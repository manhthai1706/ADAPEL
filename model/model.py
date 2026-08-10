from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timezone
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from joblib import Parallel, delayed, dump, load as jl_load
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from .base import BaseMetaLearner, fit_w, scale_estimator
from .bootstrap import fit_bootstrap, predict_clinical
from .clinical import (
    balance_check,
    calibration_check,
    fairness_report,
    negative_control_test,
    sample_size_report,
    subgroup_analysis,
    variable_importance,
)
from .config import MODE_PRESETS
from .diagnostics import compute_diagnostics, estimate_e_value, explain_surrogate
from .nuisance import (
    alpha,
    check_min_class,
    check_sample_size,
    clip_e,
    detect_missing,
    select_features,
    validate,
)
from .stacking import MIN_COEF_ACTIVE, fit_stacking

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

ADAPEL_VERSION = "0.3.0"
MIN_SAMPLES_PER_ARM = 30


# ── Default estimator factories ──


def _default_outcome_estimator(preset: dict) -> BaseEstimator:
    return HistGradientBoostingRegressor(
        random_state=42,
        max_iter=preset["outcome_iter"],
        max_depth=preset["outcome_depth"],
        learning_rate=0.05,
    )


def _default_propensity_estimator(preset: dict) -> BaseEstimator:
    return HistGradientBoostingClassifier(
        random_state=42,
        max_iter=preset["prop_iter"],
        max_depth=preset["prop_depth"],
    )


def _default_base_estimators(preset: dict) -> list[BaseEstimator]:
    return [
        HistGradientBoostingRegressor(
            random_state=42,
            max_iter=preset["gbm_iter"],
            max_depth=preset["gbm_depth"],
            learning_rate=0.05,
        ),
        ExtraTreesRegressor(
            n_estimators=preset["et_n"],
            min_samples_leaf=preset["et_leaf"],
            max_features=0.7,
            n_jobs=-1,
            random_state=42,
        ),
        Ridge(alpha=1.0),
        DecisionTreeRegressor(
            max_depth=preset["gbm_depth"], min_samples_leaf=10, random_state=42
        ),
        Lasso(alpha=0.01, max_iter=5000),
    ]


# ── Pipeline helpers ──


def _fit_arm_outcome(
    estimator: BaseEstimator, X: np.ndarray, Y: np.ndarray, mask: np.ndarray
) -> BaseEstimator:
    """Fit outcome estimator on a sub-arm if it has enough samples."""
    if mask.sum() >= MIN_SAMPLES_PER_ARM:
        return clone(estimator).fit(X[mask], Y[mask])
    return clone(estimator).fit(X, Y)


def _build_pseudo_outcome(
    mu0: np.ndarray, mu1: np.ndarray, e: np.ndarray, T: np.ndarray, Y: np.ndarray,
    fusion_gamma: float, min_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pseudo_outcome, normalised_weights, treatment_residuals)."""
    tres = T - e
    pDR = (mu1 - mu0) + tres / (e * (1.0 - e)) * (Y - np.where(T == 1, mu1, mu0))
    pX = np.where(T == 1, Y - mu0, mu1 - Y)
    a = alpha(e, fusion_gamma, min_alpha)
    pseudo = a * pX + (1.0 - a) * pDR
    sw = tres ** 2
    return pseudo, sw / max(float(sw.mean()), 1e-10), tres


def _oof_predictions(
    X: np.ndarray, T: np.ndarray, pseudo: np.ndarray, sw: np.ndarray,
    base_estimators: list[BaseEstimator], n_folds: int, oof_frac: float,
) -> np.ndarray:
    """Generate out-of-fold predictions for each base learner (parallel)."""
    n = X.shape[0]
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=42)

    def _oof_one(j: int) -> np.ndarray:
        est = base_estimators[j]
        oof_j = np.empty(n)
        for tr, val in skf.split(X, T):
            light = scale_estimator(est, oof_frac)
            oof_j[val] = fit_w(clone(light), X[tr], pseudo[tr], sw[tr]).predict(X[val])
        return oof_j

    return np.column_stack(
        Parallel(n_jobs=-1, backend="threading")(
            delayed(_oof_one)(j) for j in range(len(base_estimators))
        )
    )


def _fit_final_learners(
    X: np.ndarray, pseudo: np.ndarray, sw: np.ndarray,
    base_estimators: list[BaseEstimator], meta_coef: np.ndarray,
) -> list[Optional[BaseEstimator]]:
    """Refit each base learner on full data; skip those with near-zero weight."""
    fitted = []
    for j, est in enumerate(base_estimators):
        if j < len(meta_coef) and meta_coef[j] < MIN_COEF_ACTIVE:
            fitted.append(None)
        else:
            fitted.append(fit_w(clone(est), X, pseudo, sw))
    return fitted


# ── Main estimator ──


class ADAPEL(BaseMetaLearner):
    """Adaptive Doubly-Robust Pseudo-outcome Ensemble Learner.

    Fuses DR-, X-, and R-Learner signals via an adaptive propensity-driven
    pseudo-outcome, then stacks predictions of multiple base learners via
    NNLS with L2 regularisation.
    """

    def __init__(
        self,
        outcome_estimator: Optional[BaseEstimator] = None,
        propensity_estimator: Optional[BaseEstimator] = None,
        base_estimators: Optional[list[BaseEstimator]] = None,
        n_folds: int = 3,
        fusion_gamma: float = 1.0,
        min_alpha: float = 0.1,
        clip_propensity: float = 0.05,
        feature_select: bool = False,
        feature_frac: float = 0.5,
        mode: Literal["fast", "balanced", "accurate"] = "balanced",
        verbose: bool = False,
    ) -> None:
        preset = MODE_PRESETS.get(mode, MODE_PRESETS["balanced"])
        self._oof_frac = preset["oof_frac"]
        self._boot_frac = preset["boot_frac"]
        self._mode = mode
        self.verbose = verbose

        self.outcome_estimator = outcome_estimator or _default_outcome_estimator(preset)
        self.propensity_estimator = (
            propensity_estimator or _default_propensity_estimator(preset)
        )
        self.base_estimators = base_estimators or _default_base_estimators(preset)

        self.n_folds = n_folds
        self.fusion_gamma = fusion_gamma
        self.min_alpha = min_alpha
        self.clip_propensity = clip_propensity
        self.feature_select = feature_select
        self.feature_frac = feature_frac

        self._meta: Optional[object] = None
        self._fitted_finals: Optional[list] = None
        self._prop_full: Optional[BaseEstimator] = None
        self._m0_full: Optional[BaseEstimator] = None
        self._m1_full: Optional[BaseEstimator] = None
        self._t_res_std: float = 0.0
        self._selected_cols: Optional[np.ndarray] = None
        self._bootstrap_learners: Optional[list] = None
        self._audit: Optional[dict] = None

    # ── Internal helpers ──

    def _check_fitted(self) -> None:
        if self._fitted_finals is None:
            raise NotFittedError("ADAPEL not fitted. Call .fit() first.")

    def _prepare_X(self, X: ArrayLike) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        if self.verbose:
            missing = detect_missing(X)
            if missing["has_missing"]:
                logger.warning(
                    f"Input X has {missing['n_missing']} NaN values "
                    f"({missing['pct_missing']:.1f}%). Results may be unreliable."
                )
        return X

    def _build_audit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> dict:
        return {
            "version": ADAPEL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_treated": int(T.sum()),
            "n_control": int((1 - T).sum()),
            "params": {
                "mode": self._mode,
                "n_folds": self.n_folds,
                "fusion_gamma": self.fusion_gamma,
                "min_alpha": self.min_alpha,
                "clip_propensity": self.clip_propensity,
                "feature_select": self.feature_select,
                "feature_frac": self.feature_frac,
            },
            "missing": detect_missing(X),
            "sample_size_adequate": check_sample_size(X, T),
        }

    def _reset_state(self) -> None:
        self._meta = None
        self._fitted_finals = None
        self._prop_full = None
        self._m0_full = None
        self._m1_full = None
        self._t_res_std = 0.0
        self._selected_cols = None

    # ── Fitting ──

    def fit(self, X: ArrayLike, T: ArrayLike, Y: ArrayLike) -> ADAPEL:
        """Fit ADAPEL on covariates, treatment, and outcome."""
        X, T, Y = validate(X, T, Y)
        self._audit = self._build_audit(X, T, Y)
        self._reset_state()

        n_folds = min(self.n_folds, max(2, X.shape[0] // 100))
        check_min_class(T, n_folds)

        if self.verbose:
            logger.info(f"ADAPEL fit: n={X.shape[0]}, mode={self._mode}, folds={n_folds}")
            for w in self._audit["sample_size_adequate"]:
                logger.warning(f"  Sample size: {w}")

        if self.feature_select:
            self._selected_cols = select_features(X, Y, self.feature_frac)
            X = X[:, self._selected_cols]

        self._m0_full = _fit_arm_outcome(self.outcome_estimator, X, Y, T == 0)
        self._m1_full = _fit_arm_outcome(self.outcome_estimator, X, Y, T == 1)
        self._prop_full = clone(self.propensity_estimator).fit(X, T)

        mu0, mu1 = self._m0_full.predict(X), self._m1_full.predict(X)
        e = clip_e(self._prop_full.predict_proba(X)[:, 1], self.clip_propensity)

        pseudo, sw, tres = _build_pseudo_outcome(
            mu0, mu1, e, T, Y, self.fusion_gamma, self.min_alpha,
        )

        if self.verbose:
            logger.info(
                f"  OOF stacking: {len(self.base_estimators)} learners x {n_folds} folds"
            )

        oof = _oof_predictions(
            X, T, pseudo, sw, self.base_estimators, n_folds, self._oof_frac,
        )
        self._meta = fit_stacking(oof, pseudo, sw)
        self._fitted_finals = _fit_final_learners(
            X, pseudo, sw, self.base_estimators, self._meta.coef_,
        )
        self._t_res_std = float(tres.std())

        if self.verbose:
            active = int((self._meta.coef_ > MIN_COEF_ACTIVE).sum())
            logger.info(f"  Done. Active learners: {active}/{len(self.base_estimators)}")

        return self

    # ── Prediction ──

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict CATE tau(x) for each sample."""
        self._check_fitted()
        X = self._prepare_X(X)
        preds = [
            m.predict(X) if m is not None else np.zeros(X.shape[0])
            for m in self._fitted_finals
        ]
        return self._meta.predict(np.column_stack(preds))

    def predict_potential_outcomes(self, X: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (Y(0), Y(1)) for each sample."""
        self._check_fitted()
        X = self._prepare_X(X)
        return self._m0_full.predict(X), self._m1_full.predict(X)

    def predict_counterfactual(self, X: ArrayLike, T_observed: ArrayLike) -> np.ndarray:
        """Predict the unobserved potential outcome for each sample."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T_obs = np.asarray(T_observed).ravel()
        if X.shape[0] != T_obs.shape[0]:
            raise ValueError(
                f"X rows ({X.shape[0]}) != T_observed rows ({T_obs.shape[0]})"
            )
        y0, y1 = self.predict_potential_outcomes(X)
        return np.where(T_obs == 1, y0, y1)

    # ── Diagnostics ──

    def get_diagnostics(self, X: ArrayLike) -> dict:
        """Per-sample propensity, alpha, stacking weights, ensemble std."""
        self._check_fitted()
        return compute_diagnostics(self, self._prepare_X(X))

    def estimate_e_value(
        self, X: ArrayLike, outcome_type: Literal["binary", "continuous"] = "binary"
    ) -> float:
        """E-Value for unmeasured confounding sensitivity analysis."""
        self._check_fitted()
        return estimate_e_value(self, self._prepare_X(X), outcome_type)

    def explain_cate_surrogate(
        self,
        X: ArrayLike,
        feature_names: Optional[list[str]] = None,
        max_depth: int = 3,
    ) -> str:
        """Surrogate decision tree rules approximating CATE predictions."""
        self._check_fitted()
        return explain_surrogate(self, self._prepare_X(X), feature_names, max_depth)

    # ── Bootstrap ──

    def fit_bootstrap(
        self,
        X: ArrayLike,
        T: ArrayLike,
        Y: ArrayLike,
        n_bootstrap: int = 30,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> ADAPEL:
        """Bootstrap ensemble for confidence intervals."""
        return fit_bootstrap(self, X, T, Y, n_bootstrap, random_state, n_jobs)

    def predict_clinical(self, X: ArrayLike, alpha: float = 0.05) -> dict:
        """BMA point estimate + percentile CI + overlap flag."""
        self._check_fitted()
        return predict_clinical(self, self._prepare_X(X), alpha)

    # ── Clinical analysis ──

    def sample_size_report(self, X: ArrayLike, T: ArrayLike) -> dict:
        return sample_size_report(self, X, T)

    def subgroup_analysis(
        self,
        X: ArrayLike,
        T: ArrayLike,
        Y: ArrayLike,
        subgroups: Optional[dict[str, np.ndarray]] = None,
        feature_names: Optional[list[str]] = None,
        n_bins: int = 4,
    ) -> dict:
        self._check_fitted()
        return subgroup_analysis(
            self, self._prepare_X(X), T, Y, subgroups, feature_names, n_bins,
        )

    def variable_importance(
        self,
        X: ArrayLike,
        feature_names: Optional[list[str]] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> dict:
        self._check_fitted()
        return variable_importance(
            self, self._prepare_X(X), feature_names, n_repeats, random_state,
        )

    def balance_check(
        self,
        X: ArrayLike,
        T: ArrayLike,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        X = self._prepare_X(X)
        T = np.asarray(T, dtype=float).ravel()
        e_raw = self._prop_full.predict_proba(X)[:, 1]
        e = clip_e(e_raw, self.clip_propensity)
        weights = np.where(T == 1, 1.0 / e, 1.0 / (1.0 - e))
        return balance_check(X, T, feature_names, weights)

    def negative_control_test(
        self,
        X_outcome: ArrayLike,
        X_treatment: Optional[ArrayLike] = None,
        n_permute: int = 100,
        random_state: int = 42,
    ) -> dict:
        self._check_fitted()
        return negative_control_test(
            self, self._prepare_X(X_outcome), X_treatment, n_permute, random_state,
        )

    def calibration_check(self, X: ArrayLike, n_groups: int = 10) -> dict:
        self._check_fitted()
        return calibration_check(self, self._prepare_X(X), n_groups)

    def fairness_report(
        self,
        X: ArrayLike,
        protected_attributes: dict[str, np.ndarray],
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        self._check_fitted()
        return fairness_report(
            self, self._prepare_X(X), protected_attributes, feature_names,
        )

    # ── Audit ──

    def get_audit_trail(self) -> Optional[dict]:
        return self._audit

    # ── Serialization ──

    def save(self, path: str) -> str:
        """Save fitted model to disk."""
        self._check_fitted()
        if not path.endswith((".joblib", ".pkl")):
            path = path + ".joblib"
        dump(self, path)
        if self.verbose:
            logger.info(f"Model saved to {path}")
        return path

    @staticmethod
    def load(path: str) -> ADAPEL:
        """Load fitted model from disk."""
        return jl_load(path)
