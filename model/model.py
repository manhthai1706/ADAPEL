from __future__ import annotations
import os
import json
import hashlib
import warnings
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Literal
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed, dump, load as jl_load

from .config import MODE_PRESETS
from .base import BaseMetaLearner, scale_estimator, fit_w
from .nuisance import (
    validate,
    check_min_class,
    select_features,
    alpha,
    clip_e,
    detect_missing,
    check_sample_size,
)
from .stacking import fit_stacking, MIN_COEF_ACTIVE
from .diagnostics import compute_diagnostics, estimate_e_value, explain_surrogate
from .bootstrap import fit_bootstrap, predict_clinical
from .clinical import (
    subgroup_analysis,
    variable_importance,
    balance_check,
    negative_control_test,
    calibration_check,
    fairness_report,
    sample_size_report,
)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

_MIN_SAMPLES_PER_ARM = 30

_ADAPEL_VERSION = "0.2.0"


class ADAPEL(BaseMetaLearner):
    """Adaptive Doubly-Robust Pseudo-outcome Ensemble Learner.

    Fuses DR-Learner, X-Learner and R-Learner into a single propensity-driven
    ensemble, with NNLS stacking for the final CATE estimator.

    Parameters
    ----------
    outcome_estimator : BaseEstimator, optional
        Regression model for E[Y|X,T]. Default: HistGradientBoostingRegressor.
    propensity_estimator : BaseEstimator, optional
        Classifier for P(T=1|X). Default: HistGradientBoostingClassifier.
    base_estimators : list, optional
        List of regressors for the stacking ensemble.
    n_folds : int
        Number of OOF stacking folds.
    fusion_gamma : float
        Exponent in the alpha(e) fusion function.
    min_alpha : float
        Floor for the fusion weight alpha.
    clip_propensity : float
        Clipping threshold for extreme propensity scores.
    feature_select : bool
        Whether to perform Ridge-based feature selection before fitting.
    feature_frac : float
        Fraction of features to retain when feature_select=True.
    mode : Literal["fast", "balanced", "accurate"]
        Complexity preset controlling tree depth, iterations, etc.
    verbose : bool
        Whether to log progress.
    """

    def __init__(
        self,
        outcome_estimator: Optional[BaseEstimator] = None,
        propensity_estimator: Optional[BaseEstimator] = None,
        base_estimators: Optional[list] = None,
        n_folds: int = 3,
        fusion_gamma: float = 1.0,
        min_alpha: float = 0.1,
        clip_propensity: float = 0.05,
        feature_select: bool = False,
        feature_frac: float = 0.5,
        mode: Literal["fast", "balanced", "accurate"] = "balanced",
        verbose: bool = False,
    ) -> None:
        mp = MODE_PRESETS.get(mode, MODE_PRESETS["balanced"])
        self._oof_frac = mp["oof_frac"]
        self._boot_frac = mp["boot_frac"]
        self._mode = mode
        self.verbose = verbose

        self.outcome_estimator = (
            outcome_estimator
            or HistGradientBoostingRegressor(
                random_state=42, max_iter=mp["outcome_iter"],
                max_depth=mp["outcome_depth"], learning_rate=0.05,
            )
        )
        self.propensity_estimator = (
            propensity_estimator
            or HistGradientBoostingClassifier(
                random_state=42, max_iter=mp["prop_iter"],
                max_depth=mp["prop_depth"],
            )
        )
        self.base_estimators = base_estimators or [
            HistGradientBoostingRegressor(
                random_state=42, max_iter=mp["gbm_iter"],
                max_depth=mp["gbm_depth"], learning_rate=0.05,
            ),
            ExtraTreesRegressor(
                n_estimators=mp["et_n"], min_samples_leaf=mp["et_leaf"],
                max_features=0.7, n_jobs=-1, random_state=42,
            ),
            Ridge(alpha=1.0),
            DecisionTreeRegressor(
                max_depth=mp["gbm_depth"], min_samples_leaf=10, random_state=42,
            ),
            Lasso(alpha=0.01, max_iter=5000),
        ]
        self.n_folds = n_folds
        self.fusion_gamma = fusion_gamma
        self.min_alpha = min_alpha
        self.clip_propensity = clip_propensity
        self.feature_select = feature_select
        self.feature_frac = feature_frac
        self._meta = self._fitted_finals = self._prop_full = None
        self._m0_full = self._m1_full = None
        self._bootstrap_learners = None
        self._t_res_std = 0.0
        self._selected_cols = None
        self._audit = None
        self._fit_n = 0
        self._fit_hash = None

    # ── Audit trail helpers ──

    def _compute_hash(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(X).tobytes()
            + np.ascontiguousarray(T).tobytes()
            + np.ascontiguousarray(Y).tobytes()
        ).hexdigest()[:16]

    def _build_audit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> dict:
        return {
            "version": _ADAPEL_VERSION,
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

    # ── Core fitting / prediction ──

    def _check_fitted(self) -> None:
        if self._fitted_finals is None:
            raise NotFittedError("ADAPEL not fitted. Call .fit() first.")

    def _prepare_X(self, X: ArrayLike) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # Warn about missing data at prediction time
        missing = detect_missing(X)
        if missing["has_missing"] and self.verbose:
            logger.warning(
                f"Input X has {missing['n_missing']} NaN values "
                f"({missing['pct_missing']:.1f}%). Results may be unreliable."
            )
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        return X

    def fit(self, X: ArrayLike, T: ArrayLike, Y: ArrayLike) -> ADAPEL:
        """Fit ADAPEL on covariates, treatment, and outcome.

        Parameters
        ----------
        X : (n_samples, n_features) array-like
            Covariates.
        T : (n_samples,) array-like
            Binary treatment indicator (0/1).
        Y : (n_samples,) array-like
            Observed outcome.

        Returns
        -------
        self
        """
        X, T, Y = validate(X, T, Y)
        n, nb = X.shape[0], len(self.base_estimators)

        # Audit trail
        self._fit_hash = self._compute_hash(X, T, Y)
        self._audit = self._build_audit(X, T, Y)

        self._meta = self._fitted_finals = self._prop_full = None
        self._m0_full = self._m1_full = None
        self._t_res_std = 0.0
        self._selected_cols = None

        n_folds_eff = min(self.n_folds, max(2, n // 100))
        check_min_class(T, n_folds_eff)

        if self.verbose:
            logger.info(f"ADAPEL fit: n={n}, mode={self._mode}, folds={n_folds_eff}")
            for w in self._audit["sample_size_adequate"]:
                logger.warning(f"  Sample size: {w}")

        if self.feature_select:
            self._selected_cols = select_features(X, Y, self.feature_frac)
            X = X[:, self._selected_cols]

        i0, i1 = T == 0, T == 1
        n0, n1 = int(i0.sum()), int(i1.sum())

        if n0 >= _MIN_SAMPLES_PER_ARM:
            m0_full = clone(self.outcome_estimator).fit(X[i0], Y[i0])
        else:
            if self.verbose:
                logger.warning(
                    f"Control arm has only {n0} samples (<{_MIN_SAMPLES_PER_ARM}); "
                    "fitting outcome model on full data. CATE estimates in "
                    "control-dominated regions may be unreliable."
                )
            m0_full = clone(self.outcome_estimator).fit(X, Y)

        if n1 >= _MIN_SAMPLES_PER_ARM:
            m1_full = clone(self.outcome_estimator).fit(X[i1], Y[i1])
        else:
            if self.verbose:
                logger.warning(
                    f"Treated arm has only {n1} samples (<{_MIN_SAMPLES_PER_ARM}); "
                    "fitting outcome model on full data. CATE estimates in "
                    "treated-dominated regions may be unreliable."
                )
            m1_full = clone(self.outcome_estimator).fit(X, Y)

        e_full = clone(self.propensity_estimator).fit(X, T)

        mu0, mu1 = m0_full.predict(X), m1_full.predict(X)
        e_pred = clip_e(e_full.predict_proba(X)[:, 1], self.clip_propensity)

        tres = T - e_pred
        pDR = (mu1 - mu0) + tres / (e_pred * (1.0 - e_pred)) * (
            Y - np.where(T == 1, mu1, mu0)
        )
        pX = np.where(T == 1, Y - mu0, mu1 - Y)
        a = alpha(e_pred, self.fusion_gamma, self.min_alpha)
        pseudo = a * pX + (1.0 - a) * pDR
        sw = tres ** 2 / max(float((tres ** 2).mean()), 1e-10)

        skf = StratifiedKFold(n_folds_eff, shuffle=True, random_state=42)

        def _oof_learner(j: int) -> np.ndarray:
            est = self.base_estimators[j]
            oof_j = np.zeros(n)
            for tr, val in skf.split(X, T):
                light_est = scale_estimator(est, self._oof_frac)
                m = fit_w(clone(light_est), X[tr], pseudo[tr], sw[tr])
                oof_j[val] = m.predict(X[val])
            return oof_j

        if self.verbose:
            logger.info(f"  OOF stacking: {nb} learners x {n_folds_eff} folds")
        oof_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(_oof_learner)(j) for j in range(nb)
        )
        oof = np.column_stack(oof_list)

        self._meta = fit_stacking(oof, pseudo, sw)

        self._fitted_finals = []
        for j, m in enumerate(self.base_estimators):
            if j < len(self._meta.coef_) and self._meta.coef_[j] < MIN_COEF_ACTIVE:
                self._fitted_finals.append(None)
            else:
                self._fitted_finals.append(fit_w(clone(m), X, pseudo, sw))

        self._prop_full = e_full
        self._t_res_std = float(tres.std())
        self._m0_full = m0_full
        self._m1_full = m1_full
        self._fit_n = n

        if self.verbose:
            n_active = int((self._meta.coef_ > MIN_COEF_ACTIVE).sum())
            logger.info(f"  Done. Active learners: {n_active}/{nb}")
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict CATE tau(x) for each sample in X."""
        self._check_fitted()
        X = self._prepare_X(X)
        preds = []
        for m in self._fitted_finals:
            if m is not None:
                preds.append(m.predict(X))
            else:
                preds.append(np.zeros(X.shape[0]))
        return self._meta.predict(np.column_stack(preds))

    def predict_potential_outcomes(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict potential outcomes (Y0, Y1) for each sample.

        Returns
        -------
        y0, y1 : np.ndarray
        """
        self._check_fitted()
        X = self._prepare_X(X)
        return self._m0_full.predict(X), self._m1_full.predict(X)

    def predict_counterfactual(
        self, X: ArrayLike, T_observed: ArrayLike
    ) -> np.ndarray:
        """Predict the unobserved potential outcome for each sample.

        For treated units, predicts Y(0); for control units, predicts Y(1).
        """
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
        """Return diagnostic info: propensity, alpha, stacking weights, etc."""
        self._check_fitted()
        X = self._prepare_X(X)
        return compute_diagnostics(self, X)

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
        """Bootstrap resampling for confidence intervals.

        Fits `n_bootstrap` ADAPEL models on bootstrap samples.
        """
        return fit_bootstrap(self, X, T, Y, n_bootstrap, random_state, n_jobs)

    def predict_clinical(self, X: ArrayLike, alpha: float = 0.05) -> dict:
        """Point estimate (BMA) + percentile CI + overlap flag.

        Parameters
        ----------
        X : array-like
            Covariates.
        alpha : float
            Significance level (default 0.05 => 95% CI).

        Returns
        -------
        dict with keys: cate, cate_point, lower_ci, upper_ci,
                        bootstrap_std, propensity, propensity_raw, in_overlap.
        """
        self._check_fitted()
        X = self._prepare_X(X)
        return predict_clinical(self, X, alpha)

    # ── Sensitivity ──

    def estimate_e_value(
        self, X: ArrayLike, outcome_type: Literal["binary", "continuous"] = "binary"
    ) -> float:
        """E-Value for unmeasured confounding sensitivity analysis.

        Parameters
        ----------
        X : array-like
            Covariates.
        outcome_type : "binary" or "continuous"
            Type of outcome, determining the RR approximation.
        """
        self._check_fitted()
        X = self._prepare_X(X)
        return estimate_e_value(self, X, outcome_type)

    # ── Explainability ──

    def explain_cate_surrogate(
        self,
        X: ArrayLike,
        feature_names: Optional[list] = None,
        max_depth: int = 3,
    ) -> str:
        """Fit surrogate decision tree explaining CATE predictions.

        Returns
        -------
        str
            Text representation of the decision tree rules.
        """
        self._check_fitted()
        X = self._prepare_X(X)
        return explain_surrogate(self, X, feature_names, max_depth)

    # ── Clinical analysis ──

    def sample_size_report(self, X: ArrayLike, T: ArrayLike) -> dict:
        """Sample size adequacy report.

        Returns dict with n, n_treated, n_control, n_features,
        treatment_ratio, samples_per_feature, warnings, adequate.
        """
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
        """Analyse CATE heterogeneity across subgroups.

        Parameters
        ----------
        subgroups : dict of str -> boolean mask, optional
            Pre-defined subgroup masks.
        feature_names : list of str, optional
            Names of features for reporting.
        n_bins : int
            Number of bins for automatic subgroup creation.

        Returns
        -------
        dict with keys: overall_ate, subgroups (list of entries).
        """
        self._check_fitted()
        X = self._prepare_X(X)
        T = np.asarray(T, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        return subgroup_analysis(self, X, T, Y, subgroups, feature_names, n_bins)

    def variable_importance(
        self,
        X: ArrayLike,
        feature_names: Optional[list[str]] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> dict:
        """Permutation-based variable importance for CATE predictions.

        Returns
        -------
        dict with keys: importances_mean, importances_std, feature_names.
        """
        self._check_fitted()
        X = self._prepare_X(X)
        return variable_importance(self, X, feature_names, n_repeats, random_state)

    def balance_check(
        self,
        X: ArrayLike,
        T: ArrayLike,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Covariate balance check via standardised mean difference (SMD).

        Uses propensity weights for weighted SMD comparison.

        Returns
        -------
        dict with keys: smd_unweighted, smd_weighted, threshold_exceeded.
        """
        X = self._prepare_X(X)
        T = np.asarray(T, dtype=float).ravel()
        e_raw = self._prop_full.predict_proba(X)[:, 1]
        e = clip_e(e_raw, self.clip_propensity)
        # IPW-style weights: treated = 1/e, control = 1/(1-e)
        weights = np.where(T == 1, 1.0 / e, 1.0 / (1.0 - e))
        return balance_check(X, T, feature_names, weights)

    def negative_control_test(
        self,
        X_outcome: ArrayLike,
        X_treatment: Optional[ArrayLike] = None,
        n_permute: int = 100,
        random_state: int = 42,
    ) -> dict:
        """Placebo test: permute treatment to check for spurious signal.

        Returns
        -------
        dict with keys: observed_ate, permuted_ates_mean,
        permuted_ates_std, p_value_placebo.
        """
        self._check_fitted()
        X_out = self._prepare_X(X_outcome)
        return negative_control_test(
            self, X_out, X_treatment, n_permute, random_state
        )

    def calibration_check(
        self,
        X: ArrayLike,
        n_groups: int = 10,
    ) -> dict:
        """Calibration check: compare predicted CATE vs observed outcome
        difference within CATE quantile groups.

        Returns
        -------
        dict with keys: groups, calib_error_overall.
        """
        self._check_fitted()
        X = self._prepare_X(X)
        return calibration_check(self, X, n_groups)

    def fairness_report(
        self,
        X: ArrayLike,
        protected_attributes: dict[str, np.ndarray],
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Fairness assessment: compare CATE across protected groups.

        Parameters
        ----------
        protected_attributes : dict of str -> array-like
            Each entry defines group membership (e.g., race, gender).

        Returns
        -------
        dict with groups (list of entries with cate_mean, disparity, p_value).
        """
        self._check_fitted()
        X = self._prepare_X(X)
        return fairness_report(self, X, protected_attributes, feature_names)

    # ── Audit ──

    def get_audit_trail(self) -> Optional[dict]:
        """Return audit trail from last fit.

        Includes version, timestamp, sample counts, params, missing data
        report, and sample size warnings.
        """
        return self._audit

    # ── Serialisation ──

    def save(self, path: str) -> str:
        """Save fitted model to disk.

        Parameters
        ----------
        path : str
            File path (should end in .joblib or .pkl).

        Returns
        -------
        str
            The path the model was saved to.
        """
        self._check_fitted()
        if not path.endswith((".joblib", ".pkl")):
            path = path + ".joblib"
        dump(self, path)
        if self.verbose:
            logger.info(f"Model saved to {path}")
        return path

    @staticmethod
    def load(path: str) -> ADAPEL:
        """Load a fitted ADAPEL model from disk.

        Parameters
        ----------
        path : str
            File path to load from.
        """
        return jl_load(path)
