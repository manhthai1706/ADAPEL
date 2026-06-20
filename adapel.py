"""
adapel.py — ADAPEL: ADaptive Doubly-robust Pseudo-outcome Ensemble Learner.

Core algorithm (4 steps):
  1. DCM nuisance: fit mu0, mu1, e on FULL data → pseudo-outcome quality cao hơn
  2. DR + X pseudo-outcome fusion + R-Learner weighting
  3. Cross-fit OOF stacking trên base learners (vẫn OOF để meta non-overfit)
  4. NNLS positive stacking + L2 regularization

Improvements so far:
  - DCM-style: nuisance trên full data (Kennedy 2020), OOF chỉ cho stacking
  - Feature selection: RidgeCV pre-filter (optional)
  - Subsampling bootstrap: 63.2% không hoàn lại → nhanh hơn
  - 5 base learners: HistGBM, ExtraTrees, Ridge, DecisionTree, Lasso
  - L2 stacking, StratifiedKFold, Parallel bootstrap

Requirements: numpy, scipy, scikit-learn
"""
from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


# ── Meta-stacker ──────────────────────────────────────────────────────────────

class _PositiveStacking:
    def __init__(self, coefs: np.ndarray, intercept: float = 0.0):
        self.coef_ = np.asarray(coefs, dtype=float)
        self.intercept_ = float(intercept)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


# ── Base ABC ──────────────────────────────────────────────────────────────────

class BaseMetaLearner(ABC):
    @abstractmethod
    def fit(self, X, T, Y): ...
    @abstractmethod
    def predict(self, X): ...
    def estimate_ate(self, X): return float(np.mean(self.predict(X)))
    def estimate_att(self, X, T):
        return float(np.mean(self.predict(X[np.asarray(T).ravel() == 1])))
    def estimate_atc(self, X, T):
        return float(np.mean(self.predict(X[np.asarray(T).ravel() == 0])))


# ── Core: ADAPEL ──────────────────────────────────────────────────────────────

class ADAPEL(BaseMetaLearner):
    """ADAPEL — CATE ensemble learner.

    Parameters
    ----------
    outcome_estimator    : regressor for mu0, mu1.       Default: HistGBM(150).
    propensity_estimator : classifier for e(x).          Default: HistGBM(150).
    base_estimators      : list of base regressors.      Default: [HGB, ET, Ridge, DT, Lasso].
    n_folds              : cross-fitting folds.           Default: 3.
    fusion_gamma         : sharpness of alpha(x).         Default: 1.0.
    min_alpha            : min X-Learner weight.          Default: 0.1.
    clip_propensity      : clip e(x).                     Default: 0.05.
    feature_select       : True=RidgeCV auto-select.     Default: False.
    feature_frac         : fraction features to keep.     Default: 0.5.
    """
    def __init__(
        self,
        outcome_estimator:    Optional[BaseEstimator] = None,
        propensity_estimator: Optional[BaseEstimator] = None,
        base_estimators:      Optional[list]          = None,
        n_folds:              int   = 3,
        fusion_gamma:         float = 1.0,
        min_alpha:            float = 0.1,
        clip_propensity:      float = 0.05,
        feature_select:       bool  = False,
        feature_frac:         float = 0.5,
    ) -> None:
        self.outcome_estimator = (
            outcome_estimator
            or HistGradientBoostingRegressor(random_state=42, max_iter=150, max_depth=6, learning_rate=0.05)
        )
        self.propensity_estimator = (
            propensity_estimator
            or HistGradientBoostingClassifier(random_state=42, max_iter=150, max_depth=6)
        )
        self.base_estimators = base_estimators or [
            HistGradientBoostingRegressor(random_state=42, max_iter=200, max_depth=5, learning_rate=0.05),
            ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.7, n_jobs=-1, random_state=42),
            Ridge(alpha=1.0),
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
            Lasso(alpha=0.01, max_iter=5000),
        ]
        self.n_folds         = n_folds
        self.fusion_gamma    = fusion_gamma
        self.min_alpha       = min_alpha
        self.clip_propensity = clip_propensity
        self.feature_select  = feature_select
        self.feature_frac    = feature_frac
        self._meta = self._fitted_finals = self._prop_full = None
        self._m0_full = self._m1_full = None
        self._bootstrap_learners = None
        self._t_res_std = 0.0
        self._selected_cols = None

    # ── helpers ──

    def _alpha(self, e: np.ndarray) -> np.ndarray:
        raw = np.clip(1.0 - 4.0 * e * (1.0 - e), 0.0, 1.0) ** self.fusion_gamma
        return np.maximum(self.min_alpha, raw)

    def _clip_e(self, e: np.ndarray) -> np.ndarray:
        return np.clip(e, self.clip_propensity, 1.0 - self.clip_propensity)

    @staticmethod
    def _fit_w(est, X, y, sw):
        try:    return est.fit(X, y, sample_weight=sw)
        except TypeError: return est.fit(X, y)

    def _fit_positive_stacking(self, oof, pseudo, sw) -> _PositiveStacking:
        nb = oof.shape[1]
        sw_n = sw / max(sw.mean(), 1e-10)
        sqrt_w = np.sqrt(np.maximum(sw_n, 1e-10))
        A = np.column_stack([oof * sqrt_w[:, None], sqrt_w])
        b = pseudo * sqrt_w
        lam = 1e-3 * b.std() / nb
        A_reg = np.column_stack([np.eye(nb) * np.sqrt(lam), np.zeros(nb)])
        A = np.vstack([A, A_reg])
        b = np.concatenate([b, np.zeros(nb)])
        coefs, _ = nnls(A, b, maxiter=500 * nb)
        base_w, intercept = coefs[:nb], coefs[nb]
        if base_w.sum() < 1e-6:
            base_w = np.ones(nb) / nb
            intercept = 0.0
        return _PositiveStacking(base_w, intercept)

    def _select_features(self, X, Y):
        """RidgeCV pre-filter: giữ features có |coef| >= percentile."""
        if not self.feature_select:
            return X
        rc = RidgeCV().fit(X, Y)
        threshold = np.percentile(np.abs(rc.coef_), (1 - self.feature_frac) * 100)
        keep = np.abs(rc.coef_) >= threshold
        if keep.sum() < 2:
            keep[:2] = True
        self._selected_cols = keep
        return X[:, keep]

    def _validate(self, X, T, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T = np.asarray(T, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        assert X.shape[0] == T.shape[0] == Y.shape[0], "X, T, Y rows differ"
        assert set(np.unique(T)).issubset({0.0, 1.0}), "T must be binary 0/1"
        return X, T, Y

    def _check_fitted(self):
        if self._fitted_finals is None:
            raise RuntimeError("ADAPEL not fitted. Call .fit() first.")

    # ── core: fit (DCM-style) ──

    def fit(self, X, T, Y) -> "ADAPEL":
        X, T, Y = self._validate(X, T, Y)
        n, nb = X.shape[0], len(self.base_estimators)

        # Feature selection
        X = self._select_features(X, Y)

        # Step 1: DCM — nuisance trên FULL data
        i0, i1 = T == 0, T == 1
        m0_full = clone(self.outcome_estimator).fit(X[i0], Y[i0]) if i0.sum() >= 5 else clone(self.outcome_estimator).fit(X, Y)
        m1_full = clone(self.outcome_estimator).fit(X[i1], Y[i1]) if i1.sum() >= 5 else clone(self.outcome_estimator).fit(X, Y)
        e_full = clone(self.propensity_estimator).fit(X, T)

        mu0, mu1 = m0_full.predict(X), m1_full.predict(X)
        e_pred = self._clip_e(e_full.predict_proba(X)[:, 1])

        # Step 2: pseudo-outcome fusion (full data, full-data nuisance)
        tres = T - e_pred
        pDR = (mu1 - mu0) + tres / (e_pred * (1.0 - e_pred)) * (Y - np.where(T == 1, mu1, mu0))
        pX  = np.where(T == 1, Y - mu0, mu1 - Y)
        a = self._alpha(e_pred)
        pseudo = a * pX + (1.0 - a) * pDR
        sw = tres ** 2 / max((tres ** 2).mean(), 1e-10)

        # Step 3: OOF stacking (vẫn cross-fit để meta không overfit)
        n_folds = max(3, min(self.n_folds, int(n / 100)))
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=42)
        oof = np.zeros((n, nb))
        for j, est in enumerate(self.base_estimators):
            for tr, val in skf.split(X, T):
                oof[val, j] = self._fit_w(clone(est), X[tr], pseudo[tr], sw[tr]).predict(X[val])

        # Step 4: NNLS stacking
        self._meta = self._fit_positive_stacking(oof, pseudo, sw)
        self._fitted_finals = [self._fit_w(clone(m), X, pseudo, sw) for m in self.base_estimators]
        self._prop_full = e_full
        self._t_res_std = float(tres.std())
        self._m0_full = m0_full
        self._m1_full = m1_full
        return self

    # ── core: predict ──

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        return self._meta.predict(np.column_stack([m.predict(X) for m in self._fitted_finals]))

    def predict_potential_outcomes(self, X) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        return self._m0_full.predict(X), self._m1_full.predict(X)

    def predict_counterfactual(self, X, T_observed) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T_obs = np.asarray(T_observed).ravel()
        assert X.shape[0] == T_obs.shape[0]
        y0, y1 = self.predict_potential_outcomes(X)
        return np.where(T_obs == 1, y0, y1)

    # ── diagnostics ──

    def get_diagnostics(self, X) -> dict:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        e = self._clip_e(self._prop_full.predict_proba(X)[:, 1])
        a = self._alpha(e)
        return {
            "propensity": e, "alpha": a,
            "meta_weights": self._meta.coef_,
            "pct_dr_dominant": float((a < 0.5).mean()),
            "pct_x_dominant": float((a >= 0.5).mean()),
            "ensemble_std": np.column_stack([m.predict(X) for m in self._fitted_finals]).std(axis=1),
            "t_res_std_train": self._t_res_std,
        }

    # ── bootstrap CI (subsampling) ──

    def fit_bootstrap(self, X, T, Y, n_bootstrap: int = 30, random_state: int = 42,
                      n_jobs: int = -1) -> "ADAPEL":
        """Bootstrap CI: fit n_bootstrap models trên bootstrap samples (có hoàn lại)."""
        X, T, Y = self._validate(X, T, Y)
        n = X.shape[0]
        self.fit(X, T, Y)

        light_outcome = self._lighten(self.outcome_estimator)
        light_prop    = self._lighten(self.propensity_estimator)
        light_finals  = [self._lighten(m) for m in self.base_estimators]

        def _fit_one(seed):
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=n, replace=True)
            Xb, Tb, Yb = X[idx], T[idx], Y[idx]
            if Tb.sum() < 5 or (1 - Tb).sum() < 5:
                return None
            bl = ADAPEL(
                outcome_estimator=clone(light_outcome),
                propensity_estimator=clone(light_prop),
                base_estimators=[clone(m) for m in light_finals],
                n_folds=self.n_folds, fusion_gamma=self.fusion_gamma,
                min_alpha=self.min_alpha, clip_propensity=self.clip_propensity,
            )
            try:
                bl.fit(Xb, Tb, Yb); return bl
            except Exception:
                return None

        seeds = [random_state + i + 1 for i in range(n_bootstrap)]
        results = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(s) for s in seeds)
        self._bootstrap_learners = [r for r in results if r is not None]
        return self

    @staticmethod
    def _lighten(est) -> BaseEstimator:
        el = clone(est)
        for attr in ("max_iter", "n_estimators"):
            if hasattr(el, attr) and getattr(el, attr) is not None:
                setattr(el, attr, max(50, int(getattr(el, attr) * 0.7)))
        if hasattr(el, "n_jobs"):
            el.n_jobs = -1
        return el

    def predict_clinical(self, X, alpha: float = 0.05) -> dict:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        cate = self.predict(X)
        e_raw = self._prop_full.predict_proba(X)[:, 1]
        in_overlap = (e_raw >= self.clip_propensity) & (e_raw <= 1.0 - self.clip_propensity)
        e = self._clip_e(e_raw)
        lower = upper = std = None
        if self._bootstrap_learners:
            preds = np.column_stack([m.predict(X) for m in self._bootstrap_learners])
            cate = preds.mean(axis=1)
            std = preds.std(axis=1, ddof=1)
            from scipy.stats import norm as _norm
            zval = float(_norm.ppf(1 - alpha / 2))
            lower = cate - zval * std
            upper = cate + zval * std
        return {"cate": cate, "lower_ci": lower, "upper_ci": upper,
                "bootstrap_std": std, "propensity": e,
                "propensity_raw": e_raw, "in_overlap": in_overlap}

    # ── clinical extras ──

    def estimate_e_value(self, X, outcome_type: str = "binary") -> float:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        ate = self.estimate_ate(X)
        if outcome_type == "binary":
            p0 = float(np.clip(np.mean(self._m0_full.predict(X)), 1e-5, 1 - 1e-5))
            p1 = float(np.clip(np.mean(self._m1_full.predict(X)), 1e-5, 1 - 1e-5))
            rr = max(p1 / p0, p0 / p1)
        else:
            y0, y1 = self.predict_potential_outcomes(X)
            std = max(float(np.std(np.concatenate([y0, y1]))), 1e-5)
            rr = np.exp(0.91 * abs(ate / std))
        return 1.0 if rr <= 1.0 else float(rr + np.sqrt(rr * (rr - 1.0)))

    def explain_cate_surrogate(self, X, feature_names: Optional[list] = None, max_depth: int = 3) -> str:
        from sklearn.tree import export_text
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._selected_cols is not None:
            X = X[:, self._selected_cols]
        surrogate = DecisionTreeRegressor(max_depth=max_depth, random_state=42).fit(X, self.predict(X))
        names = feature_names or [f"F{i}" for i in range(X.shape[1])]
        out = []
        for line in export_text(surrogate, feature_names=names).split("\n"):
            if not line.strip():
                continue
            out.append(line)
        return "\n".join(out)
