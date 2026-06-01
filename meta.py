"""
meta.py — ADAPEL: ADaptive Doubly-robust Pseudo-outcome Ensemble Learner.

A meta-learning framework for Conditional Average Treatment Effect (CATE)
estimation in observational studies. Combines DR-Learner, X-Learner and
R-Learner into a single adaptive ensemble.

Core algorithm (4 steps):
  1. Cross-fitting k-fold   -> tránh overfitting bias
  2. DR + X pseudo-outcome  -> doubly robust + adaptive blending
  3. R-Learner weighting    -> upweight informative samples (T-e)^2
  4. NNLS positive stacking -> weights >= 0, on dinh

References:
  - Kunzel, Sekhon, Bickel, Yu (2019). Meta-learners for Estimating
    Heterogeneous Treatment Effects. Annals of Applied Statistics.
  - Nie & Wager (2021). Quasi-Oracle Estimation of Heterogeneous
    Treatment Effects. Biometrika.
  - Kennedy (2020). Towards Optimal Doubly Robust Estimation of
    Heterogeneous Treatment Effects. Electronic Journal of Statistics.

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
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


# ── Meta-stacker (NNLS) ───────────────────────────────────────────────────────

class _PositiveStacking:
    """NNLS stacking: trọng số >= 0, tự chọn subset khi base learners correlated."""
    def __init__(self, coefs: np.ndarray, intercept: float = 0.0):
        self.coef_ = np.asarray(coefs, dtype=float)
        self.intercept_ = float(intercept)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_
    def __repr__(self) -> str:
        return f"_PositiveStacking(coef={self.coef_}, intercept={self.intercept_:.3f})"


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
    def __repr__(self): return f"{self.__class__.__name__}()"


# ── Core: ADAPEL ──────────────────────────────────────────────────────────────

class ADAPEL(BaseMetaLearner):
    """ADAPEL: ADaptive Doubly-robust Pseudo-outcome Ensemble Learner.

    Adaptive fusion framework for CATE estimation that combines three
    complementary meta-learners (DR-Learner, X-Learner, R-Learner) into
    a single propensity-driven ensemble.

    Step 1 - Cross-fit k-fold:
        Train nuisance models (mu0, mu1, e) on each fold and predict on
        held-out fold to avoid overfitting bias.

    Step 2 - Adaptive pseudo-outcome fusion:
        Y_DR = (mu1-mu0) + (T-e)/(e(1-e)) * (Y-mu_T)         [Doubly Robust]
        Y_X  = Y-mu0  (T=1)  |  mu1-Y  (T=0)                 [X-Learner]
        alpha(x) = max(min_alpha, clip(1-4e(1-e), 0, 1)^gamma)
        pseudo = alpha * Y_X + (1-alpha) * Y_DR
        The alpha(x) function automatically upweights X-Learner in
        imbalanced regions (e near 0 or 1) where DR has high variance.

    Step 3 - R-Learner sample weighting: sw = (T-e)^2.
        Samples where T deviates most from e(x) carry the most
        information about tau(x); equivalent to R-Learner's
        residual-on-residual regression objective.

    Step 4 - NNLS positive stacking on out-of-fold base learners:
        Find non-negative weights w_i >= 0 such that
        sum_i w_i * f_i(x) ~ tau(x). NNLS automatically selects
        sparse subsets when base learners are highly correlated.

    Parameters
    ----------
    outcome_estimator    : regressor for mu0, mu1. Default: HistGBM(150).
    propensity_estimator : classifier for e(x).    Default: HistGBM(150).
    base_estimators      : list of base regressors. Default: [HistGBM(200), ET(200)].
    n_folds              : cross-fitting folds.    Default: 3.
    fusion_gamma         : sharpness of alpha(x).  Default: 1.0.
    min_alpha            : min X-Learner weight.   Default: 0.1.
    clip_propensity      : clip e(x) to avoid div0. Default: 0.05.
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
        ]
        self.n_folds         = n_folds
        self.fusion_gamma    = fusion_gamma
        self.min_alpha       = min_alpha
        self.clip_propensity = clip_propensity
        self._meta = self._fitted_finals = self._prop_full = None
        self._m0_full = self._m1_full = None
        self._bootstrap_learners = None
        self._t_res_std = 0.0

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
        coefs, _ = nnls(np.column_stack([oof * sqrt_w[:, None], sqrt_w]), pseudo * sqrt_w, maxiter=500 * nb)
        base_w, intercept = coefs[:nb], coefs[nb]
        if base_w.sum() < 1e-6:
            base_w = np.ones(nb) / nb
            intercept = 0.0
        return _PositiveStacking(base_w, intercept)

    def _validate(self, X, T, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T = np.asarray(T, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        assert X.shape[0] == T.shape[0] == Y.shape[0], "X, T, Y phải cùng số hàng."
        assert set(np.unique(T)).issubset({0.0, 1.0}), "T phải nhị phân (0/1)."
        return X, T, Y

    def _check_fitted(self):
        if self._fitted_finals is None:
            raise RuntimeError("ADAPEL chưa fit. Gọi .fit() trước.")

    # ── core: fit ──

    def fit(self, X, T, Y) -> "ADAPEL":
        X, T, Y = self._validate(X, T, Y)
        n, nb = X.shape[0], len(self.base_estimators)
        pDR, pX, ehat, tres = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        for tr, val in KFold(self.n_folds, shuffle=True, random_state=42).split(X):
            Xtr, Xv, Ttr, Tv, Ytr, Yv = X[tr], X[val], T[tr], T[val], Y[tr], Y[val]
            i0, i1 = Ttr == 0, Ttr == 1
            m0 = clone(self.outcome_estimator); m1 = clone(self.outcome_estimator)
            if i0.sum() >= 5 and i1.sum() >= 5:
                m0.fit(Xtr[i0], Ytr[i0]); m1.fit(Xtr[i1], Ytr[i1])
            else:
                m0.fit(Xtr, Ytr); m1.fit(Xtr, Ytr)
            mu0, mu1 = m0.predict(Xv), m1.predict(Xv)
            e = self._clip_e(clone(self.propensity_estimator).fit(Xtr, Ttr).predict_proba(Xv)[:, 1])
            ehat[val] = e; tres[val] = Tv - e
            pDR[val] = (mu1 - mu0) + (Tv - e) / (e * (1 - e)) * (Yv - np.where(Tv == 1, mu1, mu0))
            pX[val]  = np.where(Tv == 1, Yv - mu0, mu1 - Yv)

        a      = self._alpha(ehat)
        pseudo = a * pX + (1.0 - a) * pDR
        sw     = tres ** 2
        sw    /= max(sw.mean(), 1e-10)

        oof = np.zeros((n, nb))
        for j, est in enumerate(self.base_estimators):
            for tr2, val2 in KFold(self.n_folds, shuffle=True, random_state=0).split(X):
                oof[val2, j] = self._fit_w(clone(est), X[tr2], pseudo[tr2], sw[tr2]).predict(X[val2])

        self._meta          = self._fit_positive_stacking(oof, pseudo, sw)
        self._fitted_finals = [self._fit_w(clone(m), X, pseudo, sw) for m in self.base_estimators]
        self._prop_full     = clone(self.propensity_estimator).fit(X, T)
        self._t_res_std     = float(tres.std())
        i0f, i1f = T == 0, T == 1
        self._m0_full = clone(self.outcome_estimator).fit(X[i0f], Y[i0f])
        self._m1_full = clone(self.outcome_estimator).fit(X[i1f], Y[i1f])
        return self

    # ── core: predict & counterfactuals ──

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._meta.predict(np.column_stack([m.predict(X) for m in self._fitted_finals]))

    def predict_potential_outcomes(self, X) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
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
        e = self._clip_e(self._prop_full.predict_proba(X)[:, 1])
        a = self._alpha(e)
        return {
            "propensity":      e,
            "alpha":           a,
            "meta_weights":    self._meta.coef_,
            "pct_dr_dominant": float((a < 0.5).mean()),
            "pct_x_dominant":  float((a >= 0.5).mean()),
            "ensemble_std":    np.column_stack([m.predict(X) for m in self._fitted_finals]).std(axis=1),
            "t_res_std_train": self._t_res_std,
        }

    # ── bootstrap CI ──

    def fit_bootstrap(self, X, T, Y, n_bootstrap: int = 10, random_state: int = 42) -> "ADAPEL":
        """Train trên toàn bộ data, sau đó train n_bootstrap bản nhẹ trên bootstrap samples.

        Mỗi bootstrap FIT LẠI TOÀN BỘ pipeline (gồm cả meta-learner) để CI coverage hợp lệ.
        """
        X, T, Y = self._validate(X, T, Y)
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        self.fit(X, T, Y)

        light_outcome = self._lighten(self.outcome_estimator)
        light_prop    = self._lighten(self.propensity_estimator)
        light_finals  = [self._lighten(m) for m in self.base_estimators]

        self._bootstrap_learners = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Xb, Tb, Yb = X[idx], T[idx], Y[idx]
            if Tb.sum() < 5 or (1 - Tb).sum() < 5:
                continue
            bl = ADAPEL(
                outcome_estimator=light_outcome,
                propensity_estimator=light_prop,
                base_estimators=light_finals,
                n_folds=3,
                fusion_gamma=self.fusion_gamma,
                min_alpha=self.min_alpha,
                clip_propensity=self.clip_propensity,
            )
            try:
                bl.fit(Xb, Tb, Yb)
                self._bootstrap_learners.append(bl)
            except Exception:
                continue
        return self

    @staticmethod
    def _lighten(est) -> BaseEstimator:
        el = clone(est)
        for attr in ("max_iter", "n_estimators"):
            if hasattr(el, attr) and getattr(el, attr) is not None:
                setattr(el, attr, min(getattr(el, attr), 80))
        if hasattr(el, "n_jobs"):
            el.n_jobs = -1
        return el

    def predict_clinical(self, X, alpha: float = 0.05) -> dict:
        """Point estimate = BMA (bootstrap mean) → CI percentile bao phủ point."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        cate = self.predict(X)
        e_raw = self._prop_full.predict_proba(X)[:, 1]
        in_overlap = (e_raw >= self.clip_propensity) & (e_raw <= 1.0 - self.clip_propensity)
        e = self._clip_e(e_raw)
        lower = upper = std = None
        if self._bootstrap_learners:
            preds = np.column_stack([m.predict(X) for m in self._bootstrap_learners])
            cate  = preds.mean(axis=1)
            lower = np.percentile(preds, 100 * alpha / 2, axis=1)
            upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=1)
            std   = preds.std(axis=1)
        return {
            "cate": cate, "lower_ci": lower, "upper_ci": upper, "bootstrap_std": std,
            "propensity": e, "propensity_raw": e_raw, "in_overlap": in_overlap,
        }

    # ── clinical extras ──

    def estimate_e_value(self, X, outcome_type: str = "binary") -> float:
        """E-Value (VanderWeele & Ding 2017): RR cần thiết để unmeasured confounder giải thích ATE."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
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
        """Surrogate DecisionTree giải thích CATE bằng quy tắc dễ đọc."""
        from sklearn.tree import export_text
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        surrogate = DecisionTreeRegressor(max_depth=max_depth, random_state=42).fit(X, self.predict(X))
        names = feature_names or [f"F{i}" for i in range(X.shape[1])]
        out = []
        for line in export_text(surrogate, feature_names=names).split("\n"):
            if not line.strip():
                continue
            line = line.replace("|---", "  └─>").replace("|", "  │").replace("value:", "=> CATE TB:")
            out.append(line)
        return "\n".join(out)
