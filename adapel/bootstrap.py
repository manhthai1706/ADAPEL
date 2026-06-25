from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm as _norm
from sklearn.base import clone
from joblib import Parallel, delayed
from .base import scale_estimator
from .nuisance import validate, clip_e

_MIN_BOOT_CLASS = 5


def fit_bootstrap(
    model_self,
    X: ArrayLike,
    T: ArrayLike,
    Y: ArrayLike,
    n_bootstrap: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Fit bootstrap ensemble of ADAPEL models for uncertainty quantification.

    Stores fitted learners in ``model_self._bootstrap_learners``.
    """
    from .model import ADAPEL

    X, T, Y = validate(X, T, Y)
    n = X.shape[0]
    model_self.fit(X, T, Y)

    if model_self.verbose:
        import logging
        logging.getLogger(__name__).info(
            f"Bootstrap: {n_bootstrap} reps, {n_jobs} jobs"
        )

    light_outcome = scale_estimator(
        model_self.outcome_estimator,
        factor=getattr(model_self, "_boot_frac", 0.7),
        min_val=50,
    )
    light_prop = scale_estimator(
        model_self.propensity_estimator,
        factor=getattr(model_self, "_boot_frac", 0.7),
        min_val=50,
    )

    def _fit_one(seed: int):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n, replace=True)
        Xb, Tb, Yb = X[idx], T[idx], Y[idx]
        if Tb.sum() < _MIN_BOOT_CLASS or (1 - Tb).sum() < _MIN_BOOT_CLASS:
            return None
        bl = ADAPEL(
            outcome_estimator=clone(light_outcome),
            propensity_estimator=clone(light_prop),
            n_folds=model_self.n_folds,
            fusion_gamma=model_self.fusion_gamma,
            min_alpha=model_self.min_alpha,
            clip_propensity=model_self.clip_propensity,
            mode=getattr(model_self, "_mode", "balanced"),
        )
        try:
            bl.fit(Xb, Tb, Yb)
            return bl
        except Exception:
            return None

    seeds = [random_state + i + 1 for i in range(n_bootstrap)]
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_fit_one)(s) for s in seeds
    )
    model_self._bootstrap_learners = [r for r in results if r is not None]
    if model_self.verbose:
        import logging
        logging.getLogger(__name__).info(
            f"  Bootstrap: {len(model_self._bootstrap_learners)}/{n_bootstrap} succeeded"
        )
    return model_self


def predict_clinical(model_self, X: ArrayLike, alpha: float = 0.05) -> dict:
    """Clinical prediction with BMA point estimate and percentile CI.

    Parameters
    ----------
    alpha : float
        Significance level (0.05 => 95% confidence interval).
    """
    cate_point = model_self.predict(X)
    e_raw = model_self._prop_full.predict_proba(X)[:, 1]
    in_overlap = (
        (e_raw >= model_self.clip_propensity)
        & (e_raw <= 1.0 - model_self.clip_propensity)
    )
    e = clip_e(e_raw, model_self.clip_propensity)
    lower = upper = std = None
    cate = cate_point
    if getattr(model_self, "_bootstrap_learners", None):
        preds = np.column_stack(
            [m.predict(X) for m in model_self._bootstrap_learners]
        )
        cate = preds.mean(axis=1)
        std = preds.std(axis=1, ddof=1)
        zval = float(_norm.ppf(1.0 - alpha / 2.0))
        lower = cate - zval * std
        upper = cate + zval * std
    return {
        "cate": cate,
        "cate_point": cate_point,
        "lower_ci": lower,
        "upper_ci": upper,
        "bootstrap_std": std,
        "propensity": e,
        "propensity_raw": e_raw,
        "in_overlap": in_overlap,
    }
