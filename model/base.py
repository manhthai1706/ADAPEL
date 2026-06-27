from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone


class PositiveStacking:
    """Container for NNLS-derived positive stacking coefficients.

    Parameters
    ----------
    coefs : np.ndarray
        Positive weights for each base learner.
    intercept : float
        Bias term (typically 0 under NNLS).
    """

    def __init__(self, coefs: np.ndarray, intercept: float = 0.0):
        self.coef_ = np.asarray(coefs, dtype=float)
        self.intercept_ = float(intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted sum of base learner predictions."""
        return X @ self.coef_ + self.intercept_


class BaseMetaLearner(ABC):
    """Abstract base for meta-learners estimating CATE tau(X)."""

    @abstractmethod
    def fit(self, X: ArrayLike, T: ArrayLike, Y: ArrayLike) -> BaseMetaLearner:
        ...

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        ...

    def estimate_ate(self, X: ArrayLike) -> float:
        """Average Treatment Effect: mean of individual CATE estimates."""
        return float(np.mean(self.predict(X)))

    def estimate_att(self, X: ArrayLike, T: ArrayLike) -> float:
        """Average Treatment effect on the Treated."""
        return float(np.mean(self.predict(X[np.asarray(T).ravel() == 1])))

    def estimate_atc(self, X: ArrayLike, T: ArrayLike) -> float:
        """Average Treatment effect on the Control."""
        return float(np.mean(self.predict(X[np.asarray(T).ravel() == 0])))


def fit_w(
    est: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
) -> BaseEstimator:
    """Fit estimator with sample weights, falling back to unweighted fit."""
    try:
        return est.fit(X, y, sample_weight=sw)
    except TypeError:
        return est.fit(X, y)


def scale_estimator(
    est: BaseEstimator,
    factor: float = 0.5,
    min_val: int = 30,
) -> BaseEstimator:
    """Return a clone of `est` with reduced iteration count.

    Parameters
    ----------
    est : sklearn estimator
        Estimator to scale down.
    factor : float
        Multiplicative scaling factor for max_iter / n_estimators.
    min_val : int
        Floor for the scaled hyper-parameter.
    """
    el = clone(est)
    for attr in ("max_iter", "n_estimators"):
        if hasattr(el, attr) and getattr(el, attr) is not None:
            setattr(el, attr, max(min_val, int(getattr(el, attr) * factor)))
    if hasattr(el, "n_jobs"):
        el.n_jobs = -1
    return el


def lighter(est: BaseEstimator, factor: float = 0.5) -> BaseEstimator:
    """Backward-compat alias for aggressive down-scaling (factor=0.5)."""
    return scale_estimator(est, factor=factor, min_val=30)


def lighten(est: BaseEstimator, factor: float = 0.7) -> BaseEstimator:
    """Backward-compat alias for moderate down-scaling (factor=0.7)."""
    return scale_estimator(est, factor=factor, min_val=50)
