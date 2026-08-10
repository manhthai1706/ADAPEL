from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone


class PositiveStacking:
    """NNLS-derived positive stacking coefficients for base learners."""

    __slots__ = ("coef_", "intercept_")

    def __init__(self, coefs: np.ndarray, intercept: float = 0.0) -> None:
        self.coef_ = np.asarray(coefs, dtype=float)
        self.intercept_ = float(intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


class BaseMetaLearner(ABC):
    """Abstract base class for meta-learners estimating CATE tau(X)."""

    @abstractmethod
    def fit(self, X: ArrayLike, T: ArrayLike, Y: ArrayLike) -> BaseMetaLearner:
        ...

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        ...

    def estimate_ate(self, X: ArrayLike) -> float:
        return float(np.mean(self.predict(X)))

    def estimate_att(self, X: ArrayLike, T: ArrayLike) -> float:
        T = np.asarray(T).ravel()
        return float(np.mean(self.predict(X[T == 1])))

    def estimate_atc(self, X: ArrayLike, T: ArrayLike) -> float:
        T = np.asarray(T).ravel()
        return float(np.mean(self.predict(X[T == 0])))


def fit_w(est: BaseEstimator, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> Any:
    """Fit estimator with sample weights; fall back to unweighted fit."""
    try:
        return est.fit(X, y, sample_weight=sw)
    except TypeError:
        return est.fit(X, y)


def scale_estimator(
    est: BaseEstimator, factor: float = 0.5, min_val: int = 30
) -> BaseEstimator:
    """Return a clone of ``est`` with reduced iteration / tree count."""
    out = clone(est)
    for attr in ("max_iter", "n_estimators"):
        if getattr(out, attr, None) is not None:
            setattr(out, attr, max(min_val, int(getattr(out, attr) * factor)))
    if hasattr(out, "n_jobs"):
        out.n_jobs = -1
    return out
