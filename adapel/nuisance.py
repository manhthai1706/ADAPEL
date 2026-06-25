from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import RidgeCV


def alpha(
    e: np.ndarray,
    fusion_gamma: float = 1.0,
    min_alpha: float = 0.1,
) -> np.ndarray:
    """Compute the adaptive fusion weight alpha(e).

    alpha = max(min_alpha, clip(1 - 4*e*(1-e), 0, 1)^gamma)

    When e ≈ 0.5 (good overlap), alpha → 0 → DR-learner dominates.
    When e ≈ 0 or 1 (imbalance), alpha → 1 → X-learner dominates.
    """
    raw = np.clip(1.0 - 4.0 * e * (1.0 - e), 0.0, 1.0) ** fusion_gamma
    return np.maximum(min_alpha, raw)


def clip_e(e: np.ndarray, clip_propensity: float = 0.05) -> np.ndarray:
    """Clip propensity scores to avoid extreme weights."""
    return np.clip(e, clip_propensity, 1.0 - clip_propensity)


def select_features(X: np.ndarray, Y: np.ndarray, feature_frac: float = 0.5) -> np.ndarray:
    """Feature selection via RidgeCV coefficient magnitude thresholding.

    Returns a boolean mask keeping the top `feature_frac` fraction of features.
    """
    rc = RidgeCV(cv=5).fit(X, Y)
    threshold = np.percentile(np.abs(rc.coef_), (1.0 - feature_frac) * 100.0)
    keep = np.abs(rc.coef_) >= threshold
    if keep.sum() < 2:
        keep[:2] = True
    return keep


def _assert_finite(X: np.ndarray, name: str = "X") -> None:
    """Raise ValueError if X contains NaN or infinite values."""
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def validate(
    X: ArrayLike,
    T: ArrayLike,
    Y: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and standardise input arrays.

    Ensures correct shapes, binary treatment (0/1), and finite values.
    """
    X_arr = np.atleast_2d(np.asarray(X, dtype=float))
    T_arr = np.asarray(T, dtype=float).ravel()
    Y_arr = np.asarray(Y, dtype=float).ravel()
    if not (X_arr.shape[0] == T_arr.shape[0] == Y_arr.shape[0]):
        raise ValueError(
            f"X ({X_arr.shape[0]}), T ({T_arr.shape[0]}), Y ({Y_arr.shape[0]}) "
            "rows differ"
        )
    if not set(np.unique(T_arr)).issubset({0.0, 1.0}):
        raise ValueError(f"T must be binary 0/1, got {np.unique(T_arr)}")
    _assert_finite(X_arr, "X")
    _assert_finite(T_arr, "T")
    _assert_finite(Y_arr, "Y")
    return X_arr, T_arr, Y_arr


def check_min_class(T: np.ndarray, n_folds: int) -> None:
    """Check each treatment arm has enough samples for stratified CV."""
    n_t = int(T.sum())
    n_c = len(T) - n_t
    if n_t < n_folds or n_c < n_folds:
        raise ValueError(
            f"Each treatment arm needs at least {n_folds} samples "
            f"for {n_folds}-fold CV (T=1: {n_t}, T=0: {n_c})"
        )
