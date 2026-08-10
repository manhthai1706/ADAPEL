from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import RidgeCV


def alpha(
    e: np.ndarray,
    fusion_gamma: float = 1.0,
    min_alpha: float = 0.1,
) -> np.ndarray:
    """Adaptive fusion weight alpha(e).

    e ≈ 0.5 -> DR-Learner dominates; e ≈ 0 or 1 -> X-Learner dominates.
    """
    raw = np.clip(1.0 - 4.0 * e * (1.0 - e), 0.0, 1.0) ** fusion_gamma
    return np.maximum(min_alpha, raw)


def clip_e(e: np.ndarray, clip_propensity: float = 0.05) -> np.ndarray:
    """Clip propensity scores to avoid extreme weights."""
    return np.clip(e, clip_propensity, 1.0 - clip_propensity)


def select_features(X: np.ndarray, Y: np.ndarray, feature_frac: float = 0.5) -> np.ndarray:
    """Select top features by RidgeCV coefficient magnitude."""
    ridge = RidgeCV(cv=5).fit(X, Y)
    threshold = np.percentile(np.abs(ridge.coef_), (1.0 - feature_frac) * 100.0)
    keep = np.abs(ridge.coef_) >= threshold
    if keep.sum() < 2:
        keep[:2] = True
    return keep


def detect_missing(X: np.ndarray) -> dict:
    """Summarise missing values in X."""
    col_n_missing = np.isnan(X).sum(axis=0).astype(int)
    n_missing = int(col_n_missing.sum())
    return {
        "has_missing": n_missing > 0,
        "n_missing": n_missing,
        "pct_missing": n_missing / X.size * 100 if X.size else 0.0,
        "col_n_missing": col_n_missing,
    }


def check_sample_size(X: np.ndarray, T: np.ndarray) -> list[str]:
    """Return warnings about sample size adequacy for CATE estimation."""
    n, n1 = X.shape[0], int(T.sum())
    n0 = len(T) - n1
    warnings: list[str] = []
    if n < 200:
        warnings.append(
            f"Total sample size ({n}) is small for CATE estimation; "
            "estimates may have high variance."
        )
    if n1 < 50:
        warnings.append(
            f"Treated arm has only {n1} samples; CATE estimates in "
            "treated-dominated regions may be unreliable."
        )
    if n0 < 50:
        warnings.append(
            f"Control arm has only {n0} samples; CATE estimates in "
            "control-dominated regions may be unreliable."
        )
    ratio = max(n1, n0) / max(min(n1, n0), 1)
    if ratio > 10:
        warnings.append(
            f"Severe treatment imbalance (ratio {ratio:.1f}:1); "
            "propensity-based methods may be unstable."
        )
    return warnings


def validate(
    X: ArrayLike, T: ArrayLike, Y: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and standardise input arrays."""
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
    if not np.isfinite(X_arr).all():
        raise ValueError("X contains NaN or infinite values")
    if not np.isfinite(T_arr).all():
        raise ValueError("T contains NaN or infinite values")
    if not np.isfinite(Y_arr).all():
        raise ValueError("Y contains NaN or infinite values")
    return X_arr, T_arr, Y_arr


def check_min_class(T: np.ndarray, n_folds: int) -> None:
    """Ensure each treatment arm has enough samples for stratified CV."""
    n_t, n_c = int(T.sum()), len(T) - int(T.sum())
    if n_t < n_folds or n_c < n_folds:
        raise ValueError(
            f"Each treatment arm needs at least {n_folds} samples "
            f"for {n_folds}-fold CV (T=1: {n_t}, T=0: {n_c})"
        )
