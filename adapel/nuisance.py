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


def detect_missing(X: np.ndarray, name: str = "X") -> dict:
    """Detect and report missing values in array.

    Returns
    -------
    dict with keys: has_missing, n_missing, pct_missing, col_n_missing
    """
    n_missing = int(np.isnan(X).sum())
    has_missing = n_missing > 0
    col_n_missing = np.isnan(X).sum(axis=0).astype(int)
    return {
        "has_missing": has_missing,
        "n_missing": n_missing,
        "pct_missing": n_missing / X.size * 100 if X.size > 0 else 0.0,
        "col_n_missing": col_n_missing,
    }


def check_sample_size(X: np.ndarray, T: np.ndarray, verbose: bool = False) -> list[str]:
    """Check sample size adequacy for CATE estimation.

    Returns list of warning messages.
    """
    warnings_list = []
    n = X.shape[0]
    n1, n0 = int(T.sum()), int((1 - T).sum())

    if n < 200:
        warnings_list.append(
            f"Total sample size ({n}) is small for CATE estimation. "
            "Estimates may have high variance."
        )
    if n1 < 50:
        warnings_list.append(
            f"Treated arm has only {n1} samples; CATE estimates in "
            "treated-dominated regions may be unreliable."
        )
    if n0 < 50:
        warnings_list.append(
            f"Control arm has only {n0} samples; CATE estimates in "
            "control-dominated regions may be unreliable."
        )
    ratio = max(n1, n0) / max(min(n1, n0), 1)
    if ratio > 10:
        warnings_list.append(
            f"Severe treatment imbalance (ratio {ratio:.1f}:1). "
            "Propensity-based methods may be unstable."
        )
    return warnings_list


def _assert_finite(X: np.ndarray, name: str = "X") -> None:
    """Raise ValueError if X contains NaN or infinite values."""
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def validate(
    X: ArrayLike,
    T: ArrayLike,
    Y: ArrayLike,
    allow_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and standardise input arrays.

    Ensures correct shapes, binary treatment (0/1), and finite values.

    Parameters
    ----------
    allow_missing : bool
        If True, only check Inf (not NaN) in X.
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
    # Always check Inf
    if not np.isfinite(X_arr).all() and not allow_missing:
        has_inf = np.isinf(X_arr).any()
        has_nan = np.isnan(X_arr).any()
        if has_inf:
            _assert_finite(X_arr, "X")
        if has_nan:
            raise ValueError("X contains NaN values. Handle missing data before fitting.")
    elif not allow_missing:
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
