from __future__ import annotations
import numpy as np
from scipy.optimize import nnls
from .base import PositiveStacking

# Minimum stacking weight to consider a base learner "active"
MIN_COEF_ACTIVE = 1e-8


def fit_stacking(
    oof: np.ndarray,
    pseudo: np.ndarray,
    sw: np.ndarray,
    lam_scale: float = 1e-3,
) -> PositiveStacking:
    """NNLS positive stacking with L2 regularisation.

    Parameters
    ----------
    oof : (n_samples, n_learners)
        Out-of-fold predictions from each base learner.
    pseudo : (n_samples,)
        Pseudo-outcome target.
    sw : (n_samples,)
        Sample weights.
    lam_scale : float
        Global scaling factor for L2 penalty.

    Returns
    -------
    PositiveStacking with positive coefficients and optional intercept.
    """
    nb = oof.shape[1]
    sw_n = sw / max(float(sw.mean()), 1e-10)
    sqrt_w = np.sqrt(np.maximum(sw_n, 1e-10))
    A = np.column_stack([oof * sqrt_w[:, None], sqrt_w])
    b = pseudo * sqrt_w
    lam = lam_scale * max(float(b.std()), 1e-10) / nb
    A_reg = np.column_stack([np.eye(nb) * np.sqrt(lam), np.zeros(nb)])
    A = np.vstack([A, A_reg])
    b = np.concatenate([b, np.zeros(nb)])
    coefs, _ = nnls(A, b, maxiter=500 * nb)
    base_w, intercept = coefs[:nb], coefs[nb]
    if base_w.sum() < MIN_COEF_ACTIVE:
        base_w = np.ones(nb) / nb
        intercept = 0.0
    return PositiveStacking(base_w, intercept)
