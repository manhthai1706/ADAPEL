from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .base import PositiveStacking

MIN_COEF_ACTIVE = 1e-8


def fit_stacking(
    oof: np.ndarray,
    pseudo: np.ndarray,
    sw: np.ndarray,
    lam_scale: float = 1e-3,
) -> PositiveStacking:
    """NNLS positive stacking with L2 regularisation."""
    n, nb = oof.shape
    sqrt_w = np.sqrt(np.maximum(sw / max(float(sw.mean()), 1e-10), 1e-10))
    A = np.column_stack([oof * sqrt_w[:, None], sqrt_w])
    b = pseudo * sqrt_w

    lam = lam_scale * max(float(b.std()), 1e-10) / nb
    reg_block = np.zeros((nb, nb + 1))
    reg_block[:, :nb] = np.eye(nb) * np.sqrt(lam)
    intercept_block = np.zeros((1, nb + 1))
    A = np.vstack([A, reg_block, intercept_block])
    b = np.concatenate([b, np.zeros(nb), [0.0]])

    coefs, _ = nnls(A, b, maxiter=500 * nb)
    base_w, intercept = coefs[:nb], coefs[nb]

    if base_w.sum() < MIN_COEF_ACTIVE:
        return PositiveStacking(np.ones(nb) / nb, 0.0)
    return PositiveStacking(base_w, intercept)
