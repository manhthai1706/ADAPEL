import numpy as np
import pytest
from adapel.stacking import fit_stacking, MIN_COEF_ACTIVE
from adapel.base import PositiveStacking


class TestFitStacking:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n, nb = 200, 3
        oof = rng.normal(0, 1, (n, nb))
        pseudo = oof[:, 0] * 0.5 + oof[:, 1] * 0.3 + rng.normal(0, 0.1, n)
        sw = np.ones(n)
        result = fit_stacking(oof, pseudo, sw)
        assert isinstance(result, PositiveStacking)
        assert len(result.coef_) == nb
        assert np.all(result.coef_ >= 0)
        assert result.coef_.sum() > 0

    def test_positive_coefficients(self):
        rng = np.random.default_rng(42)
        n, nb = 100, 3
        oof = rng.normal(0, 1, (n, nb))
        pseudo = oof[:, 0] * 0.8 + rng.normal(0, 0.1, n)
        sw = np.ones(n)
        result = fit_stacking(oof, pseudo, sw)
        assert np.all(result.coef_ >= 0)

    def test_all_coefs_nonnegative(self):
        rng = np.random.default_rng(42)
        n, nb = 100, 4
        oof = rng.normal(0, 1, (n, nb))
        pseudo = rng.normal(0, 1, n)  # pure noise
        sw = np.ones(n)
        result = fit_stacking(oof, pseudo, sw, lam_scale=10.0)
        assert np.all(result.coef_ >= 0)
        assert result.coef_.sum() > 0


class TestPositiveStacking:
    def test_predict(self):
        stack = PositiveStacking(np.array([0.3, 0.7]), 0.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        preds = stack.predict(X)
        expected = X @ np.array([0.3, 0.7])
        np.testing.assert_array_almost_equal(preds, expected)
