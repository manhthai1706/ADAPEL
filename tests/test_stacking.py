import numpy as np
from model.core import PositiveStacking, fit_stacking


class TestFitStacking:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n, nb = 200, 3
        oof = rng.standard_normal((n, nb))
        pseudo = oof[:, 0] * 0.5 + oof[:, 1] * 0.3 + rng.normal(0, 0.1, n)
        result = fit_stacking(oof, pseudo, np.ones(n))
        assert isinstance(result, PositiveStacking)
        assert len(result.coef_) == nb
        assert np.all(result.coef_ >= 0)
        assert result.coef_.sum() > 0

    def test_pure_noise_returns_uniform(self):
        rng = np.random.default_rng(42)
        n, nb = 100, 4
        oof = rng.standard_normal((n, nb))
        result = fit_stacking(oof, rng.standard_normal(n), np.ones(n), lam_scale=10.0)
        assert np.all(result.coef_ >= 0)
        assert result.coef_.sum() > 0


class TestPositiveStacking:
    def test_predict_no_intercept(self):
        stack = PositiveStacking(np.array([0.3, 0.7]), 0.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(stack.predict(X), X @ np.array([0.3, 0.7]))
