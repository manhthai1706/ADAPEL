"""Unit tests for diagnostics module."""
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from adapel import ADAPEL


def _fit_tiny_model():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(0, 1, (n, 3))
    T = (rng.uniform(0, 1, n) > 0.5).astype(float)
    true_cate = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    mu0 = 0.2 * X[:, 0] - 0.1 * X[:, 1]
    Y = np.where(T == 1, mu0 + true_cate, mu0) + rng.normal(0, 0.1, n)
    model = ADAPEL(n_folds=2, mode="fast", verbose=False).fit(X, T, Y)
    return model, X


class TestDiagnostics:
    def test_compute_diagnostics_shape(self):
        model, X = _fit_tiny_model()
        d = model.get_diagnostics(X)
        assert "propensity" in d
        assert "alpha" in d
        assert "meta_weights" in d
        assert len(d["propensity"]) == X.shape[0]
        assert len(d["alpha"]) == X.shape[0]

    def test_meta_weights_positive(self):
        model, X = _fit_tiny_model()
        d = model.get_diagnostics(X)
        assert np.all(d["meta_weights"] >= 0)


class TestEValue:
    def test_binary_outcome(self):
        model, X = _fit_tiny_model()
        e_val = model.estimate_e_value(X, outcome_type="binary")
        assert isinstance(e_val, float)
        assert e_val >= 1.0

    def test_continuous_outcome(self):
        model, X = _fit_tiny_model()
        e_val = model.estimate_e_value(X, outcome_type="continuous")
        assert isinstance(e_val, float)
        assert e_val >= 1.0


class TestSurrogate:
    def test_explain_cate_surrogate(self):
        model, X = _fit_tiny_model()
        rules = model.explain_cate_surrogate(
            X, feature_names=["F1", "F2", "F3"], max_depth=2
        )
        assert isinstance(rules, str)
        assert len(rules) > 0
