"""Integration tests for ADAPEL end-to-end pipeline."""
import numpy as np
import pytest
from adapel import ADAPEL


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, (n, 5))
    logit = -1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1]
    T = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)
    true_cate = 0.8 * X[:, 0] + 0.5 * np.sin(X[:, 2])
    mu0 = 0.3 * X[:, 0] - 0.2 * X[:, 1]
    Y = np.where(T == 1, mu0 + true_cate, mu0) + rng.normal(0, 0.2, n)
    return X, T, Y, true_cate


class TestADAPELIntegration:
    def test_fit_predict_shape(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(n_folds=3, mode="fast", verbose=False).fit(X, T, Y)
        cate = model.predict(X)
        assert cate.shape == (X.shape[0],)

    def test_potential_outcomes(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(n_folds=3, mode="fast", verbose=False).fit(X, T, Y)
        y0, y1 = model.predict_potential_outcomes(X)
        assert y0.shape == (X.shape[0],)
        assert y1.shape == (X.shape[0],)

    def test_counterfactual(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(n_folds=3, mode="fast", verbose=False).fit(X, T, Y)
        cf = model.predict_counterfactual(X, T)
        assert cf.shape == (X.shape[0],)

    def test_ate_estimation(self, synthetic_data):
        X, T, Y, true_cate = synthetic_data
        model = ADAPEL(n_folds=3, mode="fast", verbose=False).fit(X, T, Y)
        ate = model.estimate_ate(X)
        true_ate = float(np.mean(true_cate))
        assert abs(ate - true_ate) < 1.0  # generous tolerance for fast mode

    def test_att_atc(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(n_folds=3, mode="fast", verbose=False).fit(X, T, Y)
        att = model.estimate_att(X, T)
        atc = model.estimate_atc(X, T)
        assert isinstance(att, float)
        assert isinstance(atc, float)

    def test_feature_select(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(
            n_folds=3, mode="fast", verbose=False,
            feature_select=True, feature_frac=0.6,
        ).fit(X, T, Y)
        cate = model.predict(X)
        assert cate.shape == (X.shape[0],)

    def test_bootstrap_predict_clinical(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(n_folds=2, mode="fast", verbose=False).fit(X, T, Y)
        model.fit_bootstrap(X, T, Y, n_bootstrap=5)
        clin = model.predict_clinical(X)
        assert "cate" in clin
        assert "lower_ci" in clin
        assert "upper_ci" in clin
        assert clin["lower_ci"] is not None
        assert clin["upper_ci"] is not None
        assert clin["cate"].shape == (X.shape[0],)

    def test_fit_twice_returns_new(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(n_folds=2, mode="fast")
        model.fit(X, T, Y)
        cate1 = model.predict(X).copy()
        X2 = X + np.random.randn(*X.shape) * 0.1
        model.fit(X2, T, Y)
        cate2 = model.predict(X2)
        assert not np.allclose(cate1, cate2)

    def test_get_diagnostics(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        model = ADAPEL(n_folds=3, mode="fast", verbose=False).fit(X, T, Y)
        d = model.get_diagnostics(X)
        assert "meta_weights" in d
        assert "pct_dr_dominant" in d
        assert "ensemble_std" in d

    def test_not_fitted_error(self):
        model = ADAPEL()
        with pytest.raises(Exception):
            model.predict(np.ones((5, 3)))

    def test_all_modes(self, synthetic_data):
        X, T, Y, _ = synthetic_data
        for mode in ("fast", "balanced", "accurate"):
            model = ADAPEL(n_folds=2, mode=mode, verbose=False).fit(X, T, Y)
            cate = model.predict(X)
            assert cate.shape == (X.shape[0],)
