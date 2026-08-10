import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from model.base import BaseMetaLearner, PositiveStacking, fit_w, scale_estimator


class TestScaleEstimator:
    def test_scales_n_estimators(self):
        est = ExtraTreesRegressor(n_estimators=200, n_jobs=1)
        assert scale_estimator(est, 0.5, 30).n_estimators == 100

    def test_scales_max_iter(self):
        est = HistGradientBoostingRegressor(max_iter=300)
        assert scale_estimator(est, 0.5, 30).max_iter == 150

    def test_min_val_floor(self):
        est = HistGradientBoostingRegressor(max_iter=40)
        assert scale_estimator(est, 0.5, 30).max_iter == 30

    def test_sets_n_jobs_neg1(self):
        est = ExtraTreesRegressor(n_estimators=100, n_jobs=2)
        assert scale_estimator(est, 0.5, 30).n_jobs == -1


class TestFitW:
    def test_with_sample_weight(self):
        X = np.random.default_rng(0).standard_normal((20, 2))
        y = np.random.default_rng(1).standard_normal(20)
        result = fit_w(Ridge(alpha=1.0), X, y, np.ones(20))
        assert hasattr(result, "coef_")

    def test_fallback_without_weight(self):
        class NoWeight(Ridge):
            def fit(self, X, y, sample_weight=None):
                if sample_weight is not None:
                    raise TypeError("no weights")
                return super().fit(X, y)

        X = np.random.default_rng(0).standard_normal((20, 2))
        y = np.random.default_rng(1).standard_normal(20)
        result = fit_w(NoWeight(alpha=1.0), X, y, np.ones(20))
        assert hasattr(result, "coef_")


class TestPositiveStacking:
    def test_predict(self):
        stack = PositiveStacking(np.array([0.4, 0.6]), 0.1)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = X @ np.array([0.4, 0.6]) + 0.1
        np.testing.assert_array_almost_equal(stack.predict(X), expected)


class DummyMetaLearner(BaseMetaLearner):
    def fit(self, X, T, Y):
        self._coef = float(np.mean(Y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._coef)


class TestBaseMetaLearner:
    def test_ate(self):
        m = DummyMetaLearner().fit(None, None, np.array([1.0, 2.0, 3.0]))
        assert m.estimate_ate(np.zeros((5, 2))) == 2.0

    def test_att(self):
        m = DummyMetaLearner().fit(None, None, np.array([1.0, 2.0, 3.0]))
        X = np.zeros((5, 2))
        T = np.array([0, 1, 0, 1, 0])
        assert m.estimate_att(X, T) == 2.0

    def test_atc(self):
        m = DummyMetaLearner().fit(None, None, np.array([1.0, 2.0, 3.0]))
        X = np.zeros((5, 2))
        T = np.array([0, 1, 0, 1, 0])
        assert m.estimate_atc(X, T) == 2.0
