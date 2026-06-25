import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from adapel.base import (
    PositiveStacking,
    BaseMetaLearner,
    fit_w,
    scale_estimator,
    lighter,
    lighten,
)


class TestScaleEstimator:
    def test_scales_n_estimators(self):
        from sklearn.ensemble import ExtraTreesRegressor
        est = ExtraTreesRegressor(n_estimators=200, n_jobs=1)
        scaled = scale_estimator(est, factor=0.5, min_val=30)
        assert scaled.n_estimators == 100

    def test_scales_max_iter(self):
        est = HistGradientBoostingRegressor(max_iter=300)
        scaled = scale_estimator(est, factor=0.5, min_val=30)
        assert scaled.max_iter == 150

    def test_min_val_floor(self):
        est = HistGradientBoostingRegressor(max_iter=40)
        scaled = scale_estimator(est, factor=0.5, min_val=30)
        assert scaled.max_iter == 30  # 40*0.5=20 < 30 -> clamped to 30

    def test_sets_n_jobs_neg1(self):
        from sklearn.ensemble import ExtraTreesRegressor
        est = ExtraTreesRegressor(n_estimators=100, n_jobs=2)
        scaled = scale_estimator(est, factor=0.5, min_val=30)
        assert scaled.n_jobs == -1

    def test_lighter_alias(self):
        est = HistGradientBoostingRegressor(max_iter=100)
        scaled = lighter(est, factor=0.5)
        assert scaled.max_iter == 50

    def test_lighten_alias(self):
        est = HistGradientBoostingRegressor(max_iter=100)
        scaled = lighten(est, factor=0.7)
        assert scaled.max_iter == 70


class TestFitW:
    def test_with_sample_weight(self):
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        sw = np.ones(20)
        est = Ridge(alpha=1.0)
        result = fit_w(est, X, y, sw)
        assert hasattr(result, "coef_")

    def test_fallback_without_weight(self):
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        sw = np.ones(20)

        class NoWeightEstimator(Ridge):
            def fit(self, X, y, sample_weight=None):
                if sample_weight is not None:
                    raise TypeError("I do not accept weights")
                return super().fit(X, y)

        est = NoWeightEstimator(alpha=1.0)
        result = fit_w(est, X, y, sw)
        assert hasattr(result, "coef_")


class TestPositiveStacking:
    def test_init_and_predict(self):
        stack = PositiveStacking(np.array([0.4, 0.6]), 0.1)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        preds = stack.predict(X)
        expected = X @ np.array([0.4, 0.6]) + 0.1
        np.testing.assert_array_almost_equal(preds, expected)


class DummyMetaLearner(BaseMetaLearner):
    def fit(self, X, T, Y):
        self._coef = np.mean(Y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._coef)


class TestBaseMetaLearner:
    def test_ate(self):
        model = DummyMetaLearner().fit(None, None, np.array([1.0, 2.0, 3.0]))
        ate = model.estimate_ate(np.zeros((5, 2)))
        assert ate == 2.0

    def test_att(self):
        model = DummyMetaLearner().fit(None, None, np.array([1.0, 2.0, 3.0]))
        X = np.zeros((5, 2))
        T = np.array([0, 1, 0, 1, 0])
        att = model.estimate_att(X, T)
        assert att == 2.0

    def test_atc(self):
        model = DummyMetaLearner().fit(None, None, np.array([1.0, 2.0, 3.0]))
        X = np.zeros((5, 2))
        T = np.array([0, 1, 0, 1, 0])
        atc = model.estimate_atc(X, T)
        assert atc == 2.0
