"""Tests for clinical analysis module."""
import numpy as np
import pytest

from model import ADAPEL
from model.analysis import balance_check, sample_size_report


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(0, 1, (n, 4))
    T = (rng.uniform(0, 1, n) > 0.5).astype(float)
    true_cate = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    mu0 = 0.2 * X[:, 0] - 0.1 * X[:, 1]
    Y = np.where(T == 1, mu0 + true_cate, mu0) + rng.normal(0, 0.1, n)
    model = ADAPEL(n_folds=2, mode="fast", verbose=False).fit(X, T, Y)
    return model, X, T, Y


class TestBalanceCheck:
    def test_unweighted_only(self):
        X = np.random.randn(100, 3)
        T = np.array([0] * 50 + [1] * 50)
        result = balance_check(X, T)
        assert "smd_unweighted" in result
        assert "smd_unweighted_max" in result
        assert "threshold_exceeded" in result

    def test_with_weights(self):
        X = np.random.randn(100, 3)
        T = np.array([0] * 50 + [1] * 50)
        w = np.ones(100)
        result = balance_check(X, T, weights=w)
        assert "smd_weighted" in result
        assert "smd_improvement" in result


class TestSampleSizeReport:
    def test_report_structure(self):
        X = np.random.randn(300, 5)
        T = np.array([0] * 150 + [1] * 150)
        report = sample_size_report(None, X, T)
        assert "n" in report
        assert "n_treated" in report
        assert "n_control" in report
        assert "adequate" in report
        assert "warnings" in report

    def test_small_data_warnings(self):
        X = np.random.randn(50, 3)
        T = np.array([0] * 45 + [1] * 5)
        report = sample_size_report(None, X, T)
        if report["n_treated"] < 50:
            assert len(report["warnings"]) > 0


class TestSubgroupAnalysis:
    def test_subgroup_basic(self, fitted_model):
        model, X, T, Y = fitted_model
        result = model.subgroup_analysis(X, T, Y, n_bins=3)
        assert "overall_ate" in result
        assert "subgroups" in result
        assert len(result["subgroups"]) > 0

    def test_subgroup_custom(self, fitted_model):
        model, X, T, Y = fitted_model
        subgroups = {"even_idx": np.arange(X.shape[0]) % 2 == 0}
        result = model.subgroup_analysis(
            X, T, Y, subgroups=subgroups, feature_names=["F1", "F2", "F3", "F4"]
        )
        assert len(result["subgroups"]) == 1
        assert result["subgroups"][0]["name"] == "even_idx"


class TestVariableImportance:
    def test_importance_shape(self, fitted_model):
        model, X, T, Y = fitted_model
        result = model.variable_importance(X, n_repeats=5)
        assert result["importances_mean"].shape == (4,)
        assert result["importances_std"].shape == (4,)
        assert len(result["feature_names"]) == 4

    def test_importance_with_names(self, fitted_model):
        model, X, T, Y = fitted_model
        names = ["Age", "BP", "HR", "BMI"]
        result = model.variable_importance(X, feature_names=names, n_repeats=3)
        assert result["feature_names"] == names


class TestNegativeControl:
    def test_placebo(self, fitted_model):
        model, X, T, Y = fitted_model
        result = model.negative_control_test(X, n_permute=20)
        assert "observed_ate" in result
        assert "p_value_placebo" in result
        assert 0 <= result["p_value_placebo"] <= 1


class TestCalibrationCheck:
    def test_calibration(self, fitted_model):
        model, X, T, Y = fitted_model
        result = model.calibration_check(X, n_groups=5)
        assert "groups" in result
        assert "calib_error_overall" in result
        assert len(result["groups"]) > 0


class TestFairnessReport:
    def test_fairness(self, fitted_model):
        model, X, T, Y = fitted_model
        rng = np.random.default_rng(42)
        protected = {
            "gender": rng.binomial(1, 0.5, X.shape[0]),
            "race": rng.choice([0, 1, 2], X.shape[0]),
        }
        result = model.fairness_report(X, protected)
        assert "groups" in result
        assert len(result["groups"]) > 0
        for g in result["groups"]:
            assert "disparity_vs_complement" in g
            assert "p_value" in g


class TestSaveLoad:
    def test_save_load(self, fitted_model, tmp_path):
        model, X, T, Y = fitted_model
        path = str(tmp_path / "test_model.joblib")
        model.save(path)
        loaded = ADAPEL.load(path)
        pred_orig = model.predict(X)
        pred_loaded = loaded.predict(X)
        np.testing.assert_array_almost_equal(pred_orig, pred_loaded)

    def test_save_not_fitted(self):
        model = ADAPEL()
        with pytest.raises(Exception):
            model.save("model.joblib")


class TestAuditTrail:
    def test_audit_after_fit(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        T = (rng.uniform(0, 1, 200) > 0.5).astype(float)
        Y = rng.normal(0, 1, 200)
        model = ADAPEL(n_folds=2, mode="fast").fit(X, T, Y)
        audit = model.get_audit_trail()
        assert audit is not None
        assert "version" in audit
        assert "timestamp" in audit
        assert "params" in audit
        assert "sample_size_adequate" in audit
        assert audit["n_samples"] == 200

    def test_audit_before_fit(self):
        model = ADAPEL()
        assert model.get_audit_trail() is None


class TestBalanceCheckMethod:
    def test_balance_check_method(self, fitted_model):
        model, X, T, Y = fitted_model
        result = model.balance_check(X, T)
        assert "smd_unweighted" in result
        assert "threshold_exceeded" in result
