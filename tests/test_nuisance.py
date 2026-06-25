import numpy as np
import pytest
from adapel.nuisance import alpha, clip_e, validate, check_min_class, _assert_finite


class TestValidate:
    def test_basic(self):
        X, T, Y = validate(np.ones((10, 3)), np.ones(10), np.ones(10))
        assert X.shape == (10, 3)
        assert T.shape == (10,)
        assert Y.shape == (10,)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="rows differ"):
            validate(np.ones((10, 3)), np.ones(5), np.ones(10))

    def test_non_binary_treatment(self):
        with pytest.raises(ValueError, match="must be binary"):
            validate(np.ones((10, 3)), np.array([0, 1, 2] * 3 + [0]), np.ones(10))

    def test_nan_values(self):
        X = np.ones((10, 3))
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            validate(X, np.ones(10), np.ones(10))

    def test_inf_values(self):
        X = np.ones((10, 3))
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite"):
            validate(X, np.ones(10), np.ones(10))


class TestClipE:
    def test_clipping(self):
        e = np.array([0.0, 0.5, 1.0])
        clipped = clip_e(e, clip_propensity=0.05)
        assert clipped[0] == 0.05
        assert clipped[1] == 0.5
        assert clipped[2] == 0.95

    def test_no_clip_needed(self):
        e = np.array([0.1, 0.5, 0.9])
        clipped = clip_e(e, clip_propensity=0.05)
        np.testing.assert_array_equal(clipped, e)


class TestAlpha:
    def test_extreme_propensity(self):
        e = np.array([0.0, 0.5, 1.0])
        a = alpha(e, fusion_gamma=1.0, min_alpha=0.1)
        assert a[0] == 1.0  # e=0 -> alpha=1 (X-learner)
        assert a[2] == 1.0  # e=1 -> alpha=1 (X-learner)

    def test_mid_propensity(self):
        e = np.array([0.5])
        a = alpha(e, fusion_gamma=1.0, min_alpha=0.1)
        assert a[0] == 0.1  # e=0.5 -> alpha=min_alpha (DR-learner)

    def test_min_alpha_floor(self):
        e = np.array([0.5])
        a = alpha(e, fusion_gamma=1.0, min_alpha=0.3)
        assert a[0] == 0.3


class TestCheckMinClass:
    def test_ok(self):
        T = np.array([0] * 10 + [1] * 10)
        check_min_class(T, 5)

    def test_fail(self):
        T = np.array([0] * 3 + [1] * 10)
        with pytest.raises(ValueError, match="treatment arm needs at least"):
            check_min_class(T, 5)


class TestAssertFinite:
    def test_ok(self):
        _assert_finite(np.array([1.0, 2.0]), "X")

    def test_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            _assert_finite(np.array([1.0, np.nan]), "X")

    def test_inf(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            _assert_finite(np.array([1.0, np.inf]), "X")
