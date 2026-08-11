import numpy as np
import pytest

from model.core import (
    alpha,
    check_min_class,
    check_sample_size,
    clip_e,
    detect_missing,
    validate,
)


class TestValidate:
    def test_basic(self):
        X, T, Y = validate(np.ones((10, 3)), np.ones(10), np.ones(10))
        assert X.shape == (10, 3) and T.shape == (10,) and Y.shape == (10,)

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
        clipped = clip_e(np.array([0.0, 0.5, 1.0]), 0.05)
        assert clipped.tolist() == [0.05, 0.5, 0.95]

    def test_no_clip_needed(self):
        e = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_equal(clip_e(e, 0.05), e)


class TestAlpha:
    def test_extreme_propensity(self):
        a = alpha(np.array([0.0, 0.5, 1.0]), 1.0, 0.1)
        assert a[0] == 1.0 and a[2] == 1.0

    def test_mid_propensity(self):
        assert alpha(np.array([0.5]), 1.0, 0.1)[0] == 0.1

    def test_min_alpha_floor(self):
        assert alpha(np.array([0.5]), 1.0, 0.3)[0] == 0.3


class TestCheckMinClass:
    def test_ok(self):
        check_min_class(np.array([0] * 10 + [1] * 10), 5)

    def test_fail(self):
        with pytest.raises(ValueError, match="treatment arm needs at least"):
            check_min_class(np.array([0] * 3 + [1] * 10), 5)


class TestDetectMissing:
    def test_no_missing(self):
        result = detect_missing(np.ones((10, 3)))
        assert not result["has_missing"] and result["n_missing"] == 0

    def test_missing_present(self):
        X = np.ones((10, 3))
        X[0, 0] = np.nan
        result = detect_missing(X)
        assert result["has_missing"] and result["n_missing"] == 1
        assert result["col_n_missing"][0] == 1


class TestCheckSampleSize:
    def test_adequate(self):
        warnings = check_sample_size(
            np.random.default_rng(0).standard_normal((500, 5)),
            np.array([0] * 250 + [1] * 250),
        )
        assert warnings == []

    def test_small_warning(self):
        warnings = check_sample_size(
            np.random.default_rng(0).standard_normal((50, 3)),
            np.array([0] * 40 + [1] * 10),
        )
        assert any("small" in w.lower() for w in warnings)

    def test_imbalance_warning(self):
        warnings = check_sample_size(
            np.random.default_rng(0).standard_normal((200, 3)),
            np.array([0] * 190 + [1] * 10),
        )
        assert any("imbalance" in w.lower() or "arm" in w.lower() for w in warnings)
