import numpy as np
import pytest

from bankai_random_forest import _core


def test_native_forest_starts_unfitted():
    forest = _core.NativeForest()

    assert forest.is_fitted() is False


def test_native_forest_fits_and_predicts_a_separable_binary_problem():
    x = np.array(
        [[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.intp)
    forest = _core.NativeForest()

    forest.fit(x, y, n_estimators=25, max_features=1, random_state=42)

    assert forest.is_fitted() is True
    np.testing.assert_array_equal(forest.predict(x), y)


def test_histogram_extra_trees_is_repeatable_across_thread_counts():
    x = np.arange(64, dtype=np.float64).reshape(64, 1)
    y = (x[:, 0] > 31).astype(np.intp)

    def fit(n_jobs, seed=6):
        forest = _core.NativeForest()
        forest.fit(
            x, y, n_estimators=1, max_features=1, random_state=4,
            n_jobs=n_jobs, max_bins=16, bootstrap=False, max_depth=1,
            extra_trees=True, extra_seed=seed,
        )
        return np.asarray(forest.predict(x)), np.asarray(forest.feature_importances())

    serial = fit(1)
    parallel = fit(3, 6)
    np.testing.assert_array_equal(serial[0], parallel[0])
    np.testing.assert_array_equal(serial[1], parallel[1])
    assert not np.array_equal(serial[0], fit(1, 0)[0])


def test_extra_trees_requires_histogram_bins():
    forest = _core.NativeForest()
    with pytest.raises(ValueError, match="extra_trees=True requires max_bins"):
        forest.fit(
            np.array([[0.0], [1.0]]), np.array([0, 1]),
            n_estimators=1, max_features=1, random_state=0, extra_trees=True,
        )


def test_sampled_binning_seed_is_repeatable_and_independent():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 2))
    y = (x[:, 0] + 0.2 * x[:, 1] > 0).astype(np.intp)

    def fit(seed):
        forest = _core.NativeForest()
        forest.fit(
            x, y, n_estimators=1, max_features=1, random_state=2,
            n_jobs=1, max_bins=16, binning_strategy="sampled_select",
            bin_sample_size=100, binning_seed=seed, bootstrap=False, max_depth=1,
        )
        return np.asarray(forest.predict(x))

    np.testing.assert_array_equal(fit(6), fit(6))
    assert not np.array_equal(fit(6), fit(7))
