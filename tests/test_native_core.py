import numpy as np
import pytest

from bankai_random_forest import _core
from bankai_random_forest import BankaiRandomForestClassifier


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


def test_exact_extra_trees_is_repeatable_across_thread_counts():
    x = np.arange(64, dtype=np.float64).reshape(64, 1)
    y = (x[:, 0] > 31).astype(np.intp)

    def fit(n_jobs):
        forest = _core.NativeForest()
        forest.fit(
            x, y, n_estimators=9, max_features=1, random_state=4,
            n_jobs=n_jobs, bootstrap=False, max_depth=1, extra_trees=True,
        )
        return np.asarray(forest.predict(x)), forest.shap_tree_arrays()

    serial = fit(1)
    parallel = fit(3)
    np.testing.assert_array_equal(serial[0], parallel[0])
    serial_thresholds = sorted(float(tree[3][0]) for tree in serial[1])
    parallel_thresholds = sorted(float(tree[3][0]) for tree in parallel[1])
    assert serial_thresholds == parallel_thresholds
    assert all(0.0 <= threshold < 63.0 for threshold in serial_thresholds)
    assert len(set(serial_thresholds)) > 1


def test_exact_extra_trees_supports_weights_missing_values_and_leaf_limits():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [np.nan], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.intp)
    model = BankaiRandomForestClassifier(
        n_estimators=3, max_features=1, random_state=9, bootstrap=False,
        max_depth=3, max_leaf_nodes=3, min_samples_leaf=2,
        extra_trees=True, max_bins=None,
    )
    model.fit(x, y, sample_weight=np.array([1, 2, 1, 2, 1, 3, 1, 2], dtype=float))

    assert np.isfinite(model.predict_proba(x)).all()
    for tree in model._forest.shap_tree_arrays():
        features = np.asarray(tree[2])
        thresholds = np.asarray(tree[3])[features >= 0]
        assert len(thresholds) <= 2
        assert np.all((thresholds >= 0.0) & (thresholds <= 6.0))


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
