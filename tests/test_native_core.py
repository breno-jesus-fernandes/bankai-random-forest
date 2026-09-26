import numpy as np

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
