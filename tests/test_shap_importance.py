import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

from bankai_random_forest import BankaiRandomForestClassifier


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("max_bins", [None, 8])
def test_shap_global_matches_reduction_of_native_tree_shap(classes, max_bins):
    rng = np.random.RandomState(12)
    x = rng.normal(size=(90, 5))
    x[::13, 2] = np.nan
    y = np.argmax(np.column_stack([x[:, 0], x[:, 1], -x[:, 0]])[:, :classes], axis=1)
    model = BankaiRandomForestClassifier(
        n_estimators=9, max_features=None, random_state=27, max_bins=max_bins,
        importance_type="shap", shap_mode="explicit",
    ).fit(x, y)
    rows = x[:17]
    values, _ = model._native_tree_shap_for_benchmark(rows)
    expected = np.abs(values).mean(axis=(0, 2))
    expected /= expected.sum()
    np.testing.assert_allclose(model.shap_importances(rows), expected, rtol=1e-11, atol=1e-12)


def test_sampled_shap_is_repeatable_and_feature_importance_is_normalized():
    rng = np.random.RandomState(4)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] > 0).astype(int)
    kwargs = dict(n_estimators=5, random_state=31, importance_type="shap", shap_sample_size=23)
    first = BankaiRandomForestClassifier(**kwargs).fit(x, y)
    second = BankaiRandomForestClassifier(**kwargs).fit(x, y)
    np.testing.assert_allclose(first.feature_importances_, second.feature_importances_)
    np.testing.assert_allclose(first.feature_importances_.sum(), 1.0)


def test_sampled_mode_uses_exactly_seeded_subset_without_replacement():
    rng = np.random.RandomState(18)
    x = rng.normal(size=(64, 5))
    y = (x[:, 0] > 0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=7,
        random_state=39,
        importance_type="shap",
        shap_sample_size=13,
    ).fit(x, y)

    selected_rows = np.sort(check_random_state(39).choice(64, size=13, replace=False))
    expected = model.shap_importances(model._fit_X[selected_rows])
    np.testing.assert_allclose(model.feature_importances_, expected, rtol=0, atol=0)


def test_sampled_mode_uses_every_row_when_sample_size_exceeds_training_size():
    rng = np.random.RandomState(21)
    x = rng.normal(size=(24, 3))
    y = (x[:, 0] > 0).astype(int)
    sampled = BankaiRandomForestClassifier(
        n_estimators=5, random_state=11, importance_type="shap",
        shap_mode="sampled", shap_sample_size=100,
    ).fit(x, y)
    all_rows = BankaiRandomForestClassifier(
        n_estimators=5, random_state=11, importance_type="shap", shap_mode="all",
    ).fit(x, y)
    np.testing.assert_array_equal(sampled.feature_importances_, all_rows.feature_importances_)


def test_shap_importances_parallel_workers_match_single_worker():
    rng = np.random.RandomState(23)
    x = rng.normal(size=(120, 6))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=12, max_features=None, random_state=8, n_jobs=1,
    ).fit(x, y)
    single = model.shap_importances(x[:37])
    model.n_jobs = 3
    parallel = model.shap_importances(x[:37])
    np.testing.assert_allclose(parallel, single, rtol=1e-12, atol=1e-13)


def test_shap_importances_accepts_csr_and_missing_values():
    rng = np.random.RandomState(9)
    x = rng.normal(size=(60, 4))
    y = (x[:, 0] > 0).astype(int)
    model = BankaiRandomForestClassifier(n_estimators=4, random_state=3).fit(x, y)
    np.testing.assert_allclose(model.shap_importances(sparse.csr_matrix(x)), model.shap_importances(x))


@pytest.mark.parametrize("max_bins", [None, 8])
@pytest.mark.parametrize("sparse_format", [sparse.csr_matrix, sparse.csc_matrix])
def test_shap_importances_handles_sparse_missing_values_and_histograms(max_bins, sparse_format):
    rng = np.random.RandomState(30)
    x = rng.normal(size=(72, 5))
    x[::11, 3] = np.nan
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=7, max_bins=max_bins, max_features=None, random_state=41,
    ).fit(x, y)
    dense_values = model.shap_importances(x[:19])
    sparse_values = model.shap_importances(sparse_format(x[:19]))
    np.testing.assert_allclose(sparse_values, dense_values, rtol=1e-12, atol=1e-13)


def test_shap_importances_zero_when_model_has_no_splits():
    x = np.ones((20, 4))
    y = np.array([0, 1] * 10)
    model = BankaiRandomForestClassifier(
        n_estimators=5, random_state=4, importance_type="shap", shap_mode="all",
    ).fit(x, y)

    np.testing.assert_array_equal(model.feature_importances_, np.zeros(x.shape[1]))
    assert np.isfinite(model.feature_importances_).all()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"shap_mode": "random"}, "shap_mode must be 'sampled', 'all', or 'explicit'"),
        ({"shap_sample_size": 0}, "shap_sample_size must be a positive integer"),
        ({"shap_sample_size": True}, "shap_sample_size must be a positive integer"),
        ({"shap_sample_size": 2.5}, "shap_sample_size must be a positive integer"),
    ],
)
def test_shap_options_reject_invalid_values(options, message):
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = BankaiRandomForestClassifier(importance_type="shap", **options)
    with pytest.raises(ValueError, match=message):
        model.fit(x, y)


def test_shap_importances_validates_fitted_state_shape_and_finite_values():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    unfitted = BankaiRandomForestClassifier()
    with pytest.raises(NotFittedError):
        unfitted.shap_importances(x)

    fitted = BankaiRandomForestClassifier(n_estimators=3, random_state=2).fit(x, y)
    with pytest.raises(ValueError):
        fitted.shap_importances(x[:, :-1])
    infinite_row = np.zeros((1, x.shape[1]))
    infinite_row[0, 0] = np.inf
    with pytest.raises(ValueError):
        fitted.shap_importances(infinite_row)
    with pytest.raises(ValueError):
        fitted.shap_importances(np.empty((0, x.shape[1])))


def test_shap_importances_remain_identical_after_joblib_round_trip(tmp_path):
    rng = np.random.RandomState(50)
    x = rng.normal(size=(80, 6))
    y = (x[:, 0] - x[:, 1] > 0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=8,
        random_state=51,
        importance_type="shap",
        shap_mode="sampled",
        shap_sample_size=17,
    ).fit(x, y)
    path = tmp_path / "shap-model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(restored.feature_importances_, model.feature_importances_)
    np.testing.assert_allclose(restored.shap_importances(x[:12]), model.shap_importances(x[:12]))


def test_all_mode_uses_all_training_rows_and_explicit_mode_defers_calculation():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(45, 4))
    y = (x[:, 0] > 0).astype(int)
    all_model = BankaiRandomForestClassifier(
        n_estimators=5, random_state=12, importance_type="shap", shap_mode="all",
    ).fit(x, y)
    np.testing.assert_allclose(all_model.feature_importances_, all_model.shap_importances(x))

    explicit_model = BankaiRandomForestClassifier(
        n_estimators=5, random_state=12, importance_type="shap", shap_mode="explicit",
    ).fit(x, y)
    assert explicit_model.feature_importances_ is None
    assert explicit_model.shap_importances(x[:7]).shape == (x.shape[1],)
