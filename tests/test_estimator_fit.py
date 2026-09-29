import numpy as np
import pandas as pd
import pickle
import pytest
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

from bankai_random_forest import BankaiRandomForestClassifier


@pytest.fixture
def separable_data():
    x = np.array(
        [[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    y = np.array(["cold", "cold", "cold", "cold", "hot", "hot", "hot", "hot"])
    return x, y


def test_fit_encodes_labels_and_predict_returns_original_labels(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42)

    fitted = classifier.fit(x, y)

    assert fitted is classifier
    np.testing.assert_array_equal(classifier.classes_, np.array(["cold", "hot"]))
    assert classifier.n_classes_ == 2
    assert classifier.n_features_in_ == 1
    np.testing.assert_array_equal(classifier.predict(x), y)


def test_dense_fit_is_row_order_invariant_with_ties_and_missing_values():
    rng = np.random.RandomState(91)
    x = rng.randint(-2, 3, size=(96, 6)).astype(np.float64)
    x[::9, 2] = np.nan
    x[1::13, 0] = -0.0
    x[2::13, 0] = 0.0
    y = rng.randint(0, 3, size=x.shape[0])
    _, encoded_y = np.unique(y, return_inverse=True)
    keys = (encoded_y, *(x[:, index] for index in range(x.shape[1] - 1, -1, -1)))
    order = np.lexsort(keys)
    params = dict(
        n_estimators=9,
        max_features=None,
        random_state=91,
        oob_score=True,
        max_bins=8,
    )

    direct = BankaiRandomForestClassifier(**params).fit(x, y)
    preordered = BankaiRandomForestClassifier(**params).fit(x[order], y[order])

    np.testing.assert_array_equal(direct.predict(x), preordered.predict(x))
    np.testing.assert_allclose(direct.predict_proba(x), preordered.predict_proba(x))
    np.testing.assert_allclose(direct.feature_importances_, preordered.feature_importances_)
    np.testing.assert_allclose(
        direct.oob_decision_function_, preordered.oob_decision_function_
    )


def test_predict_proba_and_log_proba_follow_the_fitted_classes(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    probabilities = classifier.predict_proba(x)

    assert probabilities.shape == (x.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(classifier.classes_[probabilities.argmax(axis=1)], y)
    np.testing.assert_allclose(classifier.predict_log_proba(x), np.log(probabilities))


def test_fit_sets_normalized_feature_importances(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    assert classifier.feature_importances_.shape == (1,)
    np.testing.assert_allclose(classifier.feature_importances_.sum(), 1.0)


def test_gain_importance_ranks_predictive_features_above_noise():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)

    classifier = BankaiRandomForestClassifier(n_estimators=50, random_state=42).fit(x, y)

    assert classifier.importance_type == "gain"
    assert classifier.feature_importances_[0] > classifier.feature_importances_[2]
    assert classifier.feature_importances_[1] > classifier.feature_importances_[2]
    np.testing.assert_allclose(classifier.feature_importances_.sum(), 1.0)


def test_max_bins_enables_histogram_training_and_prediction(separable_data):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, max_bins=4, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)
    assert classifier.max_bins == 4


@pytest.mark.parametrize("max_bins", [1, 256, True, 2.5])
def test_max_bins_rejects_invalid_histogram_resolution(separable_data, max_bins):
    x, y = separable_data

    with pytest.raises(ValueError, match="max_bins must be None or an integer in \\[2, 255\\]"):
        BankaiRandomForestClassifier(max_bins=max_bins).fit(x, y)


@pytest.mark.parametrize(
    "strategy", ["exact_sort", "sampled_sort", "exact_select", "sampled_select"]
)
def test_binning_strategies_fit_and_keep_predictive_signal(strategy):
    rng = np.random.RandomState(101)
    x = rng.normal(size=(800, 8))
    y = (x[:, 0] - 0.7 * x[:, 1] > 0.0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=15,
        max_depth=8,
        max_features=None,
        max_bins=16,
        binning_strategy=strategy,
        bin_sample_size=256,
        importance_type="gain",
        n_jobs=2,
        random_state=19,
    ).fit(x, y)

    assert model.score(x, y) > 0.85
    assert model.predict(x).shape == y.shape
    assert model.binning_strategy == strategy


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"binning_strategy": "quick_sort"}, "binning_strategy must be one of"),
        ({"bin_sample_size": 0}, "bin_sample_size must be a positive integer"),
        ({"bin_sample_size": True}, "bin_sample_size must be a positive integer"),
    ],
)
def test_binning_options_reject_invalid_values(separable_data, options, message):
    x, y = separable_data
    with pytest.raises(ValueError, match=message):
        BankaiRandomForestClassifier(max_bins=4, **options).fit(x, y)


def test_sampled_select_model_roundtrips_through_pickle():
    rng = np.random.RandomState(102)
    x = rng.normal(size=(240, 5))
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=5,
        max_bins=8,
        binning_strategy="sampled_select",
        bin_sample_size=120,
        random_state=12,
    ).fit(x, y)

    restored = pickle.loads(pickle.dumps(model))

    assert restored.binning_strategy == "sampled_select"
    assert restored.bin_sample_size == 120
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


def test_sampled_select_histograms_support_sparse_input():
    rng = np.random.RandomState(103)
    x = rng.normal(size=(300, 6))
    x[np.abs(x) < 0.8] = 0.0
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=7,
        max_bins=8,
        binning_strategy="sampled_select",
        bin_sample_size=100,
        random_state=13,
    ).fit(sparse.csr_matrix(x), y)

    assert model.predict(sparse.csr_matrix(x)).shape == y.shape


def test_histogram_backend_preserves_native_permutation_importance():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)

    classifier = BankaiRandomForestClassifier(
        n_estimators=30,
        max_bins=16,
        importance_type="permutation",
        random_state=42,
    ).fit(x, y)

    assert classifier.feature_importances_.shape == (3,)
    assert classifier.feature_importances_[0] > classifier.feature_importances_[2]


def test_histogram_mode_skips_constant_features_without_changing_feature_positions():
    rng = np.random.RandomState(42)
    signal = rng.normal(size=200)
    x = np.column_stack([signal, np.full(signal.size, 7.0)])
    y = (signal > 0.0).astype(int)

    classifier = BankaiRandomForestClassifier(
        n_estimators=30, max_bins=16, max_features=None, random_state=42
    ).fit(x, y)

    assert classifier.predict(x).shape == y.shape
    np.testing.assert_allclose(classifier.feature_importances_[1], 0.0)


def test_permutation_importance_type_exposes_native_oob_accuracy_decrease():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] > 0.0).astype(int)

    classifier = BankaiRandomForestClassifier(
        n_estimators=50, importance_type="permutation", random_state=42
    ).fit(x, y)

    assert classifier.feature_importances_.shape == (2,)
    assert classifier.feature_importances_[0] > classifier.feature_importances_[1]


def test_permutation_importance_type_requires_bootstrap(separable_data):
    x, y = separable_data

    with pytest.raises(ValueError, match="importance_type='permutation' requires bootstrap=True"):
        BankaiRandomForestClassifier(
            importance_type="permutation", bootstrap=False
        ).fit(x, y)


def test_permutation_importance_type_is_deterministic_across_thread_counts():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)

    sequential = BankaiRandomForestClassifier(
        n_estimators=50, n_jobs=1, importance_type="permutation", random_state=42
    ).fit(x, y)
    parallel = BankaiRandomForestClassifier(
        n_estimators=50, n_jobs=2, importance_type="permutation", random_state=42
    ).fit(x, y)

    np.testing.assert_allclose(
        sequential.feature_importances_, parallel.feature_importances_
    )


def test_rejects_unknown_importance_type(separable_data):
    x, y = separable_data

    with pytest.raises(ValueError, match="importance_type must be 'gain', 'split', or 'permutation'"):
        BankaiRandomForestClassifier(importance_type="unknown").fit(x, y)


def test_fit_accepts_uniform_sample_weight(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42)

    classifier.fit(x, y, sample_weight=np.ones(x.shape[0]))

    np.testing.assert_array_equal(classifier.predict(x), y)


def test_fit_accepts_pandas_series_sample_weight(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42)

    classifier.fit(x, y, sample_weight=pd.Series(np.ones(x.shape[0])))

    np.testing.assert_array_equal(classifier.predict(x), y)


def test_class_weight_changes_the_majority_vote():
    x = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 1])

    classifier = BankaiRandomForestClassifier(
        n_estimators=25,
        class_weight={0: 0.001, 1: 1000.0},
        random_state=42,
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict([[0.0]]), np.array([1]))


@pytest.mark.parametrize("criterion", ["entropy", "log_loss"])
def test_fit_supports_information_gain_criteria(separable_data, criterion):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, criterion=criterion, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)


@pytest.mark.parametrize("max_features", [None, "log2", 1, 1.0, 0.5])
def test_fit_supports_sklearn_max_features_forms(separable_data, max_features):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, max_features=max_features, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)


def test_fit_supports_a_maximum_tree_depth(separable_data):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, max_depth=1, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)


@pytest.mark.parametrize(
    ("parameter", "value"), [("min_samples_split", 3), ("min_samples_leaf", 2)]
)
def test_fit_supports_minimum_sample_controls(separable_data, parameter, value):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, random_state=42, **{parameter: value}
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)


def test_fit_supports_training_without_bootstrap(separable_data):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, bootstrap=False, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)


@pytest.mark.parametrize("max_samples", [4, 0.5])
def test_fit_supports_bootstrap_sample_limits(separable_data, max_samples):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, max_samples=max_samples, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)


def test_fit_exposes_out_of_bag_probabilities_and_score(separable_data):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=100, oob_score=True, random_state=42
    ).fit(x, y)

    assert classifier.oob_decision_function_.shape == (x.shape[0], 2)
    assert 0.0 <= classifier.oob_score_ <= 1.0


def test_n_jobs_preserves_seeded_predictions_and_probabilities():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(50, 4))
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)
    sequential = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)
    parallel = BankaiRandomForestClassifier(
        n_estimators=25, n_jobs=2, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(parallel.predict(x), sequential.predict(x))
    np.testing.assert_allclose(parallel.predict_proba(x), sequential.predict_proba(x))


def test_warm_start_matches_a_single_fit_with_the_final_tree_count(separable_data):
    x, y = separable_data
    warmed = BankaiRandomForestClassifier(
        n_estimators=10, warm_start=True, random_state=42
    ).fit(x, y)
    warmed.set_params(n_estimators=25).fit(x, y)
    direct = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    np.testing.assert_array_equal(warmed.predict(x), direct.predict(x))
    np.testing.assert_allclose(warmed.predict_proba(x), direct.predict_proba(x))


def test_fit_supports_minimum_impurity_decrease(separable_data):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, min_impurity_decrease=0.1, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(classifier.predict(x), y)


def test_max_leaf_nodes_limits_each_tree_to_a_leaf(separable_data):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(
        n_estimators=25, max_leaf_nodes=1, random_state=42
    ).fit(x, y)

    np.testing.assert_allclose(classifier.feature_importances_, 0.0)


def test_verbose_reports_tree_construction(separable_data, capsys):
    x, y = separable_data

    BankaiRandomForestClassifier(n_estimators=3, verbose=1, random_state=42).fit(x, y)

    assert "building 3 trees" in capsys.readouterr().out


def test_integer_sample_weight_matches_repeated_training_rows():
    x = np.array(
        [[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    weights = np.array([3, 1, 2, 1, 3, 1, 2, 1])
    repeated_x = np.repeat(x, weights, axis=0)
    repeated_y = np.repeat(y, weights)
    probe = np.linspace(-3.5, 3.5, 15, dtype=np.float64).reshape(-1, 1)

    weighted = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(
        x, y, sample_weight=weights
    )
    repeated = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(
        repeated_x, repeated_y
    )

    np.testing.assert_array_equal(weighted.predict(probe), repeated.predict(probe))


def test_integer_sample_weight_matches_repetition_with_the_sklearn_check_data():
    rng = np.random.RandomState(42)
    x = rng.rand(15, 30)
    y = rng.randint(0, 3, size=15)
    weights = rng.randint(0, 5, size=15)
    repeated_x = x.repeat(repeats=weights, axis=0)
    repeated_y = y.repeat(repeats=weights)
    weighted_x, weighted_y, weighted_weights = shuffle(x, y, weights, random_state=0)
    repeated = BankaiRandomForestClassifier(random_state=0).fit(
        repeated_x, repeated_y
    )
    weighted = BankaiRandomForestClassifier(random_state=0).fit(
        weighted_x, weighted_y, sample_weight=weighted_weights
    )

    np.testing.assert_array_equal(repeated.predict(x), weighted.predict(x))


@pytest.mark.parametrize("sparse_format", [sparse.csr_matrix, sparse.csc_matrix])
@pytest.mark.parametrize("max_bins", [None, 8])
def test_sparse_input_matches_dense_fit_prediction_and_oob(separable_data, sparse_format, max_bins):
    x, y = separable_data
    options = dict(n_estimators=15, random_state=42, oob_score=True, max_bins=max_bins)
    dense = BankaiRandomForestClassifier(**options).fit(x, y)
    sparse_model = BankaiRandomForestClassifier(**options).fit(sparse_format(x), y)

    np.testing.assert_array_equal(sparse_model.predict(sparse_format(x)), dense.predict(x))
    np.testing.assert_allclose(sparse_model.predict_proba(sparse_format(x)), dense.predict_proba(x))
    np.testing.assert_allclose(sparse_model.feature_importances_, dense.feature_importances_)
    np.testing.assert_allclose(sparse_model.oob_decision_function_, dense.oob_decision_function_)
    np.testing.assert_allclose(sparse_model.oob_score_, dense.oob_score_)


@pytest.mark.parametrize("sparse_format", [sparse.csr_matrix, sparse.csc_matrix])
def test_sparse_prediction_validation(sparse_format, separable_data):
    x, y = separable_data
    model = BankaiRandomForestClassifier(n_estimators=3, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(sparse_format(np.ones((len(y), x.shape[1] + 1))))


@pytest.mark.parametrize("training_format", [sparse.csr_matrix, sparse.csc_matrix])
def test_sparse_exact_mode_matches_sklearn_predictions(training_format):
    rng = np.random.RandomState(72)
    x = rng.normal(size=(80, 6))
    x[rng.random_sample(x.shape) < 0.7] = 0.0
    y = (x[:, 0] - x[:, 2] > 0.1).astype(int)
    x_sparse = training_format(x)
    options = dict(n_estimators=21, max_features=None, random_state=19)
    bankai = BankaiRandomForestClassifier(**options).fit(x_sparse, y)
    sklearn = RandomForestClassifier(**options).fit(x_sparse, y)

    for prediction_format in (sparse.csr_matrix, sparse.csc_matrix):
        probe = prediction_format(x)
        np.testing.assert_array_equal(bankai.predict(probe), sklearn.predict(probe))
        probabilities = bankai.predict_proba(probe)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("training_format", [sparse.csr_matrix, sparse.csc_matrix])
@pytest.mark.parametrize("max_bins", [None, 8])
def test_sparse_implicit_zeros_match_dense_backend(training_format, max_bins):
    rng = np.random.RandomState(107)
    x = rng.normal(size=(72, 7))
    x[rng.random_sample(x.shape) < 0.75] = 0.0
    y = (x[:, 0] + x[:, 2] - x[:, 5] > 0.0).astype(int)
    params = dict(n_estimators=17, max_features=4, random_state=41, max_bins=max_bins, oob_score=True)
    dense = BankaiRandomForestClassifier(**params).fit(x, y)
    sparse_model = BankaiRandomForestClassifier(**params).fit(training_format(x), y)

    np.testing.assert_array_equal(sparse_model.predict(x), dense.predict(x))
    np.testing.assert_array_equal(sparse_model.predict_proba(x), dense.predict_proba(x))
    np.testing.assert_allclose(sparse_model.feature_importances_, dense.feature_importances_)
    np.testing.assert_array_equal(sparse_model.oob_decision_function_, dense.oob_decision_function_)


@pytest.mark.parametrize("sparse_format", [sparse.csr_matrix, sparse.csc_matrix])
def test_sparse_input_validation_rejects_infinite_and_bad_csr_shape(sparse_format, separable_data):
    x, y = separable_data
    x = x.copy()
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity|infinite"):
        BankaiRandomForestClassifier().fit(sparse_format(x), y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_histogram_input_validation_rejects_infinite(dtype):
    x = np.array([[0.0, 1.0], [1.0, 0.0], [np.inf, 1.0], [3.0, 2.0]], dtype=dtype)
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="infinity|infinite"):
        BankaiRandomForestClassifier(max_bins=8, n_jobs=2).fit(x, y)


def test_sparse_fit_and_prediction_never_densify(monkeypatch):
    def fail_toarray(*args, **kwargs):
        raise AssertionError("sparse input was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", fail_toarray)
    x = sparse.random(40, 8, density=0.2, random_state=7, format="csr")
    y = np.arange(x.shape[0]) % 2
    model = BankaiRandomForestClassifier(n_estimators=5, random_state=5).fit(x, y)
    model.predict(x.tocsc())


def test_supports_multioutput_targets(separable_data):
    x, y = separable_data

    model = BankaiRandomForestClassifier(
        n_estimators=7, bootstrap=False, random_state=42
    ).fit(x, np.column_stack([y, y]))

    np.testing.assert_array_equal(model.predict(x), np.column_stack([y, y]))
    assert len(model.predict_proba(x)) == 2


def test_dataframe_column_names_are_preserved(separable_data):
    x, y = separable_data
    frame = pd.DataFrame(x, columns=["temperature"])

    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(
        frame, y
    )

    np.testing.assert_array_equal(classifier.feature_names_in_, ["temperature"])


@pytest.mark.parametrize("max_bins", [None, 8])
def test_nan_features_learn_missing_routes_and_predict_unseen_nan(max_bins):
    x = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0], [np.nan]])
    y = np.array([0, 0, 1, 1, 1, 1])
    model = BankaiRandomForestClassifier(
        n_estimators=7, max_features=1, bootstrap=False, random_state=42, max_bins=max_bins
    ).fit(x, y)
    sklearn_model = RandomForestClassifier(
        n_estimators=7, max_features=1, bootstrap=False, random_state=42
    ).fit(x, y)

    np.testing.assert_array_equal(model.predict(x), y)
    np.testing.assert_array_equal(model.predict(x), sklearn_model.predict(x))
    np.testing.assert_allclose(model.predict_proba(x), sklearn_model.predict_proba(x))
    np.testing.assert_array_equal(model.predict([[np.nan], [0.5], [3.5]]), [1, 0, 1])
    np.testing.assert_allclose(model.predict_proba([[np.nan], [0.5], [3.5]]).sum(axis=1), 1.0)
    assert model.apply(x).shape == (len(x), model.n_estimators)
    assert model.decision_path(x)[0].shape[0] == len(x)


@pytest.mark.parametrize("sparse_format", [sparse.csr_matrix, sparse.csc_matrix])
@pytest.mark.parametrize("max_bins", [None, 8])
def test_sparse_nan_routes_match_dense(sparse_format, max_bins):
    x = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0], [np.nan]])
    y = np.array([0, 0, 1, 1, 1, 1])
    params = dict(n_estimators=7, max_features=1, bootstrap=False, random_state=42, max_bins=max_bins)
    dense = BankaiRandomForestClassifier(**params).fit(x, y)
    matrix = sparse_format(x)
    sparse_model = BankaiRandomForestClassifier(**params).fit(matrix, y)

    np.testing.assert_array_equal(sparse_model.predict(matrix), dense.predict(x))
    np.testing.assert_array_equal(sparse_model.predict([[np.nan], [0.5], [3.5]]), [1, 0, 1])
    np.testing.assert_allclose(sparse_model.predict_proba(matrix), dense.predict_proba(x))


def test_nan_rows_work_with_oob_predictions():
    rng = np.random.RandomState(211)
    x = rng.normal(size=(120, 4))
    x[rng.random_sample(x.shape) < 0.15] = np.nan
    y = (np.nan_to_num(x[:, 0]) + np.nan_to_num(x[:, 1]) > 0).astype(int)
    model = BankaiRandomForestClassifier(n_estimators=25, random_state=223, oob_score=True).fit(x, y)

    assert np.isfinite(model.oob_decision_function_).all()
    assert np.isfinite(model.oob_score_)


@pytest.mark.parametrize("max_bins", [None, 8])
def test_nan_prediction_falls_back_to_larger_child_when_training_was_finite(max_bins):
    x = np.arange(8, dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = BankaiRandomForestClassifier(
        n_estimators=3, max_features=1, bootstrap=False, random_state=42, max_bins=max_bins
    ).fit(x, y)

    # The equal-sized root children follow sklearn's default direction: right.
    np.testing.assert_array_equal(model.predict([[np.nan]]), [1])


def test_accepts_float32_features(separable_data):
    x, y = separable_data

    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(
        x.astype(np.float32), y
    )

    np.testing.assert_array_equal(classifier.predict(x.astype(np.float32)), y)


def test_float_transfer_telemetry_records_preserved_dtype(separable_data):
    x, y = separable_data
    float32_model = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(
        x.astype(np.float32), y
    )
    float64_model = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    assert float32_model.copy_telemetry_["input_dtype"] == "float32"
    assert float32_model.copy_telemetry_["core_dtype"] == "float32"
    assert float32_model.copy_telemetry_["cast_to_float64"] is True
    assert float32_model._fit_X.dtype == np.float32
    assert float64_model.copy_telemetry_["cast_to_float64"] is False


def test_float32_fit_matches_float64_values_with_histograms_and_oob():
    rng = np.random.RandomState(104)
    x = rng.normal(size=(240, 7)).astype(np.float32)
    x[::17, 2] = np.nan
    y = (np.nan_to_num(x[:, 0]) - x[:, 1] > 0.0).astype(np.uint8)
    params = dict(
        n_estimators=11,
        max_features=None,
        max_bins=8,
        oob_score=True,
        random_state=53,
    )

    float32_model = BankaiRandomForestClassifier(**params).fit(x, y)
    float64_model = BankaiRandomForestClassifier(**params).fit(x.astype(np.float64), y)

    np.testing.assert_array_equal(float32_model.predict(x), float64_model.predict(x))
    np.testing.assert_allclose(
        float32_model.predict_proba(x), float64_model.predict_proba(x)
    )
    np.testing.assert_allclose(
        float32_model.oob_decision_function_, float64_model.oob_decision_function_
    )
    np.testing.assert_allclose(
        float32_model.feature_importances_, float64_model.feature_importances_
    )


def test_strided_float32_histogram_input_matches_contiguous_input():
    rng = np.random.RandomState(105)
    backing = rng.normal(size=(320, 12)).astype(np.float32)
    x = backing[:, ::2]
    y = (x[:, 0] + x[:, 1] > 0.0).astype(np.uint8)
    params = dict(
        n_estimators=7,
        max_features=None,
        max_bins=8,
        oob_score=True,
        random_state=54,
    )

    strided = BankaiRandomForestClassifier(**params).fit(x, y)
    contiguous = BankaiRandomForestClassifier(**params).fit(x.copy(), y)

    assert strided.copy_telemetry_["input_c_contiguous"] is False
    np.testing.assert_array_equal(strided.predict(x), contiguous.predict(x))
    np.testing.assert_allclose(strided.predict_proba(x), contiguous.predict_proba(x))
    np.testing.assert_allclose(
        strided.oob_decision_function_, contiguous.oob_decision_function_
    )


def test_random_state_reproduces_predictions_probabilities_and_importances():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(50, 4))
    y = (x[:, 0] > 0.0).astype(int)
    first = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)
    second = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))
    np.testing.assert_allclose(first.feature_importances_, second.feature_importances_)


def test_predict_rejects_mismatched_feature_count(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    with pytest.raises(ValueError, match="features"):
        classifier.predict(np.column_stack([x, x]))


def test_predict_rejects_changed_dataframe_feature_names(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(
        pd.DataFrame(x, columns=["temperature"]), y
    )

    with pytest.raises(ValueError, match="Feature names"):
        classifier.predict(pd.DataFrame(x, columns=["humidity"]))


def test_dataframe_noise_and_99_to_1_imbalance():
    rng = np.random.RandomState(42)
    y = np.zeros(1_000, dtype=int)
    y[-10:] = 1
    x = rng.normal(size=(1_000, 5))
    x[:, 0] = np.where(y == 1, 2.0, -2.0)
    frame = pd.DataFrame(x, columns=[f"feature_{index}" for index in range(5)])

    classifier = BankaiRandomForestClassifier(
        n_estimators=50, class_weight="balanced", random_state=42
    ).fit(frame, y)

    np.testing.assert_array_equal(classifier.predict(frame), y)


def test_pickle_round_trip_preserves_predictions(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    restored = pickle.loads(pickle.dumps(classifier))

    np.testing.assert_array_equal(restored.predict(x), classifier.predict(x))
