import numpy as np
import pandas as pd
import pickle
import pytest
from scipy import sparse
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


@pytest.mark.parametrize(
    ("parameters", "name"),
    [
        ({"max_leaf_nodes": 2}, "max_leaf_nodes"),
        ({"min_impurity_decrease": 0.1}, "min_impurity_decrease"),
        ({"n_jobs": 1}, "n_jobs"),
        ({"verbose": 1}, "verbose"),
        ({"warm_start": True}, "warm_start"),
        ({"ccp_alpha": 0.1}, "ccp_alpha"),
        ({"monotonic_cst": [1]}, "monotonic_cst"),
    ],
)
def test_rejects_parameter_semantics_not_implemented_by_the_baseline(
    separable_data, parameters, name
):
    x, y = separable_data

    with pytest.raises(NotImplementedError, match=name):
        BankaiRandomForestClassifier(**parameters).fit(x, y)


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


def test_rejects_sparse_input_without_implicit_densification(separable_data):
    x, y = separable_data

    with pytest.raises(TypeError, match="sparse"):
        BankaiRandomForestClassifier().fit(sparse.csr_matrix(x), y)


def test_pickle_round_trip_preserves_predictions(separable_data):
    x, y = separable_data
    classifier = BankaiRandomForestClassifier(n_estimators=25, random_state=42).fit(x, y)

    restored = pickle.loads(pickle.dumps(classifier))

    np.testing.assert_array_equal(restored.predict(x), classifier.predict(x))
