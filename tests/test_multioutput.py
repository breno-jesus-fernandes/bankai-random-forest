import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import get_tags

from bankai_random_forest import BankaiRandomForestClassifier


def _multioutput_data():
    rng = np.random.RandomState(91)
    x = rng.normal(size=(180, 4))
    y = np.column_stack(
        (
            np.where(x[:, 0] > 0.0, "positive", "negative"),
            np.where(x[:, 1] > 0.7, "high", np.where(x[:, 1] < -0.7, "low", "mid")),
        )
    )
    return x, y


def test_multiclass_multioutput_prediction_probability_and_sklearn_parity():
    x, y = _multioutput_data()
    params = dict(n_estimators=19, max_features=None, bootstrap=False, random_state=27)
    model = BankaiRandomForestClassifier(**params).fit(x, y)
    reference = RandomForestClassifier(**params).fit(x, y)

    prediction = model.predict(x)
    probabilities = model.predict_proba(x)

    assert prediction.shape == y.shape
    assert isinstance(model.classes_, list)
    assert isinstance(probabilities, list) and len(probabilities) == y.shape[1]
    for output, class_values in enumerate(model.classes_):
        np.testing.assert_array_equal(class_values, reference.classes_[output])
        assert probabilities[output].shape == (len(x), len(class_values))
        np.testing.assert_allclose(probabilities[output].sum(axis=1), 1.0)
        assert np.mean(prediction[:, output] == y[:, output]) > 0.95
    assert np.mean(prediction == reference.predict(x)) > 0.95
    with pytest.raises(ValueError, match="multiclass-multioutput is not supported"):
        model.score(x, y)
    with np.errstate(divide="ignore"):
        log_probabilities = model.predict_log_proba(x)
    assert isinstance(log_probabilities, list)
    assert np.isclose(model.feature_importances_.sum(), 1.0)


def test_multilabel_indicator_supports_weights_oob_and_sklearn_tags():
    rng = np.random.RandomState(93)
    x = rng.normal(size=(160, 4))
    y = np.column_stack((x[:, 0] > 0.0, x[:, 1] + x[:, 2] > 0.5)).astype(np.int64)
    weights = np.linspace(0.5, 2.0, len(x))
    model = BankaiRandomForestClassifier(
        n_estimators=31,
        max_features=None,
        random_state=29,
        oob_score=True,
        class_weight="balanced",
    ).fit(x, y, sample_weight=weights)
    reference = RandomForestClassifier(
        n_estimators=31,
        max_features=None,
        random_state=29,
        oob_score=True,
        class_weight="balanced",
    ).fit(x, y, sample_weight=weights)

    assert model.predict(x).shape == y.shape
    assert isinstance(model.predict_proba(x), list)
    assert isinstance(model.oob_decision_function_, list)
    for output, probabilities in enumerate(model.oob_decision_function_):
        assert probabilities.shape == (len(y), len(model.classes_[output]))
    assert 0.0 <= model.oob_score_ <= 1.0
    assert np.mean(model.predict(x) == reference.predict(x)) > 0.95
    assert model.score(x, y) == reference.score(x, y)
    assert abs(model.oob_score_ - reference.oob_score_) < 0.1
    assert get_tags(model).target_tags.single_output is False


def test_multioutput_joblib_round_trip_preserves_predictions_and_probabilities(tmp_path):
    x, y = _multioutput_data()
    model = BankaiRandomForestClassifier(
        n_estimators=11, max_bins=8, random_state=31
    ).fit(x[:140], y[:140])
    path = tmp_path / "multioutput.joblib"
    joblib.dump(model, path)

    restored = joblib.load(path)

    np.testing.assert_array_equal(restored.predict(x[140:]), model.predict(x[140:]))
    for actual, expected in zip(restored.predict_proba(x[140:]), model.predict_proba(x[140:])):
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)


def test_multioutput_rejects_continuous_targets():
    x = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.column_stack((np.linspace(0.0, 1.0, 20), np.linspace(1.0, 2.0, 20)))

    with pytest.raises(ValueError, match="Unknown label type"):
        BankaiRandomForestClassifier(n_estimators=3).fit(x, y)


def test_multioutput_accepts_per_output_class_weight_dictionaries():
    rng = np.random.RandomState(101)
    x = rng.normal(size=(100, 3))
    y = np.column_stack((x[:, 0] > 0.0, x[:, 1] > 0.0)).astype(np.int64)
    weights = [{0: 1.0, 1: 2.0}, {0: 3.0, 1: 1.0}]

    model = BankaiRandomForestClassifier(
        n_estimators=9, class_weight=weights, random_state=41
    ).fit(x, y)

    assert model.predict(x).shape == y.shape
    assert [child.class_weight for child in model._multioutput_models] == weights


@pytest.mark.parametrize("matrix_type", [sparse.csr_matrix, sparse.csc_matrix])
def test_multioutput_sparse_and_nan_inputs_match_dense_predictions(matrix_type):
    rng = np.random.RandomState(97)
    x = rng.normal(size=(120, 4))
    y = np.column_stack((x[:, 0] > 0.0, x[:, 1] > 0.0)).astype(np.int64)
    x[::13, 2] = np.nan
    params = dict(n_estimators=13, random_state=37, max_bins=8)
    dense = BankaiRandomForestClassifier(**params).fit(x, y)
    sparse_model = BankaiRandomForestClassifier(**params).fit(matrix_type(x), y)

    np.testing.assert_array_equal(sparse_model.predict(matrix_type(x)), dense.predict(x))
    for actual, expected in zip(
        sparse_model.predict_proba(matrix_type(x)), dense.predict_proba(x)
    ):
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
