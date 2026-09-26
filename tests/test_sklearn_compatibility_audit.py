import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from bankai_random_forest import BankaiRandomForestClassifier


def audit_data(seed=53):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(96, 5))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y


def test_bankai_keeps_sklearn_parameters_and_adds_backend_controls():
    sklearn_parameters = set(RandomForestClassifier().get_params())
    bankai_parameters = set(BankaiRandomForestClassifier().get_params())

    assert sklearn_parameters <= bankai_parameters
    assert bankai_parameters - sklearn_parameters == {"importance_type", "max_bins"}


def test_inherited_sklearn_forest_methods_work_with_bankai_trees():
    x, y = audit_data()
    model = BankaiRandomForestClassifier(n_estimators=7, random_state=59).fit(x, y)

    leaves = model.apply(x[:11])
    path, node_ptr = model.decision_path(x[:11])

    assert leaves.shape == (11, model.n_estimators)
    assert path.shape[0] == 11
    assert node_ptr.shape == (model.n_estimators + 1,)
    assert np.all(np.diff(node_ptr) >= 1)
    assert model.score(x[:11], y[:11]) >= 0.0


def test_balanced_subsample_class_weight_is_accepted():
    x, y = audit_data()
    model = BankaiRandomForestClassifier(
        n_estimators=7,
        class_weight="balanced_subsample",
        random_state=61,
    ).fit(x, y)

    assert model.predict_proba(x[:11]).shape == (11, 2)


def test_sample_weight_metadata_routing_works_when_requested():
    x, y = audit_data()
    sample_weight = np.ones(len(y))

    with config_context(enable_metadata_routing=True):
        model = BankaiRandomForestClassifier(
            n_estimators=7, random_state=63
        ).set_fit_request(sample_weight=True)
        pipeline = make_pipeline(model)
        pipeline.fit(x, y, sample_weight=sample_weight)

    assert pipeline.predict_proba(x[:11]).shape == (11, 2)


def test_sklearn_recognizes_bankai_as_a_classifier():
    assert is_classifier(BankaiRandomForestClassifier())


def test_fractional_max_samples_uses_sklearn_floor_rule():
    x, y = audit_data(seed=67)
    x, y = x[:7], y[:7]
    model = BankaiRandomForestClassifier(
        n_estimators=3, max_samples=0.5, random_state=71
    ).fit(x, y)

    assert model._fit_max_samples == int(0.5 * len(x))


def test_fractional_max_samples_uses_weighted_effective_sample_count():
    x, y = audit_data(seed=71)
    weights = np.linspace(0.5, 2.0, len(y))
    model = BankaiRandomForestClassifier(
        n_estimators=3, max_samples=0.5, random_state=13
    ).fit(x, y, sample_weight=weights)
    assert model._fit_max_samples == int(0.5 * weights.sum())


def test_balanced_subsample_without_bootstrap_uses_balanced_weights():
    x, y = audit_data(seed=72)
    weights = np.linspace(0.5, 1.5, len(y))
    model = BankaiRandomForestClassifier(
        n_estimators=3,
        bootstrap=False,
        class_weight="balanced_subsample",
        random_state=14,
    ).fit(x, y, sample_weight=weights)
    classes, inverse = np.unique(y, return_inverse=True)
    expected = weights.sum() / (len(classes) * np.bincount(inverse, weights=weights))
    np.testing.assert_allclose(
        np.bincount(model._fit_y, weights=model._fit_sample_weight)
        / np.bincount(inverse, weights=weights),
        expected,
    )


def test_oob_score_accepts_a_scoring_callable():
    x, y = audit_data(seed=73)
    score = lambda truth, predicted: float(np.mean(truth == predicted))
    model = BankaiRandomForestClassifier(
        n_estimators=12, oob_score=score, random_state=79
    ).fit(x, y)

    assert 0.0 <= model.oob_score_ <= 1.0
