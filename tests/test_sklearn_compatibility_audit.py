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


@pytest.mark.xfail(
    reason="sklearn accepts balanced_subsample; Bankai's helper rejects it.",
    strict=True,
)
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


@pytest.mark.xfail(
    condition=not is_classifier(BankaiRandomForestClassifier()),
    reason="Bankai's sklearn tags currently omit estimator_type='classifier'.",
    strict=False,
)
def test_sklearn_recognizes_bankai_as_a_classifier():
    assert is_classifier(BankaiRandomForestClassifier())


@pytest.mark.xfail(
    reason="sklearn floors fractional max_samples; Bankai currently rounds it.",
    strict=True,
)
def test_fractional_max_samples_uses_sklearn_floor_rule():
    x, y = audit_data(seed=67)
    x, y = x[:7], y[:7]
    model = BankaiRandomForestClassifier(
        n_estimators=3, max_samples=0.5, random_state=71
    ).fit(x, y)

    assert model._fit_max_samples == int(0.5 * len(x))


@pytest.mark.xfail(
    reason="sklearn accepts a callable oob_score; Bankai currently accepts booleans only.",
    strict=True,
)
def test_oob_score_accepts_a_scoring_callable():
    x, y = audit_data(seed=73)
    score = lambda truth, predicted: float(np.mean(truth == predicted))
    model = BankaiRandomForestClassifier(
        n_estimators=12, oob_score=score, random_state=79
    ).fit(x, y)

    assert 0.0 <= model.oob_score_ <= 1.0
