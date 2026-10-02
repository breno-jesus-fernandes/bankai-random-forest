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
    assert bankai_parameters - sklearn_parameters == {
        "importance_type",
        "max_bins",
        "binning_strategy",
        "bin_sample_size",
        "shap_mode",
        "shap_sample_size",
    }


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
    expected = len(y) / (len(classes) * np.bincount(inverse))
    np.testing.assert_allclose(
        np.bincount(model._fit_y, weights=model._fit_sample_weight)
        / np.bincount(inverse, weights=weights),
        expected,
    )


def test_balanced_subsample_bootstrap_is_tree_specific_and_parallel_deterministic():
    rng = np.random.RandomState(84)
    x = rng.normal(size=(360, 4))
    y = np.zeros(360, dtype=np.int64)
    y[-40:] = 1
    x[-40:, 0] += 1.0
    params = dict(
        n_estimators=31,
        max_features=None,
        max_depth=5,
        class_weight="balanced_subsample",
        random_state=91,
        oob_score=True,
        importance_type="permutation",
    )
    sequential = BankaiRandomForestClassifier(n_jobs=1, **params).fit(x, y)
    parallel = BankaiRandomForestClassifier(n_jobs=3, **params).fit(x, y)
    unweighted = BankaiRandomForestClassifier(
        n_jobs=1, **{**params, "class_weight": None, "oob_score": False}
    ).fit(x, y)

    np.testing.assert_array_equal(sequential.predict(x), parallel.predict(x))
    np.testing.assert_allclose(sequential.predict_proba(x), parallel.predict_proba(x))
    assert np.isfinite(sequential.oob_score_)
    assert np.isfinite(sequential.feature_importances_).all()
    assert not np.array_equal(sequential.predict_proba(x), unweighted.predict_proba(x))


def test_oob_score_accepts_a_scoring_callable():
    x, y = audit_data(seed=73)
    score = lambda truth, predicted: float(np.mean(truth == predicted))
    model = BankaiRandomForestClassifier(
        n_estimators=12, oob_score=score, random_state=79
    ).fit(x, y)

    assert 0.0 <= model.oob_score_ <= 1.0


@pytest.mark.parametrize("max_bins", [None, 8])
def test_ccp_alpha_prunes_trees_to_root_for_large_alpha(max_bins):
    x, y = audit_data(seed=97)
    model = BankaiRandomForestClassifier(
        n_estimators=7, max_features=None, ccp_alpha=1.0, max_bins=max_bins,
        random_state=101, oob_score=True, importance_type="permutation",
    ).fit(x, y)

    assert all(tree.tree_.node_count == 1 for tree in model.estimators_)
    np.testing.assert_array_equal(model.predict(x), np.full(len(y), model.predict(x[:1])[0]))
    assert np.isfinite(model.oob_score_)
    assert np.isfinite(model.feature_importances_).all()
    assert model.apply(x).shape == (len(y), model.n_estimators)
    path, node_ptr = model.decision_path(x)
    assert path.shape == (len(y), sum(tree.tree_.node_count for tree in model.estimators_))
    assert node_ptr.shape == (model.n_estimators + 1,)


@pytest.mark.parametrize("alpha", [-0.1, np.nan, np.inf])
def test_ccp_alpha_rejects_values_outside_sklearn_range(alpha):
    x, y = audit_data(seed=103)
    with pytest.raises(ValueError, match="ccp_alpha"):
        BankaiRandomForestClassifier(ccp_alpha=alpha, n_estimators=2).fit(x, y)


def test_ccp_alpha_matches_sklearn_single_feature_pruning_path():
    rng = np.random.RandomState(12)
    x = rng.normal(size=(80, 1))
    y = (x[:, 0] > -0.3).astype(int)
    y[rng.choice(len(y), 10, replace=False)] ^= 1

    for alpha in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        parameters = dict(
            n_estimators=1,
            bootstrap=False,
            max_features=None,
            random_state=5,
            ccp_alpha=alpha,
        )
        bankai = BankaiRandomForestClassifier(**parameters).fit(x, y)
        sklearn_model = RandomForestClassifier(**parameters).fit(x, y)

        assert bankai.estimators_[0].tree_.node_count == sklearn_model.estimators_[0].tree_.node_count
        np.testing.assert_array_equal(bankai.predict(x), sklearn_model.predict(x))


@pytest.mark.parametrize("max_bins", [None, 8])
def test_ccp_alpha_zero_preserves_unpruned_predictions_and_importances(max_bins):
    x, y = audit_data(seed=107)
    parameters = dict(n_estimators=9, max_bins=max_bins, random_state=109)
    default = BankaiRandomForestClassifier(**parameters).fit(x, y)
    explicit_zero = BankaiRandomForestClassifier(ccp_alpha=0.0, **parameters).fit(x, y)

    np.testing.assert_array_equal(explicit_zero.predict(x), default.predict(x))
    np.testing.assert_array_equal(explicit_zero.predict_proba(x), default.predict_proba(x))
    np.testing.assert_allclose(explicit_zero.feature_importances_, default.feature_importances_, rtol=1e-15, atol=0.0)
    assert [tree.tree_.node_count for tree in explicit_zero.estimators_] == [
        tree.tree_.node_count for tree in default.estimators_
    ]


@pytest.mark.parametrize("max_bins", [None, 8])
@pytest.mark.parametrize("ccp_alpha", [0.0, 0.05])
@pytest.mark.parametrize("constraint,direction", [(1, 1), (-1, -1)])
def test_monotonic_constraints_bound_positive_class_probability(max_bins, ccp_alpha, constraint, direction):
    rng = np.random.RandomState(119)
    x0 = rng.uniform(-2, 2, 320)
    x1 = rng.normal(size=320)
    probability = 1 / (1 + np.exp(-direction * (x0 + 0.8 * x1)))
    y = (rng.uniform(size=320) < probability).astype(int)
    x = np.column_stack([x0, x1])
    model = BankaiRandomForestClassifier(
        n_estimators=21,
        max_features=None,
        max_depth=6,
        monotonic_cst=[constraint, constraint],
        max_bins=max_bins,
        ccp_alpha=ccp_alpha,
        random_state=123,
    ).fit(x, y)
    reference = RandomForestClassifier(
        n_estimators=21,
        max_features=None,
        max_depth=6,
        ccp_alpha=ccp_alpha,
        monotonic_cst=[constraint, constraint],
        random_state=123,
    ).fit(x, y)

    values = np.linspace(-2.5, 2.5, 101)
    for nuisance_value in np.linspace(-2, 2, 7):
        grid = np.column_stack([values, np.full_like(values, nuisance_value)])
        positive_probability = model.predict_proba(grid)[:, 1]
        reference_probability = reference.predict_proba(grid)[:, 1]
        assert np.all(direction * np.diff(positive_probability) >= -1e-15)
        assert np.all(direction * np.diff(reference_probability) >= -1e-15)
        grid = np.column_stack([np.full_like(values, nuisance_value), values])
        positive_probability = model.predict_proba(grid)[:, 1]
        reference_probability = reference.predict_proba(grid)[:, 1]
        assert np.all(direction * np.diff(positive_probability) >= -1e-15)
        assert np.all(direction * np.diff(reference_probability) >= -1e-15)


@pytest.mark.parametrize("max_bins", [None, 8])
def test_zero_monotonic_constraints_preserve_default_predictions(max_bins):
    x, y = audit_data(seed=137)
    parameters = dict(n_estimators=11, max_bins=max_bins, random_state=139)
    default = BankaiRandomForestClassifier(**parameters).fit(x, y)
    unconstrained = BankaiRandomForestClassifier(
        monotonic_cst=np.zeros(x.shape[1], dtype=int), **parameters
    ).fit(x, y)
    np.testing.assert_array_equal(unconstrained.predict(x), default.predict(x))
    np.testing.assert_array_equal(unconstrained.predict_proba(x), default.predict_proba(x))
    assert unconstrained.apply(x).shape == (len(x), unconstrained.n_estimators)
    path, node_ptr = unconstrained.decision_path(x)
    assert path.shape[0] == len(x)
    assert node_ptr.shape == (unconstrained.n_estimators + 1,)


@pytest.mark.parametrize("constraints", [[2, 0], [1], [1, 0, 0], [1.5, 0.0]])
def test_monotonic_constraints_validate_values_and_feature_count(constraints):
    x, y = audit_data()
    with pytest.raises(ValueError, match="monotonic_cst"):
        BankaiRandomForestClassifier(
            n_estimators=3, monotonic_cst=constraints, random_state=127
        ).fit(x, y)


def test_monotonic_constraints_reject_multiclass_targets():
    x, y = audit_data()
    y = np.arange(len(y)) % 3
    with pytest.raises(ValueError, match="binary"):
        BankaiRandomForestClassifier(
            n_estimators=3, monotonic_cst=[1, 0, 0, 0, 0], random_state=131
        ).fit(x, y)
