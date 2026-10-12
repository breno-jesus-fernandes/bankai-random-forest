import pickle

import numpy as np
import pytest
from scipy import sparse

from bankai_random_forest import BankaiRandomForestClassifier


@pytest.mark.parametrize('criterion', ['gini', 'entropy'])
@pytest.mark.parametrize('bootstrap', [False, True])
@pytest.mark.parametrize('sparse_input', [False, True])
def test_extra_trees_predictions_round_trip_across_jobs(criterion, bootstrap, sparse_input):
    rng = np.random.default_rng(24)
    X = rng.normal(size=(300, 7)).astype(np.float32)
    X[:, 5] = 0
    X[::13, 3] = np.nan
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    weights = np.where(y, 3., 1.)
    if sparse_input:
        X = sparse.csr_matrix(X)
    models = [BankaiRandomForestClassifier(
        n_estimators=9, extra_trees=True, max_bins=None, random_state=19,
        criterion=criterion, bootstrap=bootstrap, min_samples_leaf=12,
        n_jobs=jobs, oob_score=bootstrap,
    ).fit(X, y, sample_weight=weights) for jobs in [1, 2]]
    np.testing.assert_array_equal(models[0].predict_proba(X), models[1].predict_proba(X))
    np.testing.assert_allclose(models[0].feature_importances_, models[1].feature_importances_)
    restored = pickle.loads(pickle.dumps(models[1]))
    np.testing.assert_array_equal(restored.predict_proba(X), models[1].predict_proba(X))
    if bootstrap:
        np.testing.assert_allclose(models[0].oob_decision_function_, models[1].oob_decision_function_, equal_nan=True)


def test_extra_trees_bootstrap_counts_distinct_rows_for_leaf_limit():
    X = np.arange(4., dtype=float).reshape(-1, 1)
    model = BankaiRandomForestClassifier(n_estimators=5, extra_trees=True,
        random_state=1, max_samples=1., min_samples_leaf=3).fit(X, [0, 0, 1, 1], sample_weight=np.full(4, 10.))
    # Even 40 draws cannot make two leaves containing three distinct rows each.
    assert all(len(tree[0]) == 1 for tree in model._forest.shap_tree_arrays())


@pytest.mark.parametrize('class_weight', [None, 'balanced', 'balanced_subsample', {0: 1., 1: 5.}])
def test_extra_trees_weighted_bootstrap_ignores_zero_probability_rows(class_weight):
    X = np.arange(20., dtype=float).reshape(-1, 1)
    y = np.array([0] * 10 + [1] * 10)
    weights = np.array([1.] * 10 + [0.] * 10)
    model = BankaiRandomForestClassifier(n_estimators=7, extra_trees=True,
        random_state=1, class_weight=class_weight).fit(X, y, sample_weight=weights)
    assert np.all(model.predict(X) == 0)


@pytest.mark.parametrize('max_bins', [1, 2, 63, 255])
def test_extra_trees_rejects_binning(max_bins):
    with pytest.raises(ValueError, match='max_bins'):
        BankaiRandomForestClassifier(extra_trees=True, max_bins=max_bins).fit([[0], [1]], [0, 1])


def test_extra_trees_skips_split_search_when_two_leaf_limits_cannot_fit(monkeypatch):
    monkeypatch.setenv('BANKAI_PROFILE_FIT', '1')
    X = np.arange(100., dtype=float).reshape(-1, 1)
    model = BankaiRandomForestClassifier(n_estimators=1, extra_trees=True,
        bootstrap=False, min_samples_leaf=51, random_state=1).fit(X, np.arange(100) % 2)
    assert model.fit_profile_['feature_visits'] == 0
    assert model.fit_profile_['split_search_seconds'] == 0


def test_extra_trees_without_bootstrap_excludes_zero_weight_rows_from_leaf_counts():
    X = np.arange(100., dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 50).astype(int)
    weights = np.zeros(100)
    weights[[0, 99]] = 1.
    model = BankaiRandomForestClassifier(n_estimators=1, extra_trees=True,
        bootstrap=False, min_samples_leaf=2, random_state=1).fit(X, y, sample_weight=weights)
    tree = model._forest.shap_tree_arrays()[0]
    assert len(tree[0]) == 1
    assert tree[4] == [2.]


def test_extra_trees_float32_and_float64_storage_preserve_values_and_ranking():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 9)).astype(np.float32)
    y = (X[:, 0] > .5).astype(int)
    models = [BankaiRandomForestClassifier(n_estimators=11, extra_trees=True,
        bootstrap=False, random_state=17, n_jobs=2).fit(X.astype(dtype), y)
        for dtype in (np.float32, np.float64)]
    np.testing.assert_array_equal(models[0].predict_proba(X), models[1].predict_proba(X))
    np.testing.assert_allclose(models[0].feature_importances_, models[1].feature_importances_)
