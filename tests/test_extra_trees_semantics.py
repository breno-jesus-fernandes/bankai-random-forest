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


@pytest.mark.parametrize('criterion', ['gini', 'entropy'])
@pytest.mark.parametrize('bootstrap', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_binned_extra_trees_round_trip_jobs_oob_and_exported_routing(criterion, bootstrap, dtype):
    from sklearn.base import clone
    rng = np.random.default_rng(71)
    X = rng.normal(size=(320, 9)).astype(dtype)
    X[:, 7] = 1
    X[::17, 3] = np.nan
    y = (X[:, 0] + X[:, 1] > .5).astype(int)
    weights = np.linspace(.5, 2, len(y))
    weights[::23] = 0
    params = dict(n_estimators=9, extra_trees=True, extra_trees_backend='binned',
                  max_bins=63, binning_strategy='sampled_select', bin_sample_size=80,
                  random_state=19, criterion=criterion, bootstrap=bootstrap,
                  min_samples_leaf=8, class_weight='balanced', oob_score=bootstrap)
    models = [BankaiRandomForestClassifier(**params, n_jobs=jobs).fit(X, y, sample_weight=weights)
              for jobs in (1, 2)]
    np.testing.assert_array_equal(models[0].predict_proba(X), models[1].predict_proba(X))
    np.testing.assert_allclose(models[0].feature_importances_, models[1].feature_importances_)
    # Public sklearn tree exports must route original values identically to native bin predictions.
    exported = np.mean([tree.predict_proba(X) for tree in models[0].estimators_], axis=0)
    np.testing.assert_allclose(exported, models[0].predict_proba(X))
    restored = pickle.loads(pickle.dumps(models[1]))
    np.testing.assert_array_equal(restored.predict_proba(X), models[1].predict_proba(X))
    np.testing.assert_allclose(restored.feature_importances_, models[1].feature_importances_)
    assert clone(models[0]).get_params()['extra_trees_backend'] == 'binned'
    if bootstrap:
        np.testing.assert_allclose(models[0].oob_decision_function_, models[1].oob_decision_function_, equal_nan=True)


@pytest.mark.parametrize('max_bins', [2, 63, 255])
@pytest.mark.parametrize('classes', [2, 3, 4])
def test_binned_extra_trees_bin_limits_multiclass_and_native_shap(max_bins, classes):
    rng = np.random.default_rng(24)
    X = rng.normal(size=(180, 6)).astype(np.float32)
    X[::13, 2] = np.nan
    y = np.arange(len(X)) % classes
    model = BankaiRandomForestClassifier(n_estimators=5, extra_trees=True,
        extra_trees_backend='binned', max_bins=max_bins, random_state=11,
        min_samples_leaf=4, max_depth=4, n_jobs=2).fit(X, y)
    probability = model.predict_proba(X[:15])
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    values, base = model._native_tree_shap_for_benchmark(X[:15])
    np.testing.assert_allclose(values.sum(axis=1) + base, probability, atol=1e-10)


@pytest.mark.parametrize('params, message', [
    ({'extra_trees_backend': 'invalid'}, 'extra_trees_backend'),
    ({'extra_trees_backend': 'binned', 'max_bins': 63}, 'requires extra_trees=True'),
    ({'extra_trees_backend': 'binned', 'extra_trees': True}, 'integer max_bins'),
    ({'extra_trees_backend': 'binned', 'extra_trees': True, 'max_bins': 256}, 'max_bins'),
])
def test_binned_backend_requires_explicit_valid_configuration(params, message):
    with pytest.raises(ValueError, match=message):
        BankaiRandomForestClassifier(**params).fit([[0], [1]], [0, 1])


def test_binned_backend_rejects_sparse_without_silent_densification():
    with pytest.raises(ValueError, match='dense input only'):
        BankaiRandomForestClassifier(extra_trees=True, extra_trees_backend='binned',
            max_bins=63).fit(sparse.csr_matrix([[0.], [1.]]), [0, 1])


def test_binned_backend_profile_records_layout_and_skips_global_histogram_cache(monkeypatch):
    monkeypatch.setenv('BANKAI_PROFILE_FIT', '1')
    X = np.arange(200., dtype=np.float32).reshape(-1, 2)
    model = BankaiRandomForestClassifier(n_estimators=3, extra_trees=True,
        extra_trees_backend='binned', max_bins=63, bootstrap=False,
        random_state=1, n_jobs=2).fit(X, np.arange(100) % 2)
    assert model.fit_profile_['bin_layout_conversion_seconds'] > 0
    assert model.fit_profile_['feature_buffer_peak_bytes_per_worker'] == 0
    assert model._fit_max_bins == 63


def test_loading_pre_backend_pickle_defaults_to_exact():
    model = BankaiRandomForestClassifier(n_estimators=3, extra_trees=True,
        random_state=1).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    state = model.__getstate__()
    state.pop('extra_trees_backend')
    state.pop('_fit_extra_trees_backend')
    restored = BankaiRandomForestClassifier.__new__(BankaiRandomForestClassifier)
    restored.__setstate__(state)
    assert restored.get_params()['extra_trees_backend'] == 'exact'
    np.testing.assert_array_equal(restored.predict_proba([[0], [3]]), model.predict_proba([[0], [3]]))
