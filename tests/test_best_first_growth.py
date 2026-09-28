import numpy as np
import pytest
from sklearn.datasets import make_classification
from bankai_random_forest import BankaiRandomForestClassifier


@pytest.mark.parametrize('max_bins', [None, 63])
@pytest.mark.parametrize('weighted', [False, True])
def test_leaf_budget_expands_global_gain_instead_of_first_child(max_bins, weighted):
    # Root splits at 27.5. Low values have a lower local gain (0.072 vs
    # 0.116), but a greater global gain because they contain 28/40 samples.
    y = np.array([0,0,1,0,0,1,0,0,1,1,1,0,1,1,1,0,1,1,0,0,
                  0,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,1,1,0])
    x = np.arange(len(y), dtype=float)[:, None]
    model = BankaiRandomForestClassifier(n_estimators=1, bootstrap=False,
        max_features=None, max_leaf_nodes=3, max_bins=max_bins, random_state=42)
    model.fit(x, y, sample_weight=np.full(len(y), 2.) if weighted else None)
    left, right, _, threshold, *_ = model._forest.shap_tree_arrays()[0]
    assert threshold[0] == 27.5
    assert left[left[0]] != -1  # The second child in Rust must win the budget.
    assert left[right[0]] == -1
    assert sum(child == -1 for child in left) == 3


@pytest.mark.parametrize('max_bins', [None, 31])
def test_bounded_growth_respects_depth_and_parallel_reproducibility(max_bins):
    x, y = make_classification(n_samples=300, n_features=6, random_state=17)
    parameters = dict(n_estimators=7, max_leaf_nodes=7, max_depth=3,
        min_samples_leaf=4, max_bins=max_bins, random_state=31)
    serial = BankaiRandomForestClassifier(n_jobs=1, **parameters).fit(x, y)
    parallel = BankaiRandomForestClassifier(n_jobs=3, **parameters).fit(x, y)
    np.testing.assert_array_equal(serial.predict_proba(x), parallel.predict_proba(x))
    for left, right, _, _, cover, *_ in serial._forest.shap_tree_arrays():
        assert sum(child == -1 for child in left) <= 7
        pending = [(0, 0)]
        while pending:
            node, depth = pending.pop()
            assert depth <= 3
            if left[node] != -1:
                pending.extend([(left[node], depth + 1), (right[node], depth + 1)])
            else:
                assert cover[node] >= 4


@pytest.mark.parametrize('max_bins', [None, 31])
def test_bounded_growth_preserves_monotonicity_and_pruning(max_bins):
    rng = np.random.default_rng(4)
    x = np.linspace(-3, 3, 300)[:, None]
    y = (x[:, 0] + rng.normal(size=300) > 0).astype(int)
    params = dict(n_estimators=9, max_leaf_nodes=9, max_bins=max_bins,
                  monotonic_cst=[1], random_state=42)
    model = BankaiRandomForestClassifier(**params).fit(x, y)
    assert np.all(np.diff(model.predict_proba(x)[:, 1]) >= -1e-12)
    pruned = BankaiRandomForestClassifier(ccp_alpha=1., **params).fit(x, y)
    assert all(len(tree[0]) == 1 for tree in pruned._forest.shap_tree_arrays())
