import numpy as np
import pytest

import shap

from bankai_random_forest import BankaiRandomForestClassifier


@pytest.mark.parametrize("max_bins", [None, 16])
@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("ccp_alpha", [0.0, 0.05])
def test_all_tree_shap_paths_are_additive_with_constant_features(max_bins, classes, ccp_alpha):
    rng = np.random.RandomState(19)
    x = rng.normal(size=(120, 5))
    x[:, 4] = 3.0
    if classes == 2:
        y = (x[:, 0] - x[:, 1] > 0).astype(int)
    else:
        y = np.digitize(x[:, 0] + x[:, 1], [-0.5, 0.5])
    model = BankaiRandomForestClassifier(
        n_estimators=12, random_state=13, max_bins=max_bins, ccp_alpha=ccp_alpha
    ).fit(x[:90], y[:90])
    explained = x[90:96]
    probabilities = model.predict_proba(explained)

    direct = shap.TreeExplainer(model)
    direct_values = np.asarray(direct.shap_values(explained))
    adapter = shap.TreeExplainer(model._tree_shap_adapter_for_benchmark())
    adapter_values = np.asarray(adapter.shap_values(explained))
    native_values, native_base = model._native_tree_shap_for_benchmark(explained)

    expected_shape = (len(explained), x.shape[1], classes)
    assert direct_values.shape == expected_shape
    assert adapter_values.shape == expected_shape
    assert native_values.shape == expected_shape
    np.testing.assert_allclose(
        direct_values.sum(axis=1) + direct.expected_value, probabilities, atol=1e-6
    )
    np.testing.assert_allclose(
        adapter_values.sum(axis=1) + adapter.expected_value, probabilities, atol=1e-6
    )
    np.testing.assert_allclose(
        native_values.sum(axis=1) + native_base, probabilities, atol=1e-6
    )


@pytest.mark.parametrize("max_bins", [None, 16])
def test_constrained_tree_shap_paths_remain_additive(max_bins):
    rng = np.random.RandomState(157)
    x = rng.normal(size=(120, 4))
    y = (x[:, 0] - x[:, 1] > 0).astype(int)
    model = BankaiRandomForestClassifier(
        n_estimators=9,
        random_state=163,
        max_bins=max_bins,
        monotonic_cst=[1, -1, 0, 0],
    ).fit(x[:90], y[:90])
    explained = x[90:96]
    probabilities = model.predict_proba(explained)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(explained))

    assert values.shape == (len(explained), x.shape[1], 2)
    np.testing.assert_allclose(
        values.sum(axis=1) + explainer.expected_value, probabilities, atol=1e-6
    )
