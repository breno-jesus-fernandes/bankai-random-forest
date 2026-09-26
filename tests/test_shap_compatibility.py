import numpy as np

import shap

from bankai_random_forest import BankaiRandomForestClassifier


def test_shap_permutation_explainer_accepts_bankai_predict_proba():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(80, 3))
    y = (x[:, 0] + x[:, 1] > 0.0).astype(int)
    classifier = BankaiRandomForestClassifier(
        n_estimators=10, random_state=42
    ).fit(x[:60], y[:60])

    explanation = shap.Explainer(
        classifier.predict_proba,
        x[:10],
        algorithm="permutation",
        seed=42,
    )(x[60:62], max_evals=7)

    assert explanation.values.shape == (2, 3, 2)
    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    np.testing.assert_allclose(
        reconstructed, classifier.predict_proba(x[60:62]), atol=1e-7
    )
