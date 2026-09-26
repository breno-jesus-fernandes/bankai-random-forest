# SHAP compatibility

Bankai works with SHAP's model-agnostic explainers through its public
`predict_proba` method. The integration test covers `shap.Explainer` with the
permutation algorithm on binary classification and checks additivity for both
class outputs.

```python
import shap

background = X_train[:50]
explainer = shap.Explainer(
    classifier.predict_proba,
    background,
    algorithm="permutation",
    seed=42,
)
explanation = explainer(X_to_explain, max_evals=2 * X_train.shape[1] + 1)
```

The returned values have shape `(samples, features, classes)`. Permutation
explainers call the model repeatedly, so runtime grows with the number of
features, samples to explain, and evaluation budget. Use a representative,
small background sample and begin with a small `X_to_explain` batch.

`shap.TreeExplainer(classifier)` is not supported yet. The current Bankai
estimator stores its forest in the native Rust core and does not expose
scikit-learn decision-tree objects or the tree arrays that SHAP's tree model
loader needs. With SHAP 0.51.0, the call raises `InvalidModelError`.

SHAP is a development dependency for the compatibility test only; it is not a
runtime dependency of Bankai.
