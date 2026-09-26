# Bankai Random Forest

Bankai is a scikit-learn-compatible random forest classifier with a native Rust
backend. The project is pre-alpha; APIs and serialized models may change. The
roadmap and compatibility details are in [`docs/ROADMAP.md`](docs/ROADMAP.md)
and [`docs/SKLEARN_COMPATIBILITY.md`](docs/SKLEARN_COMPATIBILITY.md).

## Install from source

Python 3.11 or newer and a Rust toolchain are required to build the extension.
From the repository root:

```bash
uv sync
uv run maturin develop --release
```

The `--release` flag builds the optimized native extension. Then install the
package in your Python environment using the workflow above and import it as
shown below.

## Quick start

`BankaiRandomForestClassifier` follows the familiar scikit-learn estimator API:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bankai_random_forest import BankaiRandomForestClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = BankaiRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_bins=64,  # omit this or use None for exact split search
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print("Feature importances:", model.feature_importances_)
```

The classifier works with scikit-learn utilities such as `Pipeline`,
`cross_val_score`, and `GridSearchCV`. Features should be numeric; encode
categorical columns with a transformer such as `OneHotEncoder` in a pipeline.

## Common options

- `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `max_features`, `bootstrap`, `max_samples`, `class_weight`, and
  `random_state` control forest fitting.
- `max_bins=None` uses exact split search. An integer from 2 to 255 enables
  histogram split search; for example, `max_bins=64`.
- `ccp_alpha` prunes trees. `monotonic_cst` accepts one `-1`, `0`, or `1` per
  feature and is supported for binary classification.
- `oob_score=True` enables out-of-bag estimates when `bootstrap=True`.
- `importance_type` selects `"gain"` (default), `"split"`, or
  `"permutation"`. Permutation importance requires `bootstrap=True`.
- Dense arrays and SciPy CSR/CSC matrices are supported, including NaN feature
  values. Missing values are routed by the fitted trees; categorical values
  are not automatically encoded.

Multioutput and multilabel targets are supported. Bankai trains one native
forest per output, whereas scikit-learn shares tree structures across outputs;
therefore trees and probabilities need not match exactly. For multioutput
models, `estimators_`, `apply`, `decision_path`, and TreeSHAP inspect the first
output forest. Monotonic constraints currently apply only to binary single-
output classification.

## TreeSHAP

Install SHAP in the same environment (`python -m pip install shap`), then pass
the fitted classifier directly to `TreeExplainer`:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

This integration supports single-output classifiers. See
[`docs/SKLEARN_COMPATIBILITY.md`](docs/SKLEARN_COMPATIBILITY.md) for the
multioutput inspection limitation and compatibility status.

## Save and restore a fitted model

```python
import joblib

joblib.dump(model, "classifier.joblib")
restored_model = joblib.load("classifier.joblib")
predictions = restored_model.predict(X_test)
```

Bankai's joblib state includes the training arrays and rebuilds the native
forest while loading, so loading has reconstruction cost and artifacts may be
large. Keep Bankai, Python, and dependency versions consistent when restoring
models. Only load joblib files from trusted sources. Details are in
[`docs/JOBLIB_COMPATIBILITY.md`](docs/JOBLIB_COMPATIBILITY.md).

## Development and tests

```bash
uv run pytest -q
uv run cargo test --workspace
```

The project is licensed under GPL-3.0-or-later; see [`COPYING`](COPYING).
