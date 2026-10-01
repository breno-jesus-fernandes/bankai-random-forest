# 🔥 Bankai Random Forest

[![PyPI Latest Release](https://img.shields.io/pypi/v/bankai-random-forest.svg)](https://pypi.org/project/bankai-random-forest/)
[![Package Status](https://img.shields.io/pypi/status/bankai-random-forest.svg)](https://pypi.org/project/bankai-random-forest/)
[![Python versions](https://img.shields.io/pypi/pyversions/bankai-random-forest.svg)](https://pypi.org/project/bankai-random-forest/)
[![uv managed](https://img.shields.io/badge/uv-managed-blue)](https://docs.astral.sh/uv/)
[![Downloads](https://static.pepy.tech/badge/bankai-random-forest)](https://pepy.tech/project/bankai-random-forest)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](COPYING)
[![CI](https://github.com/breno-jesus-fernandes/bankai-random-forest/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/breno-jesus-fernandes/bankai-random-forest/actions/workflows/ci.yml)
[![Dependency security](https://github.com/breno-jesus-fernandes/bankai-random-forest/actions/workflows/security.yml/badge.svg?branch=master)](https://github.com/breno-jesus-fernandes/bankai-random-forest/actions/workflows/security.yml)
[![Coverage](https://codecov.io/gh/breno-jesus-fernandes/bankai-random-forest/branch/master/graph/badge.svg)](https://codecov.io/gh/breno-jesus-fernandes/bankai-random-forest)
[![GitHub Release](https://img.shields.io/github/v/release/breno-jesus-fernandes/bankai-random-forest?include_prereleases)](https://github.com/breno-jesus-fernandes/bankai-random-forest/releases)
[![PyPI deployment](https://img.shields.io/github/deployments/breno-jesus-fernandes/bankai-random-forest/pypi?label=PyPI%20deployment)](https://github.com/breno-jesus-fernandes/bankai-random-forest/deployments)
[![GitHub stars](https://img.shields.io/github/stars/breno-jesus-fernandes/bankai-random-forest.svg?style=social)](https://github.com/breno-jesus-fernandes/bankai-random-forest)

BankaiRF is a high-performance Random Forest engine powered by a native Rust core. By leveraging LightGBM-style histograms for continuous feature pre-processing, it delivers fast and efficient training while maintaining full compatibility with the familiar scikit-learn estimator API.

⚠️ Project Status: BankaiRF is currently in Alpha. APIs and serialized models are subject to change.

The rust core is based on a maintained fork of [XRF](https://gitlab.com/mbq/xrf/), the engine
behind [FRU](https://www.sciencedirect.com/science/article/pii/S2352711026004097).

## Install

Python 3.11 or newer is required. Install the package from PyPI:

```bash
python -m pip install bankai-random-forest
```

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bankai_random_forest import BankaiRandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = BankaiRandomForestClassifier(
    n_estimators=200,
    max_bins=63,
    binning_strategy="sampled_select",
    importance_type="permutation",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print("Probabilities:", probabilities[:3])
print("Feature importances:", model.feature_importances_)
```

`max_bins` enables histogram-based split search in this example. Set
`max_bins=None` to use exact split search. Histogram cuts can change the fitted
trees and the accuracy/speed trade-off; evaluate both on your data when that
matters.

## Training options

Bankai follows the familiar `RandomForestClassifier` estimator pattern and
supports common options such as `n_estimators`, `criterion`, `max_depth`,
`min_samples_split`, `min_samples_leaf`, `max_features`, `max_leaf_nodes`,
`bootstrap`, `max_samples`, `class_weight`, `random_state`, `n_jobs`,
`oob_score`, and `ccp_alpha`.

| Option | Behavior |
| --- | --- |
| `n_jobs` | `None` uses one worker. Negative values follow joblib-style CPU-count rules; `n_jobs=-1` uses all visible logical CPUs. |
| `max_bins` | `None` (default) uses exact split search. An integer from 2 to 255 enables histogram training. |
| `binning_strategy` | With histograms, choose `exact_sort` (default), `sampled_sort`, `exact_select`, or `sampled_select`. |
| `bin_sample_size` | Maximum number of rows used per feature by sampled strategies; defaults to 200,000. |
| `importance_type` | `gain` (default), `split`, or `permutation`. Permutation importance uses out-of-bag accuracy decrease and requires `bootstrap=True`; it can add fit time. |
| `monotonic_cst` | One `-1`, `0`, or `1` constraint per feature; currently limited to binary, single-output classification without NaN values. |

Sampled binning strategies use a deterministic subset of rows to construct
feature cuts. Selection strategies avoid fully sorting each feature, while
repeated feature values can produce cuts that differ from the sort strategies.
For reproducible model comparisons, keep the strategy, sample size, seed, and
thread count fixed.

## API and supported inputs

- Dense NumPy arrays and SciPy CSR/CSC matrices are supported. Numeric features
  are expected; encode categorical columns in a scikit-learn transformer such
  as `OneHotEncoder`.
- NaN feature values are supported in dense and sparse inputs and are routed by
  the fitted trees. Infinite values are rejected.
- Binary, multiclass, multioutput, and multilabel classification targets are
  supported. Bankai trains a separate native forest for each output, so its
  multioutput trees and probabilities need not match scikit-learn's shared-tree
  implementation.
- `feature_importances_` defaults to gain-based importance; its values are not
  guaranteed to match scikit-learn's impurity importance. Bankai also provides
  split-count and out-of-bag permutation importance.
- `warm_start=True` does not append only new trees: a later `fit` rebuilds the
  forest. For multioutput models, `estimators_`, `apply`, `decision_path`, and
  TreeSHAP inspection expose the first output forest.

Tested with scikit-learn 1.9.1 (`>=1.9.1,<1.10`).

## Feature explanations

SHAP is optional and is not a runtime dependency. For TreeSHAP, pass a fitted
single-output classifier directly to `TreeExplainer`:

```python
import shap

explainer = shap.TreeExplainer(model)
explanation = explainer(X_test[:10])
```

See the [SHAP compatibility notes](docs/SHAP_COMPATIBILITY.md) for output
shapes, multioutput limitations, and model-agnostic alternatives.

## Performance

Fit time depends on the data, tree settings, hardware, and importance
calculation. The benchmark report includes stage breakdowns, later Bankai
measurements, and comparison caveats; results are specific to the recorded
workload and are not a performance guarantee.

Run the [binary classification fit benchmark in Google Colab](https://colab.research.google.com/github/breno-jesus-fernandes/bankai-random-forest/blob/master/benchmarks/benchmark_fit_100k_500f_90_relevant.ipynb)
or [view the notebook in this repository](benchmarks/benchmark_fit_100k_500f_90_relevant.ipynb).
It compares scikit-learn RF, LightGBM RF, and Bankai on one shared dataset with
`n_jobs=-1`; the first cell installs the packages. The scikit-learn comparison
times fit plus validation permutation importance against Bankai fit with OOB
permutation importance. The LightGBM comparison remains fit-only with gain on
both models.

## Save and restore models

Fitted models can be saved with joblib:

```python
import joblib

joblib.dump(model, "classifier.joblib")
restored = joblib.load("classifier.joblib")
predictions = restored.predict(X_test)
```

Bankai's serialized state includes training arrays and rebuilds the native
forest when loaded. Loading therefore has reconstruction cost and model files
can be large. Keep Bankai and Python versions consistent when restoring a
model, and only load joblib files from trusted sources. See the
[serialization notes](docs/JOBLIB_COMPATIBILITY.md).

## Development

Building from source requires Python 3.11 or newer, Rust, and `uv`:

```bash
uv sync --locked --no-install-project
uv run maturin develop --release --locked
uv run pytest -q
uv run cargo test --workspace --locked
```

Install optional benchmark dependencies with `uv sync --locked --group benchmark`.
The [roadmap](docs/ROADMAP.md) tracks project status and planned work.

The coverage badge reports Python package coverage from the Linux x86_64 CI job;
it does not include Rust line coverage.

## License

Bankai is licensed under [GPL-3.0-or-later](COPYING). The maintained XRF fork
is also GPL-3.0-or-later; see [`NOTICE`](NOTICE) for upstream attribution.
