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

BankaiRF is a high-performance Random Forest engine powered by a native Rust core. By leveraging LightGBM-style histograms for continuous feature pre-processing, it delivers fast and efficient training while maintaining full compatibility with the familiar scikit-learn estimator API. The Rust core is based on a maintained fork of [XRF](https://gitlab.com/mbq/xrf/), the engine
behind [FRU](https://www.sciencedirect.com/science/article/pii/S2352711026004097).

⚠️ BankaiRF is currently in Alpha. APIs and serialized models are subject to change.


## Install

Python 3.11 or newer is required. Install the package from PyPI:

```bash
python -m pip install bankai-random-forest
```

## Quick start

This example uses the California Housing dataset (about 20,000 records and 8 named features). It turns the house-value target into a binary label: at or above the dataset median.

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from bankai_random_forest import BankaiRandomForestClassifier

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = (housing.target >= housing.target.median()).astype("int8")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2077, stratify=y
)

model = BankaiRandomForestClassifier(
    n_estimators=40,
    criterion="gini",
    max_depth=20,
    max_leaf_nodes=511,
    min_samples_leaf=5,
    max_features=None,
    bootstrap=True,
    max_samples=0.8,
    n_jobs=-1,
    random_state=2077,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
positive_class = list(model.classes_).index(1)
probabilities = model.predict_proba(X_test)[:, positive_class]
print(f"F1: {f1_score(y_test, predictions):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

importances = pd.Series(model.feature_importances_, index=X.columns, name="importance")
print(importances.sort_values(ascending=False).head(10))
```

`fetch_california_housing` downloads the dataset on first use. The pandas column names are retained so the feature importances are labeled with the original feature names.

[![Open this project's 10k × 500-feature benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/breno-jesus-fernandes/bankai-random-forest/blob/master/benchmarks/benchmark_fit_10k_500f_90_relevant.ipynb) · [View the notebook in this repository](benchmarks/benchmark_fit_10k_500f_90_relevant.ipynb)

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
