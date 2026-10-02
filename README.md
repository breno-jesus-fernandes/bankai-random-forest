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

This example uses the Covertype forest-cover dataset, which has 581,012 rows and 54 named features. It samples 100,000 rows with seed `2077`, then treats cover type 2 as the positive class and all other cover types as the negative class. The dataset is provided by scikit-learn and downloaded on first use. [Dataset details](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html)

```python
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from bankai_random_forest import BankaiRandomForestClassifier

covertype = fetch_covtype(as_frame=True)
X = covertype.data
y = (covertype.target == 2).astype("int8")

X, _, y, _ = train_test_split(
    X, y, train_size=100_000, random_state=2077, stratify=y
)

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
    importance_type="permutation", # Native permutation importance
    max_bins=63, # Enable histogram binning
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

The pandas DataFrame keeps the 54 original feature names, so the feature importances are labeled with names such as `Elevation`, `Aspect`, and `Slope`.

## Classifier reference

`BankaiRandomForestClassifier` follows scikit-learn's estimator pattern. The defaults and parameter meanings are listed here; behavior marked as a Bankai limitation is intentionally called out because it differs from scikit-learn.

```python
from bankai_random_forest import BankaiRandomForestClassifier

model = BankaiRandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
    importance_type="gain",
    max_bins=None,
    binning_strategy="exact_sort",
    bin_sample_size=200_000,
)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `n_estimators` | `100` | Number of trees in the forest. |
| `criterion` | `"gini"` | Split criterion: `"gini"`, `"entropy"`, or `"log_loss"`. `entropy` and `log_loss` use the same entropy criterion. |
| `max_depth` | `None` | Maximum tree depth. `None` uses Bankai's internal depth cap of 512; otherwise use an integer of at least 1. |
| `min_samples_split` | `2` | Minimum samples required to split a node. Accepts an integer of at least 2 or a fraction in `(0, 1]` of the training rows. |
| `min_samples_leaf` | `1` | Minimum samples required at a leaf. Accepts an integer of at least 1 or a fraction in `(0, 1]` of the training rows. |
| `min_weight_fraction_leaf` | `0.0` | Minimum fraction of the total sample weight required at a leaf; must be between 0 and 0.5. |
| `max_features` | `"sqrt"` | Features considered at each split: `"sqrt"`, `"log2"`, `None` (all features), an integer count, or a fraction in `(0, 1]`. |
| `max_leaf_nodes` | `None` | Maximum number of leaves per tree. `None` means no explicit leaf limit; otherwise use an integer of at least 1. |
| `min_impurity_decrease` | `0.0` | A split is made only when its impurity decrease is at least this non-negative value. |
| `bootstrap` | `True` | Whether each tree is trained on a bootstrap sample. Required for out-of-bag scoring and permutation importance. |
| `oob_score` | `False` | If true, calculate out-of-bag predictions and accuracy; a callable can provide a custom score. Requires `bootstrap=True`. |
| `n_jobs` | `None` | Number of worker threads. `None` uses one worker, a positive integer sets the count, and negative values follow joblib CPU-count rules (`-1` uses all available CPUs). `0` is invalid. |
| `random_state` | `None` | Seed or `RandomState` controlling bootstrap samples, feature selection, and other randomized steps. Set it for repeatable fits. |
| `verbose` | `0` | Non-negative integer verbosity level. Values above zero print fit progress. |
| `warm_start` | `False` | Accepted for estimator compatibility. Currently, calling `fit` again rebuilds the forest instead of adding trees to the previous fit. |
| `class_weight` | `None` | `None`, `"balanced"`, `"balanced_subsample"`, a class-to-weight dictionary, or (for multioutput targets) a list of dictionaries, one per output. |
| `ccp_alpha` | `0.0` | Non-negative cost-complexity pruning strength. Larger values prune more of each tree. |
| `max_samples` | `None` | Number or fraction of rows sampled for each tree. Accepts an integer count or a fraction in `(0, 1]`; only valid with `bootstrap=True`. `None` samples the full training size. |
| `monotonic_cst` | `None` | One constraint per input feature: `-1` decreasing, `0` unconstrained, or `1` increasing. Currently supported only for binary single-output classification. |
| `importance_type` | `"gain"` | How `feature_importances_` is calculated: `"gain"` (impurity reduction), `"split"` (split counts), or `"permutation"` (out-of-bag accuracy decrease). Permutation importance requires `bootstrap=True` and adds work during `fit`. |
| `max_bins` | `None` | `None` uses exact split search. An integer from 2 to 255 enables histogram split search. |
| `binning_strategy` | `"exact_sort"` | How histogram cut points are found: `"exact_sort"`, `"sampled_sort"`, `"exact_select"`, or `"sampled_select"`. Applies only when `max_bins` is set. |
| `bin_sample_size` | `200_000` | Positive maximum number of rows used per feature by sampled histogram strategies. Applies only when `max_bins` is set. |

Call `fit(X, y, sample_weight=None)` to train. `sample_weight` accepts one finite, non-negative weight per row. The fitted estimator exposes scikit-learn-style methods including `predict`, `predict_proba`, and `predict_log_proba`, plus `classes_`, `feature_importances_`, `n_features_in_`, and `feature_names_in_` when the input has named columns.

`predict_proba` returns one probability array for a single target and a list of arrays for multioutput targets. `monotonic_cst` is not supported for multioutput classification. `oob_score=True` computes accuracy; to use another metric, pass a callable with signature `metric(y_true, y_pred)`.

## Regressor

`BankaiRandomForestRegressor` follows the `RandomForestRegressor` parameter defaults and
supports single and multioutput regression. Regression currently delegates training and
prediction to scikit-learn; `max_bins`, `binning_strategy`, and `bin_sample_size` are
accepted for API consistency but do not affect regression training. The supported criteria
are `squared_error`, `absolute_error`, and `poisson`; Poisson targets must
be nonnegative and have a positive sum. With `importance_type="permutation"`, set
`oob_score=True`; importance is the decrease in R² from aggregated OOB predictions.

## Development

To build and run the project locally, install Git, Python 3.11 or newer, a stable Rust toolchain, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/). On Windows, install the Rust MSVC build tools; on macOS, install the Xcode command-line tools.

Clone the repository and create the locked development environment:

```bash
git clone https://github.com/breno-jesus-fernandes/bankai-random-forest.git
cd bankai-random-forest
uv python install 3.11
uv sync --locked --no-install-project
uv run maturin develop --release --locked
```

Run the Python and Rust test suites from the repository root:

```bash
uv run pytest -q
uv run cargo test --workspace --locked
```

Benchmark scripts use an additional dependency group. Install it when needed with `uv sync --locked --group benchmark`. All commands use the versions recorded in `uv.lock` and `Cargo.lock`.

Contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
See the [contribution guide](CONTRIBUTING.md) for the fork and pull request workflow,
local checks, and maintainer review expectations.

## License

Bankai is licensed under [GPL-3.0-or-later](COPYING). The maintained XRF fork
is also GPL-3.0-or-later; see [`NOTICE`](NOTICE) for upstream attribution.
