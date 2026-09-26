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

## TreeSHAP through the direct route

The primary TreeSHAP integration is the direct route:

```python
explainer = shap.TreeExplainer(classifier)
explanation = explainer(X_to_explain)
```

Bankai lazily materializes sklearn-shaped tree objects through `estimators_`
when SHAP requests them. The trees use the node cover retained during native
training and Bankai's per-tree class vote as the leaf value. The resulting
values have shape `(samples, features, classes)` and add to `predict_proba`
with the explainer's expected values. SHAP is optional: users who want
TreeSHAP must install it, but it is not a Bankai runtime dependency.

The explicit sklearn adapter and Rust-native TreeSHAP methods remain available
for comparative experiments; they are not the recommended explanation API.

Run the release benchmark with:

```bash
RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release
.venv/bin/python benchmarks/run_tree_shap_benchmark.py --rows 1000 \
  --output-dir benchmarks/results-tree-shap-1k
```

The benchmark compares direct, adapter, native, and permutation routes. The
recorded run uses 1,000 rows × 20 features × 100 trees for three seeds in
exact and histogram modes; it explains 100 rows and records
export/adaptation, explainer, calculation, total, per-row latency, and process
peak RSS. The release extension was built with `target-cpu=native`. Median
CSV and Markdown reports are in `benchmarks/results-tree-shap-1k/`. Direct
TreeSHAP was additive to better than `5e-14` on this multiclass benchmark.
The permutation baseline has a 100-row background and `max_evals=41`; it is a
speed baseline only, not an expected value-for-value match with
`tree_path_dependent`.

### sklearn vs Bankai direct TreeSHAP

The direct-route comparison uses sklearn 1.9.1 and SHAP 0.51.0, with 1,000
training rows, 20 features, 100 trees, three seeds, and 100 explained rows per
seed. Bankai uses release Rust binaries built with `target-cpu=native`; the
table reports medians. Bankai exact training took 0.259 s for SHAP calculation
and 0.272 s including lazy tree export. sklearn took 0.237 s for calculation
and 0.239 s total. Bankai `max_bins=16` took 0.288 s for calculation and
0.302 s total. All routes had maximum additivity error below `5e-14`.

The direct comparison report, including raw per-seed timings, is in
`benchmarks/results-tree-shap-sklearn-vs-bankai-1k/`.

The larger comparison uses 10,000 rows and 40 features, with the same 100
trees, three seeds, and 100 explained rows. Median calculation times were
4.258 s for sklearn, 4.511 s for Bankai exact, and 4.648 s for Bankai
`max_bins=16`. Including lazy tree export, Bankai totals were 4.608 s and
4.770 s, respectively. Maximum additivity error stayed below `2.2e-12`.
Reports are in `benchmarks/results-tree-shap-sklearn-vs-bankai-10k-40f/`.

SHAP is a development dependency for the compatibility test only; it is not a
runtime dependency of Bankai.
