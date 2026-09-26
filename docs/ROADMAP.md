# Bankai Random Forest Roadmap

## Purpose

Build a GPL-3.0-or-later Rust/Python random forest package with a
scikit-learn-compatible classifier API. The initial backend is a maintained fork
of XRF. An optional histogram training mode accelerates continuous features
without leaking backend-specific types through the Python ABI.

## Decisions Locked In

- [x] Build backend: Maturin only.
- [x] Rust/Python binding: PyO3 only.
- [x] Python package management and publishing: uv.
- [x] Minimum supported Python: CPython 3.11.
- [x] Wheel ABI target: `abi3-py311`.
- [x] Reference API: `sklearn.ensemble.RandomForestClassifier` 1.9.1.
- [x] Core provenance: fork XRF at a pinned upstream revision.
- [x] Distribution license: GPL-3.0-or-later.
- [x] Initial input domain: dense numeric features and one-dimensional targets.
- [x] Initial classification domain: binary and multiclass labels.
- [x] Sparse matrices: explicit rejection, never implicit densification.
- [x] NaN and multioutput targets: explicit rejection in the first release.
- [x] Benchmark baseline B: a project-owned Rust CLI without Python.
- [x] Statistical parity gate: 95% paired confidence interval for F1 delta in
      +/-0.02 and probability RMSE <= 0.05 across ten seeded datasets.

## Mandatory TDD Protocol

Every functional change must follow this sequence:

1. Add the smallest focused pytest or Rust test.
2. Run it and retain the red result.
3. Implement only enough behavior to satisfy that test.
4. Run `uv run maturin develop` after Rust or PyO3 changes.
5. Run the focused test again, then the relevant regression suite.

Infrastructure files and upstream-source acquisition are not behavioral
implementation. All new behavior, including parameter validation and error
messages, must start with a failing test.

## Architecture

```text
Python estimator
  BankaiRandomForestClassifier
          |
          v
PyO3 extension: bankai_random_forest._core
          |
          v
bankai_core: parameter validation, data contracts, model state
          |
          +-- XRF fork: exact/sort-based backend
          +-- optional histogram split path
```

The Python layer owns sklearn validation, labels, tags, and public attributes.
The Rust layer owns tree construction, model state, prediction votes, and backend
selection. Neither XRF nor histogram-specific types may cross the public Python
interface.

## Version Milestones

### v0.1.0: Foundation

- [x] Project directory isolated from the parent Git repository.
- [x] Initial Maturin, PyO3, uv, and pytest metadata added.
- [x] Create a locked uv environment.
- [x] Red test: `from bankai_random_forest import _core` fails.
- [x] Green test: empty PyO3 `_core` module imports after `maturin develop`.
- [x] Vendor the pinned XRF revision with upstream attribution and GPL text.
- [ ] Build and test a wheel on Python 3.11, 3.12, and 3.13.

### v0.2.0: Estimator Contract

- [x] Add the full sklearn 1.9.1 constructor signature.
- [x] Implement `fit`, `predict`, `predict_proba`, and `predict_log_proba`.
- [x] Set `classes_`, `n_classes_`, `n_features_in_`, `feature_names_in_`, and
      `feature_importances_`.
- [x] Implement fitted-state, labels, validation, serialization, and tags.
- [x] Pass `check_estimator` without expected failures.

### v0.3.0: Full Parameter Semantics

- [x] Criteria: Gini, entropy, and log-loss.
- [x] Tree limits: depth, split size, leaf size, leaf weight, feature count,
      maximum leaves, and impurity decrease.
  - [x] Depth, split size, leaf size, leaf weight, feature count, and impurity decrease.
  - [x] Maximum leaves.
- [x] Sampling: bootstrap, `max_samples`, OOB predictions, and OOB score.
  - [x] Bootstrap and `max_samples`.
  - [x] OOB predictions and OOB score.
- [ ] Weights: `sample_weight`, `class_weight`, `balanced`, and
      `balanced_subsample`.
  - [x] `sample_weight`, `class_weight`, and `balanced`.
  - [ ] `balanced_subsample`.
- [x] State: `random_state`, deterministic `n_jobs`, `verbose`, and
      `warm_start`.
  - [x] `random_state` and deterministic `n_jobs`.
  - [x] `warm_start`.
  - [x] `verbose`.
- [ ] Structural controls: `ccp_alpha` and valid binary `monotonic_cst`.

### v0.4.0: Robustness and Data Boundaries

- [x] DataFrame noise and 99:1 imbalance suites.
- [x] Sparse, NaN, multioutput, dimensionality, dtype, and feature-name errors.
  - [x] Sparse and multioutput rejection; DataFrame feature-name preservation.
  - [x] NaN, dimensionality, dtype, and feature-name error coverage.
- [x] Float32/float64 transfer tests with copy telemetry.
- [x] Deterministic trees, predictions, serialization, and thread counts.

### v0.5.0: Benchmark and Distribution

- [x] Implement the pure Rust `bankai-xrf-cli` benchmark runner.
- [x] Benchmark sklearn, CLI, and PyO3 wrapper at 10k and 100k rows.
- [x] Benchmark hyperparameter profiles: criterion, feature count, depth, leaf
      size, bootstrap sampling, and balanced class weights.
- [x] Implement a permutation-importance benchmark category for sklearn, PyO3,
      and the Rust CLI.
- [x] Run the optimized permutation-importance benchmark matrix.
- [x] Export raw CSV and Markdown reports with timings, peak RSS, FFI overhead,
      F1, probability RMSE, agreement, and environment metadata.
- [ ] Build signed release artifacts with Maturin.

Benchmark policy: 100k rows is the largest dataset because of the available RAM.
Every Rust benchmark binary must use the release profile above and be built with
`RUSTFLAGS="-C target-cpu=native" cargo build --release` on the benchmark host.
The benchmark runner also installs the Maturin extension with
`RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release` before it
loads the PyO3 wrapper.

### v0.6.0: Histogram-Accelerated Trees

- [x] Add optional `max_bins` parameter; `None` preserves exact split behavior.
- [x] Pre-bin continuous features once during fit and reuse fitted cut points at
      prediction time.
- [x] Select histogram split candidates while preserving tree constraints and
      native permutation feature importance.
- [x] Run full sklearn contract and regression suites with the default backend.
- [x] Benchmark `fit` plus `feature_importances_` for permutation importance
      across exact and histogram modes, with optimized Rust builds.
- [x] Review model quality and runtime tradeoffs across histogram resolutions.

## Progress Log

| Date | Version | Progress | Notes |
| --- | --- | --- | --- |
| 2026-09-25 | v0.1.0 | Started | Scope, licensing, ABI, and TDD contract agreed. |
| 2026-09-25 | v0.1.0 | Import cycle complete | The empty `cp311-abi3` PyO3 module passed its red/green pytest cycle. |
| 2026-09-25 | v0.1.0 | XRF vendor complete | Pinned XRF source, upstream attribution, NOTICE, and GPL-3.0 text added. |
| 2026-09-25 | v0.2.0 | Baseline in progress | Dense Gini fitting, original-label prediction, sample weights, sparse rejection, and pickle reconstruction work. `class_weight` and `min_weight_fraction_leaf` remain required before the sklearn common-check gate can pass. |
| 2026-09-25 | v0.2.0 | Estimator contract complete | Probability predictions, feature importances, class weights, and minimum leaf-weight support added; `check_estimator` passes without expected failures. |
| 2026-09-25 | v0.3.0 | Parameter controls in progress | Added entropy/log-loss, max-features, depth, minimum split/leaf constraints, class weights, bootstrap selection, and `max_samples`; OOB, parallelism, pruning, and monotonic constraints remain. |
| 2026-09-25 | v0.3.0 | OOB complete | Added `oob_decision_function_` and `oob_score_` for bootstrap training. |
| 2026-09-25 | v0.3.0 | Parallel training complete | Added deterministic parallel tree construction through `n_jobs`. |
| 2026-09-25 | v0.3.0 | Warm start complete | Rebuilds the deterministic forest when the tree count increases. |
| 2026-09-25 | v0.3.0 | Impurity threshold complete | Rejects candidate splits below `min_impurity_decrease`. |
| 2026-09-25 | v0.3.0 | Tree limits complete | Added structural `max_leaf_nodes` enforcement. |
| 2026-09-25 | v0.4.0 | Data boundaries in progress | Added explicit multioutput rejection and DataFrame feature-name coverage. |
| 2026-09-25 | v0.3.0 | State controls complete | Added validated verbose construction reporting. |
| 2026-09-25 | v0.4.0 | Input boundary coverage complete | Added NaN, float32, feature-count, and feature-name mismatch tests. |
| 2026-09-25 | v0.4.0 | Imbalance suite complete | Added a noisy DataFrame 99:1 balanced-weight classifier suite. |
| 2026-09-25 | v0.4.0 | Transfer telemetry complete | Records input/core dtypes, contiguity, and float64 casts. |
| 2026-09-25 | v0.4.0 | Reproducibility complete | Seeded predictions, probabilities, and importances are reproducible. |
| 2026-09-25 | v0.5.0 | Benchmark scope adjusted | Capped datasets at 100k rows and locked optimized native Rust benchmark builds. |
| 2026-09-25 | v0.5.0 | Rust CLI runner complete | Added deterministic synthetic data, CSV/Markdown output, timings, and F1. |
| 2026-09-25 | v0.5.0 | Comparative runner complete | Added sklearn/PyO3/Rust comparison with CSV and Markdown report export. |
| 2026-09-25 | v0.5.0 | Benchmark matrix complete | Ran optimized 10k and 100k comparisons for sklearn, PyO3, and Rust CLI. |
| 2026-09-25 | v0.5.0 | Hyperparameter matrix complete | Ran the optimized 10k-row matrix across criterion, feature count, depth, leaf size, bootstrap sampling, and balanced weights; reports live in `benchmarks/results-hyperparameters-10k/`. |
| 2026-09-26 | v0.3.0 | Importance type selection complete | Added `importance_type="gain"` (criterion-weighted gain), `"split"` (split frequency), and `"permutation"` (native XRF OOB accuracy decrease) through `feature_importances_`. |
| 2026-09-26 | v0.5.0 | Feature importance benchmark complete | Compared gain, split frequency, and permutation for sklearn, PyO3, and Rust CLI on 10k training rows; permutation used a separate 10k validation set and five sklearn repeats. Reports are in `benchmarks/results-feature-importance-10k/`. |
| 2026-09-26 | v0.6.0 | Histogram mode implemented | Added opt-in `max_bins` preprocessing and histogram split search; exact sorting remains the default when `max_bins=None`. |
| 2026-09-26 | v0.6.0 | Histogram benchmark complete | Optimized 10k-row/20-feature/100-tree matrix measures fit plus permutation importance (no prediction): `max_bins=16` took 0.628 s vs exact 1.562 s (2.49x faster); LightGBM RF boosting took 0.841 s and sklearn RF 3.443 s. Higher histogram resolutions did not beat exact. Results are in `benchmarks/results-histogram-10k/`. Bankai OOB permutation is included in fit, while sklearn/LightGBM external permutation passes are added to fit; their model algorithms and Bankai's native OOB importance methodology differ. |
