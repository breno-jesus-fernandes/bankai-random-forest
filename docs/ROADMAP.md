# Bankai Random Forest Roadmap

## Purpose

Build a GPL-3.0-or-later Rust/Python random forest package with a
scikit-learn-compatible classifier API. The initial backend is a maintained fork
of XRF. The architecture must leave room for a future histogram-based backend
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
          +-- future histogram backend
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
- [ ] Tree limits: depth, split size, leaf size, leaf weight, feature count,
      maximum leaves, and impurity decrease.
  - [x] Depth, split size, leaf size, leaf weight, and feature count.
  - [ ] Maximum leaves and impurity decrease.
- [x] Sampling: bootstrap, `max_samples`, OOB predictions, and OOB score.
  - [x] Bootstrap and `max_samples`.
  - [x] OOB predictions and OOB score.
- [ ] Weights: `sample_weight`, `class_weight`, `balanced`, and
      `balanced_subsample`.
  - [x] `sample_weight`, `class_weight`, and `balanced`.
  - [ ] `balanced_subsample`.
- [ ] State: `random_state`, deterministic `n_jobs`, `verbose`, and
      `warm_start`.
- [ ] Structural controls: `ccp_alpha` and valid binary `monotonic_cst`.

### v0.4.0: Robustness and Data Boundaries

- [ ] DataFrame noise and 99:1 imbalance suites.
- [ ] Sparse, NaN, multioutput, dimensionality, dtype, and feature-name errors.
- [ ] Float32/float64 transfer tests with copy telemetry.
- [ ] Deterministic trees, predictions, serialization, and thread counts.

### v0.5.0: Benchmark and Distribution

- [ ] Implement the pure Rust `bankai-xrf-cli` benchmark runner.
- [ ] Benchmark sklearn, CLI, and PyO3 wrapper at 10k, 100k, and 1M rows.
- [ ] Export raw CSV and Markdown reports with timings, peak RSS, FFI overhead,
      F1, probability RMSE, agreement, and environment metadata.
- [ ] Build signed release artifacts with Maturin.
- [ ] Publish first to TestPyPI and then PyPI with `uv publish`.

### Future: Histogram Backend

- [ ] Define a backend-neutral Rust trait before adding histogram code.
- [ ] Preserve the sklearn contract test suite for both backends.
- [ ] Add binning, histogram caching, split selection, and deterministic merge
      rules behind the backend boundary.
- [ ] Benchmark accuracy, memory, and speed against the XRF backend.

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
