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
- [x] Cache node histograms and derive the larger child's histograms by
      subtracting the smaller child's from the parent.
- [x] Store histogram bin IDs in compact integer arrays instead of `f64`.
- [x] Exclude globally constant features from histogram-mode split sampling
      while keeping original feature IDs and importance output positions.

### v0.7.0: SHAP Compatibility

- [x] Verify model-agnostic SHAP permutation explanations through
      `predict_proba`, including binary class output shapes and additivity.
- [x] Expose Bankai's native tree structure through sklearn's tree view so
      `shap.TreeExplainer(classifier)` uses the direct route.
- [x] Validate binary and multiclass TreeSHAP additivity for exact and
      histogram-trained forests.

### v0.8.0: Joblib Serialization Compatibility

- [x] Verify `joblib.dump` and `joblib.load` round trips for fitted Bankai
      classifiers using exact and histogram training.
- [x] Check that restored estimators preserve parameters, class labels,
      predictions, probabilities, and feature importances.
- [x] Assess artifact behavior with compression and `mmap_mode`, and record
      any supported-use constraints.
- [x] Document the tested joblib, Python, and Bankai versions and the
      serialization compatibility policy.

### v0.9.0: Scikit-learn Compatibility Gap Audit

- [x] Compare Bankai's public API, parameters, attributes, estimator tags,
      validation, and prediction behavior with the pinned sklearn
      `RandomForestClassifier` contract.
- [x] Inventory each difference as supported, partial, unsupported, or an
      intentional Bankai boundary; add focused compatibility tests for the
      gaps selected for follow-up.
- [x] Reassess known gaps: `class_weight="balanced_subsample"`, `ccp_alpha`,
      `monotonic_cst`, sparse input, multioutput targets, and sklearn tree
      inspection methods such as `apply` and `decision_path`.
- [x] Prioritize implementation work and document explicit non-goals so users
      can distinguish intentional boundaries from accidental incompatibility.

### v1.0.0: Core Scikit-learn Compatibility Remediation

- [x] Fix estimator tags so sklearn identifies Bankai as a classifier and
      classifier meta-estimators can use it.
- [x] Implement `class_weight="balanced_subsample"` with per-bootstrap-sample
      weighting, including sklearn's non-bootstrap behavior.
- [x] Match sklearn's `max_samples` semantics for fractional and weighted
      samples, including flooring the computed sample count.
- [x] Support callable `oob_score` functions with sklearn-compatible inputs
      and expose the returned score through `oob_score_`.
- [x] Turn the four v0.9 `xfail` cases into passing compatibility tests and
      add focused reference-behavior coverage for the corrected semantics.
- [x] Run the full Python suite and sklearn estimator checks; update the
      compatibility matrix with verified results.
- [ ] Keep `ccp_alpha`, `monotonic_cst`, sparse/NaN inputs, and multioutput
      behavior unchanged in this milestone and document them as deferred
      backend or input-domain work.

### v1.1.0: Bootstrap Class Weight Parity

- [x] Add an XRF input hook so each tree computes `balanced_subsample` weights
      from its own bootstrap multiplicities without copying the dense matrix.
- [x] Verify weighted and unweighted per-tree behavior against sklearn,
      including deterministic sequential and parallel forests.
- [x] Preserve OOB predictions and feature importance under per-tree weights.
- [x] Verify exact and histogram behavior, non-bootstrap class weights, full
      regression tests, estimator checks, and release performance gates.

### Compatibility Expansion Quality Gates

- Every behavioral change follows the mandatory red/green TDD protocol above;
  the focused test must fail before implementation, then pass after it.
- Every milestone runs the full Python and Rust suites plus the applicable
  sklearn, joblib, TreeSHAP, exact-backend, histogram-backend, importance, and
  parallelism regression tests. Existing predictions, probabilities, public
  attributes, serialization, and supported input behavior must remain stable.
- Every performance comparison uses release binaries and an untimed warmup for
  each implementation/configuration. Discard the warmup, recreate the model
  with the same fixed seed, then collect at least three paired measurements
  over at least three fixed seeds. Record median train and prediction time,
  peak RSS, environment, and raw CSV results.
- Compare with the pre-milestone baseline on equivalent data and hardware.
  A regression greater than 5% in median training time, prediction time, or
  peak RSS blocks completion. A new feature's own cost is reported separately
  from regression on existing configurations.

### v1.2.0: Cost-Complexity Pruning

- [ ] Implement sklearn-compatible `ccp_alpha` pruning for positive alpha while
      preserving the existing unpruned tree behavior at `ccp_alpha=0`.
- [ ] Add reference and regression tests for pruned structures, predictions,
      probabilities, importances, `apply`, `decision_path`, TreeSHAP, and joblib.
- [ ] Verify exact and histogram modes and pass all compatibility quality gates.

### v1.3.0: Monotonic Constraints

- [ ] Implement sklearn-compatible `monotonic_cst` for binary classification.
- [ ] Test increasing, decreasing, and unconstrained features; invalid
      constraints; probabilities; exact and histogram modes; TreeSHAP; and joblib.
- [ ] Verify existing unconstrained models retain behavior and pass all quality
      gates.

### v1.4.0: Sparse Feature Matrices

- [ ] Support SciPy CSR and CSC inputs in fit and prediction without implicit
      densification.
- [ ] Test predictions, probabilities, OOB, importances, validation, and
      sklearn parity for both formats and supported exact/histogram modes.
- [ ] Verify dense inputs retain behavior and pass all quality gates.

### v1.5.0: Missing Feature Values

- [ ] Support NaN values in training and prediction with tree-learned missing
      value routing compatible with sklearn.
- [ ] Test missing-value patterns, features without training NaNs, OOB,
      probabilities, exact and histogram modes, and serialization.
- [ ] Verify finite-only data retains behavior and pass all quality gates.

### v1.6.0: Multioutput Classification

- [ ] Support sklearn RandomForestClassifier multioutput classification for
      multiclass targets per output and multilabel indicator targets.
- [ ] Test output shapes and values for `predict` and `predict_proba`, weights,
      metrics, validation, joblib, and sklearn parity.
- [ ] Verify single-output classification retains behavior and pass all quality
      gates.

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
| 2026-09-26 | v0.6.0 | Histogram internals optimized | Added node histogram caching/subtraction, `u8` bin storage, and histogram-mode filtering of globally constant features; original feature indices remain stable. Full Python and Rust suites pass. The optimized run reduced 255-bin time from 2.883 s to 1.616 s and 128-bin time from 1.633 s to 1.073 s; updated report is in `benchmarks/results-histogram-10k/`. |
| 2026-09-26 | v0.7.0 | SHAP API compatibility assessed | Verified SHAP 0.51.0's model-agnostic permutation explainer with `predict_proba` and additivity. Initial `TreeExplainer` use raised `InvalidModelError`; Bankai's native tree export later enabled the direct route. Details are in `docs/SHAP_COMPATIBILITY.md` and `tests/test_shap_compatibility.py`. |
| 2026-09-26 | v0.7.0 | Experimental TreeSHAP benchmark complete | Added native Rust TreeSHAP plus sklearn-shaped direct and adapter routes; release benchmark on 1k rows, 20 features, 100 trees, and three seeds measured exact and `max_bins=16` with a permutation baseline. Reports are in `benchmarks/results-tree-shap-1k/`; multiclass TreeSHAP additivity error stayed below `5e-14`. |
| 2026-09-26 | v0.7.0 | Direct TreeSHAP selected | `shap.TreeExplainer(classifier)` is the recommended integration; explicit adapter and native methods are retained only for benchmark comparisons. |
| 2026-09-26 | v0.7.0 | sklearn vs Bankai TreeSHAP benchmark complete | Compared SHAP 0.51.0 on sklearn 1.9.1 and Bankai release (`target-cpu=native`) using 1k rows, 20 features, 100 trees, three seeds, and 100 explained rows. Median calculation times were 0.237 s for sklearn, 0.259 s for Bankai exact, and 0.288 s for Bankai `max_bins=16`; all maximum additivity errors were below `5e-14`. Reports are in `benchmarks/results-tree-shap-sklearn-vs-bankai-1k/`. |
| 2026-09-26 | v0.7.0 | Larger sklearn vs Bankai TreeSHAP benchmark complete | Repeated the direct comparison at 10k rows and 40 features, with 100 trees, three seeds, and 100 explained rows. Median calculation times were 4.258 s for sklearn, 4.511 s for Bankai exact, and 4.648 s for Bankai `max_bins=16`; maximum additivity error stayed below `2.2e-12`. Reports are in `benchmarks/results-tree-shap-sklearn-vs-bankai-10k-40f/`. |
| 2026-09-26 | v0.7.0 | TreeSHAP milestone complete | Direct `TreeExplainer(classifier)` support and binary/multiclass additivity validation pass for exact and histogram forests; the focused suite reports 4 passed. |
| 2026-09-26 | v0.8.0 | Joblib compatibility complete | Joblib round trips pass for exact and histogram models, preserving estimator parameters, classes, predictions, probabilities, and importances. Uncompressed `mmap_mode="r"` works; compressed files load with joblib's mmap-unavailable warning. Bankai reconstructs the native forest from saved training arrays. Tested with joblib 1.6.0, Python 3.11.11, Bankai 0.1.0, and sklearn 1.9.1; policy is documented in `docs/JOBLIB_COMPATIBILITY.md`. |
| 2026-09-26 | v0.9.0 | sklearn compatibility gap audit complete | Documented the sklearn 1.9.1 comparison and priorities in `docs/SKLEARN_COMPATIBILITY.md`. The full suite reports 72 passed and four expected xfails (classifier tags, `balanced_subsample`, fractional `max_samples`, callable OOB scoring); two existing `log(0)` warnings remain. |
| 2026-09-26 | v1.0.0 | Core sklearn compatibility remediation planned | Correct the four highest-priority v0.9 gaps while keeping backend-heavy pruning/monotonic features and intentional sparse/NaN/multioutput boundaries outside this milestone. |
| 2026-09-26 | v1.0.0 | Compatibility remediation implemented | Fixed classifier tags, fractional/weighted `max_samples` flooring, and callable OOB scoring. `balanced_subsample` is accepted and handles `bootstrap=False`; exact per-bootstrap weighting is carried into v1.1. Release extension build passed and the full suite reports 76 passed. |
| 2026-09-26 | v1.1.0 | Bootstrap class weight parity complete | Added per-tree balanced bootstrap weights through shared dense feature storage. Verified non-bootstrap weighting, deterministic sequential/parallel forests, OOB scoring and permutation importance. Release benchmarks on 10k rows stayed within the 5% gate for existing exact and histogram paths; details and raw measurements are in `benchmarks/results-balanced-subsample-10k/`. |
| 2026-09-26 | v1.2.0-v1.6.0 | Compatibility expansion planned | Added independent pruning, monotonicity, sparse, NaN, and multioutput milestones. Each requires TDD, full compatibility regression coverage, release benchmarks with untimed warmups, and a maximum 5% regression in existing workloads. |
