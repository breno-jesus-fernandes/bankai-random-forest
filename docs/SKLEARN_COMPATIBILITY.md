# scikit-learn compatibility audit

This audit compares Bankai with the repository's pinned scikit-learn
`RandomForestClassifier` contract (scikit-learn 1.9.1). The characterization
tests are in `tests/test_sklearn_compatibility_audit.py`.

## Findings

| Area | Status | Finding |
| --- | --- | --- |
| Constructor parameters | Supported with extensions | Bankai exposes all parameters from sklearn 1.9.1 and adds `importance_type` and `max_bins`. Matching parameter names do not guarantee identical semantics for every option below. |
| Core classifier API | Supported | `fit`, `predict`, `predict_proba`, `predict_log_proba`, `score`, classes, feature count, and importances are available. Binary and multiclass single-output predictions are covered elsewhere in the suite. |
| Forest inspection | Supported in tested paths | Inherited `apply` and `decision_path` work through Bankai's sklearn-shaped `estimators_`; output dimensions were checked. |
| Metadata routing | Supported for requested sample weights | `set_fit_request(sample_weight=True)` routes through a sklearn `Pipeline` when metadata routing is enabled. |
| Classifier estimator tags | Supported | Bankai provides sklearn classifier and single-output target tags; `is_classifier` recognizes the estimator. |
| `class_weight="balanced_subsample"` | Supported | With bootstrap enabled, each tree computes class factors from its own bootstrap multiplicities; with bootstrap disabled, behavior matches `balanced`. User sample weights are multiplied by the class factors. |
| `ccp_alpha` | Supported | Cost-complexity pruning supports exact and histogram trees; nonnegative finite values are accepted and `ccp_alpha=0` preserves unpruned behavior. |
| `monotonic_cst` | Supported for binary single-output classification | Accepts one `-1`, `0`, or `1` value per feature. Constraints apply to the probability of the positive class; exact and histogram split searches enforce inherited bounds. Multiclass is rejected. NaN and multioutput remain unsupported. |
| `max_samples` | Supported for supported inputs | Fractional sample counts use floor and weighted data uses the effective sum of sample weights, matching sklearn's sample-count rule. |
| `oob_score` | Supported | Boolean scoring and callable scoring are supported; callables receive encoded targets and OOB argmax predictions. |
| `warm_start` | Partial | Increasing `n_estimators` rebuilds the deterministic forest instead of appending only the new trees. Final predictions are covered, but incremental-fit performance differs. |
| Feature importance | Partial semantic parity | The sklearn attribute exists. Bankai also exposes split-count, gain, and OOB permutation modes; values are not guaranteed to match sklearn's default impurity importance exactly. |
| Sparse input | Supported for CSR/CSC feature matrices | Fit and prediction accept SciPy CSR and CSC without materializing the full matrix as dense. Implicit entries are zero. Exact and histogram modes, OOB, importances, validation, and sklearn prediction labels have coverage. |
| NaN feature input | Supported | Dense and CSR/CSC matrices accept NaN. Exact and histogram splits learn a missing-value direction; prediction uses it, or sends unseen NaNs to the larger child. Multioutput targets remain unsupported. |
| Multioutput input | Intentional Bankai boundary | Multioutput targets remain unsupported while the v1.6 milestone is open. |

## Priorities

1. Keep multioutput targets as an explicit follow-up decision; they require
   broader estimator and prediction-shape changes.
2. Preserve the current `apply` and `decision_path` behavior with regression
   coverage; these inherited methods work in the tested sklearn version.

## Verification environment

Verification ran against scikit-learn 1.9.1 and Python 3.11.11. The release
extension build, sklearn estimator checks, Rust tests, and full Python suite
passed (140 Python tests and 19 Rust tests). NaN coverage exercises learned
routing and fallback for dense/CSR/CSC, exact/histogram, OOB, apply,
decision_path, TreeSHAP, sklearn parity, and joblib. Sparse coverage checks CSR/CSC
against dense results, sklearn predictions, OOB, validation, and joblib.
Monotonicity coverage compares
increasing and decreasing constraints against sklearn's probability invariants
in exact and histogram modes, and checks joblib and TreeSHAP compatibility.
Two existing `divide by zero` warnings from `log(0)` remain in
`predict_log_proba` coverage. Release benchmarks with warmups are documented
in `benchmarks/results-monotonic-10k/`, `benchmarks/results-sparse-10k/`, and
`benchmarks/results-nan-10k/`. The NaN implementation is still being optimized:
several finite-only workloads currently exceed the roadmap's 5% regression
limit, so v1.5.0 remains open.
