# scikit-learn compatibility audit

This audit compares Bankai with the repository's pinned scikit-learn
`RandomForestClassifier` contract (scikit-learn 1.9.1). The characterization
tests are in `tests/test_sklearn_compatibility_audit.py`.

## Findings

| Area | Status | Finding |
| --- | --- | --- |
| Constructor parameters | Supported with extensions | Bankai exposes all parameters from sklearn 1.9.1 and adds `importance_type`, `max_bins`, `binning_strategy`, and `bin_sample_size`. Matching parameter names do not guarantee identical semantics for every option below. |
| Core classifier API | Supported for single-output; multioutput public predictions supported | `fit`, `predict`, `predict_proba`, `predict_log_proba`, `score`, classes, feature count, and importances are available. Multioutput targets use independent native forests per target. |
| Forest inspection | Supported in tested paths | Inherited `apply` and `decision_path` work through Bankai's sklearn-shaped `estimators_`; output dimensions were checked. |
| Metadata routing | Supported for requested sample weights | `set_fit_request(sample_weight=True)` routes through a sklearn `Pipeline` when metadata routing is enabled. |
| Classifier estimator tags | Supported | Bankai provides sklearn classifier and multioutput target tags; `is_classifier` recognizes the estimator. |
| `class_weight="balanced_subsample"` | Supported | With bootstrap enabled, each tree computes class factors from its own bootstrap multiplicities; with bootstrap disabled, behavior matches `balanced`. User sample weights are multiplied by the class factors. |
| `ccp_alpha` | Supported | Cost-complexity pruning supports exact and histogram trees; nonnegative finite values are accepted and `ccp_alpha=0` preserves unpruned behavior. |
| `monotonic_cst` | Supported for binary single-output classification | Accepts one `-1`, `0`, or `1` value per feature. Constraints apply to the probability of the positive class; exact and histogram split searches enforce inherited bounds. Multiclass is rejected. NaN and multioutput remain unsupported. |
| `max_samples` | Supported for supported inputs | Fractional sample counts use floor and weighted data uses the effective sum of sample weights, matching sklearn's sample-count rule. |
| `oob_score` | Supported | Boolean scoring and callable scoring are supported; callables receive encoded targets and OOB argmax predictions. |
| `warm_start` | Partial | Increasing `n_estimators` rebuilds the deterministic forest instead of appending only the new trees. Final predictions are covered, but incremental-fit performance differs. |
| Feature importance | Partial semantic parity | The sklearn attribute exists. Bankai also exposes split-count, gain, and OOB permutation modes; values are not guaranteed to match sklearn's default impurity importance exactly. |
| Sparse input | Supported for CSR/CSC feature matrices | Fit and prediction accept SciPy CSR and CSC without materializing the full matrix as dense. Implicit entries are zero. Exact and histogram modes, OOB, importances, validation, and sklearn prediction labels have coverage. |
| NaN feature input | Supported | Dense and CSR/CSC matrices accept NaN. Exact and histogram splits learn a missing-value direction; prediction uses it, or sends unseen NaNs to the larger child. |
| Multioutput input | Supported with algorithmic difference | Multiclass-multioutput and multilabel targets expose sklearn-compatible prediction, probability, class, and OOB shapes. Bankai trains one native forest per output, while sklearn shares tree structures; exact trees and probabilities can differ. `estimators_`, `apply`, `decision_path`, and TreeSHAP inspect the first output forest. `score` follows sklearn's `accuracy_score` behavior: multilabel indicators are scored, while multiclass-multioutput raises the same unsupported-target error. |

## Priorities

1. Consider whether multioutput inspection and TreeSHAP should expose a
   per-output result API in a later compatibility milestone.
2. Preserve single-output `apply` and `decision_path` behavior; these inherited
   methods work in the tested sklearn version.

## Verification environment

Verification ran against scikit-learn 1.9.1 and Python 3.11.11. The release
extension build, sklearn estimator checks, Rust tests, and full Python suite
passed (147 Python tests and 20 Rust tests). NaN coverage exercises learned
routing and fallback for dense/CSR/CSC, exact/histogram, OOB, apply,
decision_path, TreeSHAP, sklearn parity, and joblib. Sparse coverage checks CSR/CSC
against dense results, sklearn predictions, OOB, validation, and joblib.
Monotonicity coverage compares
increasing and decreasing constraints against sklearn's probability invariants
in exact and histogram modes, and checks joblib and TreeSHAP compatibility.
Two existing `divide by zero` warnings from `log(0)` remain in
`predict_log_proba` coverage. Release benchmarks with warmups are documented
in `benchmarks/results-monotonic-10k/`, `benchmarks/results-sparse-10k/`,
`benchmarks/results-nan-10k/`, and `benchmarks/results-multioutput-10k-20f/`.
The paired NaN and multioutput single-output release comparisons include
untimed warmups, native prediction timings, and tree-size diagnostics; all
finite-input workloads stayed within the roadmap's 5% regression limit.
Multioutput comparisons also record label agreement and probability RMSE
against sklearn.
