# scikit-learn compatibility audit

This audit compares Bankai with the repository's pinned scikit-learn
`RandomForestClassifier` contract (scikit-learn 1.9.1). It records observed
behavior; it does not change the estimator implementation. The characterization
tests are in `tests/test_sklearn_compatibility_audit.py`. Four known gaps are
marked `xfail` so they remain visible without making the existing suite fail.

## Findings

| Area | Status | Finding |
| --- | --- | --- |
| Constructor parameters | Supported with extensions | Bankai exposes all parameters from sklearn 1.9.1 and adds `importance_type` and `max_bins`. Matching parameter names do not guarantee identical semantics for every option below. |
| Core classifier API | Supported | `fit`, `predict`, `predict_proba`, `predict_log_proba`, `score`, classes, feature count, and importances are available. Binary and multiclass single-output predictions are covered elsewhere in the suite. |
| Forest inspection | Supported in tested paths | Inherited `apply` and `decision_path` work through Bankai's sklearn-shaped `estimators_`; output dimensions were checked. |
| Metadata routing | Supported for requested sample weights | `set_fit_request(sample_weight=True)` routes through a sklearn `Pipeline` when metadata routing is enabled. |
| Classifier estimator tags | Unsupported; high priority | `is_classifier(BankaiRandomForestClassifier())` returns `False`. Bankai's `__sklearn_tags__` currently returns base estimator tags without classifier tags. This can prevent sklearn meta-estimators from recognizing Bankai as a classifier. |
| `class_weight="balanced_subsample"` | Unsupported | `fit` passes the preset to `compute_sample_weight`, which rejects it. sklearn computes this preset per bootstrap sample. |
| `ccp_alpha` and `monotonic_cst` | Unsupported beyond defaults | Non-default values raise `NotImplementedError`; the default `ccp_alpha=0.0` and `monotonic_cst=None` are accepted. Existing rejection tests are in `tests/test_estimator_fit.py`. |
| `max_samples` | Partial | Integer values are supported. Fractional values use Python `round` over row count, while sklearn floors the fraction and bases weighted sampling on the sum of sample weights. |
| `oob_score` | Partial | Boolean scoring and OOB attributes are supported. sklearn also accepts a callable scoring function; Bankai currently restricts the value to booleans. |
| `warm_start` | Partial | Increasing `n_estimators` rebuilds the deterministic forest instead of appending only the new trees. Final predictions are covered, but incremental-fit performance differs. |
| Feature importance | Partial semantic parity | The sklearn attribute exists. Bankai also exposes split-count, gain, and OOB permutation modes; values are not guaranteed to match sklearn's default impurity importance exactly. |
| Sparse, NaN, multioutput input | Intentional Bankai boundary | Bankai explicitly rejects these inputs. The rejection behavior is covered in `tests/test_estimator_fit.py` and was a locked initial scope decision. |

## Priorities

1. Fix classifier tags first because sklearn currently misidentifies the
   estimator type.
2. Decide whether to implement the sklearn-compatible semantics for
   `balanced_subsample`, fractional/weighted `max_samples`, and callable OOB
   scoring; focused `xfail` tests identify these gaps.
3. Keep pruning, monotonic constraints, sparse/NaN input, and multioutput as
   explicit follow-up decisions. They require backend or scope changes beyond
   this audit.
4. Preserve the current `apply` and `decision_path` behavior with regression
   coverage; these inherited methods work in the tested sklearn version.

## Verification environment

The audit ran against scikit-learn 1.9.1 and Python 3.11.11. The pre-audit
suite passed 69 tests. The new audit module checks supported inherited methods,
metadata routing, API parameters, and reports four known gaps as expected
failures. The full suite was rerun after adding it.
