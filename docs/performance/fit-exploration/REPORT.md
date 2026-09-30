# Fit optimization campaign report

## Outcome

The binary-classification specialization in `src/dense.rs` is accepted for the
default path. It removes heap-backed class accumulators from the hot histogram
split loop for two-class models and keeps the generic path for arbitrary class
counts. It does not change public parameters, defaults, model storage, RNG use or
the order of split arithmetic.

Baseline: `76ef49844d90c8f9e43871561ba76235219d5e49` from `origin/master`.
Feature branch: `feature/fit-evidence-driven-optimization`. All conclusions are
limited to this Apple M1, macOS arm64, and datasets of at most 100,000 rows.

## Acceptance measurements

The primary dataset contains 100,000 rows, 500 float32 features (90 informative),
80,000 training rows and 20,000 validation rows. The model uses 40 trees, depth
20, at most 511 leaves, minimum leaf size 5, all features, 80% bootstrap, 63 bins,
gain importance and `n_jobs=8`.

| Evidence | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| Final-build median fit, 5 paired runs | 7.0304 s | 5.8307 s | 17.06% lower; paired speedup 95% CI [1.1854, 1.3292] |
| Median sampled fit RSS | 1,989,459,968 B | 2,070,953,984 B | 4.10% higher |
| Maximum sampled fit RSS | 2,187,476,992 B | 2,205,810,688 B | 0.84% higher; under 10% limit |
| Quality matrix, 9 seed pairs | — | — | Exact predictions; accuracy, precision, recall and F1 deltas all 0 |

All required numeric gates pass. `acceptance.json` contains the calculations.
The 95% timing interval is a fixed-seed bootstrap over paired speedup ratios.

| Control | Median fit change | Peak RSS gate |
| --- | ---: | --- |
| `n_jobs=1` | 14.41% faster | Pass |
| `n_jobs=-1` | 16.19% faster | Pass |
| Exact training, 15 pairs | 2.42% faster | Pass |
| float64 | 38.59% faster | Pass |
| Permutation importance | 35.73% faster | Pass |
| Multiclass | 6.19% faster | Pass |
| Sparse | 19.92% faster | Pass |
| NaN | 27.84% faster | Pass |
| Few estimators | 29.47% faster | Pass |

The `n_jobs=-1` and exact-training paired intervals cross parity; their medians
show no regression above the 5% limit, but these controls do not establish a
speedup. The exact-training run was expanded from five to fifteen pairs because
the first result was near the regression boundary.

## Hypotheses and decisions

- **H1, binary split accumulators:** accepted after the primary gain, all controls,
  the nine-pair quality matrix, full tests and cross-build compatibility audit
  passed.
- **H2, Python conversion, canonical sorting and bin construction:** no rewrite.
  The instrumented profile measured canonical sorting at 0.0073 s, native
  conversion at 0.00014 s, and edge building plus bin application at 0.3671 s,
  versus 8.867 s total fit. These stages cannot alone explain a 5% gain.
- **H3, thread scheduling and histogram cache retention:** no implementation
  change. Thread controls pass the median regression and RSS limits, while
  `n_jobs=-1` remains noisy. Cache pressure is a follow-up hypothesis based on
  code inspection, not measured allocation attribution.
- **Input layouts:** C, Fortran, pandas, Polars and Arrow trials include conversion
  cost. Median conversion ranged from effectively zero for C to 0.07 s (Fortran),
  0.08–0.09 s (pandas), 0.18 s (Polars) and 0.10–0.11 s (Arrow). Fit medians
  ranged from 6.0 to 8.2 s. No default layout or public Arrow API change is
  justified by these contextual two-pair trials.
- **External references:** two sklearn RF fits had a 207.816 s median; two
  LightGBM RF fits had a 12.004 s median. These algorithms use different split
  methods and importance definitions, so they are descriptive comparisons only.

## Compatibility and validation

The cross-build audit passed 13 scenarios covering predictions, probabilities,
OOB, weights, NaNs, CSR, warm start, pruning, monotonic constraints, entropy,
permutation importance, exact training, multiclass and balanced subsampling.
Baseline pickles load in the candidate. Predictions, probabilities and OOB
outputs match exactly. Importance comparison uses an absolute 1e-15 tolerance
because repeated baseline fits themselves differed by up to about 2.8e-17.

Final checks: Python 174 passed, including seven benchmark-runner tests; Rust
workspace 58 passed. No existing tests were relaxed. `cargo fmt --all -- --check`
fails on formatting drift already present in the untouched baseline; the modified
split function was formatted, with no workspace-wide formatting applied.

## Reproduction and evidence

Run from the candidate worktree. All fit datasets are generated once per seed
outside git and shared by the workers. Raw JSON records include commands, seeds,
parameters, package versions, extension hashes and individual measurements.

```sh
uv sync --frozen --all-groups
.venv/bin/python benchmarks/run_fit_exploration.py --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/confirm-main-final --warmups 1 --repeats 5
.venv/bin/python benchmarks/run_fit_campaign.py controls --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/controls
.venv/bin/python benchmarks/run_fit_campaign.py quality --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/quality
.venv/bin/python benchmarks/run_fit_campaign.py comparators --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/comparators
uv pip install --python /path/to/baseline/.venv/bin/python polars==1.44.2 pyarrow==25.0.1
uv pip install --python /path/to/candidate/.venv/bin/python polars==1.44.2 pyarrow==25.0.1
.venv/bin/python benchmarks/run_fit_campaign.py layouts --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/layouts
.venv/bin/python benchmarks/check_fit_compatibility.py
.venv/bin/python benchmarks/evaluate_fit_campaign.py docs/performance/fit-exploration
```

See `README.md` for the detailed protocol, `DECISIONS.md` for deferred hypotheses,
and the per-scenario directories for raw evidence. No datasets, compiled
extensions or model binaries are versioned.
