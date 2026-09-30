# Fit exploration: datasets ≤100,000 rows

Baseline: `76ef49844d90c8f9e43871561ba76235219d5e49` (`origin/master`).
Started 2026-09-30 03:28 UTC; maximum campaign deadline 11:28 UTC.
Branch: `feature/fit-evidence-driven-optimization`. Separate baseline and candidate
worktrees, virtual environments and Cargo target directories. The original
modified notebook is outside both worktrees and is preserved.

## Protocol

Portable release builds (`uv sync --frozen --all-groups`, default release profile;
no target-cpu=native). Dependencies from the same uv.lock. Every worker records
actual extension SHA256, commit, Python/package versions and hardware. Datasets
are generated once per seed, float32, saved outside git, then shared by both
workers. Fit excludes generation, estimator construction and scoring. Processes
run sequentially with alternating baseline/candidate order. RSS is sampled during
fit; process high-water RSS is also reported separately (includes loading).

Main: 100,000 total rows, 80,000 train, 20,000 validation, 500 features,
90 informative; 40 trees, depth 20, 511 leaves, min leaf 5, all features,
80% bootstrap, 63 exact-sort bins, gain importance. Main threads 8; controls 1/-1.
Screen: two measurements. Confirmation: one warmup and ≥5 paired measurements.
Acceptance: ≥5% median fit improvement and paired 95% ratio CI above 1;
no mandatory control slowdown >5%; fit RSS increase ≤10%; suites pass.
Quality: three dataset × three model seeds, all four metrics mean delta ≥−0.002
and one-sided 95% lower CI >−0.002. Unconfirmed changes stay out of defaults.

Comparators sklearn RF (impurity importance) and LightGBM RF (gain) are contextual,
not acceptance baselines. No million-row workloads, architecture-specific SIMD,
public Arrow API, merge or release in this campaign.

## Journal

- Preparation: created isolated worktrees and environments from the recorded
  baseline. The modified notebook in the original worktree remains untouched.

- Initial validation: Python 169/169, Rust 58/58, executor 2/2 passed.
  Full outputs are stored alongside this journal. Both environments have identical
  package names/versions. Hardware: Apple M1, 8 GiB RAM, macOS arm64.
- Tooling note: `maturin develop` attempted `uv pip install --group`, unsupported by
  the installed uv. Rebuilds use `uv sync --frozen --all-groups
  --reinstall-package bankai-random-forest` instead, preserving locked dependencies.
- Profiling: `profile.patch` is temporary wall-clock instrumentation, not a proposed
  production change. Measurements from this build cannot pass acceptance gates.

### H1: binary split accumulators and constant class count

Hypothesis: tree construction dominates (>90% observed); per-feature split
accumulators allocate two tiny vectors and use runtime class counts in hot loops.
Specialize the binary path with two stack arrays and a constant class count;
retain dynamic multiclass handling and exact arithmetic order. Success requires
all common gates, including main ≥5%, not merely a faster microbenchmark.
Status: screening pending. No API/default/serialization changes proposed.

Initial main baseline: 8.717 / 11.389 s; median 10.053 s, sampled peak RSS median
1,203,609,600 bytes. Variation is substantial: final paired confirmation is essential.

Profile (instrumented; fit 8.867 s): canonical labels/order 0.00730 s,
native conversion 0.000136 s, bin edges 0.27477 s, applying bins 0.09234 s,
trees 8.47056 s (95.5%). Remaining Python validation, importance and bookkeeping
are collectively ~0.021 s; the profile does not resolve these tiny components
individually. Reorganizing preprocessing alone cannot plausibly meet 5% overall
on this observed profile, so prioritization moves to tree split loops.

H1 initial suites: Python 171 passed, Rust 58 passed, no tests relaxed.

H1 screening: median baseline 8.501 s, candidate 7.378 s (13.2% reduction),
paired speedup 1.152. Screening memory sampling used a Python thread which can
be blocked by the native GIL: its fit RSS is **not sufficient for acceptance**.
Before confirmation, replaced it with a separate sampling process and capture
OS high-water RSS immediately after fit (before scoring). Final memory evidence
uses only this corrected executor. This is a measurement correction, not a model
change. H1 remains pending confirmation.

H1 main confirmation (`confirm-main-final/`): one warmup + five alternating
pairs, uninstrumented release built from the final source. Median baseline
7.030409 s, candidate 5.830724 s; 17.06% reduction in medians. Paired median
speedup 1.22453, percentile-bootstrap 95% CI [1.18537, 1.32915], paired median
saving 1.25675 s. Median fit RSS 1,989,459,968 → 2,070,953,984 bytes (+4.10%);
maximum observed fit RSS 2,187,476,992 → 2,205,810,688 bytes (+0.84%). Both
memory summaries meet +10%. The earlier `confirm-main/` result is retained for
the build-hash correction history. The final binary SHA256 is
`fbc024628cafd04e6fb0b68b9e74de67bc403fc44972a011fabe6a8ca2557f31`.

H1 control matrix (`controls/`): five pairs each for `n_jobs=1`, `n_jobs=-1`,
float64, permutation importance, multiclass, sparse, NaN and few-estimator fits;
15 pairs for exact training after the initial result was near the 5% boundary.
Every control's median candidate fit was no more than 5% slower, and every peak
RSS gate passed. Exact mode's paired interval spans parity, so the result supports
absence of a material regression rather than a speedup claim.

H1 quality confirmation (`quality/`): 3 dataset seeds × 3 model seeds. Predictions
were byte-identical; accuracy, precision, recall and F1 deltas were all 0.000 in
all nine pairs. The mean and one-sided 95% lower bound for each metric are 0.000.
All numerical gates pass (`acceptance.json`).

Compatibility audit: 13 scenarios passed cross-build comparison (standard, OOB,
weights, NaNs, CSR, warm start, pruning, monotonic constraints, entropy,
permutation importance, exact training, multiclass, balanced subsample).
Predictions/probabilities/OOB require exact array equality; baseline pickles also
load in candidate. Baseline itself showed importance differences of 2.78e−17
between same-seed fits, due to reduction order; the supplemental audit records
absolute differences and uses only for importance an absolute 1e−15 tolerance.
No existing repository test was changed or relaxed. Repeated fits and pickle
round trips are checked in both environments. Raw audit JSON is included.

Context comparisons (`comparators/`) use two fits per library on the shared
100k × 500 dataset: sklearn RF median 207.816 s and LightGBM RF median 12.004 s.
These use different split algorithms and importance definitions, so they are
descriptive only. Their inputs, parameters, versions and raw timings are recorded.

Input-layout trials (`layouts/`) include one warmup and two paired measurements.
Median source-to-array conversion costs were approximately 0.072 s for Fortran,
0.084 s for pandas, 0.180 s for Polars and 0.106 s for Arrow; median fit time
remained 6.0–8.2 s. C-contiguous input needed effectively no reorganization.
Polars 1.44.2 and PyArrow 25.0.1 were installed at the same versions into both
isolated environments for these trials only; see `layout-dependencies.txt`.
The measurements do not justify a public Arrow path or changing the default
layout.

Final validation: Python 174 passed; Rust workspace 58 passed. The focused
split specialization is rustfmt-formatted. `cargo fmt --all -- --check` still
reports pre-existing formatting drift across the workspace; the same check fails
on the untouched baseline. No workspace-wide reformat was applied.

Reproduction (from candidate worktree; substitute worktree paths):

```sh
uv sync --frozen --all-groups
.venv/bin/python benchmarks/run_fit_exploration.py --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/confirm-main --warmups 1 --repeats 5
.venv/bin/python benchmarks/run_fit_campaign.py controls --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/controls
.venv/bin/python benchmarks/run_fit_campaign.py quality --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/quality
.venv/bin/python benchmarks/run_fit_campaign.py comparators --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/comparators
.venv/bin/python benchmarks/run_fit_campaign.py layouts --baseline /path/to/baseline --candidate /path/to/candidate --output docs/performance/fit-exploration/layouts
/path/to/baseline/.venv/bin/python benchmarks/check_fit_compatibility.py --write
.venv/bin/python benchmarks/check_fit_compatibility.py
.venv/bin/python benchmarks/evaluate_fit_campaign.py docs/performance/fit-exploration
```

Raw records include exact subprocess commands. Warmups use fresh worker processes,
as do measured fits; this warms machine/file caches without reusing an estimator.
The 95% timing CI resamples paired speedup ratios (20,000 draws, fixed seed 1729).
The sklearn/LightGBM contextual comparison uses two measured runs per library;
the candidate-vs-baseline acceptance comparison retains the full five-pair
confirmation protocol.
