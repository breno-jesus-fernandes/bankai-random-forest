# Decisions and bounded conclusions

## H1 — specialize binary histogram split state

Measured target: tree building (95.5% of instrumented fit). Replace two tiny heap
accumulators with fixed-size binary accumulators and expose the class count to
the compiler. The arbitrary-class path remains dynamic. Arithmetic order,
threshold selection, missing-value routing, tie-breaking, RNG calls, tree storage,
public constructor/defaults and serialization schema are unchanged.

Accepted. Final-build main confirmation improved median fit by 17.06%, with a
paired speedup ratio 95% interval of [1.185, 1.329]. The maximum sampled fit RSS
increase was 0.84%; median fit RSS increased by 4.10%. All controls passed their
5% median regression and 10% RSS limits. Nine paired quality runs produced identical
predictions and zero deltas in all four required metrics. Python and Rust suites
and cross-build compatibility checks passed. Instrumentation is removed from
production sources; its patch is retained solely for reproducible profiling.

## H2 — validation, canonical sorting and input layout

Measured canonical sorting: 0.0073 s; native conversion: 0.00014 s;
edge construction plus bin application: 0.3671 s. These together are under 5% of
fit in the phase profile. Even eliminating preprocessing entirely would not reach
the required overall improvement on that run. No sort/conversion rewrite is
justified for the default path by these measurements. Layout and Arrow/Polars
conversion experiments remain contextual and include reorganization cost. Layout
trials measured conversion medians around 0.072 s for Fortran, 0.084–0.090 s for
pandas, 0.180–0.183 s for Polars and 0.101–0.106 s for Arrow, compared with 6–8 s
fits. These are contextual two-pair trials, not grounds to change defaults. A
larger native Arrow API is explicitly out of scope.

## H3 — work distribution and histogram storage

The implementation already distributes trees across threads and uses histogram
subtraction for the larger child. Best-first candidates retain histogram caches,
which explains a plausible source of memory pressure (code inspection, not a
measured allocation attribution). Thread controls passed median fit and peak RSS
gates at `n_jobs=1` and `n_jobs=-1`. The `n_jobs=-1` paired interval is wide and
crosses parity despite a faster median, so no speedup claim is made there. No
scheduler or cache-eviction rewrite is accepted without an isolated experiment.

## Backlog (priority order)

1. Profile allocations inside best-first candidate queues and evaluate bounded
   histogram cache retention, including recomputation cost and arbitrary weights.
2. Measure tree-worker scheduling on heterogeneous CPUs with enough paired runs;
   retain explicit n_jobs behavior and avoid architecture-specific defaults.
3. If workloads with few trees make preprocessing ≥10%, revisit column/block
   layouts and edge-buffer reuse, including the complete reorganization cost.
4. Investigate deterministic importance reduction separately: baseline same-seed
   differences are ~1e−17, although predictions/probabilities/OOB repeat exactly
   in this audit. Do not silently broaden this fit optimization to change them.
5. Evaluate native Arrow only if measured conversion costs justify its API and
   lifetime/ownership complexity. Architecture-specific SIMD is deferred.

All evidence is bounded to this Apple M1/macOS machine and ≤100,000 rows.
No Linux x86_64 or million-row extrapolation is supported.
