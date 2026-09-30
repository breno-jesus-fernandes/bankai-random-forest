# Decisions and bounded conclusions

## H1 — specialize binary histogram split state

Measured target: tree building (95.5% of instrumented fit). Replace two tiny heap
accumulators with fixed-size binary accumulators and expose the class count to
the compiler. The arbitrary-class path remains dynamic. Arithmetic order,
threshold selection, missing-value routing, tie-breaking, RNG calls, tree storage,
public constructor/defaults and serialization schema are unchanged.

Main confirmation passed; control/quality gates are pending. Only this production
change is under consideration. Instrumentation is removed from production sources;
its patch is retained solely for reproducible profiling.

## H2 — validation, canonical sorting and input layout

Measured canonical sorting: 0.0073 s; native conversion: 0.00014 s;
edge construction plus bin application: 0.3671 s. These together are under 5% of
fit in the phase profile. Even eliminating preprocessing entirely would not reach
the required overall improvement on that run. No sort/conversion rewrite is
justified for the default path by these measurements. Layout and Arrow/Polars
conversion experiments remain contextual and will include reorganization cost.
A larger native Arrow API is explicitly out of scope.

## H3 — work distribution and histogram storage

The implementation already distributes trees across threads and uses histogram
subtraction for the larger child. Best-first candidates retain histogram caches,
which explains a plausible source of memory pressure (code inspection, not a
measured allocation attribution). Thread controls will quantify scaling. No
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
