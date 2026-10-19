# Experimental exact columns for Boruta

Alpha 16 adds Rust integration types `ExactColumnBank`, `ExactRowIndices` and
`ExactVirtualColumns`, plus `DenseInput::training_exact_virtual`. The Python
classifier API and its default storage are unchanged. Bankai Boruta a7 uses this
storage only with the private `BANKAI_BORUTA_EXACT_SESSION=1` experiment switch.

A bank owns one column-major snapshot of the original f32/f64 values. Original
precision remains available for canonical row sorting. Training reads cast to
f32 at access, matching alpha 15's materialized exact matrix. An augmented
column resolves a feature map, canonical row order and, for a shadow, an explicit
row permutation. Trees share immutable banks/maps through Arc. The binary split
search fills the existing reusable per-worker f32 buffer; routing and multiclass
read through the same map. Threshold sampling, bootstrap, gains and per-tree
normalization are unchanged. Binning code is unchanged.

Banks validate shape, finite values and finite f32 conversion once. Round maps
validate dimensions, ranges and bijections before training. u32 permutations are
supported through exactly 2^32 rows where addressable; larger addressable shapes
use usize. Checked shape multiplication rejects overflow. No enormous allocation
is required to test overflow rejection.

This saves the full real-plus-shadow f32 matrix, but retains the input snapshot,
row order and explicit permutations. Boruta's standard MT19937/u32 experiment shuffles directly into the session-owned
explicit permutation buffer with identical RandomState state transitions.
Custom-RNG/int64 fallbacks copy NumPy permutations into a reusable owned Rust
buffer each round. Include fallback live copies and cumulative copy volume in
memory interpretation.
`FitProfile` measures the visited binary feature buffer capacity per worker;
it does not measure total forest, masks, allocator overhead or total process RSS.
Use external RSS and Linux VmHWM for acceptance.

`cargo test --workspace --locked` checks the exact tree multiset (splits, scores,
leaf votes and covers) against materialization for binary/multiclass,
gini/entropy and bootstrap on/off. Worker completion order can reorder trees;
comparison preserves every tree while ignoring chunk arrival order, as required
by the existing parallel implementation. Boruta tests additionally compare
importances at atol=rtol=1e-12 and require identical decisions and rounds.

Activation requires lower median runtime AND peak RSS on controlled workloads
and complete Industrial fits on AMD with 1/4/16 workers, seeds 42–46, repeated
sequentially. This validation is pending; alpha 16 does not activate this storage
for ordinary classifier fits. See the Boruta and benchmark reports for host-specific
measurements; local macOS results cannot establish AMD performance.

## Release preparation

Publish alpha 16 before Boruta a7. Merging a reviewed PR into `master` publishes
its version to PyPI when that version is not on PyPI yet, then creates the
GitHub release from the same artifacts. PRs build artifacts without uploading.
A manual dispatch can retry a failed publication:

```sh
gh workflow run ci.yml --ref master -f publish=true
```

Opening a PR creates no tag or PyPI release.
