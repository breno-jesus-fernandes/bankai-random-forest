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

- Preparation: created isolated worktrees and environments. Baseline tests and
  phase profile pending. No optimization accepted yet.

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
