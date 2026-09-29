# 1M-row `fit` profile

## Original profile (before the optimizations)

The long delay before worker threads began processing in the baseline was
primarily caused by Python-side preparation and copying the entire matrix into
Rust. The Python-to-Rust dispatch itself happens only once and is not the
bottleneck: the profile spent **301.2 s** before the extension started, and the
matrix copy took another **62.5 s** inside the native call.

| Stage | 100k × 500 | 1M × 500 |
|---|---:|---:|
| sklearn validation / `float64` conversion | 0.060 s | 20.698 s |
| `_prepare_training_data` | 4.830 s | 280.504 s |
| `matrix_to_owned` (`Vec<f64>`) | 0.086 s | 62.481 s |
| Histogram cut construction | 0.202 s | 2.153 s |
| Cut application / discretized matrix | 0.083 s | 3.358 s |
| Total `DenseInput` preparation | 0.324 s | 22.552 s |
| Forest construction | 0.015 s | 1.310 s |
| Total native call | 0.425 s | 86.402 s |
| Total `fit` | **5.322 s** | **387.852 s** |
| Peak sampled RSS | 1.147 GiB | 3.782 GiB |

The internal `DenseInput` stages are included in the total shown on that row:
cut construction and application add up to 5.511 s for 1M rows; the remaining
time includes validation and setup. The table rows should not be summed.

## Measurement setup

- Machine: macOS 26.6.2, Apple Silicon ARM64, 8 logical CPUs, and 8 GiB RAM;
  Python 3.11.11.
- Reproducible synthetic input with seed 1729: `float32`, 500 features, 90
  relevant features, dense; 2 GB of data for 1M rows.
- `n_jobs=-1`, `max_bins=63`, `binning_strategy='sampled_select'`, 200,000 rows
  sampled for cut construction, and `max_samples=0.8`.
- One shallow tree (`n_estimators=1`, `max_depth=1`) and
  `importance_type='gain'` were used to isolate preparation, copying, and
  histogram processing. Dataset generation was outside the `fit` timer.
- Timings are from one run per size using a Rust `--release` build. RSS was
  sampled every 50 ms. A macOS process sample recorded a peak physical
  footprint of 8.8 GiB; memory pressure occurred, so the 1M-row timings are
  representative of this host under memory pressure, not a clean estimate for
  another machine.

This shallow-tree profile is not the notebook's full benchmark; it isolates
the initial delay. The notebook uses 40 trees, depth 20, and permutation
importance for Bankai. The `binning_strategy=BANKAI_BINNING_STRATEGY` argument
is now active. At 100k rows, `bin_sample_size=200_000` covers every row; at 1M
rows, it samples 200,000. Dataset generation is outside the `fit` timer.

## Implemented optimizations and new measurements

The prioritized work has been implemented in stages P0–P3. The timings below
use the same generator, seed, and shallow-tree configuration with a Rust
`--release` build. Generating `X` and `y` remains outside the timer.

| Size | Baseline | P0: sorting/copying in Rust | P1: preserve `float32` | P2: bin directly from matrix | P3: fused validation |
|---|---:|---:|---:|---:|---:|
| 100k × 500 (`float32`) | 5.322 s | 0.557 s | 0.517 s | 0.509 s | — |
| 1M × 500 (`float32`) | 387.852 s | 248.845 s | 41.571 s (median of 3) | 3.362 s (median of 3, final comparison) | **2.180 s** (median of 3; 177.9× vs. baseline) |

For the shallow 1M profile, P1 measured 41.571 s, 34.535 s, and 54.615 s. In
the final paired comparison, P2 measured 3.362 s (3.293–3.608 s) and P3
measured 2.180 s (1.972–2.388 s), with three fits per version. P3 reduced time
by 35.2% and was 1.54× faster than P2. An earlier P2 measurement had a median
of 3.306 s, consistent with this range. Baseline and P0 each had one run; their
gains remain diagnostic references, not statistical intervals. This host has
8 GiB RAM and experiences variable memory pressure.

The shallow 100k case in the table uses `float32`, matching the notebook. To
measure fused validation, I compared `float64` at 100k × 500 with nine fits per
version, across three processes per version in alternating order. The owned
matrix path (P1) had a median of 0.455 s (0.440–0.521 s); the P2 accessor with
a separate scan measured 0.508 s (0.487–0.546 s); and P3 measured 0.380 s
(0.375–0.411 s). P3 was 1.20× faster than P1 and 1.34× faster than P2. On this
dataset and host, fusion removed the accessor's `float64` disadvantage at
100k rows.

The shallow 1M profile uses one tree (`n_estimators=1`, `max_depth=1`) and
`importance_type='gain'` to isolate preparation. Sampled RSS was 3.782 GiB at
baseline, about 2.0 GiB in the first P0/P1 runs, and 1.983 GiB in the first P2
run; processes with multiple repetitions reached 2.6 GiB. macOS RSS is
sensitive to compression and paging, so there is no conclusive evidence of a
peak RSS reduction between P1 and P2, even though P2 removed the 4 GB
`Vec<f64>` allocation. P3-specific RSS was not measured.

The notebook's full fit was also measured with 40 trees, depth 20, 511 leaves,
`n_jobs=-1`, Bankai permutation importance, and LightGBM gain without
permutation importance:

| Data | Bankai P1 | Bankai P2 | LightGBM 4.7 RF | Repetitions |
|---|---:|---:|---:|---:|
| 100k × 500 | 7.278 s | 7.284 s | 12.822 s | 3 per configuration |
| 1M × 500 | — | **49.703 s** | 89.578 s | 3 per configuration |

For the 1M dataset, P2 ranged from 43.298 to 50.356 s, and LightGBM ranged
from 88.118 to 93.356 s. The medians indicate a 1.80× speedup for Bankai over
LightGBM, even though only Bankai computes permutation importance. This
measurement does not isolate tree construction alone, and the RF settings are
approximate rather than mathematically identical. At 100k, P1 and P2 tied on
full-fit time: preparation is small compared with training and permutation
importance.

### Local full fit at 1M: P2 vs. P3

I repeated the full Bankai fit at the requested size, keeping the notebook
parameters and changing only `N_SAMPLES` to 1,000,000: 500 features, 90
relevant features, 40 trees, depth 20, 511 leaves, `max_bins=63`,
`sampled_select` with 200,000 rows for cut construction, `max_samples=0.8`,
`n_jobs=-1`, and permutation importance. Only `fit` was timed; each process
recreated the same deterministic dataset (seed 1729) before the timer started.

| Version | Fit times (s) | Median | Range |
|---|---|---:|---:|
| P2, before fused validation | 42.701; 48.348; 49.143 | 48.348 s | 42.701–49.143 s |
| P3, fused validation | 43.476; 45.691; 48.611 | 45.691 s | 43.476–48.611 s |

The observed median decreased by **2.656 s (5.5%; 1.06×)**. The ranges overlap
substantially, and P3 was slower than P2 in one of the three runs. With this
few repetitions, the gain cannot be confidently distinguished from host
variation. The shallow profile isolates preparation better and showed about
1.18 s saved at 1M rows; in the full fit, 40 trees and permutation importance
dilute that effect. Therefore, 2.656 s is the observed difference between
medians, not a guaranteed saving for every run.

The highest sampled RSS was 1.982 GiB for the first full P2 fit at 1M rows and
3.794 GiB for the first LightGBM run; a later process alternating runs
recorded a 3.145 GiB total peak. macOS RSS varied between processes and does
not support a precise memory comparison.

### P0 — Canonical ordering and dense copying

The dense single-output path no longer builds `np.lexsort` keys and materializes
`X[order]` in Python. Rust computes a stable lexicographic order by refining
tie groups feature by feature, with NaNs after finite values, then fills the
final `Vec<f64>` directly in that order. The vector is contiguous, avoiding an
intermediate reordered NumPy matrix. The same index keeps labels, weights, and
OOB predictions aligned.

Canonical ordering was not removed: an attempt to skip it changed fixed-seed
bootstrap behavior and failed the equivalence check between integer sample
weights and duplicated rows, as well as the standard sklearn check. Multioutput
and sparse inputs still use the previous Python path; the P0 gain applies to
the dense single-output path.

### P1 — Preserve `float32` through the native boundary

`validate_data` now preserves `float32` and `float64` matrices in their input
format; other numeric types are still converted to `float64`. The PyO3
extractor accepts both formats and converts `float32` values to `f64` while
filling the native buffer, already in final order. Since every `float32` value
is represented exactly in `f64`, values are preserved and an intermediate
4 GB Python `float64` matrix is avoided for the 1M × 500 case. The training
core still stores values as `f64`.

A test was added comparing `float32` with the same values in `float64`,
including NaNs, histograms, OOB, predictions, and importances. The row-order
invariance test also covers ties, signed zeros, and NaNs.

### P2 — Discretize directly from the source matrix

When input is dense `float32` or `float64` and histograms are enabled, Rust
keeps a borrowed view of the NumPy matrix and accesses rows in canonical order.
Cut construction and bin application write directly to the compact `u8`
buffer; this path eliminates the full intermediate `Vec<f64>` allocation
(4 GB at 1M × 500). Sparse input and training without histograms retain the
previous path. A Rust test compares cuts and all bins from the accessor with
the owned-matrix path, including NaNs, sampled selection, and parallel
preprocessing; Python tests cover dtypes and strided `float32` matrices.

### P3 — Fuse validation with bin application

The accessor no longer makes a separate pass to find NaNs and reject
infinities. Each worker collects local flags while transforming values into
bins; after workers finish, the code combines those flags and preserves the
error for infinite values and NaN routing. Per-worker reduction avoids atomics
in the hot loop. Tests were added to preserve NaN detection and reject `inf` in
the direct path for `float32` and `float64` input. In the shallow 1M × 500
profile, this step reduced the median from 3.362 s to 2.180 s; at 100k × 500
`float64`, it reduced time from 0.508 s to 0.380 s versus the previous
accessor.

### Histogram controls included in the branch

The estimator exposes `binning_strategy` (`exact_sort`, `sampled_sort`,
`exact_select`, `sampled_select`) and `bin_sample_size` (200,000 by default).
Sampled strategies choose a deterministic subset per feature; select
strategies use multi-selection of ranks instead of sorting every observation.
Histogram preprocessing shares the thread budget with tree construction on
large matrices. These options change runtime and, depending on ties and
sampling, can change the cut points; compare model quality along with speed
when choosing a strategy.

## Code diagnosis before the changes

1. In the baseline, `validate_data(dtype=np.float64)` expanded the 2 GB
   `float32` matrix to 4 GB. The current path preserves `float32`/`float64` in
   [`estimator.py`](../python/bankai_random_forest/estimator.py#L193).
2. The baseline always built keys with `np.lexsort` and materialized
   `X[order]`. The dense single-output path now defers this in
   [`_prepare_training_data`](../python/bankai_random_forest/estimator.py#L781);
   the Python sorting code remains as a fallback for sparse/multioutput.
3. [`lib.rs`](../src/lib.rs#L564) determines canonical order. The baseline and
   fallback path in [`matrix_to_owned`](../src/lib.rs#L674) converts and copies
   rows into a contiguous buffer; in the baseline, this serial pass took
   62.5 s at 1M rows. The P2 histogram path accesses the view directly and
   avoids this copy.
4. [`dense.rs`](../src/dense.rs#L1049) builds cuts and applies bins. In the
   baseline `sampled_select` profile, these passes added up to 5.5 s at 1M,
   much less than preparation/copying. P3 also fuses NaN/inf validation with
   bin application and removes a full matrix scan. The Python-to-Rust bridge is
   called once; it does not by itself explain a delay of several minutes.

## Further measurements

The direct histogram path supports dense `float32` and `float64` input; P3 now
eliminates the extra validation scan on this path. The measured profiles do
not currently indicate a need for a dtype/size switch point, but this should
be revisited on Linux x86 and for small, sparse, or non-contiguous matrices.
Permutation-importance vectors were also observed to vary between multithreaded
runs with a fixed seed, while predictions and the global importance sum remain
stable. This aggregation variation should be investigated separately from
performance before relying on per-feature comparisons.

## Conclusion

The initial delay came from Python preparation and matrix transfer, not
repeated calls across the Python-to-Rust bridge. Rust-side ordering, preserving
`float32`, direct discretization, and fused validation reduced the shallow 1M
profile from 387.9 s to a median of 2.1 s on this host. For the full 1M fit,
three recent runs gave Bankai P3 a 45.7 s median, versus 48.3 s for P2 in the
paired round; this 2.7 s difference had substantial run-to-run variation. The
earlier LightGBM RF comparison measured 89.6 s, without permutation
importance. These results establish the performance level on this host, but
do not remove the difference in work or replace comparisons on production
machines.
