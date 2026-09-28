# Histogram preprocessing: base vs. parallel branch

Measured on the same Mac arm64 host (8 logical CPUs), Python 3.11.11, NumPy 2.4.6, scikit-learn 1.9.1. Both Rust extensions were built with the repository's release profile. Input is the deterministic `make_classification` workload: 100,000 rows, 100 dense features, binary target, seed 42, and 16 bins. Fits use gain importance, all features, and seeds 100 onward. Each reported run had one warmup and five timed fits; the benchmark alternated `n_jobs=1` and `n_jobs=-1` order across repetitions.

The base (`4c7b833`) was measured before and after the feature branch to reduce temporal/thermal bias. The feature branch had two runs. `fit_summary.csv` records each run's median and observed range. The comparison below uses the median of the run medians, giving equal weight to the two runs per revision.

| Trees | `n_jobs` | Base median (s) | Feature median (s) | Change |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1.3350 | 1.3571 | +1.65% slower |
| 1 | -1 | 1.3315 | 1.1751 | -11.75% faster |
| 16 | 1 | 4.0843 | 4.0924 | +0.20% slower |
| 16 | -1 | 1.8066 | 1.6895 | -6.48% faster |

## Interpretation

- No material end-to-end regression appeared in the serial path: the one-tree result is +1.65%, while the 16-tree result is +0.20%. The observed ranges overlap between base and feature runs, so these small changes are within run-to-run variation.
- With `n_jobs=-1`, the one-tree fit improved 11.75%. Since only one tree is available, this workload emphasizes preprocessing and its thread-startup cost. It is still a full `fit`, not a direct timer around the preprocessing function.
- The 16-tree multithreaded fit improved 6.48%. This includes both parallel preprocessing and forest construction; the measurement cannot attribute the entire gain to histogram preprocessing alone.
- This is one host and one bin resolution. It establishes no broad scaling claim. The output is fit time only; model metrics were not part of this comparison.

Reproduce each measurement with:

```sh
uv run python benchmarks/run_histogram_preprocessing_benchmark.py \
  --rows 100000 --features 100 --bins 16 --trees 1 16 \
  --repeats 5 --warmups 1 --output /tmp/fit.csv
```

## Small-workload crossover

I also measured one-tree fits with 20 features and 16 or 255 bins. At 1,000 rows (20,000 cells), `n_jobs=-1` did not improve fit time over `n_jobs=1`; the difference was smaller than the observed run ranges. At 10,000 rows (200,000 cells), all-core fits were 10.4% faster with 16 bins and 9.7% faster with 255 bins. Serial results stayed within a few percent across versions.

Based on that crossover, the implementation now keeps preprocessing serial below 32,768 dense matrix cells and enables worker threads above it. This threshold is a conservative initial cutoff: tested points are 20,000 and 200,000 cells, so future tuning should measure the interval between them and include more feature counts. Small-load measurements, including before and after the cutoff, are in `small_workloads.csv`. They are full-fit timings; one tree makes preprocessing a larger share but does not isolate its timer.
