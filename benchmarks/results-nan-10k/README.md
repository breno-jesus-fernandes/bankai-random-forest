# NaN support release benchmark

This is a finite-input regression check for the v1.5.0 NaN-routing changes.
Both versions were compiled with `maturin develop --release` and
`RUSTFLAGS="-C target-cpu=native"` on the same machine. The baseline is commit
`8be8ffe`; the current side includes the v1.5.0 work in progress.

The workload has 10,000 rows and 20 features with 90% zeros, 50 trees,
`max_features=4`, three seeds, and three timed repetitions per seed. Each seed
has one untimed fit and prediction warmup. Values below are medians across the
nine measured runs; raw CSVs are stored next to this report.

| Mode | Input | Fit baseline (s) | Fit current (s) | Fit change | Predict baseline (s) | Predict current (s) | Predict change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | Dense | 0.235322 | 0.234953 | -0.16% | 0.012367 | 0.014159 | +14.49% |
| Exact | CSR | 0.363878 | 0.362752 | -0.31% | 0.061248 | 0.064920 | +5.99% |
| Exact | CSC | 0.362133 | 0.382845 | +5.72% | 0.061082 | 0.067744 | +10.91% |
| Histogram (32 bins) | Dense | 0.154324 | 0.169467 | +9.81% | 0.024852 | 0.030599 | +23.13% |
| Histogram (32 bins) | CSR | 0.488648 | 0.416055 | -14.86% | 0.180450 | 0.191577 | +6.17% |
| Histogram (32 bins) | CSC | 0.493449 | 0.419849 | -14.92% | 0.184449 | 0.189188 | +2.57% |

This first comparison was the pre-optimization checkpoint. Profiling isolated
the prediction slowdown to a third iterator variant: every visited sample
entered an extra enum dispatch even when the iterator was only buffering
predictions. The fix removes that variant and buffers split routes only for
NaN-containing training input. Tree node counts and depths remain unchanged.

## Paired benchmark after traversal optimization

The rerun alternated baseline (`8be8ffe`) and current source order by workload;
both were rebuilt with `maturin develop --release` and
`RUSTFLAGS="-C target-cpu=native"`. Each seed had one untimed fit/predict
warmup followed by three timed repetitions. The workload is 10,000 × 20, 90%
zeros, 50 trees, `max_features=4`, and seeds 17, 29, and 43. The runner records
the sklearn-facing prediction time, direct native prediction time, and total
nodes/depth outside timed sections.

| Mode | Input | Fit change | Estimator predict change | Native predict change |
| --- | --- | ---: | ---: | ---: |
| Exact | Dense | -2.98% | -0.11% | -0.75% |
| Exact | CSR | -1.98% | -0.62% | +0.71% |
| Exact | CSC | +0.03% | -2.35% | +0.74% |
| Histogram (32 bins) | Dense | +1.97% | +0.86% | +1.54% |
| Histogram (32 bins) | CSR | +2.48% | -1.07% | -1.24% |
| Histogram (32 bins) | CSC | -5.14% | -0.07% | +0.59% |

All finite-input workloads meet the 5% regression ceiling. Node counts and
maximum depths match exactly for each seed/backend; the largest measured fit
regression is +2.48%, and the largest prediction regression is +0.86%.
`paired-optimized/` contains the twelve raw baseline/current CSVs. The
reproducible runner is `benchmarks/run_sparse_benchmark.py`; it rebuilds the
selected checkout in release mode and accepts `--project-root` for the
baseline worktree.
