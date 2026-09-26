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

The 5% regression gate is not met. In particular, finite dense histogram
prediction and dense exact prediction need more work before v1.5.0 can close.
These measurements are an implementation checkpoint, not a completed
performance result.
