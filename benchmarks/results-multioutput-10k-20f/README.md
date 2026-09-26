# Multioutput release benchmark

Both comparisons used 10,000 rows × 20 features, 90% zeros, two classification
outputs, 50 trees, `max_features=4`, and seeds 17, 29, and 43. The Bankai
extension was built with `maturin develop --release` and
`RUSTFLAGS="-C target-cpu=native"`. Every seed/configuration had one untimed
fit-and-predict warmup and three measured repetitions.

## Multioutput vs sklearn

The Bankai implementation trains one native forest per output; sklearn's
`RandomForestClassifier` shares each tree structure across outputs. Timings
therefore compare the public output contract and give context, while tree
construction cost is not an apples-to-apples algorithm comparison. Bankai's
predictions closely matched sklearn on this synthetic workload:

| Mode | Bankai fit median (s) | Bankai predict median (s) | sklearn fit median (s) | sklearn predict median (s) | Min label agreement | Max probability RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | 0.5839 | 0.0292 | 0.2921 | 0.0239 | 99.995% | 0.0143 |
| Histogram (32 bins) | 0.3942 | 0.0528 | 0.2854 | 0.0238 | 99.61% | 0.0490 |

Raw per-seed/repetition measurements are in `multioutput.csv`.

## Single-output regression gate

The compatibility gate compares baseline `1844bf2` with the multioutput
implementation. The baseline/current runs were release-built with
`target-cpu=native`; their order alternated by workload. The runner records one
discarded fit/predict warmup per seed and three measured repetitions.

| Mode | Input | Fit change | Estimator predict change | Native predict change |
| --- | --- | ---: | ---: | ---: |
| Exact | Dense | -0.47% | +0.06% | +0.73% |
| Exact | CSR | -4.43% | -2.64% | -0.73% |
| Exact | CSC | +2.42% | -0.47% | -0.53% |
| Histogram (32 bins) | Dense | +0.03% | -0.93% | -0.25% |
| Histogram (32 bins) | CSR | +0.43% | -0.61% | -1.26% |
| Histogram (32 bins) | CSC | +2.04% | +0.92% | +2.33% |

The worst single-output regression is +2.42%, below the 5% ceiling. Raw CSVs
are in `singleoutput-regression/`; reproduce them with
`benchmarks/run_sparse_benchmark.py --project-root <checkout>`.
