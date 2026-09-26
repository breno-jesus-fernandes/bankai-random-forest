# Sparse input benchmark

The release-built Bankai extension was measured on 10,000 rows × 20 features
with 90% zeros, 50 trees, `max_features=4`, and seeds 17, 29, and 43. Each
seed includes one untimed fit and prediction warmup followed by three measured
repetitions. The baseline is commit `74dddfc`, rebuilt cleanly with
`RUSTFLAGS="-C target-cpu=native"`; the new build uses the same release flags.

`summary.csv` reports median fit and prediction times, plus the maximum process
RSS. The six `release-*.csv` files contain the measured CSR, CSC, and dense
runs; `baseline-*.csv` contain baseline dense timings.

Dense-path change from baseline:

| Backend | Fit | Prediction |
| --- | ---: | ---: |
| Exact | +1.09% | -2.65% |
| Histogram (32 bins) | +2.34% | -10.63% |

Median fit / prediction seconds and peak RSS by format:

| Backend | Format | Fit | Prediction | Peak RSS (KiB) |
| --- | --- | ---: | ---: | ---: |
| Exact | Dense | 0.231117 | 0.012275 | 177,792 |
| Exact | CSR | 0.355013 | 0.059579 | 173,824 |
| Exact | CSC | 0.355058 | 0.059568 | 173,424 |
| Histogram (32 bins) | Dense | 0.149653 | 0.024212 | 181,760 |
| Histogram (32 bins) | CSR | 0.479296 | 0.178385 | 179,904 |
| Histogram (32 bins) | CSC | 0.483698 | 0.178430 | 180,640 |

CSR and CSC are normalized to CSR without converting the matrix to a dense
matrix. Sparse workloads use less peak RSS here, with slower training and
prediction than dense input. Timings are environment-specific and are intended
for comparison with this recorded baseline.
