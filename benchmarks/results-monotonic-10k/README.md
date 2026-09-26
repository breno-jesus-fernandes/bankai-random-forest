# Monotonic constraints benchmark

Compared release builds of baseline commit `d2d4991` with the v1.3
implementation on the same Apple Silicon host (Python 3.11.11, scikit-learn
1.9.1). Each run used 10,000 rows, 20 features, a 90:10 class balance, 100
trees, `max_features=4`, and fixed seeds 17, 29, and 43. Each seed and scenario
had one discarded fit-and-predict warmup and three measured repetitions. The
extension was built with `RUSTFLAGS="-C target-cpu=native"` and
`maturin develop --release`.

| Backend | Scenario | Median fit | Median predict | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| Exact | Baseline v1.2 | 1.3226 s | 0.03902 s | 183,344 KiB |
| Exact | v1.3, no constraints | 1.3224 s | 0.03919 s | 183,728 KiB |
| Exact | v1.3, increasing feature 0 | 1.3234 s | 0.03941 s | 184,528 KiB |
| Histogram, 32 bins | Baseline v1.2 | 0.3198 s | 0.04077 s | 186,288 KiB |
| Histogram, 32 bins | v1.3, no constraints | 0.3208 s | 0.04056 s | 187,696 KiB |
| Histogram, 32 bins | v1.3, increasing feature 0 | 0.3426 s | 0.04113 s | 186,992 KiB |

With constraints unset, the exact backend changed by -0.02% in fit, +0.45% in
prediction, and +0.21% in peak RSS. The histogram backend changed by +0.31%,
-0.52%, and +0.76%, respectively. These remain below the 5% regression gate.
The enabled constraint's own cost is listed separately: for this workload it
changed fit time by +0.07% exact and +6.79% histogram versus the v1.3 unconstrained
scenario. Raw measurements and medians are in `raw.csv` and `summary.csv`.

Reproduce the current implementation after building the release extension:

```sh
RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release
uv run python benchmarks/run_monotonic_benchmark.py \
  --mode exact --scenario after \
  --output benchmarks/results-monotonic-10k/current-exact.csv
uv run python benchmarks/run_monotonic_benchmark.py \
  --mode hist_32 --scenario monotonic \
  --output benchmarks/results-monotonic-10k/current-monotonic-hist.csv
```

The `baseline` scenario was run from the v1.2.0 release wheel built at
`d2d4991`; it omits the new constructor parameter when fitting.
