# balanced_subsample benchmark

Compared release builds of the baseline commit (`938c2da`) and the v1.1
implementation on the same host. Data had 10,000 rows, 20 features, and a
90:10 class balance; each forest had 100 trees and used three fixed seeds.
Each seed and backend configuration had one full fit/predict warmup that was discarded,
followed by three measured fits and predictions. The CSV records those nine
measurements per scenario. Peak RSS is the process high-water mark.

The unchanged exact path changed by +1.72% in median fit time and +2.63% in
median prediction time. The unchanged 32-bin histogram path changed by +3.43%
in median fit time and +2.78% in median prediction time. Peak RSS changed by
-0.60% for exact and +0.03% for histograms. All regressions stayed below the
5% gate. `balanced_subsample` medians were 1.102 s fit / 0.037 s prediction
for exact and 0.313 s / 0.041 s for 32-bin histograms; this feature cost is
reported separately from the unchanged-path regression gate.

The candidate benchmark can be repeated with:

```sh
uv run python benchmarks/run_balanced_subsample_benchmark.py \
  --mode exact --balanced-subsample \
  --output benchmarks/results-balanced-subsample-10k/reproduced-exact.csv
uv run python benchmarks/run_balanced_subsample_benchmark.py \
  --mode hist_32 --balanced-subsample \
  --output benchmarks/results-balanced-subsample-10k/reproduced-hist-32.csv
```

The runner builds the extension with Maturin in release mode and
`target-cpu=native` before measuring it.
