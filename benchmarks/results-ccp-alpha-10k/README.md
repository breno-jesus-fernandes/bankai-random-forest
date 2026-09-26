# Cost-complexity pruning benchmark

Compared release builds of the baseline commit (`47dfe1a`) with the current
implementation on the same host. Each dataset had 10,000 rows, 20 features,
and a 90:10 class balance; forests had 100 trees and three fixed seeds. Each
seed/configuration had one discarded fit-and-predict warmup followed by three
measured runs. Raw rows and medians are in `raw.csv` and `summary.csv`.

With `ccp_alpha=0`, the exact path changed by -2.77% in fit, -1.32% in
prediction, and -0.41% in peak RSS. The 32-bin histogram path changed by
+0.08% in fit, -0.42% in prediction, and +0.09% in peak RSS. Regressions stay
below the 5% gate. At `ccp_alpha=0.05`, trees are pruned during fit and
prediction becomes faster; this feature cost is reported separately.

The runner builds the extension in release mode with `target-cpu=native` and
discards one warmup for every fixed seed/configuration:

```sh
uv run python benchmarks/run_ccp_alpha_benchmark.py \
  --mode exact --ccp-alpha 0.05 \
  --output benchmarks/results-ccp-alpha-10k/pruned-exact.csv
uv run python benchmarks/run_ccp_alpha_benchmark.py \
  --mode hist_32 --ccp-alpha 0.05 \
  --output benchmarks/results-ccp-alpha-10k/pruned-hist-32.csv
```
