# Bankai Extra Trees vs LightGBM Extra Trees

Rows=8000 (6000 train / 2000 test), features=200, informative=20, signal_strength=0.12, trees=100, max_leaves/num_leaves=31, bins=63, threads=4.

Paired generated binary datasets; each variant gets the same data and model seed. One warmup per variant and seed is discarded. Fit time includes estimator input processing and histogram construction, but excludes dataset generation and prediction.

| Variant | Median fit (s) | Recall@K | False discoveries@K | Feature AP | Test ROC AUC | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| bankai | 0.2264 | 0.950 | 1.0 | 0.976 | 0.598 | 0.510 |
| bankai_extra_trees | 0.1687 | 0.900 | 2.0 | 0.955 | 0.597 | 0.499 |
| lightgbm_extra_trees | 0.3617 | 0.950 | 1.0 | 0.998 | 0.599 | 0.495 |

## Paired comparisons

- `bankai_vs_bankai_extra_trees`: paired median fit-time ratio first/second=1.350 (<1 means first model faster); median Δ recall@K (first-second)=+0.000; median Δ false discoveries (first-second)=+0.0.
- `bankai_vs_lightgbm_extra_trees`: paired median fit-time ratio first/second=0.628 (<1 means first model faster); median Δ recall@K (first-second)=+0.000; median Δ false discoveries (first-second)=+0.0.
- `bankai_extra_trees_vs_lightgbm_extra_trees`: paired median fit-time ratio first/second=0.477 (<1 means first model faster); median Δ recall@K (first-second)=+0.000; median Δ false discoveries (first-second)=+0.0.

## Fairness and limits

All variants use the same fixed train/test split, data/model seeds, tree count, leaf cap, minimum leaf count, 63-bin limit, 80% row fraction, 80% feature fraction, and gain importance. LightGBM RF requires active bagging and feature fraction below 1.0. Its bagging samples without replacement, while Bankai bootstraps with replacement; Bankai samples features per node while LightGBM feature_fraction samples per tree. LightGBM optimizes gradient statistics, while Bankai uses Gini impurity. Thus this is a matched-budget comparison, not identical algorithms.

Feature recall uses the known first K informative columns; false discoveries count selected noise columns among the top K. The target is a noisy logistic function of weak, signed effects on those columns. Medians summarize the paired seed matrix; no significance claim is made. Results are local to this machine and should not be generalized.

Environment: Python 3.11.11, macOS-26.6.2-arm64-arm-64bit, NumPy 2.4.6, scikit-learn 1.9.1, LightGBM 4.7.0.
