# Credit default: local benchmark with six jobs

Campaign: `benchmarks/campaign-extra-trees-20261012T205008024716Z`.
500,000 rows (400,000 train / 100,000 test), 500 features (24 informative),
100 trees, `n_jobs=6`, `bootstrap=True`, `max_samples=0.8`, one warmup per model
and three measured repetitions. Models share datasets, splits and nominal seeds.
Bankai Random Forest uses `max_bins=63`; Bankai Extra Trees uses `None`.

The scenario was selected from historical ranking AP gains over Bankai RF:
+0.0925 for credit default versus +0.0878 for fraud. Historical CSVs contain
metrics, not input matrices; datasets were regenerated from recorded code and seeds.
Historical observations are not measurements of this build.

| Model | Median fit (s) | Median ranking AP | Median informative features in top 24 |
| --- | ---: | ---: | ---: |
| Bankai RF, bins=63 control | 28.55 | 0.8390 | 19 |
| Bankai Extra Trees | 22.22 | 0.9232 | 22 |
| sklearn Extra Trees | 21.15 | 0.9341 | 22 |

Bankai Extra Trees was 6.33 seconds (22.2%) faster than the control by the
difference of medians, with AP higher by 0.0842 and three additional informative
features recovered. All three paired AP differences against the control were positive.

Against sklearn, the difference of median fit times was +1.07 seconds.
Paired ranking AP differences were +0.060670, +0.011005 and -0.021347
(median +0.011005); the difference of AP medians was -0.010873.
These are different summaries. Three repetitions are screening evidence,
not statistical proof of noninferiority.

AP and recall@24 assess informative feature selection, not classification recall.
Effective parameters, implementation hashes, environment and seeds are recorded
in the manifest and checkpoints. Individual observations are in `raw.csv`,
medians in `medians.csv`, and paired control comparisons in
`extra-vs-control-paired.csv`.
