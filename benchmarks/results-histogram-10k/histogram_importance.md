# Histogram and permutation importance benchmark

Times are median Bankai results across seeds; sklearn uses one fit and one external permutation pass. Bankai computes native OOB permutation importance during fit, so `feature_importance_seconds` measures the public attribute access separately and `fit_seconds` includes its computation.

| mode | bins | fit s (includes Bankai OOB permutation) | predict s | importance access/calculation s | predict + importance s | end to end s | F1 | rank corr vs exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact |  | 1.505678 | 0.049920 | 0.000001 | 0.049920 | 1.555598 | 0.960691 | 1.000000 |
| histogram_16 | 16 | 0.618576 | 0.053739 | 0.000001 | 0.053740 | 0.672316 | 0.951306 | 0.581955 |
| histogram_32 | 32 | 0.750470 | 0.052519 | 0.000001 | 0.052520 | 0.802990 | 0.952977 | 0.351880 |
| histogram_64 | 64 | 1.021732 | 0.051609 | 0.000001 | 0.051610 | 1.073341 | 0.955365 | 0.583459 |
| histogram_128 | 128 | 1.549723 | 0.051745 | 0.000000 | 0.051746 | 1.601468 | 0.957775 | 0.703759 |
| histogram_255 | 255 | 2.633414 | 0.051644 | 0.000001 | 0.051644 | 2.685059 | 0.957562 | 0.687218 |
| external_permutation |  | 1.741320 | 0.064894 | 1.403950 | 1.468844 | 3.210164 | 0.963747 | 0.505263 |
