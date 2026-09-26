# Histogram and permutation importance benchmark

Times are median Bankai results across seeds; sklearn uses one fit and one external permutation pass. Bankai computes native OOB permutation importance during fit, so its fit time already includes the importance calculation; `feature_importance_seconds` measures only public attribute access. The comparable workload is `fit_plus_importance_seconds`.

| mode | bins | fit s (includes Bankai OOB permutation) | importance access/calculation s | fit + importance s | rank corr vs exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact |  | 1.488400 | 0.000000 | 1.488400 | 1.000000 |
| histogram_16 | 16 | 0.626534 | 0.000001 | 0.626535 | 0.581955 |
| histogram_32 | 32 | 0.752621 | 0.000001 | 0.752622 | 0.351880 |
| histogram_64 | 64 | 1.041664 | 0.000001 | 1.041664 | 0.583459 |
| histogram_128 | 128 | 1.493642 | 0.000000 | 1.493642 | 0.703759 |
| histogram_255 | 255 | 2.532817 | 0.000001 | 2.532818 | 0.687218 |
| external_permutation |  | 1.752414 | 1.414104 | 3.166518 | 0.505263 |
