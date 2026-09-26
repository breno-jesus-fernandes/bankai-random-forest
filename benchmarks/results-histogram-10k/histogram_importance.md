# Histogram and permutation importance benchmark

Times are medians across model seeds; external permutation importance uses one repeat per feature. Bankai computes native OOB permutation importance during fit, so its fit time already includes the importance calculation; `feature_importance_seconds` measures only public attribute access. Scikit-learn and LightGBM calculate permutation importance externally on the same validation set. LightGBM uses `boosting_type='rf'`, 80% row bagging, all features per tree, and one thread. The comparable workload is `fit_plus_importance_seconds`.

| implementation | mode | bins | fit s (Bankai includes OOB permutation) | importance access/calculation s | fit + importance s | rank corr vs Bankai exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bankai | exact |  | 1.562304 | 0.000001 | 1.562305 | 1.000000 |
| bankai | histogram_16 | 16 | 0.628252 | 0.000002 | 0.628254 | 0.581955 |
| bankai | histogram_32 | 32 | 0.757848 | 0.000001 | 0.757849 | 0.351880 |
| bankai | histogram_64 | 64 | 1.039536 | 0.000001 | 1.039536 | 0.583459 |
| bankai | histogram_128 | 128 | 1.633072 | 0.000002 | 1.633074 | 0.703759 |
| bankai | histogram_255 | 255 | 2.883380 | 0.000002 | 2.883381 | 0.687218 |
| sklearn | random_forest |  | 1.945818 | 1.497165 | 3.442983 | 0.545865 |
| lightgbm | random_forest_boosting |  | 0.170452 | 0.623542 | 0.841442 | 0.407519 |
