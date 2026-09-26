# Histogram and permutation importance benchmark

Times are medians across model seeds; external permutation importance uses one repeat per feature. Bankai computes native OOB permutation importance during fit, so its fit time already includes the importance calculation; `feature_importance_seconds` measures only public attribute access. Scikit-learn and LightGBM calculate permutation importance externally on the same validation set. LightGBM uses `boosting_type='rf'`, 80% row bagging, all features per tree, and one thread. The comparable workload is `fit_plus_importance_seconds`.

| implementation | mode | bins | fit s (Bankai includes OOB permutation) | importance access/calculation s | fit + importance s | rank corr vs Bankai exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bankai | exact |  | 1.414400 | 0.000000 | 1.414400 | 1.000000 |
| bankai | histogram_16 | 16 | 0.644929 | 0.000001 | 0.644930 | 0.581955 |
| bankai | histogram_32 | 32 | 0.690244 | 0.000001 | 0.690244 | 0.351880 |
| bankai | histogram_64 | 64 | 0.816119 | 0.000001 | 0.816121 | 0.583459 |
| bankai | histogram_128 | 128 | 1.072805 | 0.000001 | 1.072806 | 0.703759 |
| bankai | histogram_255 | 255 | 1.616156 | 0.000001 | 1.616157 | 0.687218 |
| sklearn | random_forest |  | 1.815978 | 1.474596 | 3.290822 | 0.545865 |
| lightgbm | random_forest_boosting |  | 0.199891 | 0.638870 | 0.843389 | 0.407519 |
