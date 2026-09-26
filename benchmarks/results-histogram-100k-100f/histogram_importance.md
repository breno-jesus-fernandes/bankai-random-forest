# Histogram and permutation importance benchmark

Workload: 100,000 training rows, 100,000 validation rows, 100 features, and 100 trees; three measured seeds and one untimed warmup per seed. Benchmark execution took 53m20s, excluding release builds.

Environment: macOS 25.6.0 on Apple M1 (arm64); Python 3.11.11; NumPy 2.4.6; scikit-learn 1.9.1; LightGBM 4.7.0. The Python extension and Rust CLI were built in release mode with `RUSTFLAGS=-C target-cpu=native`.

Times are medians across model seeds after one or more untimed warmups for every implementation, mode, and seed; the measured model is recreated with the same seed. External permutation importance uses one repeat per feature. Bankai computes native OOB permutation importance during fit, so its fit time already includes the importance calculation; `feature_importance_seconds` measures only public attribute access. Scikit-learn and LightGBM calculate permutation importance externally on the same validation set. LightGBM uses `boosting_type='rf'`, 80% row bagging, all features per tree, and one thread. The comparable workload is `fit_plus_importance_seconds`.

| implementation | mode | bins | fit s (Bankai includes OOB permutation) | importance access/calculation s | fit + importance s | rank corr vs Bankai exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bankai | exact |  | 85.656981 | 0.000001 | 85.656982 | 1.000000 |
| bankai | histogram_16 | 16 | 25.225633 | 0.000001 | 25.225634 | 0.118752 |
| bankai | histogram_32 | 32 | 27.687618 | 0.000001 | 27.687619 | 0.040024 |
| bankai | histogram_64 | 64 | 34.328323 | 0.000001 | 34.328324 | 0.219838 |
| bankai | histogram_128 | 128 | 46.363346 | 0.000001 | 46.363347 | 0.051521 |
| bankai | histogram_255 | 255 | 72.938896 | 0.000001 | 72.938898 | 0.006985 |
| sklearn | random_forest |  | 95.859760 | 110.658817 | 206.606595 | 0.279832 |
| lightgbm | random_forest_boosting |  | 4.269871 | 30.061076 | 34.255443 | 0.020750 |
