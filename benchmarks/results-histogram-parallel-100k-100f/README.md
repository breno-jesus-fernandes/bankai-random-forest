# Parallel Bankai histogram and LightGBM benchmark

Workload: 100,000 training rows and validation rows, 100 features, 100 trees, 3 measured seeds, and 1 warmup(s) per seed. The machine reports 8 logical CPUs. Execution took 1:57, excluding release builds.

Environment: darwin, Python 3.11.11, NumPy 2.4.6, scikit-learn 1.9.1, LightGBM lightgbm 4.7.0. The Bankai extension was built in release mode with `RUSTFLAGS=-C target-cpu=native`.

Both estimators fit with `n_jobs=-1`. Bankai computes OOB permutation importance during fit. LightGBM fits multithreaded, then permutation importance evaluates features in parallel (`n_jobs=-1`) while each prediction uses one thread to avoid nested oversubscription. LightGBM uses RF boosting, 80% row bagging, all features per tree, and one repeat per feature for external permutation importance.

| implementation | mode | fit s | importance s | fit + importance s | rank correlation vs Bankai |
| --- | --- | ---: | ---: | ---: | ---: |
| bankai | histogram_16 | 8.585 | 0.000 | 8.585 | 1.000000 |
| lightgbm | random_forest_boosting | 1.722 | 7.802 | 9.518 | 0.159352 |

On this machine, Bankai histogram_16's fit time was 2.94x lower than the earlier single-threaded run (25.23 s). LightGBM's combined fit and importance time was 3.60x lower than its earlier single-threaded run (34.26 s); the importance phase now evaluates features in parallel, so this is a whole-workload comparison rather than a fit-only speedup.

Raw per-implementation measurements are in `histogram_parallel.csv`.
