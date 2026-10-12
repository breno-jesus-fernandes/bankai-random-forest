# Bankai Extra Trees histogram comparison

Five measured seeds per workload, paired on the same generated dense `float32` data and train/test split. One warmup per model and workload was discarded. The target is generated from three continuous features plus noise. All models use Gini, bootstrap with 80% of training rows, depth 20, up to 511 leaves, and one thread. Bankai histogram mode uses 63 bins; Bankai exact mode uses `max_bins=None`. The speed ratio is fit time divided by scikit-learn fit time, so values below 1 are faster.

| Rows × features | Trees | Variant | Median fit (s) | Fit / sklearn | Accuracy | F1 | ROC AUC |
|---:|---:|---|---:|---:|---:|---:|---:|
| 10,000 × 20 | 100 | Bankai exact | 0.738 | 1.074 | 0.8096 | 0.8067 | 0.8915 |
| 10,000 × 20 | 100 | Bankai histogram | 0.447 | 0.651 | 0.8068 | 0.8042 | 0.8918 |
| 10,000 × 20 | 100 | scikit-learn | 0.687 | 1.000 | 0.8080 | 0.8062 | 0.8938 |
| 100,000 × 20 | 100 | Bankai exact | 6.997 | 1.465 | 0.8118 | 0.8131 | 0.8928 |
| 100,000 × 20 | 100 | Bankai histogram | 1.395 | 0.292 | 0.8102 | 0.8117 | 0.8941 |
| 100,000 × 20 | 100 | scikit-learn | 4.777 | 1.000 | 0.8128 | 0.8139 | 0.8994 |
| 10,000 × 500 | 40 | Bankai exact | 8.379 | 1.340 | 0.8028 | 0.8054 | 0.8861 |
| 10,000 × 500 | 40 | Bankai histogram | 3.948 | 0.631 | 0.7992 | 0.8024 | 0.8823 |
| 10,000 × 500 | 40 | scikit-learn | 6.255 | 1.000 | 0.7976 | 0.8019 | 0.8849 |
| 100,000 × 100 | 100 | Bankai exact | 38.748 | 1.658 | 0.8128 | 0.8114 | 0.8973 |
| 100,000 × 100 | 100 | Bankai histogram | 5.296 | 0.227 | 0.8114 | 0.8106 | 0.8957 |
| 100,000 × 100 | 100 | scikit-learn | 23.375 | 1.000 | 0.8119 | 0.8104 | 0.9004 |

For these generated data and settings, Bankai with Extra Trees and 63-bin histograms was faster in all four cases, by 1.5× to 4.4×. Its median accuracy and F1 were close to the other models; ROC AUC was lower by up to 0.0053. This is a workload-specific measurement, and the histogram strategy trades exact thresholds for fit time.

The Rust extension was built in release mode with `RUSTFLAGS='-C target-cpu=native'`. Runtime versions and raw measurements are in [environment.txt](environment.txt), [runs.csv](runs.csv), and [medians.csv](medians.csv). Re-run with:

```sh
RUSTFLAGS='-C target-cpu=native' maturin develop --release
python benchmarks/run_extra_trees_sklearn_matrix.py --seeds 5 --jobs 1
```
