# Extra Trees benchmark results

## Setup

- 10,000 rows, 500 float32 features (90 informative), stratified 80/20 split; data seed 1789.
- 40 trees; max depth 20; at most 511 leaves; minimum 5 rows per leaf; bootstrap 80%; all features per split; model seed 42; 8 logical CPUs.
- One warmup and 10 timed runs per model. Times below are median fit seconds. Quality is measured on the shared validation set.
- Software: Python 3.11.11, Bankai 0.1.0a11 (local build), scikit-learn 1.9.1, LightGBM 4.7.0, NumPy 2.4.6.
- The sklearn `ExtraTreesClassifier` uses the same bootstrap and `max_samples=0.8` settings as Bankai, to keep the sampling protocol aligned.

## Exact splits, including the scikit-learn comparisons

| Model | Fit (s) | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Exact Bankai Extra Trees — split | 11.024 | 0.5525 | 0.4918 | 0.5849 |
| Exact Bankai — split | 13.045 | 0.5615 | 0.5070 | 0.5878 |
| scikit-learn Extra Trees — split count | 1.472 | 0.5440 | 0.5323 | 0.5566 |
| scikit-learn RF — split count | 14.959 | 0.5555 | 0.5535 | 0.5786 |

Bankai exact Extra Trees fit in 11.024 s, versus 1.472 s for scikit-learn Extra Trees (7.49× slower). It scored +0.85 percentage points in accuracy and +2.83 points in ROC-AUC, while F1 was 4.05 points lower. Against scikit-learn Random Forest, Bankai exact Extra Trees fit 26.3% faster and had nearly the same accuracy (-0.30 points), with F1 6.17 points lower and ROC-AUC 0.64 points higher.

Within Bankai exact mode, enabling Extra Trees reduced median fit time from 13.045 s to 11.024 s (15.5% faster). Accuracy changed by -0.90 points, F1 by -1.53 points, and ROC-AUC by -0.29 points.

## Histogram mode

| Model | Fit (s) | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Bankai histograms (63 bins), Extra Trees off — gain | 1.849 | 0.5540 | 0.5110 | 0.5770 |
| Bankai histograms Extra Trees on (63 bins) — gain | 1.636 | 0.5605 | 0.5070 | 0.5745 |
| LightGBM RF (63 bins) — gain | 4.342 | 0.5700 | 0.5657 | 0.5936 |

Within the 63-bin Bankai histogram runs, Extra Trees reduced fit time from 1.849 s to 1.636 s (11.5% faster). Accuracy rose by 0.65 points; F1 fell by 0.40 points; ROC-AUC fell by 0.25 points. The histogram models were much faster than exact Bankai and LightGBM was included as the reference in this section.

Ten repeats share the same fixed model seed, so quality scores repeat exactly; the repeats characterize runtime variation, not variation across random seeds. Full per-run measurements and all medians are in the CSV files beside this report.
