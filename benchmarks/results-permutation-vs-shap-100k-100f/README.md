# Bankai permutation vs LightGBM SHAP feature importance

Workload: 100,000 training rows and 100,000 validation rows, 100 features, 100 trees, Bankai histogram_16; 3 measured seeds plus 1 full warmup(s) per seed. Both model fits use all 8 reported logical CPU cores. The fixed SHAP sample has 1,000 validation rows and the background has 100 training rows. Execution took 1:17, excluding release builds.

Environment: macOS 26.6.2 on arm64, Python 3.11.11, NumPy 2.4.6, SHAP 0.51.0, scikit-learn 1.9.1, LightGBM 4.7.0. Bankai extension built in release mode with `RUSTFLAGS=-C target-cpu=native`.

Bankai's `importance_type='permutation'` is native out-of-bag accuracy decrease computed during fit; property access is timed separately. LightGBM uses RF boosting (80% row bagging, all features) and direct interventional TreeSHAP; its importance is mean absolute positive-class SHAP on raw margin because SHAP 0.51's probability path for this RF model fails additivity against `predict_proba`. Additivity is validated against `predict(raw_score=True)`. These methods and units differ, so compare runtime and use rank correlation as a descriptive agreement measure only; raw importance magnitudes are not comparable. Bankai's permutation work is included in fit; the comparable total is `fit_plus_importance_seconds`.

| implementation | fit s | importance setup s | importance calculation/access s | fit + importance s | validation accuracy | validation F1 | max additivity error | rank corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| histogram_16 | 8.038 | 0.000 | 0.000 | 8.038 | 0.94013 | 0.94056 | 0.000e+00 | 0.295399 |
| random_forest_boosting_tree_shap_raw_margin | 1.644 | 0.041 | 1.627 | 3.337 | 0.94085 | 0.94070 | 3.792e-06 | 0.295399 |

The three synthetic signal features (indices 0, 1, and 2) occupy the top three positions under both methods. Overall Spearman correlation is 0.295; this is descriptive because Bankai OOB permutation uses accuracy decrease, while LightGBM SHAP uses mean absolute raw-margin contributions. Bankai fit plus permutation was 2.41x the LightGBM fit plus SHAP time in this run.

Raw seed measurements are in `permutation_vs_shap_raw.csv`; summary medians are in `permutation_vs_shap.csv`; per-feature importances and average ranks are in `feature_importance.csv`.
