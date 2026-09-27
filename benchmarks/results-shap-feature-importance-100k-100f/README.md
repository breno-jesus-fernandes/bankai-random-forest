# SHAP feature importance: Bankai vs LightGBM

Dataset: 100,000 training rows, 100 features, 100 trees. Global importance is mean absolute positive-class SHAP over a fixed random sample of 1,000 validation rows; both use the same 100-row training background. There were 3 measured seeds and 1 full warmup(s) per seed. The host reports 8 logical CPUs. Run time (excluding release builds): 24:18.

Environment: macOS 26.6.2, machine arm64, Python 3.11.11, NumPy 2.4.6, SHAP 0.51.0, sklearn 1.9.1, LightGBM 4.7.0. Bankai extension built in release mode with `RUSTFLAGS=-C target-cpu=native`. Both model fits use `n_jobs=-1`.

Both models use exact interventional TreeSHAP and the same background. Bankai uses SHAP on positive-class probability with the recorded training background. SHAP 0.51's LightGBM RF path does not reproduce this model's `predict_proba` (its probability contributions sum to 0/1); LightGBM therefore uses the additive raw-margin TreeSHAP path. Additivity is checked against Bankai positive-class probability and LightGBM raw margin, respectively. Their SHAP magnitudes have different units, so compare normalized per-model importance shares and feature rankings, not absolute values. Bankai tree-export time is reported separately and included in fit+SHAP total.

| model | fit s | tree export s | explainer s | SHAP s | fit + SHAP s | max additivity error | SHAP rank corr vs other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| histogram_16 | 6.680 | 0.656 | 0.066 | 228.111 | 235.559 | 5.047e-09 | 0.295481 |
| random_forest_boosting | 1.705 | 0.000 | 0.038 | 1.641 | 3.404 | 3.792e-06 | 0.295481 |

Raw seed-level measurements are in `shap_feature_importance_raw.csv`; median summary is `shap_feature_importance.csv`; per-feature mean absolute values are in `shap_values_mean_abs.csv`.

Spearman correlation uses average ranks to account for tied zero-importance features: 0.295. The same three signal features (0, 1, 2) rank highest for both; together they hold 77.4% of Bankai importance share and 100.0% of LightGBM importance share.
