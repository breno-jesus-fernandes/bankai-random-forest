# SHAP feature importance: Bankai vs LightGBM

Dataset: 100,000 training rows, 100 features, 100 trees. Global importance is mean absolute positive-class SHAP over a fixed random sample of 1,000 validation rows; both use the same 100-row training background. There were 3 measured seeds and 1 full warmup(s) per seed. The host reports 8 logical CPUs. Run time (excluding release builds): 1:44.

Environment: macOS 26.6.2, machine arm64, Python 3.11.11, NumPy 2.4.6, SHAP 0.51.0, sklearn 1.9.1, LightGBM 4.7.0. Bankai extension built in release mode with `RUSTFLAGS=-C target-cpu=native`. Both model fits use `n_jobs=-1`.

Both models use exact interventional TreeSHAP and the same background. Bankai uses max_leaf_nodes=31; LightGBM uses num_leaves=31. Bankai uses SHAP on positive-class probability with the recorded training background. A probability-scale LightGBM explanation was attempted first; it is retained only when additivity error is <=1e-4. In this environment SHAP 0.51's LightGBM RF path fails that check, so its timing and values use the explicitly labeled raw-margin fallback. The two importance scales are not directly comparable; no cross-model rank correlation should be interpreted. Validation accuracy and F1 are reported for the matched-capacity models. Bankai tree-export time is reported separately and included in fit+SHAP total.

| model | fit s | tree export s | explainer s | SHAP s | fit + SHAP s | mean nodes/tree | accuracy | F1 | max additivity error | SHAP rank corr vs other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| histogram_16 | 1.645 | 0.003 | 0.002 | 1.642 | 3.291 | 61.0 | 0.79144 | 0.75923 | 6.625e-09 | not comparable: different output scales |
| random_forest_boosting | 1.653 | 0.000 | 0.037 | 1.622 | 5.144 | 61.0 | 0.94085 | 0.94070 | 3.792e-06 | not comparable: different output scales |

Raw seed-level measurements are in `shap_feature_importance_raw.csv`; median summary is `shap_feature_importance.csv`; per-feature mean absolute values are in `shap_values_mean_abs.csv`.

The capped Bankai model's median validation accuracy/F1 changed by -0.14869/-0.18133 versus the same-seed, unbounded Bankai reference; prediction agreement was 0.81959. The reference quality fits also had one warmup per seed. See raw CSV for all seeds, tree topology, and failed LightGBM probability additivity values.
