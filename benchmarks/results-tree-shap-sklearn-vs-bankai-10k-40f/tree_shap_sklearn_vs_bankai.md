# sklearn vs Bankai direct TreeSHAP benchmark

Medians across seeds. Both estimators use `shap.TreeExplainer(model)`; Bankai's extension is built in release mode with `target-cpu=native`. Tree export is reported separately and included in explanation total.

| model | tree export s | explainer s | calculation s | total s | s per row | max additivity error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sklearn | 0.000000 | 0.010221 | 4.257673 | 4.273107 | 0.042731 | 9.090e-13 |
| bankai_exact | 0.086754 | 0.009724 | 4.511193 | 4.607624 | 0.046076 | 2.117e-12 |
| bankai_max_bins=16 | 0.109798 | 0.012029 | 4.648248 | 4.770075 | 0.047701 | 1.903e-14 |

Dataset: 10000 rows, 40 features, 100 trees; 100 rows explained per seed. SHAP 0.51.0, sklearn 1.9.1, Python 3.11.11.
Bankai histogram mode uses `max_bins=16`; sklearn has no matching histogram mode. Fit time is excluded from TreeSHAP timings.
