# sklearn vs Bankai direct TreeSHAP benchmark

Medians across seeds. Both estimators use `shap.TreeExplainer(model)`; Bankai's extension is built in release mode with `target-cpu=native`. Tree export is reported separately and included in explanation total.

| model | tree export s | explainer s | calculation s | total s | s per row | max additivity error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sklearn | 0.000000 | 0.002211 | 0.237235 | 0.239446 | 0.002394 | 4.052e-15 |
| bankai_exact | 0.010259 | 0.002343 | 0.259111 | 0.271712 | 0.002717 | 4.152e-14 |
| bankai_max_bins=16 | 0.011717 | 0.002558 | 0.287681 | 0.301513 | 0.003015 | 1.721e-15 |

Dataset: 1000 rows, 20 features, 100 trees; 100 rows explained per seed. SHAP 0.51.0, sklearn 1.9.1, Python 3.11.11.
Bankai histogram mode uses `max_bins=16`; sklearn has no matching histogram mode. Fit time is excluded from TreeSHAP timings.
