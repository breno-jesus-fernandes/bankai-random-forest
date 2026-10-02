"""Check global SHAP importance against the full native SHAP value tensor.

Run with: uv run maturin develop --release --skip-install --features shap-benchmark-legacy &&
          uv run python benchmarks/run_shap_importance_quality_benchmark.py
"""
import json

import numpy as np
from scipy import sparse

from bankai_random_forest import BankaiRandomForestClassifier


def rank_order(values):
    return np.argsort(np.argsort(values, kind="stable"), kind="stable")


def summarize(actual, reference):
    actual_order = rank_order(actual)
    reference_order = rank_order(reference)
    top_k = min(5, len(actual))
    actual_top = set(np.argsort(actual)[-top_k:])
    reference_top = set(np.argsort(reference)[-top_k:])
    return {
        "max_abs_error": float(np.max(np.abs(actual - reference))),
        "mean_abs_error": float(np.mean(np.abs(actual - reference))),
        "rank_order_correlation": float(np.corrcoef(actual_order, reference_order)[0, 1]),
        "top_k_overlap": len(actual_top & reference_top) / top_k,
    }


def main():
    results = []
    for classes in (2, 3):
        rng = np.random.RandomState(130 + classes)
        x = rng.normal(size=(600, 12))
        x[::29, 4] = np.nan
        if classes == 2:
            y = (x[:, 0] - 0.6 * x[:, 1] + 0.15 * x[:, 2] > 0).astype(int)
        else:
            signal = x[:, 0] - 0.6 * x[:, 1]
            y = np.digitize(signal, [-0.5, 0.5])
        for max_bins in (None, 16):
            for use_sparse in (False, True):
                model_x = sparse.csr_matrix(x) if use_sparse else x
                explained = model_x[:120]
                model = BankaiRandomForestClassifier(
                    n_estimators=40,
                    max_features=None,
                    max_depth=10,
                    max_bins=max_bins,
                    random_state=44,
                    n_jobs=1,
                ).fit(model_x, y)
                shap_values, _ = model._forest.tree_shap(explained)
                shap_values = np.asarray(shap_values, dtype=np.float64)
                reference = np.abs(shap_values).mean(axis=(0, 2))
                reference /= reference.sum()
                unaccelerated = np.asarray(
                    model._forest.shap_importances_unaccelerated_for_benchmark(explained),
                    dtype=np.float64,
                )

                serial = model.shap_importances(explained)
                model.n_jobs = -1
                parallel = model.shap_importances(explained)
                results.append({
                    "classes": classes,
                    "max_bins": max_bins,
                    "input": "csr" if use_sparse else "dense",
                    "accelerated_vs_unaccelerated": summarize(serial, unaccelerated),
                    "reduced_vs_full_tensor": summarize(serial, reference),
                    "parallel_vs_unaccelerated": summarize(parallel, unaccelerated),
                    "parallel_vs_serial": summarize(parallel, serial),
                })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
