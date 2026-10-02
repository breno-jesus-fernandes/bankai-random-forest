"""Compare reduced native SHAP importance with the tensor-returning endpoint.

Run with: uv run maturin develop --release --skip-install --features shap-benchmark-legacy &&
          uv run python benchmarks/run_shap_importance_reduction_benchmark.py
"""
import json
import statistics
import time

import numpy as np

from bankai_random_forest import BankaiRandomForestClassifier


def measure(call, repeats=5):
    call()  # warmup
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = call()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), value


def main():
    rng = np.random.RandomState(2026)
    x = rng.normal(size=(1000, 20))
    y = (x[:, 0] - x[:, 1] + 0.2 * x[:, 2] > 0).astype(int)
    results = []
    for max_bins in (None, 32):
        model = BankaiRandomForestClassifier(
            n_estimators=100, max_features=None, max_depth=12, n_jobs=1,
            max_bins=max_bins, random_state=17,
        ).fit(x, y)
        explained = x[:100]
        legacy_time, legacy = measure(
            lambda: model._forest.shap_importances_unaccelerated_for_benchmark(explained)
        )
        reduced_time, reduced = measure(lambda: model.shap_importances(explained))
        tensor_time, tensor = measure(
            lambda: model._native_tree_shap_for_benchmark(explained)[0]
        )
        reference = np.abs(tensor).mean(axis=(0, 2))
        reference /= reference.sum()
        model.n_jobs = -1
        parallel_workers = model._resolve_n_jobs()
        parallel_time, parallel = measure(lambda: model.shap_importances(explained))
        model.n_jobs = 1
        tree_arrays = model._forest.shap_tree_arrays()
        max_depth = 0
        for left, right, feature, *_ in tree_arrays:
            stack = [(0, 0)]
            while stack:
                node, depth = stack.pop()
                if feature[node] >= 0:
                    max_depth = max(max_depth, depth + 1)
                    stack.extend(((left[node], depth + 1), (right[node], depth + 1)))
        workspace_bytes_per_worker = 4 * (max_depth + 2) * (min(max_depth, x.shape[1]) + 2) * 8
        results.append({
            "max_bins": max_bins,
            "rows": 1000,
            "features": 20,
            "trees": 100,
            "explained_rows": 100,
            "parallel_workers": parallel_workers,
            "unaccelerated_median_seconds": legacy_time,
            "reduced_median_seconds": reduced_time,
            "tensor_median_seconds": tensor_time,
            "all_cores_reduced_median_seconds": parallel_time,
            "reduction_vs_tensor_current_speedup": tensor_time / reduced_time,
            "speedup_over_unaccelerated_serial": legacy_time / reduced_time,
            "speedup_over_unaccelerated_parallel": legacy_time / parallel_time,
            "max_abs_error_vs_unaccelerated": float(np.max(np.abs(reduced - legacy))),
            "max_abs_error": float(np.max(np.abs(reduced - reference))),
            "parallel_max_abs_error": float(np.max(np.abs(parallel - reduced))),
            "tensor_payload_bytes": int(len(explained) * x.shape[1] * model.n_classes_ * 8),
            "workspace_bytes_per_worker_estimate": int(workspace_bytes_per_worker),
        })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
