#!/usr/bin/env python3
"""Benchmark experimental Bankai TreeSHAP routes and permutation baseline."""

import argparse
import csv
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import shap

from bankai_random_forest import BankaiRandomForestClassifier


def dataset(seed, rows=10_000, features=20):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(rows, features))
    y = np.digitize(x[:, :3].sum(axis=1) + rng.normal(scale=0.2, size=rows), [-0.5, 0.5])
    return x, y


def elapsed(callable_):
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def peak_rss_kib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / 1024 if platform.system() == "Darwin" else value


def run_route(name, model, explain, background):
    export = create = calculate = 0.0
    if name == "direct":
        model._shap_estimators_cache = None
        _, export = elapsed(lambda: model.estimators_)
        explainer, create = elapsed(lambda: shap.TreeExplainer(model))
        values, calculate = elapsed(lambda: np.asarray(explainer.shap_values(explain)))
        base = np.asarray(explainer.expected_value)
    elif name == "adapter":
        adapter, export = elapsed(model._tree_shap_adapter_for_benchmark)
        explainer, create = elapsed(lambda: shap.TreeExplainer(adapter))
        values, calculate = elapsed(lambda: np.asarray(explainer.shap_values(explain)))
        base = np.asarray(explainer.expected_value)
    elif name == "native":
        (values, base), calculate = elapsed(lambda: model._native_tree_shap_for_benchmark(explain))
    else:
        explainer, create = elapsed(
            lambda: shap.Explainer(model.predict_proba, background, algorithm="permutation", seed=42)
        )
        values, calculate = elapsed(lambda: np.asarray(explainer(explain, max_evals=41).values))
        base = None
    additive_error = ""
    if base is not None:
        additive_error = float(np.max(np.abs(values.sum(axis=1) + base - model.predict_proba(explain))))
    return {"route": name, "export_seconds": export, "explainer_seconds": create,
            "calculate_seconds": calculate, "total_seconds": export + create + calculate,
            "latency_seconds_per_row": (export + create + calculate) / len(explain),
            "peak_rss_kib": peak_rss_kib(),
            "additivity_max_error": additive_error}


def median_rows(rows):
    keys = ["export_seconds", "explainer_seconds", "calculate_seconds", "total_seconds", "latency_seconds_per_row", "peak_rss_kib"]
    output = []
    for mode in ("exact", "max_bins=16"):
        for route in ("direct", "adapter", "native", "permutation"):
            matches = [row for row in rows if row["route"] == route and row["mode"] == mode]
            if not matches:
                continue
            item = {"mode": mode, "route": route, "runs": len(matches)}
            item.update({key: statistics.median(row[key] for row in matches) for key in keys})
            errors = [row["additivity_max_error"] for row in matches if row["additivity_max_error"] != ""]
            item["additivity_max_error"] = max(errors) if errors else "not comparable"
            output.append(item)
    return output


def write_report(rows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (directory / "tree_shap_benchmark.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "tree_shap_benchmark.md").open("w") as file:
        file.write("# Experimental TreeSHAP benchmark\n\n")
        file.write("Medians of three seeds. Permutation is model-agnostic and is not value-equivalent to tree_path_dependent.\n\n")
        file.write("| mode | route | export/adapt s | explainer s | calculate s | total s | s/row | peak RSS KiB | additivity max |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            file.write("| {mode} | {route} | {export_seconds:.6f} | {explainer_seconds:.6f} | {calculate_seconds:.6f} | {total_seconds:.6f} | {latency_seconds_per_row:.6f} | {peak_rss_kib:.0f} | {additivity_max_error} |\n".format(**row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results-tree-shap-10k"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--skip-permutation", action="store_true")
    args = parser.parse_args()
    rows = []
    for max_bins in (None, 16):
        for seed in args.seeds:
            x, y = dataset(seed, args.rows, args.features)
            model = BankaiRandomForestClassifier(n_estimators=args.trees, random_state=seed, max_bins=max_bins).fit(x, y)
            for route in ("direct", "adapter", "native") + (() if args.skip_permutation else ("permutation",)):
                row = run_route(route, model, x[:100], x[100:200])
                row.update({"seed": seed, "mode": "exact" if max_bins is None else "max_bins=16", "python": platform.python_version()})
                rows.append(row)
    write_report(median_rows(rows), args.output_dir)


if __name__ == "__main__":
    main()
