#!/usr/bin/env python3
"""Compare direct SHAP TreeExplainer timing for sklearn and Bankai forests."""

import argparse
import csv
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier

from bankai_random_forest import BankaiRandomForestClassifier


def dataset(seed, rows, features):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(rows, features))
    y = np.digitize(
        x[:, :3].sum(axis=1) + rng.normal(scale=0.2, size=rows), [-0.5, 0.5]
    )
    return x, y


def elapsed(call):
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def release_build(root):
    environment = os.environ.copy()
    native_flag = "-C target-cpu=native"
    environment["RUSTFLAGS"] = " ".join(
        filter(None, [environment.get("RUSTFLAGS"), native_flag])
    )
    subprocess.run(
        ["uv", "run", "maturin", "develop", "--release"],
        check=True,
        cwd=root,
        env=environment,
    )


def time_explainer(name, model, explain, seed, rows, features, trees, mode):
    if name == "bankai":
        model._shap_estimators_cache = None
        _, export_seconds = elapsed(lambda: model.estimators_)
    else:
        export_seconds = 0.0

    explainer, explainer_seconds = elapsed(lambda: shap.TreeExplainer(model))
    values, calculate_seconds = elapsed(
        lambda: np.asarray(explainer.shap_values(explain))
    )
    base_values = np.asarray(explainer.expected_value)
    probabilities = model.predict_proba(explain)
    additivity_error = float(
        np.max(np.abs(values.sum(axis=1) + base_values - probabilities))
    )
    total = export_seconds + explainer_seconds + calculate_seconds
    return {
        "seed": seed,
        "rows": rows,
        "features": features,
        "trees": trees,
        "mode": mode,
        "implementation": name,
        "tree_export_seconds": export_seconds,
        "explainer_seconds": explainer_seconds,
        "calculate_seconds": calculate_seconds,
        "explanation_total_seconds": total,
        "seconds_per_explained_row": total / len(explain),
        "additivity_max_error": additivity_error,
        "shap_version": shap.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }


def median_rows(rows):
    fields = (
        "tree_export_seconds",
        "explainer_seconds",
        "calculate_seconds",
        "explanation_total_seconds",
        "seconds_per_explained_row",
    )
    summary = []
    for mode in ("sklearn", "bankai_exact", "bankai_max_bins=16"):
        matches = [row for row in rows if row["mode"] == mode]
        if not matches:
            continue
        result = {
            "mode": mode,
            "implementation": matches[0]["implementation"],
            "runs": len(matches),
        }
        result.update(
            {field: statistics.median(row[field] for row in matches) for field in fields}
        )
        result["additivity_max_error"] = max(
            row["additivity_max_error"] for row in matches
        )
        summary.append(result)
    return summary


def write_reports(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_fields = list(rows[0])
    summary = median_rows(rows)
    summary_fields = list(summary[0])
    with (output_dir / "tree_shap_sklearn_vs_bankai_raw.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "tree_shap_sklearn_vs_bankai.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "tree_shap_sklearn_vs_bankai.md").open("w") as handle:
        handle.write("# sklearn vs Bankai direct TreeSHAP benchmark\n\n")
        handle.write(
            "Medians across seeds. Both estimators use `shap.TreeExplainer(model)`; "
            "Bankai's extension is built in release mode with `target-cpu=native`. "
            "Tree export is reported separately and included in explanation total.\n\n"
        )
        handle.write(
            "| model | tree export s | explainer s | calculation s | total s | "
            "s per row | max additivity error |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in summary:
            handle.write(
                "| {mode} | {tree_export_seconds:.6f} | {explainer_seconds:.6f} | "
                "{calculate_seconds:.6f} | {explanation_total_seconds:.6f} | "
                "{seconds_per_explained_row:.6f} | {additivity_max_error:.3e} |\n".format(
                    **row
                )
            )
        versions = rows[0]
        handle.write(
            "\nDataset: {rows} rows, {features} features, {trees} trees; "
            "100 rows explained per seed. SHAP {shap_version}, sklearn "
            "{sklearn_version}, Python {python_version}.\n".format(**versions)
        )
        handle.write(
            "Bankai histogram mode uses `max_bins=16`; sklearn has no matching "
            "histogram mode. Fit time is excluded from TreeSHAP timings.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--explain-rows", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results-tree-shap-sklearn-vs-bankai-1k"),
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.rows < args.explain_rows:
        parser.error("--rows must be at least --explain-rows")

    root = Path(__file__).parents[1]
    if not args.skip_build:
        release_build(root)

    rows = []
    for seed in args.seeds:
        x, y = dataset(seed, args.rows, args.features)
        explain = x[: args.explain_rows]
        sklearn_model, sklearn_fit_seconds = elapsed(
            lambda: RandomForestClassifier(
                n_estimators=args.trees,
                max_features="sqrt",
                n_jobs=1,
                random_state=seed,
            ).fit(x, y)
        )
        row = time_explainer(
            "sklearn", sklearn_model, explain, seed, args.rows, args.features,
            args.trees, "sklearn"
        )
        row["fit_seconds"] = sklearn_fit_seconds
        rows.append(row)

        for mode, max_bins in (("bankai_exact", None), ("bankai_max_bins=16", 16)):
            bankai_model, fit_seconds = elapsed(
                lambda: BankaiRandomForestClassifier(
                    n_estimators=args.trees,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=seed,
                    max_bins=max_bins,
                ).fit(x, y)
            )
            row = time_explainer(
                "bankai", bankai_model, explain, seed, args.rows, args.features,
                args.trees, mode
            )
            row["fit_seconds"] = fit_seconds
            rows.append(row)

    write_reports(rows, args.output_dir)
    print(f"Wrote TreeSHAP comparison reports to {args.output_dir}")


if __name__ == "__main__":
    main()
