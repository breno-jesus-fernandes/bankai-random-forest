#!/usr/bin/env python3
"""Paired Bankai exact/histogram Extra Trees vs scikit-learn Extra Trees."""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from bankai_random_forest import BankaiRandomForestClassifier

CASES = ((10_000, 20, 100), (100_000, 20, 100), (10_000, 500, 40), (100_000, 100, 100))
VARIANTS = ("bankai_exact", "bankai_histogram", "sklearn")


def dataset(rows: int, features: int, seed: int):
    rng = np.random.default_rng(seed + 200)
    x = rng.standard_normal((rows, features), dtype=np.float32)
    signal = x[:, 0] + 0.6 * x[:, min(1, features - 1)] - 0.35 * x[:, min(2, features - 1)]
    y = (signal + rng.standard_normal(rows) * 0.8 > 0).astype(np.int64)
    train, test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=9127, stratify=y
    )
    return train, test, y_train, y_test


def make_model(variant: str, trees: int, seed: int, jobs: int, bins: int):
    if variant == "sklearn":
        return ExtraTreesClassifier(
            n_estimators=trees, criterion="gini", max_features=None,
            bootstrap=True, max_samples=0.8, max_depth=20, max_leaf_nodes=511,
            min_samples_leaf=1, random_state=seed, n_jobs=jobs,
        )
    return BankaiRandomForestClassifier(
        n_estimators=trees, criterion="gini", max_features=None,
        bootstrap=True, max_samples=0.8, max_depth=20, max_leaf_nodes=511,
        min_samples_leaf=1, random_state=seed, n_jobs=jobs,
        max_bins=None if variant == "bankai_exact" else bins,
        extra_trees=True,
    )


def fit_one(variant: str, data, trees: int, data_seed: int, model_seed: int, args, case, warmup: bool):
    x_train, x_test, y_train, y_test = data
    model = make_model(variant, trees, model_seed, args.jobs, args.bins)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = probabilities >= 0.5
    return {
        "rows": case[0], "features": case[1], "trees": trees,
        "data_seed": data_seed, "model_seed": model_seed,
        "warmup": warmup, "variant": variant,
        "fit_seconds": fit_seconds,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--bins", type=int, default=63)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results-extra-trees-sklearn"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    for case in CASES:
        rows, features, trees = case
        # Warm each implementation once on the same workload; discard these timings.
        warm_data = dataset(rows, features, 10_000)
        for variant in VARIANTS:
            result = fit_one(variant, warm_data, trees, 10_000, 10_004, args, case, True)
            records.append(result)
            print(result, flush=True)
        for seed in range(args.seeds):
            data = dataset(rows, features, seed)
            order = list(VARIANTS)
            if seed % 2:
                order.reverse()
            for variant in order:
                result = fit_one(variant, data, trees, seed, seed + 4, args, case, False)
                records.append(result)
                print(result, flush=True)
        with (args.output / "runs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    measured = [row for row in records if not row["warmup"]]
    summary_rows = []
    for rows, features, trees in CASES:
        case_records = [r for r in measured if r["rows"] == rows and r["features"] == features]
        medians = {
            variant: statistics.median(r["fit_seconds"] for r in case_records if r["variant"] == variant)
            for variant in VARIANTS
        }
        for variant, seconds in medians.items():
            own = [r for r in case_records if r["variant"] == variant]
            summary_rows.append({
                "rows": rows, "features": features, "trees": trees, "variant": variant,
                "median_fit_seconds": seconds,
                "speed_ratio_vs_sklearn": seconds / medians["sklearn"],
                "median_accuracy": statistics.median(r["accuracy"] for r in own),
                "median_f1": statistics.median(r["f1"] for r in own),
                "median_roc_auc": statistics.median(r["roc_auc"] for r in own),
            })
    with (args.output / "medians.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    environment = (
        f"Python: {platform.python_version()}\nNumPy: {np.__version__}\n"
        f"scikit-learn: {sklearn.__version__}\nPlatform: {platform.platform()}\n"
        f"Jobs: {args.jobs}\nBins: {args.bins}\nSeeds: {args.seeds}\n"
        "Bankai release build should be compiled with RUSTFLAGS='-C target-cpu=native'.\n"
    )
    (args.output / "environment.txt").write_text(environment)
    lines = [
        "# Bankai Extra Trees histogram comparison",
        "",
        "Each measured seed uses the same generated dense float32 dataset and split for all models. "
        "One warmup per model and workload is discarded. Fits use Gini, bootstrap with 80% of rows, "
        "depth 20, up to 511 leaves, all features, and one thread. Bankai histogram mode uses the "
        f"configured {args.bins} bins. A lower speed ratio means faster than scikit-learn.",
        "",
        "| Rows × features | Trees | Variant | Median fit (s) | Fit / sklearn | Accuracy | F1 | ROC AUC |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['rows']} × {row['features']} | {row['trees']} | {row['variant']} | "
            f"{row['median_fit_seconds']:.3f} | {row['speed_ratio_vs_sklearn']:.3f} | "
            f"{row['median_accuracy']:.4f} | {row['median_f1']:.4f} | {row['median_roc_auc']:.4f} |"
        )
    lines.extend(["", "## Environment", "", "```text", environment.rstrip(), "```", ""])
    (args.output / "report.md").write_text("\n".join(lines))
    for row in summary_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
