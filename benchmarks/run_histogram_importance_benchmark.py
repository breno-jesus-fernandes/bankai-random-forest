#!/usr/bin/env python3
"""Measure prediction and permutation-importance costs across histogram modes."""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmarks.run_benchmark import generate_dataset, prepare_optimized_binaries


def elapsed(call):
    started = time.perf_counter()
    result = call()
    return time.perf_counter() - started, result


def rank_correlation(left, right):
    left_rank = np.empty(left.size, dtype=np.float64)
    right_rank = np.empty(right.size, dtype=np.float64)
    left_rank[np.argsort(left, kind="stable")] = np.arange(left.size)
    right_rank[np.argsort(right, kind="stable")] = np.arange(right.size)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def benchmark(rows, features, trees, bins, repeats, x, y, validation_x, validation_y):
    from bankai_random_forest import BankaiRandomForestClassifier

    records = []
    for label, max_bins in [("exact", None), *((f"histogram_{n}", n) for n in bins)]:
        fit_samples, predict_samples, access_samples, scores, importance_values = [], [], [], [], []
        for seed in range(repeats):
            model = BankaiRandomForestClassifier(
                n_estimators=trees,
                max_bins=max_bins,
                importance_type="permutation",
                random_state=42 + seed,
                n_jobs=1,
            )
            fit_time, _ = elapsed(lambda: model.fit(x, y))
            predict_time, predictions = elapsed(lambda: model.predict(validation_x))
            access_time, importances = elapsed(lambda: model.feature_importances_)
            fit_samples.append(fit_time)
            predict_samples.append(predict_time)
            access_samples.append(access_time)
            scores.append(f1_score(validation_y, predictions))
            importance_values.append(np.asarray(importances, dtype=np.float64))

        records.append({
            "implementation": "bankai",
            "mode": label,
            "max_bins": "" if max_bins is None else max_bins,
            "rows": rows,
            "validation_rows": len(validation_y),
            "features": features,
            "trees": trees,
            "repeats": repeats,
            "fit_seconds": statistics.median(fit_samples),
            "predict_seconds": statistics.median(predict_samples),
            "feature_importance_seconds": statistics.median(access_samples),
            "predict_plus_importance_seconds": statistics.median(predict_samples)
            + statistics.median(access_samples),
            "end_to_end_seconds": statistics.median(fit_samples)
            + statistics.median(predict_samples)
            + statistics.median(access_samples),
            "validation_f1": statistics.median(scores),
            "importance_rank_correlation_vs_exact": "pending",
        })
        records[-1]["_importance"] = np.median(np.stack(importance_values), axis=0)

    exact_importance = records[0]["_importance"]
    for record in records:
        record["importance_rank_correlation_vs_exact"] = rank_correlation(
            exact_importance, record["_importance"]
        )

    sklearn = RandomForestClassifier(
        n_estimators=trees, max_features="sqrt", n_jobs=1, random_state=42
    )
    fit_time, _ = elapsed(lambda: sklearn.fit(x, y))
    predict_time, predictions = elapsed(lambda: sklearn.predict(validation_x))
    importance_time, result = elapsed(lambda: permutation_importance(
        sklearn, validation_x, validation_y, scoring="f1", n_repeats=1,
        n_jobs=1, random_state=42,
    ))
    sklearn_record = {
        "implementation": "sklearn",
        "mode": "external_permutation",
        "max_bins": "",
        "rows": rows,
        "validation_rows": len(validation_y),
        "features": features,
        "trees": trees,
        "repeats": repeats,
        "fit_seconds": fit_time,
        "predict_seconds": predict_time,
        "feature_importance_seconds": importance_time,
        "predict_plus_importance_seconds": predict_time + importance_time,
        "end_to_end_seconds": fit_time + predict_time + importance_time,
        "validation_f1": f1_score(validation_y, predictions),
        "importance_rank_correlation_vs_exact": rank_correlation(
            exact_importance, result.importances_mean
        ),
    }
    records.append(sklearn_record)
    for record in records:
        record.pop("_importance", None)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--bins", nargs="+", type=int, default=[16, 32, 64, 128, 255])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results-histogram-10k"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.rows > 100_000:
        parser.error("--rows is capped at 100000 by the benchmark memory policy")
    if not args.skip_build:
        prepare_optimized_binaries(Path(__file__).parents[1])
    all_x, all_y = generate_dataset(args.rows * 2, args.features)
    rows = benchmark(
        args.rows, args.features, args.trees, args.bins, args.repeats,
        all_x[:args.rows], all_y[:args.rows], all_x[args.rows:], all_y[args.rows:],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (args.output_dir / "histogram_importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "histogram_importance.md").open("w") as handle:
        handle.write("# Histogram and permutation importance benchmark\n\n")
        handle.write("Times are median Bankai results across seeds; sklearn uses one fit and one external permutation pass. Bankai computes native OOB permutation importance during fit, so `feature_importance_seconds` measures the public attribute access separately and `fit_seconds` includes its computation.\n\n")
        handle.write("| mode | bins | fit s (includes Bankai OOB permutation) | predict s | importance access/calculation s | predict + importance s | end to end s | F1 | rank corr vs exact |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write("| {mode} | {max_bins} | {fit_seconds:.6f} | {predict_seconds:.6f} | {feature_importance_seconds:.6f} | {predict_plus_importance_seconds:.6f} | {end_to_end_seconds:.6f} | {validation_f1:.6f} | {importance_rank_correlation_vs_exact:.6f} |\n".format(**row))
    print(f"Wrote {args.output_dir / 'histogram_importance.csv'} and Markdown report")


if __name__ == "__main__":
    main()
