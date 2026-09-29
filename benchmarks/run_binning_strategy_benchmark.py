#!/usr/bin/env python3
"""Compare exact/sampled and sort/select histogram cut construction.

The benchmark reports end-to-end ``fit`` time and holdout classification
metrics. Dataset creation, estimator setup, and metric calculation are outside
the fit timer. Run against a release build for meaningful timing.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from bankai_random_forest import BankaiRandomForestClassifier


STRATEGIES = ("exact_sort", "sampled_sort", "exact_select", "sampled_select")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--bins", type=int, default=63)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--trees", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("results-binning-strategies.csv")
    )
    args = parser.parse_args()
    if args.rows < 100 or args.features < 2:
        parser.error("--rows must be at least 100 and --features at least 2")

    x, y = make_classification(
        n_samples=args.rows,
        n_features=args.features,
        n_informative=max(2, args.features // 5),
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=123, stratify=y
    )

    fit_times = {strategy: [] for strategy in STRATEGIES}
    metrics = {}

    def run(strategy, measured):
        model = BankaiRandomForestClassifier(
            n_estimators=args.trees,
            max_features="sqrt",
            max_depth=args.depth,
            max_bins=args.bins,
            binning_strategy=strategy,
            bin_sample_size=args.sample_size,
            importance_type="gain",
            n_jobs=args.jobs,
            random_state=7,
        )
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        if measured:
            fit_times[strategy].append(fit_seconds)
            prediction = model.predict(x_test)
            metrics[strategy] = {
                "accuracy": accuracy_score(y_test, prediction),
                "precision": precision_score(y_test, prediction, zero_division=0),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "f1": f1_score(y_test, prediction, zero_division=0),
            }

    for warmup in range(args.warmups):
        order = STRATEGIES[warmup % len(STRATEGIES) :] + STRATEGIES[: warmup % len(STRATEGIES)]
        for strategy in order:
            run(strategy, measured=False)

    for repeat in range(args.repeats):
        offset = repeat % len(STRATEGIES)
        order = STRATEGIES[offset:] + STRATEGIES[:offset]
        if repeat % 2:
            order = tuple(reversed(order))
        for strategy in order:
            run(strategy, measured=True)

    records = []
    for strategy in STRATEGIES:
        records.append(
            {
                "strategy": strategy,
                "rows": args.rows,
                "features": args.features,
                "bins": args.bins,
                "bin_sample_size": args.sample_size,
                "trees": args.trees,
                "max_depth": args.depth,
                "n_jobs": args.jobs,
                "repeats": args.repeats,
                "median_fit_seconds": statistics.median(fit_times[strategy]),
                "min_fit_seconds": min(fit_times[strategy]),
                "max_fit_seconds": max(fit_times[strategy]),
                **metrics[strategy],
            }
        )

    baseline = next(
        row["median_fit_seconds"] for row in records if row["strategy"] == "exact_sort"
    )
    for row in records:
        row["speedup_vs_exact_sort"] = baseline / row["median_fit_seconds"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    columns = (
        "strategy",
        "median_fit_seconds",
        "speedup_vs_exact_sort",
        "accuracy",
        "precision",
        "recall",
        "f1",
    )
    print(" | ".join(columns))
    for row in records:
        print(" | ".join(f"{row[column]:.4f}" if isinstance(row[column], float) else str(row[column]) for column in columns))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
