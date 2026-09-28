#!/usr/bin/env python3
"""Compare serial/all-core histogram fit timings on a fixed dense workload.

Run this same script from the baseline and feature worktree to get before/after
measurements. One-tree runs make preprocessing a larger share of the measured
fit; multi-tree runs report end-to-end impact under normal parallel training.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from sklearn.datasets import make_classification

from bankai_random_forest import BankaiRandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--trees", type=int, nargs="+", default=[1, 40])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("histogram_preprocessing.csv"))
    args = parser.parse_args()

    x, y = make_classification(
        n_samples=args.rows, n_features=args.features, n_informative=max(2, args.features // 5),
        n_redundant=0, n_classes=2, random_state=42,
    )
    records = []
    for trees in args.trees:
        samples = {1: [], -1: []}
        for jobs in (1, -1):
            for repeat in range(args.warmups):
                model = BankaiRandomForestClassifier(
                    n_estimators=trees, max_bins=args.bins, max_features=None,
                    importance_type="gain", n_jobs=jobs, random_state=100 + repeat,
                )
                model.fit(x, y)
        for repeat in range(args.repeats):
            jobs_order = (1, -1) if repeat % 2 == 0 else (-1, 1)
            for jobs in jobs_order:
                model = BankaiRandomForestClassifier(
                    n_estimators=trees, max_bins=args.bins, max_features=None,
                    importance_type="gain", n_jobs=jobs, random_state=100 + args.warmups + repeat,
                )
                start = time.perf_counter()
                model.fit(x, y)
                samples[jobs].append(time.perf_counter() - start)
        for jobs in (1, -1):
            records.append({
                "rows": args.rows, "features": args.features, "bins": args.bins,
                "trees": trees, "n_jobs": jobs, "repeats": args.repeats,
                "median_fit_seconds": statistics.median(samples[jobs]),
                "min_fit_seconds": min(samples[jobs]), "max_fit_seconds": max(samples[jobs]),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        print(row)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
