#!/usr/bin/env python3
"""Measure monotonic-constraint overhead using the release-built extension."""

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import numpy as np


def peak_rss_kib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / 1024) if sys.platform == "darwin" else round(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exact", "hist_32"), required=True)
    parser.add_argument(
        "--scenario", choices=("baseline", "after", "monotonic"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from bankai_random_forest import BankaiRandomForestClassifier

    rng = np.random.RandomState(861)
    x = np.ascontiguousarray(rng.normal(size=(10_000, 20)), dtype=np.float64)
    signal = x[:, 0] + 0.8 * x[:, 1] - 0.4 * x[:, 2]
    y = (signal > np.quantile(signal, 0.9)).astype(np.int64)
    max_bins = None if args.mode == "exact" else 32
    constraints = [1] + [0] * (x.shape[1] - 1) if args.scenario == "monotonic" else None
    rows = []

    for seed in (17, 29, 43):
        params = dict(
            n_estimators=100,
            max_features=4,
            random_state=seed,
            n_jobs=1,
            max_bins=max_bins,
        )
        if args.scenario != "baseline":
            params["monotonic_cst"] = constraints

        warmup = BankaiRandomForestClassifier(**params).fit(x, y)
        warmup.predict(x)
        for repeat in range(1, 4):
            model = BankaiRandomForestClassifier(**params)
            started = time.perf_counter()
            model.fit(x, y)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            model.predict(x)
            predict_seconds = time.perf_counter() - started
            rows.append(
                {
                    "mode": args.mode,
                    "scenario": args.scenario,
                    "seed": seed,
                    "repeat": repeat,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "peak_rss_kib": peak_rss_kib(),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
