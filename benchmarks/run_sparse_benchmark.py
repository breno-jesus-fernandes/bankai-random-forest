#!/usr/bin/env python3
"""Benchmark dense, CSR, and CSC paths with a warmup on the release build."""

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse


def peak_rss_kib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / 1024) if sys.platform == "darwin" else round(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exact", "hist_32"), required=True)
    parser.add_argument("--format", choices=("dense", "csr", "csc"), required=True)
    parser.add_argument("--scenario", choices=("baseline", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from bankai_random_forest import BankaiRandomForestClassifier

    rng = np.random.RandomState(861)
    x = rng.normal(size=(10_000, 20))
    x[rng.random_sample(x.shape) < 0.9] = 0.0
    y = (x[:, 0] + 0.8 * x[:, 1] - 0.4 * x[:, 2] > 0.0).astype(np.int64)
    if args.format == "csr":
        matrix = sparse.csr_matrix(x)
    elif args.format == "csc":
        matrix = sparse.csc_matrix(x)
    else:
        matrix = np.ascontiguousarray(x)
    max_bins = None if args.mode == "exact" else 32
    rows = []

    for seed in (17, 29, 43):
        params = dict(n_estimators=50, max_features=4, random_state=seed, n_jobs=1, max_bins=max_bins)
        warmup = BankaiRandomForestClassifier(**params).fit(matrix, y)
        warmup.predict(matrix)
        for repeat in range(1, 4):
            model = BankaiRandomForestClassifier(**params)
            started = time.perf_counter()
            model.fit(matrix, y)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            model.predict(matrix)
            predict_seconds = time.perf_counter() - started
            rows.append({"mode": args.mode, "format": args.format, "scenario": args.scenario,
                         "seed": seed, "repeat": repeat, "fit_seconds": fit_seconds,
                         "predict_seconds": predict_seconds, "peak_rss_kib": peak_rss_kib()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
