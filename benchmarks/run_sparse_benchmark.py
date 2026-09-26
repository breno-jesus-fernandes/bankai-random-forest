#!/usr/bin/env python3
"""Benchmark dense, CSR, and CSC paths with a warmup on the release build."""

import argparse
import csv
import os
import resource
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).parents[1]


def peak_rss_kib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / 1024) if sys.platform == "darwin" else round(value)


def tree_metrics(model):
    arrays = model._forest.shap_tree_arrays()
    total_nodes = 0
    maximum_depth = 0
    for left, right, features, *_ in arrays:
        total_nodes += len(features)
        pending = [(0, 0)]
        while pending:
            node, depth = pending.pop()
            maximum_depth = max(maximum_depth, depth)
            if features[node] >= 0:
                pending.append((left[node], depth + 1))
                pending.append((right[node], depth + 1))
    return total_nodes, maximum_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exact", "hist_32"), required=True)
    parser.add_argument("--format", choices=("dense", "csr", "csc"), required=True)
    parser.add_argument("--scenario", choices=("baseline", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    environment = os.environ.copy()
    rustflags = environment.get("RUSTFLAGS", "")
    environment["RUSTFLAGS"] = " ".join(
        part for part in (rustflags, "-C target-cpu=native") if part
    )
    subprocess.run(
        ["maturin", "develop", "--release"],
        check=True,
        cwd=project_root,
        env=environment,
    )

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
        warmup._forest.predict(matrix)
        for repeat in range(1, 4):
            model = BankaiRandomForestClassifier(**params)
            started = time.perf_counter()
            model.fit(matrix, y)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            model.predict(matrix)
            predict_seconds = time.perf_counter() - started
            started = time.perf_counter()
            model._forest.predict(matrix)
            native_predict_seconds = time.perf_counter() - started
            total_nodes, max_depth = tree_metrics(model)
            rows.append({"mode": args.mode, "format": args.format, "scenario": args.scenario,
                         "seed": seed, "repeat": repeat, "fit_seconds": fit_seconds,
                         "predict_seconds": predict_seconds,
                         "native_predict_seconds": native_predict_seconds,
                         "total_nodes": total_nodes, "max_depth": max_depth,
                         "peak_rss_kib": peak_rss_kib()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
