#!/usr/bin/env python3
"""Measure exact/histogram forests with cost-complexity pruning."""

import argparse
import csv
import os
import resource
import subprocess
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
    parser.add_argument("--ccp-alpha", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    rustflags = environment.get("RUSTFLAGS", "")
    environment["RUSTFLAGS"] = " ".join(
        part for part in (rustflags, "-C target-cpu=native") if part
    )
    subprocess.run(
        ["uv", "run", "maturin", "develop", "--release"],
        check=True,
        cwd=root,
        env=environment,
    )

    from bankai_random_forest import _core

    rng = np.random.RandomState(861)
    x = np.ascontiguousarray(rng.normal(size=(10_000, 20)), dtype=np.float64)
    signal = x[:, 0] + 0.8 * x[:, 1] - 0.4 * x[:, 2]
    y = (signal > np.quantile(signal, 0.9)).astype(np.int64)
    max_bins = None if arguments.mode == "exact" else 32
    rows = []

    for seed in (17, 29, 43):
        params = dict(
            n_estimators=100,
            max_features=4,
            random_state=seed,
            n_jobs=1,
            max_bins=max_bins,
            ccp_alpha=arguments.ccp_alpha,
        )
        warmup = _core.NativeForest()
        warmup.fit(x, y, **params)
        warmup.predict(x)

        for repeat in range(1, 4):
            model = _core.NativeForest()
            started = time.perf_counter()
            model.fit(x, y, **params)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            model.predict(x)
            predict_seconds = time.perf_counter() - started
            rows.append(
                {
                    "mode": arguments.mode,
                    "ccp_alpha": arguments.ccp_alpha,
                    "seed": seed,
                    "repeat": repeat,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "peak_rss_kib": peak_rss_kib(),
                }
            )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
