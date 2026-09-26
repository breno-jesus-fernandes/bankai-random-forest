#!/usr/bin/env python3
"""Compare sklearn, the PyO3 wrapper, and the optimized Rust CLI."""

import argparse
import csv
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

PROFILES = {
    "default": {},
    "entropy": {"criterion": "entropy"},
    "all_features": {"max_features": None},
    "depth_8": {"max_depth": 8},
    "leaf_10": {"min_samples_leaf": 10},
    "subsample_50": {"max_samples": 0.5},
    "no_bootstrap": {"bootstrap": False},
    "balanced": {"class_weight": "balanced"},
}

def generate_dataset(rows, features):
    state = 42
    mask = (1 << 64) - 1
    values = np.empty((rows, features), dtype=np.float64)
    labels = np.empty(rows, dtype=np.intp)
    for row in range(rows):
        score = 0.0
        for feature in range(features):
            state = (state * 6_364_136_223_846_793_005 + 1) & mask
            value = ((state >> 11) / (1 << 53)) * 2.0 - 1.0
            values[row, feature] = value
            if feature < 3:
                score += value
        labels[row] = int(score > 0.0)
    return values, labels


def measure_python(name, classifier, x, y):
    started = time.perf_counter()
    classifier.fit(x, y)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = classifier.predict_proba(x)
    predict_seconds = time.perf_counter() - started
    predicted = probabilities.argmax(axis=1)
    return {
        "implementation": name,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "f1": f1_score(y, predicted),
        "predicted": predicted,
        "probabilities": probabilities,
    }


def measure_rust(binary, rows, features, trees, parameters):
    with tempfile.TemporaryDirectory() as directory:
        predictions_path = Path(directory) / "predictions.csv"
        command = [
                str(binary),
                "--rows",
                str(rows),
                "--features",
                str(features),
                "--trees",
                str(trees),
                "--predictions",
                str(predictions_path),
            ]
        if parameters.get("criterion"):
            command += ["--criterion", parameters["criterion"]]
        if parameters.get("max_features") is None and "max_features" in parameters:
            command += ["--max-features", str(features)]
        if parameters.get("max_depth"):
            command += ["--max-depth", str(parameters["max_depth"])]
        if parameters.get("min_samples_leaf"):
            command += ["--min-samples-leaf", str(parameters["min_samples_leaf"])]
        if parameters.get("bootstrap") is False:
            command += ["--no-bootstrap"]
        if parameters.get("max_samples"):
            command += ["--max-samples", str(round(parameters["max_samples"] * rows))]
        if parameters.get("class_weight") == "balanced":
            command += ["--balanced-class-weight"]
        output = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        metrics = next(csv.DictReader(output.stdout.splitlines()))
        predictions = list(csv.DictReader(predictions_path.read_text().splitlines()))
    return {
        "implementation": "rust-cli",
        "train_seconds": float(metrics["train_seconds"]),
        "predict_seconds": float(metrics["predict_seconds"]),
        "f1": float(metrics["f1"]),
        "predicted": np.array([int(row["predicted"]) for row in predictions]),
        "probabilities": np.array(
            [[float(row["probability_0"]), float(row["probability_1"])] for row in predictions]
        ),
    }


def benchmark(rows, features, trees, binary, profile, parameters):
    from bankai_random_forest import BankaiRandomForestClassifier

    x, y = generate_dataset(rows, features)
    estimator_parameters = {
        "n_estimators": trees,
        "max_features": "sqrt",
        "n_jobs": 1,
        "random_state": 42,
        **parameters,
    }
    sklearn_result = measure_python(
        "sklearn",
        RandomForestClassifier(**estimator_parameters),
        x,
        y,
    )
    bankai_result = measure_python(
        "pyO3",
        BankaiRandomForestClassifier(**estimator_parameters),
        x,
        y,
    )
    results = [sklearn_result, bankai_result, measure_rust(binary, rows, features, trees, parameters)]
    rows_out = []
    for result in results:
        probability_rmse = float(
            np.sqrt(np.mean((result["probabilities"] - sklearn_result["probabilities"]) ** 2))
        )
        agreement = float(np.mean(result["predicted"] == sklearn_result["predicted"]))
        rows_out.append(
            {
                "rows": rows,
                "features": features,
                "trees": trees,
                "profile": profile,
                "implementation": result["implementation"],
                "train_seconds": result["train_seconds"],
                "predict_seconds": result["predict_seconds"],
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "ffi_overhead_seconds": (
                    result["train_seconds"] + result["predict_seconds"]
                    - results[2]["train_seconds"]
                    - results[2]["predict_seconds"]
                    if result["implementation"] == "pyO3"
                    else 0.0
                ),
                "f1": result["f1"],
                "probability_rmse_vs_sklearn": probability_rmse,
                "agreement_vs_sklearn": agreement,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "sklearn": sklearn.__version__,
            }
        )
    return rows_out


def write_reports(rows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (directory / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "benchmark.md").open("w") as handle:
        handle.write("| rows | profile | implementation | train s | predict s | F1 | probability RMSE | agreement |\n")
        handle.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                "| {rows} | {profile} | {implementation} | {train_seconds:.6f} | {predict_seconds:.6f} | "
                "{f1:.6f} | {probability_rmse_vs_sklearn:.6f} | {agreement_vs_sklearn:.6f} |\n".format(
                    **row
                )
            )


def prepare_optimized_binaries(root):
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
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "bankai-xrf-cli"],
        check=True,
        cwd=root,
        env=environment,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[10_000, 100_000])
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--rust-binary", type=Path, default=Path("target/release/bankai-xrf-cli"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=sorted(PROFILES))
    arguments = parser.parse_args()
    root = Path(__file__).parents[1]
    if not arguments.skip_build:
        prepare_optimized_binaries(root)
    if not arguments.rust_binary.is_file():
        parser.error("optimized Rust binary missing; run the documented release build first")
    rows = [
        row
        for row_count in arguments.rows
        for profile in arguments.profiles
        for row in benchmark(row_count, arguments.features, arguments.trees, arguments.rust_binary, profile, PROFILES[profile])
    ]
    write_reports(rows, arguments.output_dir)


if __name__ == "__main__":
    main()
