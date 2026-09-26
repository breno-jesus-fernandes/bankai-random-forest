#!/usr/bin/env python3
"""Compare sklearn permutation importance with native XRF OOB importance."""

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmarks.run_benchmark import PROFILES, generate_dataset, prepare_optimized_binaries


IMPORTANCE_PROFILES = {
    name: parameters
    for name, parameters in PROFILES.items()
    if parameters.get("bootstrap", True)
}


def estimator_parameters(trees, parameters):
    return {
        "n_estimators": trees,
        "max_features": "sqrt",
        "n_jobs": 1,
        "random_state": 42,
        **parameters,
    }


def rank_correlation(reference, candidate):
    reference_ranks = np.empty(reference.size, dtype=np.float64)
    candidate_ranks = np.empty(candidate.size, dtype=np.float64)
    reference_ranks[np.argsort(reference, kind="stable")] = np.arange(reference.size)
    candidate_ranks[np.argsort(candidate, kind="stable")] = np.arange(candidate.size)
    if np.std(reference_ranks) == 0.0 or np.std(candidate_ranks) == 0.0:
        return 1.0 if np.array_equal(reference_ranks, candidate_ranks) else float("nan")
    return float(np.corrcoef(reference_ranks, candidate_ranks)[0, 1])


def top_three_overlap(reference, candidate):
    count = min(3, reference.size)
    reference_top = set(np.argsort(reference)[-count:])
    candidate_top = set(np.argsort(candidate)[-count:])
    return len(reference_top & candidate_top) / count


def measure_sklearn(parameters, x, y):
    classifier = RandomForestClassifier(**parameters)
    started = time.perf_counter()
    classifier.fit(x, y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = permutation_importance(
        classifier, x, y, scoring="f1", n_repeats=1, n_jobs=1, random_state=42
    )
    importance_seconds = time.perf_counter() - started
    return fit_seconds, importance_seconds, result.importances_mean


def measure_bankai(parameters, x, y):
    from bankai_random_forest import BankaiRandomForestClassifier

    started = time.perf_counter()
    BankaiRandomForestClassifier(**parameters).fit(x, y)
    baseline_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    classifier = BankaiRandomForestClassifier(
        **parameters, importance_type="permutation"
    ).fit(x, y)
    fit_with_importance_seconds = time.perf_counter() - started
    return (
        baseline_fit_seconds,
        fit_with_importance_seconds - baseline_fit_seconds,
        fit_with_importance_seconds,
        classifier.feature_importances_,
    )


def cli_command(binary, rows, features, trees, parameters):
    command = [
        str(binary),
        "--rows",
        str(rows),
        "--features",
        str(features),
        "--trees",
        str(trees),
    ]
    if parameters.get("criterion"):
        command += ["--criterion", parameters["criterion"]]
    if parameters.get("max_features") is None and "max_features" in parameters:
        command += ["--max-features", str(features)]
    if parameters.get("max_depth"):
        command += ["--max-depth", str(parameters["max_depth"])]
    if parameters.get("min_samples_leaf"):
        command += ["--min-samples-leaf", str(parameters["min_samples_leaf"])]
    if parameters.get("max_samples"):
        command += ["--max-samples", str(round(parameters["max_samples"] * rows))]
    if parameters.get("class_weight") == "balanced":
        command += ["--balanced-class-weight"]
    return command


def cli_train_seconds(command):
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(next(csv.DictReader(output.stdout.splitlines()))["train_seconds"])


def measure_rust(binary, rows, features, trees, parameters):
    command = cli_command(binary, rows, features, trees, parameters)
    baseline_fit_seconds = cli_train_seconds(command)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "importances.csv"
        importance_command = command + [
            "--permutation-importance",
            "--importances",
            str(path),
        ]
        fit_with_importance_seconds = cli_train_seconds(importance_command)
        importances = np.array(
            [float(row["importance"]) for row in csv.DictReader(path.read_text().splitlines())]
        )
    return (
        baseline_fit_seconds,
        fit_with_importance_seconds - baseline_fit_seconds,
        fit_with_importance_seconds,
        importances,
    )


def benchmark(rows, features, trees, binary, profile, parameters):
    x, y = generate_dataset(rows, features)
    parameters = estimator_parameters(trees, parameters)
    sklearn_fit, sklearn_importance, sklearn_values = measure_sklearn(parameters, x, y)
    measurements = [
        ("sklearn", sklearn_fit, sklearn_importance, sklearn_fit + sklearn_importance, sklearn_values),
        ("pyO3", *measure_bankai(parameters, x, y)),
        ("rust-cli", *measure_rust(binary, rows, features, trees, parameters)),
    ]
    rows_out = []
    for implementation, fit_seconds, importance_seconds, total_seconds, values in measurements:
        rows_out.append(
            {
                "rows": rows,
                "features": features,
                "trees": trees,
                "profile": profile,
                "implementation": implementation,
                "importance_source": (
                    "validation permutation (one repeat)"
                    if implementation == "sklearn"
                    else "native XRF OOB permutation"
                ),
                "fit_seconds": fit_seconds,
                "importance_seconds": importance_seconds,
                "total_seconds": total_seconds,
                "importance_rank_correlation_vs_sklearn": rank_correlation(
                    sklearn_values, values
                ),
                "top_three_overlap_vs_sklearn": top_three_overlap(sklearn_values, values),
                "top_feature": int(np.argmax(values)),
            }
        )
    return rows_out


def write_reports(rows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (directory / "permutation_importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "permutation_importance.md").open("w") as handle:
        handle.write(
            "| rows | profile | implementation | fit s | importance s | total s | rank correlation | top-3 overlap | top feature |\n"
        )
        handle.write(
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            handle.write(
                "| {rows} | {profile} | {implementation} | {fit_seconds:.6f} | "
                "{importance_seconds:.6f} | {total_seconds:.6f} | "
                "{importance_rank_correlation_vs_sklearn:.6f} | "
                "{top_three_overlap_vs_sklearn:.6f} | {top_feature} |\n".format(**row)
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[10_000])
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results-permutation-importance")
    )
    parser.add_argument(
        "--rust-binary", type=Path, default=Path("target/release/bankai-xrf-cli")
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--profiles", nargs="+", choices=sorted(IMPORTANCE_PROFILES), default=sorted(IMPORTANCE_PROFILES)
    )
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
        for row in benchmark(
            row_count,
            arguments.features,
            arguments.trees,
            arguments.rust_binary,
            profile,
            IMPORTANCE_PROFILES[profile],
        )
    ]
    write_reports(rows, arguments.output_dir)


if __name__ == "__main__":
    main()
