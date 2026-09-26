#!/usr/bin/env python3
"""Compare split and permutation feature-importance methods."""

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


def measure_sklearn(parameters, x, y, validation_x, validation_y, importance_type):
    classifier = RandomForestClassifier(**parameters)
    started = time.perf_counter()
    classifier.fit(x, y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if importance_type == "gain":
        values = classifier.feature_importances_
    elif importance_type == "split":
        counts = np.zeros(x.shape[1], dtype=np.float64)
        for tree in classifier.estimators_:
            used, occurrences = np.unique(tree.tree_.feature, return_counts=True)
            valid = used >= 0
            counts[used[valid]] += occurrences[valid]
        values = counts / counts.sum() if counts.sum() else counts
    else:
        result = permutation_importance(
            classifier,
            validation_x,
            validation_y,
            scoring="f1",
            n_repeats=5,
            n_jobs=1,
            random_state=42,
        )
        values = result.importances_mean
    importance_seconds = time.perf_counter() - started
    return fit_seconds, importance_seconds, fit_seconds + importance_seconds, values


def measure_bankai(parameters, x, y, importance_type):
    from bankai_random_forest import BankaiRandomForestClassifier

    if importance_type in ("gain", "split"):
        started = time.perf_counter()
        classifier = BankaiRandomForestClassifier(
            **parameters, importance_type=importance_type
        ).fit(x, y)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        values = (
            classifier._forest.gain_importances()
            if importance_type == "gain"
            else classifier._forest.feature_importances()
        )
        importance_seconds = time.perf_counter() - started
        total_seconds = fit_seconds + importance_seconds
    else:
        started = time.perf_counter()
        BankaiRandomForestClassifier(**parameters).fit(x, y)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        classifier = BankaiRandomForestClassifier(
            **parameters, importance_type="permutation"
        ).fit(x, y)
        total_seconds = time.perf_counter() - started
        importance_seconds = total_seconds - fit_seconds
        values = classifier.feature_importances_
    return fit_seconds, importance_seconds, total_seconds, np.asarray(values, dtype=np.float64)


def cli_command(binary, rows, features, trees, parameters):
    command = [
        str(binary), "--rows", str(rows), "--features", str(features), "--trees", str(trees)
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


def cli_measure(command):
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    metrics = next(csv.DictReader(output.stdout.splitlines()))
    extraction_seconds = 0.0
    for line in output.stderr.splitlines():
        if line.startswith(("split_importance_seconds=", "gain_importance_seconds=")):
            extraction_seconds = float(line.split("=", 1)[1])
    return float(metrics["train_seconds"]), extraction_seconds


def measure_rust(binary, rows, features, trees, parameters, importance_type):
    command = cli_command(binary, rows, features, trees, parameters)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "importances.csv"
        if importance_type == "gain":
            importance_command = command + ["--gain-importances", str(path)]
        elif importance_type == "split":
            importance_command = command + ["--split-importances", str(path)]
        else:
            importance_command = command + [
                "--permutation-importance", "--importances", str(path)
            ]
        total_seconds, extraction_seconds = cli_measure(importance_command)
        values = np.array(
            [float(row["importance"]) for row in csv.DictReader(path.read_text().splitlines())]
        )
        if importance_type == "permutation":
            fit_seconds, _ = cli_measure(command)
            importance_seconds = total_seconds - fit_seconds
        else:
            fit_seconds = total_seconds
            importance_seconds = extraction_seconds
            total_seconds += extraction_seconds
    return fit_seconds, importance_seconds, total_seconds, values


def benchmark(rows, features, trees, binary, profile, parameters, importance_type):
    all_x, all_y = generate_dataset(rows * 2, features)
    x, y = all_x[:rows], all_y[:rows]
    validation_x, validation_y = all_x[rows:], all_y[rows:]
    parameters = estimator_parameters(trees, parameters)
    sklearn_result = measure_sklearn(
        parameters, x, y, validation_x, validation_y, importance_type
    )
    measurements = [
        ("sklearn", *sklearn_result),
        ("pyO3", *measure_bankai(parameters, x, y, importance_type)),
        ("rust-cli", *measure_rust(binary, rows, features, trees, parameters, importance_type)),
    ]
    reference = sklearn_result[3]
    source = {
        "gain": "sklearn impurity gain",
        "split": "split count derived from sklearn tree nodes",
        "permutation": "sklearn inspection permutation (five repeats)",
    }[importance_type]
    rows_out = []
    for implementation, fit_seconds, importance_seconds, total_seconds, values in measurements:
        if implementation != "sklearn":
            source_name = {
                "gain": "criterion-weighted XRF split gain",
                "split": "normalized split frequency",
                "permutation": "native XRF OOB permutation",
            }[importance_type]
        else:
            source_name = source
        rows_out.append(
            {
                "rows": rows,
                "validation_rows": validation_x.shape[0],
                "features": features,
                "trees": trees,
                "profile": profile,
                "importance_type": importance_type,
                "implementation": implementation,
                "importance_source": source_name,
                "fit_seconds": fit_seconds,
                "importance_seconds": importance_seconds,
                "total_seconds": total_seconds,
                "rank_correlation_vs_sklearn": rank_correlation(reference, values),
                "top_three_overlap_vs_sklearn": top_three_overlap(reference, values),
                "top_feature": int(np.argmax(values)),
            }
        )
    return rows_out


def write_reports(rows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "feature_importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "feature_importance.md").open("w") as handle:
        handle.write(
            "| type | train rows | validation rows | profile | implementation | fit s | importance s | total s | rank correlation | top-3 overlap | top feature |\n"
        )
        handle.write(
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            handle.write(
                "| {importance_type} | {rows} | {validation_rows} | {profile} | {implementation} | "
                "{fit_seconds:.6f} | {importance_seconds:.6f} | {total_seconds:.6f} | "
                "{rank_correlation_vs_sklearn:.6f} | {top_three_overlap_vs_sklearn:.6f} | "
                "{top_feature} |\n".format(**row)
            )
        handle.write(
            "\n`gain` compares sklearn impurity decrease with accumulated XRF criterion-weighted split gain. `split` counts feature nodes in each implementation (sklearn counts are derived from fitted trees). `permutation` compares sklearn inspection permutation on the independent validation partition with native XRF per-tree OOB accuracy decrease. Sklearn permutation uses five repeats. All implementations train on the same first partition.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[10_000])
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results-feature-importance")
    )
    parser.add_argument("--rust-binary", type=Path, default=Path("target/release/bankai-xrf-cli"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--profiles", nargs="+", choices=sorted(IMPORTANCE_PROFILES), default=["default"]
    )
    parser.add_argument(
        "--importance-types", nargs="+", choices=["gain", "split", "permutation"],
        default=["gain", "split", "permutation"]
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
        for importance_type in arguments.importance_types
        for row in benchmark(
            row_count,
            arguments.features,
            arguments.trees,
            arguments.rust_binary,
            profile,
            IMPORTANCE_PROFILES[profile],
            importance_type,
        )
    ]
    write_reports(rows, arguments.output_dir)


if __name__ == "__main__":
    main()
