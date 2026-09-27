#!/usr/bin/env python3
"""Compare Bankai histogram_16 and LightGBM using all available CPU cores."""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmarks.run_benchmark import generate_dataset, prepare_optimized_binaries
from benchmarks.run_histogram_importance_benchmark import Progress, elapsed, format_duration


def rank_correlation(left, right):
    left_rank = np.empty(left.size, dtype=np.float64)
    right_rank = np.empty(right.size, dtype=np.float64)
    left_rank[np.argsort(left, kind="stable")] = np.arange(left.size)
    right_rank[np.argsort(right, kind="stable")] = np.arange(right.size)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def benchmark(args, x, y, validation_x, validation_y, progress, cores):
    from bankai_random_forest import BankaiRandomForestClassifier

    measured = {"bankai": [], "lightgbm": []}
    importances = {"bankai": [], "lightgbm": []}

    for seed_index in range(args.repeats):
        seed = 42 + seed_index
        for warmup_index in range(args.warmups):
            model = BankaiRandomForestClassifier(
                n_estimators=args.trees,
                max_bins=16,
                importance_type="permutation",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(x, y)
            _ = model.feature_importances_
            del model
            progress.advance(f"Bankai histogram_16 warmup {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}")

        model = BankaiRandomForestClassifier(
            n_estimators=args.trees,
            max_bins=16,
            importance_type="permutation",
            random_state=seed,
            n_jobs=-1,
        )
        fit_seconds, _ = elapsed(lambda: model.fit(x, y))
        importance_seconds, values = elapsed(lambda: model.feature_importances_)
        measured["bankai"].append((fit_seconds, importance_seconds))
        importances["bankai"].append(np.asarray(values, dtype=np.float64))
        del model
        progress.advance(f"Bankai histogram_16 measured, seed {seed_index + 1}/{args.repeats}")

    for seed_index in range(args.repeats):
        seed = 42 + seed_index
        for warmup_index in range(args.warmups):
            model = LGBMClassifier(
                boosting_type="rf",
                n_estimators=args.trees,
                bagging_freq=1,
                bagging_fraction=0.8,
                feature_fraction=1.0,
                n_jobs=-1,
                random_state=seed,
                verbosity=-1,
            )
            model.fit(x, y)
            model.set_params(n_jobs=1)
            permutation_importance(
                model, validation_x, validation_y, scoring="f1", n_repeats=1,
                n_jobs=-1, random_state=seed,
            )
            del model
            progress.advance(f"LightGBM RF warmup {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}")

        model = LGBMClassifier(
            boosting_type="rf",
            n_estimators=args.trees,
            bagging_freq=1,
            bagging_fraction=0.8,
            feature_fraction=1.0,
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        )
        fit_seconds, _ = elapsed(lambda: model.fit(x, y))
        # Use all cores across features while keeping each prediction single-threaded.
        model.set_params(n_jobs=1)
        importance_seconds, result = elapsed(lambda: permutation_importance(
            model, validation_x, validation_y, scoring="f1", n_repeats=1,
            n_jobs=-1, random_state=seed,
        ))
        measured["lightgbm"].append((fit_seconds, importance_seconds))
        importances["lightgbm"].append(result.importances_mean)
        del model
        progress.advance(f"LightGBM RF measured, seed {seed_index + 1}/{args.repeats}")

    medians = {
        implementation: (
            statistics.median(sample[0] for sample in samples),
            statistics.median(sample[1] for sample in samples),
        )
        for implementation, samples in measured.items()
    }
    total_medians = {
        implementation: statistics.median(fit + importance for fit, importance in samples)
        for implementation, samples in measured.items()
    }
    ranked = {
        implementation: np.median(np.stack(values), axis=0)
        for implementation, values in importances.items()
    }
    return [
        {
            "implementation": implementation,
            "mode": "histogram_16" if implementation == "bankai" else "random_forest_boosting",
            "rows": args.rows,
            "validation_rows": len(validation_y),
            "features": args.features,
            "trees": args.trees,
            "repeats": args.repeats,
            "warmups_per_seed": args.warmups,
            "available_cores": cores,
            "estimator_fit_n_jobs": -1,
            "importance_n_jobs": 1 if implementation == "bankai" else -1,
            "fit_seconds": medians[implementation][0],
            "feature_importance_seconds": medians[implementation][1],
            "fit_plus_importance_seconds": total_medians[implementation],
            "importance_rank_correlation_vs_bankai": rank_correlation(
                ranked["bankai"], ranked[implementation]
            ),
        }
        for implementation in ("bankai", "lightgbm")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmarks/results-histogram-parallel-100k-100f"),
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.rows <= 100_000:
        parser.error("--rows must be between 1 and 100000")
    if args.warmups < 1 or args.repeats < 1:
        parser.error("--warmups and --repeats must be at least 1")

    cores = os.cpu_count() or 1
    if not args.skip_build:
        prepare_optimized_binaries(Path(__file__).parents[1])
    runs_per_implementation = args.repeats * (args.warmups + 1)
    progress = Progress({"Bankai": runs_per_implementation, "LightGBM": runs_per_implementation})
    x_all, y_all = generate_dataset(args.rows * 2, args.features)
    rows = benchmark(
        args, x_all[:args.rows], y_all[:args.rows],
        x_all[args.rows:], y_all[args.rows:], progress, cores,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "histogram_parallel.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.output_dir / "README.md"
    with report_path.open("w") as handle:
        handle.write("# Parallel Bankai histogram and LightGBM benchmark\n\n")
        handle.write(
            f"Workload: {args.rows:,} training rows and validation rows, {args.features} features, "
            f"{args.trees} trees, {args.repeats} measured seeds, and {args.warmups} warmup(s) "
            f"per seed. The machine reports {cores} logical CPUs. Execution took "
            f"{format_duration(time.monotonic() - progress.started)}, excluding release builds.\n\n"
        )
        handle.write(
            f"Environment: {sys.platform}, Python {sys.version.split()[0]}, NumPy {np.__version__}, "
            f"scikit-learn {sklearn.__version__}, LightGBM {LGBMClassifier.__module__.split('.')[0]} "
            f"{__import__('lightgbm').__version__}. The Bankai extension was built in release mode "
            "with `RUSTFLAGS=-C target-cpu=native`.\n\n"
        )
        handle.write(
            "Both estimators fit with `n_jobs=-1`. Bankai computes OOB permutation importance "
            "during fit. LightGBM fits multithreaded, then permutation importance evaluates "
            "features in parallel (`n_jobs=-1`) while each prediction uses one thread to avoid "
            "nested oversubscription. LightGBM uses RF boosting, 80% row bagging, all features "
            "per tree, and one repeat per feature for external permutation importance.\n\n"
        )
        handle.write(
            "| implementation | mode | fit s | importance s | fit + importance s | rank correlation vs Bankai |\n"
            "| --- | --- | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            handle.write(
                "| {implementation} | {mode} | {fit_seconds:.3f} | "
                "{feature_importance_seconds:.3f} | {fit_plus_importance_seconds:.3f} | "
                "{importance_rank_correlation_vs_bankai:.6f} |\n".format(**row)
            )
        handle.write(
            "\nRaw per-implementation measurements are in `histogram_parallel.csv`.\n"
        )
    print(f"Wrote {csv_path} and {report_path}")


if __name__ == "__main__":
    main()
