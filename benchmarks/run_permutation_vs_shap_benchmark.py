#!/usr/bin/env python3
"""Compare Bankai OOB permutation importance with LightGBM TreeSHAP."""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import shap
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmarks.run_benchmark import generate_dataset, prepare_optimized_binaries
from benchmarks.run_histogram_importance_benchmark import Progress, elapsed, format_duration
from benchmarks.run_shap_feature_importance_benchmark import (
    positive_class_base,
    positive_class_values,
)


def rank_correlation(left, right):
    left_ranks = rankdata(left, method="average")
    right_ranks = rankdata(right, method="average")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def explain_lightgbm(model, explain, background, chunk_rows, progress, label):
    explainer_seconds, explainer = elapsed(lambda: shap.TreeExplainer(
        model,
        data=background,
        model_output="raw",
        feature_perturbation="interventional",
    ))
    progress.advance(f"lightgbm {label} TreeExplainer")

    absolute_sum = np.zeros(explain.shape[1], dtype=np.float64)
    values_by_chunk = []
    calculation_seconds = 0.0
    for start in range(0, len(explain), chunk_rows):
        batch = explain[start:start + chunk_rows]
        seconds, raw_values = elapsed(lambda: explainer.shap_values(batch))
        values = positive_class_values(raw_values, "lightgbm")
        values_by_chunk.append(values)
        absolute_sum += np.abs(values).sum(axis=0)
        calculation_seconds += seconds
        progress.advance(
            f"lightgbm {label} SHAP {min(start + len(batch), len(explain))}/{len(explain)}"
        )

    all_values = np.concatenate(values_by_chunk, axis=0)
    base_value = positive_class_base(explainer.expected_value, "lightgbm")
    raw_predictions = model.predict(explain, raw_score=True)
    additivity_error = float(np.max(np.abs(all_values.sum(axis=1) + base_value - raw_predictions)))
    return {
        "explainer_seconds": explainer_seconds,
        "shap_seconds": calculation_seconds,
        "total_seconds": explainer_seconds + calculation_seconds,
        "mean_abs_shap": absolute_sum / len(explain),
        "additivity_max_error": additivity_error,
    }


def run(args, x, y, validation_x, validation_y, background, explain, progress):
    from bankai_random_forest import BankaiRandomForestClassifier

    bankai_rows = []
    lightgbm_rows = []
    bankai_importances = []
    lightgbm_importances = []

    for seed_index in range(args.repeats):
        seed = 42 + seed_index
        for warmup_index in range(args.warmups):
            warmup = BankaiRandomForestClassifier(
                n_estimators=args.trees,
                max_bins=args.bins,
                importance_type="permutation",
                random_state=seed,
                n_jobs=-1,
            ).fit(x, y)
            _ = warmup.feature_importances_
            warmup.predict_proba(validation_x)
            progress.advance(
                f"Bankai warmup {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}"
            )
            del warmup

        model = BankaiRandomForestClassifier(
            n_estimators=args.trees,
            max_bins=args.bins,
            importance_type="permutation",
            random_state=seed,
            n_jobs=-1,
        )
        fit_seconds, _ = elapsed(lambda: model.fit(x, y))
        importance_seconds, importance = elapsed(lambda: np.asarray(model.feature_importances_, dtype=np.float64))
        predict_seconds, probabilities = elapsed(lambda: model.predict_proba(validation_x)[:, 1])
        predictions = (probabilities >= 0.5).astype(np.int8)
        bankai_rows.append({
            "implementation": "bankai",
            "mode": f"histogram_{args.bins}",
            "seed": seed,
            "fit_seconds": fit_seconds,
            "importance_setup_seconds": 0.0,
            "importance_calculation_seconds": 0.0,
            "importance_access_seconds": importance_seconds,
            "fit_plus_importance_seconds": fit_seconds + importance_seconds,
            "validation_predict_seconds": predict_seconds,
            "validation_accuracy": float(accuracy_score(validation_y, predictions)),
            "validation_f1": float(f1_score(validation_y, predictions)),
            "additivity_max_error": "",
        })
        bankai_importances.append(importance)
        progress.advance(f"Bankai measured, seed {seed_index + 1}/{args.repeats}")
        del model

    model_factory = lambda seed: LGBMClassifier(
        boosting_type="rf",
        n_estimators=args.trees,
        bagging_freq=1,
        bagging_fraction=0.8,
        feature_fraction=1.0,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )
    for seed_index in range(args.repeats):
        seed = 42 + seed_index
        for warmup_index in range(args.warmups):
            warmup_model = model_factory(seed).fit(x, y)
            explain_lightgbm(
                warmup_model, explain, background, args.chunk_rows, progress,
                f"warmup {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}",
            )
            warmup_model.predict_proba(validation_x)
            progress.advance(
                f"lightgbm warmup fit/predict {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}"
            )
            del warmup_model

        model = model_factory(seed)
        fit_seconds, _ = elapsed(lambda: model.fit(x, y))
        result = explain_lightgbm(
            model, explain, background, args.chunk_rows, progress,
            f"measured seed {seed_index + 1}/{args.repeats}",
        )
        predict_seconds, probabilities = elapsed(lambda: model.predict_proba(validation_x)[:, 1])
        predictions = (probabilities >= 0.5).astype(np.int8)
        lightgbm_rows.append({
            "implementation": "lightgbm",
            "mode": "random_forest_boosting_tree_shap_raw_margin",
            "seed": seed,
            "fit_seconds": fit_seconds,
            "importance_setup_seconds": result["explainer_seconds"],
            "importance_calculation_seconds": result["shap_seconds"],
            "importance_access_seconds": 0.0,
            "fit_plus_importance_seconds": fit_seconds + result["total_seconds"],
            "validation_predict_seconds": predict_seconds,
            "validation_accuracy": float(accuracy_score(validation_y, predictions)),
            "validation_f1": float(f1_score(validation_y, predictions)),
            "additivity_max_error": result["additivity_max_error"],
        })
        lightgbm_importances.append(result["mean_abs_shap"])
        progress.advance(f"lightgbm measured fit/predict, seed {seed_index + 1}/{args.repeats}")
        del model

    bankai_median = np.median(np.stack(bankai_importances), axis=0)
    lightgbm_median = np.median(np.stack(lightgbm_importances), axis=0)
    correlation = rank_correlation(bankai_median, lightgbm_median)
    for row in bankai_rows + lightgbm_rows:
        row["importance_rank_correlation_vs_other_method"] = correlation
        row["rows"] = args.rows
        row["validation_rows"] = len(validation_y)
        row["explained_rows"] = len(explain)
        row["background_rows"] = len(background)
        row["features"] = args.features
        row["trees"] = args.trees
        row["available_cores"] = os.cpu_count() or 1
        row["fit_n_jobs"] = -1
        row["warmups_per_seed"] = args.warmups
        row["repeats"] = args.repeats

    measurements = bankai_rows + lightgbm_rows
    summary = []
    for implementation in ("bankai", "lightgbm"):
        rows = [row for row in measurements if row["implementation"] == implementation]
        item = {
            "implementation": implementation,
            "mode": rows[0]["mode"],
            "fit_seconds": statistics.median(row["fit_seconds"] for row in rows),
            "importance_setup_seconds": statistics.median(row["importance_setup_seconds"] for row in rows),
            "importance_calculation_seconds": statistics.median(row["importance_calculation_seconds"] for row in rows),
            "importance_access_seconds": statistics.median(row["importance_access_seconds"] for row in rows),
            "fit_plus_importance_seconds": statistics.median(row["fit_plus_importance_seconds"] for row in rows),
            "validation_predict_seconds": statistics.median(row["validation_predict_seconds"] for row in rows),
            "validation_accuracy": statistics.median(row["validation_accuracy"] for row in rows),
            "validation_f1": statistics.median(row["validation_f1"] for row in rows),
            "additivity_max_error": max(float(row["additivity_max_error"] or 0.0) for row in rows),
            "importance_rank_correlation_vs_other_method": correlation,
        }
        summary.append(item)
    return measurements, summary, bankai_median, lightgbm_median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--explain-rows", type=int, default=1_000)
    parser.add_argument("--background-rows", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmarks/results-permutation-vs-shap-100k-100f"),
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.rows > 100_000 or args.rows < args.explain_rows:
        parser.error("--rows must be <=100000 and >= --explain-rows")
    if min(args.features, args.trees, args.bins, args.repeats, args.warmups,
           args.background_rows, args.chunk_rows) < 1:
        parser.error("features, trees, bins, repeats, warmups, background, and chunk sizes must be positive")

    if not args.skip_build:
        prepare_optimized_binaries(Path(__file__).parents[1])
    cores = os.cpu_count() or 1
    chunks = (args.explain_rows + args.chunk_rows - 1) // args.chunk_rows
    passes = args.repeats * (args.warmups + 1)
    progress = Progress({
        "Bankai": passes,
        "lightgbm": passes * (2 + chunks),
    })

    all_x, all_y = generate_dataset(args.rows * 2, args.features)
    x, y = all_x[:args.rows], all_y[:args.rows]
    validation_x, validation_y = all_x[args.rows:], all_y[args.rows:]
    rng = np.random.RandomState(2026)
    explain_rows = np.sort(rng.choice(len(validation_y), args.explain_rows, replace=False))
    background_rows = np.sort(rng.choice(args.rows, args.background_rows, replace=False))
    explain = validation_x[explain_rows]
    background = x[background_rows]
    measurements, summary, bankai_values, lightgbm_values = run(
        args, x, y, validation_x, validation_y, background, explain, progress
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "permutation_vs_shap_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(measurements)
    with (args.output_dir / "permutation_vs_shap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    bankai_ranks = rankdata(-bankai_values, method="average")
    lightgbm_ranks = rankdata(-lightgbm_values, method="average")
    with (args.output_dir / "feature_importance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("feature_index", "bankai_oob_permutation_accuracy_drop",
                         "bankai_rank", "lightgbm_mean_abs_shap_raw_margin",
                         "lightgbm_rank"))
        for index, values in enumerate(zip(bankai_values, bankai_ranks,
                                           lightgbm_values, lightgbm_ranks, strict=True)):
            writer.writerow((index, *values))

    with (args.output_dir / "README.md").open("w") as handle:
        handle.write("# Bankai permutation vs LightGBM SHAP feature importance\n\n")
        handle.write(
            f"Workload: {args.rows:,} training rows and {len(validation_y):,} validation rows, "
            f"{args.features} features, {args.trees} trees, Bankai histogram_{args.bins}; "
            f"{args.repeats} measured seeds plus {args.warmups} full warmup(s) per seed. "
            f"Both model fits use all {cores} reported logical CPU cores. The fixed SHAP sample has "
            f"{args.explain_rows:,} validation rows and the background has {args.background_rows} training rows. "
            f"Execution took {format_duration(time.monotonic() - progress.started)}, excluding release builds.\n\n"
        )
        handle.write(
            f"Environment: macOS {__import__('platform').mac_ver()[0] or sys.platform} on "
            f"{__import__('platform').machine()}, Python {sys.version.split()[0]}, NumPy {np.__version__}, "
            f"SHAP {shap.__version__}, scikit-learn {__import__('sklearn').__version__}, "
            f"LightGBM {__import__('lightgbm').__version__}. Bankai extension built in release mode "
            "with `RUSTFLAGS=-C target-cpu=native`.\n\n"
        )
        handle.write(
            "Bankai's `importance_type='permutation'` is native out-of-bag accuracy decrease "
            "computed during fit; property access is timed separately. LightGBM uses RF boosting "
            "(80% row bagging, all features) and direct interventional TreeSHAP; its importance is "
            "mean absolute positive-class SHAP on raw margin because SHAP 0.51's probability path "
            "for this RF model fails additivity against `predict_proba`. Additivity is validated "
            "against `predict(raw_score=True)`. These methods and units differ, so compare runtime "
            "and use rank correlation as a descriptive agreement measure only; raw importance "
            "magnitudes are not comparable. Bankai's permutation work is included in fit; the "
            "comparable total is `fit_plus_importance_seconds`.\n\n"
        )
        handle.write("| implementation | fit s | importance setup s | importance calculation/access s | fit + importance s | validation accuracy | validation F1 | max additivity error | rank corr |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary:
            import_seconds = row["importance_calculation_seconds"] or row["importance_access_seconds"]
            handle.write(
                f"| {row['mode']} | {row['fit_seconds']:.3f} | "
                f"{row['importance_setup_seconds']:.3f} | {import_seconds:.3f} | "
                f"{row['fit_plus_importance_seconds']:.3f} | {row['validation_accuracy']:.5f} | "
                f"{row['validation_f1']:.5f} | {row['additivity_max_error']:.3e} | "
                f"{row['importance_rank_correlation_vs_other_method']:.6f} |\n"
            )
        handle.write(
            "\nRaw seed measurements are in `permutation_vs_shap_raw.csv`; summary medians are in "
            "`permutation_vs_shap.csv`; per-feature importances and average ranks are in "
            "`feature_importance.csv`.\n"
        )
    print(f"Wrote Bankai permutation vs LightGBM SHAP reports to {args.output_dir}")


if __name__ == "__main__":
    main()
