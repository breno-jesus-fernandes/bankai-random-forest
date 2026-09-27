#!/usr/bin/env python3
"""Compare mean-absolute TreeSHAP importances for Bankai and LightGBM."""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import shap
import sklearn
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmarks.run_benchmark import generate_dataset, prepare_optimized_binaries
from benchmarks.run_histogram_importance_benchmark import Progress, elapsed, format_duration


def rank_correlation(left, right):
    # Average ranks are important when LightGBM assigns exact zero SHAP to
    # unused features; assigning arbitrary distinct ranks distorts Spearman rho.
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def positive_class_values(values, implementation):
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError(f"Expected binary classification SHAP output, got {len(values)} classes")
        return np.asarray(values[1], dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        return values[:, :, 1]
    if values.ndim == 2:
        # SHAP's LightGBM binary path returns only the positive-class output.
        return values
    raise ValueError(f"Unexpected {implementation} SHAP output shape: {values.shape}")


def positive_class_base(expected_value, implementation):
    expected_value = np.asarray(expected_value, dtype=np.float64)
    if expected_value.ndim == 0:
        return float(expected_value)
    if expected_value.size == 2:
        return float(expected_value.reshape(-1)[1])
    raise ValueError(f"Unexpected {implementation} SHAP baseline shape: {expected_value.shape}")


def explain_one(model, implementation, explain, background, progress, chunk_rows, label):
    if implementation == "bankai":
        model._shap_estimators_cache = None
        export_seconds, estimators = elapsed(lambda: model.estimators_)
        del estimators
        model_output = "probability"
        perturbation = "interventional"
    else:
        export_seconds = 0.0
        # SHAP 0.51 currently fails additivity for LightGBM RF probabilities.
        # Keep a valid, explicitly reported raw-margin fallback for timing.
        model_output = "probability"
        perturbation = "interventional"
    if implementation == "bankai":
        progress.advance(f"{label} tree export")

    explainer_seconds, explainer = elapsed(lambda: shap.TreeExplainer(
        model, data=background, model_output=model_output,
        feature_perturbation=perturbation,
    ))
    progress.advance(f"{label} explainer")

    absolute_sum = np.zeros(explain.shape[1], dtype=np.float64)
    class_values = []
    calculation_seconds = 0.0
    for start in range(0, len(explain), chunk_rows):
        batch = explain[start:start + chunk_rows]
        batch_seconds, raw = elapsed(lambda: explainer.shap_values(batch))
        values = positive_class_values(raw, implementation)
        absolute_sum += np.abs(values).sum(axis=0)
        class_values.append(values)
        calculation_seconds += batch_seconds
        progress.advance(f"{label} SHAP rows {min(start + len(batch), len(explain))}/{len(explain)}")

    positive_values = np.concatenate(class_values, axis=0)
    base = positive_class_base(explainer.expected_value, implementation)
    predictions = model.predict_proba(explain)[:, 1]
    additivity_error = float(np.max(np.abs(positive_values.sum(axis=1) + base - predictions)))
    probability_additivity_error = additivity_error
    output_scale = "positive_class_probability"
    fallback_probe_seconds = 0.0
    if implementation == "lightgbm" and additivity_error > 1e-4:
        # Do not silently compare invalid probability explanations. Retry in
        # LightGBM's supported additive raw-margin scale and record that scale.
        probability_probe_seconds = explainer_seconds + calculation_seconds
        fallback_probe_seconds = probability_probe_seconds
        del explainer, positive_values, class_values
        model_output = "raw"
        raw_explainer_seconds, explainer = elapsed(lambda: shap.TreeExplainer(
            model, data=background, model_output=model_output,
            feature_perturbation=perturbation,
        ))
        explainer_seconds = raw_explainer_seconds
        absolute_sum = np.zeros(explain.shape[1], dtype=np.float64)
        class_values = []
        calculation_seconds = 0.0
        for start in range(0, len(explain), chunk_rows):
            batch = explain[start:start + chunk_rows]
            batch_seconds, raw = elapsed(lambda: explainer.shap_values(batch))
            values = positive_class_values(raw, implementation)
            absolute_sum += np.abs(values).sum(axis=0)
            class_values.append(values)
            calculation_seconds += batch_seconds
            progress.advance(f"{label} raw-margin SHAP rows {min(start + len(batch), len(explain))}/{len(explain)}")
        progress.advance(f"{label} raw-margin explainer")
        positive_values = np.concatenate(class_values, axis=0)
        base = positive_class_base(explainer.expected_value, implementation)
        predictions = model.predict(explain, raw_score=True)
        additivity_error = float(np.max(np.abs(positive_values.sum(axis=1) + base - predictions)))
        output_scale = "raw_margin (probability unsupported: additivity failed)"
    return {
        "tree_export_seconds": export_seconds,
        "explainer_seconds": explainer_seconds,
        "calculation_seconds": calculation_seconds,
        "explanation_total_seconds": export_seconds + explainer_seconds + calculation_seconds + fallback_probe_seconds,
        "unsupported_probability_probe_seconds": fallback_probe_seconds,
        "mean_abs_shap": absolute_sum / len(explain),
        "additivity_max_error": additivity_error,
        "probability_additivity_max_error": probability_additivity_error,
        "output_scale": output_scale,
    }


def make_model(implementation, trees, seed, max_leaf_nodes):
    if implementation == "bankai":
        from bankai_random_forest import BankaiRandomForestClassifier
        return BankaiRandomForestClassifier(
            n_estimators=trees,
            max_bins=16,
            max_leaf_nodes=max_leaf_nodes,
            random_state=seed,
            n_jobs=-1,
        )
    return LGBMClassifier(
        boosting_type="rf",
        n_estimators=trees,
        bagging_freq=1,
        bagging_fraction=0.8,
        feature_fraction=1.0,
        num_leaves=max_leaf_nodes or 31,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )


def tree_topology(model, implementation):
    if implementation == "bankai":
        nodes = [int(est.tree_.node_count) for est in model.estimators_]
        leaves = [(count + 1) // 2 for count in nodes]
    else:
        leaves = [int(tree["num_leaves"]) for tree in model.booster_.dump_model()["tree_info"]]
        nodes = [2 * count - 1 for count in leaves]
    return float(np.mean(nodes)), max(nodes), float(np.mean(leaves))


def benchmark(args, x, y, explain, background, validation_x, validation_y, progress, cores):
    raw_rows = []
    importance_by_model = {"bankai": [], "lightgbm": []}
    for implementation in ("bankai", "lightgbm"):
        for seed_index in range(args.repeats):
            seed = 42 + seed_index
            unbounded_metrics = None
            if implementation == "bankai" and args.max_leaf_nodes:
                from bankai_random_forest import BankaiRandomForestClassifier
                for warmup_index in range(args.warmups):
                    reference_warmup = BankaiRandomForestClassifier(
                        n_estimators=args.trees, max_bins=16, random_state=seed, n_jobs=-1
                    ).fit(x, y)
                    reference_warmup.predict_proba(validation_x)
                    progress.advance(f"bankai unbounded quality warmup {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}")
                    del reference_warmup
                reference = BankaiRandomForestClassifier(
                    n_estimators=args.trees, max_bins=16, random_state=seed, n_jobs=-1
                ).fit(x, y)
                ref_probabilities = reference.predict_proba(validation_x)[:, 1]
                ref_predictions = (ref_probabilities >= 0.5).astype(np.int8)
                unbounded_metrics = (
                    float(accuracy_score(validation_y, ref_predictions)),
                    float(f1_score(validation_y, ref_predictions)),
                )
                progress.advance(f"bankai unbounded quality reference, seed {seed_index + 1}/{args.repeats}")
                del reference
            for warmup_index in range(args.warmups):
                model = make_model(implementation, args.trees, seed, args.max_leaf_nodes)
                model.fit(x, y)
                progress.advance(f"{implementation} warmup fit {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}")
                explain_one(
                    model, implementation, explain, background, progress,
                    args.chunk_rows,
                    f"{implementation} warmup {warmup_index + 1}/{args.warmups}, seed {seed_index + 1}/{args.repeats}",
                )
                del model

            model = make_model(implementation, args.trees, seed, args.max_leaf_nodes)
            fit_seconds, _ = elapsed(lambda: model.fit(x, y))
            probabilities = model.predict_proba(validation_x)[:, 1]
            predictions = (probabilities >= 0.5).astype(np.int8)
            mean_nodes, max_nodes, mean_leaves = tree_topology(model, implementation)
            progress.advance(f"{implementation} measured fit, seed {seed_index + 1}/{args.repeats}")
            result = explain_one(
                model, implementation, explain, background, progress,
                args.chunk_rows,
                f"{implementation} measured, seed {seed_index + 1}/{args.repeats}",
            )
            importance_by_model[implementation].append(result.pop("mean_abs_shap"))
            raw_rows.append({
                "implementation": implementation,
                "mode": "histogram_16" if implementation == "bankai" else "random_forest_boosting",
                "seed": seed,
                "fit_seconds": fit_seconds,
                "tree_export_seconds": result["tree_export_seconds"],
                "explainer_seconds": result["explainer_seconds"],
                "shap_calculation_seconds": result["calculation_seconds"],
                "fit_plus_shap_seconds": fit_seconds + result["explanation_total_seconds"],
                "additivity_max_error": result["additivity_max_error"],
                "probability_additivity_max_error": result["probability_additivity_max_error"],
                "unsupported_probability_probe_seconds": result["unsupported_probability_probe_seconds"],
                "shap_output_scale": result["output_scale"],
                "validation_accuracy": float(accuracy_score(validation_y, predictions)),
                "validation_f1": float(f1_score(validation_y, predictions)),
                "unbounded_bankai_accuracy": unbounded_metrics[0] if unbounded_metrics else "",
                "unbounded_bankai_f1": unbounded_metrics[1] if unbounded_metrics else "",
                "agreement_with_unbounded_bankai": (
                    float(np.mean(predictions == ref_predictions)) if unbounded_metrics else ""
                ),
                "mean_tree_nodes": mean_nodes,
                "max_tree_nodes": max_nodes,
                "mean_tree_leaves": mean_leaves,
                "rows": args.rows,
                "explained_rows": len(explain),
                "background_rows": len(background),
                "features": args.features,
                "trees": args.trees,
                "available_cores": cores,
                "estimator_fit_n_jobs": -1,
                "max_leaf_nodes": args.max_leaf_nodes or "unlimited",
            })
            del model

    median_importances = {
        implementation: np.median(np.stack(values), axis=0)
        for implementation, values in importance_by_model.items()
    }
    same_scale = all(row["shap_output_scale"] == "positive_class_probability" for row in raw_rows)
    correlation = (
        rank_correlation(median_importances["bankai"], median_importances["lightgbm"])
        if same_scale else "not comparable: different output scales"
    )
    for row in raw_rows:
        row["importance_rank_correlation_vs_other_model"] = correlation

    summary = []
    for implementation in ("bankai", "lightgbm"):
        matches = [row for row in raw_rows if row["implementation"] == implementation]
        fields = (
            "fit_seconds", "tree_export_seconds", "explainer_seconds",
            "shap_calculation_seconds", "fit_plus_shap_seconds",
        )
        record = {
            "implementation": implementation,
            "mode": matches[0]["mode"],
            "repeats": args.repeats,
            "warmups_per_seed": args.warmups,
            "rows": args.rows,
            "explained_rows": len(explain),
            "background_rows": len(background),
            "features": args.features,
            "trees": args.trees,
            "available_cores": cores,
            "importance_rank_correlation_vs_other_model": correlation,
        }
        record.update({field: statistics.median(row[field] for row in matches) for field in fields})
        record["additivity_max_error"] = max(row["additivity_max_error"] for row in matches)
        record["probability_additivity_max_error"] = max(row["probability_additivity_max_error"] for row in matches)
        record["validation_accuracy"] = statistics.median(row["validation_accuracy"] for row in matches)
        record["validation_f1"] = statistics.median(row["validation_f1"] for row in matches)
        record["mean_tree_nodes"] = statistics.median(row["mean_tree_nodes"] for row in matches)
        record["max_tree_nodes"] = max(row["max_tree_nodes"] for row in matches)
        record["mean_tree_leaves"] = statistics.median(row["mean_tree_leaves"] for row in matches)
        for field in ("unbounded_bankai_accuracy", "unbounded_bankai_f1", "agreement_with_unbounded_bankai"):
            values = [row[field] for row in matches if row[field] != ""]
            record[field] = statistics.median(values) if values else ""
        if implementation == "bankai" and record["unbounded_bankai_accuracy"] != "":
            record["accuracy_delta_vs_unbounded"] = record["validation_accuracy"] - record["unbounded_bankai_accuracy"]
            record["f1_delta_vs_unbounded"] = record["validation_f1"] - record["unbounded_bankai_f1"]
        else:
            record["accuracy_delta_vs_unbounded"] = ""
            record["f1_delta_vs_unbounded"] = ""
        summary.append(record)
    return raw_rows, summary, median_importances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-leaf-nodes", type=int, default=31,
                        help="Capacity-matched Bankai limit; LightGBM uses the same num_leaves")
    parser.add_argument("--explain-rows", type=int, default=1_000)
    parser.add_argument("--background-rows", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmarks/results-shap-feature-importance-100k-100f"),
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.rows > 100_000 or args.rows < args.explain_rows:
        parser.error("--rows must be <=100000 and >= --explain-rows")
    if min(args.repeats, args.warmups, args.chunk_rows, args.background_rows) < 1:
        parser.error("repeats, warmups, chunk_rows, and background_rows must be positive")
    if args.max_leaf_nodes < 1:
        parser.error("--max-leaf-nodes must be positive")

    cores = os.cpu_count() or 1
    if not args.skip_build:
        prepare_optimized_binaries(Path(__file__).parents[1])
    chunks = (args.explain_rows + args.chunk_rows - 1) // args.chunk_rows
    passes = args.repeats * (args.warmups + 1)
    progress = Progress({
        "bankai": passes * (3 + chunks) + args.repeats * (args.warmups + 1),  # reference fits also report progress
        "lightgbm": passes * (3 + 2 * chunks),  # includes fallback explainer step
    })

    all_x, all_y = generate_dataset(args.rows * 2, args.features)
    train_x, train_y = all_x[:args.rows], all_y[:args.rows]
    validation_x, validation_y = all_x[args.rows:], all_y[args.rows:]
    sample_rng = np.random.RandomState(2026)
    explain_indices = np.sort(sample_rng.choice(len(validation_y), args.explain_rows, replace=False))
    background_indices = np.sort(sample_rng.choice(args.rows, args.background_rows, replace=False))
    explain = validation_x[explain_indices]
    background = train_x[background_indices]

    raw_rows, summary, importances = benchmark(
        args, train_x, train_y, explain, background, validation_x, validation_y, progress, cores
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "shap_feature_importance_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw_rows)
    with (args.output_dir / "shap_feature_importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "shap_values_mean_abs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow((
            "feature_index", "bankai_mean_abs_shap_probability",
            "bankai_importance_share", "lightgbm_mean_abs_shap_model_output",
            "lightgbm_importance_share",
        ))
        bankai_total = float(importances["bankai"].sum())
        lightgbm_total = float(importances["lightgbm"].sum())
        for index, (bankai_value, lightgbm_value) in enumerate(zip(
            importances["bankai"], importances["lightgbm"], strict=True
        )):
            writer.writerow((index, bankai_value, bankai_value / bankai_total,
                             lightgbm_value, lightgbm_value / lightgbm_total))

    with (args.output_dir / "README.md").open("w") as handle:
        handle.write("# SHAP feature importance: Bankai vs LightGBM\n\n")
        handle.write(
            f"Dataset: {args.rows:,} training rows, {args.features} features, {args.trees} trees. "
            f"Global importance is mean absolute positive-class SHAP over a fixed random sample "
            f"of {args.explain_rows:,} validation rows; both use the same "
            f"{args.background_rows}-row training background. "
            f"There were {args.repeats} measured seeds and {args.warmups} full warmup(s) per seed. "
            f"The host reports {cores} logical CPUs. Run time (excluding release builds): "
            f"{format_duration(time.monotonic() - progress.started)}.\n\n"
        )
        handle.write(
            f"Environment: macOS {__import__('platform').mac_ver()[0]}, machine "
            f"{__import__('platform').machine()}, Python {sys.version.split()[0]}, "
            f"NumPy {np.__version__}, SHAP {shap.__version__}, sklearn {sklearn.__version__}, "
            f"LightGBM {__import__('lightgbm').__version__}. Bankai extension built in release "
            "mode with `RUSTFLAGS=-C target-cpu=native`. Both model fits use `n_jobs=-1`.\n\n"
        )
        handle.write(
            f"Both models use exact interventional TreeSHAP and the same background. Bankai uses max_leaf_nodes={args.max_leaf_nodes}; "
            f"LightGBM uses num_leaves={args.max_leaf_nodes}. Bankai uses SHAP on positive-class "
            "probability with the recorded training background. A probability-scale LightGBM explanation was attempted first; "
            "it is retained only when additivity error is <=1e-4. In this environment SHAP 0.51's LightGBM RF path fails that "
            "check, so its timing and values use the explicitly labeled raw-margin fallback. The two importance scales are "
            "not directly comparable; no cross-model rank correlation should be interpreted. Validation accuracy and F1 are "
            "reported for the matched-capacity models. "
            "Bankai tree-export time is reported separately and included in fit+SHAP total.\n\n"
        )
        handle.write(
            "| model | fit s | tree export s | explainer s | SHAP s | fit + SHAP s | "
            "mean nodes/tree | accuracy | F1 | max additivity error | SHAP rank corr vs other |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in summary:
            handle.write(
                "| {mode} | {fit_seconds:.3f} | {tree_export_seconds:.3f} | "
                "{explainer_seconds:.3f} | {shap_calculation_seconds:.3f} | "
                "{fit_plus_shap_seconds:.3f} | {mean_tree_nodes:.1f} | "
                "{validation_accuracy:.5f} | {validation_f1:.5f} | {additivity_max_error:.3e} | "
                "{importance_rank_correlation_vs_other_model} |\n".format(**row)
            )
        handle.write(
            "\nRaw seed-level measurements are in `shap_feature_importance_raw.csv`; "
            "median summary is `shap_feature_importance.csv`; per-feature mean absolute "
            "values are in `shap_values_mean_abs.csv`.\n"
        )
        bankai_summary = next(row for row in summary if row["implementation"] == "bankai")
        handle.write(
            f"\nThe capped Bankai model's median validation accuracy/F1 changed by "
            f"{bankai_summary['accuracy_delta_vs_unbounded']:+.5f}/"
            f"{bankai_summary['f1_delta_vs_unbounded']:+.5f} versus the same-seed, "
            f"unbounded Bankai reference; prediction agreement was "
            f"{bankai_summary['agreement_with_unbounded_bankai']:.5f}. The reference "
            "quality fits also had one warmup per seed. See raw CSV for all seeds, "
            "tree topology, and failed LightGBM probability additivity values.\n"
        )
    print(f"Wrote SHAP feature-importance reports to {args.output_dir}")


if __name__ == "__main__":
    main()
