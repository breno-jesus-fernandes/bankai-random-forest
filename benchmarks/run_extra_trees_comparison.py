#!/usr/bin/env python3
"""Paired feature-selection and fit-time benchmark: Bankai vs LightGBM RF."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(rows: int, features: int, informative: int, strength: float, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, features)).astype(np.float32)
    weights = rng.choice(np.array([-strength, strength]), size=informative)
    logits = x[:, :informative] @ weights
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probabilities).astype(np.int64)
    indices = np.arange(rows)
    train_rows, test_rows = train_test_split(
        indices, test_size=0.25, random_state=9127, stratify=y
    )
    return x[train_rows], x[test_rows], y[train_rows], y[test_rows]


def make_model(variant: str, args, model_seed: int):
    if variant.startswith("bankai"):
        from bankai_random_forest import BankaiRandomForestClassifier

        return BankaiRandomForestClassifier(
            n_estimators=args.trees,
            criterion="gini",
            max_depth=None,
            max_leaf_nodes=args.leaves,
            min_samples_leaf=args.min_leaf,
            max_features=args.feature_fraction,
            bootstrap=True,
            max_samples=args.sample_fraction,
            random_state=model_seed,
            n_jobs=args.jobs,
            max_bins=args.bins,
            importance_type="gain",
            extra_trees=variant == "bankai_extra_trees",
            extra_seed=6,
        )

    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        boosting_type="rf",
        objective="binary",
        n_estimators=args.trees,
        learning_rate=1.0,
        max_depth=-1,
        num_leaves=args.leaves,
        min_child_samples=args.min_leaf,
        min_child_weight=0.0,
        min_split_gain=0.0,
        max_bin=args.bins,
        min_data_in_bin=1,
        bagging_fraction=args.sample_fraction,
        bagging_freq=1,
        feature_fraction=args.feature_fraction,
        extra_trees=True,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=model_seed,
        bagging_seed=model_seed,
        feature_fraction_seed=model_seed,
        data_random_seed=model_seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=args.jobs,
        importance_type="gain",
        verbosity=-1,
    )


def run_one(variant, x_train, x_test, y_train, y_test, args, data_seed, model_seed, warmup):
    model = make_model(variant, args, model_seed)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    chosen = np.argsort(-importance, kind="stable")[: args.informative]
    relevant = np.zeros(args.features, dtype=bool)
    relevant[: args.informative] = True
    selected_relevant = relevant[chosen]
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = probabilities >= 0.5
    return {
        "variant": variant,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "signal_strength": args.signal_strength,
        "warmup": warmup,
        "fit_seconds": fit_seconds,
        "recall_at_k": float(selected_relevant.sum() / args.informative),
        "false_discoveries_at_k": int((~selected_relevant).sum()),
        "feature_average_precision": float(average_precision_score(relevant, importance)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def summarize(records):
    measured = [row for row in records if not row["warmup"]]
    summary = {}
    labels = ["bankai", "bankai_extra_trees", "lightgbm_extra_trees"]
    for label in labels:
        own = [row for row in measured if row["variant"] == label]
        summary[label] = {
            metric: statistics.median(float(row[metric]) for row in own)
            for metric in (
                "fit_seconds",
                "recall_at_k",
                "false_discoveries_at_k",
                "feature_average_precision",
                "test_roc_auc",
                "test_recall",
            )
        }
    pairs = {}
    index = {(row["variant"], row["data_seed"], row["model_seed"]): row for row in measured}
    for first_index, first in enumerate(labels):
        for second in labels[first_index + 1:]:
            ratios = []
            deltas = {metric: [] for metric in (
                "recall_at_k", "false_discoveries_at_k", "feature_average_precision", "test_roc_auc"
            )}
            for data_seed, model_seed in sorted({(r["data_seed"], r["model_seed"]) for r in measured}):
                left = index[(first, data_seed, model_seed)]
                right = index[(second, data_seed, model_seed)]
                ratios.append(left["fit_seconds"] / right["fit_seconds"])
                for metric in deltas:
                    deltas[metric].append(left[metric] - right[metric])
            pairs[f"{first}_vs_{second}"] = {
                "first_variant": first,
                "second_variant": second,
                "paired_median_fit_time_ratio_first_over_second": float(statistics.median(ratios)),
                "paired_fit_time_ratios_first_over_second": ratios,
                "median_deltas_first_minus_second": {
                    metric: float(statistics.median(values)) for metric, values in deltas.items()
                },
            }
    return {"median_by_variant": summary, "paired_comparisons": pairs}


def write_markdown(path: Path, args, summary, metadata):
    metrics = summary["median_by_variant"]
    lines = [
        "# Bankai Extra Trees vs LightGBM Extra Trees",
        "",
        f"Rows={args.rows} ({int(args.rows * .75)} train / {args.rows - int(args.rows * .75)} test), "
        f"features={args.features}, informative={args.informative}, signal_strength={args.signal_strength}, "
        f"trees={args.trees}, "
        f"max_leaves/num_leaves={args.leaves}, bins={args.bins}, threads={args.jobs}.",
        "",
        "Paired generated binary datasets; each variant gets the same data and model seed. "
        "One warmup per variant and seed is discarded. Fit time includes estimator input processing "
        "and histogram construction, but excludes dataset generation and prediction.",
        "",
        "| Variant | Median fit (s) | Recall@K | False discoveries@K | Feature AP | Test ROC AUC | Test recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("bankai", "bankai_extra_trees", "lightgbm_extra_trees"):
        row = metrics[label]
        lines.append(
            f"| {label} | {row['fit_seconds']:.4f} | {row['recall_at_k']:.3f} | "
            f"{row['false_discoveries_at_k']:.1f} | {row['feature_average_precision']:.3f} | "
            f"{row['test_roc_auc']:.3f} | {row['test_recall']:.3f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for label, row in summary["paired_comparisons"].items():
        lines.append(
            f"- `{label}`: paired median fit-time ratio first/second="
            f"{row['paired_median_fit_time_ratio_first_over_second']:.3f} "
            f"(<1 means first model faster); "
            f"median Δ recall@K (first-second)="
            f"{row['median_deltas_first_minus_second']['recall_at_k']:+.3f}; "
            f"median Δ false discoveries (first-second)="
            f"{row['median_deltas_first_minus_second']['false_discoveries_at_k']:+.1f}."
        )
    lines.extend([
        "",
        "## Fairness and limits",
        "",
        "All variants use the same fixed train/test split, data/model seeds, tree count, leaf cap, "
        "minimum leaf count, 63-bin limit, 80% row fraction, 80% feature fraction, and gain importance. "
        "LightGBM RF requires active bagging and feature fraction below 1.0. Its bagging samples without "
        "replacement, while Bankai bootstraps with replacement; Bankai samples features per node while "
        "LightGBM feature_fraction samples per tree. LightGBM optimizes gradient statistics, while "
        "Bankai uses Gini impurity. Thus this is a matched-budget comparison, not identical algorithms.",
        "",
        "Feature recall uses the known first K informative columns; false discoveries count selected "
        "noise columns among the top K. The target is a noisy logistic function of weak, signed effects "
        "on those columns. Medians summarize the paired seed matrix; no significance claim "
        "is made. Results are local to this machine and should not be generalized.",
        "",
        f"Environment: Python {metadata['python']}, {metadata['platform']}, "
        f"NumPy {metadata['packages']['numpy']}, scikit-learn {metadata['packages']['scikit-learn']}, "
        f"LightGBM {metadata['packages']['lightgbm']}.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results-extra-trees-local"))
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--features", type=int, default=200)
    parser.add_argument("--informative", type=int, default=20)
    parser.add_argument("--signal-strength", type=float, default=0.12)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--bins", type=int, default=63)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[1729, 1730, 1731])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[6, 42, 43])
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if not (2 <= args.informative < args.features and args.bins >= 2 and args.trees >= 1):
        parser.error("require 2 <= informative < features, bins >= 2, and trees >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = ["bankai", "bankai_extra_trees", "lightgbm_extra_trees"]
    records = []
    for data_seed in args.data_seeds:
        x_train, x_test, y_train, y_test = build_dataset(
            args.rows, args.features, args.informative, args.signal_strength, data_seed
        )
        for model_seed in args.model_seeds:
            order = variants[(data_seed + model_seed) % len(variants):] + variants[: (data_seed + model_seed) % len(variants)]
            for warmup in [True] * args.warmups + [False]:
                for variant in order:
                    record = run_one(
                        variant, x_train, x_test, y_train, y_test, args,
                        data_seed, model_seed, warmup,
                    )
                    records.append(record)
                    print(
                        f"{'warmup ' if warmup else ''}{data_seed=} {model_seed=} "
                        f"{variant}: {record['fit_seconds']:.3f}s, "
                        f"recall@K={record['recall_at_k']:.3f}, "
                        f"false_discoveries={record['false_discoveries_at_k']}",
                        flush=True,
                    )
    csv_path = args.output_dir / "raw.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": __import__("os").cpu_count(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scikit-learn", "lightgbm", "bankai-random-forest")
        },
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "bankai_extension": __import__("bankai_random_forest._core", fromlist=["__file__"]).__file__,
    }
    metadata["bankai_extension_sha256"] = sha256(Path(metadata["bankai_extension"]))
    (args.output_dir / "summary.json").write_text(json.dumps({
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "environment": metadata,
        "summary": summary,
        "raw_records": len(records),
    }, indent=2) + "\n")
    write_markdown(args.output_dir / "report.md", args, summary, metadata)
    print(f"Wrote {csv_path}, {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
