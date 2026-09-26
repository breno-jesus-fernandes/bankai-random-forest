#!/usr/bin/env python3
"""Measure fit plus permutation-importance costs across histogram modes."""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmarks.run_benchmark import generate_dataset, prepare_optimized_binaries


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


class Progress:
    """stderr progress bar with an ETA adapted to each implementation."""

    def __init__(self, group_totals):
        self.group_totals = dict(group_totals)
        self.group_remaining = dict(group_totals)
        self.group_elapsed = {group: 0.0 for group in group_totals}
        self.group_completed = {group: 0 for group in group_totals}
        self.total = sum(group_totals.values())
        self.completed = 0
        self.started = time.monotonic()
        self.last_update = self.started

    def advance(self, label):
        now = time.monotonic()
        group = label.split(maxsplit=1)[0]
        self.group_elapsed[group] += now - self.last_update
        self.group_completed[group] += 1
        self.group_remaining[group] -= 1
        self.completed += 1
        self.last_update = now
        elapsed = now - self.started
        fraction = self.completed / self.total
        overall_average = elapsed / self.completed
        external_groups = ("sklearn", "lightgbm")
        known_external = [
            self.group_elapsed[name] / self.group_completed[name]
            for name in external_groups if self.group_completed[name]
        ]
        eta = 0.0
        for name, remaining in self.group_remaining.items():
            if not remaining:
                continue
            completed = self.group_completed[name]
            if completed:
                average = self.group_elapsed[name] / completed
            elif name in external_groups and known_external:
                average = sum(known_external) / len(known_external)
            else:
                average = overall_average
            eta += average * remaining
        width = 28
        filled = int(width * fraction)
        bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
        sys.stderr.write(
            f"\r[{bar}] {self.completed}/{self.total} ({fraction:5.1%}) "
            f"elapsed {format_duration(elapsed)} ETA {format_duration(eta)} | {label[:48]:48}"
        )
        if self.completed == self.total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def elapsed(call):
    started = time.perf_counter()
    result = call()
    return time.perf_counter() - started, result


def rank_correlation(left, right):
    left_rank = np.empty(left.size, dtype=np.float64)
    right_rank = np.empty(right.size, dtype=np.float64)
    left_rank[np.argsort(left, kind="stable")] = np.arange(left.size)
    right_rank[np.argsort(right, kind="stable")] = np.arange(right.size)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def benchmark(rows, features, trees, bins, repeats, warmups, x, y, validation_x, validation_y, progress):
    from bankai_random_forest import BankaiRandomForestClassifier

    records = []
    for label, max_bins in [("exact", None), *((f"histogram_{n}", n) for n in bins)]:
        fit_samples, access_samples, importance_values = [], [], []
        for seed in range(repeats):
            random_state = 42 + seed
            for warmup in range(warmups):
                warmup_model = BankaiRandomForestClassifier(
                    n_estimators=trees,
                    max_bins=max_bins,
                    importance_type="permutation",
                    random_state=random_state,
                    n_jobs=1,
                )
                warmup_model.fit(x, y)
                _ = warmup_model.feature_importances_
                del warmup_model
                progress.advance(f"Bankai {label} warmup {warmup + 1}/{warmups}, seed {seed + 1}/{repeats}")

            model = BankaiRandomForestClassifier(
                n_estimators=trees,
                max_bins=max_bins,
                importance_type="permutation",
                random_state=random_state,
                n_jobs=1,
            )
            fit_time, _ = elapsed(lambda: model.fit(x, y))
            access_time, importances = elapsed(lambda: model.feature_importances_)
            progress.advance(f"Bankai {label} measured, seed {seed + 1}/{repeats}")
            fit_samples.append(fit_time)
            access_samples.append(access_time)
            importance_values.append(np.asarray(importances, dtype=np.float64))

        records.append({
            "implementation": "bankai",
            "mode": label,
            "max_bins": "" if max_bins is None else max_bins,
            "rows": rows,
            "validation_rows": len(validation_y),
            "features": features,
            "trees": trees,
            "repeats": repeats,
            "warmups_per_seed": warmups,
            "fit_seconds": statistics.median(fit_samples),
            "feature_importance_seconds": statistics.median(access_samples),
            "fit_plus_importance_seconds": statistics.median(fit_samples)
            + statistics.median(access_samples),
            "importance_rank_correlation_vs_exact": "pending",
        })
        records[-1]["_importance"] = np.median(np.stack(importance_values), axis=0)

    exact_importance = records[0]["_importance"]
    for record in records:
        record["importance_rank_correlation_vs_exact"] = rank_correlation(
            exact_importance, record["_importance"]
        )

    external_models = [
        (
            "sklearn",
            "random_forest",
            lambda seed: RandomForestClassifier(
                n_estimators=trees,
                max_features="sqrt",
                n_jobs=1,
                random_state=seed,
            ),
        ),
        (
            "lightgbm",
            "random_forest_boosting",
            lambda seed: LGBMClassifier(
                boosting_type="rf",
                n_estimators=trees,
                bagging_freq=1,
                bagging_fraction=0.8,
                feature_fraction=1.0,
                n_jobs=1,
                random_state=seed,
                verbosity=-1,
            ),
        ),
    ]
    for implementation, mode, model_factory in external_models:
        fit_samples, importance_samples, total_samples, importance_values = [], [], [], []
        for seed in range(repeats):
            random_state = 42 + seed
            for warmup in range(warmups):
                warmup_model = model_factory(random_state)
                warmup_model.fit(x, y)
                permutation_importance(
                    warmup_model, validation_x, validation_y, scoring="f1",
                    n_repeats=1, n_jobs=1, random_state=random_state,
                )
                del warmup_model
                progress.advance(f"{implementation} {mode} warmup {warmup + 1}/{warmups}, seed {seed + 1}/{repeats}")

            model = model_factory(random_state)
            fit_time, _ = elapsed(lambda: model.fit(x, y))
            importance_time, result = elapsed(lambda: permutation_importance(
                model, validation_x, validation_y, scoring="f1", n_repeats=1,
                n_jobs=1, random_state=42 + seed,
            ))
            progress.advance(f"{implementation} {mode} measured, seed {seed + 1}/{repeats}")
            fit_samples.append(fit_time)
            importance_samples.append(importance_time)
            total_samples.append(fit_time + importance_time)
            importance_values.append(result.importances_mean)

        external_importance = np.median(np.stack(importance_values), axis=0)
        records.append({
            "implementation": implementation,
            "mode": mode,
            "max_bins": "",
            "rows": rows,
            "validation_rows": len(validation_y),
            "features": features,
            "trees": trees,
            "repeats": repeats,
            "warmups_per_seed": warmups,
            "fit_seconds": statistics.median(fit_samples),
            "feature_importance_seconds": statistics.median(importance_samples),
            "fit_plus_importance_seconds": statistics.median(total_samples),
            "importance_rank_correlation_vs_exact": rank_correlation(
                exact_importance, external_importance
            ),
        })
    for record in records:
        record.pop("_importance", None)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--bins", nargs="+", type=int, default=[16, 32, 64, 128, 255])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results-histogram-10k"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.rows > 100_000:
        parser.error("--rows is capped at 100000 by the benchmark memory policy")
    if args.warmups < 1:
        parser.error("--warmups must be at least 1")
    if not args.skip_build:
        prepare_optimized_binaries(Path(__file__).parents[1])
    runs_per_implementation = args.repeats * (args.warmups + 1)
    progress = Progress({
        "Bankai": (len(args.bins) + 1) * runs_per_implementation,
        "sklearn": runs_per_implementation,
        "lightgbm": runs_per_implementation,
    })
    all_x, all_y = generate_dataset(args.rows * 2, args.features)
    rows = benchmark(
        args.rows, args.features, args.trees, args.bins, args.repeats, args.warmups,
        all_x[:args.rows], all_y[:args.rows], all_x[args.rows:], all_y[args.rows:], progress,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (args.output_dir / "histogram_importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "histogram_importance.md").open("w") as handle:
        handle.write("# Histogram and permutation importance benchmark\n\n")
        handle.write(f"Workload: {args.rows:,} training rows, {args.rows:,} validation rows, {args.features} features, {args.trees} trees; {args.repeats} measured seeds and {args.warmups} warmup(s) per seed. Total execution time (excluding release builds): {format_duration(time.monotonic() - progress.started)}.\n\n")
        handle.write("Environment: macOS {system} on {machine}; Python {python}; NumPy {numpy}; scikit-learn {sklearn}; LightGBM {lightgbm}. The Python extension and Rust CLI are built in release mode with `RUSTFLAGS=-C target-cpu=native`.\n\n".format(
            system=__import__("platform").mac_ver()[0] or sys.platform,
            machine=__import__("platform").machine(),
            python=sys.version.split()[0],
            numpy=np.__version__,
            sklearn=__import__("sklearn").__version__,
            lightgbm=__import__("lightgbm").__version__,
        ))
        handle.write("Times are medians across model seeds after one or more untimed warmups for every implementation, mode, and seed; the measured model is recreated with the same seed. External permutation importance uses one repeat per feature. Bankai computes native OOB permutation importance during fit, so its fit time already includes the importance calculation; `feature_importance_seconds` measures only public attribute access. Scikit-learn and LightGBM calculate permutation importance externally on the same validation set. LightGBM uses `boosting_type='rf'`, 80% row bagging, all features per tree, and one thread. The comparable workload is `fit_plus_importance_seconds`.\n\n")
        handle.write("| implementation | mode | bins | fit s (Bankai includes OOB permutation) | importance access/calculation s | fit + importance s | rank corr vs Bankai exact |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write("| {implementation} | {mode} | {max_bins} | {fit_seconds:.6f} | {feature_importance_seconds:.6f} | {fit_plus_importance_seconds:.6f} | {importance_rank_correlation_vs_exact:.6f} |\n".format(**row))
    print(f"Wrote {args.output_dir / 'histogram_importance.csv'} and Markdown report")


if __name__ == "__main__":
    main()
