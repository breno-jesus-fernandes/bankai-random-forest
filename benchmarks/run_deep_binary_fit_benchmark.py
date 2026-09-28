"""Fit and permutation-importance benchmark for the deep binary workload."""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from bankai_random_forest import BankaiRandomForestClassifier


def timed_fit(factory, x, y):
    model = factory()
    start = time.perf_counter()
    model.fit(x, y)
    return time.perf_counter() - start, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results-fit-hardware-mvp/deep-binary-permutation.json"))
    args = parser.parse_args()

    # Same synthetic workload/split and tree limits as the prior deep_binary run.
    x, y = make_classification(
        n_samples=24000, n_features=32, n_informative=16, n_classes=2,
        random_state=1729,
    )
    x = np.ascontiguousarray(x)
    x_train, y_train = x[:20000], y[:20000]
    x_valid, y_valid = x[20000:], y[20000:]
    common = dict(n_estimators=40, max_depth=20, random_state=42)
    bankai_factory = lambda: BankaiRandomForestClassifier(
        **common, max_bins=63, max_leaf_nodes=511, min_samples_leaf=5,
        max_features=None, max_samples=0.8, importance_type="permutation",
        n_jobs=-1,
    )
    lightgbm_factories = {
        "lightgbm_rf": lambda: LGBMClassifier(
            **common, boosting_type="rf", max_bin=63, num_leaves=511,
            min_child_samples=5, bagging_freq=1, bagging_fraction=0.8,
            feature_fraction=1.0, n_jobs=-1, verbosity=-1,
        ),
        "lightgbm_gbdt": lambda: LGBMClassifier(
            **common, boosting_type="gbdt", learning_rate=0.1,
            max_bin=63, num_leaves=511, min_child_samples=5,
            bagging_freq=1, bagging_fraction=0.8,
            feature_fraction=1.0, n_jobs=-1, verbosity=-1,
        ),
        "lightgbm_dart": lambda: LGBMClassifier(
            **common, boosting_type="dart", learning_rate=0.1,
            max_bin=63, num_leaves=511, min_child_samples=5,
            bagging_freq=1, bagging_fraction=0.8,
            feature_fraction=1.0, n_jobs=-1, verbosity=-1,
        ),
        "lightgbm_goss": lambda: LGBMClassifier(
            **common, boosting_type="goss", learning_rate=0.1,
            max_bin=63, num_leaves=511, min_child_samples=5,
            feature_fraction=1.0, top_rate=0.2, other_rate=0.1,
            n_jobs=-1, verbosity=-1,
        ),
    }
    factories = {"bankai": bankai_factory, **lightgbm_factories}

    # Measure fit only. Bankai's selected permutation importance is calculated
    # as part of its fit; LightGBM performs no permutation-importance work.
    for _ in range(args.warmups):
        for factory in factories.values():
            timed_fit(factory, x_train, y_train)

    records = []
    implementations = list(factories)
    for repeat in range(args.repeats):
        implementations = list(factories)
        if repeat % 2:
            implementations.reverse()
        for implementation in implementations:
            factory = factories[implementation]
            fit_seconds, model = timed_fit(factory, x_train, y_train)
            pred = model.predict(x_valid)
            metrics = {
                "accuracy": float(accuracy_score(y_valid, pred)),
                "precision": float(precision_score(y_valid, pred, zero_division=0)),
                "recall": float(recall_score(y_valid, pred, zero_division=0)),
                "f1": float(f1_score(y_valid, pred, zero_division=0)),
            }
            records.append({
                "implementation": implementation,
                "repeat": repeat,
                "fit_seconds": fit_seconds,
                **metrics,
            })
            print(implementation, repeat, f"fit={fit_seconds:.4f}s", flush=True)

    summary = {}
    for implementation in factories:
        rows = [r for r in records if r["implementation"] == implementation]
        summary[implementation] = {
            key: statistics.median(r[key] for r in rows)
            for key in ("fit_seconds", "accuracy", "precision", "recall", "f1")
        }
    digest = hashlib.sha256()
    for path in sorted([*Path("src").rglob("*.rs"), *Path("crates").rglob("*.rs")]):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    from bankai_random_forest import _core
    result = {
        "case": "deep_binary", "rows": 20000, "validation_rows": 4000,
        "features": 32, "trees": 40, "max_depth": 20, "max_leaves": 511,
        "min_samples_leaf": 5, "bins": 63, "threads": -1,
        "warmups": args.warmups, "repeats": args.repeats,
        "bankai_importance": "OOB permutation during fit",
        "lightgbm_importance": "not calculated",
        "lightgbm_configurations": {
            "lightgbm_rf": "RF boosting, 80% row bagging",
            "lightgbm_gbdt": "GBDT, learning_rate=0.1, 80% row bagging",
            "lightgbm_dart": "DART, learning_rate=0.1, 80% row bagging",
            "lightgbm_goss": "GOSS, learning_rate=0.1, top_rate=0.2, other_rate=0.1",
        },
        "source_sha256": digest.hexdigest(),
        "extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "machine": platform.machine(),
        "cpus": os.cpu_count(), "python": platform.python_version(),
        "numpy": np.__version__, "lightgbm": lightgbm.__version__,
        "summary_medians": summary, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
