#!/usr/bin/env python3
"""Compare release multioutput Bankai and sklearn forests with warmups."""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    rustflags = environment.get("RUSTFLAGS", "")
    environment["RUSTFLAGS"] = " ".join(
        part for part in (rustflags, "-C target-cpu=native") if part
    )
    subprocess.run(
        ["maturin", "develop", "--release"],
        check=True,
        cwd=root,
        env=environment,
    )

    from bankai_random_forest import BankaiRandomForestClassifier

    rng = np.random.RandomState(861)
    x = rng.normal(size=(10_000, 20))
    x[rng.random_sample(x.shape) < 0.9] = 0.0
    score_a = x[:, 0] + 0.8 * x[:, 1] - 0.4 * x[:, 2]
    score_b = x[:, 3] - 0.5 * x[:, 4]
    y = np.empty((len(x), 2), dtype=object)
    y[:, 0] = np.where(score_a > np.median(score_a), "positive", "negative")
    y[:, 1] = np.where(
        score_b < np.quantile(score_b, 1 / 3),
        "low",
        np.where(score_b > np.quantile(score_b, 2 / 3), "high", "mid"),
    )

    rows = []
    for max_bins in (None, 32):
        for seed in (17, 29, 43):
            params = dict(
                n_estimators=50,
                max_features=4,
                random_state=seed,
                n_jobs=1,
                max_bins=max_bins,
            )
            warm_bankai = BankaiRandomForestClassifier(**params).fit(x, y)
            warm_sklearn = RandomForestClassifier(**{k: v for k, v in params.items() if k != "max_bins"}).fit(x, y)
            warm_bankai.predict(x)
            warm_bankai.predict_proba(x)
            warm_sklearn.predict(x)
            warm_sklearn.predict_proba(x)

            for repeat in range(1, 4):
                bankai = BankaiRandomForestClassifier(**params)
                started = time.perf_counter()
                bankai.fit(x, y)
                bankai_fit = time.perf_counter() - started
                started = time.perf_counter()
                bankai_prediction = bankai.predict(x)
                bankai_predict = time.perf_counter() - started
                bankai_probabilities = bankai.predict_proba(x)

                sklearn_params = {k: v for k, v in params.items() if k != "max_bins"}
                reference = RandomForestClassifier(**sklearn_params)
                started = time.perf_counter()
                reference.fit(x, y)
                sklearn_fit = time.perf_counter() - started
                started = time.perf_counter()
                sklearn_prediction = reference.predict(x)
                sklearn_predict = time.perf_counter() - started
                sklearn_probabilities = reference.predict_proba(x)

                probability_rmse = np.sqrt(
                    np.mean(
                        [
                            np.mean((actual - expected) ** 2)
                            for actual, expected in zip(
                                bankai_probabilities, sklearn_probabilities
                            )
                        ]
                    )
                )
                rows.append(
                    {
                        "mode": "exact" if max_bins is None else "hist_32",
                        "seed": seed,
                        "repeat": repeat,
                        "bankai_fit_seconds": bankai_fit,
                        "bankai_predict_seconds": bankai_predict,
                        "sklearn_fit_seconds": sklearn_fit,
                        "sklearn_predict_seconds": sklearn_predict,
                        "prediction_agreement": np.mean(
                            bankai_prediction == sklearn_prediction
                        ),
                        "probability_rmse": probability_rmse,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
