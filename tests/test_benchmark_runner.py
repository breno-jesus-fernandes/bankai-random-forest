import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "bankai_benchmark", Path(__file__).parents[1] / "benchmarks" / "run_benchmark.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_dataset_is_deterministic_and_binary():
    first_x, first_y = benchmark.generate_dataset(16, 4)
    second_x, second_y = benchmark.generate_dataset(16, 4)

    assert first_x.shape == (16, 4)
    assert set(first_y) <= {0, 1}
    assert (first_x == second_x).all()
    assert (first_y == second_y).all()


def test_permutation_importance_rank_correlation_matches_identical_rankings():
    importance_spec = importlib.util.spec_from_file_location(
        "bankai_permutation_benchmark",
        Path(__file__).parents[1]
        / "benchmarks"
        / "run_permutation_importance_benchmark.py",
    )
    importance_benchmark = importlib.util.module_from_spec(importance_spec)
    importance_spec.loader.exec_module(importance_benchmark)

    correlation = importance_benchmark.rank_correlation(
        np.array([0.3, 0.2, 0.1]), np.array([3.0, 2.0, 1.0])
    )

    assert correlation == 1.0
