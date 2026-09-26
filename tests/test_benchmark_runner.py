import importlib.util
from pathlib import Path


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
