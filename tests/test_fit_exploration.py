"""The experiment runner must never pair unrelated or incomplete repeats."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('fit_exploration', Path(__file__).parents[1] / 'benchmarks/run_fit_exploration.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def record(variant, repeat, seconds):
    return dict(variant=variant, repeat=repeat, fit_seconds=seconds,
                fit_peak_rss=100, warmup=False)


def test_partial_pairs_are_excluded():
    result = runner.summarize([record('baseline', 0, 10), record('candidate', 0, 5),
                               record('baseline', 1, 100)])
    assert result['paired']['median_speedup'] == 2
    assert result['paired']['ratio_ci95'] == [2, 2]


def test_unpaired_results_have_no_speedup():
    result = runner.summarize([record('baseline', 0, 10), record('candidate', 1, 5)])
    assert 'paired' not in result
