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


def test_runner_imports_without_unix_resource_module(monkeypatch):
    import builtins

    native_import = builtins.__import__

    def import_without_resource(name, *args, **kwargs):
        if name == 'resource':
            raise ImportError('resource is unavailable')
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_resource)
    portable_spec = importlib.util.spec_from_file_location(
        'fit_exploration_without_resource', Path(__file__).parents[1] / 'benchmarks/run_fit_exploration.py')
    portable_runner = importlib.util.module_from_spec(portable_spec)
    portable_spec.loader.exec_module(portable_runner)

    assert portable_runner.resource is None


def test_quality_requires_all_nine_seed_pairs(tmp_path):
    import json
    spec = importlib.util.spec_from_file_location('quality', Path(__file__).parents[1] / 'benchmarks/summarize_fit_quality.py')
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    entry = dict(dataset={'x': 'same'}, data_seed=1, model_seed=42,
                 prediction_sha256='same', metrics=dict(accuracy=.9, precision=.9, recall=.9, f1=.9))
    directory = tmp_path / 'one'
    directory.mkdir()
    for variant in ['baseline', 'candidate']:
        (directory / f'{variant}-0.json').write_text(json.dumps(entry))
    assert quality.summarize(tmp_path)['passed'] is False


def test_quality_rejects_a_metric_outside_margin(tmp_path):
    import json
    spec = importlib.util.spec_from_file_location('quality', Path(__file__).parents[1] / 'benchmarks/summarize_fit_quality.py')
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    for data in range(3):
        for model in range(3):
            directory = tmp_path / f'{data}-{model}'
            directory.mkdir()
            entry = dict(dataset={'x': str(data)}, data_seed=data, model_seed=model,
                         prediction_sha256='same', metrics=dict(accuracy=.9, precision=.9, recall=.9, f1=.9))
            (directory / 'baseline-0.json').write_text(json.dumps(entry))
            entry['metrics']['recall'] -= .003
            (directory / 'candidate-0.json').write_text(json.dumps(entry))
    result = quality.summarize(tmp_path)
    assert result['complete']
    assert not result['passed']
    assert not result['metrics']['recall']['passed']


def test_acceptance_fails_closed_without_evidence(tmp_path):
    import json
    import subprocess
    import sys
    script = Path(__file__).parents[1] / 'benchmarks/evaluate_fit_campaign.py'
    subprocess.run([sys.executable, str(script), str(tmp_path)], check=True, capture_output=True)
    result = json.loads((tmp_path / 'acceptance.json').read_text())
    assert not result['numeric_gates_passed']
    assert not result['main']['complete']
