"""Isolated, paired fit measurements; generated data and binaries stay outside git."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import multiprocessing
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dataset(args):
    import numpy as np
    from sklearn.datasets import make_classification
    root = args.data / f'{args.rows}-{args.features}-{args.data_seed}'
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'manifest.json').exists():
        x, y = make_classification(n_samples=args.rows, n_features=args.features,
                                  n_informative=min(90, args.features - 2),
                                  random_state=args.data_seed)
        np.save(root / 'x.npy', x.astype(np.float32))
        np.save(root / 'y.npy', y)
        (root / 'manifest.json').write_text(json.dumps({
            name: digest(root / name) for name in ('x.npy', 'y.npy')}, indent=2))
    return root


def monitor_memory(pid, stop, ready, peak):
    # A separate process samples even while native fit holds the Python GIL.
    import psutil
    process = psutil.Process(pid)
    ready.set()
    while not stop.wait(.01):
        peak.value = max(peak.value, process.memory_info().rss)


def worker(args):
    import numpy as np
    import psutil
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    root = args.data
    x = np.load(root / 'x.npy')
    y = np.load(root / 'y.npy')
    if len(y) > 100000 or len(x) != len(y):
        raise ValueError('Dataset exceeds campaign limit or has inconsistent lengths')
    split = int(len(y) * .8)
    train, valid = x[:split], x[split:]
    conversion_start = time.perf_counter()
    if args.layout == 'F':
        train = np.asfortranarray(train)
    elif args.layout == 'pandas':
        import pandas as pd
        train = pd.DataFrame(train)
        valid = pd.DataFrame(valid)
    elif args.layout in ('polars', 'arrow'):
        if args.layout == 'polars':
            import polars as pl
            train = pl.DataFrame(train).to_numpy()
        else:
            import pyarrow as pa
            train = pa.table({str(i): train[:, i] for i in range(train.shape[1])}).to_pandas().to_numpy()
    conversion_seconds = time.perf_counter() - conversion_start
    if args.scenario == 'float64':
        train = train.astype(np.float64)
    elif args.scenario == 'nan':
        train = train.copy()
        train[::17, ::7] = np.nan
    elif args.scenario == 'sparse':
        from scipy.sparse import csr_matrix
        train, valid = csr_matrix(train), csr_matrix(valid)
    elif args.scenario == 'multiclass':
        y = (y + np.arange(len(y)) % 3) % 3
    params = dict(n_estimators=2 if args.scenario == 'few' else args.trees,
                  max_depth=20, max_leaf_nodes=511, min_samples_leaf=5,
                  max_features=None, max_samples=.8, random_state=args.model_seed,
                  n_jobs=args.jobs)
    extension = None
    if args.library == 'bankai':
        from bankai_random_forest import BankaiRandomForestClassifier, _core
        params.update(max_bins=None if args.scenario == 'exact' else 63,
                      importance_type='permutation' if args.scenario == 'permutation' else 'gain')
        model = BankaiRandomForestClassifier(**params)
        extension = {'path': _core.__file__, 'sha256': digest(_core.__file__)}
    elif args.library == 'sklearn':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=args.trees, max_depth=20, num_leaves=511,
                               min_child_samples=5, boosting_type='rf', bagging_freq=1,
                               bagging_fraction=.8, feature_fraction=1., max_bin=63,
                               importance_type='gain', random_state=args.model_seed,
                               n_jobs=args.jobs, verbosity=-1)
    process = psutil.Process()
    rss_before_fit = process.memory_info().rss
    context = multiprocessing.get_context('spawn')
    stop, ready = context.Event(), context.Event()
    peak = context.Value('Q', rss_before_fit)
    sampler = context.Process(target=monitor_memory, args=(os.getpid(), stop, ready, peak))
    sampler.start()
    if not ready.wait(30):
        sampler.terminate()
        sampler.join()
        raise RuntimeError('RSS sampler did not start')
    try:
        start = time.perf_counter()
        model.fit(train, y[:split])
        elapsed = time.perf_counter() - start
        peak.value = max(peak.value, process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    fit_peak_rss = peak.value
    process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    pred = model.predict(valid)
    metrics = {name: float(fn(y[split:], pred, **({} if name == 'accuracy' else
                {'average': 'macro' if args.scenario == 'multiclass' else 'binary', 'zero_division': 0})))
               for name, fn in [('accuracy', accuracy_score), ('precision', precision_score),
                                ('recall', recall_score), ('f1', f1_score)]}
    result = dict(rows=len(y), features=x.shape[1], training_rows=split, fit_seconds=elapsed, fit_peak_rss=fit_peak_rss, rss_before_fit=rss_before_fit,
                  process_peak_rss=process_peak_rss, metrics=metrics,
                  conversion_seconds=conversion_seconds, fit_plus_conversion_seconds=elapsed + conversion_seconds, extension=extension,
                  prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),
                  probability_sha256=hashlib.sha256(np.asarray(model.predict_proba(valid)).tobytes()).hexdigest(),
                  importance_sha256=hashlib.sha256(np.asarray(model.feature_importances_).tobytes()).hexdigest(),
                  params=params, library=args.library, layout=args.layout, scenario=args.scenario,
                  dataset=json.loads((root / 'manifest.json').read_text()),
                  data_seed=args.data_seed, model_seed=args.model_seed,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  platform=platform.platform(), machine=platform.machine(), cpus=os.cpu_count(),
                  python=platform.python_version(),
                  packages={p: importlib.metadata.version(p) for p in
                            ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'psutil']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


def summarize(records):
    import numpy as np
    summary = {}
    for label in sorted({r['variant'] for r in records}):
        selected = [r for r in records if r['variant'] == label and not r['warmup']]
        times = [r['fit_seconds'] for r in selected]
        summary[label] = dict(median_seconds=float(np.median(times)),
                              min_seconds=min(times), max_seconds=max(times),
                              std_seconds=float(np.std(times)),
                              median_peak_rss=float(np.median([r['fit_peak_rss'] for r in selected])))
    baseline = {r['repeat']: r for r in records if r['variant'] == 'baseline' and not r['warmup']}
    candidate = {r['repeat']: r for r in records if r['variant'] == 'candidate' and not r['warmup']}
    paired = sorted(baseline.keys() & candidate.keys())
    if paired:
        ratios = np.array([baseline[i]['fit_seconds'] / candidate[i]['fit_seconds'] for i in paired])
        rng = np.random.default_rng(1729)
        boot = np.median(rng.choice(ratios, (20000, len(ratios))), axis=1)
        summary['paired'] = dict(median_speedup=float(np.median(ratios)),
                                ratio_ci95=np.quantile(boot, [.025, .975]).tolist(),
                                median_seconds_saved=float(np.median([baseline[i]['fit_seconds'] - candidate[i]['fit_seconds'] for i in paired])),
                                note='Percentile bootstrap of paired median ratios; screening is not confirmation.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--data', type=Path, default=Path('/tmp/bankai-fit-data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--features', type=int, default=500)
    parser.add_argument('--trees', type=int, default=40)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--repeat-offset', type=int, default=0,
                        help='First measurement index; warmups retain negative indices')
    parser.add_argument('--warmups', type=int, default=0)
    parser.add_argument('--data-seed', type=int, default=1729)
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--layout', choices=['C', 'F', 'pandas', 'polars', 'arrow'], default='C')
    parser.add_argument('--scenario', choices=['main', 'exact', 'float64', 'nan', 'sparse', 'multiclass', 'permutation', 'few'], default='main')
    parser.add_argument('--library', choices=['bankai', 'sklearn', 'lightgbm'], default='bankai')
    args = parser.parse_args()
    if not 10 <= args.rows <= 100000 or args.features < 4 or args.repeats < 1:
        parser.error('Require 10..100000 total rows, >=4 features and positive repeats')
    if args.scenario != 'main' and not args.worker and args.rows > 10000:
        parser.error('Control scenarios are limited to 10000 rows')
    if args.worker:
        worker(args)
        return
    root = dataset(args)
    records = []
    variants = [(k, v) for k, v in [('baseline', args.baseline), ('candidate', args.candidate)] if v]
    if not variants:
        parser.error('Supply --baseline and/or --candidate worktree paths')
    args.output.mkdir(parents=True, exist_ok=True)
    for repeat in range(-args.warmups, args.repeats):
        measurement = repeat if repeat < 0 else repeat + args.repeat_offset
        for label, worktree in variants[::(-1 if repeat % 2 else 1)]:
            output = args.output.resolve() / f'{label}-{measurement}.json'
            command = [str(worktree / '.venv/bin/python'), str(Path(__file__).resolve()), '--worker',
                       '--data', str(root), '--output', str(output)]
            for name in ['jobs', 'trees', 'layout', 'scenario', 'library', 'data_seed', 'model_seed']:
                command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
            subprocess.run(command, cwd=worktree, check=True,
                           env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': str(max(1, args.jobs))})
            record = json.loads(output.read_text())
            record.update(variant=label, repeat=measurement, warmup=repeat < 0, command=command)
            output.write_text(json.dumps(record, indent=2) + '\n')
            records.append(record)
            print(label, repeat, round(record['fit_seconds'], 3), flush=True)
            (args.output / 'summary.json').write_text(json.dumps(summarize([r for r in records if not r['warmup']]), indent=2) + '\n') if repeat >= 0 else None
    with (args.output / 'measurements.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'repeat', 'warmup', 'fit_seconds', 'fit_peak_rss', 'accuracy', 'precision', 'recall', 'f1'])
        writer.writeheader()
        for r in records:
            writer.writerow({**{k: r[k] for k in ['variant', 'repeat', 'warmup', 'fit_seconds', 'fit_peak_rss']}, **r['metrics']})


if __name__ == '__main__':
    main()
