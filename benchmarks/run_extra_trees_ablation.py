#!/usr/bin/env python3
"""Compare two local native builds in isolated packages, alternating AB / BA.

Run from the repo venv. Copies binaries into isolated temporary packages, never
replaces the active extension. Each subprocess reuses run_extra_trees_quick.py.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-core', type=Path, required=True)
    p.add_argument('--candidate-core', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--rows', type=int, default=100000)
    p.add_argument('--features', type=int, default=500)
    p.add_argument('--trees', type=int, default=64)
    p.add_argument('--min-leaf', type=int)
    p.add_argument('--max-depth', type=int)
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--passes', type=int, default=2)
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--scenario', default='credit_default')
    p.add_argument('--no-bootstrap', action='store_true')
    p.add_argument('--include-binned', action='store_true')
    p.add_argument('--baseline-experiment', default='')
    p.add_argument('--candidate-experiment', default='')
    args = p.parse_args()
    if args.passes < 1:
        p.error('passes must be positive')
    repo = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, manifests = [], []
    with tempfile.TemporaryDirectory(prefix='bankai-ablation-') as temporary:
        roots = {}
        for label, binary in [('baseline', args.baseline_core), ('candidate', args.candidate_core)]:
            root = Path(temporary) / label
            package = root / 'bankai_random_forest'
            shutil.copytree(repo / 'python' / 'bankai_random_forest', package,
                            ignore=shutil.ignore_patterns('*.so', '__pycache__'))
            shutil.copy2(binary.resolve(), package / '_core.abi3.so')
            roots[label] = root
        for turn in range(args.passes):
            order = ['baseline', 'candidate'] if turn % 2 == 0 else ['candidate', 'baseline']
            for label in order:
                stem = f'{turn}-{label}'
                output = args.output_dir / f'{stem}.json'
                command = [sys.executable, str(repo / 'benchmarks' / 'run_extra_trees_quick.py'),
                           '--bankai-only', '--warmup', '--output', str(output), '--scenario', args.scenario,
                           '--dtype', args.dtype]
                for key in ['rows', 'features', 'trees', 'jobs', 'repeats']:
                    command.extend([f'--{key}', str(getattr(args, key))])
                for key in ['min_leaf', 'max_depth']:
                    if getattr(args, key) is not None:
                        command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
                if args.no_bootstrap:
                    command.append('--no-bootstrap')
                if args.include_binned:
                    command.append('--include-binned')
                env = os.environ | {'BANKAI_BENCHMARK_PACKAGE_ROOT': str(roots[label]),
                                    'BANKAI_EXTRA_EXPERIMENT': getattr(args, f'{label}_experiment')}
                print(f'pass {turn}: {label}', flush=True)
                with (args.output_dir / f'{stem}.log').open('w') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads(output.read_text())
                manifests.append(dict(label=label, pass_index=turn,
                                      **{k:v for k,v in result.items() if k != 'records'}))
                for record in result['records']:
                    records.append(dict(label=label, pass_index=turn, **record))
                measured = [r['fit_seconds'] for r in result['records'] if r['model'] == 'bankai']
                print(label, 'exact median', statistics.median(measured), flush=True)
    slim = [{k:v for k,v in r.items() if k not in ('importance', 'scores')} for r in records]
    with (args.output_dir / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(slim[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(slim)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifests, indent=2))
    checks = []
    for new in [r for r in records if r['label'] == 'candidate']:
        old = next(r for r in records if r['label'] == 'baseline' and
                   (r['pass_index'], r['repeat'], r['model']) ==
                   (new['pass_index'], new['repeat'], new['model']))
        checks.append(dict(pass_index=new['pass_index'], repeat=new['repeat'], model=new['model'],
                           speedup=old['fit_seconds']/new['fit_seconds'],
                           cpu_speedup=old['fit_cpu_seconds']/new['fit_cpu_seconds'],
                           same_tree_structure=old['tree_structure_sha256'] == new['tree_structure_sha256'],
                           prediction_max_abs_delta=max(abs(a-b) for a,b in zip(old['scores'],new['scores'])),
                           importance_max_abs_delta=max(abs(a-b) for a,b in zip(old['importance'],new['importance']))))
    (args.output_dir / 'paired.json').write_text(json.dumps(checks, indent=2))
    summary = []
    for model in sorted({r['model'] for r in records}):
        paired = [r for r in checks if r['model'] == model]
        row = dict(model=model,
                   baseline_fit_median=statistics.median(r['fit_seconds'] for r in records if r['label']=='baseline' and r['model']==model),
                   candidate_fit_median=statistics.median(r['fit_seconds'] for r in records if r['label']=='candidate' and r['model']==model),
                   median_paired_speedup=statistics.median(r['speedup'] for r in paired),
                   median_paired_cpu_speedup=statistics.median(r['cpu_speedup'] for r in paired),
                   same_tree_structure=all(r['same_tree_structure'] for r in paired),
                   prediction_max_abs_delta=max(r['prediction_max_abs_delta'] for r in paired),
                   importance_max_abs_delta=max(r['importance_max_abs_delta'] for r in paired))
        summary.append(row)
        print(row, flush=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
