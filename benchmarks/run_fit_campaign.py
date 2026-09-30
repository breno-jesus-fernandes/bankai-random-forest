"""Run a declared campaign phase sequentially; each child persists raw evidence."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('phase', choices=['controls', 'quality', 'comparators', 'layouts'])
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
runner = Path(__file__).with_name('run_fit_exploration.py')


def run(name, extra, paired=True, repeats=5, warmups=1):
    command = [sys.executable, str(runner), '--baseline', str(args.baseline),
               '--output', str(args.output / name), '--repeats', str(repeats), '--warmups', str(warmups)]
    if paired:
        command += ['--candidate', str(args.candidate)]
    print('RUN', name, flush=True)
    subprocess.run(command + extra, check=True)


if args.phase == 'controls':
    for jobs in (1, -1):
        run(f'jobs-{jobs}', ['--jobs', str(jobs)])
    for scenario in ('exact', 'float64', 'permutation', 'multiclass', 'sparse', 'nan', 'few'):
        run(scenario, ['--scenario', scenario, '--rows', '5000', '--features', '32'])
elif args.phase == 'quality':
    for data_seed in (1729, 1730, 1731):
        for model_seed in (42, 43, 44):
            run(f'{data_seed}-{model_seed}', ['--data-seed', str(data_seed), '--model-seed', str(model_seed)], repeats=1, warmups=0)
    subprocess.run([sys.executable, str(runner.with_name('summarize_fit_quality.py')), str(args.output)], check=True)
elif args.phase == 'comparators':
    for library in ('sklearn', 'lightgbm'):
        run(library, ['--library', library], paired=False, repeats=2, warmups=0)
else:
    for layout in ('C', 'F', 'pandas', 'polars', 'arrow'):
        run(layout, ['--layout', layout], repeats=2, warmups=1)
