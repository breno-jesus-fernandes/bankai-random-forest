#!/usr/bin/env python3
"""Short Extra Trees comparison using the banking notebook's generator.

Build with .venv/bin/maturin develop --release --skip-install.
Run with .venv/bin/python benchmarks/run_extra_trees_quick.py --output /tmp/before.json
Repeat after a release build with --compare /tmp/before.json. No Boruta loop.
Use --include-binned to also measure 63-bin sampled_select Extra Trees.
"""
import argparse
import ast
import hashlib
import json
import platform
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
# Prefer the in-place release extension produced by maturin --skip-install.
sys.path.insert(0, os.environ.get('BANKAI_BENCHMARK_PACKAGE_ROOT', str(Path(__file__).resolve().parents[1] / 'python')))
from bankai_random_forest import BankaiRandomForestClassifier
import bankai_random_forest._core as core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=100000)
    p.add_argument('--features', type=int, default=500)
    p.add_argument('--trees', type=int, default=64)
    p.add_argument('--min-leaf', type=int, help='Override notebook leaf size')
    p.add_argument('--max-depth', type=int, help='Override notebook depth; 0 means unlimited')
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--scenario', default='credit_default')
    p.add_argument('--no-bootstrap', action='store_true')
    p.add_argument('--include-binned', action='store_true')
    p.add_argument('--warmup', action='store_true', help='Fit each model once before measurements on the first dataset')
    p.add_argument('--bankai-only', action='store_true', help='Skip sklearn control for ablation screening')
    p.add_argument('--bins', type=int, default=63)
    p.add_argument('--bin-sample-size', type=int, default=20000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    args = p.parse_args()
    if args.rows < 1000 or args.features < 24 or min(args.trees, args.jobs, args.repeats) < 1:
        p.error('require rows >= 1000, features >= 24, and positive trees/jobs/repeats')
    if args.min_leaf is not None and args.min_leaf < 1 or args.max_depth is not None and args.max_depth < 0:
        p.error('min-leaf must be positive and max-depth nonnegative')
    if not 2 <= args.bins <= 255 or args.bin_sample_size < 1:
        p.error('require 2 <= bins <= 255 and positive bin-sample-size')
    if args.compare:
        previous = json.loads(args.compare.read_text())
        keys = ('rows', 'features', 'trees', 'jobs', 'repeats', 'scenario', 'no_bootstrap')
        keys += ('include_binned', 'bins', 'bin_sample_size', 'bankai_only', 'warmup', 'dtype', 'min_leaf', 'max_depth')
        defaults = dict(include_binned=False, bins=63, bin_sample_size=20000, bankai_only=False, warmup=False, dtype='float32', min_leaf=None, max_depth=None)
        if any(previous['config'].get(key, defaults.get(key)) != vars(args)[key] for key in keys):
            p.error('--compare requires matching benchmark parameters')
    notebook = Path(__file__).with_name('boruta_banking_synthetic_benchmark.ipynb')
    cells = json.loads(notebook.read_text())['cells']
    source = next(''.join(c['source']) for c in cells if ''.join(c['source']).startswith('SCENARIOS ='))
    ns = dict(np=np, expit=expit, N_ROWS=args.rows, N_FEATURES=args.features, N_INFORMATIVE=24)
    # Reuse only the generator definitions, never notebook install/benchmark cells.
    definitions = ast.parse(source)
    definitions.body = [node for node in definitions.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    exec(compile(definitions, str(notebook), 'exec'), ns)
    records = []
    for repeat in range(args.repeats):
        seed = 20261011 + repeat
        x, y, _ = ns['make_dataset'](args.scenario, seed)
        x = x.astype(args.dtype, copy=False)
        train, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
        rng = np.random.default_rng(seed + 100)
        # Boruta starts each round by appending independently permuted shadows.
        shadows = np.empty_like(x)
        for col in range(args.features):
            shadows[:, col] = rng.permutation(x[:, col])
        augmented = np.concatenate((x, shadows), axis=1)
        xt, xv = augmented[train], augmented[test]
        cfg = ns['SCENARIOS'][args.scenario]
        depth = cfg['max_depth'] if args.max_depth is None else (args.max_depth or None)
        leaf = cfg['min_leaf'] if args.min_leaf is None else args.min_leaf
        common = dict(n_estimators=args.trees, max_depth=depth,
                      min_samples_leaf=leaf, max_features='sqrt',
                      bootstrap=not args.no_bootstrap,
                      max_samples=None if args.no_bootstrap else .8,
                      class_weight='balanced', random_state=seed, n_jobs=args.jobs)
        models = dict(bankai=BankaiRandomForestClassifier(**common, extra_trees=True,
                      max_bins=None, importance_type='gain'), sklearn=ExtraTreesClassifier(**common))
        if args.include_binned:
            models['bankai_binned'] = BankaiRandomForestClassifier(
                **common, extra_trees=True, extra_trees_backend='binned',
                max_bins=args.bins, binning_strategy='sampled_select',
                bin_sample_size=args.bin_sample_size, importance_type='gain')
        if args.bankai_only:
            models = {name: model for name, model in models.items() if name != 'sklearn'}
        # Rotate positions so each of three models goes first once in three runs.
        names = list(models)
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            model = models[name]
            if args.warmup and repeat == 0:
                model.fit(xt, y[train])
            start_cpu = time.process_time()
            start = time.perf_counter()
            model.fit(xt, y[train])
            elapsed = time.perf_counter() - start
            cpu_elapsed = time.process_time() - start_cpu
            importance = model.feature_importances_
            scores = model.predict_proba(xv)[:, 1]
            truth = np.arange(2 * args.features) < 24
            if name.startswith('bankai'):
                trees = model._forest.shap_tree_arrays()
                structure = sorted(json.dumps(tree[:4]) for tree in trees)
                tree_hash = hashlib.sha256(json.dumps(structure).encode()).hexdigest()
                node_count = sum(len(tree[0]) for tree in trees)
            else:
                tree_hash = None
                node_count = sum(tree.tree_.node_count for tree in model.estimators_)
            row = dict(repeat=repeat, model=name, fit_seconds=elapsed, fit_cpu_seconds=cpu_elapsed,
                       tree_structure_sha256=tree_hash, node_count=node_count,
                       feature_ap=average_precision_score(truth, importance),
                       recall_at_24=float(truth[np.argsort(importance)[-24:]].mean()),
                       roc_auc=roc_auc_score(y[test], scores),
                       prediction_ap=average_precision_score(y[test], scores),
                       importance=importance.tolist(), scores=scores.tolist())
            records.append(row)
            print({k:v for k,v in row.items() if k not in ('importance','scores')}, flush=True)
    result = dict(experiment=os.environ.get('BANKAI_EXTRA_EXPERIMENT', ''), platform=platform.platform(), numpy_version=np.__version__, config=vars(args) | {'output': str(args.output), 'compare': str(args.compare)},
                  extension=str(core.__file__), binary_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                  records=records)
    args.output.write_text(json.dumps(result, indent=2))
    for name in models:
        rows = [r for r in records if r['model'] == name]
        print(name, {k: float(np.median([r[k] for r in rows])) for k in
              ('fit_seconds','feature_ap','recall_at_24','roc_auc','prediction_ap')})
    if args.compare:
        before = json.loads(args.compare.read_text())
        for row in records:
            old = next(r for r in before['records'] if (r['model'],r['repeat']) == (row['model'],row['repeat']))
            print('paired', row['model'], row['repeat'], 'speedup', old['fit_seconds']/row['fit_seconds'],
                  'same_tree_structure', old.get('tree_structure_sha256') == row.get('tree_structure_sha256'),
                  'importance_max_abs_delta', float(np.max(np.abs(np.array(old['importance'])-row['importance']))),
                  'prediction_max_abs_delta', float(np.max(np.abs(np.array(old['scores'])-row['scores']))))


if __name__ == '__main__':
    main()
