"""Fraud-first, build-identified banking campaign; run with the checkout venv."""
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import bankai_random_forest
import banking_scenarios as banking

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(config):
    sources = sorted(p for folder in ('src', 'crates/xrf/src', 'python/bankai_random_forest', 'benchmarks')
                     for p in (ROOT / folder).rglob('*')
                     if p.suffix in ('.rs', '.py') and 'baseline-extra-trees' not in p.parts)
    return {'config': config, 'sources': {str(p.relative_to(ROOT)): digest(p) for p in sources},
            'loaded_python_sources': {p.name: digest(p) for p in sorted(Path(bankai_random_forest.__file__).parent.glob('*.py'))},
            'core_sha256': digest(Path(bankai_random_forest._core.__file__)),
            'versions': {name: metadata.version(name) for name in
                         ('bankai-random-forest', 'scikit-learn', 'numpy', 'lightgbm', 'xgboost')},
            'python': sys.version, 'hardware': {'platform': platform.platform(),
            'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
            'physical_cores': psutil.cpu_count(logical=False), 'ram_bytes': psutil.virtual_memory().total}}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def summarize(directory, records):
    raw = pd.DataFrame(records)
    raw.to_csv(directory / 'raw.csv', index=False)
    if raw.empty:
        return
    metrics = ['fit_seconds', 'feature_average_precision', 'recall_at_24', 'precision_at_24',
               'informative_missing', 'roc_auc', 'pr_auc', 'nodes', 'depth']
    raw.groupby(['scenario', 'model'])[metrics].median().to_csv(directory / 'medians.csv')
    pairs = []
    for scenario, group in raw.groupby('scenario'):
        for repeat, subset in group.groupby('repeat'):
            by_name = subset.set_index('model')
            if banking.BANKAI_ON in by_name.index and 'Extra Trees' in by_name.index:
                a, b = by_name.loc[banking.BANKAI_ON], by_name.loc['Extra Trees']
                pairs.append({'scenario': scenario, 'repeat': int(repeat),
                              **{metric: float(a[metric] - b[metric]) for metric in metrics}})
    if pairs:
        paired = pd.DataFrame(pairs)
        paired.to_csv(directory / 'paired.csv', index=False)
        paired.groupby('scenario')[metrics].median().to_csv(directory / 'paired-medians.csv')


def structure(model):
    if hasattr(model, '_forest'):
        arrays = model._forest.shap_tree_arrays()
        children = [(a[0], a[1]) for a in arrays]
    elif hasattr(model, 'estimators_'):
        children = [(t.tree_.children_left, t.tree_.children_right) for t in model.estimators_]
    else:
        if hasattr(model, 'booster_'):
            trees = [tree['tree_structure'] for tree in model.booster_.dump_model()['tree_info']]
            descendants = lambda node: [node['left_child'], node['right_child']] if 'left_child' in node else []
        elif hasattr(model, 'get_booster'):
            trees = [json.loads(tree) for tree in model.get_booster().get_dump(dump_format='json')]
            descendants = lambda node: node.get('children', [])
        else:
            return None, None
        count, maximum = 0, 0
        for tree in trees:
            stack = [(tree, 0)]
            while stack:
                node, depth = stack.pop()
                count += 1
                maximum = max(maximum, depth)
                stack.extend((child, depth + 1) for child in descendants(node))
        return count, maximum
    depths = []
    for left, right in children:
        stack = [(0, 0)]
        maximum = 0
        while stack:
            node, depth = stack.pop()
            maximum = max(maximum, depth)
            if left[node] >= 0:
                stack.extend([(left[node], depth + 1), (right[node], depth + 1)])
        depths.append(maximum)
    return sum(len(left) for left, _ in children), max(depths, default=0)


def resolved_params(model, rows, features):
    result = {'train_rows': rows, 'features': features}
    if hasattr(model, '_fit_max_features'):
        result.update(max_features=int(model._fit_max_features),
                      min_samples_leaf=int(model._fit_min_samples_leaf),
                      min_samples_split=int(model._fit_min_samples_split),
                      max_depth=model._fit_max_depth, max_samples=model._fit_max_samples,
                      n_jobs=int(model._fit_n_jobs), random_state=int(model._fit_seed),
                      max_bins=(None if model.extra_trees and hasattr(model._forest, 'tree_normalized_gain_importances') else model._fit_max_bins),
                      bootstrap=bool(model.bootstrap))
    elif hasattr(model, 'estimators_'):
        result.update(max_features=int(model.estimators_[0].max_features_),
                      n_estimators=len(model.estimators_), bootstrap=bool(model.bootstrap),
                      max_samples=model.max_samples)
    return result


def leaf_positive_counts(model, X, y):
    """Unweighted held-out positives per leaf (not training/bootstrap cover)."""
    if hasattr(model, '_forest'):
        arrays = model._forest.shap_tree_arrays()
        trees = [(a[0], a[1], a[2], a[3], a[6]) for a in arrays]
    elif hasattr(model, 'estimators_'):
        trees = [(t.tree_.children_left, t.tree_.children_right, t.tree_.feature,
                  t.tree_.threshold, t.tree_.missing_go_to_left) for t in model.estimators_]
    else:
        return None
    results = []
    for left, right, feature, threshold, missing in trees:
        counts = []
        stack = [(0, np.arange(len(y)))]
        while stack:
            node, rows = stack.pop()
            if left[node] < 0:
                counts.append({'rows': len(rows), 'positives': int(y[rows].sum())})
            else:
                values = X[rows, feature[node]]
                low = np.where(np.isnan(values), bool(missing[node]), values <= threshold[node])
                stack.extend([(left[node], rows[low]), (right[node], rows[~low])])
        results.append(counts)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--rows', type=int, default=500_000)
    parser.add_argument('--features', type=int, default=500)
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--bootstrap', action='store_true', help='Enable bootstrap with the scenario sampling fraction.')
    parser.add_argument('--all-scenarios', action='store_true')
    parser.add_argument('--scenarios', nargs='+', choices=list(banking.SCENARIOS))
    parser.add_argument('--all-models', action='store_true')
    parser.add_argument('--models', nargs='+', choices=banking.MODEL_NAMES)
    parser.add_argument('--extra-only', action='store_true', help='Only Bankai and sklearn Extra Trees, for baseline ablations.')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--instrumented', action='store_true')
    parser.add_argument('--max-bins', type=int, default=63, help='0 means None for Random Forest; current Extra Trees always uses None.')
    args = parser.parse_args()
    if args.extra_only and args.all_models:
        parser.error('--extra-only and --all-models are mutually exclusive')
    if args.instrumented:
        os.environ['BANKAI_PROFILE_FIT'] = '1'
    elif os.environ.get('BANKAI_PROFILE_FIT'):
        raise RuntimeError('Unset BANKAI_PROFILE_FIT for timing campaigns, or pass --instrumented.')
    config = {k: v for k, v in vars(args).items() if k != 'resume'}
    config['base_seed'] = 20261012
    manifest = identity(config)
    directory = args.resume or ROOT / 'benchmarks' / ('campaign-extra-trees-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if args.resume:
        if json.loads((directory / 'manifest.json').read_text()) != manifest:
            raise RuntimeError('Implementation, environment or configuration changed; start a new campaign.')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'manifest.json', manifest)
    banking.N_ROWS, banking.N_FEATURES = args.rows, args.features
    banking.N_TREES, banking.N_JOBS = args.trees, args.jobs
    banking.BOOTSTRAP = args.bootstrap
    banking.MAX_BINS = None if args.max_bins == 0 else args.max_bins
    models = banking.MODEL_NAMES if args.all_models else [banking.BANKAI_ON, 'Extra Trees', banking.BANKAI_OFF]
    if args.extra_only:
        models = [banking.BANKAI_ON, 'Extra Trees']
    if args.models:
        models = list(dict.fromkeys(args.models))
    if args.all_models:
        primary = [banking.BANKAI_ON, 'Extra Trees', banking.BANKAI_OFF]
        models = primary + [name for name in models if name not in primary]
    scenarios = ['transaction_fraud'] + ([s for s in banking.SCENARIOS if s != 'transaction_fraud'] if args.all_scenarios else [])
    if args.scenarios:
        scenarios = [s for s in ['transaction_fraud', *banking.SCENARIOS] if s in args.scenarios]
        scenarios = list(dict.fromkeys(scenarios))
    records = [json.loads(p.read_text()) for p in sorted(directory.glob('fit-*.json'))]
    done = {(r['scenario'], r['model'], r['repeat']) for r in records}
    print(directory, flush=True)
    for scenario in scenarios:
        # Stable scenario offsets regardless of subset/order.
        offset = list(banking.SCENARIOS).index(scenario) * 1_000_000
        for repeat in range(-args.warmups, args.repeats):
            if repeat >= 0 and all((scenario, name, repeat) in done for name in models):
                continue
            seed = config['base_seed'] + offset + (repeat * 10_000 if repeat >= 0 else 777) + 500
            model_seed = config['base_seed'] + offset + (repeat * 10_000 if repeat >= 0 else 777) + 900
            X, y, informative = banking.make_dataset(scenario, seed)
            tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=(seed+73) % (2**31-1))
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            del X
            positive_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            order = models if repeat % 2 == 0 else models[::-1]
            for name in order:
                if (scenario, name, repeat) in done:
                    continue
                model = banking.estimator_factory(name, model_seed, positive_weight, scenario)
                print(scenario, repeat, name, flush=True)
                start = time.perf_counter()
                with threadpool_limits(limits=args.jobs):
                    model.fit(Xtr, ytr)
                fit_seconds = time.perf_counter() - start
                if repeat < 0:
                    del model
                    gc.collect()
                    continue
                # Commit fit timing immediately; incomplete metrics are never resumed as completed fits.
                key = f'{scenario}-{repeat}-{models.index(name)}'
                write_json(directory / f'timing-{key}.json', {'fit_seconds': fit_seconds,
                           'params': model.get_params(), 'resolved_params': resolved_params(model, len(ytr), Xtr.shape[1]),
                           'data_seed': seed, 'model_seed': model_seed})
                with threadpool_limits(limits=args.jobs):
                    probability = model.predict_proba(Xte)[:, 1]
                importance = np.asarray(model.feature_importances_, dtype=float)
                top = np.argsort(-importance, kind='stable')[:24]
                count = int(informative[top].sum())
                nodes, depth = structure(model)
                row = {'scenario': scenario, 'model': name, 'repeat': repeat, 'data_seed': seed,
                       'model_seed': model_seed, 'fit_seconds': fit_seconds,
                       'feature_average_precision': average_precision_score(informative, importance),
                       'recall_at_24': count / informative.sum(), 'precision_at_24': count / len(top),
                       'informative_missing': np.flatnonzero(informative & ~np.isin(np.arange(args.features), top)).tolist(),
                       'roc_auc': roc_auc_score(yte, probability), 'pr_auc': average_precision_score(yte, probability),
                       'nodes': nodes, 'depth': depth, 'importance': importance.tolist(), 'params': model.get_params()}
                row['resolved_params'] = resolved_params(model, len(ytr), Xtr.shape[1])
                row['heldout_leaf_positive_counts'] = leaf_positive_counts(model, Xte, yte)
                if hasattr(model, '_forest'):
                    global_gain = np.asarray(model._forest.gain_importances())
                    row['global_gain_feature_ap'] = average_precision_score(informative, global_gain)
                    row['global_gain_recall_at_24'] = int(informative[np.argsort(-global_gain, kind='stable')[:24]].sum()) / int(informative.sum())
                if args.instrumented and hasattr(model, 'fit_profile_'):
                    row['profile'] = model.fit_profile_
                row['missing_features'] = row.pop('informative_missing')
                row['informative_missing'] = len(row['missing_features'])
                write_json(directory / f'fit-{key}.json', row)
                records.append(row)
                done.add((scenario, name, repeat))
                summarize(directory, records)
                print({k: row[k] for k in ('fit_seconds', 'feature_average_precision', 'informative_missing')}, flush=True)
                del model
                gc.collect()
            del Xtr, Xte, ytr, yte, y
            gc.collect()
    summarize(directory, records)


if __name__ == '__main__':
    main()
