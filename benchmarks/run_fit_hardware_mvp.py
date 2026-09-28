"""Reproducible fit-only comparison; run once per release build in a fresh process."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import lightgbm
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from bankai_random_forest import BankaiRandomForestClassifier


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--trees', type=int, default=40)
    parser.add_argument('--jobs', type=int, nargs='+', default=[1, 4])
    args = parser.parse_args()
    records = []
    for name, rows, features, classes, bins, leaves, depth, min_leaf in [
        ('binary', 20000, 32, 2, 63, 63, 12, 20),
        ('wide_multiclass', 12000, 128, 4, 63, 63, 12, 20),
        ('deep_binary', 20000, 32, 2, 63, 511, 20, 5),
    ]:
        x, y = make_classification(n_samples=rows + 4000, n_features=features,
                                  n_informative=16, n_classes=classes, random_state=1729)
        x = np.ascontiguousarray(x)
        for jobs in args.jobs:
            # Match nominal tree/depth/leaf/bin/thread limits. Algorithms still differ:
            # Bankai uses Gini and bootstrap; LightGBM uses gradient statistics and bagging.
            factories = {
                'bankai': lambda: BankaiRandomForestClassifier(
                    n_estimators=args.trees, max_bins=bins, max_depth=depth,
                    max_leaf_nodes=leaves, min_samples_leaf=min_leaf, max_features=None,
                    max_samples=0.8, n_jobs=jobs, random_state=42),
                'lightgbm': lambda: lightgbm.LGBMClassifier(
                    boosting_type='rf', n_estimators=args.trees, max_bin=bins,
                    max_depth=depth, num_leaves=leaves, min_child_samples=min_leaf,
                    bagging_freq=1, bagging_fraction=0.8, feature_fraction=1.0,
                    n_jobs=jobs, random_state=42, verbosity=-1),
            }
            for factory in factories.values():
                factory().fit(x[:rows], y[:rows])
            for repeat in range(args.repeats):
                for implementation in (list(factories) if repeat % 2 == 0 else list(factories)[::-1]):
                    model = factories[implementation]()
                    start = time.perf_counter()
                    model.fit(x[:rows], y[:rows])
                    seconds = time.perf_counter() - start
                    prediction = model.predict(x[rows:])
                    if implementation == 'bankai':
                        arrays = model._forest.shap_tree_arrays()
                        actual_trees = len(arrays)
                        actual_leaves = sum(sum(child == -1 for child in tree[0]) for tree in arrays)
                    else:
                        info = model.booster_.dump_model()['tree_info']
                        actual_trees = len(info)
                        actual_leaves = sum(tree['num_leaves'] for tree in info)
                    record = dict(case=name, rows=rows, features=features, classes=classes,
                                  bins=bins, jobs=jobs, trees=args.trees, repeat=repeat,
                                  max_leaves=leaves, max_depth=depth, min_leaf=min_leaf,
                                  actual_trees=actual_trees, actual_leaves=actual_leaves,
                                  implementation=implementation, fit_seconds=seconds,
                                  accuracy=accuracy_score(y[rows:], prediction),
                                  prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest())
                    records.append(record)
                    print(args.label, name, jobs, implementation, repeat, round(seconds, 4), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    for path in sorted([*Path('src').rglob('*.rs'), *Path('crates').rglob('*.rs')]):
        source_digest.update(str(path).encode())
        source_digest.update(path.read_bytes())
    from bankai_random_forest import _core
    args.output.write_text(json.dumps(dict(label=args.label,
        source_sha256=source_digest.hexdigest(),
        extension_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        platform=platform.platform(), machine=platform.machine(), cpus=os.cpu_count(),
        python=platform.python_version(), numpy=np.__version__, lightgbm=lightgbm.__version__,
        records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
