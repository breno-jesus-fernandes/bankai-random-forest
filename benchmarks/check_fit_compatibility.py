"""Cross-build probability/OOB/importance and serialized-model compatibility checks.

Run baseline first with --write, then candidate without it. Artifacts live in /tmp.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from bankai_random_forest import BankaiRandomForestClassifier


def signature(model, x):
    values = dict(pred=model.predict(x), proba=model.predict_proba(x),
                  importance=model.feature_importances_)
    if hasattr(model, 'oob_decision_function_'):
        values['oob'] = model.oob_decision_function_
    return {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--directory', type=Path, default=Path('/tmp/bankai-fit-compatibility'))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    x, y = make_classification(n_samples=1000, n_features=16, n_informative=8, random_state=182)
    x = x.astype(np.float32)
    cases = dict(standard={}, oob={'oob_score': True}, weighted={}, nan={}, sparse={},
                 warm={'warm_start': True}, pruning={'ccp_alpha': .002},
                 monotonic={'monotonic_cst': [1]+[0]*15}, entropy={'criterion': 'entropy'},
                 permutation={'importance_type': 'permutation'}, exact={'max_bins': None},
                 multiclass={}, balanced={'class_weight': 'balanced_subsample'})
    result = {}
    for name, overrides in cases.items():
        train, labels = x.copy(), y.copy()
        kwargs = dict(n_estimators=8, n_jobs=2, max_bins=31, max_depth=8, random_state=92)
        kwargs.update(overrides)
        weights = None
        if name == 'weighted':
            weights = np.linspace(.1, 3, len(y))
        if name == 'nan':
            train[::13, ::3] = np.nan
        if name == 'sparse':
            train = sparse.csr_matrix(train)
        if name == 'multiclass':
            labels = (labels + np.arange(len(y)) % 3) % 3
        def fit():
            model = BankaiRandomForestClassifier(**kwargs).fit(train, labels, sample_weight=weights)
            if name == 'warm':
                model.n_estimators = 12
                model.fit(train, labels, sample_weight=weights)
            return model
        model = fit()
        observed = signature(model, train)
        assert signature(fit(), train) == observed, (name, 'fixed seed repeatability')
        assert signature(pickle.loads(pickle.dumps(model)), train) == observed, (name, 'roundtrip')
        file = args.directory / f'{name}.pkl'
        if args.write:
            file.write_bytes(pickle.dumps(model))
            (args.directory / f'{name}.json').write_text(json.dumps(observed))
        else:
            expected = json.loads((args.directory / f'{name}.json').read_text())
            assert observed == expected, (name, 'cross-build output', observed, expected)
            assert signature(pickle.loads(file.read_bytes()), train) == expected, (name, 'old model compatibility')
        result[name] = {'passed': True, 'signature': observed}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
