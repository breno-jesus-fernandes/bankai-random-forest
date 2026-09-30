"""Summarize the predeclared 3 dataset × 3 model seed quality matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def summarize(root):
    pairs = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        b, c = directory / 'baseline-0.json', directory / 'candidate-0.json'
        if b.exists() and c.exists():
            baseline, candidate = json.loads(b.read_text()), json.loads(c.read_text())
            assert baseline['dataset'] == candidate['dataset']
            assert baseline['data_seed'] == candidate['data_seed']
            assert baseline['model_seed'] == candidate['model_seed']
            pairs.append((baseline, candidate))
    seeds = {(b['data_seed'], b['model_seed']) for b, _ in pairs}
    data_seeds = {s[0] for s in seeds}
    model_seeds = {s[1] for s in seeds}
    complete = len(pairs) == len(seeds) == 9 and len(data_seeds) == len(model_seeds) == 3
    result = dict(complete=complete, pairs=len(pairs), metrics={},
                  identical_predictions=all(b['prediction_sha256'] == c['prediction_sha256'] for b, c in pairs))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        delta = np.array([c['metrics'][metric] - b['metrics'][metric] for b, c in pairs])
        mean = float(np.mean(delta)) if len(delta) else None
        lower = mean - float(t.ppf(.95, len(delta)-1) * np.std(delta, ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
        result['metrics'][metric] = dict(deltas=delta.tolist(), mean_delta=mean,
                                       lower_one_sided_95=lower,
                                       passed=complete and mean >= -.002 and lower > -.002)
    result['passed'] = complete and all(m['passed'] for m in result['metrics'].values())
    result['method'] = 'One-sided paired Student t interval across nine seed combinations; shared dataset seeds imply dependence, so interpret cautiously.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / 'quality-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
