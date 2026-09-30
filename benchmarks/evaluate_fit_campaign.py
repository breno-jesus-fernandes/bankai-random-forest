"""Evaluate timing, memory and quality gates from persisted campaign evidence."""
import argparse
import json
from pathlib import Path
from run_fit_exploration import summarize


def scenario(directory):
    records = []
    for variant in ('baseline', 'candidate'):
        records += [json.loads(p.read_text()) for p in directory.glob(f'{variant}-*.json')]
    measured = [r for r in records if not r['warmup']]
    counts = {v: len([r for r in measured if r['variant'] == v]) for v in ('baseline', 'candidate')}
    by_variant = {v: {r['repeat']: r for r in measured if r['variant'] == v} for v in counts}
    paired = by_variant['baseline'].keys() & by_variant['candidate'].keys()
    consistent = all(by_variant['baseline'][i]['dataset'] == by_variant['candidate'][i]['dataset']
                     and by_variant['baseline'][i]['params'] == by_variant['candidate'][i]['params'] for i in paired)
    complete = len(paired) >= 5 and consistent and all(any(r['variant'] == v and r['warmup'] for r in records) for v in counts)
    if not measured or not all(counts.values()):
        return {'complete': False, 'passed': False}
    summary = summarize(measured)
    b, c = summary['baseline'], summary['candidate']
    max_rss = {v: max(r['fit_peak_rss'] for r in measured if r['variant'] == v) for v in counts}
    summary.update(complete=complete, measured_counts=counts,
                   median_reduction=1-c['median_seconds']/b['median_seconds'],
                   max_fit_rss=max_rss,
                   memory_passed=max_rss['candidate'] <= 1.1*max_rss['baseline'] and c['median_peak_rss'] <= 1.1*b['median_peak_rss'])
    summary['timing_passed'] = c['median_seconds'] <= 1.05*b['median_seconds']
    summary['passed'] = complete and summary['timing_passed'] and summary['memory_passed']
    return summary


def evaluate(root):
    main = scenario(root / 'confirm-main')
    if main['complete']:
        main['timing_passed'] = main['median_reduction'] >= .05 and main['paired']['ratio_ci95'][0] > 1
        main['passed'] = main['timing_passed'] and main['memory_passed']
    controls = {name: scenario(root / 'controls' / name) for name in
                ('jobs-1', 'jobs--1', 'exact', 'float64', 'permutation', 'multiclass', 'sparse', 'nan', 'few')}
    quality_path = root / 'quality' / 'quality-summary.json'
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {'passed': False, 'complete': False}
    result = dict(main=main, controls=controls, quality=quality,
                  numeric_gates_passed=main['passed'] and all(c['passed'] for c in controls.values()) and quality['passed'],
                  note='Suite and compatibility results must also pass; this script evaluates numeric gates only.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = evaluate(args.root)
    (args.root / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
