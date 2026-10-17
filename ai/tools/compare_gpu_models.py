"""Compare two trained competitors and one fixed equal-weight ensemble."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_detectors import validate_report, localization_metrics
from evaluate_candidate_policy import evaluate
from train_detector import binary_metrics


def compare(first, second):
    left, _ = validate_report(first)
    right, _ = validate_report(second)
    if left.keys() != right.keys():
        raise ValueError('Competitors must use identical validation videos')
    for key, row in left.items():
        other = right[key]
        if any(row.get(name) != other.get(name) for name in
               ('label', 'scan_times_sec', 'annotated_event_times_sec')):
            raise ValueError('Competitors must use identical labels, inputs and point annotations')
    ensemble = copy.deepcopy(first)
    for row in ensemble['validation_video_results']:
        row['scan_scores'] = [(a+b)/2 for a,b in zip(row['scan_scores'], right[row['video_id']]['scan_scores'])]
        row['video_score'] = max(row['scan_scores'])
    rows = ensemble['validation_video_results']
    ensemble['validation_video_metrics'] = binary_metrics([r['label'] for r in rows], [r['video_score'] for r in rows])
    policy = {'min_windows': 2, 'representative': 'center'}
    results = {}
    for name, report in [('qwen25', first), ('qwen3', second), ('equal_mean', ensemble)]:
        confirmed = evaluate(report, policy)
        _, m = validate_report(confirmed)
        results[name] = {'metrics': m, 'localization': localization_metrics(
            report['validation_video_results'], .5, policy)}
    # Prioritize avoiding misses, then false alarms and annotated-point localization.
    def rank(name):
        item = results[name]
        return (item['metrics']['fn'], item['metrics']['fp'],
                -item['localization']['matched_points'], item['localization']['negative_video_candidates'],
                name == 'equal_mean')
    choice = min(results, key=rank)
    return {'results': results, 'recommended_by_validation': choice, 'candidate_policy': policy,
            'ensemble_weights': [.5, .5], 'independent_test': False,
            'limitations': ['Eight repeatedly used validation videos; selection may overfit.',
                            'Binary frame-sequence review scores do not establish event types or legal meaning.',
                            'Winner needs runtime and comparison against the existing detector before deployment.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.first.read_text(encoding='utf-8')),
                     json.loads(args.second.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
