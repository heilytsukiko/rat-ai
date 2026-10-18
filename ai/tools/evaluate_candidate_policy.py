"""Evaluate declared temporal confirmation on existing local scores; no retraining."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from train_detector import binary_metrics
from rat_ai.inference import candidate_events, validate_candidate_policy


def evaluate(report, policy):
    policy = validate_candidate_policy(policy)
    output = copy.deepcopy(report)
    threshold = output['validation_video_metrics']['threshold']
    output['candidate_policy'] = policy
    output['postprocessing_only'] = True
    for row in output['validation_video_results']:
        row['decision_score'] = max((event['score'] for event in candidate_events(row, threshold, 2, **policy)), default=-1)
    rows = output['validation_video_results']
    output['validation_video_metrics'] = binary_metrics([r['label'] for r in rows],
                                                        [r['decision_score'] for r in rows], threshold)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--min-windows', type=int, default=2)
    parser.add_argument('--representative', choices=('peak', 'center'), default='center')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.report.read_bytes()
    output = evaluate(json.loads(source), {'min_windows': args.min_windows, 'representative': args.representative})
    output['source_report_sha256'] = hashlib.sha256(source).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'postprocessing_only': True, 'metrics': output['validation_video_metrics']}))


if __name__ == '__main__':
    main()
