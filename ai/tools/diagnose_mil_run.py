"""Audit train/validation video metrics from a verified local MIL feature cache."""
import argparse
import hashlib
import json
from pathlib import Path

from train_detector import cached_video_metrics, video_groups


def diagnose(directory, threads=4):
    import torch
    from rat_ai.inference import load_detector
    torch.set_num_threads(threads)
    directory = Path(directory)
    model, config = load_detector(directory / 'best_detector.pt')
    if config.get('objective') != 'mil-video':
        raise ValueError('Diagnosis requires uniform MIL video bags')
    cache = torch.load(directory / 'prefix_features.pt', weights_only=True, map_location='cpu')
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if cache['signature'] != signature:
        raise ValueError('Feature cache does not match checkpoint configuration')
    samples = config['sample_spec']
    features = cache['features']
    if len(samples) != len(features):
        raise ValueError('Feature count does not match samples')
    summaries = {}
    with torch.inference_mode():
        for split in ('train', 'validation'):
            groups = video_groups(samples, split)
            metrics, _ = cached_video_metrics(model, features, groups, config['threshold'])
            summaries[split] = {'unique_videos': len(groups), 'metrics': metrics}
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    full = report['validation_video_metrics']
    agrees = all(full[key] == summaries['validation']['metrics'][key] for key in ('tp', 'tn', 'fp', 'fn'))
    return {'schema_version': 1, 'objective': config['objective'],
            'threshold': config['threshold'], 'splits': summaries,
            'selected_epoch': report['best_epoch'],
            'learning_curve': [{'epoch': entry['epoch'],
                                'train_roc_auc': entry['train_video_metrics']['roc_auc'],
                                'validation_roc_auc': entry['validation_video_metrics']['roc_auc']}
                               for entry in report['history'] if 'train_video_metrics' in entry],
            'cached_validation_confusion_matches_full_decode': agrees,
            'limitations': ['Frozen prefix features use float16 cache; this is a diagnostic, not a fresh decode.',
                            'Checkpoint selected using repeatedly used validation videos.',
                            'Only exact copies are grouped; no independent test set.',
                            'No source frames, labels by video, identifiers or paths are included.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('ai/local/training/mil-diagnosis.json'))
    args = parser.parse_args()
    result = diagnose(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
