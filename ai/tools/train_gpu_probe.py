"""Frozen Qwen video features plus a train-only weighted ridge classifier.

Private feature caches and per-video reports must stay outside Git.
No validation labels enter feature extraction, scaling or fitting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT = ('Describe the visible objects, their spatial relationships, and how they change '
          'over time. Do not identify people, read private information or infer intentions.')


def fit_probe(features, rows, alpha=10.0):
    """Fixed regularization, balanced classes and equal weight per source video."""
    import numpy as np
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all() or alpha <= 0:
        raise ValueError('Invalid features or regularization')
    selected = [i for i, row in enumerate(rows) if row['split'] == 'train']
    train = [rows[i] for i in selected]
    if {row['label'] for row in train} != {0, 1}:
        raise ValueError('Both training classes are required')
    groups = {}
    for row in train:
        key = row['group']
        if key in groups and groups[key][0] != row['label']:
            raise ValueError('Inconsistent group labels')
        label, count = groups.get(key, (row['label'], 0))
        groups[key] = (label, count + 1)
    class_groups = {label: sum(value[0] == label for value in groups.values()) for label in (0, 1)}
    weights = np.array([1 / (class_groups[r['label']] * groups[r['group']][1]) for r in train])
    weights *= len(train) / weights.sum()
    z = x[selected]
    mean = np.average(z, axis=0, weights=weights)
    z = z - mean
    target = np.array([2 * r['label'] - 1 for r in train], dtype=np.float64)
    bias = float(np.average(target, weights=weights))
    root = np.sqrt(weights)
    weighted = z * root[:, None]
    dual = np.linalg.solve(weighted @ weighted.T + alpha * np.eye(len(train)),
                           (target - bias) * root)
    coefficients = weighted.T @ dual
    return {'mean': mean, 'coefficients': coefficients, 'bias': bias}


def score_probe(features, fitted):
    import numpy as np
    margins = (np.asarray(features) - fitted['mean']) @ fitted['coefficients'] + fitted['bias']
    # A bounded ranking score, not a calibrated probability of a violation.
    return 1 / (1 + np.exp(-2 * np.clip(margins, -20, 20)))


def video_report(dataset, scores):
    from train_detector import binary_metrics
    rows = []
    for group in dataset['groups']:
        if group['split'] != 'validation':
            continue
        indices = group['samples']
        rows.append({'video_id': str(group['group']), 'label': group['label'],
                     'annotated_event_times_sec': group['annotated_event_times_sec'],
                     'scan_times_sec': [dataset['samples'][i]['center_sec'] for i in indices],
                     'scan_scores': [float(scores[i]) for i in indices],
                     'video_score': float(max(scores[i] for i in indices))})
    return {'training_completed': True, 'validation_video_results': rows,
            'validation_video_metrics': binary_metrics([r['label'] for r in rows],
                                                       [r['video_score'] for r in rows])}


def main():
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from train_gpu_vlm import clip_fps, synthetic_fixture
    from evaluate_candidate_policy import evaluate
    from compare_detectors import validate_report, localization_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic-fixture', action='store_true')
    args = parser.parse_args()
    if bool(args.dataset) == bool(args.synthetic_fixture):
        parser.error('Choose either --dataset or --synthetic-fixture')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.synthetic_fixture:
        args.dataset = synthetic_fixture(args.output/'synthetic-fixture')
    started = time.monotonic()
    def status(**fields):
        (args.output/'status.json').write_text(json.dumps(fields), encoding='utf-8')
        print(json.dumps(fields), flush=True)
    source = (args.dataset/'dataset.json').read_bytes()
    dataset = json.loads(source)
    config = json.loads(args.model_config.read_text(encoding='utf-8'))
    if config['model'] != 'Qwen/Qwen3-VL-2B-Instruct':
        raise ValueError('Expected the cached Qwen3-VL-2B base model')
    specification = {'model': config['model'], 'revision': config['revision'], 'prompt': PROMPT,
                     'pooling': 'unit-normalized last token + mean video tokens',
                     'dataset_sha256': hashlib.sha256(source).hexdigest(), 'alpha': 10.0,
                     'data_kind': 'synthetic' if args.synthetic_fixture else 'bank_review'}
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    cache = args.output/'features.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved['fingerprint']) != fingerprint:
                raise ValueError('Feature cache does not match dataset/model/prompt')
            features = saved['features'].copy()
    else:
        torch.set_num_threads(4)
        location = snapshot_download(config['model'], revision=config['revision'], local_files_only=True)
        processor = AutoProcessor.from_pretrained(location, local_files_only=True, trust_remote_code=False,
                                                  min_pixels=336*336, max_pixels=336*336)
        model = Qwen3VLForConditionalGeneration.from_pretrained(location, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)).eval()
        model.requires_grad_(False)
        video_id = model.config.video_token_id
        messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        durations = {group['group']: group.get('duration_sec', 4) for group in dataset['groups']}
        features = []
        with torch.inference_mode():
            for index, row in enumerate(dataset['samples']):
                with np.load(args.dataset/row['file'], allow_pickle=False) as sample:
                    frames = sample['frames']
                fps = clip_fps(row['center_sec'], durations[row['group']])
                metadata = [{'fps': fps, 'duration': len(frames)/fps, 'total_num_frames': len(frames),
                             'frames_indices': list(range(len(frames)))}]
                batch = processor(text=[text], videos=[frames], fps=fps, do_sample_frames=False,
                                  video_metadata=metadata, return_tensors='pt').to('cuda')
                output = model.model(**batch, use_cache=False, return_dict=True)
                hidden = output.last_hidden_state[0].float()
                mask = batch['input_ids'][0] == video_id
                if not bool(mask.any()):
                    raise ValueError('No video tokens to pool')
                blocks = [hidden[-1], hidden[mask].mean(0)]
                vector = torch.cat([block / block.norm().clamp_min(1e-8) for block in blocks])
                features.append(vector.cpu().numpy())
                del batch, output, hidden, blocks, vector
                if (index + 1) % 10 == 0 or index + 1 == len(dataset['samples']):
                    status(stage='extracting', completed=index+1, total=len(dataset['samples']),
                           elapsed_sec=time.monotonic()-started)
        features = np.stack(features)
        np.savez_compressed(cache, features=features, fingerprint=np.array(fingerprint))
        del model
        torch.cuda.empty_cache()
    if len(features) != len(dataset['samples']):
        raise ValueError('Feature/sample count mismatch')
    fitted = fit_probe(features, dataset['samples'])
    scores = score_probe(features, fitted)
    np.savez_compressed(args.output/'classifier.npz', **fitted, fingerprint=np.array(fingerprint))
    report = video_report(dataset, scores)
    report['config'] = specification
    report['elapsed_sec'] = time.monotonic()-started
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'config.json').write_text(json.dumps(specification, indent=2), encoding='utf-8')
    confirmed = evaluate(report, {'min_windows': 2, 'representative': 'center'})
    _, measured = validate_report(confirmed)
    summary = {'method': 'frozen Qwen3-VL-2B + weighted ridge', 'config': specification,
               'metrics': measured, 'localization': localization_metrics(report['validation_video_results'],
                   .5, confirmed['candidate_policy']), 'elapsed_sec': report['elapsed_sec'],
               'independent_test': False, 'legal_classification': False}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    status(stage='complete', metrics=measured, elapsed_sec=report['elapsed_sec'])


if __name__ == '__main__':
    main()
