"""Run a trained GPU adapter using the same frame preparation and binary prompt."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from train_gpu_vlm import inputs_for, class_token_ids, clip_fps
from prepare_gpu_dataset import frame_clip


def analyze(video, run, threshold=.5):
    import cv2
    import numpy as np
    import torch
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen3VLForConditionalGeneration, BitsAndBytesConfig)
    from huggingface_hub import snapshot_download
    from peft import PeftModel, prepare_model_for_kbit_training
    from rat_ai.inference import candidate_events
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    if report.get('training_completed') is not True or report.get('adapter_change_l2', 0) <= 0:
        raise ValueError('A completed, modified adapter is required')
    if report.get('data_kind') != 'bank_review':
        raise ValueError('A synthetic-test adapter cannot be deployed for bank review')
    architectures = {'Qwen/Qwen2.5-VL-3B-Instruct': (Qwen2_5_VLForConditionalGeneration,
                                                   'qwen2_5_vl_3b_qlora_v1'),
                     'Qwen/Qwen3-VL-2B-Instruct': (Qwen3VLForConditionalGeneration,
                                                  'qwen3_vl_2b_qlora_v1'),
                     'Qwen/Qwen2.5-VL-7B-Instruct': (Qwen2_5_VLForConditionalGeneration,
                                                   'qwen2_5_vl_7b_qlora_v1'),
                     'Qwen/Qwen3-VL-4B-Instruct': (Qwen3VLForConditionalGeneration,
                                                  'qwen3_vl_4b_qlora_v1')}
    cls, name = architectures[config['model']]
    # Runtime never contacts an inference provider or downloads model files.
    location = snapshot_download(config['model'], revision=config['revision'], local_files_only=True)
    processor = AutoProcessor.from_pretrained(location, local_files_only=True,
                                             min_pixels=144*28*28, max_pixels=144*28*28)
    base = cls.from_pretrained(location, local_files_only=True, trust_remote_code=False,
                               use_safetensors=True, device_map={'': 0}, dtype=torch.bfloat16,
                               attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
                                   load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, run/'adapter', is_trainable=False).eval()
    ids = class_token_ids(processor)
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Invalid video')
        duration = count/fps
    finally:
        capture.release()
    if not 0 < duration <= 180:
        raise ValueError('Video duration must be at most 180 seconds')
    scores, times = [], np.arange(min(.5, duration/2), duration, 2).tolist()
    started = time.monotonic()
    torch.set_num_threads(4)
    with torch.inference_mode():
        for center in times:
            batch = inputs_for(processor,frame_clip(video,center,duration),
                               modality=config.get('input_modality','ordered_images'),
                               fps=clip_fps(center,duration)).to('cuda')
            output = model(**batch, use_cache=False)
            scores.append(float(output.logits[0, -1, ids].float().softmax(-1)[1]))
            del output, batch
    peak = int(np.argmax(scores))
    scored = {'scan_times_sec': times, 'scan_scores': scores,
              'video_score': scores[peak], 'peak_time_sec': times[peak],
              'processing_sec': time.monotonic()-started}
    policy = {'min_windows': 2, 'representative': 'center'}
    return {**scored, 'model': name, 'experimental': True, 'threshold': threshold,
            'candidate_policy': policy, 'candidates': candidate_events(scored, threshold, **policy),
            'note': 'Binary adapter scores; no legal verdict or verified action type.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = analyze(args.video, args.run)
    result['total_elapsed_sec'] = time.monotonic()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result), encoding='utf-8')
    print(json.dumps({'status': 'done', 'candidates': len(result['candidates']),
                      'elapsed_sec': result['total_elapsed_sec']}))


if __name__ == '__main__':
    main()
