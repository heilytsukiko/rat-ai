"""QLoRA competitors on one GPU; private scores stay in the selected server."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('WANDB_DISABLED', 'true')

MODELS = {'qwen25': 'Qwen/Qwen2.5-VL-3B-Instruct', 'qwen3': 'Qwen/Qwen3-VL-2B-Instruct',
          'qwen25-7b': 'Qwen/Qwen2.5-VL-7B-Instruct', 'qwen3-4b': 'Qwen/Qwen3-VL-4B-Instruct'}
PROMPT = ('These four frames are in chronological order from a short clip of a bank cash desk. '
          'Classify this clip for human review of suspicious cash handling. '
          'Answer only 1 if it needs review, otherwise 0. Do not identify people or infer a legal rule.')


def prompt_messages(modality='ordered_images'):
    media = [{'type':'video'}] if modality=='video' else [{'type':'image'} for _ in range(4)]
    return [{'role': 'user', 'content': media +
             [{'type': 'text', 'text': PROMPT}]}]


def inputs_for(processor, frames, label=None, modality='ordered_images', fps=1.0):
    import torch
    from PIL import Image
    messages = prompt_messages(modality)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [Image.fromarray(image) for image in frames]
    def process(text):
        if modality!='video':
            return processor(text=[text],images=images,return_tensors='pt')
        options={'fps':fps,'do_sample_frames':False}
        if processor.__class__.__name__.startswith('Qwen3'):
            options['video_metadata']=[{'fps':fps,'duration':len(frames)/fps,
                                       'total_num_frames':len(frames),'frames_indices':list(range(len(frames)))}]
        return processor(text=[text],videos=[frames],return_tensors='pt',**options)
    prefix = process(prompt)
    if label is None:
        return prefix
    full_messages = messages + [{'role': 'assistant', 'content': str(label)}]
    text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    batch = process(text)
    n = prefix['input_ids'].shape[1]
    if not torch.equal(batch['input_ids'][:, :n], prefix['input_ids']):
        raise ValueError('Assistant-only masking boundary mismatch')
    labels = batch['input_ids'].clone()
    labels[:, :n] = -100
    batch['labels'] = labels
    return batch


def class_token_ids(processor):
    ids = [processor.tokenizer.encode(str(label), add_special_tokens=False) for label in (0, 1)]
    if any(len(item) != 1 for item in ids):
        raise ValueError('Expected single-token binary labels')
    return [item[0] for item in ids]


def clip_fps(center,duration):
    span=min(duration-.001,center+1.5)-max(0,center-1.5)
    return 3/max(.001,span)


def synthetic_fixture(root):
    """A non-bank fixture for testing actual optimizer/save/evaluation steps."""
    import numpy as np
    root.mkdir(parents=True, exist_ok=True)
    samples, groups = [], []
    for index in range(4):
        label = index % 2
        group = {'group': index, 'split': 'train' if index < 2 else 'validation',
                 'label': label, 'samples': [], 'annotated_event_times_sec': [2.5] if label else []}
        for center in (.5, 2.5):
            number = len(samples)
            filename = f'{number:05d}.npz'
            images = np.full((4,336,336,3), 90+index*10, dtype=np.uint8)
            for frame in range(4):
                x = 20 + frame*20 if label else 20
                images[frame,100:150,x:x+40] = [200,30,20]
            np.savez_compressed(root/filename, frames=images)
            samples.append({'file': filename, 'label': label, 'group': index,
                            'center_sec': center, 'split': group['split']})
            group['samples'].append(number)
        groups.append(group)
    (root/'dataset.json').write_text(json.dumps({'samples':samples,'groups':groups,
                                                'limitations':['SYNTHETIC PIPELINE TEST ONLY']}),encoding='utf-8')
    return root


def metrics(groups, scores):
    decisions = [max(scores[i] for i in group['samples']) for group in groups]
    labels = [g['label'] for g in groups]
    tp = sum(y == 1 and s >= .5 for y, s in zip(labels, decisions))
    tn = sum(y == 0 and s < .5 for y, s in zip(labels, decisions))
    fp, fn = labels.count(0) - tn, labels.count(1) - tp
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': (tp + tn) / len(groups), 'threshold': .5}


def evaluate(model, processor, dataset, root, modality='ordered_images'):
    import numpy as np
    import torch
    model.eval()
    ids = class_token_ids(processor)
    groups = [g for g in dataset['groups'] if g['split'] == 'validation']
    scores = {}
    with torch.inference_mode():
        for group in groups:
            for index in group['samples']:
                row = dataset['samples'][index]
                frames = np.load(root / row['file'], allow_pickle=False)['frames']
                batch = inputs_for(processor,frames,modality=modality,
                                   fps=clip_fps(row['center_sec'],group.get('duration_sec',4))).to('cuda')
                output = model(**batch, use_cache=False)
                scores[index] = float(output.logits[0, -1, ids].float().softmax(-1)[1])
                del output, batch
            print(json.dumps({'stage': 'validation', 'videos_completed':
                              sum(all(i in scores for i in g['samples']) for g in groups)}), flush=True)
    rows = [{'video_id': str(g['group']), 'label': g['label'],
             'annotated_event_times_sec': g['annotated_event_times_sec'],
             'scan_times_sec': [dataset['samples'][i]['center_sec'] for i in g['samples']],
             'scan_scores': [scores[i] for i in g['samples']],
             'video_score': max(scores[i] for i in g['samples'])} for g in groups]
    return {'validation_video_results': rows, 'validation_video_metrics': metrics(groups, scores)}


def main():
    import numpy as np
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,
                              Qwen3VLForConditionalGeneration, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--synthetic-fixture', action='store_true')
    parser.add_argument('--native-video', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 5:
        parser.error('epochs must be 1..5')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.synthetic_fixture and args.dataset is not None:
        parser.error('Synthetic fixture and real dataset are mutually exclusive')
    if args.synthetic_fixture:
        args.dataset = synthetic_fixture(args.output/'synthetic-fixture')
    status_path = args.output / 'status.json'
    start = time.monotonic()

    def status(**fields):
        status_path.write_text(json.dumps({'model': MODELS[args.model], 'elapsed_sec': time.monotonic()-start,
                                           **fields}, indent=2), encoding='utf-8')
        print(json.dumps(fields), flush=True)

    try:
        set_seed(42)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        status(stage='downloading', training_started=False)
        repo = MODELS[args.model]
        revision = HfApi().model_info(repo).sha
        location = snapshot_download(repo, revision=revision,
                                     allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
        config = {'model': repo, 'revision': revision, 'method': 'QLoRA NF4', 'seed': 42,
                  'rank': 8, 'alpha': 16, 'epochs': args.epochs, 'learning_rate': .0001,
                  'trainable_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  'prompt': PROMPT, 'frames_per_clip': 4, 'input_size': 336,
                  'legal_classification': False}
        config['data_kind'] = 'synthetic' if args.synthetic_fixture else 'bank_review'
        config['input_modality'] = 'video' if args.native_video else 'ordered_images'
        (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        processor = AutoProcessor.from_pretrained(location, local_files_only=True, trust_remote_code=False,
                                                 min_pixels=144*28*28, max_pixels=144*28*28)
        cls = Qwen2_5_VLForConditionalGeneration if args.model.startswith('qwen25') else Qwen3VLForConditionalGeneration
        model = cls.from_pretrained(location, local_files_only=True, trust_remote_code=False,
                                    use_safetensors=True, device_map={'': 0}, dtype=torch.bfloat16,
                                    attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
                                        load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        status(stage='probe', training_started=False)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                               target_modules=config['trainable_targets'], task_type='CAUSAL_LM'))
        # No bank frames are used for the initial forward/backward compatibility check.
        synthetic = np.full((4, 336, 336, 3), 127, dtype=np.uint8)
        synthetic[:, 100:150, 100:150] = [200, 30, 20]
        model.train()
        probe = inputs_for(processor,synthetic,0,modality=config['input_modality']).to('cuda')
        loss = model(**probe, use_cache=False).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        del probe, loss
        torch.cuda.empty_cache()
        status(stage='probe_passed', training_started=False,
               peak_vram_bytes=torch.cuda.max_memory_allocated(),
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        if args.probe_only:
            return
        if args.dataset is None:
            raise ValueError('--dataset required for training')
        root = args.dataset
        dataset = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
        for split in ('train', 'validation'):
            selected = [g for g in dataset['groups'] if g['split'] == split]
            if {g['label'] for g in selected} != {0, 1}:
                raise ValueError('Both classes required in each split')
            for group in selected:
                if any(dataset['samples'][i]['split'] != split or
                       dataset['samples'][i]['group'] != group['group'] for i in group['samples']):
                    raise ValueError('Cross-split sample detected')
        initial = {name: parameter.detach().float().cpu().clone()
                   for name, parameter in model.named_parameters() if parameter.requires_grad}
        baseline = evaluate(model,processor,dataset,root,config['input_modality'])
        (args.output / 'baseline.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
        status(stage='training', training_started=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001)
        groups = [g for g in dataset['groups'] if g['split'] == 'train']
        class_groups = {y: [g for g in groups if g['label'] == y] for y in (0, 1)}
        rng = np.random.default_rng(42)
        history = []
        # Equal class/video contribution. Each epoch visits every prepared train clip once.
        train_ids = [i for g in groups for i in g['samples']]
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0
            order = rng.permutation(train_ids).tolist()
            for step, index in enumerate(order):
                row = dataset['samples'][index]
                group = dataset['groups'][row['group']]
                frames = np.load(root / row['file'], allow_pickle=False)['frames']
                batch = inputs_for(processor,frames,row['label'],modality=config['input_modality'],
                                   fps=clip_fps(row['center_sec'],group.get('duration_sec',4))).to('cuda')
                raw_loss = model(**batch, use_cache=False).loss
                weight = len(train_ids)/(2*len(class_groups[row['label']])*len(group['samples']))
                (raw_loss * weight / min(4, len(order) - (step//4)*4)).backward()
                total += float(raw_loss.detach())
                if (step + 1) % 4 == 0 or step + 1 == len(order):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                del batch, raw_loss
                if (step+1) % 10 == 0:
                    status(stage='training', training_started=True, epoch=epoch+1,
                           completed=step+1, total=len(order), mean_loss=total/(step+1))
            history.append({'epoch': epoch+1, 'mean_loss': total/len(order)})
            model.save_pretrained(args.output / 'adapter', safe_serialization=True)
            processor.save_pretrained(args.output / 'adapter')
        final = evaluate(model,processor,dataset,root,config['input_modality'])
        report = {'training_completed': True, **final, 'baseline_metrics': baseline['validation_video_metrics'],
                  'data_kind': config['data_kind'],
                  'history': history, 'model': repo, 'revision': revision,
                  'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                  'adapter_change_l2': sum(float((p.detach().float().cpu()-initial[n]).square().sum())
                                           for n,p in model.named_parameters() if p.requires_grad)**.5,
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'elapsed_sec': time.monotonic()-start,
                  'limitations': dataset['limitations'] + ['Repeated validation; no independent test.',
                      'Scores are conditional binary-token scores, not calibrated probabilities.',
                      ('Native video tokens; nominal clip FPS is derived from sampling times.' if args.native_video
                       else 'Ordered frames, not native video tokenization.'),
                      'Action types are not supervised.']}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        status(stage='complete', training_started=True, training_completed=True,
               metrics=final['validation_video_metrics'])
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        status(stage='failed', error=type(error).__name__, training_completed=False)
        raise


if __name__ == '__main__':
    main()
