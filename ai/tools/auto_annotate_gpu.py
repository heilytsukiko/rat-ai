"""Independent private VLM annotation passes; no human-verified or legal labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ACTIONS = ('cash_counting', 'cash_transfer', 'cash_storage', 'hand_toward_pocket_or_bag',
           'phone_handling', 'leaving_workstation', 'document_handling', 'computer_use',
           'no_visible_action', 'other', 'unclear')
REVIEWS = ('possible_issue', 'ordinary_visible_action', 'unclear')
PROMPT = '''Inspect these chronological video frames independently. Describe ONLY directly visible
actions. Do not identify people, read personal details, infer intentions, ownership or legal guilt.
Four sampled frames cannot prove an action did not happen between frames or outside the image.
A hand near a pocket is not proof of money being taken; papers are not necessarily banknotes.
Use unclear when the relevant object, action or sequence is not visible. Ordinary_visible_action
means only that this short sampled clip shows an ordinary action, NOT that the full video is clean.
Phone handling, turning away and cash counting are not automatically violations. Possible_issue
requires a concrete visible operational concern, such as clearly visible cash moved toward a
pocket/bag or clearly leaving an ongoing cash service unattended; explain the visible basis without
accusing anyone. If the condition is not visible, choose unclear or ordinary_visible_action.
Return ONLY JSON: {"action":"CODE","visibility":"clear|partial|unclear",
"review":"possible_issue|ordinary_visible_action|unclear","observation":"one short Russian sentence",
"uncertainty":"one short Russian sentence, or empty"}.
Allowed action CODE values: ''' + ', '.join(ACTIONS) + '. Limit each Russian sentence to 160 characters.'
MODELS = {'qwen25-7b': 'Qwen/Qwen2.5-VL-7B-Instruct', 'qwen3-4b': 'Qwen/Qwen3-VL-4B-Instruct'}


def parse_answer(answer):
    reason = 'invalid_json'
    try:
        value = json.loads(answer[answer.index('{'):answer.rindex('}')+1])
        reason = 'unexpected_schema'
        if not isinstance(value, dict) or set(value) != {'action','visibility','review','observation','uncertainty'}:
            raise ValueError('Unexpected schema')
        for key, allowed in (('action',ACTIONS),('visibility',('clear','partial','unclear'))):
            reason = 'invalid_'+key
            if value[key] not in allowed:
                raise ValueError('Unexpected category')
        reason = 'invalid_text'
        if any(not isinstance(value[k], str) or len(value[k]) > 500 for k in ('observation','uncertainty')):
            raise ValueError('Invalid text')
        if not value['observation'].strip():
            raise ValueError('Empty observation')
        # Preserve valid visible-action fields, but never infer a verdict from
        # an unknown review category or treat it as an ordinary/positive case.
        review_was_invalid = value['review'] not in REVIEWS
        if review_was_invalid:
            value['review'] = 'unclear'
        return dict(value, parsed=True, review_normalized_to_unclear=review_was_invalid)
    except (ValueError, TypeError, KeyError, AttributeError):
        return {'action':'unclear','visibility':'unclear','review':'unclear',
                'observation':'Автоматический ответ не прошёл проверку формата.',
                'uncertainty':'Разметка этого окна не определена.', 'parsed':False,
                'parse_error':reason}


def consensus(left, right):
    valid = left.get('parsed') is True and right.get('parsed') is True
    clear = valid and left['visibility'] == right['visibility'] == 'clear'
    same = clear and left['action'] == right['action'] and left['action'] not in ('unclear','other')
    review = left['review'] if same and left['review'] == right['review'] else 'unclear'
    return {'action':left['action'] if same else 'unclear', 'review':review,
            'status':'auto_agreement' if same else 'uncertain', 'human_verified':False,
            'legal_verdict':None}


def main():
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
    from train_gpu_vlm import clip_fps, synthetic_fixture
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic-fixture', action='store_true')
    parser.add_argument('--retry-invalid', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=210)
    args = parser.parse_args()
    if not 100 <= args.max_new_tokens <= 512:
        parser.error('max-new-tokens must be between 100 and 512')
    if bool(args.dataset) == bool(args.synthetic_fixture):
        parser.error('Choose a dataset or a synthetic fixture')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.synthetic_fixture:
        args.dataset = synthetic_fixture(args.output.parent/'synthetic-fixture')
    source = (args.dataset/'dataset.json').read_bytes()
    dataset = json.loads(source)
    config = json.loads(args.model_config.read_text(encoding='utf-8'))
    if config['model'] != MODELS[args.model]:
        raise ValueError('Wrong model configuration')
    provenance = {'dataset_sha256':hashlib.sha256(source).hexdigest(), 'model':config['model'],
                  'revision':config['revision'], 'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
                  'data_kind':'synthetic' if args.synthetic_fixture else 'bank_review',
                  'schema_version':1, 'sampling':'existing four-frame three-second windows'}
    result = dict(provenance, predictions=[])
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding='utf-8'))
        if any(result.get(k) != v for k, v in provenance.items()):
            raise ValueError('Existing output provenance mismatch')
    completed = {row['sample_index'] for row in result['predictions']}
    if len(completed) != len(result['predictions']) or not completed.issubset(range(len(dataset['samples']))):
        raise ValueError('Invalid existing sample indices')
    if args.retry_invalid:
        result['predictions'] = [row for row in result['predictions'] if row.get('parsed') is True]
        completed = {row['sample_index'] for row in result['predictions']}
    if len(completed) == len(dataset['samples']):
        print(json.dumps({'status':'complete','windows':len(completed)})); return
    torch.set_num_threads(4)
    location = snapshot_download(config['model'], revision=config['revision'], local_files_only=True)
    processor = AutoProcessor.from_pretrained(location, local_files_only=True, trust_remote_code=False,
                                              min_pixels=336*336, max_pixels=336*336)
    cls = Qwen2_5_VLForConditionalGeneration if args.model == 'qwen25-7b' else Qwen3VLForConditionalGeneration
    model = cls.from_pretrained(location, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)).eval()
    messages = [{'role':'user','content':[{'type':'video'},{'type':'text','text':PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    durations = {g['group']:g.get('duration_sec',4) for g in dataset['groups']}
    started = time.monotonic()
    with torch.inference_mode():
        for index,row in enumerate(dataset['samples']):
            if index in completed:
                continue
            with np.load(args.dataset/row['file'], allow_pickle=False) as item:
                frames = item['frames']
            fps = clip_fps(row['center_sec'], durations[row['group']])
            extra = {'fps':fps,'do_sample_frames':False}
            if args.model.startswith('qwen3-'):
                extra['video_metadata'] = [{'fps':fps,'duration':len(frames)/fps,
                    'total_num_frames':len(frames),'frames_indices':list(range(len(frames)))}]
            batch = processor(text=[text], videos=[frames], return_tensors='pt', **extra).to('cuda')
            output = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            answer = processor.tokenizer.decode(output[0,batch['input_ids'].shape[1]:], skip_special_tokens=True)
            result['predictions'].append({'sample_index':index, **parse_answer(answer),
                'max_new_tokens':args.max_new_tokens,'retry_invalid':args.retry_invalid})
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            temporary.replace(args.output)
            del batch, output
            if len(result['predictions']) % 10 == 0 or len(result['predictions']) == len(dataset['samples']):
                print(json.dumps({'model':args.model,'completed':len(result['predictions']),
                                  'total':len(dataset['samples']),'elapsed_sec':time.monotonic()-started}),flush=True)


if __name__ == '__main__':
    main()
