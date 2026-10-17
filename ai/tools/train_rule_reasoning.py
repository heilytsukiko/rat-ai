"""Separate text-only QLoRA experiment using cached Qwen3-VL-4B and fictional data."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for key, value in {'USE_TF':'0', 'USE_FLAX':'0', 'HF_HUB_DISABLE_TELEMETRY':'1',
                   'TOKENIZERS_PARALLELISM':'false', 'WANDB_DISABLED':'true',
                   'HF_HUB_OFFLINE':'1'}.items():
    os.environ.setdefault(key, value)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rat_ai.evidence import assess, messages, parse_facts


def load_dataset(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    raw = (root/'examples.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError('Dataset hash mismatch')
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if any(not r.get('synthetic') or r['split'] not in ('train','challenge') for r in rows):
        raise ValueError('Only the fictional training/challenge corpus is supported')
    train = [r for r in rows if r['split'] == 'train']
    challenge = [r for r in rows if r['split'] == 'challenge']
    if not train or not challenge:
        raise ValueError('Both splits required')
    for key in ('id', 'family', 'observation'):
        if {r[key] for r in train} & {r[key] for r in challenge}:
            raise ValueError('Cross-split duplicate')
    for row in rows:
        if assess(row['rule_id'], row['facts'])['status'] != row['expected_status']:
            raise ValueError('Inconsistent target')
    return manifest, train, challenge


def encode(tokenizer, row, training=False):
    import torch
    chat = messages(row['rule_id'], row['observation'])
    prefix = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                          return_tensors='pt')
    if not training:
        return {'input_ids':prefix, 'attention_mask':torch.ones_like(prefix)}
    chat.append({'role':'assistant', 'content':json.dumps({'facts':row['facts']}, separators=(',',':'))})
    ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False,
                                       return_tensors='pt')
    n = prefix.shape[1]
    if not torch.equal(ids[:, :n], prefix) or ids.shape[1] > 2048:
        raise ValueError('Assistant mask mismatch or overlength example')
    labels = ids.clone()
    labels[:, :n] = -100
    return {'input_ids':ids, 'attention_mask':torch.ones_like(ids), 'labels':labels}


def score_predictions(rows, predictions):
    if len(rows) != len(predictions) or not rows:
        raise ValueError('Complete evaluation required')
    exact = fields = statuses = invalid = unsupported = unknown_correct = unknown_total = 0
    results = []
    by_status = {}
    for row, text in zip(rows, predictions):
        expected = row['expected_status']
        category = by_status.setdefault(expected, {'total':0, 'correct':0})
        category['total'] += 1
        try:
            facts = parse_facts(text)
            decision = assess(row['rule_id'], facts)['status']
        except (ValueError, TypeError):
            facts, decision = None, 'invalid_output'
            invalid += 1
        exact += facts == row['facts']
        statuses += decision == expected
        category['correct'] += decision == expected
        fields += sum(a == b for a,b in zip(facts or [None]*3, row['facts']))
        unsupported += decision == 'possible_issue' and expected != 'possible_issue'
        for actual, target in zip(facts or [None]*3, row['facts']):
            if target == 'unknown':
                unknown_total += 1
                unknown_correct += actual == 'unknown'
        results.append({'id':row['id'], 'expected':row['facts'], 'predicted':facts,
                        'expected_status':expected, 'predicted_status':decision})
    n = len(rows)
    return {'examples':n, 'exact_facts_correct':exact, 'exact_facts_accuracy':exact/n,
            'field_accuracy':fields/(n*3), 'status_correct':statuses, 'status_accuracy':statuses/n,
            'invalid_outputs':invalid, 'unsupported_possible_issues':unsupported,
            'unknown_fields_correct':unknown_correct, 'unknown_fields_total':unknown_total,
            'by_status':by_status, 'results':results}


def main():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cached-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh experiment directory; never overwrite a run')
    manifest, train, challenge = load_dataset(args.dataset)
    config = json.loads(args.cached_config.read_text())
    if config['model'] != 'Qwen/Qwen3-VL-4B-Instruct':
        raise ValueError('Expected cached Qwen3-VL-4B')
    args.output.mkdir(parents=True, mode=0o700)
    start = time.monotonic()
    def save(name, value):
        temp = args.output/(name+'.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(args.output/name)
    def status(stage, **kwargs):
        value = {'stage':stage, 'elapsed_sec':round(time.monotonic()-start,1), **kwargs}
        save('status.json', value)
        print(json.dumps(value), flush=True)
    try:
        set_seed(42)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        run_config = {'model':config['model'], 'revision':config['revision'],
                      'method':'text-only QLoRA NF4', 'epochs':2, 'learning_rate':0.0001,
                      'rank':8, 'alpha':16, 'seed':42, 'gradient_accumulation':8,
                      'loss_weighting':'equal total weight per rule and derived status',
                      'dataset':manifest, 'deployment':'offline experiment; not connected to website'}
        save('config.json', run_config)
        status('loading_cached_model')
        location = snapshot_download(config['model'], revision=config['revision'], local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            location, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            device_map={'':0}, dtype=torch.bfloat16, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        encoded_train = [encode(tokenizer, row, True) for row in train]
        encoded_challenge = [encode(tokenizer, row) for row in challenge]
        def evaluate(stage):
            model.eval()
            predictions = []
            with torch.inference_mode():
                for i, batch in enumerate(encoded_challenge):
                    batch = {k:v.to('cuda') for k,v in batch.items()}
                    generated = model.generate(**batch, max_new_tokens=64, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id, use_cache=True)
                    predictions.append(tokenizer.decode(generated[0,batch['input_ids'].shape[1]:],
                                                         skip_special_tokens=True))
                    if (i+1)%6 == 0:
                        status(stage, completed=i+1, total=len(challenge))
            result = score_predictions(challenge, predictions)
            save(stage+'.json', result)
            return result
        baseline = evaluate('baseline')
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant':False})
        # Explicitly exclude vision attention, even though this run supplies no images.
        targets = [name for name, _ in model.named_modules()
                   if 'language_model' in name and name.rsplit('.',1)[-1] in
                   ('q_proj','k_proj','v_proj','o_proj')]
        if not targets:
            raise RuntimeError('No language attention targets found')
        model = get_peft_model(model, LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                                               target_modules=targets,task_type='CAUSAL_LM'))
        initial = {n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001)
        counts = Counter((r['rule_id'],r['expected_status']) for r in train)
        weights = [len(train)/(len(counts)*counts[(r['rule_id'],r['expected_status'])]) for r in train]
        rng = random.Random(42)
        history = []
        for epoch in range(2):
            model.train()
            order = list(range(len(train)))
            rng.shuffle(order)
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, index in enumerate(order):
                batch = {k:v.to('cuda') for k,v in encoded_train[index].items()}
                loss = model(**batch,use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                total_loss += loss.item()
                group_size = min(8,len(order)-(step//8)*8)
                (loss*weights[index]/group_size).backward()
                if (step+1)%8 == 0 or step+1 == len(order):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if (step+1)%54 == 0:
                    status('training', epoch=epoch+1, completed=step+1, total=len(order))
            history.append({'epoch':epoch+1,'mean_unweighted_loss':total_loss/len(order)})
        delta = sum(((p.detach().float().cpu()-initial[n])**2).sum().item()
                    for n,p in model.named_parameters() if n in initial)**.5
        model.save_pretrained(args.output/'adapter',safe_serialization=True)
        tokenizer.save_pretrained(args.output/'adapter')
        trained = evaluate('trained')
        summary = {'schema_version':1,'experiment':'public-rule-text-reasoning-v1',
                   'data_kind':manifest['data_kind'],'train_examples':len(train),'challenge_examples':len(challenge),
                   'model':config['model'],'revision':config['revision'],'history':history,
                   'adapter_change_l2':delta,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'baseline':{k:v for k,v in baseline.items() if k!='results'},
                   'trained':{k:v for k,v in trained.items() if k!='results'},
                   'promoted_to_site':False,'limitations':manifest['limitations'],
                   'elapsed_sec':time.monotonic()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated()}
        save('summary.json',summary)
        status('completed', adapter_change_l2=delta)
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc)[:600])
        raise


if __name__ == '__main__':
    main()
