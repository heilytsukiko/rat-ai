"""Private offline audit: cached video captions -> base/adapter -> evidence limits.

No captions, per-video results, labels or paths are printed. Only aggregate counts
are public. With no action ground truth, this audit measures behavior, not accuracy.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key,value in {'USE_TF':'0','USE_FLAX':'0','HF_HUB_OFFLINE':'1',
                   'HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false'}.items():
    os.environ.setdefault(key,value)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rat_ai.evidence import CONDITIONS, assess, parse_facts
from rat_ai.sampled_evidence import PROTOCOL, assess_sampled
from train_rule_reasoning import encode
from auto_annotate_gpu import MODELS


def select_windows(dataset):
    """First/middle/last available windows of each validation video; never labels."""
    selected = []
    for group in dataset['groups']:
        if group['split'] != 'validation':
            continue
        indexes = sorted(group['samples'],key=lambda i:dataset['samples'][i]['center_sec'])
        if not indexes or any(dataset['samples'][i]['split']!='validation' or
                              dataset['samples'][i]['group']!=group['group'] for i in indexes):
            raise ValueError('Invalid validation grouping')
        selected.extend(indexes[i] for i in sorted({0,len(indexes)//2,len(indexes)-1}))
    if not selected or len(selected)!=len(set(selected)):
        raise ValueError('Empty or duplicate window selection')
    return selected


def build_tasks(dataset, passes, dataset_sha):
    if dataset.get('frames_per_clip')!=4 or dataset.get('clip_span_sec')!=3:
        raise ValueError('This audit supports only the existing four-frame protocol')
    if len(passes)!=2 or {p['model'] for p in passes}!=set(MODELS.values()):
        raise ValueError('Expected two independent configured caption models')
    if any(p.get('dataset_sha256')!=dataset_sha or p.get('data_kind')!='bank_review' for p in passes):
        raise ValueError('Caption dataset provenance mismatch')
    if passes[0].get('prompt_sha256')!=passes[1].get('prompt_sha256'):
        raise ValueError('Caption prompt mismatch')
    if any(p.get('sampling')!='existing four-frame three-second windows' for p in passes):
        raise ValueError('Caption sampling protocol mismatch')
    selected = select_windows(dataset)
    tasks, skipped = [], 0
    for source in sorted(passes,key=lambda p:p['model']):
        indexed = {p['sample_index']:p for p in source['predictions']}
        if len(indexed)!=len(source['predictions']) or set(indexed)!=set(range(len(dataset['samples']))):
            raise ValueError('Incomplete or duplicate caption coverage')
        for index in selected:
            caption = indexed[index]
            if caption.get('parsed') is not True:
                skipped += 1
                continue
            if any(not isinstance(caption.get(k),str) or len(caption[k])>500
                   for k in ('observation','uncertainty')):
                raise ValueError('Invalid caption text')
            observation = ('Непроверенное машинное описание четырёх отдельных кадров без звука. '
                'Полная последовательность операции, содержимое документов, принадлежность вещей '
                'и происходящее за кадром не подтверждены.\n'
                'Описание: '+caption['observation']+'\n'
                'Ограничения, указанные моделью: '+caption['uncertainty'])
            for rule in CONDITIONS:
                tasks.append({'id':f'{source["model"]}/{index}/{rule}',
                              'sample_index':index,'caption_model':source['model'],
                              'rule_id':rule,'observation':observation})
    return tasks, selected, skipped


def summarize(rows):
    valid = [r for r in rows if r['parsed']]
    overrides = [o for r in valid for o in r['bounded']['overrides']]
    return {'tasks':len(rows),'valid_outputs':len(valid),'invalid_outputs':len(rows)-len(valid),
            'raw_status_counts':dict(Counter(r['raw']['status'] for r in valid)),
            'bounded_status_counts':dict(Counter(r['bounded']['status'] for r in valid)),
            'claims_changed_to_unknown':len(overrides),
            'positive_claims_changed_to_unknown':sum(o['from']=='yes' for o in overrides),
            'negative_claims_changed_to_unknown':sum(o['from']=='no' for o in overrides),
            'override_reason_counts':dict(Counter(o['reason'] for o in overrides)),
            'per_rule':{rule:{'raw':dict(Counter(r['raw']['status'] for r in valid if r['rule_id']==rule)),
                              'bounded':dict(Counter(r['bounded']['status'] for r in valid if r['rule_id']==rule))}
                        for rule in CONDITIONS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise ValueError('Fresh output directory required')
    dataset_bytes=(args.dataset/'dataset.json').read_bytes()
    dataset=json.loads(dataset_bytes)
    digest=hashlib.sha256(dataset_bytes).hexdigest()
    passes=[json.loads(p.read_text(encoding='utf-8')) for p in (args.first,args.second)]
    tasks,selected,skipped=build_tasks(dataset,passes,digest)
    args.output.mkdir(parents=True,mode=0o700)
    started=time.monotonic()
    def save(name,data):
        path=args.output/name
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(path)
    def status(stage,**kwargs):
        state={'stage':stage,'elapsed_sec':round(time.monotonic()-started,1),**kwargs}
        save('status.json',state)
        print(json.dumps(state),flush=True)
    try:
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, set_seed
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        set_seed(42)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        config=json.loads((args.run/'config.json').read_text())
        if config['model']!='Qwen/Qwen3-VL-4B-Instruct' or config['dataset']['data_kind']!='fictional_public_rule_reasoning':
            raise ValueError('Expected separate text-reasoning adapter')
        save('protocol.json',{'dataset_sha256':digest,'selected_windows':selected,
            'caption_inputs':[{'model':p['model'],'revision':p['revision'],
                               'prompt_sha256':p['prompt_sha256']} for p in passes],
            'reasoning_config':config,'evidence_protocol':PROTOCOL,
            'training':False,'selection':'first/middle/last available validation windows; no event labels'})
        status('loading',tasks=len(tasks))
        location=snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
        tokenizer=AutoTokenizer.from_pretrained(location,local_files_only=True,trust_remote_code=False)
        encoded=[encode(tokenizer,r) for r in tasks]
        if any(batch['input_ids'].shape[1]>2048 for batch in encoded):
            raise ValueError('Caption context exceeds tested limit')
        def load_model():
            return Qwen3VLForConditionalGeneration.from_pretrained(location,local_files_only=True,
                trust_remote_code=False,use_safetensors=True,device_map={'':0},dtype=torch.bfloat16,
                attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True))
        reports={}
        for mode in ('base','adapter'):
            model=load_model()
            if mode=='adapter':
                model=PeftModel.from_pretrained(model,args.run/'adapter',local_files_only=True,is_trainable=False)
            model.eval()
            rows=[]
            with torch.inference_mode():
                for i,(task,batch_cpu) in enumerate(zip(tasks,encoded)):
                    batch={k:v.to('cuda') for k,v in batch_cpu.items()}
                    output=model.generate(**batch,max_new_tokens=64,do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id,use_cache=True)
                    text=tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
                    row={k:task[k] for k in ('id','sample_index','caption_model','rule_id')}
                    try:
                        facts=parse_facts(text)
                        row.update(parsed=True,raw=assess(task['rule_id'],facts),
                                   bounded=assess_sampled(task['rule_id'],facts,protocol=PROTOCOL))
                    except (ValueError,TypeError):
                        row.update(parsed=False,error='invalid_model_output')
                    rows.append(row)
                    del output,batch
                    if (i+1)%24==0 or i+1==len(tasks):
                        save(mode+'-private.json',{'rows':rows})
                        status(mode,completed=i+1,total=len(tasks))
            reports[mode]=summarize(rows)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        summary={'schema_version':1,'experiment':'video-caption-rule-chain-v1',
            'selected_validation_videos':len({dataset['samples'][i]['group'] for i in selected}),
            'selected_windows':len(selected),'caption_models':sorted(p['model'] for p in passes),
            'rules':len(CONDITIONS),'skipped_invalid_captions':skipped,
            'selection':'first/middle/last available validation windows; no labels or event times',
            'reports':reports,'accuracy_measured':False,'promoted_to_site':False,
            'elapsed_sec':round(time.monotonic()-started,1),
            'limitations':['Existing unverified captions reused; no fresh video inference in this audit.',
                'No human-reviewed action/condition ground truth; counts are not accuracy or false-positive rates.',
                'Zero bounded possible-issue results is guaranteed by the conservative protocol, not model quality.',
                'Three sampled windows per validation video do not cover the full video.',
                'Text adapter was trained on fictional scenarios; observations may contain shared VLM errors.']}
        save('summary.json',summary)
        status('completed',tasks_per_model=len(tasks))
    except Exception as exc:
        # Exception messages could contain private text. Keep status and stdout generic.
        status('failed',error_type=type(exc).__name__)
        raise


if __name__=='__main__':
    main()
