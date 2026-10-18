"""Draft observable actions with a pretrained VLM; never mark them reviewed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def main():
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    source=(args.dataset/'dataset.json').read_bytes()
    dataset=json.loads(source)
    config=json.loads(args.model_config.read_text(encoding='utf-8'))
    if config['model']!='Qwen/Qwen3-VL-2B-Instruct':
        raise ValueError('Expected the pretrained Qwen3 probe configuration')
    location=snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
    processor=AutoProcessor.from_pretrained(location,local_files_only=True,
                                            min_pixels=144*28*28,max_pixels=144*28*28)
    model=Qwen3VLForConditionalGeneration.from_pretrained(location,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,device_map={'':0},
        attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)).eval()
    prompt=('Describe only actions directly visible in these four chronological cash-desk frames. '
            'Do not identify people, read personal data, infer intentions, or declare a violation. '
            'If an action or object is unclear, say so. Do not invent missing actions. '
            'Reply in Russian as JSON with three short string fields: observed_action, event_type, uncertainty. '
            'event_type is a neutral action name, not an accusation. No legal references.')
    output={'schema_version':1,'dataset_sha256':hashlib.sha256(source).hexdigest(),
            'model':config['model'],'revision':config['revision'],
            'human_review_required':True,'drafts':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    for group in dataset['groups']:
        targets=group['annotated_event_times_sec'] or [group['duration_sec']/2]
        for target in targets:
            index=min(group['samples'],key=lambda i:abs(dataset['samples'][i]['center_sec']-target))
            row=dataset['samples'][index]
            frames=np.load(args.dataset/row['file'],allow_pickle=False)['frames']
            messages=[{'role':'user','content':[{'type':'image'} for _ in frames]+[{'type':'text','text':prompt}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=processor(text=[text],images=[Image.fromarray(f) for f in frames],return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated=model.generate(**batch,max_new_tokens=180,do_sample=False,use_cache=True)
            answer=processor.tokenizer.decode(generated[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
            try:
                begin,end=answer.index('{'),answer.rindex('}')+1
                parsed=json.loads(answer[begin:end])
                if not isinstance(parsed,dict):
                    raise ValueError('Expected an object')
                draft={key:str(parsed.get(key,''))[:1500] for key in ('observed_action','event_type','uncertainty')}
            except (ValueError,TypeError):
                draft={'observed_action':answer[:1500],'event_type':'','uncertainty':'Ответ требует разбора человеком'}
            output['drafts'].append({'group':group['group'],'target_sec':target,
                                    'center_sec':row['center_sec'],'review_status':'pending',**draft})
            args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'drafts_completed':len(output['drafts']),
                              'elapsed_sec':time.monotonic()-start}),flush=True)
            del batch,generated
    print(json.dumps({'status':'complete','drafts':len(output['drafts']),
                      'human_review_required':True}),flush=True)


if __name__=='__main__':
    main()
