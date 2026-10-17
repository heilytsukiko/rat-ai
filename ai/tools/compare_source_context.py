"""Additional matched visual condition: original-source 16 frames + four crops."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from compare_video_context import MODELS, PROMPT, PARSER_POLICY, build_messages, parse_context_answer, summary_for

MODE='source16_tiles'


def read_source_window(root,record):
    import numpy as np
    from PIL import Image
    root=Path(root).resolve()
    path=(root/record['file']).resolve()
    if path.parent!=root or path.suffix!='.npz':
        raise ValueError('Invalid source-window path')
    with np.load(path,allow_pickle=False) as data:
        frames,crops,times=data['frames'],data['tiles'],data['decoded_time_sec']
    if frames.shape!=(16,448,448,3) or crops.shape!=(4,448,448,3) or frames.dtype!=np.uint8 or crops.dtype!=np.uint8:
        raise ValueError('Unexpected original-source image tensors')
    if times.shape!=(16,) or not np.isfinite(times).all() or (np.diff(times)<0).any():
        raise ValueError('Invalid recorded timestamps')
    if type(record['anchor_index']) is not int or not 0<=record['anchor_index']<16:
        raise ValueError('Invalid crop anchor')
    actual_anchor=min(range(len(times)),key=lambda i:abs(float(times[i])-record['center_sec']))
    if actual_anchor!=record['anchor_index']:
        raise ValueError('Crop anchor does not match the target time')
    return [Image.fromarray(f) for f in frames], [Image.fromarray(f) for f in crops], times.tolist()


def main():
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, set_seed
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=Path,required=True)
    parser.add_argument('--qwen25-config',type=Path,required=True)
    parser.add_argument('--qwen3-config',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise ValueError('Fresh output required')
    context=json.loads((args.context/'context.json').read_text())
    protocol=json.loads((args.baseline/'protocol-private.json').read_text())
    if context.get('source_dataset_sha256')!=protocol['dataset_sha256'] or context.get('source_mapping_verified') is not True:
        raise ValueError('Different source data or unverified mapping')
    selected=protocol['selected_windows']
    by_index={r['sample_index']:r for r in context['records']}
    if set(by_index)!=set(selected) or len(by_index)!=len(context['records']):
        raise ValueError('Window coverage mismatch')
    configs={key:json.loads(path.read_text()) for key,path in
             (('qwen25-7b',args.qwen25_config),('qwen3-4b',args.qwen3_config))}
    if any(configs[k]['model']!=MODELS[k] or configs[k]['revision']!=protocol['models'][k]['revision'] for k in MODELS):
        raise ValueError('Model or revision differs from baseline')
    if protocol['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError('Prompt differs from baseline')
    if protocol.get('parser_policy')!=PARSER_POLICY:
        raise ValueError('Baseline must use the same parser policy')
    args.output.mkdir(parents=True,mode=0o700)
    started=time.monotonic()
    def save(name,value):
        path=args.output/name
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(path)
    def status(stage,**fields):
        value={'stage':stage,'elapsed_sec':round(time.monotonic()-started,1),**fields}
        save('status.json',value)
        print(json.dumps(value),flush=True)
    try:
        set_seed(42)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        predictions=[]
        save('protocol-private.json',dict(protocol,condition=MODE,source_context_manifest_sha256=
             hashlib.sha256((args.context/'context.json').read_bytes()).hexdigest()))
        for key in MODELS:
            status('loading',model=key)
            config=configs[key]
            location=snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
            processor=AutoProcessor.from_pretrained(location,local_files_only=True,trust_remote_code=False,
                min_pixels=448*448,max_pixels=448*448)
            cls=Qwen2_5_VLForConditionalGeneration if key=='qwen25-7b' else Qwen3VLForConditionalGeneration
            model=cls.from_pretrained(location,local_files_only=True,trust_remote_code=False,use_safetensors=True,
                dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)).eval()
            with torch.inference_mode():
                for position,target in enumerate([None]+selected):
                    tick=time.monotonic()
                    if target is None:
                        frames=[Image.new('RGB',(448,448),(100,30,20)) for _ in range(16)]
                        crops=[Image.new('RGB',(448,448),(100,30,20)) for _ in range(4)]
                        times=[i/2 for i in range(16)]
                        center=3.5
                    else:
                        record=by_index[target]
                        frames,crops,times=read_source_window(args.context,record)
                        center=record['center_sec']
                    refs=[{'nominal_sec':t} for t in times]
                    chat,_=build_messages(refs,center,True)
                    text=processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
                    batch=processor(text=[text],images=frames+crops,return_tensors='pt').to('cuda')
                    output=model.generate(**batch,max_new_tokens=1 if target is None else 360,
                                           do_sample=False,use_cache=True)
                    if target is None:
                        status('synthetic_shape_probe_passed',model=key,images=20,image_size=448)
                    else:
                        answer=processor.tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
                        row=parse_context_answer(answer,len(times))
                        row.update(model_key=key,mode=MODE,sample_index=target,frame_count=len(times),extra_tiles=4,
                            evidence_time_sec=[times[i-1] for i in row['evidence_frames']],timestamps_kind='decoded',
                            context_start_sec=times[0],context_end_sec=times[-1],elapsed_sec=time.monotonic()-tick,
                            human_verified=False,legal_verdict=None)
                        predictions.append(row)
                        save('predictions-private.json',{'predictions':predictions})
                        status('inference',model=key,completed=position,total=len(selected),
                               total_completed=len(predictions),total_tasks=2*len(selected))
                    del batch,output,frames,crops
            del model,processor
            gc.collect()
            torch.cuda.empty_cache()
        baseline=json.loads((args.baseline/'summary.json').read_text())
        summary={'schema_version':1,'experiment':'original-source-context-v1','selected_windows':len(selected),
            'selected_validation_videos':len({r['group'] for r in context['records']}),'models':baseline['models'],
            'comparisons':{**baseline['comparisons'],**summary_for(predictions,selected,modes=(MODE,))},
            'source_condition_elapsed_sec':round(time.monotonic()-started,1),
            'peak_vram_bytes':torch.cuda.max_memory_allocated(),'accuracy_measured':False,'promoted_to_site':False,
            'min_distinct_source_frame_times':min(r['distinct_times'] for r in context['records']),
            'limitations':['Earlier 336px conditions: '+item for item in baseline['limitations']]+[
                'Original-source condition changes sampling and detail together; their individual effects are not isolated.',
                'Original-source frames are 448px with crops made before resizing and recorded decoder timestamps.',
                'No action/object ground truth is available; agreement and visibility counts are not accuracy.']}
        save('summary.json',summary)
        status('completed',total_tasks=len(predictions))
    except Exception as exc:
        status('failed',error_type=type(exc).__name__)
        raise


if __name__=='__main__':
    main()
