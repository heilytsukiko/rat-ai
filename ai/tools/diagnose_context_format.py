"""Replay failed private answers; print fixed error codes or aggregate progress only."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from compare_video_context import MODELS, MODES, PROMPT, PARSER_POLICY, nominal_refs, read_frames, build_messages, tiles, parse_context_answer, summary_for


def main():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, set_seed
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repair-output',type=Path,help='Replay all failed responses to a fresh output directory')
    parser.add_argument('--strict-schema',action='store_true')
    args=parser.parse_args()
    os.umask(0o077)
    set_seed(42)
    torch.set_num_threads(4)
    source=(args.dataset/'dataset.json').read_bytes()
    dataset=json.loads(source)
    protocol=json.loads((args.run/'protocol-private.json').read_text())
    rows=json.loads((args.run/'predictions-private.json').read_text())['predictions']
    if protocol['dataset_sha256']!=hashlib.sha256(source).hexdigest():
        raise ValueError('Dataset differs from original run')
    if protocol['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError('Prompt differs from original run')
    if any(protocol['models'][key]['model']!=MODELS[key] for key in MODELS):
        raise ValueError('Model differs from original run')
    summary_for(rows,protocol['selected_windows'])
    repaired=list(rows)
    if args.repair_output:
        if args.repair_output.exists(): raise ValueError('Fresh repair directory required')
        args.repair_output.mkdir(parents=True,mode=0o700)
        protocol['parser_policy']=PARSER_POLICY
        (args.repair_output/'protocol-private.json').write_text(json.dumps(protocol,indent=2))
    output=[]
    replay_started=time.monotonic()
    for key in MODELS:
        chosen=([r for r in rows if r['model_key']==key and not r['parsed']] if args.repair_output else
                [next((r for r in rows if r['model_key']==key and r['mode']==mode and not r['parsed']),None)
                 for mode in MODES])
        chosen=[r for r in chosen if r is not None]
        if not chosen:
            continue
        config=protocol['models'][key]
        location=snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
        processor=AutoProcessor.from_pretrained(location,local_files_only=True,trust_remote_code=False,
                                                min_pixels=336*336,max_pixels=336*336)
        cls=Qwen2_5_VLForConditionalGeneration if key=='qwen25-7b' else Qwen3VLForConditionalGeneration
        model=cls.from_pretrained(location,local_files_only=True,trust_remote_code=False,use_safetensors=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)).eval()
        with torch.inference_mode():
            for row in chosen:
                tick=time.monotonic()
                target,mode=row['sample_index'],row['mode']
                center=dataset['samples'][target]['center_sec']
                refs=nominal_refs(dataset,target,4 if mode=='four_frames' else 16)
                images=read_frames(args.dataset,dataset,refs)
                chat,anchor=build_messages(refs,center,mode=='context16_tiles')
                if mode=='context16_tiles': images+=tiles(images[anchor])
                text=processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
                batch=processor(text=[text],images=images,return_tensors='pt').to('cuda')
                generated=model.generate(**batch,max_new_tokens=360,do_sample=False,use_cache=True)
                n=generated.shape[1]-batch['input_ids'].shape[1]
                answer=processor.tokenizer.decode(generated[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
                parsed=parse_context_answer(answer,len(refs),strict_schema=args.strict_schema and not args.repair_output)
                item={'model':key,'mode':mode,'parsed':parsed['parsed'],'reason':parsed.get('parse_error','valid'),
                      'generated_tokens':n,'hit_token_limit':n==360,'sample_index':target,'raw_answer':answer}
                output.append(item)
                safe={k:v for k,v in item.items() if k not in ('raw_answer','sample_index')}
                if args.repair_output:
                    revised={k:v for k,v in row.items() if k not in
                             ('action','visibility','review','observation','uncertainty','parsed','parse_error',
                              'objects','evidence_frames','evidence_nominal_sec')}
                    revised.update(parsed,evidence_nominal_sec=[refs[i-1]['nominal_sec'] for i in parsed['evidence_frames']],
                                   elapsed_sec=time.monotonic()-tick,replayed_after_format_fix=True)
                    position=next(i for i,r in enumerate(repaired) if r['model_key']==key and r['mode']==mode and r['sample_index']==target)
                    repaired[position]=revised
                    (args.repair_output/'predictions-private.json').write_text(json.dumps({'predictions':repaired},ensure_ascii=False,indent=2),encoding='utf-8')
                    state={'stage':'replaying_invalid','completed':len(output),'total':sum(not r['parsed'] for r in rows)}
                    (args.repair_output/'status.json').write_text(json.dumps(state))
                    print(json.dumps(state),flush=True)
                else:
                    print(json.dumps(safe),flush=True)
                    (args.run/'format-diagnostics-private.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
                del batch,generated
        del model,processor
        gc.collect()
        torch.cuda.empty_cache()
    if args.repair_output:
        summary=json.loads((args.run/'summary.json').read_text())
        summary.update(experiment='video-context-format-repair-v2',
                       comparisons=summary_for(repaired,protocol['selected_windows']),
                       replayed_responses=len(output),parser_policy=protocol['parser_policy'])
        summary['limitations'].append('Post-hoc parser repair and replay of invalid answers; not a new independent accuracy test.')
        summary['original_run_elapsed_sec']=summary.pop('elapsed_sec')
        summary['replay_elapsed_sec']=round(time.monotonic()-replay_started,1)
        summary['original_peak_vram_bytes']=summary['peak_vram_bytes']
        summary['peak_vram_bytes']=torch.cuda.max_memory_allocated()
        (args.repair_output/'summary.json').write_text(json.dumps(summary,indent=2))
        (args.repair_output/'status.json').write_text(json.dumps({'stage':'completed','replayed':len(output)}))
    print(json.dumps({'completed':True,'cases':len(output),'reason_counts':dict(Counter(r['reason'] for r in output))}),flush=True)


if __name__=='__main__':
    main()
