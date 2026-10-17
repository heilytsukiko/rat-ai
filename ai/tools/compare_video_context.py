"""Private matched ablation: four frames vs longer context vs context plus tiles.

Reuses the authorized server bundle. Nominal times are reconstructed, not decoded
presentation timestamps. More context and model agreement are not ground truth.
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
from auto_annotate_gpu import ACTIONS, MODELS, consensus, parse_answer
from audit_video_rule_chain import select_windows

MODES=('four_frames','context16','context16_tiles')
PARSER_POLICY='retain_valid_extra_references_unknown_objects_abstain_v2'
OBJECTS=('banknotes','papers','phone','bag','counter','computer','person','unclear')
PROMPT='''Analyze the chronological frames independently. Focus on the target interval,
using earlier/later context to understand the sequence. The images are sparse samples,
not continuous video. Image times are approximate. Extra tiles, if present, are crops
of one already-shown frame, NOT new time steps. Describe only directly visible actions
and objects. Do not identify people, read personal details, infer ownership, intentions,
legal guilt, document contents or events outside the images. A hand near a bag is not
proof of cash being taken. Papers are not necessarily banknotes. Phone handling and
cash counting are not automatically violations. Ordinary_visible_action does not mean
the whole video is clean. Use possible_issue only for a concrete visible operational
concern, otherwise ordinary_visible_action or unclear. Choose unclear when uncertain.
Return ONLY JSON with exactly these keys:
{"action":"CODE","visibility":"clear|partial|unclear",
"review":"possible_issue|ordinary_visible_action|unclear",
"observation":"one short Russian sentence","uncertainty":"one short Russian sentence or empty",
"objects":["OBJECT"],"evidence_frames":[1,2]}.
Use at most 3 evidence frame numbers from the chronological sequence, never tile numbers.
Clear actions must cite a frame. These references locate observations, not exact event boundaries.
Limit each Russian sentence to 140 characters.
Allowed action CODE: '''+', '.join(ACTIONS)+'. Allowed OBJECT: '+', '.join(OBJECTS)+'.'


def nominal_refs(dataset, target, count):
    """No image reads, class labels or event annotations influence frame selection."""
    if dataset.get('frames_per_clip')!=4 or dataset.get('clip_span_sec')!=3 or count not in (4,16):
        raise ValueError('Unsupported source or frame budget')
    sample=dataset['samples'][target]
    groups=[g for g in dataset['groups'] if g['group']==sample['group']]
    if len(groups)!=1 or sample['split']!='validation' or groups[0]['split']!='validation':
        raise ValueError('Only validation windows are used')
    group=groups[0]
    duration=group['duration_sec']
    center=sample['center_sec']
    if not duration>0 or not 0<=center<duration:
        raise ValueError('Invalid video timing')
    def refs(index):
        row=dataset['samples'][index]
        if row['group']!=group['group'] or row['split']!='validation':
            raise ValueError('Cross-group/split frame reference')
        start=max(0,row['center_sec']-1.5)
        end=min(duration-.001,row['center_sec']+1.5)
        return [{'sample_index':index,'frame_index':j,'nominal_sec':round(start+(end-start)*j/3,4)}
                for j in range(4)]
    focus=refs(target)
    if count==4:
        return focus
    # Keep all original target frames. Fill the wider 12-second context using
    # farthest-time sampling; do not manufacture/repeat frames to reach 16.
    selected={r['nominal_sec']:r for r in focus}
    candidates={}
    for index in sorted(group['samples']):
        for ref in refs(index):
            timestamp=ref['nominal_sec']
            if max(0,center-6)<=timestamp<=min(duration,center+6) and timestamp not in selected:
                candidates.setdefault(timestamp,ref)
    while candidates and len(selected)<count:
        chosen=max(candidates,key=lambda t:(min(abs(t-s) for s in selected),-t))
        selected[chosen]=candidates.pop(chosen)
    return [selected[t] for t in sorted(selected)]


def read_frames(root,dataset,refs):
    import numpy as np
    from PIL import Image
    cached={}
    result=[]
    root=root.resolve()
    for ref in refs:
        index=ref['sample_index']
        if index not in cached:
            path=(root/dataset['samples'][index]['file']).resolve()
            if path.parent!=root or path.suffix!='.npz':
                raise ValueError('Invalid bundle frame path')
            with np.load(path,allow_pickle=False) as archive:
                frames=archive['frames']
            if frames.shape!=(4,336,336,3) or frames.dtype!=np.uint8:
                raise ValueError('Unexpected frame tensor')
            cached[index]=frames
        result.append(Image.fromarray(cached[index][ref['frame_index']]))
    return result


def tiles(image):
    from PIL import Image
    if image.size!=(336,336):
        raise ValueError('Expected prepared square frame')
    # Fixed overlapping regions; NOT detected hands or objects. Upscaling adds
    # attention to existing pixels but cannot recover original-resolution detail.
    boxes=[(x,y,x+224,y+224) for y in (0,112) for x in (0,112)]
    return [image.crop(box).resize((336,336),Image.Resampling.BICUBIC) for box in boxes]


def build_messages(refs,center,with_tiles):
    content=[{'type':'text','text':PROMPT+f'\nTarget interval: approximately {max(0,center-1.5):.2f}'
             f' to {center+1.5:.2f} seconds. The same target is used for every input variant.'}]
    for index,ref in enumerate(refs):
        content.extend([{'type':'text','text':f'Chronological frame {index+1}; approximate time {ref["nominal_sec"]:.2f}s.'},
                        {'type':'image'}])
    anchor=min(range(len(refs)),key=lambda i:abs(refs[i]['nominal_sec']-center))
    if with_tiles:
        for name in ('top-left','top-right','bottom-left','bottom-right'):
            content.extend([{'type':'text','text':f'{name} enlarged tile of frame {anchor+1}; same instant, not a new frame.'},
                            {'type':'image'}])
    return [{'role':'user','content':content}],anchor


def parse_context_answer(answer,frame_count,strict_schema=False):
    reason='invalid_json'
    try:
        value=json.loads(answer[answer.index('{'):answer.rindex('}')+1])
        reason='unexpected_schema'
        if not isinstance(value,dict) or set(value)!={'action','visibility','review','observation','uncertainty','objects','evidence_frames'}:
            raise ValueError('Schema')
        objects=value.pop('objects')
        indices=value.pop('evidence_frames')
        reason='invalid_objects'
        if not isinstance(objects,list) or len(objects)>16 or any(not isinstance(x,str) or len(x)>80 for x in objects):
            raise ValueError('Objects')
        if len(objects)!=len(set(objects)):
            raise ValueError('Duplicate objects')
        if strict_schema and (len(objects)>8 or any(x not in OBJECTS for x in objects)):
            raise ValueError('Strict object vocabulary')
        reason='invalid_frame_references'
        if not isinstance(indices,list) or len(indices)>frame_count or any(type(i) is not int or not 1<=i<=frame_count for i in indices):
            raise ValueError('Frame references')
        if strict_schema and len(indices)>3:
            raise ValueError('Strict reference limit')
        reason='missing_or_duplicate_frame_references'
        if len(indices)!=len(set(indices)) or (value['visibility']=='clear' and not indices):
            raise ValueError('Missing or duplicate frame references')
        parsed=parse_answer(json.dumps(value,ensure_ascii=False))
        if not parsed['parsed']:
            reason=parsed.get('parse_error','annotation_fields')
            raise ValueError('Annotation fields')
        unknown=sum(x not in OBJECTS for x in objects)
        normalized=[x for x in objects if x in OBJECTS]
        if unknown:
            if 'unclear' not in normalized: normalized.append('unclear')
            if parsed['visibility']=='clear': parsed['visibility']='partial'
            parsed['review']='unclear'
        return dict(parsed,objects=normalized,evidence_frames=indices,
                    unknown_object_codes=unknown,extra_evidence_references=len(indices)>3)
    except (ValueError,TypeError,KeyError,AttributeError):
        return dict(parse_answer('invalid'),objects=[],evidence_frames=[],parse_error=reason)


def summary_for(predictions,selected,modes=MODES):
    expected={(m,mode,i) for m in MODELS for mode in modes for i in selected}
    indexed={(r['model_key'],r['mode'],r['sample_index']):r for r in predictions}
    if set(indexed)!=expected or len(indexed)!=len(predictions):
        raise ValueError('Complete matched comparison required')
    summary={}
    for mode in modes:
        pairs=[consensus(indexed[('qwen25-7b',mode,i)],indexed[('qwen3-4b',mode,i)]) for i in selected]
        by_model={}
        for model in MODELS:
            rows=[indexed[(model,mode,i)] for i in selected]
            by_model[model]={'parsed':sum(r['parsed'] for r in rows),
                'clear':sum(r['parsed'] and r['visibility']=='clear' for r in rows),
                'review_counts':dict(Counter(r['review'] for r in rows)),
                'action_counts':dict(Counter(r['action'] for r in rows)),
                'object_counts':dict(Counter(o for r in rows for o in r['objects'])),
                'review_normalized_to_unclear':sum(r.get('review_normalized_to_unclear',False) for r in rows),
                'elapsed_sec':round(sum(r['elapsed_sec'] for r in rows),2)}
        summary[mode]={'windows':len(selected),'models':by_model,
            'two_model_agreement':dict(Counter(r['status'] for r in pairs)),
            'joint_review_counts':dict(Counter(r['review'] for r in pairs)),
            'agreed_action_counts':dict(Counter(r['action'] for r in pairs if r['status']=='auto_agreement')),
            'min_chronological_frames':min(indexed[('qwen25-7b',mode,i)]['frame_count'] for i in selected),
            'max_chronological_frames':max(indexed[('qwen25-7b',mode,i)]['frame_count'] for i in selected)}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--qwen25-config',type=Path,required=True)
    parser.add_argument('--qwen3-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--strict-schema',action='store_true',help='Reproduce the original strict-format comparison')
    args=parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise ValueError('Fresh experiment directory required')
    source=(args.dataset/'dataset.json').read_bytes()
    dataset=json.loads(source)
    selected=select_windows(dataset)
    args.output.mkdir(parents=True,mode=0o700)
    started=time.monotonic()
    def save(name,value):
        path=args.output/name
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)
    def status(stage,**fields):
        value={'stage':stage,'elapsed_sec':round(time.monotonic()-started,1),**fields}
        save('status.json',value)
        print(json.dumps(value),flush=True)
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, set_seed
        set_seed(42)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        configs={key:json.loads(path.read_text()) for key,path in
                 (('qwen25-7b',args.qwen25_config),('qwen3-4b',args.qwen3_config))}
        if any(configs[key]['model']!=MODELS[key] for key in MODELS):
            raise ValueError('Unexpected cached model')
        save('protocol-private.json',{'dataset_sha256':hashlib.sha256(source).hexdigest(),
            'selected_windows':selected,'models':configs,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
            'selection':'first/middle/last validation windows, no labels or event times',
            'nominal_times_only':True,'modes':MODES,'max_new_tokens':360,
            'parser_policy':'strict_v1' if args.strict_schema else PARSER_POLICY})
        predictions=[]
        for key in MODELS:
            status('loading',model=key)
            config=configs[key]
            location=snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
            processor=AutoProcessor.from_pretrained(location,local_files_only=True,trust_remote_code=False,
                min_pixels=336*336,max_pixels=336*336)
            cls=Qwen2_5_VLForConditionalGeneration if key=='qwen25-7b' else Qwen3VLForConditionalGeneration
            model=cls.from_pretrained(location,local_files_only=True,trust_remote_code=False,use_safetensors=True,
                dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)).eval()
            # Check the largest input shape using artificial images before reading
            # any bank frame for this model. No synthetic output becomes a label.
            from PIL import Image
            probe_refs=[{'nominal_sec':i/2} for i in range(16)]
            probe_chat,_=build_messages(probe_refs,3.5,True)
            probe_text=processor.apply_chat_template(probe_chat,tokenize=False,add_generation_prompt=True)
            probe=processor(text=[probe_text],images=[Image.new('RGB',(336,336),(100,30,20))
                                                      for _ in range(20)],return_tensors='pt').to('cuda')
            with torch.inference_mode():
                probe_output=model.generate(**probe,max_new_tokens=1,do_sample=False,use_cache=True)
            del probe,probe_output
            status('synthetic_shape_probe_passed',model=key,images=20)
            with torch.inference_mode():
                for mode in MODES:
                    for index,target in enumerate(selected):
                        tick=time.monotonic()
                        center=dataset['samples'][target]['center_sec']
                        refs=nominal_refs(dataset,target,4 if mode=='four_frames' else 16)
                        images=read_frames(args.dataset,dataset,refs)
                        chat,anchor=build_messages(refs,center,mode=='context16_tiles')
                        if mode=='context16_tiles':
                            images+=tiles(images[anchor])
                        text=processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
                        batch=processor(text=[text],images=images,return_tensors='pt').to('cuda')
                        output=model.generate(**batch,max_new_tokens=360,do_sample=False,use_cache=True)
                        answer=processor.tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
                        row=parse_context_answer(answer,len(refs),strict_schema=args.strict_schema)
                        row.update(model_key=key,mode=mode,sample_index=target,frame_count=len(refs),
                            extra_tiles=4 if mode=='context16_tiles' else 0,
                            evidence_nominal_sec=[refs[i-1]['nominal_sec'] for i in row['evidence_frames']],
                            context_nominal_start_sec=refs[0]['nominal_sec'],context_nominal_end_sec=refs[-1]['nominal_sec'],
                            elapsed_sec=time.monotonic()-tick,human_verified=False,legal_verdict=None)
                        predictions.append(row)
                        save('predictions-private.json',{'predictions':predictions})
                        del batch,output,images
                        status('inference',model=key,mode=mode,completed=index+1,total=len(selected),
                               total_completed=len(predictions),total_tasks=len(selected)*len(MODES)*len(MODELS))
            del model,processor
            gc.collect()
            torch.cuda.empty_cache()
        summary={'schema_version':1,'experiment':'video-context-ablation-v1',
            'selected_validation_videos':len({dataset['samples'][i]['group'] for i in selected}),
            'selected_windows':len(selected),'models':{k:{'model':c['model'],'revision':c['revision']} for k,c in configs.items()},
            'comparisons':summary_for(predictions,selected),'elapsed_sec':round(time.monotonic()-started,1),
            'peak_vram_bytes':torch.cuda.max_memory_allocated(),'accuracy_measured':False,'promoted_to_site':False,
            'limitations':['Fresh visual inference, but no human-reviewed action/object labels: agreement is not accuracy.',
                'Validation windows were selected without event labels; no model training or promotion in this run.',
                'Context reconstructed from existing 336px samples; actual decoded timestamps were not retained.',
                '16 is a maximum, not fabricated padding; end clips may contain fewer unique nominal times.',
                'Fixed overlapping tiles do not detect hands or recover detail lost during original resizing.',
                'Sparse images and generated frame references do not establish full procedures or exact event boundaries.',
                'Both models may share errors; possible_issue remains a machine suggestion for review.']}
        save('summary.json',summary)
        status('completed',total_tasks=len(predictions))
    except Exception as exc:
        status('failed',error_type=type(exc).__name__)
        raise


if __name__=='__main__':
    main()
