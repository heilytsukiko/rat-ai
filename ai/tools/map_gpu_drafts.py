"""Map private GPU suggestions back to local video IDs after checking dataset identity."""
import argparse
import hashlib
import json
from pathlib import Path
from train_detector import unique_videos


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('ai/local/dataset/manifest.jsonl'))
    parser.add_argument('--dataset',type=Path,default=Path('ai/local/gpu-dataset/dataset.json'))
    parser.add_argument('--drafts',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('ai/local/review/ai-drafts.json'))
    args=parser.parse_args()
    source=args.dataset.read_bytes()
    dataset=json.loads(source)
    drafts=json.loads(args.drafts.read_text(encoding='utf-8'))
    if drafts.get('human_review_required') is not True or drafts.get('dataset_sha256')!=hashlib.sha256(source).hexdigest():
        raise ValueError('Suggestions do not match the exact prepared dataset')
    rows=[json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines()]
    unique=unique_videos(rows)
    if len(unique)!=len(dataset['groups']):
        raise ValueError('Manifest changed since GPU dataset preparation')
    group_by_sha={}
    for i,(video,group) in enumerate(zip(unique,dataset['groups'])):
        if (group['group']!=i or video['label']!=group['label'] or video['split']!=group['split']
            or video['duration_sec']!=group['duration_sec']
            or video['event_times_sec']!=group['annotated_event_times_sec']):
            raise ValueError('Manifest does not match GPU group order')
        group_by_sha[video['sha256']]=i
    mapped={}
    for row in rows:
        options=[d for d in drafts['drafts'] if d['group']==group_by_sha[row['sha256']]]
        if not options:
            continue
        target=row['event_time_sec'] if row['event_time_sec'] is not None else row['duration_sec']/2
        item=min(options,key=lambda d:abs(d['target_sec']-target))
        mapped[row['video_id']]={key:item.get(key,'') for key in ('observed_action','event_type','uncertainty')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'human_review_required':True,'by_video_id':mapped},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'videos_with_drafts':len(mapped),'human_review_required':True}))


if __name__=='__main__':
    main()
