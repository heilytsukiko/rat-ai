"""Prepare private 16-frame context and pre-resize crops from local original videos."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_video_rule_chain import select_windows


def letterbox(image,size=448):
    import cv2
    import numpy as np
    h,w=image.shape[:2]
    scale=size/max(h,w)
    nh,nw=max(1,round(h*scale)),max(1,round(w*scale))
    canvas=np.full((size,size,3),127,dtype=np.uint8)
    y,x=(size-nh)//2,(size-nw)//2
    canvas[y:y+nh,x:x+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    return canvas


def main():
    import numpy as np
    from train_detector import unique_videos
    from prepare_gpu_dataset import frame_clip
    from rat_ai.video import sample_frames
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh source-context directory required')
    raw=(args.dataset/'dataset.json').read_bytes()
    dataset=json.loads(raw)
    selected=select_windows(dataset)
    videos=unique_videos([json.loads(x) for x in args.manifest.read_text(encoding='utf-8').splitlines()])
    if len(videos)!=len(dataset['groups']):
        raise ValueError('Source group mapping changed')
    groups={g['group']:g for g in dataset['groups']}
    selected_groups=sorted({dataset['samples'][i]['group'] for i in selected})
    # Verify source bytes AND reproduce one original bundle window per video.
    # Group order alone is not accepted as proof that the clips correspond.
    for group_id in selected_groups:
        source=videos[group_id]
        group=groups[group_id]
        if source['split']!='validation' or abs(source['duration_sec']-group['duration_sec'])>.001:
            raise ValueError('Source timing or split mismatch')
        with Path(source['video_path']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=source['sha256']:
                raise ValueError('Original video changed')
        index=group['samples'][0]
        row=dataset['samples'][index]
        recreated=frame_clip(source['video_path'],row['center_sec'],group['duration_sec'])
        with np.load(args.dataset/row['file'],allow_pickle=False) as archive:
            if not np.array_equal(recreated,archive['frames']):
                raise ValueError('Original video does not reproduce the authorized bundle')
    args.output.mkdir(parents=True)
    records=[]
    for index,target in enumerate(selected):
        row=dataset['samples'][target]
        source=videos[row['group']]
        center=row['center_sec']
        start=max(0,center-6)
        end=min(source['duration_sec']-.001,center+6)
        frames,times,metadata=sample_frames(Path(source['video_path']),start,max(.001,end-start),16,2048)
        anchor=min(range(len(times)),key=lambda i:abs(times[i]-center))
        image=frames[anchor]
        h,w=image.shape[:2]
        th,tw=round(h*2/3),round(w*2/3)
        crops=[letterbox(image[y:y+th,x:x+tw]) for y in (0,h-th) for x in (0,w-tw)]
        filename=f'{target:05d}.npz'
        np.savez_compressed(args.output/filename,frames=np.stack([letterbox(f) for f in frames]),
                            tiles=np.stack(crops),decoded_time_sec=np.asarray(times,dtype=np.float64))
        records.append({'sample_index':target,'group':row['group'],'center_sec':center,'file':filename,
                        'anchor_index':anchor,'frame_count':len(frames),'distinct_times':len(set(times)),
                        'source_sha256':source['sha256'],'seek_fallbacks':metadata['seek_fallbacks']})
        print(json.dumps({'prepared':index+1,'total':len(selected)}),flush=True)
    value={'schema_version':1,'source_dataset_sha256':hashlib.sha256(raw).hexdigest(),
           'data_kind':'private_original_video_context','source_mapping_verified':True,
           'frame_size':448,'frames_per_window':16,'extra_tiles':4,'records':records,
           'limitations':['Fixed crops are not object or hand detections.',
                          'Recorded frame times do not establish event boundaries; sparse sampling is not continuous footage.']}
    (args.output/'context.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({'completed':True,'windows':len(records),'videos':len(selected_groups),
                      'source_mapping_verified':True,'bytes':sum(p.stat().st_size for p in args.output.iterdir())}))


if __name__=='__main__':
    main()
