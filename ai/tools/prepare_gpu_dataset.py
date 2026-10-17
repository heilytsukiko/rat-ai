"""Create a private frame bundle for the team's explicitly selected GPU server."""
import argparse
import hashlib
import json
from pathlib import Path


def frame_clip(path, center, duration, size=336):
    import cv2
    import numpy as np
    from rat_ai.video import sample_frames
    frames = []
    start = max(0, center - 1.5)
    end = min(duration - .001, center + 1.5)
    images, _, _ = sample_frames(Path(path), start, max(.001, end-start), 4, size)
    for image in images:
        h, w = image.shape[:2]
        scale = size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        canvas = np.full((size, size, 3), 127, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
        frames.append(canvas)
    return np.stack(frames)


def main():
    import numpy as np
    from train_detector import unique_videos
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('ai/local/dataset/manifest.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('ai/local/gpu-dataset'))
    args = parser.parse_args()
    videos = unique_videos([json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines()])
    args.output.mkdir(parents=True, exist_ok=True)
    samples, groups = [], []
    for index, row in enumerate(videos):
        with Path(row['video_path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                raise ValueError('Source changed since split preparation')
        duration = row['duration_sec']
        if row['split'] == 'validation':
            centers = np.arange(min(.5, duration / 2), duration, 2).tolist()
        elif row['label']:
            centers = sorted(set(min(max(p + offset, .1), max(.1, duration - .15))
                                 for p in row['event_times_sec'] for offset in (-.5, 0, .5)))
        else:
            centers = np.linspace(min(.5, duration / 2), max(.5, duration - .5), 5).tolist()
        group = {'group': index, 'split': row['split'], 'label': row['label'],
                 'duration_sec': duration, 'samples': [],
                 'annotated_event_times_sec': row['event_times_sec']}
        for center in centers:
            sample_id = len(samples)
            filename = f'{sample_id:05d}.npz'
            np.savez_compressed(args.output / filename,
                                frames=frame_clip(row['video_path'], center, duration))
            samples.append({'file': filename, 'center_sec': center, 'group': index,
                            'label': row['label'], 'split': row['split']})
            group['samples'].append(sample_id)
        groups.append(group)
        print(json.dumps({'videos_prepared': index + 1, 'windows': len(samples)}), flush=True)
    payload = {'schema_version': 1, 'frames_per_clip': 4, 'clip_span_sec': 3,
               'input_size': 336, 'validation_stride_sec': 2,
               'samples': samples, 'groups': groups,
               'limitations': ['Exact copies grouped; related filming sessions may cross splits.',
                               'Positive training clips use point annotations, without verified action types.']}
    (args.output / 'dataset.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'ready', 'videos': len(groups), 'windows': len(samples)}))


if __name__ == '__main__':
    main()
