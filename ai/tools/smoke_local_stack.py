"""Exercise the real local Vite proxy -> API -> detector -> JPEG pipeline."""
import argparse
import base64
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse


def synthetic_video(path):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 8, (192, 128))
    if not writer.isOpened():
        raise RuntimeError('Cannot create the synthetic integration video')
    try:
        for index in range(48):
            frame = np.full((128, 192, 3), 160, dtype=np.uint8)
            frame[50:90, (index * 3) % 150:(index * 3) % 150 + 30] = (20, 30, 200)
            writer.write(frame)
    finally:
        writer.release()


def smoke(url, video, output):
    import httpx
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username or parsed.password:
        raise ValueError('This smoke test accepts only local HTTP servers')
    started = time.monotonic()
    deadline = started + 300
    with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
        page = client.get('/')
        page.raise_for_status()
        if '<div id="root"' not in page.text:
            raise ValueError('Expected the actual frontend server')
        with Path(video).open('rb') as stream:
            submitted = client.post('/api/v1/analyze', files={'file': ('integration.mov', stream, 'video/quicktime')})
        if submitted.status_code != 202:
            raise ValueError(f'Upload failed with HTTP {submitted.status_code}')
        job_id = submitted.json()['job_id']
        while True:
            response = client.get('/api/v1/jobs/' + job_id)
            response.raise_for_status()
            result = response.json()
            if result.get('job_id') != job_id:
                raise ValueError('Job identity mismatch')
            if result['status'] == 'done':
                break
            if result['status'] != 'processing' or time.monotonic() >= deadline:
                raise RuntimeError('Analysis failed or timed out; inspect local backend log')
            time.sleep(.5)
    if result.get('analysis_mode') != 'experimental' or result.get('experimental') is not True:
        raise ValueError('Expected real experimental detector output')
    if 'verdict' in result or 'violations' in result:
        raise ValueError('Raw detector output must not invent a legal verdict')
    duration, frames, described = result['duration_sec'], 0, 0
    if not math.isfinite(duration) or not 0 < duration <= 180:
        raise ValueError('Invalid duration')
    for candidate in result['candidates']:
        if 'observation_draft' in candidate:
            if (result.get('observation_model')!='Qwen/Qwen2.5-VL-7B-Instruct'
                or not isinstance(candidate['observation_draft'],str)
                or not 0<len(candidate['observation_draft'].strip())<=1600):
                raise ValueError('Invalid observation draft')
            described+=1
        if not 0 <= candidate['timecode_sec'] < duration or not 0 <= candidate['score'] <= 1:
            raise ValueError('Invalid candidate')
        if 'source' in candidate or 'type' in candidate:
            raise ValueError('Unexpected classification or normative source')
        if not 1 <= len(candidate['evidence_frames']) <= 3:
            raise ValueError('Missing evidence frames')
        for frame in candidate['evidence_frames']:
            if not base64.b64decode(frame['image_base64'], validate=True).startswith(b'\xff\xd8'):
                raise ValueError('Invalid JPEG')
            if not 0 <= frame['timestamp_sec'] < duration or abs(frame['timestamp_sec'] - candidate['timecode_sec']) > 5:
                raise ValueError('Invalid evidence timestamp')
            frames += 1
    report = {'status': 'passed', 'real_detector': True, 'via_frontend_proxy': True,
              'analysis_mode': result['analysis_mode'], 'duration_sec': duration,
              'candidate_policy': result.get('candidate_policy'),
              'observation_model': result.get('observation_model'), 'candidates_described': described,
              'candidates': len(result['candidates']), 'jpeg_frames_checked': frames,
              'elapsed_sec': time.monotonic() - started,
              'quality_evaluated': False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5173')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, default=Path('ai/local/integration/smoke.json'))
    args = parser.parse_args()
    video = args.video or Path('ai/local/integration/synthetic.mov')
    if args.video is None:
        synthetic_video(video)
    print(json.dumps(smoke(args.url, video, args.output)))


if __name__ == '__main__':
    main()
