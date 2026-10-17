"""Local inference adapter. No network clients, generated verdicts or face identification."""
import base64
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    title: str = Field(min_length=1, max_length=1000)
    section: str = Field(min_length=1, max_length=500)
    url: str = Field(pattern=r'^https://(adilet\.zan\.kz|nationalbank\.kz|www\.nationalbank\.kz)/')


class Event(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str = Field(min_length=1, max_length=300)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1, max_length=4000)
    source: Source
    evidence_timestamps_sec: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=1, max_length=3)


class Prediction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    verdict: Literal['violation', 'no_violation']
    violations: list[Event] = Field(max_length=100)


def analyze(video: Path, deadline: float | None = None):
    local_deadline = time.monotonic() + min(240, max(1, int(os.getenv('ANALYSIS_TIMEOUT_SEC', '240'))))
    deadline = min(deadline, local_deadline) if deadline is not None else local_deadline

    def run(args):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Analysis exceeded time budget')
        return subprocess.run(args, capture_output=True, check=True, timeout=remaining)

    probe = json.loads(run([os.getenv('FFPROBE_BIN', 'ffprobe'), '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(video)]).stdout)
    duration = float(probe['format']['duration'])
    if not math.isfinite(duration) or not 0 < duration <= 180:
        raise ValueError('Expected video duration between 0 and 180 seconds')
    if not any(s.get('codec_type') == 'video' for s in probe['streams']):
        raise ValueError('No video stream')
    command = json.loads(os.getenv('ANALYZER_COMMAND', '[]'))
    if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
        raise RuntimeError('Configure ANALYZER_COMMAND as a JSON argv array for the local model')
    with tempfile.TemporaryDirectory(prefix='rat-inference-') as directory:
        output = Path(directory) / 'prediction.json'
        run([*command, '--video', str(video.resolve()), '--output', str(output)])
        if output.stat().st_size > 1024 * 1024:
            raise ValueError('Model output too large')
        prediction = Prediction.model_validate_json(output.read_text(encoding='utf-8'))
        if (prediction.verdict == 'violation') != bool(prediction.violations):
            raise ValueError('Verdict conflicts with events')
        events = []
        for event in prediction.violations:
            if event.timecode_sec >= duration:
                raise ValueError('Event outside video')
            frames = []
            for timestamp in event.evidence_timestamps_sec:
                if timestamp >= duration or abs(timestamp - event.timecode_sec) > 5:
                    raise ValueError('Evidence outside event window')
                jpeg = run([os.getenv('FFMPEG_BIN', 'ffmpeg'), '-v', 'error', '-ss', str(timestamp), '-i', str(video), '-map', '0:v:0', '-frames:v', '1', '-vf', 'scale=960:-2', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1']).stdout
                if not jpeg.startswith(b'\xff\xd8'):
                    raise ValueError('Failed to extract JPEG evidence')
                frames.append({'timestamp_sec': timestamp, 'image_base64': base64.b64encode(jpeg).decode('ascii')})
            events.append({**event.model_dump(exclude={'evidence_timestamps_sec'}), 'evidence_frames': frames})
    return {'verdict': prediction.verdict, 'violations': events, 'duration_sec': duration}
