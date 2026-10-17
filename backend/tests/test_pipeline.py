import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from app import pipeline


def configure(monkeypatch, prediction):
    monkeypatch.setenv('ANALYZER_COMMAND', '["local-model"]')

    def run(args, **kwargs):
        assert kwargs['timeout'] <= 240
        if args[0] == 'ffprobe':
            return SimpleNamespace(stdout=json.dumps({'format': {'duration': '20'}, 'streams': [{'codec_type': 'video'}]}).encode())
        if args[0] == 'local-model':
            Path(args[-1]).write_text(json.dumps(prediction), encoding='utf-8')
            return SimpleNamespace(stdout=b'')
        return SimpleNamespace(stdout=b'\xff\xd8jpeg\xff\xd9')

    monkeypatch.setattr(pipeline.subprocess, 'run', run)


def event():
    return {'type': 'Test', 'description': 'Observed action', 'timecode_sec': 12, 'evidence_timestamps_sec': [12], 'source': {'title': 'Test fixture only', 'section': '1', 'url': 'https://adilet.zan.kz/test'}}


def test_evidence_and_source(monkeypatch):
    configure(monkeypatch, {'verdict': 'violation', 'violations': [event()]})
    result = pipeline.analyze(Path('test.mov'))
    assert result['verdict'] == 'violation'
    assert result['violations'][0]['source']['section'] == '1'
    frame = result['violations'][0]['evidence_frames'][0]
    assert frame['timestamp_sec'] == 12
    assert base64.b64decode(frame['image_base64']).startswith(b'\xff\xd8')


@pytest.mark.parametrize('prediction', [
    {'verdict': 'violation', 'violations': []},
    {'verdict': 'no_violation', 'violations': [event()]},
    {'verdict': 'violation', 'violations': [{**event(), 'timecode_sec': 21}]},
    {'verdict': 'violation', 'violations': [{**event(), 'evidence_timestamps_sec': [1]}]},
])
def test_invalid_model_results(monkeypatch, prediction):
    configure(monkeypatch, prediction)
    with pytest.raises(ValueError):
        pipeline.analyze(Path('test.mov'))


def test_queue_wait_reduces_remaining_inference_budget(monkeypatch):
    configure(monkeypatch, {'verdict': 'no_violation', 'violations': []})
    fake_run = pipeline.subprocess.run
    timeouts = []

    def measured_run(args, **kwargs):
        timeouts.append(kwargs['timeout'])
        return fake_run(args, **kwargs)

    monkeypatch.setattr(pipeline.subprocess, 'run', measured_run)
    result = pipeline.analyze(Path('test.mov'), deadline=pipeline.time.monotonic() + 2)
    assert result['verdict'] == 'no_violation'
    assert timeouts and all(0 < timeout <= 2 for timeout in timeouts)
