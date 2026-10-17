import os
import time

os.environ['API_KEY'] = 'test-key-at-least-16-chars'

from fastapi.testclient import TestClient
from app import main

HEADERS = {'X-API-Key': os.environ['API_KEY']}


def test_api_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA', tmp_path)
    from app import pipeline
    monkeypatch.setattr(pipeline, 'analyze', lambda _: {'verdict': 'no_violation', 'violations': []})
    main.initialize()
    with TestClient(main.app) as client:
        assert client.get('/api/v1/jobs/missing').status_code == 401
        assert client.get('/api/v1/jobs/missing', headers=HEADERS).status_code == 404
        assert client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('x.txt', b'data')}).status_code == 415
        assert client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('x.mov', b'')}).status_code == 422
        monkeypatch.setattr(main, 'MAX_BYTES', 3)
        assert client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('x.mov', b'large')}).status_code == 413
        monkeypatch.setattr(main, 'MAX_BYTES', 100)
        response = client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('x.mov', b'video')})
        assert response.status_code == 202
        job_id = response.json()['job_id']
        for _ in range(100):
            result = client.get('/api/v1/jobs/' + job_id, headers=HEADERS).json()
            if result['status'] != 'processing':
                break
            time.sleep(.01)
        assert result == {'job_id': job_id, 'status': 'done', 'verdict': 'no_violation', 'violations': []}
        assert not list(tmp_path.glob('*.video'))


def test_restart_and_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA', tmp_path)
    main.initialize()
    with main.db() as conn:
        conn.execute("INSERT INTO jobs(id,status) VALUES ('interrupted','processing')")
    main.initialize()
    assert main.get_job('interrupted')['status'] == 'failed'
    monkeypatch.setattr(main, 'CAPACITY', 0)
    # No lifespan: the first test has already shut down the executor; this request never submits work.
    client = TestClient(main.app)
    assert client.post('/api/v1/analyze', headers=HEADERS, files={'file': ('x.mov', b'data')}).status_code == 429


def test_model_schema_rejects_invalid_evidence():
    import pytest
    from pydantic import ValidationError
    from app.pipeline import Prediction
    with pytest.raises(ValidationError):
        Prediction.model_validate({'verdict': 'unknown', 'violations': []})


def test_failure_is_persisted_and_video_deleted(tmp_path, monkeypatch):
    from app import pipeline
    monkeypatch.setattr(main, 'DATA', tmp_path)
    main.initialize()
    with main.db() as conn:
        conn.execute("INSERT INTO jobs(id,status) VALUES ('broken','processing')")
    video = tmp_path / 'broken.video'
    video.write_bytes(b'bad video')

    def fail(_):
        raise RuntimeError('private model diagnostic')

    monkeypatch.setattr(pipeline, 'analyze', fail)
    main.process('broken', video)
    result = main.get_job('broken')
    assert result['status'] == 'failed'
    assert 'private model diagnostic' not in result['error']
    assert not video.exists()
