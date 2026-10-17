"""Authenticated, persistent local video analysis API."""
import asyncio
import hmac
import json
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

DATA = Path(os.getenv('DATA_DIR', './storage'))
MAX_BYTES = int(os.getenv('MAX_UPLOAD_MB', '100')) * 1024 * 1024
CAPACITY = int(os.getenv('MAX_PENDING_JOBS', '8'))
API_KEY = os.getenv('API_KEY', '')
executor = ThreadPoolExecutor(max_workers=1)


def db():
    conn = sqlite3.connect(DATA / 'jobs.sqlite3', timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def initialize():
    DATA.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
        conn.execute("UPDATE jobs SET status='failed', payload=? WHERE status='processing'", (json.dumps({'error': 'Service restarted during analysis'}),))


def save(job_id, status, payload):
    with db() as conn:
        conn.execute('UPDATE jobs SET status=?, payload=? WHERE id=?', (status, json.dumps(payload, ensure_ascii=False), job_id))


def process(job_id, path):
    try:
        from .pipeline import analyze
        save(job_id, 'done', analyze(path))
    except Exception:
        import logging
        logging.exception('Analysis failed for job %s', job_id)
        save(job_id, 'failed', {'error': 'Analysis failed. Check local model configuration and server logs.'})
    finally:
        path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app):
    if len(API_KEY) < 16:
        raise RuntimeError('Set API_KEY to a random secret of at least 16 characters')
    initialize()
    yield
    executor.shutdown(wait=True)


app = FastAPI(title='RAT AI — local cash discipline analysis', version='0.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(','), allow_methods=['GET', 'POST'], allow_headers=['X-API-Key', 'Content-Type'])


def authenticate(x_api_key: str = Header(default='')):
    if not API_KEY or not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(401, 'Invalid API key')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/api/v1/analyze', status_code=202, dependencies=[Depends(authenticate)])
async def submit(file: UploadFile = File(...)):
    if Path(file.filename or '').suffix.lower() not in {'.mov', '.mp4'}:
        raise HTTPException(415, 'Expected a .mov or .mp4 video')
    job_id = uuid.uuid4().hex
    path = DATA / (job_id + '.video')
    # Reserve queue capacity atomically, including uploads currently in flight.
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        pending = conn.execute("SELECT count(*) FROM jobs WHERE status='processing'").fetchone()[0]
        if pending >= CAPACITY:
            raise HTTPException(429, 'Analysis queue is full; retry later')
        conn.execute("INSERT INTO jobs(id,status) VALUES (?,'processing')", (job_id,))
    try:
        size = 0
        with path.open('wb') as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Video exceeds upload limit')
                target.write(chunk)
        if not size:
            raise HTTPException(422, 'Empty video')
        executor.submit(process, job_id, path)
    except BaseException:
        path.unlink(missing_ok=True)
        with db() as conn:
            conn.execute('DELETE FROM jobs WHERE id=?', (job_id,))
        raise
    finally:
        await file.close()
    return {'job_id': job_id}


@app.get('/api/v1/jobs/{job_id}', dependencies=[Depends(authenticate)])
def get_job(job_id: str):
    with db() as conn:
        row = conn.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, 'Job not found')
    return {'job_id': row['id'], 'status': row['status'], **json.loads(row['payload'] or '{}')}
