"""Run the existing frontend/backend with the real local experimental detector."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'ai/local/integration'
STATE = RUNTIME / 'stack.json'


def alive(record):
    import psutil
    try:
        process = psutil.Process(record['pid'])
        return process if process.create_time() == record['created_at'] and process.is_running() else None
    except psutil.NoSuchProcess:
        return None


def stop_record(record):
    import psutil
    process = alive(record)
    if process is None:
        return
    children = process.children(recursive=True)
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
    psutil.wait_procs([*children, process], timeout=5)


def save_state(state):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(STATE)


def stop_saved():
    if not STATE.exists():
        print(json.dumps({'status': 'not_running'}))
        return
    state = json.loads(STATE.read_text(encoding='utf-8'))
    for key in ('frontend', 'backend'):
        if key in state:
            stop_record(state[key])
    save_state({'status': 'stopped'})
    if 'launcher' in state and state['launcher']['pid'] != os.getpid():
        stop_record(state['launcher'])
    print(json.dumps({'status': 'stopped'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg-bin', type=Path)
    parser.add_argument('--ffprobe-bin', type=Path)
    parser.add_argument('--backend-port', type=int, default=8000)
    parser.add_argument('--frontend-port', type=int, default=5173)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    import psutil
    if args.stop:
        stop_saved()
        return
    if STATE.exists():
        previous = json.loads(STATE.read_text(encoding='utf-8'))
        if previous.get('status') in ('starting', 'ready') and alive(previous['launcher']):
            raise RuntimeError('Local stack is already running; use --stop before restarting')
    for module in ('uvicorn', 'fastapi', 'rat_ai'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f'Missing dependency: {module}; install backend requirements and ./ai')
    executables = {}
    for name in ('ffmpeg', 'ffprobe'):
        explicit = getattr(args, f'{name}_bin')
        found = str(explicit.resolve()) if explicit else shutil.which(name)
        if not found or not Path(found).is_file():
            raise RuntimeError(f'Set --{name}-bin to the local {name} executable')
        executables[name] = found
    node = shutil.which('node')
    vite = ROOT / 'frontend/node_modules/vite/bin/vite.js'
    if not node or not vite.is_file():
        raise RuntimeError('Install Node and run npm ci in frontend first')
    if args.backend_port == args.frontend_port:
        parser.error('Frontend and backend need different ports')
    for port in (args.backend_port, args.frontend_port):
        if not 1024 <= port <= 65535:
            parser.error('Ports must be between 1024 and 65535')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / 'tmp').mkdir(exist_ok=True)
    env = os.environ.copy()
    # Shared only between local server processes; never written to a VITE_ variable.
    env.update(API_KEY=secrets.token_urlsafe(32), ANALYSIS_MODE='experimental',
               DATA_DIR=str(RUNTIME / 'storage'), TEMP=str(RUNTIME / 'tmp'), TMP=str(RUNTIME / 'tmp'),
               FFPROBE_BIN=executables['ffprobe'], FFMPEG_BIN=executables['ffmpeg'],
               ANALYZER_COMMAND=json.dumps([sys.executable, str(ROOT / 'ai/tools/infer_detector.py')]),
               BACKEND_URL=f'http://127.0.0.1:{args.backend_port}')
    env['BACKEND_API_KEY'] = env['API_KEY']
    commands = {
        'backend': ([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
                     '--port', str(args.backend_port), '--workers', '1'], ROOT / 'backend'),
        'frontend': ([node, str(vite), '--host', '127.0.0.1', '--port', str(args.frontend_port), '--strictPort'], ROOT / 'frontend'),
    }
    current = psutil.Process()
    state = {'status': 'starting', 'launcher': {'pid': current.pid, 'created_at': current.create_time()},
             'frontend_url': f'http://127.0.0.1:{args.frontend_port}',
             'backend_url': env['BACKEND_URL'], 'analysis_mode': 'experimental'}
    logs, processes = [], []
    try:
        save_state(state)
        for name, (command, directory) in commands.items():
            log = (RUNTIME / f'{name}.log').open('ab')
            logs.append(log)
            process = subprocess.Popen(command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            processes.append(process)
            child = psutil.Process(process.pid)
            state[name] = {'pid': child.pid, 'created_at': child.create_time()}
            save_state(state)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 45
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('A server stopped during startup; inspect ai/local/integration/*.log')
            try:
                with opener.open(state['backend_url'] + '/health', timeout=1) as response:
                    assert json.loads(response.read())['status'] == 'ok'
                with opener.open(state['frontend_url'], timeout=1) as response:
                    assert response.status == 200
                break
            except (OSError, AssertionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Local servers did not start within 45 seconds')
                time.sleep(.25)
        state['status'] = 'ready'
        save_state(state)
        print(json.dumps({'status': 'ready', 'url': state['frontend_url'], 'analysis_mode': 'experimental'}), flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for key in ('frontend', 'backend'):
            if key in state:
                stop_record(state[key])
        for log in logs:
            log.close()
        save_state({**state, 'status': 'stopped'})


if __name__ == '__main__':
    main()
