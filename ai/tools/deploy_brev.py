"""Start the repository's site/API on its allocated Brev Linux host."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.request


def main():
    root = Path(__file__).resolve().parents[2]
    runtime = root / 'ai/local/server'
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret_file = runtime / 'api.env'
    if not secret_file.exists():
        secret_file.write_text('API_KEY=' + secrets.token_urlsafe(40) + '\n', encoding='utf-8')
    secret_file.chmod(0o600)
    key = secret_file.read_text(encoding='utf-8').strip().split('=', 1)[1]
    template = runtime / 'nginx.conf.template'
    template.write_text('''server {
    listen 8080;
    root /usr/share/nginx/html;
    client_max_body_size 110m;
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header X-API-Key "${API_KEY}";
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
    location / { try_files $uri $uri/ /index.html; }
}
''', encoding='utf-8')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open('http://127.0.0.1:8000/health', timeout=2):
            api_ready = True
    except OSError:
        api_ready = False
    if not api_ready:
        env = os.environ.copy()
        env.update(API_KEY=key, ANALYSIS_MODE='experimental', DATA_DIR=str(runtime / 'storage'),
                   PYTHONPATH=str(root / 'ai'), USE_TF='0', USE_FLAX='0',
                   ANALYZER_COMMAND=json.dumps([sys.executable, str(root / 'ai/tools/infer_detector.py')]))
        log = (runtime / 'backend.log').open('ab')
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app',
                                    '--host', '127.0.0.1', '--port', '8000', '--workers', '1'],
                                   cwd=root / 'backend', env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        (runtime / 'backend.pid').write_text(str(process.pid), encoding='utf-8')
        for _ in range(60):
            try:
                with opener.open('http://127.0.0.1:8000/health', timeout=1):
                    break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError('API startup failed; inspect private backend.log')
                time.sleep(.5)
        else:
            raise RuntimeError('API startup timed out')
    existing = subprocess.run(['docker', 'container', 'inspect', 'rat-ai-site'],
                              capture_output=True, text=True)
    if existing.returncode:
        subprocess.run(['docker', 'run', '-d', '--name', 'rat-ai-site', '--network', 'host',
                        '--restart', 'unless-stopped', '--label', 'rat-ai.project=rat-ai',
                        '--env-file', str(secret_file), '-v',
                        str(template) + ':/etc/nginx/templates/default.conf.template:ro',
                        'rat-ai-site:night'], check=True, capture_output=True)
    elif json.loads(existing.stdout)[0]['Config'].get('Labels', {}).get('rat-ai.project') != 'rat-ai':
        raise RuntimeError('Container name belongs to another deployment')
    for _ in range(30):
        try:
            with opener.open('http://127.0.0.1:8080/', timeout=2) as response:
                if b'<div id="root"' in response.read():
                    print(json.dumps({'status': 'ready', 'site_port': 8080,
                                      'api_bind': '127.0.0.1:8000', 'mode': 'experimental'}))
                    return
        except OSError:
            pass
        time.sleep(.5)
    raise RuntimeError('Site startup timed out')


if __name__ == '__main__':
    main()
