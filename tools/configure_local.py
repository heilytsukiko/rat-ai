"""Create ignored local configuration without overwriting existing settings."""
import argparse
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]


def configure(root=ROOT):
    backend = root / 'backend/.env'
    if not backend.exists():
        template = (root / 'backend/.env.example').read_text(encoding='utf-8')
        template = template.replace('replace-with-a-random-secret-at-least-16-characters', secrets.token_urlsafe(32))
        # Exclusive creation prevents overwriting configuration from another process.
        with backend.open('x', encoding='utf-8') as stream:
            stream.write(template)
        backend.chmod(0o600)
        return True
    return False


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    created = configure()
    print('Created backend/.env with a random API key.' if created else 'Kept existing backend/.env unchanged.')
    print('Start: docker compose up --build -d. Open http://127.0.0.1:8080')
