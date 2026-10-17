"""Download pinned model artifacts. This command never opens dataset videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_ID = "helenai/Qwen3-VL-2B-Instruct-int4"
REVISION = "61f75fcd9f922da40af9463a30bdd231fe4f6f14"


def download_part(task):
    url, destination, start, end, total = task
    expected = end - start + 1
    if destination.exists() and destination.stat().st_size == expected:
        return
    for attempt in range(6):
        try:
            completed = destination.stat().st_size if destination.exists() else 0
            if completed == expected:
                return
            if completed > expected:
                destination.unlink()
                completed = 0
            actual_start = start + completed
            request = urllib.request.Request(url, headers={"Range": f"bytes={actual_start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    if actual_start != 0 or end != total - 1 or response.status != 200:
                        raise ValueError("Server did not honor the requested download range")
                elif response.headers.get("Content-Range") != f"bytes {actual_start}-{end}/{total}":
                    raise ValueError("Unexpected download range")
                with destination.open("ab" if completed else "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if destination.stat().st_size != expected:
                raise ValueError(f"Incomplete download part {destination.name}: "
                                 f"{destination.stat().st_size}/{expected} bytes")
            print(json.dumps({"stage": "downloaded_part", "file": destination.name,
                              "bytes": expected}), flush=True)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1)


def download_snapshot(output: Path, workers: int, mirror: bool = False):
    lock_path = Path(__file__).resolve().parents[1] / "model-artifacts.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["repository"] != MODEL_ID or lock["revision"] != REVISION:
        raise ValueError("Artifact checksum lock does not match the pinned model")
    entries = lock["entries"]
    # These hashes were captured from the original HF API, never from a mirror.
    (output / "artifact_checksums.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    endpoint = "https://hf-mirror.com" if mirror else "https://huggingface.co"
    parts_dir = output / ".download-parts"
    parts_dir.mkdir(exist_ok=True)
    tasks = []
    parts_by_file = {}
    for entry in sorted(entries, key=lambda e: e["size"], reverse=True):
        name = entry["path"]
        if Path(name).name != name:
            raise ValueError("Unexpected nested model artifact")
        path = output / name
        expected_hash = entry.get("lfs", {}).get("oid", entry["oid"])
        is_lfs = "lfs" in entry
        if path.exists() and file_hash(path, is_lfs) == expected_hash:
            continue
        size = entry["size"]
        chunk = 32 * 1024 * 1024
        parts = []
        for index, start in enumerate(range(0, size, chunk)):
            end = min(start + chunk, size) - 1
            part = parts_dir / f"{name}.{index:04d}"
            parts.append(part)
            tasks.append((f"{endpoint}/{MODEL_ID}/resolve/{REVISION}/{name}",
                          part, start, end, size))
        parts_by_file[name] = parts
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(download_part, tasks))
    for entry in entries:
        name = entry["path"]
        if name not in parts_by_file:
            continue
        temporary = output / f"{name}.assembling"
        with temporary.open("wb") as stream:
            for part in parts_by_file[name]:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, stream)
        expected_hash = entry.get("lfs", {}).get("oid", entry["oid"])
        if file_hash(temporary, "lfs" in entry) != expected_hash:
            for part in parts_by_file[name]:
                part.unlink()
            raise ValueError(f"Model artifact checksum mismatch: {name}")
        temporary.replace(output / name)
        for part in parts_by_file[name]:
            part.unlink()
    (output / "artifact_checksums.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def file_hash(path: Path, is_lfs: bool) -> str:
    digest = hashlib.sha256() if is_lfs else hashlib.sha1()
    if not is_lfs:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("ai/local/models/qwen3-vl-2b-int4"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--transport", choices=["http", "hf"], default="http")
    parser.add_argument("--mirror", action="store_true",
                        help="Use a mirror for file bytes; verify original HF checksum lock")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.transport == "hf":
        if args.mirror:
            parser.error("mirror is only supported by the HTTP transport")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=args.output,
                          allow_patterns=["*.json", "*.xml", "*.bin", "*.txt", "*.jinja", "README.md"],
                          max_workers=args.workers)
    # Also verifies every artifact downloaded by the Hugging Face transport.
    download_snapshot(args.output, args.workers, args.mirror)
    provenance = {"base_model": "Qwen/Qwen3-VL-2B-Instruct",
                  "artifact_repository": MODEL_ID, "revision": REVISION,
                  "format": "OpenVINO INT4", "trust_remote_code": False,
                  "download_mirror": "https://hf-mirror.com" if args.mirror else None,
                  "note": "Third-party preview export; validate compatibility and quality."}
    (args.output / "rat_ai_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"downloaded": True, "path": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
