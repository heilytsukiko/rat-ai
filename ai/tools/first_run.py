"""Resume setup and run offline benchmarks; video observations stay local."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from process_utils import run_with_timeout

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    state_path = ROOT / "ai/local/first_run_status.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "stages": [], "training_started": False}

    def save():
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    def run(name, script, extra=(), timeout=600):
        stage = {"name": name, "status": "running"}
        state["stages"].append(stage)
        save()
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        try:
            process = run_with_timeout([sys.executable, str(ROOT / "ai/tools" / script), *extra],
                                       cwd=ROOT, timeout=timeout)
            stage["returncode"] = process.returncode
            stage["status"] = "done" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            stage.update(status="failed", reason="stage_timeout")
        save()
        return stage["status"] == "done"

    try:
        downloads = ["--workers", str(args.workers)] + (["--mirror"] if args.mirror else [])
        if not run("download_and_verify", "download_model.py", downloads, timeout=7200):
            raise RuntimeError("Model download or checksum verification failed")
        if not run("prepare_tokenizer", "prepare_tokenizer.py"):
            raise RuntimeError("Offline tokenizer preparation failed")
        successes = []
        for device in ("CPU", "GPU"):
            output = f"ai/local/benchmarks/smoke-{device.lower()}"
            ok = run(f"synthetic_smoke_{device.lower()}", "benchmark_vlm.py",
                     ["--device", device, "--max-new-tokens", "40", "--output", output], timeout=330)
            if ok:
                successes.append(device)
        if not successes:
            raise RuntimeError("Neither CPU nor GPU completed a synthetic model inference")
        device = "GPU" if "GPU" in successes else "CPU"
        manifest = ROOT / "ai/local/dataset/manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        # Diagnostic on a training clip, never an accuracy evaluation.
        row = next(r for r in rows if r["split"] == "train" and r["label"] == 1)
        start = max(0, row["event_time_sec"] - 4)
        ok = run("training_clip_observations", "benchmark_vlm.py",
                 ["--device", device, "--video", row["video_path"],
                  "--start-sec", str(start), "--duration-sec", "8", "--frames", "4",
                  "--output", "ai/local/benchmarks/training-clip"], timeout=330)
        state["status"] = ("done" if ok and all(s["status"] == "done" for s in state["stages"])
                           else "done_with_errors" if ok else "failed")
        state["clip_device"] = device
        state["note"] = "Diagnostic model run only; no training or detection accuracy claim."
        save()
    except Exception as error:
        state.update(status="failed", reason=str(error))
        save()
        raise
    if state["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
