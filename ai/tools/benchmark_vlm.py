"""Offline Qwen VLM smoke/clip benchmark; content stays in local output files."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from process_utils import run_with_timeout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rat_ai.video import sample_frames

# Inference must not access Hugging Face or send telemetry.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["OPENVINO_TELEMETRY_DISABLED"] = "1"




def benchmark(args):
    import numpy as np
    import openvino as ov
    import openvino_genai as genai
    import psutil

    if args.start_sec < 0 or args.duration_sec <= 0 or not 1 <= args.frames <= 16:
        raise ValueError("Require nonnegative start, positive duration and 1–16 frames")
    if args.max_side < 56 or not 1 <= args.max_new_tokens <= 512:
        raise ValueError("Invalid image size or token budget")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak = [0]
    stop = threading.Event()
    process = psutil.Process()

    def measure_memory():
        while not stop.wait(0.1):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=measure_memory, daemon=True)
    monitor.start()
    try:
        if args.video:
            frames, timestamps, video_meta = sample_frames(
                args.video, args.start_sec, args.duration_sec, args.frames, args.max_side)
            prompt = (
                "These images are chronological frames of one video. "
                f"Their timestamps in seconds are {timestamps}. "
                "Describe only directly visible actions and changes between frames. "
                "Do not identify people or infer intent, authorization, accounting entries "
                "or legal violations. If an action is unclear, say so. "
                "Return JSON with keys observations (list of objects with timestamp_sec "
                "and visible_action) and uncertainty (string). Be concise."
            )
        else:
            # Public/artificial smoke input, unrelated to the bank dataset.
            frame = np.full((224, 224, 3), 255, dtype=np.uint8)
            frame[48:176, 48:176] = [255, 0, 0]
            frames, timestamps, video_meta = [frame], [], None
            prompt = "Describe the shape and its color in one short sentence."
        decoded = time.perf_counter()
        available_devices = ov.Core().available_devices
        print(json.dumps({"stage": "loading", "device": args.device,
                          "available_devices": available_devices}), flush=True)
        pipe = genai.VLMPipeline(str(args.model.resolve()), args.device)
        loaded = time.perf_counter()
        print(json.dumps({"stage": "generating", "load_sec": loaded - decoded}), flush=True)
        images = [ov.Tensor(np.ascontiguousarray(frame[None])) for frame in frames]
        result = pipe.generate(prompt, images=images, max_new_tokens=args.max_new_tokens,
                               do_sample=False)
        finished = time.perf_counter()
        # Never print video observations; they remain on the user's computer.
        text = result.texts[0]
        (args.output / "response.txt").write_text(text, encoding="utf-8")
        metrics = {
            "status": "done", "mode": "clip_observations" if args.video else "synthetic_smoke",
            "device": args.device, "available_devices": available_devices,
            "decode_sec": decoded - started, "load_sec": loaded - decoded,
            "generation_sec": finished - loaded, "total_sec": finished - started,
            "peak_process_rss_bytes": max(peak[0], process.memory_info().rss),
            "sampled_frames": len(frames), "sampled_timestamps_sec": timestamps,
            "input_shapes": [list(frame.shape) for frame in frames],
            "video_metadata": video_meta, "max_new_tokens": args.max_new_tokens,
            "response_chars": len(text), "response_file": "response.txt",
            "limitations": ["Diagnostic observations, not a violation detector.",
                            "Frames treated as an ordered image sequence.",
                            "RSS excludes some GPU/shared memory allocations.",
                            "Timestamp precision depends on the decoder."]
        }
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics), flush=True)
    finally:
        stop.set()
        monitor.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("ai/local/models/qwen3-vl-2b-int4"))
    parser.add_argument("--device", choices=["CPU", "GPU", "NPU"], default="CPU")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-sec", type=float, default=0)
    parser.add_argument("--duration-sec", type=float, default=4)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=336)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("ai/local/benchmarks/smoke-cpu"))
    parser.add_argument("--timeout-sec", type=float, default=300)
    parser.add_argument("--min-free-ram-gb", type=float, default=3,
                        help="Refuse model loading below this available RAM threshold")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0:
        parser.error("timeout must be positive and finite")
    if not math.isfinite(args.min_free_ram_gb) or args.min_free_ram_gb < 0:
        parser.error("RAM threshold must be nonnegative and finite")
    if args.worker:
        benchmark(args)
        return
    import psutil
    available = psutil.virtual_memory().available
    if available < args.min_free_ram_gb * 1024 ** 3:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "reason": "insufficient_available_ram",
                   "ram_available_bytes": available,
                   "minimum_available_ram_gb": args.min_free_ram_gb,
                   "device": args.device}
        (args.output / "metrics.json").write_text(json.dumps(failure), encoding="utf-8")
        print(json.dumps(failure), flush=True)
        raise SystemExit(2)
    try:
        result = run_with_timeout([sys.executable, str(Path(__file__).resolve()),
                                   *sys.argv[1:], "--worker"], timeout=args.timeout_sec)
        if result.returncode:
            raise RuntimeError(f"Inference worker exited with code {result.returncode}")
    except (subprocess.TimeoutExpired, RuntimeError) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        # A failed or timed-out run must never be reported as a successful analysis.
        failure = {"status": "failed", "reason": str(error), "device": args.device}
        (args.output / "metrics.json").write_text(json.dumps(failure), encoding="utf-8")
        print(json.dumps(failure), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
