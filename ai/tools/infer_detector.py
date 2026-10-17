"""Run a trained detector offline; output statistical candidates, not legal conclusions."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=None, help="Defaults to the packaged experimental detector")
    parser.add_argument("--output", type=Path, default=Path("ai/local/prediction.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    started = time.perf_counter()
    import torch
    from rat_ai.inference import candidate_events, load_detector, score_video, validate_candidate_policy
    torch.set_num_threads(4)
    model, config = load_detector(args.checkpoint)
    policy = validate_candidate_policy(config.get("candidate_policy"))
    scored = score_video(model, args.video)
    result = {**scored, "experimental": True, "candidate_policy": policy,
              "candidates": candidate_events(scored, args.threshold, **policy),
              "threshold": args.threshold, "model": config["architecture"],
              "total_elapsed_sec": time.perf_counter() - started,
              "note": "Binary statistical candidates; event types and legal sources are not assigned."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", "candidates": len(result["candidates"]),
                      "processing_sec": result["processing_sec"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
