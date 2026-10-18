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
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument('--describe-with-config',type=Path,help='Optional cached GPU base-model config')
    args = parser.parse_args()
    started = time.perf_counter()
    import torch
    from rat_ai.inference import candidate_events, load_detector, score_video, validate_candidate_policy
    torch.set_num_threads(4)
    model, config = load_detector(args.checkpoint)
    policy = validate_candidate_policy(config.get("candidate_policy"))
    threshold = config["threshold"] if args.threshold is None else args.threshold
    candidate_events({"scan_times_sec": [], "scan_scores": []}, threshold, **policy)
    analysis_started=time.perf_counter()
    scored = score_video(model, args.video)
    result = {**scored, "experimental": True, "candidate_policy": policy,
              "candidates": candidate_events(scored, threshold, **policy),
              "threshold": threshold, "model": config["architecture"],
              "total_elapsed_sec": time.perf_counter() - started,
              "note": "Binary statistical candidates; event types and legal sources are not assigned."}
    if args.describe_with_config:
        from describe_gpu_candidates import describe
        result['candidates']=describe(args.video,result['candidates'],args.describe_with_config)
        result['observation_model']='Qwen/Qwen2.5-VL-7B-Instruct'
        result['processing_sec']=time.perf_counter()-analysis_started
    result['total_elapsed_sec']=time.perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", "candidates": len(result["candidates"]),
                      "processing_sec": result["processing_sec"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
