"""Locally decode beginning/middle/end frames; report health, never image content."""
import argparse
import json
from pathlib import Path

from benchmark_vlm import sample_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("ai/local/dataset/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("ai/local/dataset/decode_checks.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    checks = []
    for row in rows:
        try:
            frames, times, metadata = sample_frames(Path(row["video_path"]), 0,
                                                    row["duration_sec"], 3, 336)
            checks.append({"video_id": row["video_id"], "status": "ok",
                           "sampled_times_sec": times, "metadata": metadata,
                           "decoded_shapes": [list(f.shape) for f in frames]})
        except Exception as error:
            checks.append({"video_id": row["video_id"], "status": "failed",
                           "reason": str(error)})
    report = {"checked": len(checks), "failed": sum(r["status"] != "ok" for r in checks),
              "limitation": "Three sample frames per video; not a full decode integrity check.",
              "videos": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checked": report["checked"], "failed": report["failed"],
                      "report": str(args.output)}))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
