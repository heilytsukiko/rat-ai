"""Compare local full-video reports without publishing video IDs or annotations."""
import argparse
import json
import math
from pathlib import Path

from train_detector import binary_metrics


def validate_report(report):
    if not report.get("training_completed"):
        raise ValueError("Training must be complete")
    rows = report["validation_video_results"]
    indexed = {row["video_id"]: row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError("Validation IDs must be nonempty and unique")
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Both validation classes are required")
    for row in rows:
        scores, times = row["scan_scores"], row["scan_times_sec"]
        if not scores or len(scores) != len(times):
            raise ValueError("Full scan scores and times are required")
        if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise ValueError("Invalid scan scores")
        if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Invalid scan times")
        if not math.isclose(row["video_score"], max(scores), abs_tol=1e-7):
            raise ValueError("Video score must equal peak window score")
    threshold = report["validation_video_metrics"]["threshold"]
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    metrics = binary_metrics([r["label"] for r in rows], [r["video_score"] for r in rows], threshold)
    positives = [r["video_score"] for r in rows if r["label"]]
    negatives = [r["video_score"] for r in rows if not r["label"]]
    # Pairwise ROC AUC includes half credit for tied scores; no threshold search.
    metrics["roc_auc"] = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (
        len(positives) * len(negatives))
    for key in ("tp", "tn", "fp", "fn"):
        if metrics[key] != report["validation_video_metrics"][key]:
            raise ValueError("Stored metrics disagree with full-video predictions")
    return indexed, metrics


def compare_reports(baseline, candidate):
    before, old = validate_report(baseline)
    after, new = validate_report(candidate)
    if before.keys() != after.keys() or old["threshold"] != new["threshold"]:
        raise ValueError("Comparison requires the same videos and fixed threshold")
    for key, row in before.items():
        other = after[key]
        if row["label"] != other["label"] or row["scan_times_sec"] != other["scan_times_sec"]:
            raise ValueError("Comparison requires identical labels and scan timestamps")
    # A practical gate for this experiment, not a claim of test-set generalization.
    promote = new["fp"] < old["fp"] and new["fn"] <= old["fn"]
    return {
        "schema_version": 1, "validation_videos": len(before),
        "baseline": old, "candidate": new,
        "false_alarms_reduced_without_more_misses": promote,
        "limitations": ["Repeatedly used validation set; no independent test set.",
                        "A positive video can be flagged at an incorrect timestamp.",
                        "Exact copies grouped; related filming sessions may cross splits.",
                        "Binary candidates do not establish a violation type or legal source."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_reports(json.loads(args.baseline.read_text(encoding="utf-8")),
                             json.loads(args.candidate.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
