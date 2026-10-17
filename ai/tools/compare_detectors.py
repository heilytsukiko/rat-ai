"""Compare local full-video reports without publishing video IDs or annotations."""
import argparse
import json
import math
from pathlib import Path

from train_detector import binary_metrics
from rat_ai.inference import candidate_events, validate_candidate_policy


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
        if not all(math.isfinite(t) and t >= 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Invalid scan times")
        if "annotated_event_times_sec" in row:
            points = row["annotated_event_times_sec"]
            if (not isinstance(points, list) or any(not math.isfinite(t) or t < 0 for t in points)
                or bool(points) != bool(row["label"])):
                raise ValueError("Invalid point annotations")
        if not math.isclose(row["video_score"], max(scores), abs_tol=1e-7):
            raise ValueError("Video score must equal peak window score")
    threshold = report["validation_video_metrics"]["threshold"]
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    policy = validate_candidate_policy(report.get("candidate_policy"))
    decisions = []
    for row in rows:
        decision = row["video_score"]
        if "candidate_policy" in report:
            decision = max((e["score"] for e in candidate_events(row, threshold, 2, **policy)), default=-1)
            if row.get("decision_score") != decision:
                raise ValueError("Decision score disagrees with candidate policy")
        decisions.append(decision)
    metrics = binary_metrics([r["label"] for r in rows], decisions, threshold)
    positives = [score for r, score in zip(rows, decisions) if r["label"]]
    negatives = [score for r, score in zip(rows, decisions) if not r["label"]]
    # Pairwise ROC AUC includes half credit for tied scores; no threshold search.
    metrics["roc_auc"] = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (
        len(positives) * len(negatives))
    for key in ("tp", "tn", "fp", "fn"):
        if metrics[key] != report["validation_video_metrics"][key]:
            raise ValueError("Stored metrics disagree with full-video predictions")
    return indexed, metrics


def localization_metrics(rows, threshold, policy=None):
    """Evaluate only emitted candidate peaks, with one-to-one point matching.

    Unannotated parts of positive videos do not become false-positive labels.
    A fixed two-second tolerance measures point proximity, not event semantics.
    """
    policy = validate_candidate_policy(policy)
    if any("annotated_event_times_sec" not in row for row in rows):
        return None
    total, matched, negative_candidates = 0, 0, 0
    for row in rows:
        peaks = sorted(event["timecode_sec"] for event in candidate_events(row, threshold, stride_sec=2, **policy))
        if not row["label"]:
            negative_candidates += len(peaks)
            continue
        points = sorted(row["annotated_event_times_sec"])
        total += len(points)
        i, j = 0, 0
        while i < len(points) and j < len(peaks):
            if peaks[j] < points[i] - 2:
                j += 1
            elif peaks[j] > points[i] + 2:
                i += 1
            else:
                matched += 1
                i += 1
                j += 1
    return {"tolerance_sec": 2, "annotated_points": total, "matched_points": matched,
            "point_recall": matched / total if total else None,
            "negative_video_candidates": negative_candidates}


def compare_reports(baseline, candidate):
    before, old = validate_report(baseline)
    after, new = validate_report(candidate)
    if before.keys() != after.keys() or old["threshold"] != new["threshold"]:
        raise ValueError("Comparison requires the same videos and fixed threshold")
    for key, row in before.items():
        other = after[key]
        if row["label"] != other["label"] or row["scan_times_sec"] != other["scan_times_sec"]:
            raise ValueError("Comparison requires identical labels and scan timestamps")
        if row.get("annotated_event_times_sec") != other.get("annotated_event_times_sec"):
            raise ValueError("Comparison requires identical point annotations")
    # A practical gate for this experiment, not a claim of test-set generalization.
    promote = new["fp"] < old["fp"] and new["fn"] <= old["fn"]
    old_policy = validate_candidate_policy(baseline.get("candidate_policy"))
    new_policy = validate_candidate_policy(candidate.get("candidate_policy"))
    old_points = localization_metrics(list(before.values()), old["threshold"], old_policy)
    new_points = localization_metrics(list(after.values()), new["threshold"], new_policy)
    eligible = bool(promote and old_points is not None and new_points is not None
                    and new_points["matched_points"] >= old_points["matched_points"]
                    and new_points["negative_video_candidates"] <= old_points["negative_video_candidates"])
    return {
        "schema_version": 1, "validation_videos": len(before),
        "baseline": old, "candidate": new,
        "false_alarms_reduced_without_more_misses": promote,
        "baseline_localization": old_points, "candidate_localization": new_points,
        "baseline_candidate_policy": old_policy, "candidate_policy": new_policy,
        "candidate_eligible_for_packaging": eligible,
        "limitations": ["Repeatedly used validation set; no independent test set.",
                        "Point proximity does not verify action type, event duration or legal meaning.",
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
