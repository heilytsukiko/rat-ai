import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_detectors import compare_reports, localization_metrics
from train_detector import binary_metrics


def report(scores):
    labels = [0, 0, 1, 1]
    return {"training_completed": True,
            "validation_video_results": [{"video_id": f"PRIVATE_{i}", "label": label,
                "scan_scores": [score], "scan_times_sec": [0.5], "video_score": score}
                for i, (label, score) in enumerate(zip(labels, scores))],
            "validation_video_metrics": binary_metrics(labels, scores)}


class ComparisonTests(unittest.TestCase):
    def test_classification_improvement_cannot_conceal_worse_candidate_timestamps(self):
        baseline, candidate = report([.9, .9, .9, .9]), report([.1, .9, .9, .9])
        for fixture in (baseline, candidate):
            for row in fixture["validation_video_results"]:
                row["scan_times_sec"] = [.5, 2.5, 4.5]
                row["scan_scores"] = [.1, row["video_score"], .1]
                row["annotated_event_times_sec"] = [2.5] if row["label"] else []
        good = compare_reports(baseline, candidate)
        self.assertTrue(good["candidate_eligible_for_packaging"])
        for row in candidate["validation_video_results"]:
            if row["label"]:
                row["scan_scores"] = [row["video_score"], .1, .1]
                row["annotated_event_times_sec"] = [4.5]
        # Keep labels identical, but move the emitted candidate away from the point.
        for row in baseline["validation_video_results"]:
            if row["label"]:
                row["scan_scores"] = [.1, .1, row["video_score"]]
                row["annotated_event_times_sec"] = [4.5]
        worse = compare_reports(baseline, candidate)
        self.assertTrue(worse["false_alarms_reduced_without_more_misses"])
        self.assertFalse(worse["candidate_eligible_for_packaging"])
        self.assertEqual(worse["candidate_localization"]["matched_points"], 0)

    def test_one_candidate_peak_cannot_match_two_annotations_or_hidden_positive_windows(self):
        row = {"label": 1, "scan_times_sec": [.5, 2.5, 4.5, 6.5],
               "scan_scores": [.8, .9, .8, .1], "annotated_event_times_sec": [1.5, 3.5]}
        metrics = localization_metrics([row], .5)
        self.assertEqual(metrics["matched_points"], 1)
        self.assertEqual(metrics["point_recall"], .5)
        # A threshold-crossing window near a point does not count if its group peak is far away.
        row["scan_scores"] = [.6, .7, .9, .1]
        row["annotated_event_times_sec"] = [.5]
        self.assertEqual(localization_metrics([row], .5)["matched_points"], 0)

    def test_fewer_false_alarms_cannot_hide_more_misses(self):
        baseline = report([0.9, 0.9, 0.9, 0.9])
        improved = compare_reports(baseline, report([0.1, 0.9, 0.9, 0.9]))
        self.assertTrue(improved["false_alarms_reduced_without_more_misses"])
        self.assertEqual(improved["baseline"]["roc_auc"], 0.5)
        self.assertEqual(improved["candidate"]["roc_auc"], 0.75)
        self.assertNotIn("PRIVATE", str(improved))
        worse = compare_reports(baseline, report([0.1, 0.1, 0.1, 0.9]))
        self.assertFalse(worse["false_alarms_reduced_without_more_misses"])

    def test_auc_detects_reversed_ranking_despite_full_recall(self):
        comparison = compare_reports(report([0.9, 0.9, 0.8, 0.8]), report([0.1, 0.1, 0.8, 0.8]))
        self.assertEqual(comparison["baseline"]["recall"], 1)
        self.assertEqual(comparison["baseline"]["roc_auc"], 0)
        self.assertEqual(comparison["candidate"]["roc_auc"], 1)

    def test_incomparable_or_tampered_reports_fail(self):
        baseline = report([0.9, 0.9, 0.9, 0.9])
        for field, value in [("scan_times_sec", [1.5]), ("label", 1),
                             ("video_id", "different"), ("video_score", 0.1),
                             ("scan_scores", [float("nan")])]:
            changed = copy.deepcopy(baseline)
            changed["validation_video_results"][0][field] = value
            with self.assertRaises(ValueError):
                compare_reports(baseline, changed)
        changed = copy.deepcopy(baseline)
        changed["validation_video_metrics"]["threshold"] = 0.6
        with self.assertRaises(ValueError):
            compare_reports(baseline, changed)


if __name__ == "__main__":
    unittest.main()
