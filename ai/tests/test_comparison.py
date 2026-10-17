import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_detectors import compare_reports
from train_detector import binary_metrics


def report(scores):
    labels = [0, 0, 1, 1]
    return {"training_completed": True,
            "validation_video_results": [{"video_id": f"PRIVATE_{i}", "label": label,
                "scan_scores": [score], "scan_times_sec": [0.5], "video_score": score}
                for i, (label, score) in enumerate(zip(labels, scores))],
            "validation_video_metrics": binary_metrics(labels, scores)}


class ComparisonTests(unittest.TestCase):
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
