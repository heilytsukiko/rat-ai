import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_detector import binary_metrics, training_samples, unique_videos
from rat_ai.inference import candidate_events


class TrainingDataTests(unittest.TestCase):
    def test_copies_share_all_annotations_and_cannot_cross_splits(self):
        a = {"sha256": "a", "label": 1, "split": "train", "event_time_sec": 57}
        b = {**a, "event_time_sec": 68}
        c = {"sha256": "b", "label": 0, "split": "validation", "event_time_sec": None}
        groups = unique_videos([a, b, c])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["event_times_sec"], [57, 68])
        with self.assertRaises(ValueError):
            unique_videos([a, {**b, "split": "validation"}, c])

    def test_unannotated_positive_periods_are_not_assumed_negative(self):
        rows = [{"video_id": "X-1", "sha256": "a", "video_path": "a.mov",
                 "duration_sec": 20, "label": 1, "split": "train", "event_times_sec": [10]},
                {"video_id": "X-2", "sha256": "b", "video_path": "b.mov",
                 "duration_sec": 20, "label": 0, "split": "train", "event_times_sec": []}]
        samples = training_samples(rows)
        positives = [s for s in samples if s["sha256"] == "a"]
        self.assertEqual([s["center_sec"] for s in positives], [9.5, 10, 10.5])
        self.assertTrue(all(s["label"] == 1 for s in positives))
        self.assertEqual(len([s for s in samples if s["label"] == 0]), 4)
        dense = training_samples(rows, dense_negative_stride=2)
        self.assertEqual(len([s for s in dense if s["label"] == 0]), 10)
        validation_rows = [{**row, "split": "validation"} for row in rows]
        self.assertEqual(len([s for s in training_samples(validation_rows, 2) if s["label"] == 0]), 4)

    def test_metrics_do_not_hide_false_alarms_or_misses(self):
        m = binary_metrics([1, 1, 0, 0], [0.9, 0.2, 0.7, 0.1])
        self.assertEqual([m[k] for k in ["tp", "tn", "fp", "fn"]], [1, 1, 1, 1])
        self.assertEqual(m["f1"], 0.5)
        self.assertIsNone(binary_metrics([1], [0.1])["precision"])

    def test_consecutive_candidates_merge_but_distinct_events_do_not(self):
        scored = {"scan_times_sec": [1, 3, 5, 7, 9], "scan_scores": [0.6, 0.9, 0.1, 0.8, 0.7]}
        events = candidate_events(scored)
        self.assertEqual([e["timecode_sec"] for e in events], [3, 7])
        self.assertEqual(len(candidate_events(scored, threshold=0.95)), 0)
        with self.assertRaises(ValueError):
            candidate_events(scored, threshold=2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires CPU training dependencies")
class FineTuningTests(unittest.TestCase):
    def test_gradients_reach_visual_tail_but_not_frozen_prefix(self):
        import torch
        from rat_ai.temporal import TemporalDetector
        torch.set_num_threads(2)
        model = TemporalDetector()
        model.train()
        cached = torch.randn(2, 3, 96, 7, 7)
        before = model.tail[0].weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model.forward_cached(cached).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.prefix.parameters()))
        self.assertIsNotNone(model.tail[0].weight.grad)
        optimizer.step()
        self.assertGreater(float((model.tail[0].weight.detach() - before).abs().sum()), 0)
        self.assertFalse(model.prefix.training)


if __name__ == "__main__":
    unittest.main()
