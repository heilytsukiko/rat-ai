import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_detector import (binary_metrics, training_samples, unique_videos, video_groups,
                            video_loss, balanced_video_loss, mil_samples, mil_video_loss, cached_video_metrics)
from rat_ai.inference import candidate_events


class TrainingDataTests(unittest.TestCase):
    def test_mil_validation_inputs_ignore_event_timestamps_and_training_auxiliary_is_separate(self):
        row = {"video_id": "X-1", "sha256": "a", "video_path": "a.mov", "duration_sec": 8,
               "label": 1, "split": "validation", "event_times_sec": [3]}
        samples = mil_samples([row])
        self.assertEqual(samples, mil_samples([{**row, "event_times_sec": [7]}]))
        self.assertEqual([s["center_sec"] for s in samples], [.5, 2.5, 4.5, 6.5])
        self.assertFalse(any(s["point_supervised"] for s in samples))
        train = mil_samples([{**row, "split": "train"}])
        self.assertEqual([s["center_sec"] for s in train if s["point_supervised"]], [3])
        self.assertTrue(all(not s["scan_window"] for s in train if s["point_supervised"]))
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
        self.assertEqual(len([s for s in training_samples(validation_rows, 2, dense_validation=True)
                              if s["label"] == 0]), 10)

    def test_video_groups_exclude_other_split_and_merge_windows(self):
        rows = [{"sha256": "negative", "label": 0, "split": "train"},
                {"sha256": "positive", "label": 1, "split": "train"},
                {"sha256": "negative", "label": 0, "split": "train"},
                {"sha256": "held-out", "label": 1, "split": "validation"}]
        groups = video_groups(rows, "train")
        self.assertEqual([g["indices"] for g in groups], [[0, 2], [1]])
        with self.assertRaises(ValueError):
            video_groups(rows + [{**rows[0], "label": 1}], "train")

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
    def test_diagnostic_scores_exclude_high_auxiliary_annotation_logits(self):
        import torch
        class Scores:
            def forward_cached(self, features):
                return features
        groups = [{"label": 0, "indices": [0, 1, 2], "scan_positions": [0, 1]},
                  {"label": 1, "indices": [3, 4], "scan_positions": [0]}]
        metrics, loss = cached_video_metrics(Scores(), torch.tensor([-2.0, -1.0, 10.0, 2.0, -10.0]), groups)
        self.assertEqual([metrics[k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 0, 0])
        self.assertEqual(metrics["roc_auc"], 1)
        self.assertGreater(loss, 0)

    def test_mil_gradient_uses_peak_and_known_points_without_labeling_all_positive_windows(self):
        import torch
        logits = torch.tensor([-3.0, 1.0, -2.0], requires_grad=True)
        mil_video_loss(logits, 1, [0, 1], [2]).backward()
        self.assertEqual(float(logits.grad[0]), 0)
        self.assertLess(float(logits.grad[1]), 0)
        self.assertLess(float(logits.grad[2]), 0)
        negative = torch.tensor([-3.0, 1.0], requires_grad=True)
        mil_video_loss(negative, 0).backward()
        self.assertEqual(float(negative.grad[0]), 0)
        self.assertGreater(float(negative.grad[1]), 0)
        with self.assertRaises(ValueError):
            mil_video_loss(negative, 0, [])

    def test_mil_validation_matches_deployment_peak_binary_loss(self):
        import torch
        logits = torch.tensor([-.3, 1.7, .2])
        for label in (0, 1):
            expected = torch.nn.functional.binary_cross_entropy(logits.sigmoid().max(), torch.tensor(float(label)))
            self.assertAlmostEqual(float(mil_video_loss(logits, label)), float(expected), places=6)

    def test_hard_negative_loss_targets_false_alarm_and_pushes_scores_down(self):
        import torch
        logits = torch.tensor([-4.0, -4.0, 3.0], requires_grad=True)
        loss = video_loss(logits, 0)
        loss.backward()
        self.assertTrue(all(value > 0 for value in logits.grad.tolist()))
        self.assertGreater(float(logits.grad[2]), 10 * float(logits.grad[0]))
        self.assertGreater(float(loss.detach()), float(torch.nn.functional.softplus(logits.detach()).mean()))

    def test_each_video_has_equal_weight_independent_of_window_count(self):
        import torch
        class Scores:
            def forward_cached(self, features):
                return features
        groups = [{"label": 0, "indices": [0]}, {"label": 0, "indices": [1]},
                  {"label": 1, "indices": [2]}]
        short = balanced_video_loss(Scores(), torch.tensor([-1.0, 2.0, 0.0]), groups)
        groups[1]["indices"] = [1, 3, 4, 5]
        long = balanced_video_loss(Scores(), torch.tensor([-1.0, 2.0, 0.0, 2.0, 2.0, 2.0]), groups)
        self.assertAlmostEqual(float(short), float(long), places=6)

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
