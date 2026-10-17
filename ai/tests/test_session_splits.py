import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI / "tools"))
from prepare_dataset import apply_review, assign_session_splits, write_outputs
from review_dataset import FIELDS
from train_detector import unique_videos


def fixture():
    return [{"video_id": f"X-{label * 10 + i}", "label": label,
             "sha256": f"sha-{label}-{i}", "session_group": f"scene-{i}",
             "event_time_sec": 2.0 if label else None}
            for label in (0, 1) for i in range(6)]


class SessionSplitTests(unittest.TestCase):
    def test_mixed_label_sessions_and_transitive_duplicates_stay_together(self):
        rows = fixture()
        # One file bridges two named sessions. This must merge both sessions.
        rows.append({**rows[0], "video_id": "X-99", "session_group": "scene-1"})
        assign_session_splits(rows)
        for key in ("sha256", "session_group", "split_group"):
            train = {r[key] for r in rows if r["split"] == "train"}
            validation = {r[key] for r in rows if r["split"] == "validation"}
            self.assertFalse(train & validation)
        for split in ("train", "validation"):
            self.assertEqual({r["label"] for r in rows if r["split"] == split}, {0, 1})
        a, b = [r for r in rows if r["session_group"] == "scene-0"][0], rows[1]
        self.assertEqual(a["split_group"], b["split_group"])
        self.assertEqual(len(unique_videos(rows)), 12)

    def test_source_order_and_duplicate_annotation_count_do_not_change_split(self):
        first = fixture()
        second = list(reversed(copy.deepcopy(first)))
        second.extend([{**second[0], "video_id": "X-101"}, {**second[0], "video_id": "X-102"}])
        assign_session_splits(first)
        assign_session_splits(second)
        self.assertEqual({r["sha256"]: r["split"] for r in first},
                         {r["sha256"]: r["split"] for r in second})
        self.assertEqual(sum(r["split"] == "validation" for r in first), 2)

    def test_incomplete_impossible_and_conflicting_groups_fail_without_mutation(self):
        incomplete = fixture()
        incomplete[0]["session_group"] = ""
        impossible = [{**r, "session_group": "all-one-scene"} for r in fixture()]
        conflict = fixture()
        conflict[0]["sha256"] = conflict[-1]["sha256"]
        for rows in (incomplete, impossible, conflict):
            original = copy.deepcopy(rows)
            with self.assertRaises(ValueError):
                assign_session_splits(rows)
            self.assertEqual(rows, original)

    def test_training_rejects_manually_broken_session_manifest(self):
        rows = fixture()
        assign_session_splits(rows)
        tampered = copy.deepcopy(rows)
        tampered[0]["split"] = "train" if tampered[0]["split"] == "validation" else "validation"
        with self.assertRaisesRegex(ValueError, "session crosses"):
            unique_videos(tampered)
        for row in rows:
            del row["split_strategy"]
        with self.assertRaisesRegex(ValueError, "rebuilding"):
            unique_videos(rows)


class ReviewImportTests(unittest.TestCase):
    def setUp(self):
        scratch = AI / "local/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "review.csv"
        self.rows = fixture()
        self.records = [{**{field: r.get(field, "") for field in FIELDS},
                         "review_status": "pending"} for r in self.rows]

    def save(self, records):
        with self.path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)

    def test_import_preserves_source_and_accepts_multiline_human_annotations(self):
        self.records[-1].update(event_type="example", observed_action='quoted "action"\nsecond line',
                                rule_id="rule-1", review_status="reviewed")
        self.save(self.records)
        imported = apply_review(self.rows, self.path)
        self.assertEqual(imported[-1]["observed_action"], self.records[-1]["observed_action"])
        self.assertEqual(imported[-1]["event_time_sec"], 2.0)
        self.assertNotIn("observed_action", self.rows[-1])

    def test_source_edits_unknown_ids_duplicates_and_missing_rows_fail(self):
        for field, value in (("label", 1), ("event_time_sec", 5), ("video_id", "X-500"),
                             ("review_status", "anything")):
            records = copy.deepcopy(self.records)
            records[0][field] = value
            self.save(records)
            with self.assertRaises(ValueError):
                apply_review(self.rows, self.path)
        for records in (self.records[:-1], self.records + [self.records[0]]):
            self.save(records)
            with self.assertRaises(ValueError):
                apply_review(self.rows, self.path)

    def test_reviewed_status_requires_annotations_and_rerun_preserves_csv(self):
        self.records[-1]["review_status"] = "reviewed"
        self.save(self.records)
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "required human"):
            apply_review(self.rows, self.path)
        output = Path(self.directory.name)
        existing = output / "event_review.csv"
        existing.write_bytes(original)
        rows = [{**r, "video_path": "fixture.mov", "duration_sec": 10,
                 "bytes": 12, "track_width": 96, "track_height": 64} for r in fixture()]
        assign_session_splits(rows)
        summary = write_outputs(rows, output)
        self.assertEqual(existing.read_bytes(), original)
        self.assertEqual(summary["session_groups"], 6)
        self.assertEqual(summary["split_strategy"], "reviewed_sessions_v1")


if __name__ == "__main__":
    unittest.main()
