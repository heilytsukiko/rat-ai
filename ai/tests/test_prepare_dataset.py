import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_dataset import (assign_splits, atoms, movie_metadata,
                             timestamp_seconds, video_id)


def box(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


class DatasetTests(unittest.TestCase):
    def test_real_timestamp_formats(self):
        self.assertEqual(timestamp_seconds("1минута 8 секунда"), 68)
        self.assertEqual(timestamp_seconds("57 секунда"), 57)
        self.assertEqual(timestamp_seconds("1 минута"), 60)
        self.assertIsNone(timestamp_seconds(None))

    def test_unknown_or_negative_timestamps_fail(self):
        for value in ("", "8 bananas", "1:08", "-8 секунд", -1, float("nan")):
            if value == "":
                self.assertIsNone(timestamp_seconds(value))
            else:
                with self.assertRaises(ValueError):
                    timestamp_seconds(value)

    def test_cyrillic_and_latin_ids_match(self):
        self.assertEqual(video_id("Х-47.mov"), "X-47")
        self.assertEqual(video_id("X-47"), "X-47")
        with self.assertRaises(ValueError):
            video_id("other-47")

    def test_duplicate_videos_do_not_cross_splits(self):
        rows = [{"video_id": str(i), "sha256": f"hash-{label}-{i}", "label": label}
                for label in (0, 1) for i in range(10)]
        rows.append({**rows[0], "video_id": "copy", "event_time_sec": 68})
        assign_splits(rows)
        self.assertEqual(rows[0]["split"], rows[-1]["split"])
        self.assertEqual(rows[-1]["event_time_sec"], 68)
        validation_hashes = {r["sha256"] for r in rows if r["split"] == "validation"}
        train_hashes = {r["sha256"] for r in rows if r["split"] == "train"}
        self.assertFalse(validation_hashes & train_hashes)
        first = [r["split"] for r in rows]
        assign_splits(rows)
        self.assertEqual(first, [r["split"] for r in rows])

    def test_identical_videos_with_conflicting_labels_fail(self):
        with self.assertRaises(ValueError):
            assign_splits([{"sha256": "same", "label": label} for label in (0, 1)])

    def test_read_movie_versions_without_decoding(self):
        for version in (0, 1):
            mvhd = (bytes([version, 0, 0, 0]) + bytes(16 if version else 8)
                    + struct.pack(">IQ" if version else ">II", 1000, 72700))
            tkhd = bytes(76) + struct.pack(">II", 464 * 65536, 848 * 65536)
            hdlr = bytes(8) + b"vide" + bytes(12)
            trak = box(b"trak", box(b"tkhd", tkhd) + box(b"mdia", box(b"hdlr", hdlr)))
            movie = box(b"moov", box(b"mvhd", mvhd) + trak)
            scratch = Path(__file__).resolve().parents[1] / "local" / "tests"
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=scratch) as tmp:
                path = Path(tmp) / "fixture.mov"
                path.write_bytes(movie)
                meta = movie_metadata(path)
                self.assertAlmostEqual(meta["duration_sec"], 72.7)
                self.assertEqual(meta["track_width"], 464)
                self.assertEqual(meta["track_height"], 848)

    def test_corrupt_box_and_extended_box(self):
        with self.assertRaises(ValueError):
            list(atoms(io.BytesIO(struct.pack(">I4s", 99, b"moov")), 0, 8))
        extended = struct.pack(">I4sQ", 1, b"free", 16)
        self.assertEqual(list(atoms(io.BytesIO(extended), 0, 16)), [(b"free", 16, 16)])


if __name__ == "__main__":
    unittest.main()
