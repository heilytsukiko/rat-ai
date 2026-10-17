import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_vlm import sample_frames


@unittest.skipUnless(importlib.util.find_spec("cv2"), "Requires inference environment")
class VideoSamplingTests(unittest.TestCase):
    def test_decode_order_timestamps_resize_and_rgb(self):
        import cv2
        import numpy as np

        scratch = Path(__file__).resolve().parents[1] / "local" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "synthetic.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                     2, (128, 64))
            self.assertTrue(writer.isOpened())
            try:
                for color in [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]:
                    writer.write(np.full((64, 128, 3), color, dtype=np.uint8))
            finally:
                writer.release()
            frames, times, metadata = sample_frames(path, 0, 1, 3, 64)
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, (32, 64, 3))
            self.assertEqual(times, [0, 0.5, 1])
            self.assertEqual(metadata["fps"], 2)
            self.assertGreater(frames[0][0, 0, 0], 240)  # BGR red converted to RGB
            self.assertGreater(frames[2][0, 0, 2], 240)  # third frame is blue
            with self.assertRaises(ValueError):
                sample_frames(path, 20, 1, 3, 64)

    def test_missing_video_is_an_error(self):
        with self.assertRaises(ValueError):
            sample_frames(Path("ai/local/does-not-exist.mov"), 0, 1, 2, 64)

    def test_failed_tail_seek_uses_actual_last_frame_timestamp(self):
        import cv2
        import numpy as np

        class Capture:
            def __init__(self):
                self.timestamp = 0
                self.tail = False
                self.samples = iter([True, False, True, True, False])

            def isOpened(self):
                return True

            def get(self, key):
                return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 11,
                        cv2.CAP_PROP_POS_MSEC: self.timestamp}[key]

            def set(self, key, value):
                self.tail = value == 500

            def read(self):
                ok = next(self.samples)
                if not ok:
                    self.timestamp = 0  # a decoder may reset this at EOF
                    return False, None
                self.timestamp = (900 if self.timestamp == 800 else 800) if self.tail else 0
                return True, np.zeros((64, 64, 3), dtype=np.uint8)

            def release(self):
                pass

        existing = Path(__file__)
        with patch("cv2.VideoCapture", return_value=Capture()):
            frames, times, meta = sample_frames(existing, 0, 1, 2, 64)
        self.assertEqual(times, [0, 0.9])
        self.assertEqual(meta["seek_fallbacks"], 1)
        self.assertEqual(len(frames), 2)


if __name__ == "__main__":
    unittest.main()
