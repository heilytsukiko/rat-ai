import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
sys.path.insert(0, str(AI / "tools"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires detector dependencies")
class PublicPackageTests(unittest.TestCase):
    def setUp(self):
        scratch = AI / "local/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)

    def test_export_drops_private_training_configuration_and_preserves_tensors(self):
        import torch
        from safetensors import safe_open
        from export_detector import export_checkpoint
        from rat_ai.inference import load_detector

        model, config = load_detector()
        source = Path(self.directory.name) / "private.pt"
        output = Path(self.directory.name) / "public.safetensors"
        private_config = {**config, "pretrained_url": "https://example.invalid/public-weights",
                          "pretrained_sha256": "test", "sample_spec": [
                              {"video_path": "PRIVATE_PATH_SENTINEL", "event_time_sec": 123}]}
        torch.save({"state_dict": model.state_dict(), "config": private_config, "epoch": 6}, source)
        policy = {"min_windows": 2, "representative": "center"}
        exported = export_checkpoint(source, output, candidate_policy=policy)
        serialized = output.with_suffix(".json").read_text(encoding="utf-8")
        for private_key in ("PRIVATE_PATH_SENTINEL", "sample_spec", "event_time_sec", "video_path"):
            self.assertNotIn(private_key, serialized)
        with safe_open(output, framework="pt", device="cpu") as tensors:
            self.assertEqual(set(tensors.metadata()), {"architecture", "format"})
        restored, restored_config = load_detector(output)
        self.assertEqual(restored_config['candidate_policy'], policy)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        self.assertEqual(exported["best_epoch"], 6)

    def test_corrupted_weights_are_rejected_before_parsing(self):
        from rat_ai.inference import load_detector
        artifact = AI / "rat_ai/artifacts/detector-v2.safetensors"
        output = Path(self.directory.name) / artifact.name
        body = bytearray(artifact.read_bytes())
        body[-1] ^= 1
        output.write_bytes(body)
        output.with_suffix(".json").write_bytes(artifact.with_suffix(".json").read_bytes())
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_detector(output)

    def test_packaged_detector_runs_on_synthetic_video_without_network(self):
        import cv2
        import numpy as np
        from rat_ai import VideoDetector

        video = Path(self.directory.name) / "synthetic.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (96, 64))
        self.assertTrue(writer.isOpened())
        try:
            for i in range(12):
                frame = np.full((64, 96, 3), 100, dtype=np.uint8)
                frame[20:40, i * 4:i * 4 + 12] = (0, 0, 255)
                writer.write(frame)
        finally:
            writer.release()
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network during inference")):
            result = VideoDetector(threads=2).analyze_video(video)
        self.assertTrue(result["experimental"])
        self.assertGreater(len(result["scan_scores"]), 0)
        self.assertTrue(all(0 <= score <= 1 for score in result["scan_scores"]))
        self.assertTrue(all(0 <= t < 3 for t in result["scan_times_sec"]))
        self.assertNotIn("verdict", result)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
