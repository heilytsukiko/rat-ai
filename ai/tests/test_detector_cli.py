"""The CLI must use the calibrated checkpoint threshold unless overridden."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('infer_detector', Path(__file__).resolve().parents[1] / 'tools/infer_detector.py')
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class DetectorCliTests(unittest.TestCase):
    def run_cli(self, extra):
        scored = {'scan_times_sec': [1.0], 'scan_scores': [0.6], 'processing_sec': 0.1}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'result.json'
            with patch('sys.argv', ['infer_detector', '--video', 'sample.mov', '--output', str(output), *extra]), patch('rat_ai.inference.load_detector', return_value=(object(), {'threshold': 0.8, 'architecture': 'test'})), patch('rat_ai.inference.score_video', return_value=scored) as scan, contextlib.redirect_stdout(io.StringIO()):
                CLI.main()
                return json.loads(output.read_text()), scan.call_count

    def test_checkpoint_threshold_is_used(self):
        result, scans = self.run_cli([])
        self.assertEqual(result['threshold'], 0.8)
        self.assertEqual(result['candidates'], [])
        self.assertEqual(scans, 1)

    def test_explicit_override_is_used(self):
        result, _ = self.run_cli(['--threshold', '0.5'])
        self.assertEqual(len(result['candidates']), 1)

    def test_invalid_threshold_fails_before_decoding(self):
        with patch('sys.argv', ['infer_detector', '--video', 'missing.mov', '--threshold', 'nan']), patch('rat_ai.inference.load_detector', return_value=(object(), {'threshold': 0.8})), patch('rat_ai.inference.score_video') as scan:
            with self.assertRaises(ValueError):
                CLI.main()
            scan.assert_not_called()
