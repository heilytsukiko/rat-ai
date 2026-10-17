import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BenchmarkTimeoutTests(unittest.TestCase):
    def test_expired_worker_is_failed_and_cannot_leave_a_successful_status(self):
        root = Path(__file__).resolve().parents[1]
        scratch = root / "local" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            output = Path(directory)
            (output / "metrics.json").write_text('{"status":"done"}', encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(root / "tools" / "benchmark_vlm.py"),
                 "--timeout-sec", "0.001", "--min-free-ram-gb", "0", "--output", str(output)],
                capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["status"], "failed")
            self.assertIn("timed out", metrics["reason"])


if __name__ == "__main__":
    unittest.main()
