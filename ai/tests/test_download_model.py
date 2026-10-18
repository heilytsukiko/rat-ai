import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from download_model import download_part, file_hash


class FakeResponse(io.BytesIO):
    status = 206
    headers = {"Content-Range": "bytes 0-3/4"}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "local" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)

    def test_model_hashes_support_lfs_and_git_blobs(self):
        path = Path(self.directory.name) / "file"
        path.write_bytes(b"test")
        self.assertEqual(file_hash(path, True), hashlib.sha256(b"test").hexdigest())
        self.assertEqual(file_hash(path, False), hashlib.sha1(b"blob 4\0test").hexdigest())

    def test_download_validates_server_range(self):
        path = Path(self.directory.name) / "file.part"
        task = ("https://example.invalid/file", path, 0, 3, 4)
        with patch("download_model.urllib.request.urlopen", return_value=FakeResponse(b"test")):
            download_part(task)
        self.assertEqual(path.read_bytes(), b"test")
        with patch("download_model.urllib.request.urlopen") as network:
            download_part(task)
            network.assert_not_called()  # resume a completed part

    def test_wrong_range_is_rejected(self):
        path = Path(self.directory.name) / "file.part"
        task = ("https://example.invalid/file", path, 4, 7, 8)
        with patch("download_model.urllib.request.urlopen",
                   side_effect=lambda *a, **kw: FakeResponse(b"test")), \
                patch("download_model.time.sleep"):
            with self.assertRaises(ValueError):
                download_part(task)
        self.assertFalse(path.exists())

    def test_resume_partial_response(self):
        path = Path(self.directory.name) / "file.part"
        path.write_bytes(b"te")
        response = FakeResponse(b"st")
        response.headers = {"Content-Range": "bytes 2-3/4"}
        with patch("download_model.urllib.request.urlopen", return_value=response) as network:
            download_part(("https://example.invalid/file", path, 0, 3, 4))
            request = network.call_args.args[0]
            self.assertEqual(request.get_header("Range"), "bytes=2-3")
        self.assertEqual(path.read_bytes(), b"test")


if __name__ == "__main__":
    unittest.main()
