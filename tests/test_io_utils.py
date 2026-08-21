import http.server
import json
import os
import threading
import unittest
from unittest.mock import MagicMock

from io_utils import (
    IoError,
    download_audio,
    filename_from_url,
    parse_gcs_uri,
    upload_json,
)


class ParseGcsUriTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(
            parse_gcs_uri("gs://my-bucket/episodes/ep.json"),
            ("my-bucket", "episodes/ep.json"),
        )

    def test_rejects_http(self):
        with self.assertRaises(IoError):
            parse_gcs_uri("https://example.com/a.wav")

    def test_rejects_bucket_only(self):
        with self.assertRaises(IoError):
            parse_gcs_uri("gs://my-bucket")

    def test_rejects_trailing_slash(self):
        with self.assertRaises(IoError):
            parse_gcs_uri("gs://my-bucket/prefix/")


class FilenameFromUrlTests(unittest.TestCase):
    def test_http(self):
        self.assertEqual(
            filename_from_url("https://cdn.example.com/shows/ep35.mp3"),
            "ep35.mp3",
        )

    def test_gcs(self):
        self.assertEqual(
            filename_from_url("gs://audio/raw/ep35.wav"),
            "ep35.wav",
        )

    def test_empty_path(self):
        self.assertEqual(filename_from_url("https://example.com"), "audio.bin")


class FakeBlob:
    def __init__(self, data=b"abc", size=None):
        self._data = data
        self.size = size if size is not None else len(data)
        self.uploaded = None
        self.content_type = None

    def reload(self):
        return None

    def download_to_filename(self, path):
        with open(path, "wb") as handle:
            handle.write(self._data)

    def upload_from_string(self, data, content_type=None):
        self.uploaded = data
        self.content_type = content_type


class FakeClient:
    def __init__(self, blob):
        self._blob = blob

    def bucket(self, _name):
        bucket = MagicMock()
        bucket.blob.return_value = self._blob
        return bucket


class GcsIoTests(unittest.TestCase):
    def test_download_gcs(self, tmp=None):
        import tempfile

        blob = FakeBlob(b"wav-bytes")
        with tempfile.TemporaryDirectory() as dest:
            path = download_audio(
                "gs://audio/clip.wav", dest, client=FakeClient(blob)
            )
            self.assertTrue(path.endswith("clip.wav"))
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"wav-bytes")

    def test_download_gcs_too_large(self):
        import tempfile

        blob = FakeBlob(b"x", size=10**12)
        with tempfile.TemporaryDirectory() as dest:
            with self.assertRaises(IoError):
                download_audio(
                    "gs://audio/huge.wav", dest, max_bytes=100, client=FakeClient(blob)
                )

    def test_upload_json(self):
        blob = FakeBlob()
        payload = {"0": {"start_sec": 1.0, "end_sec": 1.5}}
        upload_json("gs://out/ep.json", payload, client=FakeClient(blob))
        self.assertEqual(json.loads(blob.uploaded), payload)
        self.assertEqual(blob.content_type, "application/json")


class HttpDownloadTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = os.path.join(self.tmp.name, "clip.wav")
        with open(self.audio_path, "wb") as handle:
            handle.write(b"RIFF....WAVE")

        directory = self.tmp.name

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, _format, *args):
                return

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def test_http_download(self):
        import tempfile

        url = f"http://127.0.0.1:{self.port}/clip.wav"
        with tempfile.TemporaryDirectory() as dest:
            path = download_audio(url, dest)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"RIFF....WAVE")

    def test_rejects_unsupported_scheme(self):
        import tempfile

        with tempfile.TemporaryDirectory() as dest:
            with self.assertRaises(IoError):
                download_audio("ftp://example.com/a.wav", dest)


if __name__ == "__main__":
    unittest.main()
