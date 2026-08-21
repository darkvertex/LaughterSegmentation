import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

import inference


class ResolveInputsTests(unittest.TestCase):
    def test_local_path(self):
        args = inference.parse_args(["--audio_path", "a.wav"])
        audio_path, audio_url, output_gcs = inference.resolve_inputs(args)
        self.assertEqual(audio_path, "a.wav")
        self.assertIsNone(audio_url)
        self.assertIsNone(output_gcs)

    def test_audio_url_requires_output_gcs(self):
        args = inference.parse_args(["--audio-url", "https://example.com/a.wav"])
        with self.assertRaises(SystemExit):
            inference.resolve_inputs(args)

    def test_audio_url_and_gcs(self):
        args = inference.parse_args([
            "--audio-url", "https://example.com/a.wav",
            "--output-gcs", "gs://bucket/out.json",
        ])
        audio_path, audio_url, output_gcs = inference.resolve_inputs(args)
        self.assertIsNone(audio_path)
        self.assertEqual(audio_url, "https://example.com/a.wav")
        self.assertEqual(output_gcs, "gs://bucket/out.json")

    def test_env_fallback(self):
        args = inference.parse_args([])
        env = {
            "AUDIO_URL": "gs://audio/ep.wav",
            "OUTPUT_GCS": "gs://out/ep.json",
        }
        with patch.dict(os.environ, env, clear=False):
            audio_path, audio_url, output_gcs = inference.resolve_inputs(args)
        self.assertIsNone(audio_path)
        self.assertEqual(audio_url, "gs://audio/ep.wav")
        self.assertEqual(output_gcs, "gs://out/ep.json")

    def test_both_sources_rejected(self):
        args = inference.parse_args([
            "--audio_path", "a.wav",
            "--audio-url", "https://example.com/a.wav",
        ])
        with self.assertRaises(SystemExit):
            inference.resolve_inputs(args)

    def test_neither_source_rejected(self):
        args = inference.parse_args([])
        with patch.dict(os.environ, {"AUDIO_URL": "", "OUTPUT_GCS": ""}, clear=False):
            with self.assertRaises(SystemExit):
                inference.resolve_inputs(args)


class MergeEventsTests(unittest.TestCase):
    def test_overlaps_merge(self):
        events = {
            "0": {"start_sec": 1.0, "end_sec": 2.0},
            "1": {"start_sec": 1.5, "end_sec": 3.0},
        }
        merged = inference.merge_events([events])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged["0"]["start_sec"], 1.0)
        self.assertEqual(merged["0"]["end_sec"], 3.0)


class MainJobPathTests(unittest.TestCase):
    def test_url_path_uploads_to_gcs(self):
        laughter = {"0": {"start_sec": 1.0, "end_sec": 1.5}}
        with patch("inference.load_model", return_value=(object(), "cpu", 16000)), \
             patch("inference.download_audio", return_value="/tmp/a.wav") as dl, \
             patch("inference.segment_audio", return_value=laughter) as seg, \
             patch("inference.upload_json") as up:
            result = inference.main([
                "--audio-url", "https://example.com/a.wav",
                "--output-gcs", "gs://bucket/out.json",
            ])
        self.assertEqual(result, laughter)
        dl.assert_called_once()
        self.assertEqual(dl.call_args[0][0], "https://example.com/a.wav")
        seg.assert_called_once()
        up.assert_called_once_with("gs://bucket/out.json", laughter)

    def test_local_path_writes_json(self):
        import tempfile

        laughter = {"0": {"start_sec": 1.0, "end_sec": 1.5}}
        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "clip.wav")
            with open(audio, "wb") as handle:
                handle.write(b"x")
            out_dir = os.path.join(tmp, "out")
            with patch("inference.load_model", return_value=(object(), "cpu", 16000)), \
                 patch("inference.segment_audio", return_value=laughter):
                inference.main(["--audio_path", audio, "--output_dir", out_dir])
            out_file = os.path.join(out_dir, "clip.json")
            with open(out_file, encoding="utf-8") as handle:
                self.assertEqual(json.loads(handle.read()), laughter)


class FakeModel:
    """Returns logits so frames 50-80 of the first window look like laughter."""

    def __call__(self, input_values=None):
        batch = input_values.shape[0]
        frames = 349
        logits = torch.full((batch, frames), -10.0)
        logits[0, 50:80] = 10.0
        return (None, logits)


class SegmentAudioTests(unittest.TestCase):
    def test_fake_model_emits_segment(self):
        import soundfile as sf
        import tempfile

        sr = 16000
        audio = np.zeros(sr * 8, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.wav")
            sf.write(path, audio, sr)
            laughter = inference.segment_audio(path, FakeModel(), sr=sr)

        self.assertGreaterEqual(len(laughter), 1)
        first = laughter["0"]
        self.assertIn("start_sec", first)
        self.assertIn("end_sec", first)
        self.assertLess(first["start_sec"], first["end_sec"])
        # Frames 50-80 of 349 over a 7s window land around 1.0-1.6s.
        self.assertGreater(first["start_sec"], 0.5)
        self.assertLess(first["end_sec"], 2.5)


if __name__ == "__main__":
    unittest.main()
