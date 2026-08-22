import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evaluation._utils.utils import concat_close, remove_short
from inference import ensure_model_weights, merge_events, segment_laughter, write_segments


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.safetensors"
COG_YAML = ROOT / "cog.yaml"


class MergeEventsTest(unittest.TestCase):
    def test_merges_overlapping_intervals(self):
        events = {
            "0": {"start_sec": 1.0, "end_sec": 2.0},
            "1": {"start_sec": 1.5, "end_sec": 3.0},
        }
        merged = merge_events([events])
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged["0"]["start_sec"], 1.0)
        self.assertAlmostEqual(merged["0"]["end_sec"], 3.0)

    def test_keeps_disjoint_intervals_sorted(self):
        events = {
            "0": {"start_sec": 4.0, "end_sec": 5.0},
            "1": {"start_sec": 0.2, "end_sec": 0.8},
        }
        merged = merge_events([events])
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged["0"]["start_sec"], 0.2)
        self.assertAlmostEqual(merged["1"]["start_sec"], 4.0)

    def test_concat_close_and_remove_short(self):
        events = {
            "0": {"start_sec": 1.0, "end_sec": 1.05},
            "1": {"start_sec": 1.15, "end_sec": 1.5},
            "2": {"start_sec": 3.0, "end_sec": 3.5},
        }
        cleaned = remove_short(concat_close(events, 0.2), 0.2)
        self.assertEqual(len(cleaned), 2)
        self.assertAlmostEqual(cleaned["0"]["start_sec"], 1.0)
        self.assertAlmostEqual(cleaned["0"]["end_sec"], 1.5)


class CogConfigTest(unittest.TestCase):
    def test_cog_yaml_uses_python_requirements(self):
        text = COG_YAML.read_text(encoding="utf-8")
        self.assertIn('predict: "predict.py:Predictor"', text)
        self.assertIn("gpu: true", text)
        self.assertIn('python_version: "3.11"', text)
        self.assertIn("python_requirements: replicate-requirements.txt", text)
        self.assertNotIn("python_packages:", text)
        self.assertIn("- ffmpeg", text)
        self.assertIn("image: \"r8.im/", text)

    def test_predict_module_parses(self):
        source = (ROOT / "predict.py").read_text(encoding="utf-8")
        compile(source, "predict.py", "exec")

    def test_github_action_lives_under_dot_github(self):
        workflow = ROOT / ".github" / "workflows" / "push.yaml"
        self.assertTrue(workflow.is_file())
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("cog push", text)
        self.assertIn("replicate/setup-cog@v2", text)


class InferenceValidationTest(unittest.TestCase):
    def test_input_sec_must_exceed_overlap(self):
        with self.assertRaises(ValueError):
            segment_laughter(model=None, audio_path="unused.wav", input_sec=2.0, batch_size=1)


class WeightsHelperTest(unittest.TestCase):
    def test_uses_existing_checkpoint_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            with open(path, "wb") as f:
                f.write(b"checkpoint")
            with patch("inference.hf_hub_download") as download:
                result = ensure_model_weights(path)
            self.assertEqual(result, path)
            download.assert_not_called()


class WriteSegmentsTest(unittest.TestCase):
    def test_writes_json_dict(self):
        laughter = {"0": {"start_sec": 0.5, "end_sec": 1.2}}
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "clip.json")
            write_segments(laughter, out_path)
            with open(out_path, encoding="utf-8") as f:
                loaded = json.load(f)
        self.assertEqual(loaded["0"]["start_sec"], 0.5)


@unittest.skipUnless(MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0, "model.safetensors is not available")
class InferenceSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inference import load_model

        cls.model = load_model(str(MODEL_PATH))

    def test_short_wav_returns_segment_dict(self):
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(16000).astype(np.float32) * 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "noise.wav")
            sf.write(wav_path, audio, 16000)
            laughter = segment_laughter(
                self.model,
                wav_path,
                input_sec=3.0,
                batch_size=1,
            )
        self.assertIsInstance(laughter, dict)
        for item in laughter.values():
            self.assertIn("start_sec", item)
            self.assertIn("end_sec", item)
            self.assertLessEqual(item["start_sec"], item["end_sec"])


if __name__ == "__main__":
    unittest.main()
