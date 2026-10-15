"""Runpod Serverless queue worker for laughter and applause segmentation.

The Cog predictor in ``predict.py`` is unchanged. This process loads the model
once, then serves queue jobs until Runpod stops the worker.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import runpod

from serving import InputError, load_model, materialize_audio, parse_job_input, segment_audio

_MODEL = None
_MODEL_PATH: str | None = None
_INPUT_ROOT = Path("/tmp/runpod-input")
_OUTPUT_ROOT = Path("/tmp/laughter-output")


def handler(job: dict):
    """Segment one audio job.

    Success returns the same ``{"laughter", "applause"}`` object Cog writes to
    a file. Invalid input and download failures return ``{"error": ...}``.
    """
    if not isinstance(job, dict):
        return {"error": "Job must be an object"}

    job_id = _safe_job_id(job.get("id"))
    work = _INPUT_ROOT / job_id
    output_dir = _OUTPUT_ROOT / job_id
    try:
        parsed = parse_job_input(job.get("input"))
        if _MODEL is None or _MODEL_PATH is None:
            raise RuntimeError("Model is not loaded")
        shutil.rmtree(work, ignore_errors=True)
        audio_path = materialize_audio(parsed, work / f"audio{parsed.suffix}")
        print(
            f"Segmenting job {job_id} input_sec={parsed.input_sec} batch_size={parsed.batch_size}",
            flush=True,
        )
        return segment_audio(
            audio_path=audio_path,
            model=_MODEL,
            model_path=_MODEL_PATH,
            output_dir=str(output_dir),
            input_sec=parsed.input_sec,
            batch_size=parsed.batch_size,
        )
    except InputError as exc:
        return {"error": str(exc)}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def _safe_job_id(raw: object) -> str:
    text = raw if isinstance(raw, str) else "job"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", text).strip("._")[:80]
    return cleaned or "job"


if __name__ == "__main__":
    # Load before the worker accepts jobs so a cold start finishes in init.
    _MODEL, _MODEL_PATH = load_model()
    runpod.serverless.start({"handler": handler})
