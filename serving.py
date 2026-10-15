"""Shared laughter/applause inference used by Cog and the Runpod worker.

This module must not import ``cog`` or ``runpod``. Weight loading and torch
imports stay inside functions so unit tests can import the job parser alone.
"""

from __future__ import annotations

import base64
import ipaddress
import os
import re
import shutil
import socket
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

HF_REPO_ID = "omine-me/LaughterSegmentation"
HF_WEIGHTS_FILE = "model.safetensors"
AUDIO_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLE_RATE = 16000

INPUT_SEC_DEFAULT = 7.0
INPUT_SEC_MIN = 2.1
INPUT_SEC_MAX = 30.0
BATCH_SIZE_DEFAULT = 10
BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 64

_SUFFIX_RE = re.compile(r"^\.[a-z0-9]{1,5}$")
_LOCAL_HOSTS = {"localhost", "localhost.localdomain", "metadata.google.internal"}
_MIME_SUFFIX = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/x-m4a": ".m4a",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


class InputError(ValueError):
    """Job input or audio fetch that should fail the request, not the worker."""


@dataclass(frozen=True)
class JobRequest:
    """Normalized Runpod job parameters. Exactly one of ``url`` or ``data`` is set."""

    url: str | None
    data: bytes | None
    suffix: str
    input_sec: float
    batch_size: int


def local_weights_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", HF_WEIGHTS_FILE)


def ensure_weights(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    from huggingface_hub import hf_hub_download

    dest_dir = os.path.dirname(os.path.abspath(model_path))
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS_FILE)
    if os.path.abspath(cached) != os.path.abspath(model_path):
        shutil.copy2(cached, model_path)
    return model_path


def load_model(model_path: str | None = None):
    """Load the fine-tuned checkpoint once. Returns ``(model, weights_path)``."""
    import safetensors
    import torch
    from train.model import Model

    resolved = ensure_weights(model_path or local_weights_path())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # The fine-tuned checkpoint contains every parameter, so skip the base-weight download.
    model = Model(AUDIO_MODEL_NAME, device, SAMPLE_RATE, pretrained=False).to(device)
    state_dict = safetensors.torch.load_file(
        resolved,
        device.index if device.type == "cuda" else "cpu",
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, resolved


def segment_audio(
    *,
    audio_path: str,
    model,
    model_path: str,
    output_dir: str,
    input_sec: float,
    batch_size: int,
) -> dict:
    """Run inference and return the validated laughter/applause JSON object."""
    import json

    from inference import main as run_inference

    os.makedirs(output_dir, exist_ok=True)
    run_inference(
        audio_path=audio_path,
        output_dir=output_dir,
        model_path=model_path,
        input_sec=float(input_sec),
        batch_size=int(batch_size),
        model=model,
    )

    output_json = Path(output_dir) / f"{Path(audio_path).stem}.json"
    if not output_json.exists():
        raise RuntimeError("Inference completed but output JSON was not created.")

    with output_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_job_input(job_input: object) -> JobRequest:
    """Validate a Runpod ``input`` object. Does not download audio."""
    if not isinstance(job_input, dict):
        raise InputError("Job input must be an object")

    audio = job_input.get("audio")
    audio_base64 = job_input.get("audio_base64")
    has_audio = isinstance(audio, str) and audio.strip() != ""
    has_base64 = isinstance(audio_base64, str) and audio_base64.strip() != ""

    if audio is not None and not isinstance(audio, str):
        raise InputError("audio must be a string")
    if audio_base64 is not None and not isinstance(audio_base64, str):
        raise InputError("audio_base64 must be a string")
    if has_audio and has_base64:
        raise InputError("Provide either audio or audio_base64, not both")
    if not has_audio and not has_base64:
        raise InputError("audio is required")

    input_sec = _bounded_number(
        "input_sec",
        job_input.get("input_sec", INPUT_SEC_DEFAULT),
        INPUT_SEC_MIN,
        INPUT_SEC_MAX,
        as_int=False,
    )
    batch_size = _bounded_number(
        "batch_size",
        job_input.get("batch_size", BATCH_SIZE_DEFAULT),
        BATCH_SIZE_MIN,
        BATCH_SIZE_MAX,
        as_int=True,
    )

    if has_base64:
        filename = job_input.get("filename", "audio.wav")
        if not isinstance(filename, str):
            raise InputError("filename must be a string")
        data = _decode_base64(audio_base64.strip(), field="audio_base64")
        return JobRequest(
            url=None,
            data=data,
            suffix=_safe_suffix(Path(filename).suffix),
            input_sec=float(input_sec),
            batch_size=int(batch_size),
        )

    assert isinstance(audio, str)
    audio = audio.strip()
    if audio.startswith("data:"):
        data, suffix = _decode_data_uri(audio)
        return JobRequest(
            url=None,
            data=data,
            suffix=suffix,
            input_sec=float(input_sec),
            batch_size=int(batch_size),
        )

    assert_fetchable_url(audio, resolve=False)
    return JobRequest(
        url=audio,
        data=None,
        suffix=_suffix_from_url(audio),
        input_sec=float(input_sec),
        batch_size=int(batch_size),
    )


def materialize_audio(job: JobRequest, dest: Path) -> str:
    """Write job audio to ``dest`` and return that path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if job.data is not None:
        dest.write_bytes(job.data)
        return str(dest)
    if not job.url:
        raise InputError("audio is required")

    assert_fetchable_url(job.url, resolve=True)
    request = urllib.request.Request(
        job.url,
        headers={"User-Agent": "laughter-segmentation-runpod"},
    )
    opener = urllib.request.build_opener(_GuardedRedirectHandler)
    try:
        with opener.open(request, timeout=300) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Failed to download audio: {exc}") from exc

    if not dest.is_file() or dest.stat().st_size == 0:
        raise InputError("Downloaded audio is empty")
    return str(dest)


def assert_fetchable_url(url: str, *, resolve: bool) -> str:
    """Accept a public http(s) URL. Reject other schemes, local paths, and private hosts."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InputError("audio must be an http(s) URL or a base64 data URI")

    host = parsed.hostname
    if host is None or host == "":
        raise InputError("audio URL is missing a host")
    host = host.rstrip(".").lower()
    if host in _LOCAL_HOSTS or host.endswith(".local") or host.endswith(".localhost"):
        raise InputError("audio URL must not point at a local address")
    if host.isdigit():
        raise InputError("audio URL must not point at a local address")

    addresses = _host_addresses(host, resolve=resolve)
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise InputError("audio URL host did not resolve to an IP address") from exc
        if not ip.is_global:
            raise InputError("audio URL must not point at a local or private address")
    return url


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        assert_fetchable_url(newurl, resolve=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _host_addresses(host: str, *, resolve: bool) -> list[str]:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not resolve:
            return []
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise InputError(f"Failed to resolve audio URL host: {exc}") from exc
        addresses = []
        for info in infos:
            sockaddr = info[4]
            if sockaddr:
                addresses.append(sockaddr[0])
        if not addresses:
            raise InputError("audio URL host did not resolve to an IP address")
        return addresses
    return [host]


def _bounded_number(name: str, value: object, minimum: float, maximum: float, *, as_int: bool):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        kind = "an integer" if as_int else "a number"
        raise InputError(f"{name} must be {kind}")
    if as_int:
        if isinstance(value, float) and not value.is_integer():
            raise InputError(f"{name} must be an integer")
        number = int(value)
    else:
        number = float(value)
    if number < minimum or number > maximum:
        raise InputError(f"{name} must be between {minimum} and {maximum}")
    return number


def _decode_base64(payload: str, *, field: str) -> bytes:
    try:
        data = base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
    except Exception as exc:
        raise InputError(f"{field} is not valid base64") from exc
    if not data:
        raise InputError(f"{field} is empty")
    return data


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    header, sep, payload = uri.partition(",")
    if not sep or not header.startswith("data:"):
        raise InputError("audio data URI is missing a payload")
    meta = header[len("data:") :]
    parts = [part.strip().lower() for part in meta.split(";") if part.strip()]
    if "base64" not in parts:
        raise InputError("audio data URI must be base64-encoded")
    mime = parts[0] if parts and parts[0] != "base64" else ""
    suffix = _MIME_SUFFIX.get(mime, ".wav")
    return _decode_base64(payload, field="audio"), suffix


def _suffix_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    return _safe_suffix(Path(path).suffix)


def _safe_suffix(suffix: str, default: str = ".wav") -> str:
    cleaned = suffix.lower()
    if _SUFFIX_RE.match(cleaned):
        return cleaned
    return default
