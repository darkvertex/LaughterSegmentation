"""Runpod job-input contract. These tests do not import torch, cog, or runpod."""

import ast
import base64
import importlib
import socket
import sys
import types
from pathlib import Path

import pytest

from serving import (
    InputError,
    assert_fetchable_url,
    materialize_audio,
    parse_job_input,
)

ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_cog_predictor_contract_is_unchanged():
    tree = ast.parse((ROOT / "predict.py").read_text(encoding="utf-8"))
    methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Predictor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.add(item.name)
    assert {"setup", "predict"} <= methods

    source = (ROOT / "predict.py").read_text(encoding="utf-8")
    assert "default=7.0" in source
    assert "ge=2.1" in source
    assert "le=30.0" in source
    assert "default=10" in source
    assert "ge=1" in source
    assert "le=64" in source
    assert 'predict: "predict.py:Predictor"' in (ROOT / "cog.yaml").read_text(encoding="utf-8")


def test_serving_does_not_import_cog_or_runpod():
    serving_imports = _imported_modules(ROOT / "serving.py")
    assert "cog" not in serving_imports
    assert "runpod" not in serving_imports
    predict_imports = _imported_modules(ROOT / "predict.py")
    assert "cog" in predict_imports
    assert "runpod" not in predict_imports
    assert "serving" in predict_imports


def test_worker_image_starts_handler_and_skips_test_input():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["python", "-u", "handler.py"]' in dockerfile
    assert "cog.yaml" not in dockerfile.split("CMD", 1)[-1]
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "test_input.json" in dockerignore
    handler = (ROOT / "handler.py").read_text(encoding="utf-8")
    main = handler.split('if __name__ == "__main__":', 1)[1]
    assert main.index("load_model()") < main.index("serverless.start")


def test_url_job_uses_replicate_defaults_and_suffix():
    job = parse_job_input(
        {"audio": "https://cdn.example.com/shows/clip.opus?download=1"}
    )
    assert job.url == "https://cdn.example.com/shows/clip.opus?download=1"
    assert job.data is None
    assert job.suffix == ".opus"
    assert job.input_sec == 7.0
    assert job.batch_size == 10


def test_explicit_window_and_batch_bounds():
    job = parse_job_input(
        {
            "audio": "https://cdn.example.com/a.wav",
            "input_sec": 2.1,
            "batch_size": 64,
        }
    )
    assert job.input_sec == 2.1
    assert job.batch_size == 64

    integral = parse_job_input(
        {"audio": "https://cdn.example.com/a.wav", "input_sec": 7, "batch_size": 10.0}
    )
    assert integral.input_sec == 7.0
    assert integral.batch_size == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_sec", 2.09),
        ("input_sec", 30.1),
        ("batch_size", 0),
        ("batch_size", 65),
        ("batch_size", 1.5),
        ("batch_size", True),
        ("input_sec", "7"),
    ],
)
def test_rejects_out_of_range_and_mistyped_numbers(field, value):
    payload = {"audio": "https://cdn.example.com/a.wav", field: value}
    with pytest.raises(InputError):
        parse_job_input(payload)


def test_data_uri_and_audio_base64(tmp_path: Path):
    raw = b"RIFFfake-wav"
    uri = "data:audio/mpeg;base64," + base64.b64encode(raw).decode()
    from_uri = parse_job_input({"audio": uri, "input_sec": 30, "batch_size": 1})
    assert from_uri.data == raw
    assert from_uri.suffix == ".mp3"
    assert from_uri.url is None

    encoded = base64.b64encode(raw).decode()
    from_field = parse_job_input(
        {"audio_base64": encoded, "filename": "../../etc/passwd.wav"}
    )
    assert from_field.data == raw
    assert from_field.suffix == ".wav"
    written = materialize_audio(from_field, tmp_path / f"audio{from_field.suffix}")
    assert Path(written).read_bytes() == raw
    assert Path(written).parent == tmp_path


def test_rejects_missing_audio_both_sources_and_local_urls():
    with pytest.raises(InputError, match="audio is required"):
        parse_job_input({})
    with pytest.raises(InputError, match="not both"):
        parse_job_input(
            {
                "audio": "https://cdn.example.com/a.wav",
                "audio_base64": base64.b64encode(b"x").decode(),
            }
        )
    for audio in (
        "/tmp/show.wav",
        "file:///tmp/show.wav",
        "ftp://cdn.example.com/a.wav",
        "http://127.0.0.1/a.wav",
        "http://169.254.169.254/latest/meta-data",
        "https://localhost/a.wav",
        "http://[::1]/a.wav",
        "http://10.0.0.5/a.wav",
        "http://2130706433/a.wav",
    ):
        with pytest.raises(InputError):
            parse_job_input({"audio": audio})


@pytest.mark.parametrize(
    "url",
    [
        "http://127.1/a.wav",
        "http://0177.0.0.1/a.wav",
        "http://0x7f.0.0.1/a.wav",
    ],
)
def test_fetch_rejects_obscured_loopback(url):
    with pytest.raises(InputError):
        assert_fetchable_url(url, resolve=True)


def test_dns_resolution_rejects_private_addresses(monkeypatch: pytest.MonkeyPatch):
    def _private(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _private)
    with pytest.raises(InputError, match="private"):
        assert_fetchable_url("https://weights.example/a.wav", resolve=True)

    def _public(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _public)
    assert (
        assert_fetchable_url("https://cdn.example.com/a.wav", resolve=True)
        == "https://cdn.example.com/a.wav"
    )


def _load_handler():
    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(start=lambda _config: None)
    sys.modules["runpod"] = fake
    sys.modules.pop("handler", None)
    return importlib.import_module("handler")


def test_handler_returns_input_errors_without_loading_weights():
    handler_mod = _load_handler()
    missing = handler_mod.handler({"id": "job-1", "input": {}})
    assert missing == {"error": "audio is required"}

    # Sentinel only: download failures are reported after the worker is ready,
    # and this must not call load_model().
    handler_mod._MODEL = object()
    handler_mod._MODEL_PATH = "/tmp/model.safetensors"

    def _boom(*_args, **_kwargs):
        raise InputError("Failed to download audio: timed out")

    handler_mod.materialize_audio = _boom
    downloaded = handler_mod.handler(
        {"id": "job-2", "input": {"audio": "https://cdn.example.com/a.wav"}}
    )
    assert downloaded == {"error": "Failed to download audio: timed out"}


def test_handler_returns_segmentation_json(monkeypatch: pytest.MonkeyPatch):
    handler_mod = _load_handler()
    handler_mod._MODEL = object()
    handler_mod._MODEL_PATH = "/tmp/model.safetensors"
    raw = b"RIFFfake-wav"
    seen: dict = {}

    def _segment(**kwargs):
        seen.update(kwargs)
        assert Path(kwargs["audio_path"]).read_bytes() == raw
        return {"laughter": {}, "applause": {}}

    monkeypatch.setattr(handler_mod, "segment_audio", _segment)
    result = handler_mod.handler(
        {
            "id": "job/2",
            "input": {
                "audio_base64": base64.b64encode(raw).decode(),
                "filename": "clip.wav",
                "input_sec": 8,
                "batch_size": 4,
            },
        }
    )
    assert result == {"laughter": {}, "applause": {}}
    assert seen["input_sec"] == 8.0
    assert seen["batch_size"] == 4
    assert seen["model"] is handler_mod._MODEL
    assert not Path("/tmp/runpod-input/job_2").exists()
