"""Download audio and upload JSON results for Cloud Run jobs."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
HTTP_TIMEOUT_SEC = 300


class IoError(Exception):
    """Raised when audio download or GCS upload fails."""


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, blob_name) for a gs://bucket/object URI."""
    if not uri.startswith("gs://"):
        raise IoError(f"Not a GCS URI: {uri}")
    rest = uri[5:]
    if "/" not in rest:
        raise IoError(f"GCS URI must be gs://bucket/object, got: {uri}")
    bucket, blob_name = rest.split("/", 1)
    if not bucket or not blob_name or blob_name.endswith("/"):
        raise IoError(f"GCS URI must be gs://bucket/object, got: {uri}")
    return bucket, blob_name


def filename_from_url(url: str) -> str:
    if url.startswith("gs://"):
        _bucket, blob_name = parse_gcs_uri(url)
        name = os.path.basename(blob_name)
    else:
        name = os.path.basename(urlparse(url).path)
    return name or "audio.bin"


def _gcs_client():
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise IoError(
            "google-cloud-storage is required for gs:// URLs. "
            "Install it with: pip install google-cloud-storage"
        ) from exc
    return storage.Client()


def download_gcs(
    gcs_uri: str,
    dest_path: str,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    client: Optional[Any] = None,
) -> None:
    storage_client = client or _gcs_client()
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    try:
        blob.reload()
        size = getattr(blob, "size", None)
        if size is not None and size > max_bytes:
            raise IoError(f"GCS object too large: {size} bytes (max {max_bytes})")
        blob.download_to_filename(dest_path)
    except IoError:
        raise
    except Exception as exc:
        raise IoError(f"Failed to download {gcs_uri}: {exc}") from exc


def download_http(
    url: str,
    dest_path: str,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    opener: Optional[Any] = None,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IoError(f"Unsupported URL scheme: {url}")
    request = Request(url, method="GET")
    open_fn = opener or urlopen
    written = 0
    try:
        with open_fn(request, timeout=HTTP_TIMEOUT_SEC) as resp:
            length = resp.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise IoError(
                    f"Remote file too large: {length} bytes (max {max_bytes})"
                )
            with open(dest_path, "wb") as handle:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise IoError(
                            f"Remote file too large: exceeded {max_bytes} bytes"
                        )
                    handle.write(chunk)
    except IoError:
        raise
    except Exception as exc:
        raise IoError(f"Failed to download {url}: {exc}") from exc
    if written == 0:
        raise IoError(f"Downloaded empty file from {url}")


def download_audio(
    url: str,
    dest_dir: str,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    client: Optional[Any] = None,
    opener: Optional[Any] = None,
) -> str:
    """Download http(s) or gs:// audio into dest_dir. Returns the local path."""
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename_from_url(url))
    if url.startswith("gs://"):
        download_gcs(url, dest_path, max_bytes=max_bytes, client=client)
    elif urlparse(url).scheme in ("http", "https"):
        download_http(url, dest_path, max_bytes=max_bytes, opener=opener)
    else:
        raise IoError(f"Unsupported audio URL (use http(s) or gs://): {url}")
    return dest_path


def upload_json(
    gcs_uri: str,
    payload: dict,
    client: Optional[Any] = None,
) -> None:
    """Write a JSON object to gs://bucket/object."""
    storage_client = client or _gcs_client()
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    try:
        blob.upload_from_string(
            json.dumps(payload),
            content_type="application/json",
        )
    except Exception as exc:
        raise IoError(f"Failed to upload {gcs_uri}: {exc}") from exc
