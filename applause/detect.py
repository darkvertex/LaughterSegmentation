"""Applause detection: AMP MLP ported to NumPy.

Weights come from AudiovisualMetadataPlatform/applause-detection
pretrained/applause-binary-20210203 (class 0 = applause, class 1 = non-applause).

MFCC settings match AMP feature.py: 40 coefficients, 16 kHz, 10 ms hop.
"""

from __future__ import annotations

import os
from typing import Mapping

import librosa
import numpy as np

from evaluation._utils.utils import annotate_loudness

MFCC_SIZE = 40
FRAME_SIZE_MS = 10
SAMPLE_RATE = 16000
N_FFT = 2048
APPLAUSE_CLASS = 0
DEFAULT_MIN_SEGMENT_MS = 1000

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS_PATH = os.path.join(_ROOT, "models", "applause_mlp.npz")

SegmentMap = dict[str, dict[str, float]]


def load_mlp_weights(weights_path: str = DEFAULT_WEIGHTS_PATH) -> dict[str, np.ndarray]:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Applause MLP weights not found: {weights_path}. "
            "Run scripts/export_applause_weights.py to regenerate them."
        )
    with np.load(weights_path) as data:
        return {key: data[key] for key in data.files}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward_mlp(features: np.ndarray, weights: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return per-frame class logits/probabilities after the final Dense layer."""
    if features.ndim != 2 or features.shape[1] != MFCC_SIZE:
        raise ValueError(
            f"Expected features of shape (n_frames, {MFCC_SIZE}), got {features.shape}"
        )
    activations = [str(name) for name in np.asarray(weights["activations"]).tolist()]
    hidden = features.astype(np.float32, copy=False)
    for index, activation in enumerate(activations):
        kernel = np.asarray(weights[f"kernel_{index}"], dtype=np.float32)
        bias = np.asarray(weights[f"bias_{index}"], dtype=np.float32)
        hidden = hidden @ kernel + bias
        if activation == "sigmoid":
            hidden = _sigmoid(hidden)
        elif activation == "softmax":
            hidden = _softmax(hidden)
        else:
            raise ValueError(f"Unsupported activation {activation!r}")
    return hidden


def predict_classes(features: np.ndarray, weights: Mapping[str, np.ndarray]) -> np.ndarray:
    logits = forward_mlp(features, weights)
    return np.argmax(logits, axis=1).astype(np.int32)


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    if audio.size == 0:
        return np.empty((0, MFCC_SIZE), dtype=np.float32)
    hop_length = int(sr // (1000 / FRAME_SIZE_MS))
    mfccs = librosa.feature.mfcc(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        n_mfcc=MFCC_SIZE,
        hop_length=hop_length,
        n_fft=N_FFT,
    ).T
    return np.asarray(mfccs, dtype=np.float32)


def _merge_short_sounds(predictions: np.ndarray, threshold_ms: int) -> np.ndarray:
    """Port of AMP smoothing.merge_short_sounds.

    Short runs are rewritten to the previous long-enough label. The AMP source
    used `if not previous_value`, which treats class 0 (applause) as missing;
    this port uses `is None` so the first long segment is tracked correctly.
    """
    if threshold_ms <= 0 or len(predictions) == 0:
        return np.asarray(predictions, dtype=np.int32)

    min_n_frames = round(threshold_ms / FRAME_SIZE_MS)
    merged: list[int] = []
    previous_value: int | None = None
    current_seg_start = 0
    preds = np.asarray(predictions)

    while current_seg_start < len(preds):
        current_value = int(preds[current_seg_start])
        find_next = np.where(preds[current_seg_start:] != current_value)[0]
        if find_next.size == 0:
            next_different = len(preds)
            current_seg_len = len(preds) - current_seg_start
        else:
            current_seg_len = int(find_next[0])
            next_different = current_seg_len + current_seg_start

        if current_seg_len >= min_n_frames:
            segment = preds[current_seg_start:next_different]
            previous_value = current_value
            merged.extend(int(value) for value in segment)
        elif previous_value is None:
            # Keep frames so later timestamps stay aligned with the audio.
            # AMP omitted leading shorts from the output list; we still drop
            # them later by duration when converting to applause ranges.
            merged.extend(int(value) for value in preds[current_seg_start:next_different])
        else:
            merged.extend([previous_value] * current_seg_len)
        current_seg_start = next_different

    return np.asarray(merged, dtype=np.int32)


def _num_to_label(class_index: int, binary: bool) -> str:
    if binary:
        return "applause" if int(class_index) == APPLAUSE_CLASS else "non-applause"
    raise ValueError("Only binary applause labeling is supported")


def _group_frames(predictions: np.ndarray, binary: bool) -> list[dict[str, float | str]]:
    if len(predictions) == 0:
        return []
    results: list[dict[str, float | str]] = []
    index = 0
    preds = np.asarray(predictions)
    current = int(preds[0])
    while index < len(preds):
        next_different = np.where(preds[index:] != current)[0]
        if len(next_different) == 0:
            results.append(
                {
                    "label": _num_to_label(current, binary),
                    "start": index * FRAME_SIZE_MS / 1000,
                    "end": (len(preds) - 1) * FRAME_SIZE_MS / 1000,
                }
            )
            break
        seg_length = int(next_different[0])
        results.append(
            {
                "label": _num_to_label(current, binary),
                "start": index * FRAME_SIZE_MS / 1000,
                "end": (index + seg_length - 1) * FRAME_SIZE_MS / 1000,
            }
        )
        index += seg_length
        current = int(preds[index])
    return results


def smooth_predictions(
    predictions: np.ndarray,
    min_segment_ms: int = DEFAULT_MIN_SEGMENT_MS,
    binary: bool = True,
) -> list[dict[str, float | str]]:
    merged = _merge_short_sounds(predictions, min_segment_ms)
    return _group_frames(merged, binary)


def segments_from_grouped(
    grouped: list[dict[str, float | str]],
    min_segment_ms: int = DEFAULT_MIN_SEGMENT_MS,
) -> SegmentMap:
    min_duration_sec = min_segment_ms / 1000.0
    segments: SegmentMap = {}
    index = 0
    for row in grouped:
        if row["label"] != "applause":
            continue
        start_sec = float(row["start"])
        end_sec = float(row["end"])
        if end_sec - start_sec + FRAME_SIZE_MS / 1000.0 < min_duration_sec:
            continue
        segments[str(index)] = {
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        index += 1
    return segments


def build_timed_ranges_result(laughter: SegmentMap, applause: SegmentMap) -> dict[str, SegmentMap]:
    return {"laughter": laughter, "applause": applause}


def detect_applause(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    min_segment_ms: int = DEFAULT_MIN_SEGMENT_MS,
    annotate: bool = True,
    weights: Mapping[str, np.ndarray] | None = None,
) -> SegmentMap:
    """Return indexed applause ranges in the same shape as laughter segments."""
    loaded_weights = weights if weights is not None else load_mlp_weights(weights_path)
    audio_16k = np.asarray(audio, dtype=np.float32)
    if sr != SAMPLE_RATE:
        audio_16k = librosa.resample(audio_16k, orig_sr=sr, target_sr=SAMPLE_RATE)
    features = extract_mfcc(audio_16k, SAMPLE_RATE)
    if len(features) == 0:
        return {}
    class_ids = predict_classes(features, loaded_weights)
    grouped = smooth_predictions(class_ids, min_segment_ms=min_segment_ms, binary=True)
    segments = segments_from_grouped(grouped, min_segment_ms=min_segment_ms)
    if annotate:
        segments = annotate_loudness(segments, audio_16k, SAMPLE_RATE)
    return segments
