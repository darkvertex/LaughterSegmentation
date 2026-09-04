import numpy as np
import pytest

from applause.detect import (
    MFCC_SIZE,
    build_timed_ranges_result,
    detect_applause,
    extract_mfcc,
    forward_mlp,
    load_mlp_weights,
    predict_classes,
    smooth_predictions,
)


LOUDNESS_KEYS = {"rms_db", "peak_db", "crest_db", "rel_rms_db"}


def test_build_timed_ranges_result_nests_both_maps():
    laughter = {
        "0": {
            "start_sec": 1.2,
            "end_sec": 2.0,
            "rms_db": -21.4,
            "peak_db": -9.1,
            "crest_db": 12.3,
            "rel_rms_db": -4.0,
        }
    }
    applause = {}
    result = build_timed_ranges_result(laughter, applause)
    assert set(result.keys()) == {"laughter", "applause"}
    assert result["laughter"] is laughter
    assert result["applause"] == {}


def test_mlp_forward_shape_and_class_range():
    weights = load_mlp_weights()
    frames = np.zeros((8, MFCC_SIZE), dtype=np.float32)
    probs = forward_mlp(frames, weights)
    assert probs.shape == (8, 2)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    classes = predict_classes(frames, weights)
    assert classes.shape == (8,)
    assert set(np.unique(classes)).issubset({0, 1})


def test_smooth_predictions_emits_applause_range():
    # 10 ms frames: 50 non-applause (leading short, kept), 150 applause (1.5s),
    # 120 non-applause (>= 1s so it is not merged into applause).
    predictions = np.concatenate(
        [
            np.ones(50, dtype=np.int32),
            np.zeros(150, dtype=np.int32),
            np.ones(120, dtype=np.int32),
        ]
    )
    grouped = smooth_predictions(predictions, min_segment_ms=1000, binary=True)
    applause_rows = [row for row in grouped if row["label"] == "applause"]
    assert len(applause_rows) == 1
    assert applause_rows[0]["start"] == pytest.approx(0.5)
    assert applause_rows[0]["end"] == pytest.approx(1.99)


def test_smooth_merges_short_gap_after_applause():
    predictions = np.concatenate(
        [
            np.zeros(120, dtype=np.int32),
            np.ones(20, dtype=np.int32),
            np.zeros(120, dtype=np.int32),
        ]
    )
    grouped = smooth_predictions(predictions, min_segment_ms=1000, binary=True)
    applause_rows = [row for row in grouped if row["label"] == "applause"]
    assert len(applause_rows) == 1
    assert applause_rows[0]["start"] == pytest.approx(0.0)
    assert applause_rows[0]["end"] == pytest.approx(2.59)


def test_smooth_drops_short_applause_after_long_non_applause():
    predictions = np.concatenate(
        [
            np.ones(120, dtype=np.int32),
            np.zeros(20, dtype=np.int32),
            np.ones(120, dtype=np.int32),
        ]
    )
    grouped = smooth_predictions(predictions, min_segment_ms=1000, binary=True)
    assert all(row["label"] == "non-applause" for row in grouped)


def test_detect_applause_on_silence_returns_object_map():
    sr = 16000
    silence = np.zeros(sr * 3, dtype=np.float32)
    segments = detect_applause(silence, sr=sr, min_segment_ms=1000)
    assert isinstance(segments, dict)
    for key, segment in segments.items():
        assert key.isdigit()
        assert {"start_sec", "end_sec"} <= set(segment)
        assert LOUDNESS_KEYS <= set(segment)


def test_extract_mfcc_frame_count():
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    mfccs = extract_mfcc(audio, sr)
    assert mfccs.ndim == 2
    assert mfccs.shape[1] == MFCC_SIZE
    assert mfccs.shape[0] > 0


def test_detect_applause_annotates_loudness_for_forced_positive(monkeypatch):
    sr = 16000
    audio = np.full(sr * 3, 0.1, dtype=np.float32)
    monkeypatch.setattr(
        "applause.detect.predict_classes",
        lambda features, weights: np.zeros(len(features), dtype=np.int32),
    )
    segments = detect_applause(audio, sr=sr, min_segment_ms=1000)
    assert set(segments.keys()) == {"0"}
    assert segments["0"]["start_sec"] == pytest.approx(0.0)
    assert LOUDNESS_KEYS <= set(segments["0"])
    assert segments["0"]["rms_db"] > -80.0
