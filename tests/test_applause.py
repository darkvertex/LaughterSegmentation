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


def test_smooth_joins_fragmented_exact_men_clapping_sample_pattern():
    # Exact run-length encoding of the classifier output for men_clapping.mp3
    # (http://tmpfiles.comedyonmackay.com/shared/men_clapping.mp3): 1,247
    # 10 ms frames, 391 applause frames in 38 runs, and no applause run longer
    # than 340 ms. Keeping the encoded output avoids a network/audio fixture.
    run_lengths = np.fromstring(
        "118,14,6,2,4,1,114,14,92,7,10,3,20,9,8,1,7,3,31,8,"
        "9,8,68,34,4,2,30,27,7,7,34,33,39,2,6,24,6,7,22,2,"
        "9,28,4,7,3,8,16,7,4,9,10,1,20,8,23,14,31,1,14,1,"
        "3,21,4,18,5,12,28,2,2,9,8,2,26,33,5,2,6",
        dtype=np.int32,
        sep=",",
    )
    runs = [
        (1 - index % 2, int(length))
        for index, length in enumerate(run_lengths)
    ]
    predictions = np.concatenate(
        [np.full(length, label, dtype=np.int32) for label, length in runs]
    )

    assert len(runs) == 77
    assert len(predictions) == 1247
    assert np.count_nonzero(predictions == 0) == 391
    assert sum(label == 0 for label, _ in runs) == 38
    assert max(length for label, length in runs if label == 0) == 34

    grouped = smooth_predictions(predictions, min_segment_ms=1000, binary=True)
    applause_rows = [row for row in grouped if row["label"] == "applause"]

    assert applause_rows == [
        {
            "label": "applause",
            "start": pytest.approx(5.57),
            "end": pytest.approx(12.4),
        }
    ]


def test_smooth_does_not_promote_sparse_applause_hits():
    # Synthetic false-positive guard: bridging spans over one second, but only
    # 320 ms of the span is direct applause evidence.
    predictions = np.concatenate(
        [
            np.ones(73, dtype=np.int32),
            np.zeros(3, dtype=np.int32),
            np.ones(21, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.ones(3, dtype=np.int32),
            np.zeros(3, dtype=np.int32),
            np.ones(26, dtype=np.int32),
            np.zeros(3, dtype=np.int32),
            np.ones(30, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.ones(13, dtype=np.int32),
            np.zeros(18, dtype=np.int32),
            np.ones(11, dtype=np.int32),
            np.zeros(3, dtype=np.int32),
            np.ones(92, dtype=np.int32),
        ]
    )

    grouped = smooth_predictions(predictions, min_segment_ms=1000, binary=True)

    assert all(row["label"] == "non-applause" for row in grouped)


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
