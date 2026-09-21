import json
from pathlib import Path

import numpy as np
import pytest

from laughter_levels import (
    LevelsConfig,
    amplitude_to_db,
    db_to_amplitude,
    laughter_mask,
    load_audio,
    merge_ranges,
    parse_markers,
    pre_lift_envelope,
    process_array,
    process_laughter_levels,
    resolve_markers_source,
    slice_peak,
    slice_rms,
)


SR = 48000


def _tone(freq: float, start: float, end: float, amp: float, n: int, sr: int = SR) -> np.ndarray:
    signal = np.zeros(n, dtype=np.float64)
    i0 = int(start * sr)
    i1 = int(end * sr)
    t = np.arange(i1 - i0) / sr
    signal[i0:i1] = amp * np.sin(2 * np.pi * freq * t)
    return signal


def synthetic_show(sr: int = SR, seconds: float = 5.0) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Stereo bed + quiet laugh + hot laugh."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    speech = 0.03 * np.sin(2 * np.pi * 180.0 * t)
    quiet = _tone(400.0, 1.0, 1.5, 0.08, n, sr)
    hot = _tone(400.0, 3.0, 3.6, 0.99, n, sr)
    mono = speech + quiet + hot
    audio = np.vstack([mono, mono]).astype(np.float32)
    ranges = [(1.0, 1.5), (3.0, 3.6)]
    return audio, ranges


MARKERS = {
    "laughter": {
        "0": {
            "start_sec": 1.0,
            "end_sec": 1.5,
            "rms_db": -25.0,
            "peak_db": -22.0,
            "crest_db": 3.0,
            "rel_rms_db": -5.0,
        },
        "1": {
            "start_sec": 3.0,
            "end_sec": 3.6,
            "rms_db": -3.0,
            "peak_db": -0.1,
            "crest_db": 2.9,
            "rel_rms_db": 17.0,
        },
    },
    "applause": {
        "0": {"start_sec": 4.2, "end_sec": 4.8},
    },
}


def test_parse_markers_reads_start_end_and_ignores_baked_loudness():
    ranges = parse_markers(MARKERS)
    assert ranges == [(1.0, 1.5), (3.0, 3.6)]


def test_parse_markers_can_include_applause():
    ranges = parse_markers(MARKERS, process_applause=True)
    assert (4.2, 4.8) in ranges
    assert len(ranges) == 3


def test_missing_laughter_key_fails_clearly():
    with pytest.raises(ValueError, match="missing a 'laughter' object"):
        parse_markers({"applause": {}})


def test_laughter_must_be_an_object():
    with pytest.raises(ValueError, match="must be an object"):
        parse_markers({"laughter": [{"start_sec": 1.0, "end_sec": 2.0}]})


def test_segment_missing_bounds_fails_clearly():
    with pytest.raises(ValueError, match="needs start_sec and end_sec"):
        parse_markers({"laughter": {"0": {"start_sec": 1.0}}})


def test_bad_json_string_fails_clearly():
    with pytest.raises(ValueError, match="JSON"):
        resolve_markers_source(markers_json="not-json")


def test_reject_both_marker_sources():
    with pytest.raises(ValueError, match="only one"):
        resolve_markers_source(markers_path="markers.json", markers_json="{}")


def test_neither_marker_source_returns_none():
    assert resolve_markers_source(None, "") is None
    assert resolve_markers_source(None, None) is None


def test_merge_ranges_joins_close_laughs():
    merged = merge_ranges([(1.0, 1.2), (1.3, 1.5), (3.0, 3.4)], gap_sec=0.2)
    assert merged == [(1.0, 1.5), (3.0, 3.4)]


def test_pre_lift_raises_rms_just_before_onset():
    audio, ranges = synthetic_show()
    sr = SR
    n = audio.shape[1]
    env = pre_lift_envelope(n, sr, ranges, lift_db=1.5, lift_ms=280.0, attack_ms=100.0)
    lifted = audio * env.astype(np.float32)
    # Hold region is roughly 180 ms to 0 ms before onset.
    start = int(1.0 * sr)
    pre0 = start - int(0.12 * sr)
    pre1 = start - int(0.02 * sr)
    before = slice_rms(audio, pre0, pre1)
    after = slice_rms(lifted, pre0, pre1)
    expected = db_to_amplitude(1.5)
    assert after > before
    assert after / before == pytest.approx(expected, rel=0.05)


def test_full_chain_lifts_pre_window_and_levels_laughs(tmp_path: Path):
    audio, ranges = synthetic_show()
    sr = SR
    quiet = (int(1.0 * sr), int(1.5 * sr))
    hot = (int(3.0 * sr), int(3.6 * sr))
    pre0 = int(0.88 * sr)
    pre1 = int(0.98 * sr)

    before_pre = slice_rms(audio, pre0, pre1)
    before_quiet = slice_rms(audio, *quiet)
    before_hot = slice_rms(audio, *hot)
    before_spread = amplitude_to_db(before_hot) - amplitude_to_db(before_quiet)

    processed = process_array(audio, sr, ranges, LevelsConfig())

    after_pre = slice_rms(processed, pre0, pre1)
    after_quiet = slice_rms(processed, *quiet)
    after_hot = slice_rms(processed, *hot)
    after_spread = amplitude_to_db(after_hot) - amplitude_to_db(after_quiet)

    assert after_pre > before_pre
    assert after_spread < before_spread
    ceiling = db_to_amplitude(-1.0)
    assert slice_peak(processed) <= ceiling + 1e-6

    wav_path = tmp_path / "show.wav"
    markers_path = tmp_path / "markers.json"
    out_path = tmp_path / "show_leveled.wav"
    import soundfile as sf

    sf.write(wav_path, audio.T, sr)
    markers_path.write_text(json.dumps(MARKERS), encoding="utf-8")
    process_laughter_levels(wav_path, markers_path, out_path, LevelsConfig())
    loaded, loaded_sr = load_audio(out_path)
    assert loaded_sr == sr
    assert loaded.shape[0] == 2
    assert slice_peak(loaded) <= ceiling + 1e-5


def test_empty_laughter_still_limits_peaks():
    sr = SR
    n = sr
    t = np.arange(n) / sr
    hot = (0.99 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)
    audio = np.vstack([hot, hot])
    processed = process_array(audio, sr, [], LevelsConfig())
    assert slice_peak(processed) <= db_to_amplitude(-1.0) + 1e-6


def test_inline_json_string_roundtrip(tmp_path: Path):
    audio, _ranges = synthetic_show()
    wav_path = tmp_path / "in.wav"
    out_path = tmp_path / "out.wav"
    import soundfile as sf

    sf.write(wav_path, audio.T, SR)
    process_laughter_levels(wav_path, json.dumps(MARKERS), out_path, LevelsConfig())
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_laughter_mask_is_one_inside_range():
    sr = 1000
    mask = laughter_mask(5000, sr, [(1.0, 2.0)], attack_ms=10.0, release_ms=20.0)
    assert mask[1500] == pytest.approx(1.0)
    assert mask[0] == pytest.approx(0.0)
    assert mask[4000] == pytest.approx(0.0)


def test_invalid_output_format():
    with pytest.raises(ValueError, match="output_format"):
        LevelsConfig(output_format="ogg")
