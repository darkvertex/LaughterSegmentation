import json

import numpy as np
import pytest

from applause.detect import build_timed_ranges_result, detect_applause


def assert_smoke_payload(result):
    """Same contract as the Cog CI 'Validate output' step."""
    assert isinstance(result, dict)
    assert set(result.keys()) == {"laughter", "applause"}
    required = {"start_sec", "end_sec"}
    for kind in ("laughter", "applause"):
        segments = result[kind]
        assert isinstance(segments, dict)
        for key, seg in segments.items():
            assert isinstance(seg, dict)
            assert required <= seg.keys(), {kind: key, "seg": seg}


def test_json_payload_has_laughter_and_applause_keys():
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
    applause = detect_applause(
        np.zeros(16000, dtype=np.float32),
        sr=16000,
        min_segment_ms=1000,
    )
    payload = json.loads(json.dumps(build_timed_ranges_result(laughter, applause)))
    assert_smoke_payload(payload)
    assert payload["laughter"]["0"]["start_sec"] == 1.2
    assert isinstance(payload["applause"], dict)


def test_smoke_payload_allows_empty_event_maps():
    payload = {"laughter": {}, "applause": {}}
    assert_smoke_payload(payload)


def test_legacy_flat_segments_fail_smoke_payload():
    with pytest.raises(AssertionError):
        assert_smoke_payload(
            {"0": {"start_sec": 1.0, "end_sec": 2.0}}
        )
