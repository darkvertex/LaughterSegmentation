import json

import numpy as np

from applause.detect import build_timed_ranges_result, detect_applause


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
    assert set(payload.keys()) == {"laughter", "applause"}
    assert payload["laughter"]["0"]["start_sec"] == 1.2
    assert isinstance(payload["applause"], dict)
