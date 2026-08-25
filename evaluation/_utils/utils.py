import numpy as np

EPS = 1e-10
FLOOR_DB = -80.0


def _amplitude_to_db(amplitude):
    return max(FLOOR_DB, float(20.0 * np.log10(float(amplitude) + EPS)))


def file_rms_db(audio):
    if audio is None or len(audio) == 0:
        return FLOOR_DB
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return _amplitude_to_db(rms)


def annotate_loudness(segments, audio, sr):
    file_db = file_rms_db(audio)
    n_samples = 0 if audio is None else len(audio)
    for segment in segments.values():
        start = int(segment["start_sec"] * sr)
        end = int(segment["end_sec"] * sr)
        start = max(0, min(start, n_samples))
        end = max(0, min(end, n_samples))
        slice_ = audio[start:end] if n_samples else np.array([])
        if len(slice_) == 0:
            rms_db = FLOOR_DB
            peak_db = FLOOR_DB
        else:
            rms = float(np.sqrt(np.mean(np.square(slice_, dtype=np.float64))))
            peak = float(np.max(np.abs(slice_)))
            rms_db = _amplitude_to_db(rms)
            peak_db = _amplitude_to_db(peak)
        segment["rms_db"] = round(rms_db, 2)
        segment["peak_db"] = round(peak_db, 2)
        segment["crest_db"] = round(peak_db - rms_db, 2)
        segment["rel_rms_db"] = round(rms_db - file_db, 2)
    return segments


def concat_close(laughters, gap_threshold):
    # concat laughters which are close to each other
    laughters_concat = []
    for laughter in laughters.values():
        if len(laughters_concat) == 0:
            laughters_concat.append(laughter.copy())
            continue
        if abs(laughters_concat[-1]["end_sec"] - laughter["start_sec"]) < gap_threshold:
            laughters_concat[-1]["end_sec"] = laughter["end_sec"]
        else:
            laughters_concat.append(laughter.copy())
    # to dict
    return {str(i): laughter for i, laughter in enumerate(laughters_concat)}

def remove_short(laughters, min_length):
    # remove short laughters
    laughters_concat = []
    for laughter in laughters.values():
        if laughter["end_sec"] - laughter["start_sec"] < min_length:
            continue
        laughters_concat.append(laughter.copy())
    # to dict
    return {str(i): laughter for i, laughter in enumerate(laughters_concat)}

def remove_inappropriate(laughters):
    # remove inappropriate laughters
    laughters_concat = []
    for laughter in laughters.values():
        if "not_a_laugh" in laughter and laughter["not_a_laugh"]:
            continue
        if laughter["end_sec"] - laughter["start_sec"] <= 0.0:
            continue
        laughters_concat.append(laughter.copy())
    # to dict
    return {str(i): laughter for i, laughter in enumerate(laughters_concat)}
