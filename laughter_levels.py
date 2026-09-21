"""Laughter-aware compressor, normalizer, and limiter.

Takes analysis JSON ({laughter, applause} timed ranges) plus the original
audio and levels the laughs: a subtle pre-onset lift, a gated compressor,
per-segment RMS matching, and a full-file limiter.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np

EPS = 1e-8
MERGE_GAP_SEC = 0.2
OUTPUT_FORMATS = ("wav", "flac", "mp3")

MarkersSource = Union[str, os.PathLike[str], Mapping[str, object]]
RangeList = list[tuple[float, float]]


@dataclass
class LevelsConfig:
    process_applause: bool = False
    output_format: str = "wav"
    pre_lift_db: float = 1.5
    pre_lift_ms: float = 280.0
    pre_lift_attack_ms: float = 100.0
    comp_threshold_db: float = -22.0
    comp_ratio: float = 2.5
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 200.0
    norm_target_rms_db: float = -18.0
    norm_max_gain_db: float = 6.0
    norm_max_cut_db: float = 12.0
    norm_attack_ms: float = 50.0
    norm_release_ms: float = 250.0
    limiter_ceiling_db: float = -1.0
    limiter_release_ms: float = 80.0
    merge_gap_sec: float = MERGE_GAP_SEC

    def __post_init__(self) -> None:
        fmt = str(self.output_format).lower().lstrip(".")
        if fmt not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {self.output_format!r}"
            )
        self.output_format = fmt
        if self.comp_ratio < 1.0:
            raise ValueError(f"comp_ratio must be >= 1, got {self.comp_ratio}")
        if self.pre_lift_db < 0:
            raise ValueError(f"pre_lift_db must be >= 0, got {self.pre_lift_db}")
        if self.norm_max_gain_db < 0 or self.norm_max_cut_db < 0:
            raise ValueError("norm_max_gain_db and norm_max_cut_db must be >= 0")


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amplitude_to_db(amplitude: float) -> float:
    return float(20.0 * np.log10(float(amplitude) + EPS))


def slice_rms(audio: np.ndarray, start: int, end: int) -> float:
    if end <= start or audio.size == 0:
        return 0.0
    window = audio[..., start:end]
    if window.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))


def slice_peak(audio: np.ndarray, start: int = 0, end: int | None = None) -> float:
    window = audio if end is None else audio[..., start:end]
    if window.size == 0:
        return 0.0
    return float(np.max(np.abs(window)))


def _as_2d(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"Audio must be 1-D or 2-D, got shape {array.shape}")
    return array


def _ms_to_samples(ms: float, sr: int) -> int:
    return max(1, int(round(sr * float(ms) / 1000.0)))


def _cosine_01(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.linspace(0.0, 1.0, length, endpoint=True, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _cosine_10(length: int) -> np.ndarray:
    return 1.0 - _cosine_01(length)


def _parse_json_payload(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Markers must be a JSON object, an existing JSON file, or a JSON string. "
            f"{exc}"
        ) from exc


def load_markers(source: MarkersSource) -> dict[str, object]:
    """Load markers from a dict, JSON file path, or JSON string."""
    if isinstance(source, Mapping):
        return dict(source)
    text = os.fspath(source)
    stripped = text.lstrip()
    if stripped[:1] in "{[":
        payload = _parse_json_payload(text)
    else:
        path = Path(text)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = _parse_json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("Markers JSON must be an object")
    return payload


def _is_present(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in ("", "None")


def resolve_markers_source(
    markers_path: object | None = None,
    markers_json: object | None = None,
) -> dict[str, object] | None:
    """Return parsed markers, or None if neither source is set.

    Raises if both a file and an inline JSON string are provided.
    """
    has_path = _is_present(markers_path)
    has_json = _is_present(markers_json)
    if has_path and has_json:
        raise ValueError("Provide only one of markers file or markers_json, not both")
    if not has_path and not has_json:
        return None
    if has_path:
        return load_markers(str(markers_path))
    return load_markers(str(markers_json))


def _segments_from_map(segment_map: object, kind: str) -> RangeList:
    if not isinstance(segment_map, dict):
        raise ValueError(f"{kind!r} must be an object mapping indices to segments")
    ranges: RangeList = []
    for key, segment in segment_map.items():
        if not isinstance(segment, dict):
            raise ValueError(f"{kind}[{key}] must be an object")
        if "start_sec" not in segment or "end_sec" not in segment:
            raise ValueError(f"{kind}[{key}] needs start_sec and end_sec")
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        if end <= start:
            continue
        ranges.append((start, end))
    return ranges


def parse_markers(payload: Mapping[str, object], process_applause: bool = False) -> RangeList:
    if "laughter" not in payload:
        raise ValueError("Markers JSON is missing a 'laughter' object")
    ranges = _segments_from_map(payload["laughter"], "laughter")
    if process_applause:
        applause = payload.get("applause", {})
        if applause is None:
            applause = {}
        ranges.extend(_segments_from_map(applause, "applause"))
    return ranges


def merge_ranges(ranges: Sequence[tuple[float, float]], gap_sec: float = MERGE_GAP_SEC) -> RangeList:
    if not ranges:
        return []
    ordered = sorted(((float(start), float(end)) for start, end in ranges), key=lambda item: item[0])
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start - merged[-1][1] < gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def load_audio(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    """Load audio at native sample rate as float32 (channels, samples)."""
    audio_path = os.fspath(path)
    try:
        from pedalboard.io import AudioFile

        with AudioFile(audio_path) as handle:
            audio = np.asarray(handle.read(handle.frames), dtype=np.float32)
            sr = int(handle.samplerate)
        return _as_2d(audio), sr
    except Exception:
        import soundfile as sf

        data, sr = sf.read(audio_path, always_2d=True)
        return np.asarray(data.T, dtype=np.float32), int(sr)


def write_audio(
    path: str | os.PathLike[str],
    audio: np.ndarray,
    sr: int,
    output_format: str = "wav",
) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lower().lstrip(".")
    array = _as_2d(np.asarray(audio, dtype=np.float32))
    interleaved = np.ascontiguousarray(array.T)

    if fmt == "mp3":
        from pydub import AudioSegment

        pcm = np.clip(interleaved, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        segment = AudioSegment(
            pcm_i16.tobytes(),
            frame_rate=int(sr),
            sample_width=2,
            channels=int(pcm_i16.shape[1]),
        )
        segment.export(os.fspath(output_path), format="mp3", bitrate="320k")
        return str(output_path)

    import soundfile as sf

    subtype = "PCM_24" if fmt == "wav" else "PCM_24"
    sf.write(os.fspath(output_path), interleaved, int(sr), format=fmt, subtype=subtype)
    return str(output_path)


def pre_lift_envelope(
    n_samples: int,
    sr: int,
    ranges: Sequence[tuple[float, float]],
    lift_db: float,
    lift_ms: float,
    attack_ms: float,
) -> np.ndarray:
    envelope = np.ones(n_samples, dtype=np.float64)
    if lift_db <= 0 or n_samples <= 0:
        return envelope
    gain = db_to_amplitude(lift_db)
    lift_samples = _ms_to_samples(lift_ms, sr)
    attack = _ms_to_samples(attack_ms, sr)
    for start_sec, _end_sec in ranges:
        start = int(round(start_sec * sr))
        start = min(max(0, start), n_samples)
        pre0 = max(0, start - lift_samples)
        ramp_end = min(start, pre0 + attack)
        if ramp_end > pre0:
            ramp = 1.0 + (gain - 1.0) * _cosine_01(ramp_end - pre0)
            envelope[pre0:ramp_end] = np.maximum(envelope[pre0:ramp_end], ramp)
        if start > ramp_end:
            envelope[ramp_end:start] = np.maximum(envelope[ramp_end:start], gain)
        rel_end = min(n_samples, start + attack)
        if rel_end > start:
            release = 1.0 + (gain - 1.0) * _cosine_10(rel_end - start)
            envelope[start:rel_end] = np.maximum(envelope[start:rel_end], release)
    return envelope


def laughter_mask(
    n_samples: int,
    sr: int,
    ranges: Sequence[tuple[float, float]],
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=np.float64)
    if n_samples <= 0:
        return mask
    attack = _ms_to_samples(attack_ms, sr)
    release = _ms_to_samples(release_ms, sr)
    for start_sec, end_sec in ranges:
        start = int(round(start_sec * sr))
        end = int(round(end_sec * sr))
        start = min(max(0, start), n_samples)
        end = min(max(0, end), n_samples)
        a0 = max(0, start - attack)
        r1 = min(n_samples, end + release)
        if start > a0:
            mask[a0:start] = np.maximum(mask[a0:start], _cosine_01(start - a0))
        if end > start:
            mask[start:end] = 1.0
        if r1 > end:
            mask[end:r1] = np.maximum(mask[end:r1], _cosine_10(r1 - end))
    return mask


def linked_compress(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Compress with gain reduction from max(|L|, |R|) applied to every channel."""
    from pedalboard import Compressor

    array = _as_2d(audio).astype(np.float32, copy=False)
    compressor = Compressor(
        threshold_db=float(threshold_db),
        ratio=float(ratio),
        attack_ms=float(attack_ms),
        release_ms=float(release_ms),
    )
    peak = np.max(np.abs(array), axis=0)
    mid = np.mean(array, axis=0)
    sign = np.sign(mid)
    sign[sign == 0] = 1.0
    sidechain = (peak * sign).astype(np.float32)
    compressed = np.asarray(compressor(sidechain, sr), dtype=np.float32)
    if compressed.ndim == 2:
        compressed = compressed[0]
    gain = np.ones(array.shape[1], dtype=np.float32)
    nonzero = np.abs(sidechain) > EPS
    gain[nonzero] = compressed[nonzero] / sidechain[nonzero]
    gain = np.abs(gain)
    return array * gain


def normalize_laughter(
    audio: np.ndarray,
    sr: int,
    ranges: Sequence[tuple[float, float]],
    target_rms_db: float,
    max_gain_db: float,
    max_cut_db: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    array = _as_2d(audio)
    n_samples = array.shape[1]
    envelope = np.ones(n_samples, dtype=np.float64)
    target_rms = db_to_amplitude(target_rms_db)
    attack = _ms_to_samples(attack_ms, sr)
    release = _ms_to_samples(release_ms, sr)
    for start_sec, end_sec in ranges:
        start = int(round(start_sec * sr))
        end = int(round(end_sec * sr))
        start = min(max(0, start), n_samples)
        end = min(max(0, end), n_samples)
        rms = slice_rms(array, start, end)
        if rms < EPS:
            continue
        gain = target_rms / rms
        gain_db = float(np.clip(amplitude_to_db(gain), -max_cut_db, max_gain_db))
        gain = db_to_amplitude(gain_db)
        region = np.ones(n_samples, dtype=np.float64)
        a0 = max(0, start - attack)
        r1 = min(n_samples, end + release)
        if start > a0:
            region[a0:start] = 1.0 + (gain - 1.0) * _cosine_01(start - a0)
        if end > start:
            region[start:end] = gain
        if r1 > end:
            region[end:r1] = 1.0 + (gain - 1.0) * _cosine_10(r1 - end)
        envelope *= region
    return array * envelope.astype(array.dtype, copy=False)


def apply_limiter(
    audio: np.ndarray,
    sr: int,
    ceiling_db: float,
    release_ms: float,
) -> np.ndarray:
    from pedalboard import Limiter

    array = _as_2d(audio).astype(np.float32, copy=False)
    limiter = Limiter(threshold_db=float(ceiling_db), release_ms=float(release_ms))
    limited = np.asarray(limiter(array, sr), dtype=np.float32)
    limited = _as_2d(limited)
    ceiling = db_to_amplitude(ceiling_db)
    return np.clip(limited, -ceiling, ceiling)


def process_array(
    audio: np.ndarray,
    sr: int,
    ranges: Sequence[tuple[float, float]],
    config: LevelsConfig | None = None,
) -> np.ndarray:
    """Run the four-stage chain on an in-memory (channels, samples) buffer."""
    cfg = config or LevelsConfig()
    array = _as_2d(audio).astype(np.float32, copy=True)
    n_samples = array.shape[1]
    merged = merge_ranges(ranges, cfg.merge_gap_sec)

    lift = pre_lift_envelope(
        n_samples,
        sr,
        merged,
        cfg.pre_lift_db,
        cfg.pre_lift_ms,
        cfg.pre_lift_attack_ms,
    )
    array = array * lift.astype(np.float32, copy=False)

    if merged:
        wet = linked_compress(
            array,
            sr,
            cfg.comp_threshold_db,
            cfg.comp_ratio,
            cfg.comp_attack_ms,
            cfg.comp_release_ms,
        )
        mask = laughter_mask(
            n_samples,
            sr,
            merged,
            cfg.comp_attack_ms,
            cfg.comp_release_ms,
        ).astype(np.float32, copy=False)
        array = array * (1.0 - mask) + wet * mask
        array = normalize_laughter(
            array,
            sr,
            merged,
            cfg.norm_target_rms_db,
            cfg.norm_max_gain_db,
            cfg.norm_max_cut_db,
            cfg.norm_attack_ms,
            cfg.norm_release_ms,
        )

    return apply_limiter(array, sr, cfg.limiter_ceiling_db, cfg.limiter_release_ms)


def process_laughter_levels(
    audio_path: str | os.PathLike[str],
    markers: MarkersSource,
    output_path: str | os.PathLike[str],
    config: LevelsConfig | None = None,
) -> str:
    cfg = config or LevelsConfig()
    payload = load_markers(markers)
    ranges = parse_markers(payload, process_applause=cfg.process_applause)
    audio, sr = load_audio(audio_path)
    processed = process_array(audio, sr, ranges, cfg)
    return write_audio(output_path, processed, sr, cfg.output_format)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply laughter-aware compressor, normalizer, and limiter."
    )
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--markers", required=True, help="Analysis JSON file")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--process_applause", action="store_true", default=False)
    parser.add_argument("--output_format", default="wav", choices=OUTPUT_FORMATS)
    parser.add_argument("--pre_lift_db", type=float, default=1.5)
    parser.add_argument("--pre_lift_ms", type=float, default=280.0)
    parser.add_argument("--pre_lift_attack_ms", type=float, default=100.0)
    parser.add_argument("--comp_threshold_db", type=float, default=-22.0)
    parser.add_argument("--comp_ratio", type=float, default=2.5)
    parser.add_argument("--comp_attack_ms", type=float, default=10.0)
    parser.add_argument("--comp_release_ms", type=float, default=200.0)
    parser.add_argument("--norm_target_rms_db", type=float, default=-18.0)
    parser.add_argument("--norm_max_gain_db", type=float, default=6.0)
    parser.add_argument("--norm_max_cut_db", type=float, default=12.0)
    parser.add_argument("--norm_attack_ms", type=float, default=50.0)
    parser.add_argument("--norm_release_ms", type=float, default=250.0)
    parser.add_argument("--limiter_ceiling_db", type=float, default=-1.0)
    parser.add_argument("--limiter_release_ms", type=float, default=80.0)
    return parser


def main(argv: Sequence[str] | None = None) -> str:
    args = _build_arg_parser().parse_args(argv)
    config = LevelsConfig(
        process_applause=bool(args.process_applause),
        output_format=str(args.output_format),
        pre_lift_db=float(args.pre_lift_db),
        pre_lift_ms=float(args.pre_lift_ms),
        pre_lift_attack_ms=float(args.pre_lift_attack_ms),
        comp_threshold_db=float(args.comp_threshold_db),
        comp_ratio=float(args.comp_ratio),
        comp_attack_ms=float(args.comp_attack_ms),
        comp_release_ms=float(args.comp_release_ms),
        norm_target_rms_db=float(args.norm_target_rms_db),
        norm_max_gain_db=float(args.norm_max_gain_db),
        norm_max_cut_db=float(args.norm_max_cut_db),
        norm_attack_ms=float(args.norm_attack_ms),
        norm_release_ms=float(args.norm_release_ms),
        limiter_ceiling_db=float(args.limiter_ceiling_db),
        limiter_release_ms=float(args.limiter_release_ms),
    )
    return process_laughter_levels(args.audio, args.markers, args.output, config)


if __name__ == "__main__":
    print(main())
