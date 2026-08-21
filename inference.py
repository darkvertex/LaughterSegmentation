import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tempfile

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
import safetensors
from scipy import signal
import torch
from transformers.trainer_utils import set_seed

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/train/")
from evaluation._utils.utils import concat_close, remove_short
from train.model import Model

from io_utils import IoError, download_audio, upload_json

AUDIO_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
DEFAULT_SR = 16000
DEFAULT_SEED = 42


def merge_events(event_lists):
    merged_events = {}
    merged_event_idx = 0
    has_merged = False
    for event_list in event_lists:
        for event in event_list.values():
            if not merged_events:
                # If merged_events is empty, add the first event
                merged_events[str(merged_event_idx)] = event.copy()
                merged_event_idx += 1
            else:
                merged = False
                for merged_event in merged_events.values():
                    if event["start_sec"] <= merged_event["end_sec"] and event["end_sec"] >= merged_event["start_sec"]:
                        # Events overlap, merge them
                        merged_event["start_sec"] = min(event["start_sec"], merged_event["start_sec"])
                        merged_event["end_sec"] = max(event["end_sec"], merged_event["end_sec"])
                        merged = True
                        has_merged = True
                        # break
                if not merged:
                    # If the event does not overlap with any merged event, add it to merged_events
                    merged_events[str(merged_event_idx)] = event.copy()
                    merged_event_idx += 1
    if has_merged:
        merged_events = merge_events([merged_events])
    merged_events = sorted(merged_events.values(), key=lambda x: x["start_sec"])
    merged_events = {str(idx): val for idx, val in enumerate(merged_events)}
    return merged_events

# bandpass
def bandpass(x, samplerate, fp=np.array([1000,3000]), fs=np.array([1000,3000]), gpass=3, gstop=40):
    fn = samplerate / 2 # nyquist frequency
    wp = fp / fn  # normalizing the passband frequency by the Nyquist frequency
    ws = fs / fn  # normalizing the stopband frequency by the Nyquist frequency
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # calculate the order and normalized frequency of the Butterworth
    b, a = signal.butter(N, Wn, "band") # calculate the numerator and denominator of the filter transfer function
    y = signal.filtfilt(b, a, x) # filter the signal
    return y

def custom_amplituder_small_portion(array, sr, mul_fac=5):
    # 32767 is max value of signed short
    dub_audio = AudioSegment(
                (array*32767).astype("int16").tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
                )

    dub_audio = dub_audio.set_frame_rate(sr)
    silent_section = detect_silence(dub_audio, min_silence_len=270, silence_thresh=-35)

    sr_mul = sr // 1000
    for sec in silent_section:
        fade_len = int(sr*.15) # 0.15 sec
        if (sec[1]-sec[0])*sr_mul > (fade_len*2):
            array[sec[0]*sr_mul: sec[0]*sr_mul + fade_len] *= np.linspace(1, mul_fac, fade_len)
            array[sec[0]*sr_mul + fade_len: sec[1]*sr_mul - fade_len] *= mul_fac
            if sec[1]*sr_mul < len(array):
                array[sec[1]*sr_mul - fade_len: sec[1]*sr_mul] *= np.linspace(mul_fac, 1, fade_len)
        else:
            array[sec[0]*sr_mul: sec[1]*sr_mul] *= mul_fac
    array = librosa.util.normalize(array)
    return array


def load_model(model_path, device=None):
    """Load the laughter segmentation model once. Returns (model, device, sr)."""
    sr = DEFAULT_SR
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(DEFAULT_SEED)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Model(AUDIO_MODEL_NAME, device, sr).to(device)
    # Training-era helper that randomly kills subprocesses; never run during jobs.
    model.kill_subprocess_randomly = lambda: None

    if not osp.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Download the model file and place it in the specified path."
        )
    state_dict = safetensors.torch.load_file(
        model_path, device.index if device.type == "cuda" else "cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, device, sr


def segment_audio(audio_path, model, sr=DEFAULT_SR, input_sec=7, batch_size=10):
    """Run segmentation and return the laughter dict written by the original CLI."""
    over_lap_sec = 2.0
    assert input_sec > over_lap_sec

    laughter = {}
    laughter_idx = 0

    with torch.no_grad():
        audio_array = librosa.load(audio_path, sr=sr, mono=True)[0]
        audio_array = custom_amplituder_small_portion(audio_array, sr)

        for array_idx in range(0, len(audio_array), int(sr * (input_sec - over_lap_sec)) * batch_size):
            batched_arrays = []
            should_break = False
            for batch_idx in range(batch_size):
                array = audio_array[
                    array_idx + batch_idx * int(sr * (input_sec - over_lap_sec)):
                    array_idx + batch_idx * int(sr * (input_sec - over_lap_sec)) + sr * input_sec
                ]
                if len(array) < sr * input_sec:
                    array = np.append(array, np.zeros(sr * input_sec - len(array)))
                    should_break = True
                batched_arrays.append(array)
                if should_break:
                    break

            input_values = torch.from_numpy(np.array(batched_arrays)).type(torch.FloatTensor)
            outputs = model(input_values=input_values)

            logits = outputs[1]
            preds = torch.sigmoid(logits.to(torch.float32))

            for batch_idx, pred in enumerate(preds):
                frame_pred = list(map(round, pred.cpu().tolist(), [3] * len(pred)))
                frame_pred = (np.array(frame_pred) >= 0.5).astype(int)

                batch_start_sec = (
                    array_idx + batch_idx * int(sr * (input_sec - over_lap_sec))
                ) / float(sr)
                frame_count = len(frame_pred)
                start_idx = None
                end_idx = None
                status = "not_laughing"
                for idx, frame in enumerate(frame_pred):
                    if frame == 1:
                        if status == "not_laughing":
                            start_idx = idx
                            status = "laughing"

                        if status == "laughing" and idx == frame_count - 1:
                            laughter[str(laughter_idx)] = {
                                "start_sec": batch_start_sec + (input_sec / frame_count) * start_idx,
                                "end_sec": batch_start_sec + input_sec,
                            }
                            laughter_idx += 1
                            start_idx = None
                            end_idx = None
                    elif frame == 0:
                        if status == "laughing":
                            end_idx = idx
                            status = "not_laughing"
                            laughter[str(laughter_idx)] = {
                                "start_sec": batch_start_sec + (input_sec / frame_count) * start_idx,
                                "end_sec": batch_start_sec + (input_sec / frame_count) * end_idx,
                            }
                            laughter_idx += 1
                            start_idx = None
                            end_idx = None

        if over_lap_sec > 0.0:
            laughter = merge_events([laughter])

    laughter = concat_close(laughter, 0.2)
    laughter = remove_short(laughter, 0.2)
    return laughter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment laughter in an audio file (local path or remote URL)."
    )
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Local audio file path.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for local JSON output (used with --audio_path).")
    parser.add_argument("--audio-url", dest="audio_url", type=str, default=None,
                        help="http(s) or gs:// URL of the sound file. "
                             "Can also be set via AUDIO_URL.")
    parser.add_argument("--output-gcs", dest="output_gcs", type=str, default=None,
                        help="gs://bucket/object.json destination. "
                             "Required with --audio-url. Can also be set via OUTPUT_GCS.")
    parser.add_argument("--model_path", type=str, default="./models/model.safetensors")
    return parser.parse_args(argv)


def resolve_inputs(args):
    audio_path = args.audio_path
    audio_url = args.audio_url or os.environ.get("AUDIO_URL") or None
    output_gcs = args.output_gcs or os.environ.get("OUTPUT_GCS") or None
    if bool(audio_path) == bool(audio_url):
        raise SystemExit("Provide exactly one of --audio_path or --audio-url (or AUDIO_URL).")
    if audio_url and not output_gcs:
        raise SystemExit("--output-gcs (or OUTPUT_GCS) is required with --audio-url.")
    return audio_path, audio_url, output_gcs


def write_local_json(laughter, audio_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    basename = osp.splitext(osp.basename(audio_path))[0]
    out_file = osp.join(output_dir, basename + ".json")
    with open(out_file, mode="w", encoding="utf-8") as handle:
        json.dump(laughter, handle)
    return out_file


def main(argv=None):
    args = parse_args(argv)
    audio_path, audio_url, output_gcs = resolve_inputs(args)
    model, _device, sr = load_model(args.model_path)

    tmp_dir = None
    try:
        if audio_url:
            tmp_dir = tempfile.mkdtemp(prefix="laughterseg-")
            try:
                audio_path = download_audio(audio_url, tmp_dir)
            except IoError as exc:
                raise SystemExit(str(exc)) from exc

        laughter = segment_audio(audio_path, model, sr=sr)

        if output_gcs:
            try:
                upload_json(output_gcs, laughter)
            except IoError as exc:
                raise SystemExit(str(exc)) from exc
            print(f"Wrote {len(laughter)} segments to {output_gcs}")
        else:
            out_file = write_local_json(laughter, audio_path, args.output_dir)
            print(f"Wrote {len(laughter)} segments to {out_file}")
        return laughter
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

# .\.venv\Scripts\python .\inference.py --model_path=.\models\model.safetensors --output_dir=.\output --audio_path "bordel.wav"
# then open in https://omine-me.github.io/AudioDatasetChecker/compare.html
