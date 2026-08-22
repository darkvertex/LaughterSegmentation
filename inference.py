import argparse
import json
import os
import os.path as osp
import sys

import librosa
import numpy as np
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from pydub.silence import detect_silence
import safetensors
from scipy import signal
import torch
from transformers.trainer_utils import set_seed

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "train"))
from evaluation._utils.utils import concat_close, remove_short
from train.model import Model

AUDIO_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLE_RATE = 16000
OVERLAP_SEC = 2.0
DEFAULT_MODEL_PATH = "./models/model.safetensors"
DEFAULT_OUTPUT_DIR = "./output"
SEED = 42
WEIGHTS_REPO = "omine-me/LaughterSegmentation"
WEIGHTS_FILENAME = "model.safetensors"


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


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_model_weights(model_path):
    """Use a local checkpoint when present; otherwise download from Hugging Face."""
    if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    dest_dir = os.path.dirname(os.path.abspath(model_path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    download_kwargs = {
        "repo_id": WEIGHTS_REPO,
        "filename": WEIGHTS_FILENAME,
        "local_dir": dest_dir,
    }
    try:
        downloaded = hf_hub_download(local_dir_use_symlinks=False, **download_kwargs)
    except TypeError:
        downloaded = hf_hub_download(**download_kwargs)
    if os.path.abspath(downloaded) != os.path.abspath(model_path):
        os.replace(downloaded, model_path)
    return model_path


def load_model(model_path, device=None, audio_model_name=AUDIO_MODEL_NAME, sr=SAMPLE_RATE, seed=SEED):
    """Load the laughter segmentation model once and return it in eval mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    if device is None:
        device = get_device()

    if not osp.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Download the model file and place it in the specified path."
        )

    model = Model(audio_model_name, device, sr).to(device)
    map_location = device.index if device.type == "cuda" else "cpu"
    state_dict = safetensors.torch.load_file(model_path, map_location)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def segment_laughter(
    model,
    audio_path,
    input_sec=7,
    batch_size=10,
    overlap_sec=OVERLAP_SEC,
    sr=SAMPLE_RATE,
):
    """Run laughter segmentation and return a dict of {idx: {start_sec, end_sec}}."""
    if input_sec <= overlap_sec:
        raise ValueError(f"input_sec ({input_sec}) must be greater than overlap_sec ({overlap_sec})")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    laughter = {}
    laughter_idx = 0

    with torch.no_grad():
        audio_array = librosa.load(audio_path, sr=sr, mono=True)[0]
        audio_array = custom_amplituder_small_portion(audio_array, sr)

        # get each array of input_sec
        for array_idx in range(0, len(audio_array), int(sr*(input_sec-overlap_sec))*batch_size):
            batched_arrays = []
            should_break = False
            for batch_idx in range(batch_size):
                array = audio_array[array_idx+batch_idx*int(sr*(input_sec-overlap_sec)): array_idx+batch_idx*int(sr*(input_sec-overlap_sec))+sr*input_sec]
                if len(array) < sr*input_sec:
                    # fill 0 to the end of array
                    array = np.append(array, np.zeros(sr*input_sec-len(array)))
                    should_break = True
                batched_arrays.append(array)
                if should_break:
                    break

            input_values = torch.from_numpy(np.array(batched_arrays)).type(torch.FloatTensor)
            outputs = model(input_values=input_values)

            logits = outputs[1]

            #  --- predict ends ---

            preds = torch.sigmoid(logits.to(torch.float32))

            for batch_idx, pred in enumerate(preds): # each batch
                frame_pred = list(map(round, pred.cpu().tolist(), [3]*len(pred)))

                # change to 0, 1
                frame_pred = (np.array(frame_pred)>=0.5).astype(int)

                batch_start_sec = (array_idx+batch_idx*int(sr*(input_sec-overlap_sec)))/float(sr)
                frame_count = len(frame_pred)
                start_idx = None
                end_idx = None
                status = "not_laughing"
                for idx, frame in enumerate(frame_pred):
                    if frame == 1:
                        if status == "not_laughing":
                            start_idx = idx
                            status = "laughing"

                        # if the last frame is laughing
                        if status == "laughing" and idx == frame_count-1:
                            laughter[str(laughter_idx)] = {
                                "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                "end_sec": batch_start_sec + input_sec,
                            }
                            laughter_idx += 1
                            start_idx = None
                            end_idx = None
                    elif frame == 0:
                        # end of laughter
                        if status == "laughing":
                            end_idx = idx
                            status = "not_laughing"
                            if start_idx == 0:
                                laughter[str(laughter_idx)] = {
                                "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                "end_sec": batch_start_sec + (input_sec/frame_count)*end_idx,
                                }
                            else:
                                laughter[str(laughter_idx)] = {
                                    "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                    "end_sec": batch_start_sec + (input_sec/frame_count)*end_idx,
                                }
                            laughter_idx += 1
                            start_idx = None
                            end_idx = None

    if overlap_sec > .0:
        laughter = merge_events([laughter])

    laughter = concat_close(laughter, 0.2)
    laughter = remove_short(laughter, 0.2)
    return laughter


def write_segments(laughter, output_path):
    output_dir = osp.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(laughter, f)
    return output_path


def main(audio_path, output_dir, model_path, input_sec=7, batch_size=10):
    model = load_model(model_path)
    laughter = segment_laughter(
        model,
        audio_path,
        input_sec=input_sec,
        batch_size=batch_size,
    )
    basename = osp.splitext(osp.basename(audio_path))[0]
    out_file = osp.join(output_dir, basename + ".json")
    write_segments(laughter, out_file)
    return out_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    
    main(args.audio_path, args.output_dir, args.model_path)
