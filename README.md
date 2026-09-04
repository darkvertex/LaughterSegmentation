# Laughter Segmentation

## Overview
You can extract a exact segment of laughter from various talking audio using trained model and code. You can also train your own model.

Code, annotations, and model are described in the following paper:
[Taisei Omine, Kenta Akita, and Reiji Tsuruno, "Robust Laughter Segmentation with Automatic Diverse Data Synthesis", Interspeech 2024.](https://doi.org/10.21437/Interspeech.2024-1644)

## What's special about this fork

This fork of [omine-me/LaughterSegmentation](https://github.com/omine-me/LaughterSegmentation) keeps the original laughter segmenter and adds audience-event extras used in comedy-show post-production.

**Applause detection.** In addition to laughter ranges, inference runs a NumPy port of the [AMP applause-detection](https://github.com/AudiovisualMetadataPlatform/applause-detection) binary MLP (`pretrained/applause-binary-20210203`). Weights are `models/applause_mlp.npz` (regenerate with `scripts/export_applause_weights.py`). TensorFlow is not a runtime dependency. Applause segments use the same timed-range shape as laughter.

**Sound level peaks.** Each laughter and applause event is annotated on the original audio (dBFS) with RMS, peak, crest (`peak_db - rms_db`), and RMS relative to the whole file. Downstream tools can bucket intensity (for example chuckle vs riot) from these fields; this repo does not assign those labels.

**Hosted Replicate model.** A Cog image is published at [replicate.com/darkvertex/laughtersegmentation](https://replicate.com/darkvertex/laughtersegmentation). Local `cog predict` and that hosted model return the same JSON file.

**JSON output on success.** The result is one object with `laughter` and `applause` maps. Keys are string indices in time order (`"0"`, `"1"`, …). A kind with no detections is `{}`. Each segment looks like:

```json
{
  "laughter": {
    "0": {
      "start_sec": 12.41,
      "end_sec": 14.08,
      "rms_db": -21.4,
      "peak_db": -9.1,
      "crest_db": 12.3,
      "rel_rms_db": -4.0
    }
  },
  "applause": {
    "0": {
      "start_sec": 103.2,
      "end_sec": 108.7,
      "rms_db": -18.0,
      "peak_db": -2.3,
      "crest_db": 15.7,
      "rel_rms_db": 0.7
    }
  }
}
```

| Field | Meaning |
| --- | --- |
| `start_sec` / `end_sec` | Event bounds in seconds |
| `rms_db` | Segment RMS loudness (dBFS) |
| `peak_db` | Segment peak amplitude (dBFS) |
| `crest_db` | `peak_db - rms_db` |
| `rel_rms_db` | Segment RMS minus whole-file RMS |

Silent or empty slices floor at `-80.0` dB.

## Installation
```Batchfile
git clone https://github.com/omine-me/LaughterSegmentation.git
cd LaughterSegmentation
python -m pip install -r requirements.txt
# ↓ Depends on your environment. See https://pytorch.org/get-started/locally/
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
Run in Venv environment is recommended. Also, download `model.safetensors` from [Huggingface](https://huggingface.co/omine-me/LaughterSegmentation/tree/main) (1.26 GB) and place it in `models` directory and make sure the name is `model.safetensors`.

Python<=3.11 is required ([#2](https://github.com/omine-me/LaughterSegmentation/issues/2)).

Tested on Windows 11 with GeForce RTX 2060 SUPER.

## Usage
1. Prepare audio file.
2. Open Terminal and go to the directory where `inference.py` is located.
3. Run `python inference.py --audio_path audio.wav`. You have to change *audio.wav* to your own audio path. You can use common audio format like `mp3`, `wav`, `opus`, etc. 16kHz wav audio is faster. If the audio fails to load, run the following command and also download FFmpeg and add it to the PATH.
    ```Batchfile
    python -m pip uninstall pysoundfile
    python -m pip uninstall soundfile
    python -m pip install soundfile
    ```
4. If you want to change output directory, use  `--output_dir` option. If you want to use your own model, use `--model_path` option.
5. Result will be saved in output directory in json format (`laughter` and `applause` timed ranges). To visualize the results, you can use [this site](https://omine-me.github.io/AudioDatasetChecker/compare.html) (not perfect because it's for debugging).

## Replicate (Cog)
This repository includes a Cog wrapper for deploying on Replicate.

The build downloads `model.safetensors` (and the base wav2vec2 `config.json`) into the image via the `build.run` steps in `cog.yaml`, so no weights need to be present locally and cold boots on Replicate do no network I/O. Any local `models/*.safetensors` is excluded from the build context by `.dockerignore`.

1. Install Cog CLI: https://cog.run/getting-started/
2. Build and test locally:
  ```Batchfile
  cog build
  cog predict -i audio=@./your_audio.wav
  ```
3. Push to Replicate (after `replicate login`):
  ```Batchfile
  cog push r8.im/<your-username>/laughter-segmentation
  ```

### GitHub Actions
- `.github/workflows/cog-build.yml` runs on every pull request: `pytest` and `cog build` + a CPU `cog predict` smoke test, in parallel. Nothing is pushed.
- `.github/workflows/cog-release.yml` runs on every push to `main` (or manually): `cog build` then `cog push`. It runs in the `production` GitHub Environment and needs the secret `REPLICATE_CLI_AUTH_TOKEN` defined there (Settings > Environments > production). The value must be the CLI auth token copied from https://replicate.com/auth/token, not an API token (`r8_...`) from the account settings page: `cog login` rejects API tokens. The job fails early with a clear message if the secret is missing or is an API token.

Inputs exposed by the predictor:
- `audio`: input audio file.
- `input_sec` (default `7.0`): inference window size in seconds.
- `batch_size` (default `10`): number of windows per forward pass.

Output: a JSON file in the format documented under [What's special about this fork](#whats-special-about-this-fork).

## Training
Read [README](/train/README.md) in train directory.

## Evaluation (Includes our evaluation dataset)
Read [README](/evaluation/README.md) in evaluavtion directory.

## License
This repository is MIT-licensed, but [the publicly available trained model](https://huggingface.co/omine-me/LaughterSegmentation/tree/main) is currently available for research use only.

## Citation
Cite as: `Omine, T., Akita, K., Tsuruno, R. (2024) Robust Laughter Segmentation with Automatic Diverse Data Synthesis. Proc. Interspeech 2024, 4748-4752, doi: 10.21437/Interspeech.2024-1644`

or
```
@inproceedings{omine24_interspeech,
  title     = {Robust Laughter Segmentation with Automatic Diverse Data Synthesis},
  author    = {Taisei Omine and Kenta Akita and Reiji Tsuruno},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4748--4752},
  doi       = {10.21437/Interspeech.2024-1644},
}
```

## Contact
Use Issues or reach out my [X(Twitter)](https://x.com/mineBeReal).
