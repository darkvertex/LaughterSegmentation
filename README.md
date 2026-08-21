# Laughter Segmentation

## Overview
You can extract a exact segment of laughter from various talking audio using trained model and code. You can also train your own model.

Code, annotations, and model are described in the following paper:
[Taisei Omine, Kenta Akita, and Reiji Tsuruno, "Robust Laughter Segmentation with Automatic Diverse Data Synthesis", Interspeech 2024.](https://doi.org/10.21437/Interspeech.2024-1644)

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
1. Open Terminal and go to the directory where `inference.py` is located.
1. Run `python inference.py --audio_path audio.wav`. You have to change *audio.wav* to your own audio path. You can use common audio format like `mp3`, `wav`, `opus`, etc. 16kHz wav audio is faster. If the audio fails to load, run the following command and also download FFmpeg and add it to the PATH.
    ```Batchfile
    python -m pip uninstall pysoundfile
    python -m pip uninstall soundfile
    python -m pip install soundfile
    ```
1. If you want to change output directory, use  `--output_dir` option. If you want to use your own model, use `--model_path` option.
1. Result will be saved in output directory in json format. To visualize the results, you can use [this site](https://omine-me.github.io/AudioDatasetChecker/compare.html) (not perfect because it's for debugging).

## Cloud Run Job

The same CLI runs as a **Cloud Run Job** (GPU recommended). There is no public HTTP API and no API key: IAM decides who can execute, and the job writes JSON to a `gs://` object you pass in.

Cloud Run tracks each execution (`PENDING` / `RUNNING` / `SUCCEEDED` / `FAILED` / `CANCELLED`). GCS is the artifact store — jobs do not keep output files after the container exits.

### Inputs

| Flag | Env var | Description |
|---|---|---|
| `--audio-url` | `AUDIO_URL` | `https://...` or `gs://bucket/object` sound file |
| `--output-gcs` | `OUTPUT_GCS` | `gs://bucket/path.json` destination (required with `--audio-url`) |
| `--model_path` | | Defaults to `/app/models/model.safetensors` in the image |

Passing per-run URLs requires **execute with overrides** (`roles/run.jobsExecutorWithOverrides`). Plain Job Invoker can only run the job with default args.

Use `--args` when URLs have no commas. Use `--update-env-vars` if a URL might contain commas (gcloud splits `--args` on `,`).

### IAM and storage

Create a dedicated runtime service account and output bucket. Grant the job SA write access on that bucket only.

```powershell
$PROJECT = "YOUR_PROJECT"
$REGION = "us-central1"
$SA = "laughter-job@$PROJECT.iam.gserviceaccount.com"
$BUCKET = "$PROJECT-laughter-segments"

gcloud iam service-accounts create laughter-job --display-name "Laughter segmentation job"
gsutil mb -l $REGION gs://$BUCKET
gsutil iam ch "serviceAccount:${SA}:objectAdmin" gs://$BUCKET

# Callers who start jobs with a custom audio/output URL:
gcloud run jobs add-iam-policy-binding laughter-segmentation `
  --region $REGION `
  --member "user:YOU@example.com" `
  --role roles/run.jobsExecutorWithOverrides

# Callers who only need to read results:
gsutil iam ch "user:YOU@example.com:objectViewer" gs://$BUCKET
```

If `--audio-url` is `gs://`, also grant the job SA `objectViewer` on that audio bucket. Public `https://` audio needs no extra IAM.

Keep `--output-gcs` under `gs://$PROJECT-laughter-segments/`.

### Build and deploy

Image is large (CUDA PyTorch + wav2vec2 + 1.26 GB checkpoint). First push is slow. GPU jobs need a GPU region such as `us-central1`.

```powershell
$PROJECT = "YOUR_PROJECT"
$REGION = "us-central1"
$IMAGE = "$REGION-docker.pkg.dev/$PROJECT/laughter/laughter-segmentation:latest"

gcloud artifacts repositories create laughter --repository-format=docker --location=$REGION
gcloud builds submit --tag $IMAGE --timeout=3600

gcloud run jobs deploy laughter-segmentation `
  --image $IMAGE `
  --region $REGION `
  --gpu 1 --gpu-type nvidia-l4 `
  --gpu-zonal-redundancy=disabled `
  --cpu 8 --memory 16Gi `
  --task-timeout 3600 `
  --tasks 1 `
  --max-retries 0 `
  --service-account "laughter-job@$PROJECT.iam.gserviceaccount.com"
```

Task timeout default is 10 minutes; the deploy sets **1 hour** (max 168 hours). Episode-length CPU inference is much slower — use the L4 GPU.

### Execute

```powershell
gcloud run jobs execute laughter-segmentation --region us-central1 `
  --args="--audio-url,https://example.com/episode.wav,--output-gcs,gs://YOUR_BUCKET/episodes/episode.json"

gcloud run jobs executions describe-latest --job laughter-segmentation --region us-central1
gsutil cat gs://YOUR_BUCKET/episodes/episode.json
```

Equivalent with env vars:

```powershell
gcloud run jobs execute laughter-segmentation --region us-central1 `
  --update-env-vars="AUDIO_URL=https://example.com/episode.wav,OUTPUT_GCS=gs://YOUR_BUCKET/episodes/episode.json"
```

JSON format matches local `inference.py` output (`"0": { "start_sec", "end_sec" }`, ...).

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
