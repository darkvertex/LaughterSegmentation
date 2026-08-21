# Cloud Run Job image: GPU inference for laughter segmentation.
# Base includes CUDA runtime + PyTorch 2.1.2 (Python 3.10, which is <= 3.11).
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    HF_HUB_CACHE=/models/hf

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-job.txt .
RUN pip install --no-cache-dir -r requirements-job.txt

# Bake wav2vec2 backbone so job startup does not hit the network.
RUN python -c "from transformers import Wav2Vec2ForAudioFrameClassification as C; \
    C.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english', \
    num_labels=1, problem_type='single_label_classification')"

# Trained checkpoint (~1.26 GB). Downloaded at build time, not copied from the repo.
RUN mkdir -p /app/models \
    && curl -L --fail --retry 4 --retry-delay 5 \
        -o /app/models/model.safetensors \
        "https://huggingface.co/omine-me/LaughterSegmentation/resolve/main/model.safetensors?download=true"

COPY train/__init__.py train/model.py /app/train/
COPY evaluation/__init__.py /app/evaluation/
COPY evaluation/_utils/__init__.py evaluation/_utils/utils.py /app/evaluation/_utils/
COPY inference.py io_utils.py /app/

ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "inference.py"]
