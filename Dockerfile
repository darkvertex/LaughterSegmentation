# Runpod Serverless worker image.
# Replicate builds stay on cog.yaml; `cog build` does not read this file.
#
# Weights and the wav2vec2 config are baked in so a cold start does no
# network I/O, matching the build.run steps in cog.yaml. The weight download
# happens before COPY so source edits do not re-fetch the checkpoint.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/src/models/model.safetensors \
    HF_HOME=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        gnupg \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Checkpoint first (curl only) so dependency changes do not re-download it.
RUN mkdir -p /src/models \
    && curl -fsSL -o /src/models/model.safetensors \
        https://huggingface.co/omine-me/LaughterSegmentation/resolve/main/model.safetensors \
    && test "$(stat -c%s /src/models/model.safetensors)" -gt 1200000000

COPY replicate-requirements.txt runpod-requirements.txt /tmp/reqs/
RUN python -m pip install --no-cache-dir -r /tmp/reqs/runpod-requirements.txt \
    && python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english')"

WORKDIR /src
COPY . /src

CMD ["python", "-u", "handler.py"]
