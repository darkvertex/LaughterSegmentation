#!/usr/bin/env bash
# Idempotent setup for the LaughterSegmentation Cloud Agent environment.
# Safe to run repeatedly: every step checks for existing state before doing work.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Persist heavy state outside the (re-checked-out) repo so it survives in the snapshot.
VENV_DIR="${LAUGHTERSEG_VENV:-$HOME/.venvs/laughterseg}"
MODEL_CACHE_DIR="$HOME/.cache/laughterseg"
MODEL_CACHE_PATH="$MODEL_CACHE_DIR/model.safetensors"
MODEL_URL="https://huggingface.co/omine-me/LaughterSegmentation/resolve/main/model.safetensors?download=true"
MODEL_MIN_BYTES=1200000000
BASE_AUDIO_MODEL="jonatasgrosman/wav2vec2-large-xlsr-53-english"

echo "==> LaughterSegmentation env setup (repo: $REPO_DIR)"

# --- System packages (ffmpeg for audio decode, libsndfile for soundfile,
#     build-essential so native wheels like pyroomacoustics can compile). ---
need_apt=0
for c in ffmpeg gcc g++; do command -v "$c" >/dev/null 2>&1 || need_apt=1; done
ldconfig -p 2>/dev/null | grep -q libsndfile || need_apt=1
if [ "$need_apt" -eq 1 ]; then
  echo "==> Installing system packages"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libsndfile1 build-essential
else
  echo "==> System packages already present"
fi

# --- uv (fast Python/venv manager) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- Python 3.11 (README requires Python <= 3.11) + virtualenv ---
uv python install 3.11
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "==> Creating venv at $VENV_DIR"
  uv venv --python 3.11 "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"

# The system default `c++` is clang, which cannot locate libstdc++ headers here;
# force gcc/g++ so source builds (e.g. pyroomacoustics) succeed.
export CC=gcc CXX=g++

# --- Python dependencies ---
# CPU PyTorch build (no GPU in Cloud Agent VMs). Matches the versions in README.
echo "==> Installing PyTorch (CPU)"
uv pip install --python "$PY" \
  torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cpu

# requirements.txt is a Windows pip-freeze: strip Windows-only packages and the
# legacy PySoundFile (soundfile is kept, per the project README's guidance).
echo "==> Installing requirements.txt (Linux-filtered)"
REQ_FILTERED="$(mktemp)"
grep -viE '^(pywin32|pywinpty|PySoundFile)([=<>~! ]|$)' "$REPO_DIR/requirements.txt" > "$REQ_FILTERED"
uv pip install --python "$PY" -r "$REQ_FILTERED"
rm -f "$REQ_FILTERED"

# librosa imports pkg_resources, which lives in setuptools (absent from the freeze).
# Pin <81 so pkg_resources remains available.
uv pip install --python "$PY" 'setuptools<81' wheel

# --- Trained model weights (~1.26 GB), downloaded once and cached in $HOME ---
mkdir -p "$MODEL_CACHE_DIR"
if [ ! -f "$MODEL_CACHE_PATH" ] || [ "$(stat -c%s "$MODEL_CACHE_PATH" 2>/dev/null || echo 0)" -lt "$MODEL_MIN_BYTES" ]; then
  echo "==> Downloading trained model.safetensors"
  curl -L --fail --retry 4 --retry-delay 5 -C - -o "$MODEL_CACHE_PATH" "$MODEL_URL"
else
  echo "==> Trained model already cached"
fi
# inference.py defaults to ./models/model.safetensors; link the cached copy in.
# The same script also runs as the `start` step, since /workspace is re-checked
# out on each boot (removing this untracked link) and install does not re-run.
bash "$REPO_DIR/.cursor/link_model.sh"

# --- Pre-cache the base wav2vec2 model so inference runs without network ---
echo "==> Pre-caching base audio model ($BASE_AUDIO_MODEL)"
"$PY" - <<PYEOF
from transformers import Wav2Vec2ForAudioFrameClassification
Wav2Vec2ForAudioFrameClassification.from_pretrained(
    "$BASE_AUDIO_MODEL", num_labels=1, problem_type="single_label_classification"
)
print("base audio model cached")
PYEOF

echo
echo "==> Done. Activate with:  source $VENV_DIR/bin/activate"
echo "    Run inference with:   $PY inference.py --audio_path <audio>"
