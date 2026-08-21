#!/usr/bin/env bash
# (Re)create the models/model.safetensors symlink pointing at the cached weights.
#
# This runs on every container start: /workspace is re-checked out on each boot
# (removing this untracked symlink), and the `install` step does not re-run per
# boot when booting from a prebuilt environment build. inference.py defaults to
# ./models/model.safetensors, so this keeps the default command working.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CACHE_PATH="$HOME/.cache/laughterseg/model.safetensors"

mkdir -p "$REPO_DIR/models"
ln -sfn "$MODEL_CACHE_PATH" "$REPO_DIR/models/model.safetensors"
echo "Linked $REPO_DIR/models/model.safetensors -> $MODEL_CACHE_PATH"
