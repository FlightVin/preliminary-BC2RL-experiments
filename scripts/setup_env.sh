#!/usr/bin/env bash
# Build the BC2RL environment under $PROJ_DATA (compute node only) and fetch the pinned model.
# Idempotent: re-running reuses the venv and the model snapshot.
set -euo pipefail
: "${PROJ_DATA:?set PROJ_DATA}"
export UV_CACHE_DIR="$PROJ_DATA/cache/uv" HF_HOME="$PROJ_DATA/hf" UV_PYTHON_INSTALL_DIR="$PROJ_DATA/cache/uv-python"
VENV="$PROJ_DATA/venv"
UV="${UV:-$HOME/.local/bin/uv}"
mkdir -p "$PROJ_DATA" "$UV_CACHE_DIR" "$HF_HOME"
[ -x "$VENV/bin/python" ] || "$UV" venv --python 3.11 "$VENV"
"$UV" pip install --python "$VENV/bin/python" -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
"$UV" pip freeze --python "$VENV/bin/python" > "$PROJ_DATA/env.lock.txt"
"$VENV/bin/python" - <<'PY'
import json, os
from huggingface_hub import snapshot_download, HfApi
repo = "Qwen/Qwen2.5-0.5B-Instruct"
info = HfApi().model_info(repo)
path = snapshot_download(repo, revision=info.sha)
out = {"repo": repo, "revision": info.sha, "path": path}
json.dump(out, open(os.path.join(os.environ["PROJ_DATA"], "model_revision.json"), "w"), indent=1)
print(out)
PY
