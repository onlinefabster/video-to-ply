#!/usr/bin/env bash
# Runs INSIDE the HF Job GPU container. Turns the mounted video into an
# uploaded .ply. Env vars are set by the submitter: VIDEO_FILE, NUM_FRAMES,
# ITERATIONS, CKPT_REPO. HF_TOKEN is passed as a secret.
set -euo pipefail

echo "[bootstrap] video=$VIDEO_FILE frames=$NUM_FRAMES iters=$ITERATIONS"
mkdir -p /work && cd /work

# 1+2. Extract frames + run SfM (ns-process-data wraps both).
# `--matching-method sequential` keeps the camera lock frame-to-frame for
# ordered video, preventing the mis-estimated "bowed arc" trajectory that
# bends the room around the camera. `--num-downscales 1` keeps images
# higher-res so more matchable features survive for robust registration.
echo "[bootstrap] extracting frames + SfM (sequential matching, low downscale)"
ns-process-data video \
  --data "/src/$VIDEO_FILE" \
  --output-dir ./processed \
  --num-frames-target "$NUM_FRAMES" \
  --matching-method sequential \
  --num-downscales 1

# 3. Train the Gaussian splat.
echo "[bootstrap] training splatfacto"
ns-train splatfacto \
  --data ./processed \
  --output-dir ./outputs \
  --max-num-iterations "$ITERATIONS" \
  --viewer.quit-on-train-completion True

# 4. Export to PLY.
echo "[bootstrap] exporting gaussian splat"
CONFIG=$(ls ./outputs/processed/splatfacto/*/config.yml | tail -1)
mkdir -p ./exports
ns-export gaussian-splat --load-config "$CONFIG" --output-dir ./exports
PLY=$(ls ./exports/*.ply | head -1)
echo "[bootstrap] generated: $PLY"

# 5. Upload the result to the model repo so the submitter can fetch it.
# The Nerfstudio image has NO plain `pip` binary, but `python3 -m pip` works.
# huggingface_hub is not preinstalled, so install it, then use its commit+LFS-
# aware upload_file (a raw HTTP PUT to the hub no longer works — 404).
echo "[bootstrap] uploading $PLY -> $CKPT_REPO"
python3 -m pip install -q --disable-pip-version-check huggingface_hub
PLY="$PLY" CKPT_REPO="$CKPT_REPO" python3 - <<'PY'
import os
from huggingface_hub import upload_file
ply = os.environ["PLY"]
repo = os.environ["CKPT_REPO"]
info = upload_file(
    path_or_fileobj=ply,
    path_in_repo="splat.ply",
    repo_id=repo,
    repo_type="model",
)
print(f"[bootstrap] uploaded splat.ply to {repo} ({getattr(info, 'commit_url', '')})")
PY

echo "[bootstrap] done"