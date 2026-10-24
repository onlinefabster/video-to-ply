#!/usr/bin/env bash
# Runs INSIDE the HF Job GPU container. Turns the mounted video into an
# uploaded .ply. Env vars are set by the submitter: VIDEO_FILE, NUM_FRAMES,
# ITERATIONS, CKPT_REPO. HF_TOKEN is passed as a secret.
set -euo pipefail

echo "[bootstrap] video=$VIDEO_FILE frames=$NUM_FRAMES iters=$ITERATIONS"
mkdir -p /work && cd /work

# 1+2. Extract frames + run COLMAP (ns-process-data wraps both).
echo "[bootstrap] extracting frames + COLMAP"
ns-process-data video \
  --data "/src/$VIDEO_FILE" \
  --output-dir ./processed \
  --num-frames-target "$NUM_FRAMES"

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
echo "[bootstrap] uploading $PLY -> $CKPT_REPO"
PLY="$PLY" CKPT_REPO="$CKPT_REPO" python3 - <<'PY'
import os
from huggingface_hub import HfApi
ply = os.environ["PLY"]
repo = os.environ["CKPT_REPO"]
HfApi().upload_file(
    path_or_fileobj=ply,
    path_in_repo="splat.ply",
    repo_id=repo,
    repo_type="model",
)
print(f"[bootstrap] uploaded splat.ply to {repo}")
PY

echo "[bootstrap] done"