#!/usr/bin/env bash
# Runs INSIDE the HF Job GPU container. Turns the mounted video into an
# uploaded .ply. Env vars are set by the submitter: VIDEO_FILE, NUM_FRAMES,
# ITERATIONS, CKPT_REPO. HF_TOKEN is passed as a secret.
set -euo pipefail

echo "[bootstrap] video=$VIDEO_FILE frames=$NUM_FRAMES iters=$ITERATIONS"
mkdir -p /work && cd /work

# hloc needs `wget` to fetch NetVLAD/SuperPoint weights; the image lacks it and
# the package index is empty until `apt-get update`. Also lacks `git`, and its
# SuperGluePretrainedNetwork dependency is neither pip-installable nor importable
# via a plain PYTHONPATH (namespace-package quirk: works only at sys.path[0]).
# So: install wget, pull the source tarball, and drop a sitecustomize.py that
# force-inserts /work at the front of sys.path for EVERY python subprocess.
apt-get update -qq >/dev/null 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wget >/dev/null 2>&1
python3 - <<'PY'
import urllib.request, tarfile, os
URL = "https://github.com/magicleap/SuperGluePretrainedNetwork/archive/refs/heads/master.tar.gz"
urllib.request.urlretrieve(URL, "/tmp/sg.tar.gz")
with tarfile.open("/tmp/sg.tar.gz") as t:
    t.extractall("/work")
d = "/work/SuperGluePretrainedNetwork"
if not os.path.isdir(d):
    os.rename("/work/SuperGluePretrainedNetwork-master", d)
open("/work/sitecustomize.py", "w").write('import sys; sys.path.insert(0, "/work")\n')
print("[bootstrap] wget + superpoint setup done")
PY
export PYTHONPATH=/work${PYTHONPATH:+:$PYTHONPATH}

# 1+2. Extract frames + run SfM (ns-process-data wraps both).
# Use HLoc with SuperPoint features + SuperGlue matcher: learned deep
# features dramatically out-register stock COLMAP/SIFT for ordered video
# (the stock matcher only locked 4-8 of 400 frames here, producing a
# bowed-arc shell). SuperGlue tracks frames reliably -> clean camera path
# -> solid core instead of a curved cloud.
echo "[bootstrap] extracting frames + SfM (hloc superpoint+superglue)"
ns-process-data video \
  --data "/src/$VIDEO_FILE" \
  --output-dir ./processed \
  --num-frames-target "$NUM_FRAMES" \
  --sfm-tool hloc \
  --feature-type superpoint \
  --matcher-type superglue \
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