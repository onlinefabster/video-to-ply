# video-to-ply

Turn a room walkthrough video into a Gaussian-splat `.ply` file — with all GPU
work running on **Hugging Face Jobs** (no local CUDA card required).

## How it works

1. `app.py` (Gradio) uploads your video to a private HF dataset repo.
2. `submitter.submit()` launches an HF Job on a GPU container
   (`ghcr.io/nerfstudio-project/nerfstudio`) running `bootstrap.sh`.
3. Inside the job: `ns-process-data video` (ffmpeg + COLMAP) →
   `ns-train splatfacto` → `ns-export gaussian-splat` → upload `splat.ply`.
4. `submitter.wait_and_download()` polls to COMPLETED and downloads the `.ply`.

Design is adapted from
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)'s
`--hf-jobs` submitter (`src/mjlab_microduck/hf_jobs.py`).

## Setup (host without GPU)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
hf auth login           # or export HF_TOKEN
chmod +x bootstrap.sh   # only for local sanity runs
```

Needs an HF token with the **Jobs** permission and a positive credit balance.

## Run

```bash
python app.py           # web UI at http://localhost:7860
```

CLI (uses a live job — costs money):

```bash
python submitter.py room.mp4 --flavor l4x1 --frames 300 --iterations 15000
```

`--dry-run` prints the full job spec without spending a cent:

```bash
python submitter.py room.mp4 --dry-run
```

## Repos created (private, per run)

- `<ns>/video2ply-src` — dataset repo holding the uploaded video (input volume)
- `<ns>/video2ply-<timestamp>` — model repo holding `splat.ply` (output)

## Notes / caveats

- **Cost:** billed per Start+Run minute; `--timeout` caps *duration, not spend*.
- **No `.ply`** on a timed-out/cancelled job is possible — check job logs.
- Nerfstudio images are large (~minutes cold pull, during the billed
  SCHEDULING phase); pin a specific tag in `submitter.py` once you've tested
  one that works on the current CUDA stack.
- Open the result in a splat viewer: [SuperSplat](https://playcanvas.com/supersplat/editor)
  (free), or a Three.js-based splat viewer for embedding in a site.