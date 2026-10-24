# Video → PLY via HF Jobs — Architecture Overview

## What it does
A Gradio web app where a user uploads a room walkthrough video and later downloads a ready Gaussian-splat `.ply` file. All GPU work runs on Hugging Face managed hardware via **HF Jobs** — no local CUDA card needed.

## Core idea
The front end never touches the heavy ML. It packages the job and lets Hugging Face's GPU do `COLMAP → splatfacto → export`, then fetches the result. Design borrows directly from pollen-robotics/microduck_rl's `--hf-jobs` submitter (`hf_jobs.py`). Live on an ordinary host (this Linux box); jobs bill per minute only while running.

## Data flow

```
[User]
  │  uploads video (.mp4)
  ▼
[Gradio app]  ──────────────────────────────── runs on host (no GPU)
  │ upload_video()        → private dataset repo   (input volume)
  │ run_job()             → HF Jobs boots GPU container
  │                        image: ghcr.io/nerfstudio-project/nerfstudio
  │                        flavors: l4x1 / a10g-large / a100-large
  ▼
[GPU container (HF job)]  ─────────────────────── one-shot, billed per min
  bootstrap.sh:
    1. ns-process-data video   (ffmpeg frames + COLMAP poses)
    2. ns-train splatfacto      (Gaussian splat)
    3. ns-export gaussian-splat → splat.ply
    4. upload splat.ply          → private model repo  (output volume)
  │
  ▼
[Gradio app]  poll: inspect_job → COMPLETED → hf_hub_download(splat.ply)
  │
  ▼
[User]  downloads .ply
```

## Components

| Piece | File | Role |
|-------|------|------|
| UI | `app.py` | Upload, params, progress, file download (Gradio) |
| Submitter | `submitter.py` | Auth/namespace, video upload, `run_job`, supervision loop, download |
| In-container script | `bootstrap.sh` | The 3-step nerfstudio pipeline + final `.ply` upload |
| Image | `ghcr.io/nerfstudio-project/nerfstudio` | Prebuilt CUDA+COLMAP+nerfstudio runtime |

## Key exchanges (HF Jobs API, via `huggingface_hub`)

- **Input:** read-only volume `Volume(type="dataset", source=<repo>, mount_path="/src")`.
- **Auth:** `HF_TOKEN` passed as a job secret; also `--dry-run` guard against accidental spend.
- **Lifecycle:** `run_job` → `inspect_job` (`SCHEDULING→RUNNING→COMPLETED/ERROR`) → `fetch_job_logs` → `cancel_job`.
- **Output:** job uploads `splat.ply` to a private model repo; submittter downloads it.

## Why HF Jobs over a HF Space
- One-shot: boots per video, exits, bills only actual run time — no idle GPU.
- Avoids Space limits: no sleep, no persistent-disk constraints, short-job timeouts are fine.
- (Space option exists but is more constrained — see the full plan.)

## Risks at a glance
- Billing is per Start+Run minute; **timeout caps duration, not spend** → use `--dry-run` + short timeouts.
- Flaky GPU-node volume mounts (known issue) → resubmit-on-failure loop.
- A timed-out/cancelled job isn't a promise the `.ply` was exported → surface a clear "no output" error.

_Full detail: `.hermes/plans/2026-10-24_164233-video2ply-hf-jobs-gradio.md`_