# Video → PLY (Gaussian Splat) Pipeline: Build Guide

Goal: upload a walkthrough video of a room and get back a `.ply` Gaussian splat file, via a small web app (Gradio) running on a GPU machine (local PC, rented GPU, or a Hugging Face Docker Space).

---

## 1. Architecture

```
Video (.mp4)
   │
   ▼
[1] Frame extraction        ffmpeg
   │
   ▼
[2] Camera poses (SfM)      COLMAP  → sparse point cloud + camera positions
   │
   ▼
[3] Splat training          Nerfstudio "splatfacto" (GPU required)
   │
   ▼
[4] Export                  ns-export gaussian-splat → splat.ply
   │
   ▼
[5] Front end               Gradio: upload video → download PLY
   │
   ▼
[6] (Optional) Viewer       Three.js / React Three Fiber splat viewer
```

Nerfstudio's `ns-process-data video` already wraps steps 1 and 2 (ffmpeg + COLMAP), so the minimum pipeline is **three commands**.

---

## 2. Requirements

| Item | Needed |
|------|--------|
| GPU | Nvidia, ideally 8 GB+ VRAM (training is CUDA-only) |
| OS | Linux (or Windows with WSL2 / Docker) |
| Software | Docker + NVIDIA Container Toolkit |
| Python | 3.10+ (inside the container) |
| Disk | 10–20 GB free for frames, COLMAP data, and outputs |

---

## 3. Minimal manual pipeline (test this first, before building the app)

Run these inside a Nerfstudio environment:

```bash
# 1+2. Extract frames and run COLMAP
ns-process-data video \
  --data room.mp4 \
  --output-dir ./processed \
  --num-frames-target 300

# 3. Train the Gaussian splat
ns-train splatfacto \
  --data ./processed \
  --output-dir ./outputs \
  --max-num-iterations 15000 \
  --viewer.quit-on-train-completion True

# 4. Export to PLY
ns-export gaussian-splat \
  --load-config ./outputs/processed/splatfacto/<timestamp>/config.yml \
  --output-dir ./exports
# → ./exports/splat.ply
```

If this works on your machine, everything below is just packaging. Flag names can change between Nerfstudio versions, so check `ns-train splatfacto --help`.

---

## 4. What to build

### 4.1 Dockerfile

Easiest base: the official Nerfstudio image, which already contains CUDA, PyTorch, and a CUDA-enabled COLMAP. Pin a specific tag instead of `latest` once you know one that works.

```dockerfile
FROM ghcr.io/nerfstudio-project/nerfstudio:latest

USER root
RUN pip install --no-cache-dir gradio

WORKDIR /app
COPY app.py pipeline.py /app/

EXPOSE 7860
CMD ["python", "app.py"]
```

### 4.2 `pipeline.py`: the processing logic

```python
import subprocess, shutil, glob, os, tempfile
from pathlib import Path

def run(cmd):
    subprocess.run(cmd, check=True)

def video_to_ply(video_path, num_frames=300, iterations=15000, progress=None):
    work = Path(tempfile.mkdtemp(prefix="splat_"))
    processed, outputs, exports = work / "processed", work / "outputs", work / "exports"

    if progress: progress(0.1, desc="Extracting frames + COLMAP")
    run(["ns-process-data", "video", "--data", video_path,
         "--output-dir", str(processed),
         "--num-frames-target", str(num_frames)])

    if progress: progress(0.4, desc="Training splat")
    run(["ns-train", "splatfacto", "--data", str(processed),
         "--output-dir", str(outputs),
         "--max-num-iterations", str(iterations),
         "--viewer.quit-on-train-completion", "True"])

    if progress: progress(0.9, desc="Exporting PLY")
    config = sorted(glob.glob(str(outputs / "**" / "config.yml"), recursive=True))[-1]
    run(["ns-export", "gaussian-splat", "--load-config", config,
         "--output-dir", str(exports)])

    ply = next(exports.glob("*.ply"))
    final = Path("/tmp") / f"{work.name}.ply"
    shutil.copy(ply, final)
    shutil.rmtree(work, ignore_errors=True)   # free disk
    return str(final)
```

### 4.3 `app.py`: Gradio front end

```python
import gradio as gr
from pipeline import video_to_ply

def convert(video, frames, iterations, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Please upload a video.")
    return video_to_ply(video, int(frames), int(iterations), progress)

demo = gr.Interface(
    fn=convert,
    inputs=[
        gr.Video(label="Room video"),
        gr.Slider(100, 600, value=300, step=50, label="Frames to extract"),
        gr.Slider(5000, 30000, value=15000, step=1000, label="Training iterations"),
    ],
    outputs=gr.File(label="Gaussian splat (.ply)"),
    title="Video → Gaussian Splat (PLY)",
)

demo.queue(max_size=2)                      # one job at a time on one GPU
demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 5. Deployment options

**A. Local PC / homelab with an Nvidia GPU (simplest)**
```bash
docker build -t video2ply .
docker run --gpus all -p 7860:7860 video2ply
```
Open `http://localhost:7860`.

**B. Hugging Face Docker Space**
- Create a Space with SDK = **Docker** and select a GPU hardware tier (paid, so check current pricing).
- Put the `Dockerfile`, `app.py`, `pipeline.py` in the repo.
- Add this header to the Space's `README.md`:
  ```yaml
  ---
  title: Video to Splat
  sdk: docker
  app_port: 7860
  ---
  ```
- Notes: disk is ephemeral, Spaces may sleep when idle, and long jobs can hit time limits. Keep videos short.

**C. Rented GPU (RunPod, Vast.ai, etc.)**
- Same Docker image, pay per hour, shut down when finished.

---

## 6. Capture tips (these matter more than any setting)

- 1–2 minutes of video per room, **slow** walking, no sudden turns.
- Avoid motion blur: good light, steady hands, 4K if possible.
- Overlap your passes, moving around the room in two or three loops at different heights.
- Keep exposure and focus locked if the camera allows it.
- Mirrors, windows, TVs, and plain white walls cause artifacts.
- Scan one room at a time rather than the whole house.

---

## 7. Known pitfalls

| Problem | Likely cause / fix |
|---------|--------------------|
| COLMAP fails or registers few frames | Blurry or low-overlap video: refilm slower, add more frames |
| CUDA out of memory | Fewer frames, lower resolution, or a bigger GPU |
| Floating blobs / fog in the splat | Poor poses or reflections: crop in SuperSplat editor |
| Huge PLY file | Compress or crop (see below) before putting it on the web |
| Build errors in the Docker image | Pin working versions of Nerfstudio, CUDA, and PyTorch |

---

## 8. After you have a PLY

- **Clean up and crop:** SuperSplat (free web editor) removes floaters and trims the scene.
- **Compress for the web:** convert to a compressed splat format (e.g. SPZ or SOG) with the tools in SuperSplat or similar.
- **View it in your own site:** use a Three.js-based splat library (for example Spark or `gaussian-splats-3d`) inside a React Three Fiber component, with orbit controls and optional walkthrough camera.

---

## 9. Suggested build order

1. Get the three manual commands working on a short test video.
2. Wrap them in `pipeline.py`, tested from the command line.
3. Add the Gradio `app.py`, tested locally.
4. Containerize with the Dockerfile.
5. Deploy (local, rented GPU, or Hugging Face Space).
6. Build the React Three Fiber viewer for the exported PLY.

---

## 10. Optional improvements

- Allow uploading a ZIP of photos instead of a video.
- Add quality presets (fast / balanced / high).
- Email or notify when a job finishes.
- Store results and serve them through a link instead of a direct download.
- Auto-run SuperSplat-style cropping or compression after export.
