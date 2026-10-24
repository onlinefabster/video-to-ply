"""Gradio app: upload a room video -> submit an HF Job -> download the .ply."""

import os
import threading

import gradio as gr
from huggingface_hub import HfApi

import submitter

FLAT_STATUS = {
    "VIDEO2PLY_DEFAULT": gr.update(visible=False),
    "SESSION": gr.update(visible=False),
}

# One GPU job at a time on this submitter; hold the job id so a refresh can
# re-poll. We don't persist across processes — fine for a single-user app.
_state = {"lock": threading.Lock(), "job_id": None, "ckpt_repo": None, "namespace": None}


def convert(video, frames, iterations, flavor, dry_run, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Please upload a video.")

    api = HfApi()
    namespace = submitter.default_namespace(api)
    if not namespace:
        raise gr.Error("No HF token. Run `hf auth login` / export HF_TOKEN on the host.")

    progress(0.05, desc="Uploading video to Hub")
    src_repo, video_file = submitter.upload_video(api, namespace, video)

    run_name = submitter.make_run_name()
    ckpt_repo = f"{namespace}/{run_name}"

    if dry_run:
        return (
            gr.update(),
            "DRY RUN — no job submitted. Video uploaded to dataset only.",
        )

    progress(0.15, desc="Submitting HF job")
    job_id = submitter.submit(
        api, namespace, src_repo, video_file, ckpt_repo,
        frames=int(frames), iterations=int(iterations), flavor=flavor,
    )
    with _state["lock"]:
        _state.update(job_id=job_id, ckpt_repo=ckpt_repo, namespace=namespace)

    progress(0.2, desc="Waiting for job to complete")
    # wait_and_download blocks on streaming logs; run it on a worker thread so
    # the web UI stays responsive. It writes outputs/splat.ply and returns the
    # path; we surface that path as the downloadable file.
    result = threading.Event()
    holder: dict = {}

    def _run():
        try:
            local = submitter.wait_and_download(api, job_id, namespace, ckpt_repo)
            holder["path"] = local
            holder["err"] = None
        except BaseException as e:  # noqa: BLE001
            holder["err"] = str(e)
        finally:
            result.set()

    threading.Thread(target=_run, daemon=True).start()

    # Report progress without a hard sleep-loop dependency: short polls on the
    # job status until completion or error.
    import time
    last = 0.2
    while not result.is_set():
        time.sleep(5)
        try:
            st = api.inspect_job(job_id=job_id, namespace=namespace).status
        except Exception:  # noqa: BLE001
            st = None
        if st is not None:
            stage = st.stage
            if stage == "RUNNING":
                last = min(0.9, 0.2 + 0.6 * 0.5)
            elif stage == "COMPLETED":
                last = 0.97
            progress(last, desc=f"Job {stage}")
    result.wait()

    if holder.get("err"):
        raise gr.Error(f"Job failed: {holder['err']}")
    return gr.update(value=holder["path"]), "Done — .ply ready to download."


demo = gr.Interface(
    fn=convert,
    inputs=[
        gr.Video(label="Room video (.mp4)"),
        gr.Slider(100, 600, value=submitter.DEFAULT_FRAMES, step=50, label="Frames to extract"),
        gr.Slider(5000, 30000, value=submitter.DEFAULT_ITERS, step=1000, label="Training iterations"),
        gr.Dropdown(["l4x1", "a10g-large", "a100-large"], value=submitter.DEFAULT_FLAVOR, label="GPU flavor"),
        gr.Checkbox(label="Dry run (don't submit job)", value=False),
    ],
    outputs=[
        gr.File(label="Gaussian splat (.ply)"),
        gr.Textbox(label="Status", interactive=False),
    ],
    title="Video → Gaussian Splat (PLY) — HF Jobs",
    description=(
        "Upload a room walkthrough; the GPU work runs on Hugging Face managed"
        " hardware (billed per minute). Download the resulting .ply and open it"
        " in a Gaussian-splat viewer such as SuperSplat."
    ),
)

demo.queue(max_size=2)
demo.launch(server_name="0.0.0.0", server_port=7860)