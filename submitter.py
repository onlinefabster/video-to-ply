"""
Submit a video->PLY (Gaussian splat) job to Hugging Face Jobs and fetch the
resulting .ply. Adapted from pollen-robotics/microduck_rl's hf_jobs.py submitter.

Flow: upload video to a private dataset repo (input volume) -> run_job on a
GPU container running bootstrap.sh -> supervise scheduling (resubmit on flaky
mount) -> poll to COMPLETED -> download splat.ply from the private model repo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token
from huggingface_hub.errors import EntryNotFoundError

DEFAULT_IMAGE = "ghcr.io/nerfstudio-project/nerfstudio:latest"
DEFAULT_FLAVOR = "l4x1"
DEFAULT_TIMEOUT = "12h"
DEFAULT_FRAMES = 300
DEFAULT_ITERS = 15000

# The in-container pipeline. Kept as a separate file is avoidable since script
# and bootstrap.sh should stay in sync; we read it from disk at import time so
# bootstrap.sh is a single source of truth.
_HERE = Path(__file__).resolve().parent
BOOTSTRAP_SCRIPT = (_HERE / "bootstrap.sh").read_text()

# Repos/bucket naming --------------------------------------------------------

def default_namespace(api: HfApi) -> str:
    info = api.whoami()
    return info.get("name") or info.get("email")


def make_run_name(prefix: str = "video2ply") -> str:
    return f"{prefix}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}".lower()


# Auth / validation ----------------------------------------------------------

def require_token() -> str:
    token = get_token()
    if not token:
        print(
            "error: no cached HF token. Run `hf auth login` or export HF_TOKEN.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def _friendly_job_error(e: Exception, namespace: str) -> None:
    msg = str(e)
    if "402" in msg or "Payment Required" in msg or "credit" in msg.lower():
        print(
            "\n[job] ✗ Hugging Face Jobs billing error.\n"
            f"  Namespace {namespace!r} has insufficient Jobs credits.\n"
            "  → Add credits: https://huggingface.co/settings/billing\n"
            "  → Or get HF Pro: https://huggingface.co/settings/billing/subscription",
            file=sys.stderr,
        )
        sys.exit(2)
    if "403" in msg or "Forbidden" in msg or "required permissions" in msg.lower():
        print(
            "\n[job] ✗ Hugging Face Jobs permission error (403).\n"
            "  Your token authenticates but is NOT allowed to use the Jobs API.\n"
            "  → Create/edit a fine-grained token WITH the Jobs permission:\n"
            "      https://huggingface.co/settings/tokens\n"
            "  then re-login locally:  hf auth login",
            file=sys.stderr,
        )
        sys.exit(3)
    raise


# Video upload ---------------------------------------------------------------

def upload_video(api: HfApi, namespace: str, video_path: str, src_repo: str | None = None) -> tuple[str, str]:
    """Upload the video to a private dataset repo. Returns (src_repo, filename)."""
    src_repo = src_repo or f"{namespace}/video2ply-src"
    filename = os.path.basename(video_path)
    api.create_repo(src_repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"[src] uploading {filename} -> dataset {src_repo}")
    api.upload_file(
        path_or_fileobj=video_path,
        path_in_repo=filename,
        repo_id=src_repo,
        repo_type="dataset",
    )
    return src_repo, filename


# Scheduling supervision -----------------------------------------------------

def _await_scheduling(api: HfApi, job_id: str, namespace: str, budget_s: float = 1200.0) -> tuple[str, str | None]:
    """Poll until the job leaves SCHEDULING (image pull / queue / mounts)."""
    deadline = time.monotonic() + budget_s
    last_note = time.monotonic()
    stage, message = "", None
    while time.monotonic() < deadline:
        status = api.inspect_job(job_id=job_id, namespace=namespace).status
        stage, message = status.stage, status.message
        if stage and stage != "SCHEDULING":
            return stage, message
        if time.monotonic() - last_note > 60:
            print("[job] still scheduling (queue / image pull / volume mounts)...")
            last_note = time.monotonic()
        time.sleep(10)
    return stage, message


# Job submission -------------------------------------------------------------

def submit(
    api: HfApi,
    namespace: str,
    src_repo: str,
    video_file: str,
    ckpt_repo: str,
    *,
    frames: int = DEFAULT_FRAMES,
    iterations: int = DEFAULT_ITERS,
    flavor: str = DEFAULT_FLAVOR,
    image: str = DEFAULT_IMAGE,
    timeout: str = DEFAULT_TIMEOUT,
) -> str:
    """Submit the job and return its id. Supervises scheduling with resubmit."""
    volumes = [
        Volume(type="dataset", source=src_repo, mount_path="/src", read_only=True)
    ]
    env = {
        "VIDEO_FILE": video_file,
        "NUM_FRAMES": str(frames),
        "ITERATIONS": str(iterations),
        "CKPT_REPO": ckpt_repo,
    }
    secrets = {"HF_TOKEN": require_token()}

    print(f"[job] submitting (namespace={namespace}, flavor={flavor}, timeout={timeout})")
    print(f"[ckpt] .ply will land in https://huggingface.co/{ckpt_repo}")

    job = None
    stage, message = "", None
    for attempt in range(3):
        if attempt:
            print(f"[job] ✗ volume mount failed (flaky node) — resubmitting ({attempt + 1}/3)")
            time.sleep(10)
        try:
            job = api.run_job(
                image=image,
                command=["bash", "-c", BOOTSTRAP_SCRIPT],
                env=env,
                secrets=secrets,
                flavor=flavor,
                timeout=timeout,
                volumes=volumes,
                namespace=namespace,
            )
        except Exception as e:  # noqa: BLE001
            _friendly_job_error(e, namespace)
        print(f"[job] id: {job.id}  url: {getattr(job, 'url', '')}")
        stage, message = _await_scheduling(api, job.id, namespace)
        if stage == "ERROR" and "mount" in (message or "").lower():
            continue  # flaky node — resubmit
        break

    if stage == "ERROR":
        print(f"[job] ✗ failed to start: {message}", file=sys.stderr)
        sys.exit(1)
    return job.id


# Wait + download ------------------------------------------------------------

def wait_and_download(
    api: HfApi,
    job_id: str,
    namespace: str,
    ckpt_repo: str,
    out: str = str(_HERE / "outputs"),
) -> str:
    """Poll the job to completion, then download splat.ply. Returns local path."""
    os.makedirs(out, exist_ok=True)
    local = os.path.join(out, "splat.ply")

    print("[job] streaming logs (Ctrl-C detaches; the job keeps running)")
    try:
        while True:
            try:
                for line in api.fetch_job_logs(job_id=job_id, namespace=namespace, follow=True):
                    print(line)
            except Exception as e:  # noqa: BLE001
                print(f"[job] log stream dropped ({e}); re-attaching")
            status = api.inspect_job(job_id=job_id, namespace=namespace).status
            if status.stage == "COMPLETED":
                print("[job] ✓ completed")
                break
            if status.stage in ("ERROR", "DELETED", "CANCELED"):
                raise RuntimeError(f"job {status.stage}: {status.message}")
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n[job] detached. Job {job_id} still running: https://huggingface.co/")
        sys.exit(0)

    try:
        api.hf_hub_download(
            repo_id=ckpt_repo, repo_type="model",
            filename="splat.ply", local_dir=out,
        )
    except EntryNotFoundError:
        print(
            "\n[result] ✗ job completed but no splat.ply found (timed out or export"
            " failed before uploading). Check the job logs.",
            file=sys.stderr,
        )
        sys.exit(1)
    return local


# CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submit a video->PLY job to HF Jobs")
    ap.add_argument("video", help="path to the input .mp4")
    ap.add_argument("--namespace", default=None, help="HF account/org to run under")
    ap.add_argument("--flavor", default=DEFAULT_FLAVOR)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default=str(_HERE / "outputs"))
    ap.add_argument(
        "--dry-run", action="store_true",
        help="build the spec and print it, do not submit",
    )
    args = ap.parse_args(argv)

    api = HfApi()
    namespace = args.namespace or default_namespace(api)
    if not namespace:
        print("error: could not determine HF username. Run `hf auth login`.", file=sys.stderr)
        return 1
    if not args.dry_run:
        require_token()

    run_name = args.run_name or make_run_name()
    ckpt_repo = f"{namespace}/{run_name}"
    src_repo, video_file = upload_video(api, namespace, args.video)

    if args.dry_run:
        print("[dry-run] would submit job:")
        print(f"  video:       {args.video} -> dataset {src_repo}")
        print(f"  image:       {args.image}")
        print(f"  flavor:      {args.flavor}  timeout: {args.timeout}")
        print(f"  frames:      {args.frames}  iterations: {args.iterations}")
        print(f"  ckpt repo:   https://huggingface.co/{ckpt_repo}")
        print("  cost: billed per Start+Run minute; timeout is a DURATION cap, not spend.")
        return 0

    job_id = submit(
        api, namespace, src_repo, video_file, ckpt_repo,
        frames=args.frames, iterations=args.iterations,
        flavor=args.flavor, image=args.image, timeout=args.timeout,
    )
    local = wait_and_download(api, job_id, namespace, ckpt_repo, out=args.out)
    print(f"\n[result] PLY saved to: {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())