#!/usr/bin/env python3
"""Watch a named HF job; when it finishes and a fresh splat.ply lands under
the mounted output dir, render multi-view snapshots and send them to WhatsApp.

Usage: watch_and_snap.py <job_id> [--channel ID] [--out-dir /docker/video-to-ply/outputs] [--interval 60]
"""
import argparse
import os
import time
import sys
from pathlib import Path

import snapshot_and_send as snap


def job_stage(api, job_id, namespace):
    st = api.inspect_job(job_id=job_id, namespace=namespace).status
    return st.stage, (st.message or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--out-dir", default="/docker/video-to-ply/outputs")
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--namespace", default=None)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    from submitter import default_namespace
    ns = args.namespace or default_namespace(api)
    chat = args.channel or os.environ.get("WHATSAPP_HOME_CHANNEL")
    if not chat:
        print("error: no --channel / WHATSAPP_HOME_CHANNEL"); return 1

    target = Path(args.out_dir) / "splat.ply"
    seen_size = None
    print(f"[watch] {args.job_id} -> poll; outputs at {target}", flush=True)

    while True:
        stage, msg = job_stage(api, args.job_id, ns)
        print(f"[watch] stage={stage}", flush=True)

        if stage in ("ERROR", "DELETED", "CANCELED"):
            print(f"[watch] job ended {stage}: {msg}"); return 2
        if stage == "COMPLETED":
            # wait until the container has downloaded + closed the file
            for _ in range(40):
                if target.exists():
                    sz = target.stat().st_size
                    if seen_size is None:
                        seen_size = sz
                    elif sz == seen_size and sz > 0:
                        time.sleep(3)
                        print(f"[watch] splat.ply stable ({sz} bytes) -> snapshotting", flush=True)
                        snap.main([str(target), "--channel", chat])
                        return 0
                    seen_size = sz
                time.sleep(5)
            print("[watch] completed but no stable splat.ply found"); return 3
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())