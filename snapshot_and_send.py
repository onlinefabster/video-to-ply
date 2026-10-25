#!/usr/bin/env python3
"""Render a splat .ply from several views and send the images to WhatsApp.

Uses the same CPU projection as render_thumb.py (numpy+matplotlib, no GPU).
Sends each view via the local Baileys bridge (localhost:3000 /send-media).
Usage: snapshot_and_send.py <splat.ply> [--channel <id>] [--out <dir>]
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load(path):
    v = PlyData.read(path)["vertex"]
    pos = np.column_stack([v["x"], v["y"], v["z"]])
    color = np.clip(SH_C0 * np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]) + 0.5, 0, 1)
    op = 1.0 / (1.0 + np.exp(-v["opacity"]))
    scale = np.exp(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]))
    size = scale.mean(axis=1)
    m = op > 0.15
    return pos[m], color[m], size[m]


def render(pos, color, size, out, W=1024, H=768, eye_f=(0.6, 0.15, 0.9), dist_f=2.0, f=1000.0):
    c = pos.mean(axis=0)
    diag = np.quantile(np.linalg.norm(pos - c, axis=1), 0.99)
    eye = np.array(eye_f) * dist_f * diag
    up = np.array([0.0, 0.0, 1.0])
    z = c - eye; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x); R = np.stack([x, y, z]); t = -R @ eye
    cam = pos @ R.T + t; depth = cam[:, 2]
    cx, cy = W / 2, H / 2
    nx = cam[:, 0] * f / depth + cx
    ny = cam[:, 1] * f / depth + cy
    r = size * f / depth
    keep = (depth > 0.1) & np.isfinite(nx) & np.isfinite(ny) & (r > 0.05) & (r < 500)
    nx, ny, depth, color, r = nx[keep], ny[keep], depth[keep], color[keep], r[keep]
    order = np.argsort(depth)[::-1]
    nx, ny, color, r = nx[order], ny[order], color[order], r[order]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_facecolor("black")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis()
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for i in range(0, len(nx), 40000):
        sl = slice(i, i + 40000)
        ax.scatter(nx[sl], ny[sl], s=(r[sl] * 1.2) ** 2, c=color[sl],
                   alpha=0.9, edgecolors="none", marker=".", rasterized=True)
    fig.savefig(out, dpi=100, facecolor="black")
    plt.close(fig)
    return out


def send_whatsapp(chat_id, file_path, caption):
    with open(file_path, "rb") as _:
        pass  # bridge reads the path; just validate existence
    body = json.dumps({
        "chatId": chat_id, "filePath": os.path.abspath(file_path),
        "mediaType": "image", "caption": caption,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:3000/send-media", data=body,
        headers={"Content-Type": "application/json", "Host": "localhost"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


VIEWS = [
    ("overview", (0.6, 0.15, 0.9), 2.0),
    ("front",   (1.0, 0.0, 0.55), 2.0),
    ("right",   (0.4, 0.9, 0.6),  2.0),
    ("close",   (0.4, 0.05, 0.5), 1.1),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--channel")
    ap.add_argument("--out", default="/tmp/snapshots")
    args = ap.parse_args(argv)

    pos, color, size = load(args.ply)
    os.makedirs(args.out, exist_ok=True)

    chat = args.channel or os.environ.get("WHATSAPP_HOME_CHANNEL")
    if not chat:
        print("error: no --channel and no WHATSAPP_HOME_CHANNEL")
        return 1

    sent = []
    for name, eye_f, dist_f in VIEWS:
        img = os.path.join(args.out, f"snapshot_{name}.png")
        render(pos, color, size, img, eye_f=eye_f, dist_f=dist_f)
        r = send_whatsapp(chat, img, f"video->ply view: {name}")
        ok = r.get("success") or bool(r.get("messageId"))
        print(f"[send] {name}: {'OK' if ok else r}")
        sent.append((name, ok))
    print("done. sent:", sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())