#!/usr/bin/env python3
"""CPU thumbnail render of a Nerfstudio Gaussian-splat PLY (no GPU needed).

Projects splats into a synthetic perspective camera, sizes them by their
3D scale, depth-orders them, and rasterizes with matplotlib. Produces a PNG
"thumbnail" so the scene can be sanity-checked without CUDA.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def project(pts, R, t, f, cx, cy):
    """Perspective-project world points -> (nx, ny, depth). R rotates world->cam."""
    cam = pts @ R.T + t
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = cam[:, 0] * f / depth + cx
        ny = cam[:, 1] * f / depth + cy
    return nx, ny, depth


def main(path, out="thumbnail.png", W=1024, H=768, f=1000.0, view_angle=None,
         eye_f=(0.6, 0.15, 0.9), dist_f=2.2):
    pd = PlyData.read(path)
    v = pd["vertex"]
    pos = np.column_stack([v["x"], v["y"], v["z"]])
    color = np.clip(SH_C0 * np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]) + 0.5, 0, 1)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"]))
    scale = np.exp(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]))
    # mean projected size proxy: mean of 3 scales
    size = scale.mean(axis=1)

    # discard near-invisible splats to speed up
    m = opacity > 0.15
    pos, color, size = pos[m], color[m], size[m]

    # center camera on the point-cloud centroid
    c = pos.mean(axis=0)
    # view direction + distance to frame the whole scene
    diag = np.quantile(np.linalg.norm(pos - c, axis=1), 0.99)
    dist = 2.2 * diag

    # build a camera: world +Z is up for nerfstudio (vertical axis z)
    eye = np.array(eye_f) * dist_f * dist
    cam_center = c
    up = np.array([0.0, 0.0, 1.0])
    z = cam_center - eye
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ eye

    cx, cy = W / 2, H / 2
    if view_angle:
        f = (H / 2) / np.tan(np.deg2rad(view_angle) / 2)
    nx, ny, depth = project(pos, R, t, f, cx, cy)

    # keep points in front of camera and inside bounds (with margin)
    sane = (depth > 0.1) & np.isfinite(nx) & np.isfinite(ny)
    # a splat that projects large: radius in pixels ~ (size/depth)*f
    r_px = size * f / depth
    keep = sane & (r_px > 0.05) & (r_px < 500)
    nx, ny, depth, color, r_px = nx[keep], ny[keep], depth[keep], color[keep], r_px[keep]

    # depth-sort far -> near so front splats paint last
    order = np.argsort(depth)[::-1]
    nx, ny, color, r_px = nx[order], ny[order], color[order], r_px[order]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_facecolor("black")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # scatter respects call order for overdraw: draw far->near in chunks.
    max_pts = 40000
    for i in range(0, len(nx), max_pts):
        sl = slice(i, i + max_pts)
        ax.scatter(
            nx[sl], ny[sl], s=(r_px[sl] * 1.2) ** 2,
            c=color[sl], alpha=0.9, edgecolors="none", marker=".", rasterized=True,
        )
    fig.savefig(out, dpi=100, facecolor="black")
    plt.close(fig)
    print(f"rendered {out} ({len(order)} splats projected)")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/splat.ply",
         sys.argv[2] if len(sys.argv) > 2 else "thumbnail.png")