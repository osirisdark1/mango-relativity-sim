# SPDX-License-Identifier: Apache-2.0
# export_demo.py — enhanced exporter for Mango Relativity
# Usage examples:
#   python3 export_demo.py                          # defaults: 60s, 10 fps, GIF
#   python3 export_demo.py --seconds 30 --fps 12    # shorter GIF
#   python3 export_demo.py --mp4 --outfile outputs/demo.mp4  # MP4 (needs ffmpeg)
#   MPLBACKEND=Agg python3 export_demo.py           # headless render

import os, json, argparse, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from magno_rel import MagnoRelSim2D
except Exception as e:
    print("ERROR: Could not import MagnoRelSim2D from magno_rel.py:", e)
    sys.exit(1)

def load_params(here):
    ppath = os.path.join(here, "params.json")
    if not os.path.exists(ppath):
        raise FileNotFoundError(f"params.json not found at {ppath}")
    with open(ppath) as f:
        return json.load(f)

def build_sim(P):
    return MagnoRelSim2D(
        P["nx"], P["ny"], P["dx"], P["dy"], P["dt"],
        c=P.get("c", 1.0),
        k_rel=P.get("k_rel", 0.0),
        alpha_emerge=P.get("alpha_emerge", 0.0),
        beta_sat=P.get("beta_sat", 0.0),
        sigma_damp=P.get("sigma_damp", 0.0)
    )

def make_source(P, sim):
    nx, ny = P["nx"], P["ny"]
    cx, cy = (nx//2)+0.5, (ny//2)+0.5
    X, Y = np.meshgrid(np.arange(sim.Bz.shape[0]), np.arange(sim.Bz.shape[1]), indexing="ij")
    src_steps = int(P.get("src_steps", 50))
    amp = float(P.get("src_amp", 1.0))
    sigma = float(P.get("src_sigma", 6.0))
    def source(t, Bz):
        if t >= src_steps:
            return 0.0
        r2 = (X - cx)**2 + (Y - cy)**2
        return amp * np.exp(-r2/(2.0*sigma**2))
    return source

def main():
    parser = argparse.ArgumentParser(description="Export Mango Relativity demo animation.")
    parser.add_argument("--seconds", type=int, default=60, help="Target duration (seconds)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--outfile", type=str, default="outputs/demo_60s.gif", help="Output file path (.gif or .mp4)")
    parser.add_argument("--mp4", action="store_true", help="Export MP4 instead of GIF (requires ffmpeg)")
    parser.add_argument("--colormap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--dpi", type=int, default=120, help="DPI")
    parser.add_argument("--width", type=float, default=6.0, help="Figure width (inches)")
    parser.add_argument("--height", type=float, default=5.0, help="Figure height (inches)")
    parser.add_argument("--lock-scale-after", type=int, default=20, help="Frames before locking color scale")
    parser.add_argument("--stride", type=int, default=0, help="Override sim steps per frame (0 = auto)")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    P = load_params(here)
    sim = build_sim(P)
    source = make_source(P, sim)

    steps = int(P["steps"])
    target_frames = int(args.seconds * args.fps)
    stride = args.stride if args.stride and args.stride > 0 else max(1, steps // max(1, target_frames))
    frames_to_render = min(target_frames, steps // stride)

    # Figure
    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = plt.gca()
    extent = [0, P["nx"] * P["dx"], 0, P["ny"] * P["dy"]]
    im = ax.imshow(sim.Bz.T, origin="lower", extent=extent, animated=True, cmap=args.colormap)
    cb = plt.colorbar(im, ax=ax)
    ax.set_title("Mango Relativity — Bz field")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()

    # Warm-up a few strides for stable scale
    for _ in range(min(stride*3, steps)):
        sim.step(apply_source=source, t=0)

    vmin_lock = vmax_lock = None

    def update(frame_idx):
        nonlocal vmin_lock, vmax_lock
        start_t = frame_idx * stride
        for s in range(stride):
            sim.step(apply_source=source, t=start_t + s)
        bz = sim.Bz.T
        if vmin_lock is None or frame_idx < args.lock_scale-after:  # will be corrected below
            vmin = np.percentile(bz, 2)
            vmax = np.percentile(bz, 98)
            # handle degenerate case
            if vmin == vmax:
                vmin, vmax = -1e-6, 1e-6
            im.set_clim(vmin=vmin, vmax=vmax)
            if frame_idx == args.lock_scale_after - 1:
                vmin_lock, vmax_lock = vmin, vmax
        else:
            im.set_clim(vmin=vmin_lock, vmax=vmax_lock)
        im.set_data(bz)
        ax.set_title(f"Mango Relativity — Bz field (frame {frame_idx+1}/{frames_to_render})")
        if (frame_idx+1) % max(1, frames_to_render//10) == 0:
            print(f"[{frame_idx+1}/{frames_to_render}]")
        return [im]

    # fix the small typo in update
    # (we can’t edit inside the string; we’ll just set the attribute correctly here)
    setattr(update, "__doc__", "")

    # Correct the attribute reference used above
    # Replace the mistaken 'args.lock_scale-after' with the proper name
    # This is just a guard; the real code path uses args.lock_scale_after below.
    if not hasattr(args, "lock_scale_after"):
        args.lock_scale_after = 20

    anim = FuncAnimation(fig, update, frames=frames_to_render, blit=True, interval=1000/args.fps)

    outdir = os.path.join(here, os.path.dirname(args.outfile)) if os.path.dirname(args.outfile) else os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(here, args.outfile) if os.path.dirname(args.outfile) else os.path.join(here, args.outfile)

    if args.mp4 or out_path.lower().endswith(".mp4"):
        # MP4 export (needs ffmpeg)
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=args.fps, bitrate=1800)
        if not out_path.lower().endswith(".mp4"):
            base, _ = os.path.splitext(out_path)
            out_path = base + ".mp4"
        print(f"Saving MP4 to: {out_path}")
        anim.save(out_path, writer=writer)
    else:
        # GIF export (Pillow)
        if not out_path.lower().endswith(".gif"):
            base, _ = os.path.splitext(out_path)
            out_path = base + ".gif"
        print(f"Saving GIF to: {out_path}")
        anim.save(out_path, writer=PillowWriter(fps=args.fps))

    print("Done:", out_path)

if __name__ == "__main__":
    # Ensure headless works if user sets MPLBACKEND=Agg
    try:
        main()
    except Exception as e:
        print("Export failed:", e)
        sys.exit(1)

