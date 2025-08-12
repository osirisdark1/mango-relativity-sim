import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from magno_rel import MagnoRelSim2D

here = os.path.dirname(__file__)
with open(os.path.join(here, "params.json")) as f:
    P = json.load(f)

nx, ny = P["nx"], P["ny"]
dx, dy = P["dx"], P["dy"]
dt = P["dt"]
steps = P["steps"]

sim = MagnoRelSim2D(nx, ny, dx, dy, dt,
    c=P["c"],
    k_rel=P["k_rel"],
    alpha_emerge=P["alpha_emerge"],
    beta_sat=P["beta_sat"],
    sigma_damp=P["sigma_damp"]
)

cx, cy = (nx//2)+0.5, (ny//2)+0.5
X, Y = np.meshgrid(np.arange(sim.Bz.shape[0]), np.arange(sim.Bz.shape[1]), indexing="ij")
def source(t, Bz):
    if t >= P["src_steps"]:
        return 0.0
    r2 = (X - cx)**2 + (Y - cy)**2
    g = P["src_amp"] * np.exp(-r2/(2.0*P["src_sigma"]**2))
    return g

target_seconds = 60
target_fps = 10
target_frames = target_seconds * target_fps
stride = max(1, steps // target_frames)
frames_to_render = min(target_frames, steps // stride)

fig = plt.figure(figsize=(6,5), dpi=120)
ax = plt.gca()
im = ax.imshow(sim.Bz.T, origin="lower", extent=[0, nx*dx, 0, ny*dy], animated=True)
plt.colorbar(im, ax=ax)
ax.set_title("Mango Relativity — Bz field")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

for _ in range(stride):
    sim.step(apply_source=source, t=0)

def update(frame_idx):
    start_t = frame_idx * stride
    for s in range(stride):
        sim.step(apply_source=source, t=start_t + s)
    bz = sim.Bz.T
    vmin = np.percentile(bz, 2)
    vmax = np.percentile(bz, 98)
    if vmin == vmax:
        vmin, vmax = -1e-6, 1e-6
    im.set_data(bz)
    im.set_clim(vmin=vmin, vmax=vmax)
    ax.set_title(f"Mango Relativity — Bz field (frame {frame_idx+1}/{frames_to_render})")
    return [im]

anim = FuncAnimation(fig, update, frames=frames_to_render, blit=True, interval=1000/target_fps)

outdir = os.path.join(here, "outputs")
os.makedirs(outdir, exist_ok=True)
gif_path = os.path.join(outdir, "demo_60s.gif")
anim.save(gif_path, writer=PillowWriter(fps=target_fps))
print("Saved:", gif_path)
