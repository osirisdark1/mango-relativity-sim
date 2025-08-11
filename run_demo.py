import json, os
import numpy as np
import matplotlib.pyplot as plt
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

for t in range(steps):
    sim.step(apply_source=source, t=t)

outdir = os.path.join(here, "outputs")
os.makedirs(outdir, exist_ok=True)

plt.figure()
plt.imshow(sim.Bz.T, origin="lower", extent=[0, nx*dx, 0, ny*dy])
plt.colorbar()
plt.title("Final Bz")
plt.savefig(os.path.join(outdir, "field_B_final.png"), dpi=160, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(sim.energy_hist)
plt.xlabel("Step")
plt.ylabel("Energy (arb)")
plt.title("Total Energy vs Time")
plt.savefig(os.path.join(outdir, "energy_timeseries.png"), dpi=160, bbox_inches="tight")
plt.close()

summary = {
    "params": P,
    "final_energy": sim.energy_hist[-1] if sim.energy_hist else None,
    "steps": steps
}
with open(os.path.join(outdir, "run_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Done. Outputs in:", outdir)
