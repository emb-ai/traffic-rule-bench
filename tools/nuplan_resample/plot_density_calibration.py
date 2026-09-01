#!/usr/bin/env python3
"""traffic_density calibration plot: simulator response curve vs nuPlan targets."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEW = Path("/Users/victoria_s/sdc_new_signs/nuplan_statistics_v2")

r = json.loads((NEW / "density_calibration.json").read_text())
curve = r["sim_curve"]
ds = np.array(sorted(float(d) for d in curve))
p50 = np.array([curve[str(d)]["p50"] for d in ds])
p25 = np.array([curve[str(d)]["p25"] for d in ds])
p75 = np.array([curve[str(d)]["p75"] for d in ds])
resp = r["sim_curve_respawn_mode"]
rds = np.array(sorted(float(d) for d in resp))
rp50 = np.array([resp[str(d)]["p50"] for d in rds])

INK, MUT = "#1f2430", "#6b7280"
C_TRIG, C_RESP, C_TGT, C_LVL = "#2563eb", "#0e9f6e", "#8a8f98", "#d97706"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d7dade", "axes.grid": True, "grid.color": "#eceef1",
    "axes.axisbelow": True, "font.size": 10, "text.color": INK,
    "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT,
    "legend.frameon": False,
})
fig, ax = plt.subplots(figsize=(8.6, 5.4))
fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.11)

ax.fill_between(ds, p25, p75, color=C_TRIG, alpha=0.12, lw=0)
ax.plot(ds, p50, color=C_TRIG, lw=2, marker="o", ms=5,
        label="simulator, trigger mode (bench default)")
ax.plot(rds, rp50, color=C_RESP, lw=2, ls=(0, (4, 2)), marker="s", ms=5,
        label="simulator, respawn mode")

for tgt, name in [(5, "nuPlan p25"), (11, "nuPlan p50"), (18, "nuPlan p75")]:
    ax.axhline(tgt, color=C_TGT, lw=1.1, ls=(0, (3, 2)))
    ax.text(0.615, tgt + 0.3, f"{name} = {tgt}", color="#5c6067", fontsize=9,
            ha="right")

levels = [(0.1, 1), (0.4, 3), (0.5, 5)]
ax.scatter([d for d, _ in levels], [n for _, n in levels], s=90, color=C_LVL,
           zorder=5, label="low / medium / high levels")

ax.set_xlabel("MetaDrive traffic_density")
ax.set_ylabel("moving vehicles within 50 m of ego (median)")
ax.set_xlim(0.03, 0.62)
ax.set_ylim(0, 20.8)
ax.set_title("traffic_density calibration: simulator response vs nuPlan targets",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)

out = NEW / "plots" / "density_calibration.png"
fig.savefig(out, dpi=170)
print("saved:", out)
