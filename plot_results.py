"""
plot_results.py — final-accuracy lollipop with the heterogeneity gap.

One row per federated method, a dot at its final global test accuracy,
a dashed line for the centralised upper bound, and a bracket that
measures the FedAvg even-vs-by-person gap (the headline finding).
Zoomed x-axis so the small but real differences are actually visible.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager

CENTRALISED = 0.9294
RESULTS = Path("results")

INK   = "#3A3A3A"
GREY  = "#B4ADA3"
GRID  = "#ECE7DF"
CREAM = "#FBF9F6"

# each row: (json file, label, colour, y-position)
rows = [
    ("fedavg_even.json",       "FedAvg · even split",         "#5B8A72", 3),
    ("fedavg_by_person.json",  "FedAvg · by person",          "#C97B63", 2),
    ("fedprox_by_person.json", "FedProx · by person  (μ=0.1)", "#8B7BA8", 1),
]

for pref in ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]:
    if any(pref in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = pref
        break

vals = {}
for filename, _, _, _ in rows:
    vals[filename] = json.loads((RESULTS / filename).read_text())["history"][-1]

fig, ax = plt.subplots(figsize=(9, 4.6))
fig.patch.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

BASE = 0.90  # left end of the stems

# centralised upper bound reference
ax.axvline(CENTRALISED, linestyle=(0, (5, 4)), color=GREY, linewidth=1.6, zorder=1)
ax.text(CENTRALISED, 3.7, f"centralised upper bound  {CENTRALISED:.3f}",
        ha="center", va="bottom", fontsize=9.5, color=GREY, style="italic")

# lollipops
for filename, label, colour, y in rows:
    v = vals[filename]
    ax.plot([BASE, v], [y, y], color=colour, linewidth=2.4,
            solid_capstyle="round", zorder=2, alpha=0.9)
    ax.scatter(v, y, s=210, color=colour, zorder=3,
               edgecolor=CREAM, linewidth=2)
    ax.text(v + 0.0012, y, f"{v:.3f}", va="center", ha="left",
            fontsize=12, color=colour, fontweight="bold")

# the headline: heterogeneity gap between the two FedAvg runs
even = vals["fedavg_even.json"]
byp  = vals["fedavg_by_person.json"]
gap_pts = (even - byp) * 100
ax.annotate("", xy=(even, 2.5), xytext=(byp, 2.5),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text((even + byp) / 2, 2.62,
        f"heterogeneity gap ≈ {gap_pts:.1f} pts",
        ha="center", va="bottom", fontsize=9.5, color=INK)

# axes styling
ax.set_yticks([r[3] for r in rows])
ax.set_yticklabels([r[1] for r in rows], fontsize=11, color=INK)
ax.set_ylim(0.4, 4.1)
ax.set_xlim(BASE, 0.955)
ax.set_xticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
ax.set_xticklabels([f"{t:.2f}" for t in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]],
                   fontsize=10, color=INK)
ax.set_xlabel("Final global test accuracy  (after 50 rounds)",
              fontsize=11, color=INK, labelpad=8)

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(length=0)
ax.grid(axis="x", color=GRID, linewidth=1)
ax.set_axisbelow(True)

ax.set_title("Federated HAR: where each method lands",
             fontsize=14, color=INK, fontweight="bold", loc="left", pad=44)
ax.text(BASE, 4.35, "UCI HAR · 21 clients · axis zoomed to 0.90–0.95",
        fontsize=9.5, color=GREY)

plt.tight_layout()
out = RESULTS / "comparison.png"
plt.savefig(out, dpi=200, facecolor=CREAM)
print("saved", out)