import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path

# -----------------------------
# Global style
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# -----------------------------
# Event data (same for all panels)
# Each event: (x, t)
# -----------------------------
events = np.array([
    [3, 1],
    [4, 2],
    [5, 2],
    [5, 3],
    [6, 1],
    [6, 3],
    [7, 2],
    [7, 4],
    [8, 3],
    [8, 5],
], dtype=float)

velocities = [0.0, 1.0, 1.5]

# -----------------------------
# Projection rule
# b = x - v * t
# For the bottom histograms:
# - v = 0.0 and 1.0: use discrete integer positions
# - v = 1.5: keep all distinct projected positions (7 bars)
# -----------------------------
def project_points(events, v):
    x = events[:, 0]
    t = events[:, 1]
    return x - v * t

def discrete_positions_for_hist(proj, v):
    if np.isclose(v, 1.5):
        # Keep all unique projected positions exactly
        # This produces 7 bars: 0.5, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5
        vals = np.round(proj, 2)
        counter = Counter(vals)
    else:
        # For v=0 and v=1, use nearest integer locations
        vals = np.rint(proj).astype(int)
        counter = Counter(vals)

    xs = sorted(counter.keys())
    ys = [counter[k] for k in xs]
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

# -----------------------------
# Plot helpers
# -----------------------------
def format_top_axis(ax, title):
    ax.set_title(title, pad=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 2, 4, 6])
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.tick_params(width=1.4, length=6)

def format_bottom_axis(ax, xlabel=r"$b$"):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.tick_params(width=1.4, length=6)

# -----------------------------
# Create figure
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
plt.subplots_adjust(wspace=0.22, hspace=0.28)

bar_width = 0.36   # same width for all three histograms
variance_positions = np.arange(0, 10.5, 0.5)  # shared support, including empty bins
projection_line_style = dict(color="#18a118", linewidth=2.0,
                             linestyle=(0, (4, 3)), dash_capstyle="butt", zorder=2)

for col, v in enumerate(velocities):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    x = events[:, 0]
    t = events[:, 1]
    b = project_points(events, v)

    # ---- Top row: events + projections + arrows ----
    # Event points
    ax_top.scatter(x, t, s=140, color="#1f77b4", edgecolor="none", label="Event", zorder=3)

    # Projection points shown at t=0
    # To avoid too much overlap, plot unique projection locations
    if np.isclose(v, 1.5):
        proj_vis = sorted(set(np.round(b, 2)))
        proj_vis = np.array(proj_vis, dtype=float)
    else:
        proj_vis = np.unique(np.rint(b).astype(int)).astype(float)

    ax_top.scatter(proj_vis, np.zeros_like(proj_vis), s=140, color="#ff7f0e",
                   edgecolor="none", label="Projection", zorder=4)

    # Start each dashed line at its projection so overlapping lines share a phase.
    for xi, ti, bi in zip(x, t, b):
        if np.isclose(v, 1.5):
            bx = bi
        else:
            bx = np.rint(bi)

        ax_top.plot([bx, xi], [0, ti], **projection_line_style)
        # Draw only the arrowhead above the projection marker; keep the shaft dashed.
        arrow = ax_top.annotate(
            "", xy=(bx, 0), xytext=(xi, ti),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.65,head_width=0.3",
                color="#18a118", linewidth=0, mutation_scale=18,
                shrinkA=0, shrinkB=0,
            ),
            annotation_clip=False, zorder=5,
        )
        arrow.arrow_patch.set_clip_on(False)

    format_top_axis(ax_top, rf"$v \,=\, {v}$")
    ax_top.legend(loc="upper left", frameon=True, fancybox=True,
                  framealpha=0.92, borderpad=0.6, handlelength=1.2, handletextpad=0.5)

    # ---- Bottom row: histogram / bar counts ----
    bx, by = discrete_positions_for_hist(b, v)

    ax_bot.bar(bx, by, width=bar_width, color="#0072BD", edgecolor="#0072BD", alpha=0.95)

    # add count labels
    for xi, yi in zip(bx, by):
        ax_bot.text(xi, yi + 0.08, f"{int(yi)}", ha="center", va="bottom", fontsize=16)

    format_bottom_axis(ax_bot, xlabel=r"$b$")
    # Population variance of counts on the same grid for all three panels.
    counts = np.zeros_like(variance_positions)
    counts[np.rint(bx / 0.5).astype(int)] = by
    variance = np.var(counts)
    ax_bot.text(
        0.96, 0.94, rf"$\mathrm{{Var}} = {variance:.3f}$",
        transform=ax_bot.transAxes, ha="right", va="top", fontsize=18,
    )
    print(f"v={v:.1f}: variance={variance:.6f}")

# Optional: keep x-label of bottom-left and bottom-middle as b too
# If you want the first two to show x instead, replace with:
# axes[1, 0].set_xlabel(r"$x$")
# axes[1, 1].set_xlabel(r"$x$")

output_file = Path(__file__).resolve().parent / "discrete_IWE_vote.pdf"
fig.savefig(output_file, format="pdf", bbox_inches="tight")
print(f"Saved: {output_file}")
plt.show()
