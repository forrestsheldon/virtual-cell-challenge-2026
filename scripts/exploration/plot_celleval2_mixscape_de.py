"""Does Mixscape responder filtering sharpen the DE signal, or just shrink the sample?

Three arms per target: all labelled cells, the Mixscape KO-like subset, and a
size-matched random subset. The matched arm is the one that makes the comparison
readable -- it holds cell count fixed so the KO arm shows selection alone.

Palette is the single-hue blue sequential ramp (responder fraction is a magnitude),
with neutrals for reference lines and the matched arm.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
BLUE = LinearSegmentedColormap.from_list("blue", RAMP)
ACCENT, NEUTRAL, LINE = "#256abf", "#6b6a66", "#c3cbd6"
INK, MUTED = "#131820", "#555f6d"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
args = parser.parse_args()

s = pd.read_csv(args.output / "celleval2_mixscape_de_summary.csv")
s = s[s.n_post > 0].copy()
print(f"{len(s)} targets with KO-like cells")

figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)

# 1. KO against the size-matched random arm: does selection beat losing cells?
limit = max(s.matched_de.max(), s.post_de.max()) * 1.4
axes[0].plot([1, limit], [1, limit], color=LINE, linewidth=1, zorder=1)
points = axes[0].scatter(s.matched_de + 1, s.post_de + 1, c=s.responder_fraction, cmap=BLUE,
                         s=28, vmin=0.3, vmax=1, zorder=2, edgecolors=NEUTRAL, linewidths=0.35)
axes[0].set(xscale="log", yscale="log", xlim=(0.8, limit), ylim=(0.8, limit),
            xlabel="size-matched random cells, DE genes + 1",
            ylabel="Mixscape KO-like cells, DE genes + 1",
            title="Selection, with cell count held fixed")
figure.colorbar(points, ax=axes[0], label="Mixscape KO-like fraction")

# 2. The gain over the matched arm, against how much was filtered away.
gain = np.log2((s.post_de + 1) / (s.matched_de + 1))
axes[1].axhline(0, color=LINE, linewidth=1)
axes[1].scatter(s.responder_fraction, gain, s=28, color=ACCENT, alpha=0.8,
                edgecolors=NEUTRAL, linewidths=0.35)
axes[1].set(xlabel="Mixscape KO-like fraction", ylabel="log2(KO DE / matched DE)",
            title="Gain is largest where most cells are filtered")

# 3. Set membership against the full-cell test.
axes[2].scatter(s.responder_fraction, s.jaccard, s=28, color=ACCENT, alpha=0.8,
                label="KO-like", edgecolors=NEUTRAL, linewidths=0.35)
axes[2].scatter(s.responder_fraction, s.jaccard_matched, s=28, color=NEUTRAL, alpha=0.5,
                label="size-matched random", edgecolors=NEUTRAL, linewidths=0.35)
axes[2].set(xlabel="Mixscape KO-like fraction", ylabel="Jaccard with the all-cell DE set",
            title="Membership overlap with the full test", ylim=(0, 1))
axes[2].legend(frameon=False, loc="lower right", labelcolor=MUTED)

for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(MUTED), axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(INK)
figure.suptitle("Differential expression before and after Mixscape responder filtering, cell-eval2 universe",
                fontsize=13, color=INK)
figure.savefig(args.output / "celleval2_mixscape_de_comparison.png", dpi=200, facecolor="#fcfcfb")
print(f"wrote {args.output / 'celleval2_mixscape_de_comparison.png'}")

print(f"\nKO beats size-matched random in {(s.post_de > s.matched_de).sum()} of {len(s)} targets")
print(f"median log2 gain over matched: {gain.median():.3f}")
print(f"median Jaccard with all-cell set: KO {s.jaccard.median():.3f}, matched {s.jaccard_matched.median():.3f}")
