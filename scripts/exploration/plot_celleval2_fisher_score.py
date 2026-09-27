"""Illustrating the responder/non-responder mixture within a perturbed population.

Simplified successor to fisher_h1_directions.py's figure: project cells onto a
held-out, perturbation-specific direction and compare Control vs Perturbed score
distributions, computed in the cell-eval2 universe with a small capped marker set
(celleval2_h1_fisher_score.py). The direction always excludes the target's own
gene (markers_by_target drops it), so the score measures the downstream response,
not the knockdown of the targeted transcript.

Layout: two representative targets (rows) for each of the three DE-count bins
(columns). Representatives are the two targets closest to their bin's median
responder fraction among those with >= MIN_CELLS cells -- enough to render a smooth
histogram, and central rather than cherry-picked. Chosen by DE-count bin rather
than by responder fraction, so the columns are the same negligible/subtle/strong
partition as the rest of the post; the responder fractions come out roughly
monotonic across bins anyway (a consequence, not an input).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTROL, LINE, INK, MUTED = "#6b6a66", "#c3cbd6", "#131820", "#555f6d"
BIN_COLOR = {"negligible": "#6da7ec", "subtle": "#256abf", "strong": "#0d366b"}
MIN_CELLS = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--xmax", type=float, help="override the upper x-limit (in control SD units); "
                    "keeps the lower limit and the bin width, writes a _to<xmax>sd variant")
args = parser.parse_args()

scores = pd.read_parquet(args.output / "celleval2_fisher_scores.parquet")
summary = pd.read_csv(args.output / "celleval2_fisher_summary.csv").set_index("target")
bins = pd.read_csv(args.output / "celleval2_response_bins.csv").set_index("target")["bin"]
summary["bin"] = bins
summary["responder_fraction"] = 1 - summary.mean_control_like_fraction

# Two representatives per bin: closest to the bin's median responder fraction,
# among targets with enough cells for a smooth histogram.
picks = {}
for name in ["negligible", "subtle", "strong"]:
    eligible = summary[(summary["bin"] == name) & (summary.n_target >= MIN_CELLS)]
    median = eligible.responder_fraction.median()
    picks[name] = (eligible.responder_fraction - median).abs().nsmallest(2).index.tolist()
    print(f"{name}: {picks[name]}")


def panel(axis, target, colour, bins_):
    data = scores[scores.target == target]
    perturbed = data.cell_group == "Perturbed"
    axis.hist(data.loc[~perturbed, "score"], bins=bins_, density=True, color=CONTROL, alpha=0.6, label="Control")
    axis.hist(data.loc[perturbed, "score"], bins=bins_, density=True, color=colour, alpha=0.6, label="Perturbed")
    threshold = data.loc[~perturbed].groupby("fold", observed=True).score.quantile(0.95).median()
    axis.axvline(threshold, color=INK, linestyle="--", linewidth=1.1)
    fraction = summary.loc[target, "responder_fraction"]
    auc = summary.loc[target, "mean_roc_auc"]
    axis.set_title(f"{target}\n{fraction:.0%} responder-like · AUC {auc:.2f}", fontsize=10.5, color=INK)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.set_yticks([])


low, high = scores.score.quantile([0.005, 0.995]).to_numpy()
if args.xmax is None:
    edges = np.linspace(low, high, 70)
else:
    # Keep the default bin width so bars look the same, just extend the range.
    width = (high - low) / 69
    edges = np.linspace(low, args.xmax, round((args.xmax - low) / width) + 1)
figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True)

for col, name in enumerate(["negligible", "subtle", "strong"]):
    axes[0, col].text(0.5, 1.34, name, transform=axes[0, col].transAxes,
                      ha="center", fontsize=13, weight="bold", color=BIN_COLOR[name])
    for row, target in enumerate(picks[name]):
        panel(axes[row, col], target, BIN_COLOR[name], edges)

for axis in axes[-1]:
    axis.set_xlabel("held-out responder score", fontsize=9.5, color=MUTED)
for axis in axes[:, 0]:
    axis.set_ylabel("cell density", fontsize=9.5, color=MUTED)

handles = [
    plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=CONTROL, alpha=0.6, label="control cells"),
    plt.Line2D([], [], marker="s", linestyle="", markersize=9, color="#3d6da3", alpha=0.6, label="labelled perturbed cells"),
    plt.Line2D([], [], linestyle="--", color=INK, linewidth=1.1, label="control 95th percentile"),
]
figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
              labelcolor=MUTED, bbox_to_anchor=(0.5, -0.02))
figure.suptitle("Responders and non-responders within a perturbed population", fontsize=13.5, color=INK, y=1.0)
figure.tight_layout(rect=(0, 0.03, 1, 0.97))
name = "celleval2_fisher_score_distributions"
if args.xmax is not None:
    name += f"_to{int(args.xmax)}sd"
figure.savefig(args.output / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
print(f"wrote {name}.png")
