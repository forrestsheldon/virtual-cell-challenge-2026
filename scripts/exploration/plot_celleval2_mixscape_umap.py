"""Mixscape classes on the UMAP, split by the same response bins.

Companion to plot_celleval2_response_bins.py: same panels, same axes, but each
perturbed cell is now coloured by whether Mixscape called it KO-like or
non-perturbed. The question the figure answers is whether the negligible bin looks
control-like because the perturbation did nothing, or because it is a mixture in
which few cells responded.

Palette reuses the validated ordinal blue ramp -- non-perturbed to KO-like is a
magnitude -- with the same two neutrals for controls and backdrop.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NP, KO = "#6da7ec", "#0d366b"
CONTROL, BACKDROP = "#6b6a66", "#d6d5d1"
INK, MUTED = "#131820", "#555f6d"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
args = parser.parse_args()

bins = pd.read_csv(args.output / "celleval2_response_bins.csv")
calls = pd.read_parquet(args.output / "mixscape_cells.parquet", columns=["cell_id", "mixscape_class_global"])
embedding = pd.read_parquet(args.output / "embedding.parquet", columns=["cell_id", "target_gene", "UMAP1", "UMAP2"])
data = embedding.merge(calls, on="cell_id", how="left")
data["bin"] = data.target_gene.map(dict(zip(bins.target, bins["bin"])))

control = data[data.target_gene == "non-targeting"]
perturbed = data[data.target_gene != "non-targeting"]
names = ["negligible", "subtle", "strong"]

# KO and NP get their own row. Overlaying them in one panel hides whichever is drawn
# first, and because alpha compounds with density the panel then reads as cell count
# rather than as class.
figure, axes = plt.subplots(2, 4, figsize=(17.5, 9.6), sharex=True, sharey=True)

for row, (klass, colour, label) in enumerate([("KO", KO, "KO-like"), ("NP", NP, "non-perturbed")]):
    reference = control if row == 0 else perturbed
    axes[row, 0].scatter(data.UMAP1, data.UMAP2, s=0.8, c=BACKDROP, linewidths=0, rasterized=True)
    axes[row, 0].scatter(reference.UMAP1, reference.UMAP2, s=0.8, c=CONTROL, alpha=0.4,
                         linewidths=0, rasterized=True)
    axes[row, 0].set_title(
        f"{'non-targeting control' if row == 0 else 'all perturbed cells'}\n{len(reference):,} cells",
        fontsize=11, color=INK, pad=10)

    for column, name in enumerate(names, start=1):
        cells = data[(data["bin"] == name) & (data.mixscape_class_global == klass)]
        axis = axes[row, column]
        axis.scatter(data.UMAP1, data.UMAP2, s=0.8, c=BACKDROP, linewidths=0, rasterized=True)
        axis.scatter(cells.UMAP1, cells.UMAP2, s=0.8, c=colour, alpha=0.4, linewidths=0, rasterized=True)
        binned = data[data["bin"] == name]
        share = (binned.mixscape_class_global == klass).sum() / len(binned)
        axis.set_title(f"{name} · {label}\n{len(cells):,} cells · {share:.0%} of the bin",
                       fontsize=11, color=INK, pad=10)

for axis in axes.flat:
    axis.set_xticks([]), axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

handles = [
    plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=CONTROL, label="reference population"),
    plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=KO, label="KO-like (top row)"),
    plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=NP, label="non-perturbed (bottom row)"),
]
figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
              labelcolor=MUTED, bbox_to_anchor=(0.5, -0.015))
figure.suptitle("Mixscape classification within each response bin", fontsize=13, color=INK, y=1.01)
figure.tight_layout()
figure.savefig(args.output / "umap_mixscape_bins.png", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
print(f"wrote {args.output / 'umap_mixscape_bins.png'}")
for name in ["negligible", "subtle", "strong"]:
    c = data[data["bin"] == name]
    print(f"  {name:11} KO {(c.mixscape_class_global=='KO').sum():6}  NP {(c.mixscape_class_global=='NP').sum():6}")
