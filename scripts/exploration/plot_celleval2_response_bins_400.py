"""Where the 400-cell response bins sit on the UMAP.

Companion to plot_celleval2_response_bins.py, restricted to the 126 targets that
have 400 cells to give and re-binned on their power-equalised DE count
(celleval2_response_bins_400.csv) rather than the full-cell bins. The 24 undersized
targets are dropped from every panel, including the backdrop, rather than shown
unbinned -- the point of this figure is the bins computed at competition size, and
an unbinned residual would invite reading it as a fourth group.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BINS = [("negligible", "#6da7ec"), ("subtle", "#256abf"), ("strong", "#0d366b")]
CONTROL, BACKDROP = "#6b6a66", "#d6d5d1"
INK, MUTED = "#131820", "#555f6d"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
args = parser.parse_args()

bins = pd.read_csv(args.output / "celleval2_response_bins_400.csv")
embedding = pd.read_parquet(args.output / "embedding.parquet", columns=["target_gene", "UMAP1", "UMAP2"])

control = embedding[embedding.target_gene == "non-targeting"]
kept = embedding[embedding.target_gene.isin(bins.target)].copy()
kept["bin"] = kept.target_gene.map(dict(zip(bins.target, bins["bin"])))
backdrop = pd.concat([control, kept])  # excludes the 24 undersized targets throughout
dropped = embedding.target_gene.nunique() - 1 - bins.target.nunique()
print(f"{len(control):,} control cells; {len(kept):,} cells across {bins.target.nunique()} targets; "
      f"{dropped} undersized targets excluded from every panel")

figure, axes = plt.subplots(1, 4, figsize=(17.5, 5.0), sharex=True, sharey=True)

axes[0].scatter(backdrop.UMAP1, backdrop.UMAP2, s=0.8, c=BACKDROP, linewidths=0, rasterized=True)
axes[0].scatter(control.UMAP1, control.UMAP2, s=0.8, c=CONTROL, alpha=0.4, linewidths=0, rasterized=True)
axes[0].set_title(f"non-targeting control\n{len(control):,} cells", fontsize=11, color=INK, pad=10)

for axis, (name, colour) in zip(axes[1:], BINS):
    targets = bins[bins["bin"] == name]
    cells = kept[kept["bin"] == name]
    axis.scatter(backdrop.UMAP1, backdrop.UMAP2, s=0.8, c=BACKDROP, linewidths=0, rasterized=True)
    axis.scatter(cells.UMAP1, cells.UMAP2, s=0.8, c=colour, alpha=0.4, linewidths=0, rasterized=True)
    axis.set_title(f"{name}\n{len(targets)} targets · {targets.n_de.min():,.0f}–{targets.n_de.max():,.0f} DE genes",
                   fontsize=11, color=INK, pad=10)

for axis in axes:
    axis.set_xticks([]), axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

figure.text(0.5, -0.015,
           "grey backdrop is the 126 targets shown plus controls; 24 targets with fewer than 400 cells are excluded",
           ha="center", fontsize=10, color=MUTED)
figure.suptitle("Response bins at 400 cells, on the shared H1 UMAP", fontsize=13, color=INK, y=1.02)
figure.tight_layout()
figure.savefig(args.output / "umap_response_bins_400.png", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
print(f"wrote {args.output / 'umap_response_bins_400.png'}")
