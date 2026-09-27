"""Where do the three response bins sit on the H1 UMAP?

Perturbations are split into negligible / subtle / strong by tertiles of DE count
in the cell-eval2 universe. Tertiles are the principled cut here because Arc
sampled 100 perturbations from each of three DE-count bins and then split
150/100/50 balancing across them, so the 150 training targets should be about 50
per bin by construction. The underlying spectrum is continuous -- there is no gap
at either cut -- so the bins reconstruct Arc's design rather than discovering
structure.

Palette is a single-hue ordinal blue ramp (magnitude, so sequential rather than
categorical) plus two neutrals, stepped so every pair clears the all-pairs floors:
worst normal-vision dE 15.6, worst CVD dE 9.5, lightest step 2.44:1 on the surface.

Each panel draws every cell as a backdrop and then one group on top. Drawing the
controls *underneath* instead hides them completely -- the perturbed cells cover
the same manifold -- which defeats the comparison the figure exists to make.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = [("negligible", "#6da7ec"), ("subtle", "#256abf"), ("strong", "#0d366b")]
CONTROL, BACKDROP = "#6b6a66", "#d6d5d1"
INK, MUTED = "#0b0b0b", "#52514e"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
args = parser.parse_args()

summary = pd.read_csv(args.output / "celleval2_summary.csv")
cuts = summary.n_de.quantile([1 / 3, 2 / 3]).to_numpy()
summary["bin"] = pd.cut(summary.n_de, [-np.inf, *cuts, np.inf], labels=[b for b, _ in BINS])
summary[["target", "n_cells", "n_de", "bin"]].to_csv(args.output / "celleval2_response_bins.csv", index=False)
print(f"tertile cuts at {cuts[0]:.0f} and {cuts[1]:.0f} DE genes")

embedding = pd.read_parquet(
    args.output / "embedding.parquet", columns=["target_gene", "UMAP1", "UMAP2"]
)
embedding["bin"] = embedding.target_gene.map(dict(zip(summary.target, summary["bin"]))).astype(object)
control = embedding[embedding.target_gene == "non-targeting"]
print(f"{len(control)} control cells; " + ", ".join(
    f"{b} {(embedding['bin'] == b).sum()}" for b, _ in BINS))

panels = [("non-targeting control", CONTROL, control, f"{len(control):,} cells · 31 guide pairs")]
for name, colour in BINS:
    targets = summary[summary["bin"] == name]
    panels.append((name, colour, embedding[embedding["bin"] == name],
                   f"{len(targets)} targets · {targets.n_de.min():,}–{targets.n_de.max():,} DE genes"))

figure, axes = plt.subplots(1, 4, figsize=(17.5, 5.0), sharex=True, sharey=True)
for axis, (name, colour, cells, subtitle) in zip(axes, panels):
    axis.scatter(embedding.UMAP1, embedding.UMAP2, s=0.8, c=BACKDROP, linewidths=0, rasterized=True)
    axis.scatter(cells.UMAP1, cells.UMAP2, s=0.8, c=colour, alpha=0.4, linewidths=0, rasterized=True)
    axis.set_title(f"{name}\n{subtitle}", fontsize=11, color=INK, pad=10)
    axis.set_xticks([]), axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

figure.text(0.5, -0.015, "grey backdrop is all 221,273 cells; each panel highlights one group",
            ha="center", fontsize=10, color=MUTED)
figure.suptitle("Where the response bins sit on the shared H1 UMAP", fontsize=13, color=INK, y=1.02)
figure.tight_layout()
figure.savefig(args.output / "umap_response_bins.png", dpi=200, bbox_inches="tight",
               facecolor="#fcfcfb")
print(f"wrote {args.output / 'umap_response_bins.png'}")
