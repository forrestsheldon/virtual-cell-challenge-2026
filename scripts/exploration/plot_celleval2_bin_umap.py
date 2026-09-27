"""Diagnose each bin's fresh embedding before trusting any structure it shows.

Two negative controls per bin -- colour by batch and by total UMI -- since a
population with little real biological variance (negligible, and to a lesser
extent subtle) is exactly where HVG selection can end up tracking technical
structure instead. If either negative control shows visible clustering, the
positive-control panel next to it should not be read as biology.

The positive control is control-vs-perturbed separation, plus (where any exist)
the CORUM co-complex pairs fully contained in that bin, from
celleval2_h1_specificity.py -- do known complex partners sit near each other in
a bin-specific embedding the way they do in the shift-vector analysis, or was
that result only visible in aggregate. Negligible has no such pairs to test.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTROL, PERTURBED = "#6b6a66", "#0d366b"
INK, MUTED = "#131820", "#555f6d"
HIGHLIGHT = ["#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#4a3aa7", "#e34948"]

PAIRS_BY_BIN = {
    "strong": [("METTL3", "METTL14"), ("MED12", "MED13"), ("MED1", "MED24"), ("KAT2A", "TADA1")],
    "subtle": [("NDUFB4", "NDUFB6"), ("BIRC2", "IKBKG")],
    "negligible": [],
}

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
args = parser.parse_args()

for name in ["negligible", "subtle", "strong"]:
    frame = pd.read_parquet(args.output / f"celleval2_bin_umap_{name}.parquet")
    pairs = PAIRS_BY_BIN[name]
    n_cols = 3 + bool(pairs)
    figure, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.6), constrained_layout=True)

    control = frame.target_gene == "non-targeting"
    axes[0].scatter(frame.UMAP1[~control], frame.UMAP2[~control], s=1, c=PERTURBED, alpha=0.35, linewidths=0, rasterized=True)
    axes[0].scatter(frame.UMAP1[control], frame.UMAP2[control], s=1, c=CONTROL, alpha=0.35, linewidths=0, rasterized=True)
    axes[0].set_title(f"control vs perturbed\n{(~control).sum():,} vs {control.sum():,} cells", fontsize=10.5, color=INK)

    batch_code = frame.batch.astype("category").cat.codes
    axes[1].scatter(frame.UMAP1, frame.UMAP2, s=1, c=batch_code, cmap="tab20", alpha=0.5, linewidths=0, rasterized=True)
    axes[1].set_title(f"batch (negative control)\n{frame.batch.nunique()} batches", fontsize=10.5, color=INK)

    depth = axes[2].scatter(frame.UMAP1, frame.UMAP2, s=1, c=frame.total_counts, cmap="viridis",
                            alpha=0.5, linewidths=0, rasterized=True, vmin=frame.total_counts.quantile(0.02),
                            vmax=frame.total_counts.quantile(0.98))
    axes[2].set_title("total UMI (negative control)", fontsize=10.5, color=INK)
    figure.colorbar(depth, ax=axes[2], fraction=0.046, pad=0.03)

    if pairs:
        axes[3].scatter(frame.UMAP1, frame.UMAP2, s=1, c="#d6d5d1", linewidths=0, rasterized=True)
        for (a, b), colour in zip(pairs, HIGHLIGHT):
            for gene, marker in [(a, "o"), (b, "^")]:
                cells = frame[frame.target_gene == gene]
                axes[3].scatter(cells.UMAP1, cells.UMAP2, s=10, c=colour, marker=marker,
                                label=f"{gene} ({len(cells)})", linewidths=0)
        axes[3].legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="best", ncol=1)
        axes[3].set_title("CORUM co-complex pairs\nwithin this bin", fontsize=10.5, color=INK)

    for axis in axes:
        axis.set_xticks([]), axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(f"{name.capitalize()} bin: independent embedding, gated 10,780-gene universe",
                    fontsize=12.5, color=INK)
    figure.savefig(args.output / f"umap_bin_specific_{name}.png", dpi=200, facecolor="#fcfcfb")
    print(f"wrote umap_bin_specific_{name}.png")
