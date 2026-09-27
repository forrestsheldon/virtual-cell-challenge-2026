"""The response spectrum in the cell-eval2 universe.

Successor to response_spectrum.png. That figure carried three nested criteria
because the earlier null appeared to demand effect-size thresholds on top of FDR;
the size-matched random null showed the floor was five off-target control guides
and 7,059 genes below the CPM gate, so FDR < 0.05 alone is the criterion now and
the nested lines collapse to one.

The space freed goes to two things the earlier figure could not show: where the
measured null floor actually sits, and where the competition's own metric stops
being able to score a target.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = [("negligible", "#6da7ec"), ("subtle", "#256abf"), ("strong", "#0d366b")]
NEUTRAL, LINE, INK, MUTED = "#6b6a66", "#c3cbd6", "#131820", "#555f6d"
EFFECT = "#eb6834"
NMAE_FLOOR = 10
MIN_LFC = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--at400", action="store_true",
                    help="draw both curves from the 400-cell subsample, so the two criteria are "
                         "compared at equal power on the 126 targets that have 400 cells to give")
args = parser.parse_args()

bins = pd.read_csv(args.output / "celleval2_response_bins.csv")

# The same DE set with an effect-size floor added. Kept as a separate line rather than
# a separate figure because the gap between the two is the point: FDR alone counts
# every gene the test can resolve, however small the shift.
source = "celleval2_subsample_de.parquet" if args.at400 else "celleval2_de.parquet"
columns = ["target", "p_adj", "log2_fold_change"] + (["replicate"] if args.at400 else [])
de = pd.read_parquet(args.output / source, columns=columns)
significant = de[de.p_adj < 0.05]

if args.at400:
    # Both counts per replicate, then the median across replicates for each -- so the two
    # curves are computed on the same draws and neither is favoured by a lucky sample.
    undersized = set(pd.read_csv(args.output / "celleval2_subsample_summary.csv")
                     .query("undersized").target)
    significant = significant[~significant.target.isin(undersized)]
    per = significant.groupby(["target", "replicate"]).agg(
        n_de=("p_adj", "size"),
        n_de_lfc=("log2_fold_change", lambda v: int((v.abs() > MIN_LFC).sum())))
    counts = per.groupby("target").median()
    bins = bins[bins.target.isin(counts.index)].drop(columns=["n_de"])
    bins = bins.merge(counts, left_on="target", right_index=True)
    # Re-bin on the power-equalised count: the old tertiles were cut on a different variable.
    cuts = bins.n_de.quantile([1 / 3, 2 / 3]).to_numpy()
    bins["bin"] = pd.cut(bins.n_de, [-np.inf, *cuts, np.inf], labels=[b for b, _ in BINS])
    print(f"tertile cuts at 400 cells: {cuts[0]:.0f} and {cuts[1]:.0f} DE genes")
else:
    strict = (significant[significant.log2_fold_change.abs() > MIN_LFC]
              .groupby("target").size().rename("n_de_lfc"))
    bins = bins.merge(strict, left_on="target", right_index=True, how="left").fillna({"n_de_lfc": 0})

ordered = bins.sort_values("n_de").reset_index(drop=True)
rank = np.arange(1, len(ordered) + 1)

if not args.at400:
    subsample = pd.read_csv(args.output / "celleval2_subsample_summary.csv")
    ordered["n_de_400"] = ordered.target.map(
        subsample[~subsample.undersized].groupby("target").n_de.median())

figure, axis = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)

# The measured null: 2 significant genes across 90 held-out comparisons, so at most
# one in any single comparison. Drawn as a band because zero has no place on a log axis.
axis.axhspan(0.55, 1, color=LINE, alpha=0.45, zorder=0)
null_note = ("measured null floor — 1 gene across 10 held-out draws of 400 control cells"
             if args.at400 else
             "measured null floor — at most 1 gene in any of 90 held-out comparisons")
axis.text(len(ordered), 0.72, null_note, ha="right", va="center", fontsize=8.5, color=MUTED)

axis.axhline(NMAE_FLOOR, color=NEUTRAL, linestyle="--", linewidth=1.1, zorder=1)
# Right-aligned: the left of this line is crowded by the 400-cell points, and the
# right half is empty at this height.
axis.text(len(ordered), NMAE_FLOOR * 1.25, "cell-eval2 lfc_nmae needs 10 genes to score a target",
          ha="right", fontsize=8.5, color=MUTED, va="bottom")

if not args.at400:
    axis.scatter(rank, ordered.n_de_400, s=11, color=NEUTRAL, alpha=0.5, linewidths=0, zorder=2,
                 label="same targets at 400 cells (competition size)")

for name, colour in BINS:
    mask = (ordered["bin"] == name).to_numpy()
    axis.plot(rank[mask], ordered.n_de[mask], color=colour, linewidth=2.2, zorder=4,
              label=f"{name} ({mask.sum()})")
axis.plot(rank, ordered.n_de_lfc, color=EFFECT, linewidth=1.8, zorder=3,
          label=f"same genes, also |log2FC| > {MIN_LFC}")
for boundary in np.flatnonzero(ordered["bin"].ne(ordered["bin"].shift()))[1:]:
    axis.axvline(boundary + 0.5, color=LINE, linewidth=1, zorder=1)

ticks = [0, 24, 49, 74, 99, 124, 149] if not args.at400 else [0, 20, 41, 62, 83, 104, 125]
axis.set_xticks(np.array(ticks) + 1)
axis.set_xticklabels(ordered.loc[ticks, "target"], rotation=45, ha="right", fontsize=9)
axis.set(yscale="log", xlim=(0, len(ordered) + 1), ylim=(0.55, 20000),
         xlabel=f"{len(ordered)} perturbations, ordered by DE genes", ylabel="DE genes (FDR < 0.05)")
axis.set_title("Response spectrum at 400 cells" if args.at400 else "Response spectrum",
               fontsize=13, color=INK)
axis.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=MUTED)
axis.spines[["top", "right"]].set_visible(False)
axis.tick_params(colors=MUTED)
axis.xaxis.label.set_color(MUTED), axis.yaxis.label.set_color(MUTED)

name = "celleval2_response_spectrum_400.png" if args.at400 else "celleval2_response_spectrum.png"
figure.savefig(args.output / name, dpi=200, facecolor="#fcfcfb")
print(f"wrote {args.output / name}")
print(f"range {ordered.n_de.min()}-{ordered.n_de.max()}, median {int(ordered.n_de.median())}")
print(f"below the lfc_nmae floor: {(ordered.n_de < NMAE_FLOOR).sum()} of {len(ordered)}")
print(f"with |log2FC| > {MIN_LFC}: range {int(ordered.n_de_lfc.min())}-{int(ordered.n_de_lfc.max())}, "
      f"median {int(ordered.n_de_lfc.median())}")
print("Spearman between the two orderings:",
      round(ordered.n_de.corr(ordered.n_de_lfc, method="spearman"), 3))
