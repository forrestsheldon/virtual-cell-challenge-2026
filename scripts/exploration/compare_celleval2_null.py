"""Read the cell-eval2-universe target run against its held-out control null.

Prints the four things needed to decide whether a DE count means anything:
the null floor as a function of group size, where the real targets sit relative
to it, whether the on-target repression positive control holds, and how the
counts moved when the universe changed from the earlier 10,000-count run.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
args = parser.parse_args()

null = pd.read_csv(args.output / "celleval2_null_summary.csv")
targets = pd.read_csv(args.output / "celleval2_summary.csv")

print("=== Null floor: random non-targeting cells, held out of their own reference ===")
floor = null.groupby("n_cells").n_de.agg(["median", "min", "max", "mean"])
floor["genes"] = null.groupby("n_cells").n_genes.median().astype(int)
floor["up"] = null.groupby("n_cells").n_up.median()
floor["down"] = null.groupby("n_cells").n_down.median()
print(floor.to_string())

print("\n=== Real targets against the null at their own cell count ===")
sizes, medians = floor.index.to_numpy(), floor["median"].to_numpy()
worst = null.groupby("n_cells").n_de.max().to_numpy()
targets["null_median"] = np.exp(np.interp(np.log(targets.n_cells), np.log(sizes), np.log(medians + 1))) - 1
targets["null_worst"] = np.exp(np.interp(np.log(targets.n_cells), np.log(sizes), np.log(worst + 1))) - 1
targets["excess"] = targets.n_de / targets.null_median.clip(lower=1)
below = targets.null_median > targets.n_de
print(f"{below.sum()} of {len(targets)} targets call fewer DE genes than the median null at their size")
print(f"{(targets.n_de < targets.null_worst).sum()} of {len(targets)} fall below the worst null draw at their size")
print("\nweakest 15 targets by DE count:")
print(
    targets.nsmallest(15, "n_de")[
        ["target", "n_cells", "n_de", "null_median", "null_worst", "excess", "target_log2_fold_change", "target_p_adj"]
    ].to_string(index=False, float_format=lambda v: f"{v:.3g}")
)

print("\n=== Positive control: does every target repress its own gene? ===")
gated = targets.target_gene_gated
print(f"{gated.sum()} targets have their own gene below the CPM gate (no on-target row)")
scored = targets[~gated]
repressed = (scored.target_p_adj < 0.05) & (scored.target_log2_fold_change < 0)
print(f"{repressed.sum()} of {len(scored)} scored targets significantly repress their own gene")
if (~repressed).any():
    print(scored[~repressed][["target", "n_cells", "target_log2_fold_change", "target_p_adj"]].to_string(index=False))

old_path = args.output / "wilcoxon_summary.csv"
if old_path.exists():
    print("\n=== Universe change: 10,000-count/all-genes run versus cell-eval2 universe ===")
    old = pd.read_csv(old_path)[["target", "n_de"]].rename(columns={"n_de": "n_de_old"})
    both = targets.merge(old, on="target")
    both["ratio"] = both.n_de / both.n_de_old.clip(lower=1)
    print(f"median DE genes: old {both.n_de_old.median():.0f}, cell-eval2 {both.n_de.median():.0f}")
    print(f"targets with more DE genes in the new universe: {(both.n_de > both.n_de_old).sum()} of {len(both)}")
    print(f"ratio quartiles: {both.ratio.quantile([0.25, 0.5, 0.75]).round(3).to_dict()}")
