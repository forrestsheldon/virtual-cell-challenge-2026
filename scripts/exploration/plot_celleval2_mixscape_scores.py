"""Mixscape perturbation scores, with each cell coloured by the class it was given.

The existing version of this figure histograms control against perturbed cells and
draws the fitted threshold between them. That shows the mixture but not the outcome:
the perturbed histogram contains both classes, so you cannot see which cells the
threshold actually assigned to KO.

Here the perturbed population is split into its KO-like and non-perturbed parts,
using the classes from the production run rather than the ones refitted here, since
those are the labels every DE result in this report was computed with.

Representative guides span the range of responder fractions, as before; each panel
also names the target's response bin so this figure and the UMAPs can be read together.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_h1_mixscape import perturbation_scores, select_guides

NT, NP, KO = "#6b6a66", "#6da7ec", "#0d366b"
INK, MUTED, LINE = "#131820", "#555f6d", "#c3cbd6"

parser = argparse.ArgumentParser()
parser.add_argument("--signatures", type=Path, default=Path("data/derived/vcc2025_h1_mixscape/signatures"))
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--control-cells", type=int, default=5_000)
args = parser.parse_args()

bins = pd.read_csv(args.output / "celleval2_response_bins.csv").set_index("target")["bin"]
calls = pd.read_parquet(args.output / "mixscape_cells.parquet", columns=["cell_id", "mixscape_class_global"])
selected = select_guides(pd.read_csv(args.output / "mixscape_guides.csv"))
print(selected[["target_gene", "n_cells", "responder_fraction"]].to_string(index=False), flush=True)

scores = perturbation_scores(args.signatures, selected, args.control_cells)
scores = scores.merge(calls, on="cell_id", how="left")
scores["klass"] = np.where(scores.population == "Control", "NT", scores.mixscape_class_global)
scores.to_parquet(args.output / "celleval2_mixscape_scores.parquet", index=False)

# The threshold is refitted here while the labels come from the production run, so the
# two need not agree exactly. Report the disagreement rather than hiding it.
perturbed = scores[scores.population == "Perturbed"]
agree = ((perturbed.pvec > perturbed.threshold) == (perturbed.klass == "KO")).mean()
print(f"\nrefitted threshold agrees with the production label for {agree:.1%} of perturbed cells")

targets = scores.target.drop_duplicates().tolist()
limits = scores.pvec.quantile([0.005, 0.995]).to_numpy()
edges = np.linspace(*limits, 65)
figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharex=True, constrained_layout=True)

for axis, target in zip(axes, targets):
    data = scores[scores.target == target]
    for label, colour, name in (("NT", NT, "control"), ("NP", NP, "non-perturbed"), ("KO", KO, "KO-like")):
        values = data.loc[data.klass == label, "pvec"]
        axis.hist(values, bins=edges, histtype="stepfilled", alpha=0.55, color=colour,
                  label=f"{name} (n={len(values):,})")
    axis.axvline(data.threshold.iloc[0], color=INK, linestyle="--", linewidth=1.3, label="Mixscape threshold")
    axis.set_title(f"{target} · {bins.get(target, '?')}\n{data.responder_fraction.iloc[0]:.0%} KO-like",
                   fontsize=11, color=INK)
    axis.set(xlabel="Mixscape perturbation score", ylabel="cells per bin")
    axis.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(MUTED), axis.yaxis.label.set_color(MUTED)

figure.suptitle("Mixscape perturbation scores, split by the class each cell was assigned",
                fontsize=13, color=INK)
figure.savefig(args.output / "celleval2_mixscape_scores.png", dpi=200, facecolor="#fcfcfb")
print(f"wrote {args.output / 'celleval2_mixscape_scores.png'}")
