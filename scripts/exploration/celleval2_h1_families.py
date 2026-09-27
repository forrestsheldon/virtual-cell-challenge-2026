"""Do perturbations fall into families with shared responses?

Two complementary partitions of the 150 perturbations, both on the same object
the co-complex test used: the per-perturbation shift vectors (celleval2_shifts.npy,
Systema O(X) - O_control over the 10,780 gated genes), compared by cosine.

1. DATA-DRIVEN families: hierarchical clustering of the 150x150 shift-cosine
   matrix. Most perturbations are near-orthogonal (median pairwise cosine ~0.02),
   so only genuinely coherent groups form tight blocks; the rest is a diffuse
   background. The tight blocks are the families.

2. ANNOTATION test + transfer ceiling: for CORUM complexes with >=2 panel targets,
   is a held-out member's response predicted by the mean of its complex-mates? This
   is the ceiling of the "predict-by-pathway" baseline -- how well family membership
   alone reconstructs a response -- measured leave-one-out against a random-partner
   null.
"""

import argparse
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--corum", type=Path, default=Path("data/external/annotations/humanComplexes.txt"))
parser.add_argument("--min-cosine", type=float, default=0.3,
                    help="a data-driven family requires ALL member pairs above this cosine")
args = parser.parse_args()

shifts = np.load(args.output / "celleval2_shifts.npy")
targets = pd.read_csv(args.output / "celleval2_systema_summary.csv").target.to_numpy()
bins = pd.read_csv(args.output / "celleval2_response_bins.csv").set_index("target")["bin"]
index = {t: i for i, t in enumerate(targets)}

unit = shifts / np.linalg.norm(shifts, axis=1, keepdims=True)
cosine = unit @ unit.T
np.clip(cosine, -1, 1, out=cosine)

# --- 1. data-driven families ---------------------------------------------------
# Complete linkage + a distance cut: every pair inside a family clears the cosine
# threshold, so only genuinely tight groups form families and the near-orthogonal
# background stays as singletons -- the honest structure, not one chained blob.
distance = squareform(1 - cosine, checks=False)
Z = linkage(distance, method="complete")
family = fcluster(Z, t=1 - args.min_cosine, criterion="distance")
order = leaves_list(Z)

fam = pd.DataFrame({"target": targets, "family": family, "bin": bins.loc[targets].to_numpy()})
sizes = fam.family.value_counts()
print(f"=== data-driven families (complete linkage, all pairs cosine >= {args.min_cosine}) ===")
print(f"{(sizes >= 2).sum()} families with >=2 members; "
      f"{(sizes == 1).sum()} singletons (near-orthogonal to everything)\n")
for f, n in sizes[sizes >= 2].items():
    members = fam.target[fam.family == f].tolist()
    within = np.mean([cosine[index[a], index[b]] for a, b in combinations(members, 2)])
    tag = fam.bin[fam.family == f].value_counts().idxmax()
    print(f"  family {f:2} (n={n:2}, mostly {tag:10}, within-cos {within:.2f}): {', '.join(members)}")
fam.to_csv(args.output / "celleval2_families.csv", index=False)

# --- 2. CORUM complexes as families + transfer ceiling -------------------------
corum = pd.read_csv(args.corum, sep="\t")
corum = corum[corum.organism == "Human"] if "organism" in corum else corum
target_set = set(targets)
complexes = [set(str(s).split(";")) & target_set for s in corum.subunits_gene_name.fillna("")]
complexes = [c for c in complexes if len(c) >= 2]

partners = {t: set() for t in targets}          # co-complex panel partners of each target
for c in complexes:
    for t in c:
        partners[t] |= (c - {t})
has_partner = [t for t in targets if partners[t]]
print(f"\n=== CORUM transfer ceiling ===")
print(f"{len(has_partner)} of 150 targets share a complex with >=1 other panel target")

rng = np.random.default_rng(0)
rows = []
for t in has_partner:
    truth = unit[index[t]]
    pred = shifts[[index[p] for p in partners[t]]].mean(axis=0)      # mean shift of complex-mates
    pred = pred / np.linalg.norm(pred)
    # null: same number of RANDOM non-partner targets
    others = [x for x in targets if x != t and x not in partners[t]]
    null = []
    for _ in range(200):
        pick = rng.choice(others, len(partners[t]), replace=False)
        r = shifts[[index[p] for p in pick]].mean(axis=0)
        null.append(float(truth @ (r / np.linalg.norm(r))))
    rows.append({"target": t, "n_partners": len(partners[t]), "bin": bins[t],
                 "cos_to_partner_mean": float(truth @ pred), "null_mean": float(np.mean(null))})
transfer = pd.DataFrame(rows).sort_values("cos_to_partner_mean", ascending=False)
transfer.to_csv(args.output / "celleval2_transfer_ceiling.csv", index=False)
print(f"cosine(true shift, complex-mate mean): median {transfer.cos_to_partner_mean.median():.3f} "
      f"vs random-partner null {transfer.null_mean.median():.3f}")
print(f"beats its own null for {(transfer.cos_to_partner_mean > transfer.null_mean).sum()}/{len(transfer)} targets")
print("\ntop transfers (family membership alone predicts the response):")
print(transfer.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# --- figure: clustered cosine heatmap ------------------------------------------
cmap = LinearSegmentedColormap.from_list("div", ["#256abf", "#eef1f5", "#a4442d"])
BIN_COLOR = {"negligible": "#6da7ec", "subtle": "#256abf", "strong": "#0d366b"}
fig, ax = plt.subplots(figsize=(9.6, 9.6), constrained_layout=True)
M = cosine[np.ix_(order, order)]
im = ax.imshow(M, cmap=cmap, vmin=-0.4, vmax=0.4, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
strip = np.array([[tuple(int(BIN_COLOR[bins[targets[i]]].lstrip("#")[k:k+2], 16) / 255 for k in (0, 2, 4))
                   for i in order]])
ax.imshow(strip, extent=(0, len(order), -3.5, -0.5), aspect="auto")
ax.set_ylim(len(order) - 0.5, -3.5)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="shift cosine similarity")
ax.set_title("Perturbation response similarity, hierarchically ordered\n"
             "(bottom strip: negligible / subtle / strong)", fontsize=12, color="#131820")
fig.savefig(args.output / "celleval2_family_heatmap.png", dpi=200, facecolor="#fcfcfb")
print(f"\nwrote celleval2_family_heatmap.png")

(args.output / "celleval2_families_run.json").write_text(json.dumps({
    "created_utc": datetime.now(UTC).isoformat(),
    "object": "per-perturbation shift vectors (celleval2_shifts.npy), cosine similarity",
    "data_driven": {"method": "complete linkage on 1-cosine", "min_cosine": args.min_cosine,
                    "families_ge2": int((sizes >= 2).sum()), "singletons": int((sizes == 1).sum())},
    "corum_transfer": {"n_targets_with_partner": len(has_partner),
                       "median_cos_to_partner_mean": float(transfer.cos_to_partner_mean.median()),
                       "median_null": float(transfer.null_mean.median())},
}, indent=2) + "\n")
