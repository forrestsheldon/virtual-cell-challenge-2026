"""How much of the H1 perturbation response is Systema's "systematic variation"?

Systema (Vinas Torne et al., Nat Biotechnol 2025, doi 10.1038/s41587-025-02777-8)
defines the shared component geometrically: for perturbation X with centroid O(X) and
control centroid O_control, the perturbation-specific shift is s_X = O(X) - O_control,
the average perturbation effect is a = mu_pert - O_control where mu_pert is the
per-cell mean over all perturbed cells, and *systematic variation* is the mean over
perturbations of cos(s_X, a). They report 0.41 +/- 0.18 across ten datasets, with
Replogle K562 at 0.32 +/- 0.16 and the lowest datasets around 0.2-0.3.

Centroids are means over cells of log-normalized expression, so they cannot be read
off the pseudobulk means already computed -- mean(log1p(x)) is not log1p(mean(x)).
This streams the file once in contiguous chunks to accumulate them exactly.

Also computes Systema's re-referencing: replacing the control centroid with the
unweighted mean of per-perturbation centroids should move the cosines to near zero,
which is the check that the shared direction has actually been removed.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from celleval2_h1_de import CPM_GATE

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--chunk", type=int, default=8000)
args = parser.parse_args()

adata = ad.read_h5ad(args.input, backed="r")
labels = adata.obs["target_gene"].astype(str).to_numpy()
groups = sorted(set(labels))
code = {g: i for i, g in enumerate(groups)}
codes = np.array([code[g] for g in labels])
n_groups, n_genes = len(groups), adata.n_vars

totals = np.zeros((n_groups, n_genes))       # sum of log1p(CPM) per group
counts = np.zeros(n_groups)
cpm_totals = np.zeros((n_groups, n_genes))   # sum of linear CPM, for the gate

for start in range(0, adata.n_obs, args.chunk):
    stop = min(start + args.chunk, adata.n_obs)
    block = adata.X[start:stop].tocsr().astype(np.float64)
    block.data /= np.repeat(np.asarray(block.sum(axis=1)).ravel() / 1e6, np.diff(block.indptr))
    dense = block.toarray()
    for g in np.unique(codes[start:stop]):
        rows = dense[codes[start:stop] == g]
        cpm_totals[g] += rows.sum(axis=0)
        totals[g] += np.log1p(rows).sum(axis=0)
        counts[g] += len(rows)
    print(f"  {stop}/{adata.n_obs} cells", end="\r", flush=True)

control = code["non-targeting"]
centroids = totals / counts[:, None]                       # O(X), log-normalized space
keep = (cpm_totals[control] / counts[control]) > CPM_GATE  # the same reference CPM gate
print(f"\n{n_groups - 1} perturbations, {int(counts.sum())} cells, {keep.sum()} genes past the gate")

targets = [g for g in groups if g != "non-targeting"]
rows = np.array([code[t] for t in targets])
O_control = centroids[control][keep]
shifts = centroids[rows][:, keep] - O_control              # s_X

# a = mu_pert - O_control, with mu_pert the PER-CELL mean over all perturbed cells
mu_pert = totals[rows][:, keep].sum(axis=0) / counts[rows].sum()
a_cell = mu_pert - O_control
a_unweighted = shifts.mean(axis=0)

def cosines(matrix, direction):
    unit = direction / np.linalg.norm(direction)
    return (matrix @ unit) / np.linalg.norm(matrix, axis=1)

result = {}
for name, direction in [("cell_weighted", a_cell), ("unweighted", a_unweighted)]:
    c = cosines(shifts, direction)
    result[name] = {"mean": float(c.mean()), "sd": float(c.std()), "median": float(np.median(c)),
                    "q25": float(np.percentile(c, 25)), "q75": float(np.percentile(c, 75))}
    print(f"systematic variation, a = {name:13}: mean {c.mean():.3f} +/- {c.std():.3f}  median {np.median(c):.3f}")

# Systema's re-referencing: control centroid -> unweighted mean of perturbation centroids.
O_pert = centroids[rows][:, keep].mean(axis=0)
rereferenced = centroids[rows][:, keep] - O_pert
c_re = cosines(rereferenced, rereferenced.mean(axis=0))
result["rereferenced"] = {"mean": float(c_re.mean()), "sd": float(c_re.std())}
print(f"after re-referencing to the perturbed centroid : mean {c_re.mean():.3f} +/- {c_re.std():.3f}  (Systema report -0.06)")

pairs = shifts / np.linalg.norm(shifts, axis=1)[:, None]
C = pairs @ pairs.T
off = C[np.triu_indices_from(C, k=1)]
print(f"pairwise cosine between shifts: median {np.median(off):.3f}, 90th {np.percentile(off,90):.3f}, max {off.max():.3f}")

frame = pd.DataFrame({"target": targets, "n_cells": counts[rows].astype(int),
                      "cos_shared": cosines(shifts, a_cell),
                      "shift_norm": np.linalg.norm(shifts, axis=1)})
frame.to_csv(args.output / "celleval2_systema_summary.csv", index=False)
np.save(args.output / "celleval2_shifts.npy", shifts)
np.save(args.output / "celleval2_shared_axis.npy", a_cell)
pd.Series(adata.var_names.to_numpy()[keep]).to_csv(args.output / "celleval2_shift_genes.csv", index=False, header=["gene"])

(args.output / "celleval2_systema_run.json").write_text(json.dumps({
    "created_utc": datetime.now(UTC).isoformat(),
    "reference": "Vinas Torne et al., Nature Biotechnology 2025, doi 10.1038/s41587-025-02777-8",
    "definition": "systematic variation = mean over perturbations of cos(O(X) - O_control, mu_pert - O_control)",
    "space": "per-cell CPM then log1p; centroids are means over cells",
    "gene_set": f"reference mean CPM > {CPM_GATE}",
    "n_genes": int(keep.sum()), "n_perturbations": len(targets),
    "systema_reported": {"all_datasets_mean": 0.41, "all_datasets_sd": 0.18,
                         "replogle_k562": 0.32, "adamson": 0.76, "norman": 0.50},
    "result": result}, indent=2) + "\n")
