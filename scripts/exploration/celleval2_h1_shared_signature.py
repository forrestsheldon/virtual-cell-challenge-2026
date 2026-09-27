"""A sparse gene signature for the shared perturbation axis.

The systematic-variation axis a = mu_pert - O_control (celleval2_h1_systema.py) is
diffuse: the top 200 genes by |a| carry only ~27% of its L2 norm, so no small set
of genes dominates its raw coefficients. This asks a different question: how few
genes are needed to *reconstruct the per-cell projection* onto that axis, and which
ones -- a sparse linear regression of the global perturbation score on gene
expression. Because correlated genes are redundant, the minimal panel can be far
smaller than the raw loading vector suggests.

The score s_i = a . y_i is an exact linear function of y, so this is compression,
not discovery: with every gene the reconstruction is trivially perfect. The
informative outputs are the held-out R^2 vs. panel-size curve, and the identity of
the genes the LARS/Lasso path admits first. Panel target genes are dropped from the
features (not the axis) so the signature is downstream biology, not knockdown of the
150 panel genes themselves; each panel gene's own contribution to a is ~1/150 of a
single knockdown and therefore negligible.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler

N_CONTROL, N_PERTURBED = 15_000, 25_000
MAX_GENES = 200


def build_matrix(adata, rows, gene_cols, chunk=4_000):
    out = np.empty((len(rows), len(gene_cols)), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        block = adata.X[rows[start:start + chunk]].tocsr().astype(np.float64)
        totals = np.asarray(block.sum(axis=1)).ravel()
        dense = block[:, gene_cols].toarray() * (1e6 / totals)[:, None]
        out[start:start + chunk] = np.log1p(dense)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
    args = parser.parse_args()

    a = np.load(args.output / "celleval2_shared_axis.npy")
    genes = pd.read_csv(args.output / "celleval2_shift_genes.csv").gene.to_numpy()
    panel = set(pd.read_csv(args.output / "celleval2_response_bins.csv").target)

    adata = ad.read_h5ad(args.input, backed="r")
    gene_cols = pd.Series(np.arange(adata.n_vars), index=adata.var_names).loc[genes].to_numpy()
    labels = adata.obs["target_gene"].astype(str).to_numpy()

    rng = np.random.default_rng(0)
    control_rows = rng.choice(np.flatnonzero(labels == "non-targeting"), N_CONTROL, replace=False)
    perturbed_rows = rng.choice(np.flatnonzero(labels != "non-targeting"), N_PERTURBED, replace=False)
    rows = np.sort(np.concatenate([control_rows, perturbed_rows]))
    is_control = labels[rows] == "non-targeting"
    print(f"{len(rows):,} cells sampled ({is_control.sum():,} control, {(~is_control).sum():,} perturbed)", flush=True)

    Y = build_matrix(adata, rows, gene_cols)            # log1p-CPM over the 10,780 gated genes
    score = Y @ a                                        # global perturbation score, per cell (Systema axis)
    np.save(args.output / "celleval2_global_score_sample.npy", score)
    pd.DataFrame({"cell_id": adata.obs_names[rows], "target": labels[rows],
                  "is_control": is_control, "global_score": score}
                 ).to_parquet(args.output / "celleval2_global_score_sample.parquet", index=False)

    feature = ~np.isin(genes, list(panel))               # drop the 150 panel target genes from the features
    X_genes = genes[feature]
    half = len(rows) // 2
    order = rng.permutation(len(rows))
    train, test = order[:half], order[half:]

    scaler = StandardScaler().fit(Y[train][:, feature])
    Xtr = scaler.transform(Y[train][:, feature])
    Xte = scaler.transform(Y[test][:, feature])
    ytr, yte = score[train] - score[train].mean(), score[test] - score[test].mean()

    print("running LARS-Lasso path ...", flush=True)
    _, active, coefs = lars_path(Xtr, ytr, method="lasso", max_iter=MAX_GENES, verbose=False)

    denom = (yte**2).sum()
    curve = []
    for step in range(coefs.shape[1]):
        beta = coefs[:, step]
        k = int((beta != 0).sum())
        if k == 0:
            continue
        r2 = 1 - ((yte - Xte @ beta) ** 2).sum() / denom
        curve.append({"n_genes": k, "test_r2": float(r2)})
    curve = pd.DataFrame(curve).drop_duplicates("n_genes", keep="last")
    curve.to_csv(args.output / "celleval2_shared_signature_curve.csv", index=False)

    entry_order = [X_genes[i] for i in active]           # genes in the order the path admits them
    final = coefs[:, -1]
    signature = pd.DataFrame({
        "gene": X_genes[final != 0],
        "coef": final[final != 0],
        "axis_loading": a[feature][final != 0],
        "entry_rank": [entry_order.index(g) + 1 if g in entry_order else -1 for g in X_genes[final != 0]],
    }).sort_values("entry_rank")
    signature.to_csv(args.output / "celleval2_shared_signature.csv", index=False)

    print("\nheld-out reconstruction of the global perturbation score:")
    for k in [5, 10, 20, 50, 100, 200]:
        row = curve[curve.n_genes <= k].tail(1)
        if len(row):
            print(f"  <= {k:3} genes: test R^2 = {row.test_r2.iloc[0]:.3f}")
    print(f"\nfirst 30 genes admitted (sign = direction on the shared axis):")
    for g in entry_order[:30]:
        print(f"  {'+' if a[genes == g][0] > 0 else '-'} {g}")

    (args.output / "celleval2_shared_signature_run.json").write_text(json.dumps({
        "created_utc": datetime.now(UTC).isoformat(),
        "target": "per-cell global perturbation score s = a . y (Systema shared axis)",
        "features": "log1p-CPM over gated genes, panel target genes excluded, standardized",
        "method": "LARS-Lasso path, held-out R^2 on a 50/50 cell split",
        "n_cells": len(rows), "n_features": int(feature.sum()),
        "final_panel_size": int((final != 0).sum()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
