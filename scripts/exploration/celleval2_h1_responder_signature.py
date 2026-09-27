"""Is there a gene signature shared by responders across all perturbations?

Regresses the per-cell Fisher responder score (celleval2_h1_fisher_score.py) on
gene expression, pooled over all perturbed cells. Unlike the shared axis a, the
Fisher score is not a single linear functional of expression: each perturbation t
contributes z_i = f_t(y_i) with its own marker set and variance-normalized,
z-scored weights. Pooled across the 150 perturbations it is therefore NOT any
single a . y, so a global sparse regression does real work -- its held-out R^2 is
the fraction of "responder-ness" that a single shared gene direction explains
across perturbations, and the selected genes are that shared responder program
(discovered, not recovered). Low R^2 would mean each perturbation's responder axis
is idiosyncratic.

Only perturbed cells enter, each with the score on its own perturbation's axis
(comparable across targets: all in control-SD units). Panel target genes are
dropped from the features so the signature is downstream biology, not each cell's
own knockdown.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from celleval2_h1_shared_signature import build_matrix
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler

N_CELLS = 40_000
MAX_GENES = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
    args = parser.parse_args()

    genes = pd.read_csv(args.output / "celleval2_shift_genes.csv").gene.to_numpy()
    panel = set(pd.read_csv(args.output / "celleval2_response_bins.csv").target)
    fisher = pd.read_parquet(args.output / "celleval2_fisher_scores.parquet",
                             columns=["cell_id", "target", "cell_group", "score"])
    fisher = fisher[fisher.cell_group == "Perturbed"].set_index("cell_id")

    adata = ad.read_h5ad(args.input, backed="r")
    gene_cols = pd.Series(np.arange(adata.n_vars), index=adata.var_names).loc[genes].to_numpy()
    cell_pos = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)

    rng = np.random.default_rng(0)
    sample = fisher.index.to_numpy()[rng.choice(len(fisher), N_CELLS, replace=False)]
    rows = np.sort(cell_pos.loc[sample].to_numpy())
    ordered_ids = adata.obs_names[rows]
    score = fisher.loc[ordered_ids, "score"].to_numpy()
    print(f"{len(rows):,} perturbed cells; Fisher score range {score.min():.1f}..{score.max():.1f}", flush=True)

    Y = build_matrix(adata, rows, gene_cols)
    feature = ~np.isin(genes, list(panel))
    X_genes = genes[feature]

    half = len(rows) // 2
    order = rng.permutation(len(rows))
    train, test = order[:half], order[half:]
    scaler = StandardScaler().fit(Y[train][:, feature])
    Xtr, Xte = scaler.transform(Y[train][:, feature]), scaler.transform(Y[test][:, feature])
    ytr, yte = score[train] - score[train].mean(), score[test] - score[test].mean()

    print("running LARS-Lasso path ...", flush=True)
    _, active, coefs = lars_path(Xtr, ytr, method="lasso", max_iter=MAX_GENES, verbose=False)

    denom = (yte**2).sum()
    curve = []
    for step in range(coefs.shape[1]):
        beta = coefs[:, step]
        k = int((beta != 0).sum())
        if k:
            curve.append({"n_genes": k, "test_r2": float(1 - ((yte - Xte @ beta) ** 2).sum() / denom)})
    curve = pd.DataFrame(curve).drop_duplicates("n_genes", keep="last")
    curve.to_csv(args.output / "celleval2_responder_signature_curve.csv", index=False)

    entry = [X_genes[i] for i in active]
    final = coefs[:, -1]
    pd.DataFrame({"gene": X_genes[final != 0], "coef": final[final != 0],
                  "entry_rank": [entry.index(g) + 1 for g in X_genes[final != 0]]}
                 ).sort_values("entry_rank").to_csv(args.output / "celleval2_responder_signature.csv", index=False)

    print("held-out R^2 of a shared sparse responder direction:")
    for k in [5, 10, 20, 50, 100, 200]:
        row = curve[curve.n_genes <= k].tail(1)
        if len(row):
            print(f"  <= {k:3} genes: {row.test_r2.iloc[0]:.3f}")
    print("\nfirst 30 genes admitted (+ up in responders / - down):")
    for g in entry[:30]:
        c = final[list(X_genes).index(g)]
        print(f"  {'+' if c > 0 else '-'} {g}")

    (args.output / "celleval2_responder_signature_run.json").write_text(json.dumps({
        "created_utc": datetime.now(UTC).isoformat(),
        "target": "per-cell Fisher responder score, pooled over all perturbed cells, own-perturbation axis",
        "features": "log1p-CPM over gated genes minus panel targets, standardized",
        "method": "LARS-Lasso path, held-out R^2 on a 50/50 cell split",
        "n_cells": len(rows), "n_features": int(feature.sum()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
