"""Per-cell responder scores: does a perturbed population separate into cells
that moved along the perturbation's own direction and cells that didn't?

Simplified successor to fisher_h1_directions.py. That script used Pearson
residuals under a negative-binomial model and excluded all 150 targets globally
from every marker set; this uses log1p(CPM) on the gated universe (the same
representation as celleval2_de.parquet) and excludes only each target's own gene.
Both keep the essential piece: weights are fit on cells from two of three Flex
batch groups and scores are read off the third, so a cell's score never depends
on a direction fit using that same cell -- without this, gene selection and
weight-fitting on the same cells being scored would show clean separation by
construction, regardless of whether the perturbation does anything.

Markers are the top 50 genes by |log2FC| among each target's FDR<0.05 genes
(celleval2_de.parquet), not all FDR-significant genes: a diagonal Fisher score
sums per-gene contributions, so a strong perturbation's thousands of tiny-effect
significant genes would mostly add noise to the direction. Capping at the
largest 50 effects never gives fewer genes than the FDR-only set already has
(minimum 4 across all 150 targets).
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_MARKERS = 50


def markers_by_target(de_path):
    de = pd.read_parquet(de_path, columns=["target", "gene", "p_adj", "log2_fold_change"])
    significant = de[de.p_adj < 0.05].copy()
    significant["abs_lfc"] = significant.log2_fold_change.abs()
    top = significant.sort_values("abs_lfc", ascending=False).groupby("target").head(N_MARKERS)
    return {t: g.gene[g.gene != t].tolist() for t, g in top.groupby("target")}


def prepare(adata, rows):
    """Read a cell block once: its CSR data plus full-transcriptome library sizes.
    Control blocks are identical across every target in a fold, so this is called
    once per fold for controls and reused, rather than re-read per target."""
    block = adata.X[rows].tocsr()
    totals = np.asarray(block.sum(axis=1)).ravel()
    return block, totals


def log1p_cpm(prepared, columns):
    block, totals = prepared
    values = block[:, columns].toarray() * (1e6 / totals)[:, None]
    return np.log1p(values)


def fit_and_score(positions, genes, train_target, train_control, heldout_target, heldout_control):
    """Each `*_target`/`*_control` argument is a (block, totals) pair from `prepare`."""
    columns = positions.loc[genes].to_numpy()
    tt, tc = log1p_cpm(train_target, columns), log1p_cpm(train_control, columns)

    difference = tt.mean(axis=0) - tc.mean(axis=0)
    variance = (tt.var(axis=0, ddof=1) + tc.var(axis=0, ddof=1)) / 2
    ridge = 0.1 * np.median(variance)
    weights = difference / (variance + ridge)

    target_score = log1p_cpm(heldout_target, columns) @ weights
    control_score = log1p_cpm(heldout_control, columns) @ weights
    center, scale = control_score.mean(), control_score.std(ddof=1)
    return (target_score - center) / scale, (control_score - center) / scale


def separation(target, control):
    variance = (target.var(ddof=1) + control.var(ddof=1)) / 2
    d_prime = (target.mean() - control.mean()) / np.sqrt(variance)
    labels = np.r_[np.ones(len(target)), np.zeros(len(control))]
    auc = roc_auc_score(labels, np.r_[target, control])
    return d_prime, auc


def run(adata, positions, labels, flex, markers, groups, group_col):
    """`groups`: iterable of group labels to score (e.g. 150 targets, or one
    fake target for the negative control). `group_col`: array of each cell's
    group membership, same length as `labels` (usually just `labels` itself)."""
    scores, summaries = [], []
    for fold in ("1", "2", "3"):
        held_out = flex == fold
        control_all = labels == "non-targeting"
        train_control_rows = np.flatnonzero(control_all & ~held_out)
        heldout_control_rows = np.flatnonzero(control_all & held_out)
        train_control = prepare(adata, train_control_rows)
        heldout_control = prepare(adata, heldout_control_rows)
        print(f"  Flex_{fold}: {len(train_control_rows):,} training + "
              f"{len(heldout_control_rows):,} held-out controls read once", flush=True)

        for group in groups:
            in_group = group_col == group
            train_target_rows = np.flatnonzero(in_group & ~held_out)
            heldout_target_rows = np.flatnonzero(in_group & held_out)
            if len(train_target_rows) < 10 or len(heldout_target_rows) < 5:
                continue
            train_target = prepare(adata, train_target_rows)
            heldout_target = prepare(adata, heldout_target_rows)

            target_score, control_score = fit_and_score(
                positions, markers[group], train_target, train_control, heldout_target, heldout_control
            )
            threshold = np.quantile(control_score, 0.95)
            d_prime, auc = separation(target_score, control_score)
            summaries.append({
                "target": group, "fold": f"Flex_{fold}", "n_genes": len(markers[group]),
                "n_target": len(target_score), "d_prime": d_prime, "roc_auc": auc,
                "control_like_fraction": float(np.mean(target_score <= threshold)),
            })
            for score, cell_group, rows in [
                (target_score, "Perturbed", heldout_target_rows),
                (control_score, "Control", heldout_control_rows),
            ]:
                scores.append(pd.DataFrame({
                    "cell_id": adata.obs_names[rows], "target": group, "fold": f"Flex_{fold}",
                    "cell_group": cell_group, "score": score, "responder": score > threshold,
                }))
    return pd.concat(scores, ignore_index=True), pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input, backed="r")
    labels = adata.obs["target_gene"].astype(str).to_numpy()
    flex = adata.obs["batch"].str.extract(r"Flex_(\d)", expand=False).to_numpy()
    positions = pd.Series(np.arange(adata.n_vars), index=adata.var_names)
    markers = markers_by_target(args.output / "celleval2_de.parquet")

    print("negative control: scoring an inert NT guide pair as if it were a target", flush=True)
    guide = adata.obs["guide_id"].astype(str).to_numpy()
    nt_guide = "non-targeting_00006|non-targeting_00706"  # 0 calls in celleval2_guide_null_summary.csv
    fake_group = np.where((labels == "non-targeting") & (guide == nt_guide), "NT_CONTROL_CHECK", labels)
    nt_markers = {"NT_CONTROL_CHECK": markers["MED12"][:N_MARKERS]}  # borrow a real gene set; only the score matters
    _, null_summary = run(adata, positions, labels, flex, nt_markers, ["NT_CONTROL_CHECK"], fake_group)
    print(null_summary[["fold", "n_target", "d_prime", "roc_auc", "control_like_fraction"]]
          .to_string(index=False))
    mean_fpr = 1 - null_summary.control_like_fraction.mean()
    print(f"mean false-positive rate on an inert guide: {mean_fpr:.1%} (expect ~5%, the threshold's own rate)\n")

    print(f"scoring {len(markers)} targets against all non-targeting controls", flush=True)
    scores, summary = run(adata, positions, labels, flex, markers, list(markers), labels)

    scores.to_parquet(args.output / "celleval2_fisher_scores.parquet", index=False)
    per_target = (
        summary.groupby("target")
        .agg(n_genes=("n_genes", "first"), n_target=("n_target", "sum"),
             mean_d_prime=("d_prime", "mean"), mean_roc_auc=("roc_auc", "mean"),
             mean_control_like_fraction=("control_like_fraction", "mean"))
        .reset_index()
    )
    per_target.to_csv(args.output / "celleval2_fisher_summary.csv", index=False)
    print(per_target.sort_values("mean_control_like_fraction").head(5).to_string(index=False))
    print(per_target.sort_values("mean_control_like_fraction").tail(5).to_string(index=False))

    (args.output / "celleval2_fisher_run.json").write_text(json.dumps({
        "markers": f"top {N_MARKERS} genes by |log2FC| among FDR<0.05 (celleval2_de.parquet), own gene excluded",
        "representation": "log1p(CPM), full-transcriptome library size, gated genes only as candidates",
        "direction": "cross-fitted diagonal Fisher (per-gene mean difference / pooled variance + ridge)",
        "ridge": "0.1 * median per-gene pooled variance among that target's markers",
        "cross_fit": "3 folds by Flex batch group; weights trained on 2, scored on the held-out third",
        "negative_control": {"guide": nt_guide, "mean_false_positive_rate": float(mean_fpr)},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
