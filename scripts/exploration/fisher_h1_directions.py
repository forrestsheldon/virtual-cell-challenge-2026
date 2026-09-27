"""Cross-fitted Fisher directions for the VCC 2025 H1 screen."""

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

BLUE = "#4C78A8"
ORANGE = "#F58518"
GRAY = "#555555"


def load_markers(output, excluded_genes):
    summary = pl.read_csv(output / "wilcoxon_summary.csv").select("target", "n_cells")
    controls = json.loads((output / "wilcoxon_run.json").read_text())["n_controls"]
    markers = (
        pl.scan_parquet(output / "wilcoxon_de.parquet")
        .join(summary.lazy(), on="target")
        .with_columns(
            (pl.col("statistic") / (pl.col("n_cells") * controls) - 0.5)
            .abs()
            .alias("auc_delta")
        )
        .filter(
            (pl.col("fdr") <= 0.05)
            & (pl.col("log2_fold_change").abs() >= 0.5)
            & (pl.col("auc_delta") >= 0.05)
            & (~pl.col("gene").is_in(excluded_genes))
        )
        .select("target", "gene")
        .collect()
        .to_pandas()
    )
    return {target: group.gene.to_list() for target, group in markers.groupby("target")}


def residuals(counts, totals, columns, probabilities, theta, clip):
    observed = counts[:, columns].toarray()
    expected = totals[:, None] * probabilities[columns]
    values = (observed - expected) / np.sqrt(expected + expected**2 / theta)
    np.clip(values, -clip, clip, out=values)
    return values


def separation(target, control):
    variance = (target.var(ddof=1) + control.var(ddof=1)) / 2
    d_prime = (target.mean() - control.mean()) / np.sqrt(variance)
    labels = np.r_[np.ones(len(target)), np.zeros(len(control))]
    auc = roc_auc_score(labels, np.r_[target, control])
    return d_prime, auc


def calculate(input_path, output, theta, train_controls, heldout_controls, ridge):
    source = ad.read_h5ad(input_path, backed="r")
    obs = source.obs.copy()
    obs["flex"] = obs.batch.str.extract(r"Flex_(\d)", expand=False)
    targets = sorted(
        obs.loc[obs.target_gene != "non-targeting", "target_gene"].unique()
    )
    markers = load_markers(output, targets)
    excluded = sorted(set(targets) - set(markers))
    positions = pd.Series(np.arange(source.n_vars), index=source.var_names)
    score_frames = []
    fold_summary = []

    for fold in ("1", "2", "3"):
        rng = np.random.default_rng(int(fold))
        training_control_rows = np.flatnonzero(
            obs.target_gene.eq("non-targeting") & obs.flex.ne(fold)
        )
        reference = source.X[training_control_rows].tocsr()
        gene_counts = np.asarray(reference.sum(axis=0)).ravel()
        probabilities = (gene_counts + 0.5) / (gene_counts.sum() + 0.5 * source.n_vars)
        del reference
        clip = np.sqrt(len(training_control_rows))

        training_control_rows = np.sort(
            rng.choice(
                training_control_rows,
                min(train_controls, len(training_control_rows)),
                replace=False,
            )
        )
        heldout_control_rows = np.flatnonzero(
            obs.target_gene.eq("non-targeting") & obs.flex.eq(fold)
        )
        heldout_control_rows = np.sort(
            rng.choice(
                heldout_control_rows,
                min(heldout_controls, len(heldout_control_rows)),
                replace=False,
            )
        )
        training_control_counts = source.X[training_control_rows].tocsr()
        training_control_totals = np.asarray(
            training_control_counts.sum(axis=1)
        ).ravel()
        heldout_control_counts = source.X[heldout_control_rows].tocsr()
        heldout_control_totals = np.asarray(heldout_control_counts.sum(axis=1)).ravel()

        for target, genes in markers.items():
            columns = positions.loc[genes].to_numpy()
            training_target_rows = np.flatnonzero(
                obs.target_gene.eq(target) & obs.flex.ne(fold)
            )
            heldout_target_rows = np.flatnonzero(
                obs.target_gene.eq(target) & obs.flex.eq(fold)
            )
            training_target_counts = source.X[training_target_rows].tocsr()
            heldout_target_counts = source.X[heldout_target_rows].tocsr()
            training_target = residuals(
                training_target_counts,
                np.asarray(training_target_counts.sum(axis=1)).ravel(),
                columns,
                probabilities,
                theta,
                clip,
            )
            training_control = residuals(
                training_control_counts,
                training_control_totals,
                columns,
                probabilities,
                theta,
                clip,
            )
            difference = training_target.mean(axis=0) - training_control.mean(axis=0)
            variance = (
                training_target.var(axis=0, ddof=1)
                + training_control.var(axis=0, ddof=1)
            ) / 2
            weights = difference / (variance + ridge)

            heldout_target = residuals(
                heldout_target_counts,
                np.asarray(heldout_target_counts.sum(axis=1)).ravel(),
                columns,
                probabilities,
                theta,
                clip,
            )
            heldout_control = residuals(
                heldout_control_counts,
                heldout_control_totals,
                columns,
                probabilities,
                theta,
                clip,
            )
            target_score = heldout_target @ weights
            control_score = heldout_control @ weights
            center = control_score.mean()
            scale = control_score.std(ddof=1)
            target_score = (target_score - center) / scale
            control_score = (control_score - center) / scale
            threshold = np.quantile(control_score, 0.95)
            d_prime, auc = separation(target_score, control_score)
            fold_summary.append(
                {
                    "target": target,
                    "fold": f"Flex_{fold}",
                    "n_genes": len(columns),
                    "d_prime": d_prime,
                    "roc_auc": auc,
                    "control_like_fraction": np.mean(target_score <= threshold),
                }
            )
            score_frames.append(
                pd.DataFrame(
                    {
                        "cell_id": source.obs_names[heldout_target_rows],
                        "target": target,
                        "fold": f"Flex_{fold}",
                        "cell_group": "Labelled perturbed",
                        "fisher_score": target_score,
                        "response_like": target_score > threshold,
                    }
                )
            )
            score_frames.append(
                pd.DataFrame(
                    {
                        "cell_id": source.obs_names[heldout_control_rows],
                        "target": target,
                        "fold": f"Flex_{fold}",
                        "cell_group": "Control",
                        "fisher_score": control_score,
                        "response_like": control_score > threshold,
                    }
                )
            )
    return (
        pd.concat(score_frames, ignore_index=True),
        pd.DataFrame(fold_summary),
        excluded,
    )


def density(ax, scores, target, title, bins):
    data = scores if target is None else scores[scores.target == target]
    perturbed = data.cell_group == "Labelled perturbed"
    ax.hist(
        data.loc[~perturbed, "fisher_score"],
        bins=bins,
        density=True,
        color=BLUE,
        alpha=0.65,
        label="Controls",
    )
    ax.hist(
        data.loc[perturbed, "fisher_score"],
        bins=bins,
        density=True,
        color=ORANGE,
        alpha=0.55,
        label="Labelled perturbed",
    )
    threshold = (
        data.loc[~perturbed]
        .groupby(["target", "fold"], observed=True)
        .fisher_score.quantile(0.95)
        .median()
    )
    ax.axvline(
        threshold,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label="Control 95th percentile",
    )
    fraction = 1 - data.loc[perturbed, "response_like"].mean()
    detail = f"{fraction:.0%} control-like"
    ax.set(
        xlabel="Held-out Fisher score",
        ylabel="Density",
        title=f"{title}\n{detail}",
    )
    ax.spines[["top", "right"]].set_visible(False)


def plot_distributions(scores, output):
    perturbed = scores.cell_group == "Labelled perturbed"
    grouped = scores.loc[perturbed].groupby("target").response_like
    fractions = 1 - grouped.mean()
    well_sampled = fractions[grouped.size() >= 500]
    representatives = [
        fractions.idxmax(),
        (fractions - 0.5).abs().idxmin(),
        (well_sampled - 0.2).abs().idxmin(),
    ]
    shown = scores[scores.target.isin(representatives)]
    limits = np.quantile(
        pd.concat([scores.fisher_score, shown.fisher_score]), [0.005, 0.995]
    )
    bins = np.linspace(*limits, 70)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    density(
        axes[0, 0], scores, None, "All perturbation-specific directions pooled", bins
    )
    for ax, target in zip(axes.flat[1:], representatives, strict=True):
        density(ax, scores, target, target, bins)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fisher-score distributions before Mixscape")
    fig.savefig(output / "fisher_score_distributions.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/vcc2025_h1/adata_Training.h5ad"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--theta", type=float, default=100)
    parser.add_argument("--train-controls", type=int, default=2_000)
    parser.add_argument("--heldout-controls", type=int, default=500)
    parser.add_argument("--ridge", type=float, default=0.1)
    args = parser.parse_args()

    scores, fold_summary, excluded = calculate(
        args.input,
        args.output,
        args.theta,
        args.train_controls,
        args.heldout_controls,
        args.ridge,
    )
    plot_distributions(scores, args.output)
    summary = (
        fold_summary.groupby("target")
        .agg(
            n_genes=("n_genes", "first"),
            mean_d_prime=("d_prime", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_control_like_fraction=("control_like_fraction", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "fisher_summary.csv", index=False)
    pd.DataFrame(
        {"target": excluded, "reason": "no non-target stringent DE genes"}
    ).to_csv(args.output / "fisher_excluded.csv", index=False)
    scores.to_parquet(args.output / "fisher_scores.parquet", index=False)
    (args.output / "fisher_run.json").write_text(
        json.dumps(
            {
                "candidate_genes": "stringent DE genes excluding all 150 CRISPR targets",
                "direction": "cross-fitted diagonal Fisher using every eligible gene",
                "pearson_theta": args.theta,
                "ridge": args.ridge,
                "training_controls_per_fold": args.train_controls,
                "heldout_controls_per_fold": args.heldout_controls,
                "seed": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
