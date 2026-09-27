"""Plot Mixscape labels on the default UMAP and representative perturbation scores."""

import argparse
import gc
from pathlib import Path
from typing import ClassVar

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pertpy as pt
import pertpy.tools._perturbation_efficacy._mixscape as mixscape_module
from scipy.optimize import brentq

COLORS = {
    "excluded_control": "#D0D0D0",
    "NT": "#8C8C8C",
    "NP": "#4C78A8",
    "KO": "#F58518",
}


class RecordingMixture(mixscape_module.MixscapeGaussianMixture):
    fits: ClassVar[list] = []

    def fit(self, X, y=None):
        result = super().fit(X, y)
        self.scores = np.asarray(X).ravel().copy()
        self.fits.append(self)
        return result


def subset(shard, rows):
    rows = np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    return shard[rows].to_memory()


def plot_umap(embedding, calls, output):
    data = embedding[["cell_id", "UMAP1", "UMAP2"]].merge(
        calls[["cell_id", "mixscape_class_global"]], on="cell_id"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    for label in ("excluded_control", "NT", "NP", "KO"):
        cells = data.mixscape_class_global == label
        axes[0].scatter(
            data.loc[cells, "UMAP1"],
            data.loc[cells, "UMAP2"],
            s=0.45,
            alpha=0.5,
            color=COLORS[label],
            linewidth=0,
            rasterized=True,
            label=f"{label.replace('_', ' ')} ({cells.sum():,})",
        )
    axes[0].legend(markerscale=8, frameon=False, loc="upper right")
    axes[0].set_title("Revised labels")

    for ax, label in zip(axes[1:], ("NP", "KO"), strict=True):
        cells = data.mixscape_class_global == label
        ax.scatter(
            data.UMAP1,
            data.UMAP2,
            s=0.35,
            alpha=0.12,
            color="#BDBDBD",
            linewidth=0,
            rasterized=True,
        )
        ax.scatter(
            data.loc[cells, "UMAP1"],
            data.loc[cells, "UMAP2"],
            s=0.6,
            alpha=0.65,
            color=COLORS[label],
            linewidth=0,
            rasterized=True,
        )
        ax.set_title(f"{label} cells ({cells.sum():,})")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set(xticks=[], yticks=[], xlabel="UMAP1", ylabel="UMAP2")
        ax.spines[:].set_visible(False)
    fig.suptitle("Default UMAP with Mixscape classifications")
    fig.savefig(output / "mixscape_umap.png", dpi=180)
    plt.close(fig)


def select_guides(summary):
    eligible = summary[(summary.n_cells >= 500) & (summary.responder_fraction > 0)]
    rows = [
        eligible.loc[(eligible.responder_fraction - fraction).abs().idxmin()]
        for fraction in (0.3, 0.5, 0.9)
    ]
    return pd.DataFrame(rows)


def perturbation_scores(signature_dir, selected, control_cells):
    paths = sorted(signature_dir.glob("*.h5ad"))
    shards = [ad.read_h5ad(path, backed="r") for path in paths]
    rng = np.random.default_rng(0)
    per_batch = int(np.ceil(control_cells / len(shards)))
    control_parts = []
    for shard in shards:
        rows = np.flatnonzero(shard.obs.mixscape_perturbation.to_numpy() == "NT")
        control_parts.append(
            subset(shard, rng.choice(rows, min(per_batch, len(rows)), replace=False))
        )
    controls = ad.concat(control_parts, join="inner")[:control_cells].copy()
    mixscape = pt.tl.Mixscape()
    scores = []

    for row in selected.itertuples():
        parts = [
            subset(
                shard,
                shard.obs.mixscape_perturbation.to_numpy() == row.guide_id,
            )
            for shard in shards
            if (shard.obs.mixscape_perturbation == row.guide_id).any()
        ]
        work = ad.concat([ad.concat(parts, join="inner"), controls], join="inner")
        original_mixture = mixscape_module.MixscapeGaussianMixture
        RecordingMixture.fits = []
        mixscape_module.MixscapeGaussianMixture = RecordingMixture
        try:
            mixscape.mixscape(
                work,
                pert_key="mixscape_perturbation",
                control="NT",
                layer="X_pert",
                test_method="wilcoxon",
                split_by=None,
                perturbation_type="KO",
                random_state=0,
            )
        finally:
            mixscape_module.MixscapeGaussianMixture = original_mixture

        model = RecordingMixture.fits[-1]
        control_mean, perturbed_mean = model.means_.ravel()
        threshold = brentq(
            lambda value, fitted=model: fitted.predict_proba([[value]])[0, 1] - 0.5,
            control_mean,
            perturbed_mean,
        )
        score = pd.DataFrame(
            {
                "cell_id": work.obs_names,
                "pvec": model.scores,
                "population": np.where(
                    work.obs.mixscape_perturbation == "NT", "Control", "Perturbed"
                ),
            }
        )
        score["target"] = row.target_gene
        score["responder_fraction"] = row.responder_fraction
        score["threshold"] = threshold
        scores.append(score)
        del work, parts
        gc.collect()
        print(f"Scored {row.target_gene}", flush=True)
    return pd.concat(scores, ignore_index=True)


def plot_scores(scores, output):
    targets = scores.target.drop_duplicates().tolist()
    limits = scores.pvec.quantile([0.005, 0.995]).to_numpy()
    bins = np.linspace(*limits, 65)
    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4), sharex=True, constrained_layout=True
    )
    for ax, target in zip(axes, targets, strict=True):
        data = scores[scores.target == target]
        for label, color in (("Control", COLORS["NT"]), ("Perturbed", COLORS["KO"])):
            values = data.loc[data.population == label, "pvec"]
            ax.hist(
                values,
                bins=bins,
                histtype="stepfilled",
                alpha=0.5,
                color=color,
                label=f"{label} (n={len(values):,})",
            )
        ax.axvline(
            data.threshold.iloc[0],
            color="#202020",
            linestyle="--",
            linewidth=1.5,
            label="Mixscape threshold",
        )
        fraction = data.responder_fraction.iloc[0]
        ax.set(
            title=f"{target}\n{fraction:.0%} KO-like",
            xlabel="Mixscape perturbation score",
            ylabel="Cells per bin",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Raw cell counts along representative Mixscape directions")
    fig.savefig(output / "mixscape_perturbation_scores.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signatures",
        type=Path,
        default=Path("data/derived/vcc2025_h1_mixscape/signatures"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--control-cells", type=int, default=5_000)
    args = parser.parse_args()

    embedding = pd.read_parquet(args.output / "embedding.parquet")
    calls = pd.read_parquet(args.output / "mixscape_cells.parquet")
    summary = pd.read_csv(args.output / "mixscape_guides.csv")
    selected = select_guides(summary)
    plot_umap(embedding, calls, args.output)
    scores = perturbation_scores(args.signatures, selected, args.control_cells)
    plot_scores(scores, args.output)


if __name__ == "__main__":
    main()
