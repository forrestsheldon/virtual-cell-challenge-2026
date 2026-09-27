"""Run non-targeting guides through Mixscape as pseudo-perturbations."""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def null_signature(
    shard,
    pseudo_guide,
    excluded_guides,
    control_cells,
    rng,
    n_neighbors,
    n_dimensions,
):
    obs = shard.obs
    nt = obs.target_gene.to_numpy() == "non-targeting"
    pseudo = nt & (obs.guide_id.astype(str).to_numpy() == pseudo_guide)
    excluded = obs.guide_id.astype(str).isin(excluded_guides).to_numpy()
    reference_rows = np.flatnonzero(nt & ~pseudo & ~excluded)
    pseudo_rows = np.flatnonzero(pseudo)
    sampled_rows = rng.choice(
        reference_rows, min(control_cells, len(reference_rows)), replace=False
    )
    query_rows = np.r_[pseudo_rows, sampled_rows]

    reference_pcs = np.asarray(shard.obsm["X_pca"])[reference_rows, :n_dimensions]
    query_pcs = np.asarray(shard.obsm["X_pca"])[query_rows, :n_dimensions]
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(reference_pcs)
    indices = neighbors.kneighbors(query_pcs, return_distance=False)

    reference = shard.X[reference_rows].tocsr().copy()
    np.expm1(reference.data, out=reference.data)
    row_indices = np.repeat(np.arange(len(query_rows)), n_neighbors)
    weights = sparse.csr_matrix(
        (
            np.full(len(row_indices), 1 / n_neighbors, dtype=np.float32),
            (row_indices, indices.ravel()),
        ),
        shape=(len(query_rows), len(reference_rows)),
    )
    neighbor_mean = weights @ reference
    np.log1p(neighbor_mean.data, out=neighbor_mean.data)
    expression = shard.X[query_rows].tocsr()

    query_obs = obs.iloc[query_rows].copy()
    query_obs["null_perturbation"] = np.where(pseudo[query_rows], pseudo_guide, "NT")
    result = ad.AnnData(X=expression, obs=query_obs, var=shard.var.copy())
    result.layers["X_pert"] = neighbor_mean - expression
    return result[: len(pseudo_rows)], result[len(pseudo_rows) :]


def run_guide(
    shards, guide, excluded_guides, control_cells, n_neighbors, n_dimensions
):
    rng = np.random.default_rng(0)
    per_batch = int(np.ceil(control_cells / len(shards)))
    target_parts = []
    control_parts = []
    for shard in shards:
        target, controls = null_signature(
            shard,
            guide,
            excluded_guides,
            per_batch,
            rng,
            n_neighbors,
            n_dimensions,
        )
        target_parts.append(target)
        control_parts.append(controls)

    target = ad.concat(target_parts, join="inner")
    controls = ad.concat(control_parts, join="inner")[:control_cells]
    work = ad.concat([target, controls], join="inner")
    pt.tl.Mixscape().mixscape(
        work,
        pert_key="null_perturbation",
        control="NT",
        layer="X_pert",
        test_method="wilcoxon",
        split_by=None,
        perturbation_type="KO",
        random_state=0,
    )
    result = work.obs.iloc[: len(target)].copy()
    result["cell_id"] = result.index
    result["pseudo_guide"] = guide
    result["direction_available"] = guide in work.uns["mixscape"]
    return result


def load_control_shards(paths):
    backed = [ad.read_h5ad(path, backed="r") for path in paths]
    shards = []
    for shard in backed:
        rows = np.flatnonzero(shard.obs.target_gene.to_numpy() == "non-targeting")
        batch = ad.AnnData(
            X=shard.X[rows].tocsr(),
            obs=shard.obs.iloc[rows].copy(),
            var=shard.var.copy(),
        )
        batch.obsm["X_pca"] = np.asarray(shard.obsm["X_pca"])[rows]
        shards.append(batch)
    for shard in backed:
        shard.file.close()
    return shards


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
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--dimensions", type=int, default=15)
    parser.add_argument("--guide")
    parser.add_argument("--exclude-guides", nargs="*", default=[])
    parser.add_argument("--prefix", default="mixscape_null")
    args = parser.parse_args()

    paths = sorted(args.signatures.glob("*.h5ad"))
    shards = load_control_shards(paths)
    obs = pd.concat([shard.obs for shard in shards])
    guides = (
        obs.loc[obs.target_gene == "non-targeting", "guide_id"]
        .astype(str)
        .value_counts()
        .sort_index()
    )
    guides = guides.loc[~guides.index.isin(args.exclude_guides)]
    if args.guide:
        guides = guides.loc[[args.guide]]

    calls = []
    summaries = []
    started = time.monotonic()
    for number, (guide, n_cells) in enumerate(guides.items(), 1):
        result = run_guide(
            shards,
            guide,
            args.exclude_guides,
            args.control_cells,
            args.neighbors,
            args.dimensions,
        )
        ko = result.mixscape_class_global == "KO"
        posterior = result.mixscape_class_p_ko
        calls.append(
            result[
                [
                    "cell_id",
                    "pseudo_guide",
                    "batch",
                    "mixscape_class_global",
                    "mixscape_class_p_ko",
                    "direction_available",
                ]
            ]
        )
        summaries.append(
            {
                "pseudo_guide": guide,
                "n_cells": n_cells,
                "n_batches": result.batch.nunique(),
                "direction_available": result.direction_available.iloc[0],
                "n_ko": int(ko.sum()),
                "ko_fraction": ko.mean(),
                "maximum_posterior": posterior.max(),
            }
        )
        elapsed = time.monotonic() - started
        eta = elapsed / number * (len(guides) - number) / 60
        print(
            f"[{number}/{len(guides)}] {guide}: {ko.sum()} KO, ETA {eta:.1f} min",
            flush=True,
        )

    cells = pd.concat(calls, ignore_index=True)
    summary = pd.DataFrame(summaries)
    cells.to_parquet(args.output / f"{args.prefix}_cells.parquet", index=False)
    summary.to_csv(args.output / f"{args.prefix}_guides.csv", index=False)
    (args.output / f"{args.prefix}_run.json").write_text(
        json.dumps(
            {
                "pseudo_guides": len(guides),
                "signature_shards": len(shards),
                "control_cells_per_guide": args.control_cells,
                "neighbors": args.neighbors,
                "pca_dimensions": args.dimensions,
                "neighbor_reference": "other non-targeting guides in the same batch",
                "excluded_guides": args.exclude_guides,
                "neighbor_search": f"exact Euclidean distance in PC1-PC{args.dimensions}",
                "test_method": "wilcoxon",
                "classification_split": None,
                "seed": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
