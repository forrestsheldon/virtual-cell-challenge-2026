"""Compute batch-local Mixscape signatures for the VCC 2025 H1 screen."""

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt

EXCLUDED_CONTROL_GUIDES = [
    "non-targeting_00018|non-targeting_00127",
    "non-targeting_00026|non-targeting_02263",
    "non-targeting_00047|non-targeting_00882",
    "non-targeting_00062|non-targeting_02518",
    "non-targeting_00121|non-targeting_00339",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/vcc2025_h1/adata_Training.h5ad"),
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path("reports/crispri-h1-exploration/generated/embedding.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/vcc2025_h1_mixscape/signatures"),
    )
    parser.add_argument("--batch")
    parser.add_argument(
        "--exclude-control-guides", nargs="*", default=EXCLUDED_CONTROL_GUIDES
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.input, backed="r")
    pc_columns = [f"PC{i}" for i in range(1, 51)]
    embedding = pd.read_parquet(args.embedding, columns=["cell_id", *pc_columns])
    embedding = embedding.set_index("cell_id").loc[source.obs_names]
    batches = [args.batch] if args.batch else sorted(source.obs.batch.unique())
    mixscape = pt.tl.Mixscape()

    for number, batch in enumerate(batches, 1):
        destination = args.output / f"{batch}.h5ad"
        if destination.exists():
            print(f"[{number}/{len(batches)}] {batch}: already complete", flush=True)
            continue

        started = time.monotonic()
        rows = np.flatnonzero(source.obs.batch.to_numpy() == batch)
        counts = source.X[rows].tocsr()
        totals = np.asarray(counts.sum(axis=1)).ravel()
        counts.data *= np.repeat(10_000 / totals, np.diff(counts.indptr))
        np.log1p(counts.data, out=counts.data)

        obs = source.obs.iloc[rows].copy()
        excluded = (obs.target_gene == "non-targeting") & obs.guide_id.astype(
            str
        ).isin(args.exclude_control_guides)
        obs["mixscape_perturbation"] = np.select(
            [excluded, obs.target_gene == "non-targeting"],
            ["excluded_control", "NT"],
            default=obs.guide_id.astype(str),
        )
        result = ad.AnnData(X=counts, obs=obs, var=source.var.copy())
        result.obsm["X_pca"] = embedding.iloc[rows][pc_columns].to_numpy()
        mixscape.perturbation_signature(
            result,
            pert_key="mixscape_perturbation",
            control="NT",
            use_rep="X_pca",
            n_dims=15,
            n_neighbors=20,
            random_state=0,
        )
        result.uns["mixscape_signature"] = {
            "batch": batch,
            "normalization_target": 10_000,
            "neighbors": 20,
            "pca_dimensions": 15,
            "control": "NT",
            "excluded_control_guides": args.exclude_control_guides,
            "perturbation_key": "mixscape_perturbation",
        }
        result.write_h5ad(destination, compression="gzip")
        minutes = (time.monotonic() - started) / 60
        size = destination.stat().st_size / 1024**3
        print(
            f"[{number}/{len(batches)}] {batch}: {minutes:.1f} min, {size:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
