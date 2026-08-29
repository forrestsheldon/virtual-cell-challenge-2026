"""Build a deterministic, construct-balanced control-resampling baseline."""

import argparse
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CELLS_PER_TARGET = 400
TARGETS_PER_CHUNK = 10


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_rows(obs: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    guides = sorted(obs["ntc_id"].unique())
    base, remainder = divmod(CELLS_PER_TARGET, len(guides))
    extras = set(rng.choice(len(guides), remainder, replace=False))
    rows = []
    for index, guide in enumerate(guides):
        available = np.flatnonzero(obs["ntc_id"].to_numpy() == guide)
        count = base + (index in extras)
        rows.extend(rng.choice(available, count, replace=False))
    rng.shuffle(rows)
    return np.asarray(rows)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(args.controls / "gene_names.csv").iloc[:, 0].astype(str)
    targets = pd.read_csv(args.controls / "pert_counts.csv")["target_gene"].astype(str)
    paths = sorted(args.controls.glob("context_*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"no context_*.h5ad files in {args.controls}")

    chunks, total_nnz = [], 0
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        temporary = Path(temporary)
        for context_index, path in enumerate(paths):
            data = ad.read_h5ad(path)
            context = str(data.obs["context"].iloc[0])
            if data.var_names.tolist() != genes.tolist():
                raise ValueError(f"gene order mismatch in {path}")
            if data.obs["target_gene"].ne("non-targeting").any():
                raise ValueError(f"non-control rows found in {path}")

            for start in range(0, len(targets), TARGETS_PER_CHUNK):
                target_block = targets.iloc[start : start + TARGETS_PER_CHUNK]
                rows, labels = [], []
                for target_index, target in enumerate(target_block, start=start):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([args.seed, context_index, target_index])
                    )
                    rows.extend(sample_rows(data.obs, rng))
                    labels.extend([target] * CELLS_PER_TARGET)
                counts = data.X[np.asarray(rows)].tocsr().astype(np.int32)
                counts.eliminate_zeros()
                total_nnz += counts.nnz
                obs = pd.DataFrame({"target_gene": labels, "context": context})
                obs.index = [f"{context}_{start + i // 400:03d}_{i % 400:03d}" for i in range(len(obs))]
                block = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=genes))
                chunk = temporary / f"{context}_{start:03d}.h5ad"
                block.write_h5ad(chunk, compression="gzip")
                chunks.append(chunk)
            print(f"Prepared {len(targets):,} targets for context {context}")

        if total_nnz > 4_750_000_000:
            raise ValueError(f"prediction has {total_nnz:,} stored entries")
        ad.experimental.concat_on_disk(
            chunks, args.output, max_loaded_elems=25_000_000, merge="same"
        )
    print(f"Wrote {len(paths) * len(targets) * CELLS_PER_TARGET:,} cells")
    print(f"Stored entries: {total_nnz:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
