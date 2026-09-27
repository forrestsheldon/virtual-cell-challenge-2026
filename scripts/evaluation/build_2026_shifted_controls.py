"""Prepare a 2026 zero-shot submission: each context's control cells shifted by the
global perturbation axis learned on H1.

2026 is zero-shot -- only non-targeting controls are supplied -- so the only
perturbation direction available is the one learned on 2025 H1:
    Delta_g = mean_perturbed CPM_g - mean_control CPM_g   (over the H1 data)
mapped onto the 2026 gene axis by gene name (2026-only genes get no shift). For each
context A/B/C and each of the 300 targets, 400 control cells are drawn, shifted by
Delta in CPM, and written back as integer counts. This is exactly the shifted-control
baseline the H1 harness estimated the value of, now applied to the real val panel.

Counts are emitted at a capped depth (default 5000 UMI/cell) so the 360k-cell file is
tractable to build, prep, and upload; cell-eval2 normalizes per group, so the score is
depth-invariant. Each context is built in memory and streamed into the combined h5ad
with h5py so peak memory is one context, not the whole panel.
"""

import argparse
import zlib
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = "non-targeting"
CELLS_PER_TARGET = 400
CONTEXTS = ["A", "B", "C"]


def h1_global_axis(h1_path, gene_names_2026, cache, chunk=20_000):
    """Delta = mean_pert CPM - mean_ctrl CPM over H1, reindexed to the 2026 genes."""
    if cache.exists():
        return np.load(cache)
    adata = ad.read_h5ad(h1_path, backed="r")
    is_ctrl = adata.obs["target_gene"].astype(str).to_numpy() == CONTROL
    n = adata.n_vars
    s_pert, s_ctrl, n_pert, n_ctrl = np.zeros(n), np.zeros(n), 0, 0
    for start in range(0, adata.n_obs, chunk):
        block = adata.X[start:start + chunk].tocsr().astype(np.float64)
        lib = np.asarray(block.sum(axis=1)).ravel()
        cpm = block.multiply(1e6 / lib[:, None]).tocsr()
        c = is_ctrl[start:start + chunk]
        s_ctrl += np.asarray(cpm[c].sum(axis=0)).ravel(); n_ctrl += int(c.sum())
        s_pert += np.asarray(cpm[~c].sum(axis=0)).ravel(); n_pert += int((~c).sum())
    delta_h1 = s_pert / n_pert - s_ctrl / n_ctrl
    on_2026 = pd.Series(delta_h1, index=adata.var_names.to_numpy()).reindex(gene_names_2026).fillna(0.0)
    out = on_2026.to_numpy()
    np.save(cache, out)
    print(f"global axis: {np.count_nonzero(out)} of {len(out)} 2026 genes shifted; "
          f"L1 {np.abs(out).sum():.0f} CPM", flush=True)
    return out


def shifted_counts(X, delta, depth, seed):
    """Shift control CPM by delta, clip, and emit integer counts capped at `depth`."""
    lib = np.asarray(X.sum(axis=1)).ravel()
    cpm = np.asarray(X.todense()) * (1e6 / lib[:, None])
    shifted = np.clip(cpm + delta[None, :], 0.0, None)
    target_lib = np.minimum(lib, depth)                       # cap deep cells, keep shallow ones
    counts = np.rint(shifted * (target_lib[:, None] / 1e6)).astype(np.int64)
    return sp.csr_matrix(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls-dir", type=Path, default=Path("data/controls"))
    parser.add_argument("--h1", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/vcc2026_submission/shifted_controls.h5ad"))
    parser.add_argument("--depth", type=int, default=5000)
    parser.add_argument("--limit-targets", type=int, help="build only the first N targets (smoke test)")
    args = parser.parse_args()

    genes = pd.read_csv(args.controls_dir / "gene_names.csv")["gene_name"].tolist()
    targets = pd.read_csv(args.controls_dir / "pert_counts.csv")["target_gene"].tolist()
    if args.limit_targets:
        targets = targets[: args.limit_targets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    delta = h1_global_axis(args.h1, genes, args.output.parent / "h1_global_axis_2026genes.npy")

    n_cells = len(CONTEXTS) * len(targets) * CELLS_PER_TARGET
    with h5py.File(args.output, "w") as f:
        Xg = f.create_group("X")
        data = Xg.create_dataset("data", (0,), maxshape=(None,), dtype="float32", chunks=(2**20,))
        indices = Xg.create_dataset("indices", (0,), maxshape=(None,), dtype="int32", chunks=(2**20,))
        indptr = [0]
        obs_target, obs_context = [], []

        for context in CONTEXTS:
            controls = ad.read_h5ad(args.controls_dir / f"context_{context}.h5ad")  # in memory: fast per-target indexing
            pool = np.arange(controls.n_obs)
            ctx_blocks = []
            for target in targets:
                seed = zlib.crc32(f"shift2026:{context}:{target}".encode())
                rows = np.sort(pool[np.random.default_rng(seed).choice(len(pool), CELLS_PER_TARGET, replace=False)])
                ctx_blocks.append(shifted_counts(controls.X[rows].tocsr().astype(np.float64), delta, args.depth, seed))
                obs_target.extend([target] * CELLS_PER_TARGET)
                obs_context.extend([context] * CELLS_PER_TARGET)
            del controls
            block = sp.vstack(ctx_blocks, format="csr")       # this context only, in memory
            data.resize((data.shape[0] + block.data.shape[0],)); data[-block.data.shape[0]:] = block.data
            indices.resize((indices.shape[0] + block.indices.shape[0],)); indices[-block.indices.shape[0]:] = block.indices
            indptr.extend((np.asarray(indptr[-1]) + block.indptr[1:]).tolist())
            print(f"context {context}: {block.shape[0]} cells, density {block.nnz/(block.shape[0]*block.shape[1]):.1%}", flush=True)
            del block, ctx_blocks

        Xg.create_dataset("indptr", data=np.asarray(indptr, dtype="int64"))
        Xg.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                         "shape": np.array([n_cells, len(genes)], dtype="int64")})
        f.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})

        # obs / var as anndata-encoded dataframes
        def write_df(group_name, columns, index, index_name):
            g = f.create_group(group_name)
            g.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0",
                            "_index": index_name, "column-order": list(columns)})
            idx = g.create_dataset(index_name, data=np.asarray(index, dtype=object),
                                   dtype=h5py.string_dtype())
            idx.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})
            for name, values in columns.items():
                d = g.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype())
                d.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})

        write_df("obs", {"target_gene": obs_target, "context": obs_context},
                 [f"cell_{i}" for i in range(n_cells)], "_index")
        write_df("var", {}, genes, "_index")

    size_gb = args.output.stat().st_size / 1e9
    print(f"wrote {args.output}: {n_cells} cells x {len(genes)} genes, {size_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
