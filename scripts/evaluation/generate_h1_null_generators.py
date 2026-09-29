"""Null tests for alternative cell generators on the H1 harness: no perturbation at all.

A generator is only usable if unchanged controls passed through it score like real controls
(about -0.044 natively, -0.30 calibrated to 2026). Every prediction reuses the canonical 400
H1 control cells per target (same cells as `vcc-h1 score-control-baseline`) for their library
sizes and, for metacells, their state.

* pooled_nb:        counts ~ NB(N_c * p, r_g), p the pooled control profile, r_g the gene's
                    dispersion estimated across all controls.
* metacell_poisson: counts ~ Poisson(N_c * p_m), p_m the profile of the cell's metacell
                    (k-means on 30 PCs of ln(1 + CPM/20) over 2,000 variable genes, ~100 cells).
* metacell_nb:      counts ~ NB(N_c * p_m, r_g within metacells).
* knn_poisson:      counts ~ Poisson(N_c * p_k), p_k the pooled profile of the cell's k nearest
                    neighbours in the same PCA space (the cell itself excluded).

Distribution check: for genes at 0.2-1 control counts per cell, each block's zero fraction and
variance are compared with the 400 real cells it replaces.
Run: pixi run python -m scripts.evaluation.generate_h1_null_generators KIND OUT.h5ad
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

from scripts.evaluation.generate_replogle_h1_phase1 import CONTROLS, sha256
from scripts.evaluation.h1_generation import ROOT, canonical_axes
from scripts.linear_response.kernel import write_prediction

REPORT = ROOT / "reports/h1-harness-review/generators"
METACELL_SIZE = 100
N_HVG, N_PCS = 2000, 30


def dispersion(X, lib, profile_of_rows, chunk=4000):
    """Method-of-moments NB dispersion per gene, streamed over cells:
    phi_g = sum((y - mu)^2 - mu) / sum(mu^2), with mu = N_c * profile."""
    num = np.zeros(X.shape[1])
    den = np.zeros_like(num)
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        y = X[start:stop].toarray()
        mu = lib[start:stop, None] * profile_of_rows(start, stop)
        num += (np.square(y - mu) - mu).sum(axis=0)
        den += np.square(mu).sum(axis=0)
    phi = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.clip(phi, 1e-6, 1e3)


def nb_draw(rng, mean, phi):
    r = 1.0 / phi
    return rng.negative_binomial(r, r / (r + mean))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["pooled_nb", "metacell_poisson", "metacell_nb", "knn_poisson"])
    parser.add_argument("--k", type=int, default=30, help="neighbours for knn_poisson")
    parser.add_argument("output")
    args = parser.parse_args()
    targets, genes, rows = canonical_axes()
    controls = ad.read_h5ad(CONTROLS)
    X = sparse.csr_matrix(controls.X).astype(np.float64)
    lib = np.asarray(X.sum(axis=1)).ravel()
    pooled = np.asarray(X.sum(axis=0)).ravel()
    p = pooled / pooled.sum()
    per_cell = pooled / X.shape[0]
    sparse_band = (per_cell > 0.2) & (per_cell <= 1.0)

    if args.kind == "pooled_nb":
        phi = dispersion(X, lib, lambda a, b: p[None, :])
        member = None
    else:
        norm = X.multiply(5e4 / lib[:, None]).tocsr()
        norm.data = np.log1p(norm.data)
        mean = np.asarray(norm.mean(axis=0)).ravel()
        var = np.asarray(norm.multiply(norm).mean(axis=0)).ravel() - mean**2
        hvg = np.argsort(-var)[:N_HVG]
        dense = norm[:, hvg].toarray() - mean[hvg]
        u, s, _ = np.linalg.svd(dense, full_matrices=False)
        pcs = u[:, :N_PCS] * s[:N_PCS]
        if args.kind == "knn_poisson":
            from sklearn.neighbors import NearestNeighbors

            chosen = np.unique(rows.ravel())
            nn = NearestNeighbors(n_neighbors=args.k + 1).fit(pcs)
            _, idx = nn.kneighbors(pcs[chosen])
            neighbour = {int(c): row[row != c][: args.k] for c, row in zip(chosen, idx, strict=True)}
            member = None
        k = X.shape[0] // METACELL_SIZE
        if args.kind != "knn_poisson":
            member = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3).fit_predict(pcs)
            indicator = sparse.csr_matrix((np.ones(X.shape[0]), (member, np.arange(X.shape[0]))), shape=(k, X.shape[0]))
            profiles = np.asarray((indicator @ X).todense())
            profiles /= profiles.sum(axis=1, keepdims=True)
            if args.kind == "metacell_nb":
                phi = dispersion(X, lib, lambda a, b: profiles[member[a:b]])
            sizes = np.bincount(member)
            print(f"{k} metacells, size median {int(np.median(sizes))} (range {sizes.min()}-{sizes.max()})", flush=True)

    blocks, dz, vr = {}, [], []
    for t_index, t in enumerate(targets):
        rng = np.random.default_rng(zlib.crc32(f"{args.kind}{args.k if args.kind == 'knn_poisson' else ''}:{t}".encode()))
        cells = rows[t_index]
        real = X[cells].toarray()
        n = lib[cells, None]
        if args.kind == "knn_poisson":  # each cell's neighbourhood profile, one sparse product
            cols = np.concatenate([neighbour[int(c)] for c in cells])
            owner = np.repeat(np.arange(len(cells)), args.k)
            W = sparse.csr_matrix((np.ones(len(cols)), (owner, cols)), shape=(len(cells), X.shape[0]))
            prof = np.asarray((W @ X).todense())
            mean = n * prof / prof.sum(axis=1, keepdims=True)
        else:
            mean = n * (p[None, :] if member is None else profiles[member[cells]])
        if args.kind in ("metacell_poisson", "knn_poisson"):
            counts = rng.poisson(mean)
        else:
            counts = nb_draw(rng, mean, phi[None, :])
        blocks[t] = sparse.csr_matrix(counts.astype(np.int32))
        dz.append(np.mean((counts[:, sparse_band] == 0).mean(0) - (real[:, sparse_band] == 0).mean(0)))
        vr.append(np.median(counts[:, sparse_band].var(0) / np.maximum(real[:, sparse_band].var(0), 1e-12)))
    write_prediction(blocks, genes.tolist(), Path(args.output))
    summary = {
        "generator": args.kind,
        "metacell_size": args.k if args.kind == "knn_poisson" else (None if member is None else METACELL_SIZE),
        "sparse_band_genes": int(sparse_band.sum()),
        "mean_zero_fraction_difference_vs_real": float(np.mean(dz)),
        "median_variance_ratio_vs_real": float(np.median(vr)),
        "prediction_sha256": sha256(Path(args.output)),
    }
    out = REPORT / (f"knn_poisson_k{args.k}" if args.kind == "knn_poisson" else args.kind)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
