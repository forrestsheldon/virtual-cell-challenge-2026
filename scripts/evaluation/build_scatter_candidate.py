"""The a = 1 available-source transfer on H1, with sparse increases added as scattered counts.

Same prediction as `build_available_average_candidate --pseudocount 20 --direct`; only the
cells differ. The canonical 400 H1 control cells per target are transformed as in
h1_generation (factor re-aimed at the sample, multiplicative scaling, gene-wise rounding),
except for sparse genes (pooled control mean below SPARSE counts per cell) whose predicted
total rises. Those keep every cell's observed count and receive the extra counts

    dT_g = N q_g - Y_g   (N, Y_g: the sample's total library and gene count; q_g predicted)

scattered multinomially over cells with probability N_c / N, dT_g stochastically rounded.
With --match-first, sample matching stays multiplicative for every gene and only the
perturbation is scattered: dT_g = N (q_g - p_g), p_g the pooled control share. With no
predicted change nothing is scattered, so the null is the multiplicative generator's.
Multiplicative scaling cannot turn a zero into a count; scattering can, at the rate extra
molecules would land in each cell. Decreases thin naturally and stay multiplicative.
Run: pixi run python -m scripts.evaluation.build_scatter_candidate OUT.h5ad [--null]
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import scripts.evaluation.h1_transfer_regression as reg
from scripts.evaluation.build_quantile_mapping_candidate import SPARSE, available_factor, truth_blocks
from scripts.evaluation.generate_replogle_h1_phase1 import CONTROLS, sha256
from scripts.evaluation.h1_generation import decode_lfc_genewise, expected_counts, harness_log
from scripts.evaluation.kolf_h1_transfer_rules import cpm
from scripts.linear_response.kernel import write_prediction

REPORT = reg.ROOT / "reports/h1-transfer-regression/cell_eval_v3/available_avg_direct_c20_scatter"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--null", action="store_true", help="no perturbation: the generator's null test")
    parser.add_argument("--match-first", action="store_true", help="scatter the perturbation only")
    args = parser.parse_args()
    report = REPORT.with_name("scatter_null") if args.null else REPORT
    if args.match_first:
        report = report.with_name(report.name.replace("scatter", "scatter_mf"))
    report.mkdir(parents=True, exist_ok=True)
    targets, genes, rows, factor, profile = available_factor()
    controls = ad.read_h5ad(CONTROLS)
    X = sparse.csr_matrix(controls.X).astype(np.float64)
    lib = np.asarray(X.sum(axis=1)).ravel()
    pooled_counts = np.asarray(X.sum(axis=0)).ravel()
    per_cell = pooled_counts / X.shape[0]
    sparse_genes = (per_cell > 0) & (per_cell < SPARSE)
    band = (per_cell > 0.2) & (per_cell <= 1.0)
    pooled = cpm(pooled_counts)
    if args.null:
        factor[:] = 0.0
        profile[:] = pooled / pooled.sum()
    truth = truth_blocks(targets)

    blocks, cos, diag, scattered = {}, [], [], []
    for k, t in enumerate(targets):
        raw = X[rows[k]].tocsr()
        sample_counts = np.asarray(raw.sum(axis=0)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(sample_counts > 0, np.log2(pooled / cpm(sample_counts)), 0.0)
        lfc = factor[k] + np.where(np.isfinite(shift), shift, 0.0)
        rng = np.random.default_rng(zlib.crc32(f"scatter:{t}".encode()))
        y = raw.toarray().astype(np.int64)
        n = lib[rows[k]]
        if args.match_first:  # match multiplicatively; scatter only the rise over the pooled share
            extra = n.sum() * (profile[k] - pooled / pooled.sum())
            up = np.flatnonzero(sparse_genes & (extra > 0))
            lfc[up] -= factor[k, up]
            counts = decode_lfc_genewise(raw, lfc, rng).toarray().astype(np.int64)
            base_up = counts[:, up]
        else:
            counts = decode_lfc_genewise(raw, lfc, rng).toarray().astype(np.int64)
            extra = n.sum() * profile[k] - sample_counts
            up = np.flatnonzero(sparse_genes & (extra > 0))
            base_up = y[:, up]
        add = np.floor(extra[up]) + (rng.random(len(up)) < extra[up] % 1)
        counts[:, up] = base_up + rng.multinomial(add.astype(np.int64), n / n.sum()).T
        scattered.append(len(up))
        blocks[t] = sparse.csr_matrix(counts.astype(np.int32))
        base = harness_log(sample_counts.astype(float))
        ideal = expected_counts(raw, lfc).sum(0)
        ideal[up] = ideal[up] + extra[up] if args.match_first else n.sum() * profile[k, up]
        want = harness_log(ideal) - base
        real = harness_log(counts.sum(0).astype(float)) - base
        if not want.any():
            continue
        cos.append(float(want @ real / (np.linalg.norm(want) * np.linalg.norm(real))))
        tr = truth[t].toarray()
        ctrl_mean = y[:, band].mean(0)
        for label, block in (("truth", tr), ("scatter", counts)):
            m = block[:, band].mean(0)
            rise = m >= 1.5 * np.maximum(ctrl_mean, 1e-9)
            diag.append(pd.DataFrame({"generator": label, "mean_per_cell": m[rise],
                                      "zero_fraction": (block[:, band] == 0).mean(0)[rise]}))
    write_prediction(blocks, genes.tolist(), Path(args.output))
    d = pd.concat(diag)
    d["mean_bin"] = pd.cut(d.mean_per_cell, [0, 0.5, 1, 2, 4, 1e9])
    curve = d.groupby(["mean_bin", "generator"], observed=True).zero_fraction.agg(["mean", "count"]).unstack("generator")
    curve.to_csv(report / "zero_fraction_vs_mean_increases.csv")
    summary = {"median_genes_scattered": float(np.median(scattered)), "fidelity_median_cosine": float(np.median(cos)),
               "prediction_sha256": sha256(Path(args.output))}
    assert summary["fidelity_median_cosine"] >= 0.95, summary
    (report / "generation.json").write_text(json.dumps(summary, indent=1))
    print(curve.round(3).to_string(), flush=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
