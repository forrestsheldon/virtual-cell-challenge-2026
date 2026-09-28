"""VCC 2026 submission: transfer the five-screen average perturbation change onto each context.

Donors: K562 GWPS (Replogle 2022, raw sums reconstructed from the author means), HCT116 and
HEK293T (X-Atlas/Orion), primary CD4+ T (Zhu 2026) and KOLF2.1J (Nourreddine). For target t
and gene g, each donor's change is phi_20 = log((y_p + 20) / (y_0 + 20)), with CPM computed on
that donor's genes that are on the 2026 axis. The prediction averages the change over the
donors that measure both t and g; missing entries are not zeros, and a gene no donor measures
keeps no change. For each context, x_hat = (x_0 + 20) * exp(a * dbar) - 20, floored at zero and
renormalized, with x_0 the context's pooled control CPM (c = 20 matches the scorer's
ln(1 + CPM/20) space; see Baselines II).

Cells: for every target-context pair, the same ntc_id-balanced 400 control cells as the
control-resampling baseline (scripts/baselines/resample_controls.py, seed 0). The per-gene
factor is re-aimed at those cells' own pseudobulk and integerized gene-wise
(scripts/evaluation/h1_generation.decode_lfc_genewise), so each pseudobulk hits the intended
profile to within one count per gene. Native control depth; no depth inflation.
Run: pixi run python -m scripts.baselines.transfer_average OUT.h5ad --scale A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.baselines.resample_controls import (
    CELLS_PER_TARGET,
    TARGETS_PER_CHUNK,
    sample_rows,
)
from scripts.evaluation.context_transfer_blog import collapse_columns
from scripts.evaluation.context_transfer_h1 import SOURCE_SPECS
from scripts.evaluation.h1_generation import (
    decode_lfc_genewise,
    expected_counts,
    harness_log,
)
from scripts.evaluation.prepare_replogle_h1_transfer import (
    CONTROL,
    REPLOGLE,
    collapse_duplicate_symbols,
    reconstruct_replogle_rows,
    target_from_population,
)

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ROOT / "data/controls"
C = 20.0
SEED = 0  # control-cell selection, identical to control-resampling-seed0
MAX_ENTRIES = 4_750_000_000
MIN_COSINE = 0.95
GLOBAL_DELTA = ROOT / "data/derived/vcc2026_h1/global_shift_delta.npz"
H1_PHASE1 = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
H1_EFFECTS = ROOT / "data/derived/replogle_h1_transfer/effects.npz"


def cpm(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (1e6 / x.sum(axis=-1, keepdims=True))


def load_spec_source(spec, targets):
    a = ad.read_h5ad(spec.target_path, backed="r")
    try:
        labels = a.obs.gene_target.astype(str).to_numpy()
        rows = np.flatnonzero(np.isin(labels, targets) | (labels == spec.control))
        matrix, genes = collapse_columns(a.X[rows], a.var.gene_name.astype(str).tolist())
    finally:
        a.file.close()
    labels = labels[rows]
    is_control = labels == spec.control
    assert is_control.sum() == 1 and len(set(labels[~is_control])) == (~is_control).sum()
    dense = matrix.toarray().astype(np.float64)
    return list(labels[~is_control]), dense[~is_control], dense[is_control].ravel(), genes


def load_k562(targets):
    data = ad.read_h5ad(REPLOGLE, backed="r")
    try:
        row_targets = np.array([target_from_population(v) for v in data.obs_names.astype(str)])
        n_cells = data.obs["num_cells_filtered"].to_numpy(dtype=float)
        finite = np.isfinite(n_cells)
        core = data.obs["core_control"].to_numpy(dtype=bool)
        control_rows = np.flatnonzero((row_targets == CONTROL) & core & finite)
        matched = [t for t in targets if t in set(row_targets[finite])]
        guide_rows = np.flatnonzero(np.isin(row_targets, matched) & finite)
        retained = np.concatenate([control_rows, guide_rows])
        counts, audit = reconstruct_replogle_rows(data, np.flatnonzero(finite), retained)
        symbols = data.var["gene_name"].astype(str).tolist()
    finally:
        data.file.close()
    # author means are float32: mean * n_cells lands within ~0.02 of an integer
    assert audit["max_distance_from_integer"] < 0.05, audit["max_distance_from_integer"]
    collapsed, genes = collapse_duplicate_symbols(counts, symbols)
    control = collapsed[: len(control_rows)].sum(axis=0).astype(np.float64)
    guides = collapsed[len(control_rows) :]
    guide_targets = row_targets[guide_rows]
    per_target = np.vstack([guides[guide_targets == t].sum(axis=0) for t in matched])
    return matched, per_target.astype(np.float64), control, list(genes)


def average_change(targets, genes):
    """Mean phi_20 change over the donors measuring each (target, gene)."""
    ti = {t: i for i, t in enumerate(targets)}
    gi = {g: i for i, g in enumerate(genes)}
    total = np.zeros((len(targets), len(genes)))
    count = np.zeros((len(targets), len(genes)), dtype=np.int16)
    coverage = {}
    donors = {"K562_GWPS": load_k562(targets)}
    for name in ("HCT116", "HEK293T", "CD4T", "KOLF2.1J"):
        donors[name] = load_spec_source(SOURCE_SPECS[name], targets)
    for name, (d_targets, perturbed, control, d_genes) in donors.items():
        cols = [j for j, g in enumerate(d_genes) if g in gi]
        axis = np.array([gi[d_genes[j]] for j in cols])
        yp, y0 = cpm(perturbed[:, cols]), cpm(control[cols])
        change = np.log((yp + C) / (y0 + C))
        rows = np.array([ti[t] for t in d_targets])
        total[np.ix_(rows, axis)] += change
        count[np.ix_(rows, axis)] += 1
        coverage[name] = {"targets": len(d_targets), "genes_on_axis": len(cols)}
        print(f"{name}: {len(d_targets)} targets, {len(cols)} genes on the 2026 axis", flush=True)
    dbar = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    per_target = pd.DataFrame(
        {"target_gene": targets, "donors_max": count.max(axis=1), "genes_covered": (count > 0).sum(axis=1)}
    )
    return dbar, coverage, per_target


def global_change(genes):
    """H1's global response (mean perturbed minus mean control CPM over all 2025 H1 cells, the
    Exploring the Data II axis) as a phi_20 change on H1's pooled control, mapped to the 2026
    axis; genes not on the H1 axis get no change."""
    with np.load(GLOBAL_DELTA) as a:
        delta = a["delta"]
    with np.load(H1_PHASE1) as a:
        h1_genes = a["output_gene"].astype(str)
    with np.load(H1_EFFECTS) as a:
        h0 = cpm(a["h1_control_count_sum"])
    d = np.log((np.maximum(h0 + delta, 0) + C) / (h0 + C))
    lookup = dict(zip(h1_genes, d, strict=True))
    out = np.array([lookup.get(g, 0.0) for g in genes])
    print(f"global change: {np.count_nonzero(out)} genes, norm {np.linalg.norm(out):.3f}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, required=True, help="a in x0 exp(a * dbar)")
    parser.add_argument("--global-shift", action="store_true",
                        help="add H1's global perturbation response as a phi_20 change")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(CONTROLS / "gene_names.csv").iloc[:, 0].astype(str).tolist()
    targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    dbar, coverage, per_target = average_change(targets, genes)
    label = f"transfer_average_c20_a{args.scale:g}"
    change = args.scale * dbar
    if args.global_shift:
        change = change + global_change(genes)[None, :]
        label += "_plus_global"

    chunks, total_nnz, cosines, max_total = [], 0, [], 0
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        temporary = Path(temporary)
        for context_index, path in enumerate(sorted(CONTROLS.glob("context_*.h5ad"))):
            data = ad.read_h5ad(path)
            context = str(data.obs["context"].iloc[0])
            assert data.var_names.tolist() == genes
            assert data.obs["target_gene"].eq("non-targeting").all()
            pooled = cpm(np.asarray(data.X.sum(axis=0)).ravel())
            raw_pred = (pooled + C) * np.exp(change) - C
            pred = cpm(np.maximum(raw_pred, 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(pooled > 0, np.log2(pred / pooled), 0.0)
            factor = np.where(np.isfinite(factor), factor, -60.0)
            for start in range(0, len(targets), TARGETS_PER_CHUNK):
                blocks, labels = [], []
                for target_index in range(start, min(start + TARGETS_PER_CHUNK, len(targets))):
                    target = targets[target_index]
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, context_index, target_index]))
                    raw = data.X[sample_rows(data.obs, rng)].tocsr()
                    sample = cpm(np.asarray(raw.sum(axis=0)).ravel())
                    with np.errstate(divide="ignore", invalid="ignore"):
                        shift = np.where(sample > 0, np.log2(pooled / sample), 0.0)
                    lfc = factor[target_index] + np.where(np.isfinite(shift), shift, 0.0)
                    round_rng = np.random.default_rng(zlib.crc32(f"{label}:{context}:{target}".encode()))
                    block = decode_lfc_genewise(raw, lfc, round_rng)
                    base = harness_log(np.asarray(raw.sum(axis=0)).ravel())
                    ideal = harness_log(expected_counts(raw, lfc).sum(axis=0))
                    got = harness_log(np.asarray(block.sum(axis=0)).ravel())
                    want, real = ideal - base, got - base
                    cosines.append(float(want @ real / (np.linalg.norm(want) * np.linalg.norm(real))))
                    max_total = max(max_total, int(block.sum(axis=1).max()))
                    blocks.append(block)
                    labels.extend([target] * CELLS_PER_TARGET)
                counts = sparse.vstack(blocks, format="csr")
                counts.eliminate_zeros()
                total_nnz += counts.nnz
                obs = pd.DataFrame({"target_gene": labels, "context": context})
                obs.index = [f"{context}_{start + i // 400:03d}_{i % 400:03d}" for i in range(len(obs))]
                chunk = temporary / f"{context}_{start:03d}.h5ad"
                ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(chunk, compression="gzip")
                chunks.append(chunk)
            print(f"context {context}: done; median fidelity cosine so far {np.median(cosines):.4f}", flush=True)
            del data
        assert np.median(cosines) >= MIN_COSINE, np.median(cosines)
        assert total_nnz <= MAX_ENTRIES and max_total <= 1_000_000, (total_nnz, max_total)
        ad.experimental.concat_on_disk(chunks, args.output, max_loaded_elems=25_000_000, merge="same")

    digest = hashlib.sha256()
    with args.output.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "method": "five-screen average phi_20 change, transferred onto each context's pooled controls",
        "pseudocount_cpm": C,
        "scale": args.scale,
        "global_shift": bool(args.global_shift),
        "control_cell_seed": SEED,
        "cells": 3 * len(targets) * CELLS_PER_TARGET,
        "genes": len(genes),
        "stored_entries": total_nnz,
        "max_cell_total": max_total,
        "fidelity_median_cosine": float(np.median(cosines)),
        "fidelity_min_cosine": float(np.min(cosines)),
        "donor_coverage": coverage,
        "targets_by_donor_count": per_target.donors_max.value_counts().sort_index().to_dict(),
        "prediction_sha256": digest.hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1, default=int))
    per_target.to_csv(args.output.with_suffix(".coverage.csv"), index=False)
    print(json.dumps(manifest, indent=1, default=int), flush=True)


if __name__ == "__main__":
    main()
