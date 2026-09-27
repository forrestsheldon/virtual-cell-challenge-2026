"""Fixed transfer rules from every locally available source screen into H1.

No fitting or downloads. Matched panels share both targets and normalization genes.
Run: pixi run python -m scripts.evaluation.all_source_h1_transfer
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from scripts.evaluation.context_transfer_blog import CONTEXTS, collapse_columns, sha256
from scripts.evaluation.context_transfer_h1 import (
    H1_AGGREGATION,
    H1_CONTROLS,
    H1_PHASE1,
    REPLOGLE_RAW,
    SOURCE_SPECS,
)
from scripts.evaluation.kolf_h1_transfer_rules import (
    METRICS,
    RULES,
    close_prediction,
    cpm,
    evaluate,
    log_profile,
    stochastic_pseudobulk,
    transfer,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/all-source-h1-transfer"
DERIVED = ROOT / "data/derived/all_source_h1_transfer"
SEED = 20260925
ANCHORS = ["K562_GWPS", "CD4T", "HCT116", "HEK293T", "KOLF2.1J"]
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"


@dataclass
class Counts:
    name: str
    genes: list[str]
    targets: list[str]
    counts: np.ndarray
    control: np.ndarray
    cells: np.ndarray
    provenance: dict


def load_counts():
    """Load existing count artifacts; restrict compact rows before densification."""
    inputs = [
        H1_AGGREGATION,
        H1_PHASE1,
        REPLOGLE_RAW,
        ROOT / "reports/replogle-h1-transfer/checkpoint1_manifest.json",
        ROOT / "reports/replogle-h1-transfer/target_overlap_and_knockdown.csv",
    ]
    original = json.loads(inputs[3].read_text())
    assert (
        sha256(H1_AGGREGATION)
        == original["outputs"][str(H1_AGGREGATION.relative_to(ROOT))]
    )
    with np.load(H1_PHASE1) as a:
        hgenes = a["output_gene"].astype(str).tolist()
    with np.load(H1_AGGREGATION) as a:
        h = Counts(
            "H1",
            hgenes,
            a["target_gene"].astype(str).tolist(),
            a["h1_count_sum"],
            a["h1_control_count_sum"],
            np.full(126, 400),
            original,
        )
        raw = ad.read_h5ad(REPLOGLE_RAW, backed="r")
        _, kgenes = pd.factorize(raw.var.gene_name.astype(str), sort=False)
        raw.file.close()
        targets = a["matched_target_gene"].astype(str).tolist()
        table = pd.read_csv(inputs[4]).set_index("target_gene")
        sources = {
            "K562_GWPS": Counts(
                "K562_GWPS",
                kgenes.tolist(),
                targets,
                a["replogle_count_sum"],
                a["replogle_control_count_sum"],
                table.loc[targets, "n_cells"].to_numpy(),
                original,
            )
        }
    for name, spec in {**CONTEXTS, **SOURCE_SPECS}.items():
        manifest = json.loads(spec.manifest_path.read_text())
        assert (
            sha256(spec.target_path)
            == manifest["outputs"][spec.target_path.name]["sha256"]
        )
        inputs.extend([spec.target_path, spec.manifest_path, spec.audit_path])
        a = ad.read_h5ad(spec.target_path, backed="r")
        labels = a.obs.gene_target.astype(str).to_numpy()
        rows = np.flatnonzero(np.isin(labels, h.targets) | (labels == spec.control))
        matrix, genes = collapse_columns(
            a.X[rows], a.var.gene_name.astype(str).tolist()
        )
        selected = labels[rows] != spec.control
        assert (~selected).sum() == 1
        targets = labels[rows][selected].tolist()
        counts = matrix[selected].toarray().astype(np.int64)
        control = matrix[~selected].toarray().ravel().astype(np.int64)
        cells = a.obs.n_cells.to_numpy()[rows][selected]
        a.file.close()
        assert (counts >= 0).all() and (control >= 0).all()
        sources[name] = Counts(name, genes, targets, counts, control, cells, manifest)
    return sources, h, inputs


def axes(sources, h):
    targets = sorted(set(h.targets).intersection(*(set(s.targets) for s in sources)))
    genes = sorted(set(h.genes).intersection(*(set(s.genes) for s in sources)))
    return targets, genes


def extract(c, targets, genes):
    rows = [c.targets.index(t) for t in targets]
    cols = [c.genes.index(g) for g in genes]
    return (
        c.counts[np.ix_(rows, cols)].astype(float),
        c.control[cols].astype(float),
        c.cells[rows],
    )


def mean_metrics(frame):
    row = frame[list(METRICS)].mean().to_dict()
    row["profile_mse_ratio"] = frame.squared_error.sum() / frame.truth_energy.sum()
    return row


def run_panel(panel, sources, h, targets, genes, n_controls, strong_table):
    out = REPORT / panel
    derived = DERIVED / panel
    out.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    hc, hc0, _ = extract(h, targets, genes)
    hp, h0 = cpm(hc), cpm(hc0)
    keep = ~np.isin(genes, targets)
    strong = np.zeros_like(hp, dtype=bool)
    ti = {t: i for i, t in enumerate(targets)}
    gi = {g: i for i, g in enumerate(genes)}
    for r in strong_table.itertuples():
        if r.target_gene in ti and r.feature in gi:
            strong[ti[r.target_gene], gi[r.feature]] = True
    y = (log_profile(hp) - log_profile(h0))[:, keep]
    energy = np.square(y).sum()
    depth = hc0.sum() / n_controls
    pd.DataFrame({"target_gene": targets}).to_csv(out / "targets.csv", index=False)
    pd.DataFrame({"gene": genes, "metric_included": keep}).to_csv(
        out / "genes.csv", index=False
    )
    frames = []
    summaries = []
    boundaries = []
    nulls = []
    roundings = []
    depths = []
    decompositions = []
    common_rows = []
    wrong_rows = []
    unchanged = evaluate(np.broadcast_to(h0, hp.shape), hp, h0, keep, strong)
    assert np.allclose(unchanged.retrieval, 0.5)
    assert np.allclose(unchanged.squared_error, unchanged.truth_energy)
    oracle = evaluate(hp, hp, h0, keep, strong)
    assert np.allclose(oracle.retrieval, 1) and np.allclose(oracle.squared_error, 0)
    summaries.append(
        {"source": "unchanged", "rule": "unchanged", **mean_metrics(unchanged)}
    )
    for name, s in sources.items():
        print(
            f"{panel}: {name} ({len(targets)} targets, {len(genes)} genes)", flush=True
        )
        counts, control, cells = extract(s, targets, genes)
        sp, s0 = cpm(counts), cpm(control)
        libs = counts.sum(axis=1).astype(np.int64)
        depths.extend(
            {
                "source": name,
                "target_gene": t,
                "cells": int(cells[i]),
                "shared_UMIs": int(libs[i]),
                "UMIs_per_cell": libs[i] / cells[i],
                "h1_shared_UMIs": int(hc[i].sum()),
                "zero_control_genes": int((s0 == 0).sum()),
            }
            for i, t in enumerate(targets)
        )
        predictions = {}
        for rule in RULES:
            raw = transfer(sp, s0, h0, rule)
            pred, audit = close_prediction(raw)
            predictions[rule] = pred
            f = evaluate(pred, hp, h0, keep, strong)
            frames.append(f.assign(source=name, rule=rule, target_gene=targets))
            summaries.append({"source": name, "rule": rule, **mean_metrics(f)})
            boundaries.append(audit.assign(source=name, rule=rule, target_gene=targets))
            p = (log_profile(pred) - log_profile(h0))[:, keep]
            pre = (log_profile(np.maximum(raw, 0)) - log_profile(h0))[:, keep]
            pe = np.square(p).sum() / energy
            cross = 2 * np.sum(p * y) / energy
            np.testing.assert_allclose(f.squared_error.sum() / energy, 1 + pe - cross)
            decompositions.append(
                {
                    "source": name,
                    "rule": rule,
                    "prediction_energy_over_truth": pe,
                    "twice_dot_over_truth": cross,
                    "before_closure_error_ratio": np.square(pre - y).sum() / energy,
                    "after_closure_error_ratio": f.squared_error.sum() / energy,
                }
            )
            # Fixed common-response control, independent of H1 perturbation truth.
            cf = evaluate(
                np.broadcast_to(pred.mean(axis=0), hp.shape), hp, h0, keep, strong
            )
            assert np.isclose(cf.retrieval.mean(), 0.5)
            common_rows.append({"source": name, "rule": rule, **mean_metrics(cf)})
            # Twenty wrong-target permutations: retrieval matrix rows can be reused.
            from scipy.stats import rankdata

            from scripts.evaluation.kolf_h1_transfer_rules import unit_rows

            similarity = unit_rows(p) @ unit_rows(y).T
            ranks = 1 - (rankdata(-similarity, axis=1, method="average") - 1) / (
                len(targets) - 1
            )
            rng = np.random.default_rng(SEED)
            for repeat in range(20):
                perm = rng.permutation(len(targets))
                while np.any(perm == np.arange(len(targets))):
                    perm = rng.permutation(len(targets))
                wrong_rows.append(
                    {
                        "source": name,
                        "rule": rule,
                        "repeat": repeat,
                        "retrieval": ranks[perm, np.arange(len(targets))].mean(),
                        "cosine": similarity[perm, np.arange(len(targets))].mean(),
                    }
                )
            for seed in range(SEED, SEED + 5):
                rates = pred * depth / 1e6
                integer = stochastic_pseudobulk(rates, np.random.default_rng(seed))
                error = integer / 400 - rates
                frac = rates - np.floor(rates)
                se = np.sqrt(np.sum(frac * (1 - frac) / 400)) / error.size
                assert abs(error.mean()) < 7 * se
                rf = evaluate(cpm(integer), hp, h0, keep, strong)
                roundings.append(
                    {
                        "source": name,
                        "rule": rule,
                        "seed": seed,
                        "mean_signed_UMI_error": error.mean(),
                        "mean_absolute_UMI_error": np.abs(error).mean(),
                        **mean_metrics(rf),
                    }
                )
        np.savez_compressed(
            derived / f"{name}.npz",
            targets=targets,
            genes=genes,
            source_control=s0,
            source_perturbed=sp,
            h1_control=h0,
            h1_truth=hp,
            **predictions,
        )
        # The same technical-count null as KOLF: no true donor perturbation effect.
        rng = np.random.default_rng(SEED)
        for repeat in range(10):
            sampled = np.vstack([rng.multinomial(int(n), s0 / s0.sum()) for n in libs])
            np.testing.assert_array_equal(sampled.sum(axis=1), libs)
            for rule in RULES:
                pred, _ = close_prediction(transfer(cpm(sampled), s0, h0, rule))
                p = (log_profile(pred) - log_profile(h0))[:, keep]
                nulls.append(
                    {
                        "source": name,
                        "rule": rule,
                        "repeat": repeat,
                        "null_prediction_energy_over_truth": np.square(p).sum()
                        / energy,
                        "null_error_ratio": np.square(p - y).sum() / energy,
                    }
                )
    tables = {
        "per_target": pd.concat(frames, ignore_index=True),
        "summary": pd.DataFrame(summaries),
        "boundary_audit": pd.concat(boundaries),
        "count_sampling_null": pd.DataFrame(nulls),
        "rounding": pd.DataFrame(roundings),
        "depth": pd.DataFrame(depths),
        "decomposition": pd.DataFrame(decompositions),
        "common_response": pd.DataFrame(common_rows),
        "wrong_target": pd.DataFrame(wrong_rows),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    paired_comparisons(tables["per_target"], out)
    return tables["summary"].assign(panel=panel, targets=len(targets), genes=len(genes))


def paired_comparisons(frame, out):
    targets = sorted(frame.target_gene.unique())
    boot = np.random.default_rng(SEED).integers(len(targets), size=(2000, len(targets)))
    rows = []
    # Compare sources within each fixed rule; do not select a different best rule per source.
    for rule in RULES:
        sub = frame.query("rule == @rule")
        for left, right in combinations(sorted(sub.source.unique()), 2):
            a = sub[sub.source.eq(left)].set_index("target_gene").loc[targets]
            b = sub[sub.source.eq(right)].set_index("target_gene").loc[targets]
            for metric in ["retrieval", "cosine", "signed_top100", "profile_mse_ratio"]:
                if metric == "profile_mse_ratio":
                    d = (a.squared_error - b.squared_error).to_numpy()
                    denom = a.truth_energy.to_numpy()
                    values = d[boot].sum(axis=1) / denom[boot].sum(axis=1)
                    estimate = d.sum() / denom.sum()
                else:
                    d = (a[metric] - b[metric]).to_numpy()
                    values = d[boot].mean(axis=1)
                    estimate = d.mean()
                lo, hi = np.quantile(values, [0.025, 0.975])
                rows.append(
                    {
                        "rule": rule,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        "difference": estimate,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    pd.DataFrame(rows).to_csv(out / "paired_source_differences.csv", index=False)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    sources, h, inputs = load_counts()
    controls = ad.read_h5ad(H1_CONTROLS, backed="r")
    n_controls = controls.n_obs
    controls.file.close()
    inputs.extend([H1_CONTROLS, STRONG])
    strong = pd.read_csv(STRONG).query("stable_strong")
    inventory = []
    for name, s in sources.items():
        t, g = axes([s], h)
        inventory.append(
            {
                "source": name,
                "available_H1_targets": len(t),
                "available_H1_genes": len(g),
                "study_description": s.provenance.get(
                    "dataset", "Replogle 2022 GWPS author pseudobulk"
                ),
                "representation": s.provenance.get(
                    "output_representation",
                    "raw UMI sums reconstructed from author means",
                ),
            }
        )
    pd.DataFrame(inventory).to_csv(REPORT / "inventory.csv", index=False)
    results = []
    for panel, subset in [
        ("all_nine", sources),
        ("five_anchors", {n: sources[n] for n in ANCHORS}),
    ]:
        t, g = axes(list(subset.values()), h)
        results.append(run_panel(panel, subset, h, t, g, n_controls, strong))
    # Supplementary full pairwise overlap: never a cross-source ranking.
    for name, s in sources.items():
        t, g = axes([s], h)
        results.append(
            run_panel("available_" + name, {name: s}, h, t, g, n_controls, strong)
        )
    summary = pd.concat(results, ignore_index=True)
    summary.to_csv(REPORT / "summary.csv", index=False)
    for panel in ["all_nine", "five_anchors"]:
        f = summary.query('panel == @panel and rule != "unchanged"')
        names = (
            f[f.rule.eq("log1p")]
            .sort_values("retrieval", ascending=False)
            .source.tolist()
        )
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, metric in zip(
            axs, ["retrieval", "cosine", "profile_mse_ratio"], strict=True
        ):
            for j, rule in enumerate(RULES):
                v = f[f.rule.eq(rule)].set_index("source").loc[names, metric]
                ax.bar(np.arange(len(names)) + (j - 1) * 0.25, v, 0.25, label=rule)
            ax.set_xticks(
                np.arange(len(names)), names, rotation=55, ha="right", fontsize=8
            )
            ax.set_title(metric)
            if metric == "retrieval":
                ax.axhline(0.5, color="black", lw=0.6)
            if metric == "profile_mse_ratio":
                ax.axhline(1, color="black", lw=0.6)
        axs[0].legend(fontsize=8)
        fig.suptitle(f"{panel}: fixed transfer to H1, matched targets and genes")
        fig.savefig(REPORT / f"{panel}.png", dpi=160)
        plt.close(fig)
    code = [
        Path(__file__),
        ROOT / "scripts/evaluation/kolf_h1_transfer_rules.py",
        ROOT / "scripts/evaluation/context_transfer_blog.py",
    ]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "lineage": {n: s.provenance for n, s in sources.items()},
        "design": "Same fixed additive, ratio (undefined=1), and log1p CPM transfer; floor rates then close to CPM. Shared genes define denominators within each panel. No H1 fitting. All panel-target genes excluded from every metric.",
        "panels": "Nine local source screens on all-common targets/genes; five genome-scale anchors on all-common targets/genes; full per-source overlaps supplementary, not ranked against one another.",
        "limitations": "K562 GWPS and essential are different screens of one study/context; CD4T pools donors and conditions. Jiang and McFaline have metadata but no local expression artifacts; not downloaded. H1 is destination only. Archives, views and split copies are not extra datasets.",
        "uncertainty": "2000 paired target bootstraps, unadjusted pairwise intervals conditional on observed datasets. 20 wrong-target derangements. 10 multinomial count-null replicates, controls fixed; not a biological reliability correction. Five stochastic rounding seeds using exact 400-cell pooled distribution.",
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "code": {str(p.relative_to(ROOT)): sha256(p) for p in code},
        "outputs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                *REPORT.rglob("*.csv"),
                *REPORT.glob("*.png"),
                *DERIVED.rglob("*.npz"),
            ]
        },
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        summary.query(
            'panel in ["all_nine","five_anchors"] and rule == "log1p"'
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
