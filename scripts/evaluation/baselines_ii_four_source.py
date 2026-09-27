"""Four-source sensitivity panel for Baselines II: do low-count genes move the best pseudocount?

CD4T, HCT116, HEK293T and KOLF2.1J on their common H1 targets, built from the frozen per-source
`available_*` profiles (no new data). Three views of the same sources and targets:

* k562_genes:  the five-source common gene set (genes expressed in K562 GWPS), CPM on that set;
* all_genes:   all genes the four sources share (~17.6k), CPM on that set;
* all_genes_scored_on_k562_genes: transferred on all genes, scored only on the k562_genes set.

The last view separates transferring/normalizing on more genes from being scored on them.
Run: pixi run python -m scripts.evaluation.baselines_ii_four_source
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.evaluation.h1_transfer_regression as reg
from scripts.evaluation.baselines_ii_noise import run
from scripts.evaluation.kolf_h1_transfer_rules import cpm

SOURCES = ("CD4T", "HCT116", "HEK293T", "KOLF2.1J")
SWEEP = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, np.inf)
H1 = reg.ROOT / "data/derived/replogle_h1_transfer/effects.npz"
PHASE1 = reg.ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"


def build(gene_set, targets, raw, depth, h1, strong, name):
    genes = sorted(gene_set)
    sources, libs = {}, {}
    for s in SOURCES:
        z = raw[s]
        r = [z["ti"][t] for t in targets]
        g = [z["gi"][x] for x in genes]
        sp = z["source_perturbed"][np.ix_(r, g)]
        libs[s] = depth[s].reindex(targets).to_numpy() * sp.sum(axis=1) / 1e6  # counts on the set
        sources[s] = {"source_perturbed": cpm(sp), "source_control": cpm(z["source_control"][g])}
    hg = [h1["gi"][x] for x in genes]
    ht = [h1["ti"][t] for t in targets]
    common = {
        "targets": np.array(targets),
        "genes": np.array(genes),
        "h1_control": cpm(h1["control"][hg]),
        "h1_truth": cpm(h1["perturbed"][np.ix_(ht, hg)]),
    }
    return reg.Panel(name, sources, common, strong), libs


def load_inputs(strong):
    raw, depth = {}, {}
    for s in SOURCES:
        with np.load(reg.SOURCE / f"available_{s}/{s}.npz") as z:
            t, g = z["targets"].astype(str), z["genes"].astype(str)
            raw[s] = {
                "source_perturbed": z["source_perturbed"],
                "source_control": z["source_control"],
                "ti": {x: i for i, x in enumerate(t)},
                "gi": {x: i for i, x in enumerate(g)},
            }
        d = pd.read_csv(reg.ROOT / f"reports/all-source-h1-transfer/available_{s}/depth.csv")
        depth[s] = d.set_index("target_gene").shared_UMIs
    with np.load(PHASE1) as a:
        h1_genes = a["output_gene"].astype(str)
    with np.load(H1) as a:
        h1_targets = a["target_gene"].astype(str)
        h1 = {
            "control": a["h1_control_count_sum"].astype(float),
            "perturbed": a["h1_count_sum"].astype(float),
            "gi": {x: i for i, x in enumerate(h1_genes)},
            "ti": {x: i for i, x in enumerate(h1_targets)},
        }
    targets = sorted(set(h1_targets).intersection(*(raw[s]["ti"] for s in SOURCES)))
    all_genes = set(h1_genes).intersection(*(raw[s]["gi"] for s in SOURCES))
    with np.load(reg.SOURCE / "five_anchors/KOLF2.1J.npz") as z:
        k562_genes = set(z["genes"].astype(str))
    return raw, depth, h1, targets, all_genes, k562_genes


def main():
    strong = pd.read_csv(reg.STRONG).query("stable_strong")
    raw, depth, h1, targets, all_genes, k562_genes = load_inputs(strong)
    print(f"{len(targets)} targets; all genes {len(all_genes)}; K562-expressed genes {len(k562_genes)}", flush=True)

    tables = []
    for name, genes in (("k562_genes", k562_genes), ("all_genes", all_genes)):
        panel, libs = build(genes, targets, raw, depth, h1, strong, name)
        tables.append(run(panel, libs, f"four_source_{name}", sweep=SWEEP).assign(panel=name))
    # Transfer and normalize on all genes, score only on the K562-expressed genes.
    panel, libs = build(all_genes, targets, raw, depth, h1, strong, "all_genes_scored_on_k562_genes")
    panel.keep &= np.isin(panel.genes, sorted(k562_genes))
    panel.y = panel.effect(panel.hp)
    panel.energy = np.square(panel.y).sum()
    tables.append(
        run(panel, libs, "four_source_all_genes_scored_on_k562_genes", sweep=SWEEP).assign(
            panel="all_genes_scored_on_k562_genes"
        )
    )
    out = pd.concat(tables)
    out.to_csv(reg.ROOT / "reports/baselines-ii/four_source_summary.csv", index=False)
    print(out[["panel", "model", "retrieval", "cosine", "profile_mse_ratio", "noise_fraction"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
