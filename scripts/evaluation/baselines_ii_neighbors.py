"""Baselines II: how much of the transferred signal is CRISPRi neighbor knockdown?

dCas9-KRAB silencing can spread to genes near the target. The scorer excludes only the target
gene, so a knocked-down neighbor counts as a response, and a CRISPRi donor transfers it for free.
Five-source panel (110 H1 targets, 7,446 genes), direct transfer at c = 20.

1. Knockdown by distance: mean change of genes at each genomic distance from the target, in the
   H1 truth, each donor, and the equal-screen average (evaluation space ln(1 + CPM/20)).
2. Share of alignment from neighbors: <P, Y> restricted to neighbors within d, over the total.
3. Re-scoring with the union of all targets' neighbors within d excluded (panel-style, so every
   profile keeps one gene axis), against the same number of random genes excluded, and against
   scoring on the neighbors only.
Gene coordinates: the local K562 GWPS var table (chr, start, end), which covers the panel.
Run: pixi run python -m scripts.evaluation.baselines_ii_neighbors
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

import scripts.evaluation.h1_transfer_regression as reg

C = 20.0
DISTANCES = (10_000, 50_000, 100_000, 250_000, 1_000_000)
BINS = [(-1, 0), (0, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, 250_000),
        (250_000, 1_000_000), (1_000_000, np.inf)]
K562 = reg.ROOT / "data/external/replogle2022/ReplogleWeissman2022_K562_gwps.h5ad"
REPORT = reg.ROOT / "reports/baselines-ii"


def coordinates(genes):
    a = ad.read_h5ad(K562, backed="r")
    var = a.var[["chr", "start", "end"]].copy()
    a.file.close()
    var = var[~var.index.duplicated()]
    return var.reindex(genes)


def distance_matrix(targets, genes, coord):
    """Interval distance from each target to each gene; inf across chromosomes or if unknown."""
    t = coord.reindex(targets)
    out = np.full((len(targets), len(genes)), np.inf)
    gchr, gs, ge = coord.chr.to_numpy(), coord.start.to_numpy(float), coord.end.to_numpy(float)
    for i, (c, s, e) in enumerate(t[["chr", "start", "end"]].itertuples(index=False)):
        if pd.isna(c):
            continue
        same = gchr == c
        out[i, same] = np.maximum(0, np.maximum(gs[same] - e, s - ge[same]))
    return out


def rescore(panel, keep):
    panel.keep = keep
    panel.y = panel.effect(panel.hp)
    panel.energy = np.square(panel.y).sum()


def main():
    strong = pd.read_csv(reg.STRONG).query("stable_strong")
    panel = reg.Panel("five_anchors", *reg.load_panel("five_anchors"), strong)
    targets, genes = panel.targets, panel.genes
    coord = coordinates(genes)
    dist = distance_matrix(targets, genes, coord)
    own = genes[None, :] == targets[:, None]
    print(f"coordinates for {coord.chr.notna().sum()} of {len(genes)} genes; "
          f"{np.isfinite(dist).any(axis=1).sum()} of {len(targets)} targets placed", flush=True)
    names = list(panel.sources)
    d = {s: reg.direction(a["source_perturbed"], a["source_control"], C) for s, a in panel.sources.items()}
    d["avg"] = sum(d[s] for s in names) / len(names)
    base = reg.log_profile(panel.h0)
    full = {k: reg.log_profile(reg.realize(v, panel.h0, C)) - base for k, v in d.items()}
    full["H1_truth"] = reg.log_profile(panel.hp) - base

    # 1. Knockdown by distance (own gene = distance -1 bin).
    rows = []
    dd = np.where(own, -0.5, dist)
    for lo, hi in BINS:
        mask = (dd > lo) & (dd <= hi) if lo >= 0 else own
        for k, eff in full.items():
            rows.append({"bin_low": lo, "bin_high": hi, "profile": k, "pairs": int(mask.sum()),
                         "mean_change": float(eff[mask].mean())})
    knock = pd.DataFrame(rows)
    knock.to_csv(REPORT / "neighbor_knockdown_c20.csv", index=False)
    print(knock.pivot_table(index=["bin_low", "bin_high"], columns="profile", values="mean_change").round(4).to_string(), flush=True)

    # 2 + 3. Alignment share and re-scoring.
    base_keep = ~np.isin(genes, targets)
    rng = np.random.default_rng(reg.SEED)
    out = []
    for dmax in (0, *DISTANCES):
        near = (dist <= dmax) & ~own & base_keep[None, :]
        union = near.any(axis=0)
        n_ex = int(union.sum())
        views = {"exclude_neighbors": base_keep & ~union}
        if dmax:
            random = np.zeros_like(base_keep)
            random[rng.choice(np.flatnonzero(base_keep & ~union), n_ex, replace=False)] = True
            views["exclude_random_same_n"] = base_keep & ~random
            views["neighbors_only"] = base_keep & union
        for view, keep in views.items():
            rescore(panel, keep)
            for k in [*names, "avg"]:
                q = full[k][:, keep]
                frame = panel.score(q)
                y = full["H1_truth"][:, keep]
                row = {"max_distance": dmax, "view": view, "excluded_genes": n_ex,
                       "scored_genes": int(keep.sum()), "model": k,
                       "retrieval": frame.retrieval.mean(), "cosine": frame.cosine.mean()}
                if view == "exclude_neighbors" and dmax == 0:
                    total = np.sum(full[k][:, base_keep] * full["H1_truth"][:, base_keep])
                    for dm in DISTANCES:
                        m = (dist <= dm) & ~own & base_keep[None, :]
                        row[f"alignment_share_{dm // 1000}kb"] = float(np.sum((full[k] * full["H1_truth"])[m]) / total)
                row["pooled_rho"] = float(np.sum(q * y) / np.sqrt(np.sum(q * q) * np.sum(y * y)))
                out.append(row)
    table = pd.DataFrame(out)
    table.to_csv(REPORT / "neighbor_check_c20.csv", index=False)
    share = table[(table.max_distance == 0)].set_index("model").filter(like="alignment_share")
    print(share.round(4).to_string(), flush=True)
    view = table[table.model == "avg"].pivot_table(index=["max_distance", "excluded_genes"], columns="view", values="retrieval")
    print("avg retrieval\n", view.round(4).to_string(), flush=True)
    view = table[table.model == "avg"].pivot_table(index=["max_distance", "excluded_genes"], columns="view", values="cosine")
    print("avg cosine\n", view.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
