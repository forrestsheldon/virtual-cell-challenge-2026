"""GO GSEA on the shared axis, and on each perturbation's residual after removing it.

The trap in per-perturbation GSEA is that the shared pluripotency-exit axis is
engaged in proportion to response size, so ranking each perturbation's raw shift
returns the SAME differentiation / growth terms for every strong perturbation --
the shared program rediscovered 150 times, not each perturbation's own biology.

So two passes:
  1. GSEA on the shared axis a itself -- names the shared program formally
     (validates "pluripotency exit + growth-down" beyond the by-eye gene list).
  2. GSEA on the residual shift  resid = shift - (shift . a_hat) a_hat  for every
     perturbation -- the shared component projected out, leaving the specific part.
Then the per-perturbation residual terms are aggregated by data-driven family to
ask whether family-mates share the SAME specific program.

Ranking metric is the shift value per gene (Systema centroid difference), the same
object the families and co-complex tests use. GO BP is the primary library (the
per-perturbation sweep); Reactome is added for the shared-axis headline only.
"""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

ANN = Path("data/external/annotations")
GO, REACTOME = ANN / "GO_Biological_Process_2023.gmt", ANN / "Reactome_Pathways_2024.gmt"


def load_gmt(path):
    sets = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        genes = [g for g in parts[2:] if g]
        if genes:
            sets[parts[0]] = genes
    return sets


def prerank(ranking, gene_sets, genes, perms, seed=0):
    rnk = pd.DataFrame({"gene": genes, "score": ranking}).sort_values("score", ascending=False)
    res = gp.prerank(rnk=rnk, gene_sets=gene_sets, min_size=10, max_size=500,
                     permutation_num=perms, threads=4, seed=seed, no_plot=True, outdir=None).res2d
    res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
    res["FDR q-val"] = pd.to_numeric(res["FDR q-val"], errors="coerce")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
    parser.add_argument("--fdr", type=float, default=0.10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    shifts = np.load(args.output / "celleval2_shifts.npy")
    a = np.load(args.output / "celleval2_shared_axis.npy")
    genes = pd.read_csv(args.output / "celleval2_shift_genes.csv").gene.to_list()
    targets = pd.read_csv(args.output / "celleval2_systema_summary.csv").target.to_numpy()
    family = pd.read_csv(args.output / "celleval2_families.csv").set_index("target")["family"]

    go = load_gmt(GO)
    a_hat = a / np.linalg.norm(a)

    # 1. shared axis -----------------------------------------------------------
    print("=== shared axis (GO BP + Reactome) ===", flush=True)
    shared = pd.concat([prerank(a, lib, genes, 1000).assign(library=name)
                        for name, lib in [("GO", go), ("Reactome", load_gmt(REACTOME))]])
    shared.to_csv(args.output / "celleval2_gsea_shared_axis.csv", index=False)
    for tail, label in [(False, "positive NES  (up in the average perturbation)"),
                        (True, "negative NES  (down in the average perturbation)")]:
        top = shared[shared["FDR q-val"] < 0.05].sort_values("NES", ascending=tail).head(10)
        print(f"\n{label}:")
        for _, r in top.iterrows():
            print(f"  NES {r.NES:+.2f}  FDR {r['FDR q-val']:.1e}  {r.Term[:70]}  [{r.library}]")

    # 2. per-perturbation residual --------------------------------------------
    order = targets if not args.limit else targets[: args.limit]
    print(f"\n=== residual GSEA (GO BP) over {len(order)} perturbations ===", flush=True)
    rows = []
    for i, t in enumerate(order):
        idx = list(targets).index(t)
        resid = shifts[idx] - (shifts[idx] @ a_hat) * a_hat
        res = prerank(resid, go, genes, 200, seed=0)
        hits = res[res["FDR q-val"] < args.fdr]
        for _, r in hits.iterrows():
            rows.append({"target": t, "family": int(family[t]), "term": r.Term,
                         "NES": float(r.NES), "fdr": float(r["FDR q-val"])})
        print(f"  {i+1:3}/{len(order)} {t:10} {len(hits):3} terms at FDR<{args.fdr}", flush=True)

    specific = pd.DataFrame(rows)
    specific.to_csv(args.output / "celleval2_gsea_residual.csv", index=False)

    # 3. family agreement ------------------------------------------------------
    print("\n=== terms recurring across family-mates (specific programs) ===")
    fam_sizes = family.value_counts()
    for f in fam_sizes[fam_sizes >= 2].index:
        members = family[family == f].index.tolist()
        sub = specific[specific.target.isin(members)]
        if sub.empty:
            continue
        shared_terms = Counter(sub.term)
        recur = [(term, n) for term, n in shared_terms.items() if n >= 2]
        if recur:
            recur.sort(key=lambda x: -x[1])
            names = ", ".join(f"{term[:45]} (x{n})" for term, n in recur[:4])
            print(f"  family {f:3} {members}: {names}")

    (args.output / "celleval2_gsea_residual_run.json").write_text(json.dumps({
        "created_utc": datetime.now(UTC).isoformat(),
        "ranking": "Systema shift; residual = shift - (shift.a_hat) a_hat (shared axis removed)",
        "libraries": {"per_perturbation": "GO_Biological_Process_2023",
                      "shared_axis": ["GO_Biological_Process_2023", "Reactome_Pathways_2024"]},
        "fdr_threshold": args.fdr, "n_perturbations": len(order),
        "n_specific_hits": len(specific),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
