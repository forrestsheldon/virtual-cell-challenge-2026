"""Do perturbations of the same protein complex produce the same transcriptional response?

Systema's systematic-variation statistic says how much of the response is shared.
It says nothing about whether what remains is real biology. This is the positive
test: if the residual carried no target-specific information, perturbations of
co-complex members would be no more similar to each other than random pairs.

The shifts are Systema's own s_X = O(X) - O_control from celleval2_h1_systema.py,
so the two analyses are computed on one object. The null is a permutation over the
target labels, which keeps the number of co-complex pairs and the whole similarity
distribution fixed and only breaks the correspondence between them.
"""

import argparse
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--corum", type=Path, default=Path("data/external/annotations/humanComplexes.txt"))
parser.add_argument("--permutations", type=int, default=20000)
args = parser.parse_args()

shifts = np.load(args.output / "celleval2_shifts.npy")
targets = pd.read_csv(args.output / "celleval2_systema_summary.csv").target.tolist()
bins = pd.read_csv(args.output / "celleval2_response_bins.csv").set_index("target")
index = {t: i for i, t in enumerate(targets)}

unit = shifts / np.linalg.norm(shifts, axis=1)[:, None]
similarity = unit @ unit.T

corum = pd.read_csv(args.corum, sep="\t")
corum = corum[corum.organism == "Human"] if "organism" in corum else corum
members = [set(str(s).split(";")) & set(targets) for s in corum.subunits_gene_name.fillna("")]
complexes = [(name, m) for name, m in zip(corum.complex_name, members) if len(m) >= 2]
print(f"CORUM: {len(corum)} human complexes; {len(complexes)} contain 2+ of our 150 targets")

co_pairs = {tuple(sorted(p)) for _, m in complexes for p in combinations(sorted(m), 2)}
all_pairs = list(combinations(range(len(targets)), 2))
is_co = np.array([tuple(sorted((targets[i], targets[j]))) in co_pairs for i, j in all_pairs])
values = np.array([similarity[i, j] for i, j in all_pairs])
print(f"{is_co.sum()} co-complex pairs among {len(all_pairs)} total pairs\n")

observed = values[is_co].mean() - values[~is_co].mean()
rng = np.random.default_rng(0)
null = np.empty(args.permutations)
order = np.arange(len(targets))
for k in range(args.permutations):
    rng.shuffle(order)
    permuted = np.array([tuple(sorted((targets[order[i]], targets[order[j]]))) in co_pairs for i, j in all_pairs])
    null[k] = values[permuted].mean() - values[~permuted].mean()
p = (1 + (null >= observed).sum()) / (1 + args.permutations)

print(f"mean cosine, co-complex pairs : {values[is_co].mean():.4f}")
print(f"mean cosine, other pairs      : {values[~is_co].mean():.4f}")
print(f"difference                    : {observed:.4f}")
print(f"permutation null              : {null.mean():.4f} +/- {null.std():.4f}")
print(f"one-sided p                   : {p:.2e}   (z = {(observed - null.mean()) / null.std():.1f})")

print("\nco-complex pairs, most to least similar:")
rows = []
for i, j in all_pairs:
    key = tuple(sorted((targets[i], targets[j])))
    if key in co_pairs:
        shared = [n for n, m in complexes if key[0] in m and key[1] in m]
        rows.append({"a": key[0], "b": key[1], "cosine": similarity[i, j],
                     "complexes": len(shared), "example": shared[0][:52]})
pairs_frame = pd.DataFrame(rows).sort_values("cosine", ascending=False)
print(pairs_frame.head(20).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
pairs_frame.to_csv(args.output / "celleval2_cocomplex_pairs.csv", index=False)

(args.output / "celleval2_specificity_run.json").write_text(json.dumps({
    "created_utc": datetime.now(UTC).isoformat(),
    "question": "do co-complex perturbations give more similar responses than random pairs?",
    "shifts": "Systema s_X from celleval2_h1_systema.py (log-normalized centroid differences)",
    "annotation": {"source": "CORUM human complexes", "file": str(args.corum),
                   "url": "https://mips.helmholtz-munich.de/corum/",
                   "complexes_with_2plus_targets": len(complexes), "co_complex_pairs": int(is_co.sum())},
    "mean_cosine_cocomplex": float(values[is_co].mean()),
    "mean_cosine_other": float(values[~is_co].mean()),
    "difference": float(observed),
    "permutations": args.permutations,
    "null_mean": float(null.mean()), "null_sd": float(null.std()),
    "p_one_sided": float(p)}, indent=2) + "\n")
