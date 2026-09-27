# Choosing the pseudocount without H1: donor-to-donor transfer

Run 2026-09-25 with `scripts/evaluation/donor_pair_pseudocount.py`. Each source in the frozen
`five_anchors` (110 targets) and `all_nine` (24 targets) panels is used in turn as the
destination: its control CPM plays h0 and its perturbed CPM plays the truth. Predictions come
from every other single source and from the equal-weight average of all the others. No H1
profile is used. The targets are still the H1-overlap targets, because those are the frozen
panels. Evaluation is the same ln(1+CPM/20) space, so the c = 20 confound in the H1 sweep
applies here too. Scaled MSE uses the same five-fold, effect-space, least-squares scale.

## Held-out cosine by pseudocount (mean over destinations)

| Panel | Donor | c=0 | 0.1 | 1 | 3 | 10 | 30 | 100 | inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| five_anchors | single, cross-study | .018 | .021 | .024 | .025 | **.025** | .025 | .023 | .020 |
| five_anchors | single, same-study | .028 | .034 | .037 | .038 | **.039** | .039 | .038 | .034 |
| five_anchors | average of others | – | .036 | .042 | .044 | **.045** | .044 | .041 | .031 |
| all_nine | single, cross-study | .095 | .098 | .100 | .101 | .102 | **.102** | .099 | .083 |
| all_nine | average of others | – | .191 | .201 | .206 | .211 | **.212** | .207 | .174 |

Pairs whose cosine is highest at each c (c > 0): five_anchors 15 of 25 at c = 10 (all 5
average-of-others destinations); all_nine 52 of 81 at c = 10 or 30.
Scaled-MSE optima fall at c = 10–30 as well. Retrieval is flat in c for single donors.
