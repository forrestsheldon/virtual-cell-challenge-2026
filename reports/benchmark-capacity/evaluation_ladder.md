# Evaluation ladder

```text
E0 Sampling / metric ceiling
    |
E1 Oracle-response distribution generation
    |
E2 Cross-study K562 calibration
    |
E3a Pairwise within-study context transfer
    |
E3b Multi-source within-study context holdout
    |
E4 Replogle-anchored cross-study context transfer
    |
E5 Multi-source canonical-context holdout
```

Each rung adds a distinct uncertainty. E0 isolates finite sampling and the metric.
E1 adds cell-distribution generation while exposing the true response. E2 changes
study/protocol while nominally holding K562 fixed. E3a measures one ordered
within-study source-target context pair at a time. E3b uses all permitted source
contexts in the study and holds out one target context. E4 fixes the response
anchor to Replogle K562 and transfers only to other studies. E5 learns from all
permitted sources while withholding every dataset-context that shares the target
canonical context. Holding out only the target experimental dataset-context is
retained as an **E5 ablation**, not a separate main rung.

## Nominal capacity

| rung | target contexts | unique genes | gene-evaluation units | main contributors and limits |
|---|---:|---:|---:|---|
| E0 | 5 quantified | 9,884 | 17,198 | Per-target counts known for H1, Replogle, and Nadig; the summary uses the minimal >=2-cell split only. Jiang and McFaline-Figueroa remain unquantified, and H1 is not currently raw counts. |
| E1 | 13 | 10,125 | 19,923 | Nominal control + raw-count contexts from Replogle, Nadig, Jiang, and McFaline-Figueroa; no minimum cell count or response rule. H1 excluded from this count-level total. |
| E2 | 1 | 187 | 187 | Replogle K562 -> Jiang K562; changes 10x to Parse as well as study. A matching name does not prove a matched state. |
| E3a | 13 across 40 ordered pairs | 3,017 | 19,242 | Each directed within-study source-target pair uses `target ∩ source`. |
| E3b | 13 | 3,017 | 12,444 | Main eligibility is `target ∩ union(source contexts)`; the all-source intersection is retained as a stricter reported subset. |
| E4 | 12 | 2,828 | 6,926 | Replogle K562 anchor with cross-study targets only; same-study RPE1 belongs in E3a/E3b. |
| E5 | 14 | 3,083 | 13,020 | Hold out all instances of the canonical context. |
| E5 ablation | 14 | 3,083 | 13,020 | Hold out only one experimental dataset-context. Current nominal totals match E5, but K562 permits one extra training context and the information boundary is weaker. |

## E0 split-size view

The five contexts with per-perturbation cell-count metadata have the following
descriptive counts. A 400-vs-400 split requires at least 800 reported cells, and
so on.

| dataset-context | targets with counts | 400v400 | 200v200 | 100v100 | 50v50 | verified raw counts |
|---|---:|---:|---:|---:|---:|---|
| `vcc2025_h1::H1` | 150 | 96 | 126 | 139 | 145 | no |
| `replogle2022::K562` | 9,870 | 29 | 502 | 4,257 | 8,127 | yes |
| `replogle2022::RPE1` | 2,393 | 14 | 37 | 186 | 783 | yes |
| `nadig2025::HepG2` | 2,393 | 1 | 13 | 58 | 255 | yes |
| `nadig2025::Jurkat` | 2,393 | 11 | 42 | 184 | 905 | yes |

The H1 row is useful for sampling arithmetic but not yet for a raw-count
`cell-eval2` execution claim. Per-target counts for Jiang and
McFaline-Figueroa require inspection of larger source objects.

## E3a pairwise capacity

| study | ordered source-target pairs | shared-gene evaluation units |
|---|---:|---:|
| Replogle | 2 | 4,786 |
| Nadig | 2 | 4,784 |
| Jiang | 30 | 6,540 |
| McFaline-Figueroa | 6 | 3,132 |

## E3b per held-out context

| study | held-out target | source contexts | main genes | strict genes |
|---|---|---:|---:|---:|
| Replogle | K562 | 1 | 2,393 | 2,393 |
| Replogle | RPE1 | 1 | 2,393 | 2,393 |
| Nadig | HepG2 | 1 | 2,392 | 2,392 |
| Nadig | Jurkat | 1 | 2,392 | 2,392 |
| Jiang | each of A549, MCF7, HT29, HAP1, BxPC3, K562 | 5 | 218 each | 218 each |
| McFaline-Figueroa | each of A172, T98G, U87MG | 2 | 522 each | 522 each |

Main and strict capacities happen to match because target panels are shared
within these study groups. The main rule remains the union rule for future data.

## E4 per target context

| target dataset-context | shared with Replogle | VCC-300 | non-VCC | relationship |
|---|---:|---:|---:|---|
| `vcc2025_h1::H1` | 136 | 13 | 123 | cross-study, cross-context |
| `nadig2025::HepG2` | 2,393 | 0 | 2,393 | cross-study, cross-context |
| `nadig2025::Jurkat` | 2,393 | 0 | 2,393 | cross-study, cross-context |
| each Jiang context (6) | 187 | 9 | 178 | cross-study; Jiang K562 is the E2 same-context case |
| each McFaline-Figueroa context (3) | 294 | 17 | 277 | cross-study, cross-context |

## E5 exposure capacity

For target perturbation `g`, let `k_g` be the number of permitted source
contexts in which `g` has been observed. Labels are **Seen-1** (`k_g == 1`),
**Seen-multi** (`k_g >= 2`), and **Unseen** (`k_g == 0`).

| target dataset-context | target genes | Seen-1 | Seen-multi | Unseen |
|---|---:|---:|---:|---:|
| `vcc2025_h1::H1` | 150 | 66 | 73 | 11 |
| `replogle2022::K562` | 9,870 | 66 | 2,762 | 7,042 |
| `replogle2022::RPE1` | 2,393 | 0 | 2,393 | 0 |
| `nadig2025::HepG2` | 2,393 | 0 | 2,393 | 0 |
| `nadig2025::Jurkat` | 2,393 | 0 | 2,393 | 0 |
| each Jiang context (6) | 218 | 0 | 218 | 0 |
| each McFaline-Figueroa context (3) | 522 | 0 | 522 | 0 |

The E5 ablation has the same class totals in the current atlas, although its
per-gene `k_g` differs for K562 because the other K562 study remains permitted.
`benchmark_gene_exposure.csv` records `k_g` and the label for every target gene
under both the main E5 boundary and the ablation.
