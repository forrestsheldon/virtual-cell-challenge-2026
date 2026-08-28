# Perturbation-context bridge map

Treat each response-bearing `dataset_context_id` as a node and the number of
shared perturbation genes as an edge weight. This preserves study/protocol identity
while allowing a separate canonical-context collapse for biological summaries.
The machine-readable 91-edge graph is `metadata/datasets/bridge_edges.csv`.

## Strategic topology

- `replogle2022::K562` is the broad response anchor: 9,870 genes and positive
  overlap with every other response-bearing node.
- `vcc2025_h1::H1` is the challenge-relevant bridge: 150 genes, including 136
  shared with Replogle K562. It provides direct gene overlap, but its currently
  verified representation and unresolved assay family do not establish an
  Arc-like protocol-equivalence bridge.
- `replogle2022::K562` and `jiang2025::K562` are the only primary CRISPRi nodes
  sharing a canonical context across independent studies. They retain separate
  dataset-context IDs.
- Replogle, Nadig, Jiang, and McFaline-Figueroa each have multi-context structure.
  Jiang has six cell-line nodes; McFaline-Figueroa has three; their environmental
  conditions are collapsed in this first cell-line-level view.
- `vcc2026::A`, `vcc2026::B`, and `vcc2026::C` are control-only target nodes.
  They have no perturbation-response edges and their biological identities are
  intentionally not inferred.

## Strong edges

| source | target | shared genes | bridge type |
|---|---|---:|---|
| Replogle K562 | Replogle RPE1 | 2,393 | different context, within study |
| Replogle K562 | Nadig HepG2 | 2,393 | different context, cross-study |
| Replogle K562 | Nadig Jurkat | 2,393 | different context, cross-study |
| Replogle RPE1 | Nadig HepG2/Jurkat | 2,392 each | different context, cross-study |
| Nadig HepG2 | Nadig Jurkat | 2,392 | different context, within study |
| McFaline A172/T98G/U87MG pairs | each other | 522 each | different context, within study |
| Replogle K562 | each McFaline context | 294 each | different context, cross-study |
| Jiang context pairs | each other | 218 each | different context, within study |
| Replogle K562 | Jiang K562 | 187 | same named context, cross-study |

The strongest same-canonical cross-study edge is therefore K562 with 187 genes:
1.89% of the Replogle K562 panel and 85.78% of the Jiang K562 panel. This pair also
changes study and assay family (`10x_Perturb-seq` to `Parse_Perturb-seq`), and
named-cell-line equality does not guarantee equal cell state.

## Replogle anchor

Of 9,870 Replogle K562 genes, 2,828 occur in at least one other dataset-context,
2,762 in at least two, and 2,759 in at least three. All 2,828 occur in at least
one independent study; 177 occur in at least two independent non-Replogle
studies. The anchor contains 272 VCC-300 genes and 9,598 genes outside that panel.

Per-node Replogle overlaps are 136 for H1; 2,393 each for RPE1, HepG2, and Jurkat;
187 for each of six Jiang nodes; and 294 for each of three McFaline-Figueroa nodes.

## H1 connections

H1 shares 136 genes with Replogle K562, 50 with Replogle RPE1, 50 with each Nadig
node, 15 with each Jiang node, and 13 with each McFaline-Figueroa node. Collapsed
by independent study, H1's direct overlaps are 136 with Replogle, 50 with Nadig,
15 with Jiang, and 13 with McFaline-Figueroa. These are gene bridges only; assay,
state, and preprocessing comparability remain separate review questions.

## Broader registry repetitions

The context registry also repeats A549 and MCF7 across Jiang CRISPRi and Srivatsan
chemical-perturbation datasets. Those rows are useful provenance links, but the
chemical data are not included in this primary CRISPRi atlas or its E2 capacity.
