# Does the best transfer pseudocount track the evaluation space?

Run 2026-09-25 with `scripts/evaluation/rescore_eval_pseudocount.py`. This rescores the H1
sweep (single sources and the equal-source average, direct cosine/retrieval, and cross-fitted
effect-space scale MSE) and the donor-to-donor sweep in the spaces ln(1 + CPM/c_eval) for
c_eval = 1, 20 and 100. Positive controls: c_eval = 20 reproduces
`reports/h1-transfer-regression/summary.csv` and `reports/donor-pair-pseudocount/pairs.csv`
exactly.

## Transfer pseudocount with the highest held-out cosine (average predictor)

| Setting | c_eval = 1 | c_eval = 20 | c_eval = 100 |
|---|---:|---:|---:|
| H1, five anchors (110 targets) | 1 | 10 | 30 |
| H1, all nine (24 targets) | 3 | 10 | 30 |
| donor to donor, five anchors | 1 | 10 | 30 |
| donor to donor, all nine | 10 | 30 | 30 |
| H1 five anchors, lowest scaled MSE | 3 | 30 | 30 |

The optimum moves with the evaluation space but less than proportionally: a 100-fold change
in c_eval moves it about 10–30-fold. Metric values are not comparable across c_eval, because
each space weights low-count genes differently.
