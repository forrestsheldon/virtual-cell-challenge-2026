# Responder detection — deferred research notes

**Status:** deferred by Forrest on 2026-09-24; algorithm development is not part of the current analysis; raw cell retention is requested.

Establish signal/noise methodology first. A future responder analysis requires single-cell
calculations, potentially iterating responsibilities and differential-expression genes, with
held-out calibration and null controls. No algorithm is selected or authorized here.
Forrest subsequently requested durable retention of responder-ready raw counts for likely use
within two weeks: broad scPerturb multi-construct panels (115/135 targets), the original 17
X-Atlas multi-construct targets, and all eligible controls from their batches. Added X-Atlas
genes remain pseudobulk-only. Preserve native genes, source/cell identifiers and assignments;
no automatic expiry or inferred responder labels. See the artifact contract and preflight for
the 31.71 GiB extra CSR estimate and independent acceptance checks. Build approval is still
required; this decision retains inputs without implementing the model.

The following preserves the earlier illustrative approach for later review, not implementation.

## 5. Test responder dilution without selecting the answer

Responder analysis changes the question from an all-labelled-cell effect to a latent conditional
effect. It is an optional second layer, not a preprocessing fix or part of the core pseudobulk
release. Run it only if approved cell-level counts and held-out folds are available; otherwise
stop this branch. Added count-ranked X-Atlas genes do not automatically enter this panel.

For each target:

1. Learn a P1-versus-control diagonal Fisher direction in training batches, excluding the target
   gene and selecting genes inside the training fold.
2. Fit and calibrate the responder model without using evaluation cells.
3. Apply the frozen P1 direction to held-out P2 cells. A high-score P2 tail is direct evidence for
   responders sharing the P1 program.
4. Independently learn a P2 direction and validate it on held-out P2 cells to test for a different
   reproducible program.
5. Form soft posterior-weighted responder pseudobulks in two disjoint held-out cell splits and
   rerun the split-power calculation.

On the linear rate scale, dilution predicts

\[
\delta_{\rm all}
=\omega\delta_R+(1-\omega)\delta_N
\approx\omega\delta_R,
\]

where `omega` is the responder share of exposure, not necessarily the fraction of cells. A dilution
claim requires all of the following on held-out cells:

- positive responder-only split power;
- higher P1/P2 responder-direction agreement than all-cell agreement;
- all-cell and responder-only effects aligned;
- attenuation close to the exposure-weighted responder fraction;
- separation from non-targeting pseudo-perturbations passed through the same classifier.

Mixscape is a sensitivity, not the sole label authority. The earlier H1 analysis showed both real
heterogeneity and large false responder fractions for five non-inert control guides. The primary
responder analysis will therefore use nested cross-fitting and retain continuous scores/posteriors.

