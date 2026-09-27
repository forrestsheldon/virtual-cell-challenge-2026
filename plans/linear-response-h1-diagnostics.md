# Linear-response H1 diagnostics

**Status:** superseded by `plans/linear-response-ladder.md`; retained for history  
**Authority:** historical only; do not execute any instruction below  
**Prepared:** 2026-09-03  
**Supersedes:** `linear-response-three-model-handoff.md` and `cipher-fixes-and-eval-split.md`

The current phase implements and reviews three H1 diagnostic models without producing
VCC 2026 A/B/C predictions or using leaderboard slots.

1. Fix Model 1's calibration in the 50,000-count pseudobulk coordinate and benchmark its
   exact expected output before sampled-cell effects.
2. Keep Model 2 as a diagnostic. Measure both its near-identity in raw direction and the
   downstream magnitude inflation caused by target-matched calibration.
3. Keep truth ownership in a lightweight evaluation-side profile script. It consumes
   model-only expected profiles and uses the official `cell_eval2` PDS kernel; the full
   H1 harness remains a later, separate cell-level evaluation.
4. Estimate H1 reliability with disjoint 200-vs-200 splits of the canonical 400 cells;
   there is no second disjoint 400-cell panel for all 126 targets.
5. Implement Model 3 only after the shared calibration and lightweight evaluator pass.
   Retain rank 50 and stop for review if its likelihood or posterior-predictive checks fail.
6. Generate future multi-GB H1 artifacts one at a time, retain hashes and compact results,
   then remove replaceable intermediates before continuing.

The shared code stays under `scripts/linear_response/` with a small `kernel.py`; no
`src/vcc_lr`, YAML configuration, or local `metrics.py` is introduced. Batch centering,
top-mode projection, correlation columns, and diagonal shrinkage are retired diagnostics,
not implementation work. No commit, VCC package, authentication, upload, or submission is
authorized by this plan.
