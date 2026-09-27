# Fixing Model 1 and splitting evaluation from modelling

**Status:** superseded by `plans/linear-response-h1-diagnostics.md`; retained for history
**Authority:** historical only; do not execute any instruction below
**Prepared:** 2026-09-03

Two jobs, done together because they touch the same code:

1. Fix the defects found in the normalized CIPHER-style linear response (Model 1).
2. Move evaluation out of the modelling code and onto Codex's self-contained H1 harness
   (`scripts/evaluation/vcc2026_h1.py`), leaving a single narrow seam between them.

Every number below was computed against the repository, most of them twice.

---

## 1. What changed since the review

### Confirmed defects

| id | defect | corrected magnitude |
|----|--------|--------------------|
| **F1** | The MSE baseline is unmatched: the prediction is a multinomial *resample* of the 400 source cells, the baseline is those same cells' raw counts. At `a=0` the decoder draws again, so the model pays one sampling layer the baseline never does. | Null/base = **1.0894**; model/base = 1.2747; the exact generator-free ratio is **1.1863**. About 32% of the reported excess is generator artefact. |
| **F2** | Amplitude calibration runs in a different coordinate from the metric. | `calibrate_amplitudes` measures the on-target shift at `log_pseudobulk(..., 10_000)` while `main` evaluates at `50_000` — **in the same script**. Median on-target shift is −0.639 at 10k vs −1.501 at 50k. That 2.3× is the defect; the per-cell-vs-bulk distinction is worth only 1.6% (Pearson 0.9987 at a common scale). Realized on-target shift −1.026 vs intended −0.547 vs observed −1.465; rank correlation with truth 0.13. Nothing checks it, because every metric excludes the target gene. |
| **F3** | "The fit itself is stable" cites the bootstrap interval of the *median* (−10.60, −8.93). | The transfer-relevant spread is per-target: `a_t` ∈ [−51.4, −1.07], MAD 2.51, 3.41× p10–p90. |
| **F4** | No noise ceiling, so no metric is interpretable. | Split-half cosine of two disjoint 400-cell truth vectors: median **0.594**; sampling noise is 15% of a truth vector's energy; **PDS ceiling 0.990**. In the cleanest ceiling quartile the model still scores cosine 0.008. |
| **F5** | The mechanism is under-reported. | 61.4% of the unit-normalized covariance columns' variance sits in **3 axes**; effective rank 11.1. Truth effects: 17.3% in 3 axes, effective rank 88.3. The axes are the **S→G2M cell-cycle axis** (Spearman −0.78 with S score, +0.76 with G2M), a second cycling axis, and a **hypoxia/glycolysis stress** axis. A target's predicted direction sits at cosine 0.81 from its nearest panel neighbour; the truth effects sit at 0.26. |
| **F7a** | `assert not np.allclose(directions[:,0], directions[:,1])` is the only specificity check. | Replaced by the F5 geometry as a tracked artifact plus a real assertion. |
| **F7b** | Benchmark cosine/Spearman use un-differenced vectors. | Paired form gives median per-target cosine +0.0079. Moot after the split — cell_eval2 owns this. |
| **F7e** | Model 1's prediction never went through the harness. | Fixed by the seam. |

### Retracted from the review

- **F6 (batch structure) is a null. Do not build the batch-held-out fix.** Batch-centering the control
  matrix within each of 48 batches moves the covariance column by cosine **0.9993** (q10 0.9970);
  a 24-vs-24 batch-held-out split reproduces the within-batch number; the columns hold 0.925–0.930
  median cosine across four independent held-out splits including whole sequencing runs and whole
  guide identities. The columns are extraordinarily reproducible — and reproducibly wrong. That is
  a better sentence for the write-up than anything the batch diagnostic would have produced.
- **"Project the top modes out" is refuted.** It looks like a real PDS gain (0.577 vs 0.493) and is
  not: fit independently on the two control halves, the complement direction's split-half cosine is
  **0.019** (comp-100) against 0.929 for the full column, and the PDS lift falls to 0.531 on the
  other half. Negative controls confirm the specificity: dropping mid-spectrum modes instead of the
  top ones changes nothing (PDS 0.494, split-half 0.929).
- **Root-finding the amplitude to the held-out target's own observed shift leaks that target.** The
  review proposed it; it is replaced below.
- **Per-target regression coefficients `C_{:g}/C_gg` are admissible but empirically worse.** They
  force an identical intended on-target shift for every target, whereas `d_tt` is genuinely
  correlated with the observed shift (Spearman +0.53 → 0). Not adopted.

### New, and decision-relevant

**Model 2 is not a distinct rung.** Verified twice, independently:

| k | % of control variance | median cos(C_k[:,g], C[:,g]) | q10 | median `d_tt` fraction |
|---|---|---|---|---|
| 3 | 5.2% | 0.777 | 0.577 | 0.034 |
| 20 | 11.2% | 0.978 | 0.930 | 0.093 |
| 50 | 13.5% | **0.984** | 0.941 | 0.113 |
| 100 | 15.1% | 0.986 | 0.943 | 0.121 |

`C_{:g}` weights modes by λ, so the trailing near-isotropic bulk lands on the **diagonal**, not on the
off-diagonal direction. Top-50 carries 13.5% of the variance and reproduces 98.4% of the direction
while recovering only 11.3% of `d_tt`. Scored against truth: median cosine 0.0079 (top-50) vs 0.0074
(full), mean PDS 0.4912 vs 0.4931 — a difference of ~0.15σ of Model 1's own five-replicate seed noise
(cosine sd 0.0035, PDS sd 0.0121).

Worse, under the plan's own rule `a_t = Δ_tt/d_tt`, the shrunken diagonal makes Model 2 run at **8.9×**
Model 1's amplitude along an essentially identical direction — a purely notational artefact that would
push its MSE ratio toward 20. Truncation *is* a mild denoiser (independent split-half cosine 0.982 vs
0.929 for the full column); that is its only real benefit, and it moves no downstream metric.

**This is what makes output-matched calibration the load-bearing fix**, not cleanup: as specified, the
three rungs calibrate against three different diagonals (per-cell log1CP10k variance, its top-50
truncation, and a latent log-rate variance) and would differ by an order of magnitude in effect size
for reasons that have nothing to do with the models.

---

## 2. The seam

**The seam is a prediction `.h5ad` on disk. Nothing else crosses it.**

- **Modelling** writes `data/derived/linear_response/<model>/<arm>.h5ad` (gitignored):
  50,400 cells × 18,080 genes, `obs` exactly `["target_gene"]`, the 126 canonical targets at exactly
  400 cells each, no controls, sparse CSR, finite non-negative integer-valued, per-cell totals in
  [1, 10⁶], **no explicitly stored zeros**.
- **Evaluation** consumes it: `pixi run python scripts/evaluation/vcc2026_h1.py score <path> --output reports/linear-response-three-models/scores/<model>/<arm>/`.
- **Nothing in `scripts/evaluation/` changes.** The modelling side imports the harness for exactly one
  purpose — calling `validate_prediction` in a test — and never for scoring.
- **The modelling side stops owning truth entirely.** Not "adopt crc32": it never reads a truth cell.
  `reports/vcc2026-h1/reference_cells.csv` is the authoritative panel and split.

Three consequences worth stating:

- `validate_prediction` does **not** check for explicitly stored zeros, but the handoff plan requires it
  and `vcc prep` will. Our writer guarantees it; a test asserts it.
- `augment_prediction` appends the real H1 controls itself, so **the model must not generate control
  cells**. The evaluator differences against all 38,176 NTCs while the model fits and samples from the
  strict-26 pool (32,616 cells). Measured, that mismatch is a shared offset of norm 0.124 against a
  median truth-effect norm of 4.23 — 2.9%, negligible in energy. It is a documentation item. (Its
  cosine against truth is systematically ≈ −0.058, contributing ≈ −0.002 to the measured cosine, which
  is the same order as the entire ±0.003 result. One more reason F4's ceiling has to be published.)
- `score` hard-requires the scale bundle, which does not exist yet — Codex's `build-scale` **anchor
  stage is running now**. Until it lands, use `validate` (which does not need the bundle) plus raw
  metrics. Do not ask Codex to make the bundle optional.

**Codex collision protocol.** `data/derived/vcc2026_h1/reference_cache` is Codex's, and `cell_eval2`'s
`CacheStore` takes an exclusive `fcntl.flock` on `manifest.json.lock` there. Both `validate` and `score`
default to it. **Always pass `--reference-cache data/derived/linear_response/eval_cache`.** Never write
to `scripts/evaluation/`, `tests/test_vcc2026_h1_eval.py`, `reports/vcc2026-h1/`, or
`data/derived/vcc2026_h1/`.

---

## 3. Work items

Ordered. Each leaves the repo working. **model** = modelling side, **seam** = crosses, **diag** = diagnostics.

### Stage 0 — safety

| id | side | what |
|----|------|------|
| `gitignore-parquets` | — | `git add --dry-run reports/crispri-h1-exploration/` currently stages a 96 MB parquet. The `.gitignore` diff does not cover them. Fix before any commit is contemplated. **Control:** re-run the dry run; assert no `*.parquet` and no `*.h5ad`. |
| `eval-cache-dir` | seam | Create `data/derived/linear_response/eval_cache` and use it in every harness invocation. **Control:** `lsof` shows no handle on Codex's cache during our runs. |

*Do not commit anything without asking Forrest first (AGENTS.md). Note that essentially all of this
work is untracked, so `git checkout`/`stash`/`restore` are inert — there is no undo.*

### Stage 1 — fix the model (must precede any prediction file)

**`fix-calibration-scale`** · model · *the actual F2 bug, ~10 minutes*
Make the scored pseudobulk scale one named constant `BULK_TARGET_SUM = 50_000` with its provenance
recorded (it equals `cell_eval2`'s `vcc2026` preset `bulk_target_sum`, and `log_pseudobulk` is exactly
`cell_eval2.prep.bulk_lognorm_means`). Replace both `10_000` calls in `calibrate_amplitudes`.
**Test:** `log_pseudobulk(c, 50_000)` equals `bulk_lognorm_means(c[None], 50000)[0]`.
**Control:** the recorded median on-target shift moves from −0.639 to −1.501.

**`knockdown-depth`** · model · *the transferable quantity*
Estimate the transferable quantity as **knockdown depth in log-fraction space**, leave-one-target-out:
`κ̄_t = median_{j≠t} log(f_pert,j / f_ctrl,j)`. This is the coordinate in which the quantity is actually
constant — MAD/|median| **0.073** and 1.37× p10–p90, against 0.24/2.61× for the scored-space shift and
0.26/3.41× for the current `a_t`.
**Test:** the LOO median excludes the held-out target (existing test pattern).
**Control:** report the spread of κ̄ alongside `a_t`'s, which is the honest replacement for F3.

**`invert-calibration`** · model · *replaces the leaky option (B)*
Per target, set the calibration target from control-only expression and the transferred depth:
```
target_shift_t = log1p(50000·p_ctrl,t·exp(κ̄_t)) − log1p(50000·p_ctrl,t)
a_t            = root of  realized_on_target_shift(a) = target_shift_t
```
Leakage-free: κ̄_t is a LOO median over *other* targets, `p_ctrl,t` is control-only. Monotone 1-D root
find, ~5 evaluations. Use the **closed form** for the realized shift (the per-cell weight total holds at
10⁴ to within 0.5%, giving accuracy 0.0084 max against the exact decoder) and run it on the full
32,616-cell control pool rather than the 400-cell draw — removes seed dependence for a ~1% change in `a_t`.
**Control:** the closed form matches the exact decoder on one target to <0.01; the root find recovers a
known `a` when handed a synthetic target shift generated at that `a`.

**`on-target-artifact`** · model · *the missing positive control*
Write `reports/linear-response-three-models/on_target_closure.csv`: per target the intended, realized,
and observed on-target shift, plus `a_t`, `κ_t`, `κ̄_t`. This is the one statistic the harness
structurally cannot supply, because every official metric excludes the target gene. It stays model-side
as a calibration diagnostic, never as a score.
**Control:** post-fix, realized/target ratio → 1.00 by construction; realized-vs-observed rank
correlation is reported (currently 0.13) as the honest measure of what calibration transfer achieves.

> **State the tradeoff in the report:** this fix makes the headline MSE *worse* (MSE is monotone in |a|
> with its optimum at a = 0) and leaves cosine and PDS **unchanged** — per-target scalars are invariant
> for both to <1% over a 160× range. The gain is that the model's one calibrated quantity becomes
> checkable, and is checked.

**`seed-split-arms`** · model · *F1 mechanics*
Split the single rng in `generate_one`: source-cell stream stays `[0, 1, MODEL_INDEX, source_order,
replicate]`, decode moves to a new `[0, 8, MODEL_INDEX, source_order, replicate, arm]` with
`arm ∈ {model:0, null:1}`. Same source cells across arms (56% of generator-noise energy cancels there),
independent decode draws. Add stream 8 to `seed_namespace()`. Read `data.X[source_rows]` once, decode
both arms off it. Rename to `generate_arms` returning a dict — **and update all four call sites**:
`main()`'s smoke test, and the three tests that import the old names.
**Control:** the null arm's source rows are byte-identical to the model arm's.
*Do not share the rng between arms — measured worthless (9.30 vs 9.33 difference variance) and it
makes decode noise inseparable from the source draw.*

**`expected-profile-arm`** · model · *the generator-free comparison*
Add the exact, draw-free expected pseudobulk of the multinomial decoder as a third arm. Densify before
computing totals (`raw.sum(axis=1)` on a csr returns an `(n,1)` matrix and mis-broadcasts).
**Control:** as the replicate count grows, the sampled arm's mean converges to the expected arm.

### Stage 2 — the seam

**`write-prediction`** · seam · *the writer both loops share*
One function, `write_prediction(counts_by_target, path)`, producing an h5ad satisfying
`validate_prediction`. Guarantee no explicit stored zeros (`eliminate_zeros()` then assert
`(X.data == 0).sum() == 0`).
**Test:** imports `vcc2026_h1.validate_prediction` and passes a generated file through it.
**Control:** a deliberately malformed file (399 cells for one target; a stored zero; wrong gene order)
is rejected on each count.

**`adopt-harness-truth`** · seam · *delete truth ownership*
Delete `collect_truth_counts`, the `h1_truth_cells.csv` read, `truth_effects`, `control_profile`, and
the `SeedSequence([0,0,target_index])` sampler — **and the call sites at
`benchmark_cipher_h1.py:527–529, 576`** and `diagnose_cipher_h1.py:233`. Read the panel from
`reports/vcc2026-h1/reference_cells.csv`. Keep `benchmark_candidate` in `h1_target_counts.csv` or fix
the four call sites that read it (`build_stage1_manifest.py:218,276`, `diagnose_cipher_h1.py:233`).

**`strip-evaluation`** · model · *delete the second scorer*
Delete `pds_scores`, `evaluate_profiles`, `summarize_metrics`, `plot_overview`'s metric panel from
`benchmark_cipher_h1.py`. **Keep `log_pseudobulk`** — move it to the shared kernel; the diagnostics
still need pseudobulks. Delete the three tests that cover departed functions
(`test_pds_scores_rank_the_matching_truth_first`, `test_profile_metrics_exclude_the_target_coordinate`,
`test_metric_summary_contains_model_and_control`) and fix
`test_tiny_end_to_end_response_pipeline`, which calls `evaluate_profiles`.
**Control (run it before deleting):** score one prediction through both the old `evaluate_profiles`
and the harness, and record the agreement in the report. After deletion this is no longer possible, so
it is the last chance to confirm the parity the review established for `pds_scores`.
Update `run_manifest.json`'s `evaluation` **and** `outputs` blocks — the latter carries sha256 entries
for the two CSVs being deleted.

**`emit-arms`** · seam · *the files*
Emit `model.h5ad` and `null.h5ad` for replicate 0. **One replicate scored officially, not five** — each
file is ~3.5 GB, and `score` leaves `cache_pred=None` so every run recomputes prediction-side Wilcoxon
DE. Five replicates × two arms × three models would be ~105 GB and ~30 full DE runs. Multi-replicate
generator-variance work stays as in-memory pseudobulk diagnostics.

### Stage 3 — diagnostics (model side)

**`shared-axes`** · diag · *the write-up's central figure*
Tracked artifact: pairwise-cosine distributions for the covariance columns, the decoded predictions,
and the observed truth effects; spectral concentration (top-3 = 61.4% vs 17.3%; effective rank 11.1 vs
88.3); and **what the axes are** — correlate the top shared axes against per-cell S/G2M scores and
library size, and list top-loading genes.
**Controls:** random Gaussian directions (median truth cosine −0.0017, PDS 0.468) and gene-shuffled C
(cosine 0.0002, PDS 0.489) bracket the measurement scale.

**`specificity-assert`** · diag · replace `assert not np.allclose(...)` with an assertion on the
pairwise-cosine distribution, thresholded from the ceiling work.

**`spectral-note`** · diag · Record the four Model 2 numbers **instead of running Model 2**:
cos-to-full at k = 20/50/100, the 11.3% diagonal fraction, split-half 0.982 (top-50) vs 0.929 (full),
and 61.4%-in-3-axes.

**`noise-ceiling`** · diag+seam · *F4*
The harness owns the *cells*; both sides compute their own ceiling from them. Ask Codex to publish
`reports/vcc2026-h1/ceiling_cells.csv` (the disjoint second-half truth cells, same schema as
`reference_cells.csv`) — **this is the one thing that must be added on Codex's side**, and it is a CSV
plus a `score` run of a `ceiling.h5ad`, no new metric code. The modelling side reads that CSV and
computes a per-target cosine/Spearman/energy ceiling in its own space for stratification.
**Control:** the estimator returns ≈1.0 on two identical samples and ≈0 on two independent noise
vectors. *Do not* use "the cosine route and the energy route agree" as a control — that is an exact
algebraic identity, not an independent check.

### Stage 4 — score and rewrite

**`score-through-harness`** · seam · once `build-scale` lands, score model / null / ceiling through the
real `score` with `--reference-cache` pointed at our directory.

**`rewrite-report`** · — · Rewrite `reports/linear-response-three-models/README.md` **in the same steps
that invalidate it**, not at the end. Headline the **exact generator-free ratio 1.1863**, and debias
**additively** — `model/base − (null/base − 1) = 1.1852` recovers the exact value; the ratio
`model/null = 1.1700` does not. Report `null/base = 1.0894` beside it as the published size of the
artefact. For cosine/Spearman/PDS the null is a measured **chance band** (report q10/q50/q90), never a
denominator — do not divide any cosine by anything.

---

## 4. What becomes stale

| artifact | fate |
|---|---|
| `h1_truth_cells.csv` | **delete** — the LR side stops owning truth |
| `h1_summary.csv`, `h1_per_target.csv` | **delete** — replaced by the harness's `per_target.csv` / `aggregates.csv` / `scores.csv` |
| `amplitude_calibration.csv` | regenerate (new coordinate, new `a_t`) |
| `run_manifest.json` | regenerate — `evaluation` **and** `outputs` blocks |
| `data_split_manifest.json` | regenerate — new seed stream 8; the truth-split section goes |
| `cipher_de_*.csv`, `cipher_scaling_summary.csv`, `cipher_gene_scaling.csv`, `cipher_diagnostics_manifest.json` | regenerate — all derive from `cipher_h1_arrays.npz`, whose hash changes |
| `control_fit_diagnostics.csv` | keep the split-half columns, drop the batch framing (F6 retired) |
| `figures/cipher_h1_overview.png` | regenerate; the metric panel moves to harness numbers |
| `h1_target_counts.csv`, `h1_strict_controls.csv` | survive untouched |

---

## 5. Decisions for Forrest

**D1 — Cut Model 2?** *Recommend: yes.* Replace the rung with the four spectral numbers in
`spectral-note`. Its direction is cosine 0.984 to Model 1's, its scored difference is ~0.15σ of seed
noise, and under the plan's own amplitude rule it would run at 8.9× amplitude along the same direction.
Running it produces a redundant result, not a null one. Against: the plan explicitly freezes scope, and
"we ran it and it was identical" is a stronger claim in a write-up than "we computed that it would be."

**D2 — Does Model 1 get a leaderboard slot?** *Recommend: one submission, after the calibration fix, with the score preregistered in the report.* The predicted score is ≈0.00 to −0.02 (PDS 0.477 vs baseline
0.500 → ≈ −0.05; MSE ratio > 1 → capped 0). Nothing else is ready to compete for the slot, so today's
opportunity cost is zero, and it buys the one number no local computation can produce: the H1→A/B/C
transfer coefficient. Framed as a test of a stated numeric prediction rather than a fishing expedition,
it is worth the slot. **Do not submit the current miscalibrated artifact.** Note the control-resampling
baseline was packaged but never submitted, so no slot has been spent.

**D3 — Repo layout.** *Recommend: reject the handoff plan's `src/vcc_lr/{io,normalize,amplitude,generation,metrics}.py + models/ + configs/`.* Six modules and two YAML files for three run-once scripts
whose shared surface is five functions, against an AGENTS.md that explicitly prefers a short script to a
framework. Adopt: everything stays in `scripts/linear_response/`, one shared `kernel.py`
(`log1cp10k`, `log_pseudobulk`, `sample_balanced_controls`, `decode_multinomial`, `write_prediction`),
one script per model, no `configs/`. **`metrics.py` must not be created** — after the split every
function it would hold is either duplicated `cell_eval2` logic or departed scoring, and creating it
would silently re-establish the second unvalidated scorer that F1 and F2 came from.

**D4 — Model 3 before or after this?** *Recommend: after.* Two reasons beyond ordering. Its amplitude
is specified as `a_t = Δ_t / C_tt` with `C_tt` the **latent** log-rate variance, which excludes shot
noise — so it will be inflated exactly the way Model 2's is, and `invert-calibration` is the convention
that makes all three rungs comparable. And the constraint `qᵀL ≈ 0` will **not** act as a shared-axis
filter: only 16.1% of `q`'s norm lies in span(PC1–3) (|cos(PCᵢ, q)| = 0.067, 0.115, 0.090 for i=1,2,3),
so Model 3 inherits the same failure mode. Feasibility is not the blocker — Laplace-MAP at k=50 over
32,616 cells costs ~32 s/sweep with a batched Hessian (1809 s naive), 5.6 s at k=20.

**D5 — One `ceiling_cells.csv` from Codex.** The only thing this plan asks of the evaluation side.
Confirm with Codex before assuming it.

---

## 6. Deliberately not doing

- **The batch-held-out covariance fix (F6).** Empirically null: cosine 0.9993. Retired, not deferred.
- **The top-mode complement, correlation columns, diagonal shrinkage, per-target regression
  coefficients.** The first is sampling noise (independent split-half cosine 0.019); the second is a
  null (PDS 0.4913); the last two leave the downstream direction algebraically unchanged — they are
  amplitude conventions, not models.
- **Five officially-scored replicates.** One per arm. `expr_mse_unbiased_capped_norm` already subtracts
  a delete-1 jackknife estimate of the prediction's own sampling variance, so the F1 layer is corrected
  inside the metric; multi-replicate work stays a model-side pseudobulk diagnostic.
- **Any fourth model.** Scope is frozen; D1 subtracts a rung rather than adding one.
- **Touching `scripts/evaluation/`** beyond the single `ceiling_cells.csv` request in D5.

---

## 7. Provenance

Numbers marked verified twice: the Model 2 truncation table and the covariance positive control
(cos 0.9999999529 against the cached `directions`) were recomputed independently with a separate
randomized SVD. F1's energy decomposition, F4's ceiling, F5's geometry, and the strict-vs-all-NTC
offset were computed directly against `data/derived/linear_response/cipher_h1_arrays.npz` and the H1
H5AD. The cell-cycle identification of the shared axes, the κ̄ dispersion statistics, the closed-form
decoder accuracy, the additive-vs-ratio debiasing comparison, and the Model 3 timings come from the
workflow agents and carry their own controls, but were **not** independently re-derived here.
