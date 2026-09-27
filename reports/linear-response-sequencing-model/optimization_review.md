# Poisson-lognormal optimization review

## What made the previous fits slow

The full `LL^T + D` implementation spent essentially all of its time in the
per-cell E-step. Each 64-cell batch ran a fixed 30 Newton iterations. Every
Newton iteration ran a fixed 40-trial Armijo scan over all 18,080 genes, even
after individual cells had converged. The E-step also formed posterior blocks
and `G x 50` moments, restarted every cell at `z=0, x=mu` at every outer
iteration, and used float64 throughout.

| Recorded operation | Time |
|---|---:|
| Pure-factor differentiable Laplace fit, one start | 80–82 min |
| Full `LL^T + D` E-step, median of five complete iterations | 1.92 h |
| Full M-step plus checkpoint write | 23–51 s |
| Intended full run | up to 10 outer iterations × 3 starts |
| Factorial-moment fit, control and 151-state balanced operators | 161 s |
| Cold expected-profile evaluation after I/O/vectorization fixes | 39 s |

The five saved full-model updates had not converged. The final NLL improvement
was 8.46 per cell versus a declared 0.01 tolerance, covariance change was 1.40%
versus 0.1%, and the M-step gradient RMS was 0.0666 versus 0.01. The old job
record still says `running`, but no worker exists; its last complete checkpoint
is outer iteration 5 and it stopped partway through iteration 6.

## Fast estimator implemented

For observed depth `s_c` and counts `Y_cg`, the draft measurement model gives

\[
Y_{cg}\mid x_{cg},s_c\sim\operatorname{Poisson}(s_c e^{x_{cg}}).
\]

The cell-equal factorial moments obey

\[
E[Y_{cg}/s_c]=E[e^{x_{cg}}],
\qquad
E\left[\frac{Y_{cg}(Y_{ct}-1[g=t])}{s_c^2}\right]
=E[e^{x_{cg}+x_{ct}}].
\]

If `x` is Gaussian, the required covariance columns follow directly:

\[
\Sigma_{gt}=
\log\frac{E[Y_g(Y_t-1[g=t])/s^2]}
{E[Y_g/s]E[Y_t/s]}.
\]

This removes Poisson variance with the factorial diagonal, estimates only the
126 requested columns on the 10,783-gene panel, and has no latent E-step or
iterative optimizer. The control operator uses the strict 26-guide pool. The
all-state operator equally averages first and factorial-second moments across
the control state and 150 perturbation states before moment-matching a single
lognormal covariance.

Positive controls:

- dense and streaming factorial-moment calculations agree;
- the identity reconstructs an exact supplied lognormal covariance;
- a simulated Poisson-lognormal sample recovers its known covariance column;
- all 126 final target variances are positive;
- both final covariance artifacts are finite;
- every normalized target response coefficient is exactly `-1`;
- deterministic within-batch halves are complete, disjoint, and balanced.

## Initial H1 result

Every model used one shared fixed 400-cell strict-control panel. Normalized
responses were decoded in log1p-CP10K space; Poisson-lognormal responses were
decoded as multiplicative latent log-rate shifts. Magnitude was selected
separately for each model by leave-one-target-out downstream L1 transfer.

| Model | Transductive | Median gamma | Signed top-K recovery | Wrong-target mean | Wrong-target p | Strong-DE NMAE |
|---|---:|---:|---:|---:|---:|---:|
| Control log1p-CP10K | no | 0.968 | 0.0365 | 0.0350 | 0.293 | 1.007 |
| Control factorial PLN | no | 0.189 | 0.0424 | 0.0405 | 0.263 | 1.006 |
| Control iterative rank-50 PLN | no | 0.232 | 0.0188 | 0.0164 | 0.124 | 1.016 |
| All-state combined log1p-CP10K | yes | 2.410 | 0.0752 | 0.0490 | 0.0001 | 0.960 |
| All-state combined factorial PLN | yes | 0.718 | 0.0618 | 0.0482 | 0.0015 | 0.998 |

The sequencing-aware control result does not exceed its wrong-target null. The
all-state result contains target-specific information, but it is transductive
because each evaluated target state contributed to the fitted operator. The
factorial PLN weakens both recovery and magnitude fidelity relative to the
normalized all-state operator.

The latent-log-rate response is genuinely much denser. The median control
response has 53.8% of downstream coefficients above `0.1`, versus 0.69% for
log1p-CP10K, and downstream norm 20.5 versus 2.87. Cross-fitted magnitude
shrinks accordingly. Despite that density, its unnormalized expected total-rate
ratio remains close to one: median 1.0007 and 5th–95th percentile
1.0002–1.0020 for the control factorial model.

## Options for likelihood-based successors

| Option | Expected speed | What it adds | Main cost or caveat |
|---|---|---|---|
| Factorial target moments | minutes; implemented | Correct Poisson first/second moments and the exact LR columns requested | Does not produce a globally PSD covariance or cell posteriors; the all-state mixture is only moment-matched |
| Poisson-residual randomized factorization | minutes | PSD low-rank generator and arbitrary new columns | Uses a small-covariance/working-residual approximation rather than the exact lognormal likelihood |
| Stop-gradient alternating MAP | tens of minutes on CPU; potentially minutes on GPU | A fitted low-rank Poisson factor model | MAP ignores posterior uncertainty unless a separate correction is added |
| Collapsed `LL^T + D` Laplace | likely 10–30× faster than the old full fit | Closest continuation of the document-matched likelihood | More implementation work; still unnecessary if only target columns are wanted |
| Amortized variational encoder | fast for repeated large datasets | Scales to external atlases and rapid per-cell inference | Adds a neural inference model and variational bias to a simple scientific comparison |

For the collapsed model, the conditional mode of each residual log rate has a
closed form given `z`:

\[
x_g=\eta_g+d_gY_g-W\!\left(d_gs\exp(\eta_g+d_gY_g)\right),
\qquad \eta=\mu+Lz.
\]

Using this relation, or a few vectorized scalar Newton updates for the Lambert-W
term, eliminates the old joint `G + k` Newton solve and its 40-step line search.
The remaining rank-50 mode should be warm-started across outer iterations and
stopped adaptively. The E-step should be detached from autodiff and the M-step
should consume sufficient posterior moments. Float32, the 10,783-gene panel,
and larger batches are secondary gains after removing the fixed nested scans.

## Recommendation

Use factorial target moments as the default sequencing-aware LR estimator.
Build the Poisson-residual randomized factorization next only if a PSD generator
or arbitrary unrequested covariance columns become necessary. Keep a collapsed
Laplace fit on a small gene/cell subset as a validation reference. Do not resume
the old full EM loop unchanged.

The independent-Poisson model is an approximation after conditioning on an
observed library total; the corresponding multinomial correction is order
`1/s_c` and is tiny at H1 depths. Gene capture efficiencies remain unidentified
across experiments, but their constant log offsets cancel in within-experiment
covariances. Neither statement licenses direct cross-experiment covariance
comparison without an anchor population.

## Log1p limit and factorial-PLN regularization

Let `kappa` be the normalization target and

\[
L_{cg}=\log(1+\kappa Y_{cg}/s_c).
\]

When both the normalized abundance and expected raw count are large,

\[
L_{cg}\simeq \log\kappa+X_{cg}+\text{small Poisson error}.
\]

The off-diagonal covariance of log1p-normalized expression therefore approaches
the latent log-rate covariance. At low abundance, however,

\[
L_{cg}\simeq \kappa Y_{cg}/s_c
\]

and, for distinct genes,

\[
\operatorname{Cov}(L_g,L_t)
\simeq
\kappa^2 E[e^{X_g}]E[e^{X_t}](e^{\Sigma_{gt}}-1).
\]

Log1p consequently attenuates latent covariance by the product of the mean
rates. The factorial estimator reverses that attenuation, which is correct
under the model but noisy when either rate is small.

Five deterministic guide-by-batch-balanced control splits were used to estimate
coefficient uncertainty. A zero-centered normal scale mixture was then fitted
by marginal likelihood, and posterior means supplied the regularized
off-diagonal covariance columns. No perturbation truth was read during fitting.

The adaptive estimator improved median split-half cosine from 0.846 to 0.883,
reduced held-out log-covariance MSE from `6.33e-5` to `5.28e-5`, and retained
77.9% of raw covariance energy. In the 5--10 CPM bin it improved cosine from
0.743 to 0.802 while retaining 64.9% of the energy. By contrast, a 100-CPM
abundance pseudocount raised overall cosine to 0.925 only by retaining 2.5% of
the covariance energy.

The fitted mixture put only 0.025% weight on the exact-zero component and most
of its weight on small continuous scales between roughly 0.0036 and 0.013.
Within this approximate exchangeable-coefficient model, the control covariance
looks weakly dense rather than spike-sparse; dependence among coefficients and
the repeated-split standard-error approximation make this descriptive rather
than a formal sparsity estimate.

After freezing the regularizer, H1 signed top-K recovery was 0.0386 versus a
wrong-target mean of 0.0356 (`p=0.159`). Raw factorial PLN gave 0.0424 versus
0.0405 (`p=0.263`). Regularization therefore improves covariance estimation but
does not establish target-specific perturbation information.
