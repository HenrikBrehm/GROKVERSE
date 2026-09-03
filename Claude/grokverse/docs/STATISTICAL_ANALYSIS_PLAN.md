# Statistical analysis plan

**AI-drafted (Claude), 2026-09-03 — not yet human-reviewed.** Written before any statistic was computed
from the new runs. Binding values come from `docs/dev/PREREG_BRIEF.md`; this document says *how* they are
tested. Implementation: `training/grokverse/analysis/statistics.py` (94 checks pass).

---

## 1. Unit of analysis

**A seed pair.** Seed `s` gives one transformer run and one MLP run that share the training split
(asserted by the SHA256 `split_hash` in each manifest; a mismatch is a hard error, not a warning). Every
primary comparison is therefore **paired**, and the pairing is by seed, never by rank or by any property
measured after training.

Seeds 0–9 in the primary block. A block with fewer seeds (confound cells, two-hot control: 3 seeds) is
reported as such and never pooled with the primary block.

## 2. What is computed for every primary comparison

For a paired quantity `x` measured on both architectures:

| statistic | definition |
|---|---|
| per-seed differences | `d_s = x_transformer(s) − x_mlp(s)`, **all ten listed in the output**, never only summarized |
| location | mean and median of `d` |
| interval | **percentile bootstrap CI 95 %**, `n_boot = 10 000`, `seed = 0`, resampling the *pairs* |
| sign test | exact two-sided binomial on the signs of `d`; ties dropped and counted |
| Wilcoxon | exact two-sided signed-rank, by enumerating all `2^n` sign assignments for `n ≤ 20`; zeros dropped, average ranks for tied absolute values |
| effect size | Cohen's `d_z = mean(d) / sd(d, ddof=1)` and Cliff's delta between the two samples |
| direction | `a>b`, `a<b`, or `mixed`, with `all_positive` / `all_negative` flags |
| robust spread | median, MAD (scaled 1.4826) and IQR of `d` |
| completeness | `n_pairs`, `n_total_a`, `n_total_b`, `n_dropped` and the dropped seeds |

**Every primary comparison is also plotted with one line per seed** (`plot_paired`), so a reader sees the
individual runs and not only an interval.

## 3. The pre-specified primary comparisons

Exactly **eight** paired comparisons carry the study's claims. They are listed here so the count is fixed
in advance and cannot grow after the fact:

| # | quantity | at which measurement point |
|---|---|---|
| 1 | generalization step (with its evaluation interval, §6) | — |
| 2 | grokking gap (with its interval) | — |
| 3 | structured-neuron fraction, primary definition | crossing and final, separately |
| 4 | phase-relation resultant length `R` | crossing and final, separately |
| 5 | fraction of neurons best fit by a square or odd-harmonic model (H2) | final |
| 6 | family-fraction vs top-1 gap (H3b) | final |
| 7 | causal ablation damage, structured minus size-matched random control (H5) | final |
| 8 | end-to-end logit-fit R² of the best Fourier formula (G3) | final |

Everything else — spectral entropy, participation ratio, IPR, periodicity score, top-k concentration,
weight norms, `additivity_r2`, per-class quantities — is **secondary and descriptive**, reported with
intervals but never used to decide a hypothesis.

## 4. Multiplicity, honestly

The evidence gate (§5 of the pre-registration) and every hypothesis refutation criterion are stated in
terms of **confidence intervals and control distributions**, not p-values. P-values are reported as
descriptive companions to the eight comparisons above and are **never the sole evidence for any claim**.

Because eight primary comparisons are pre-specified, a single nominal p ≈ 0.05 among them is not treated
as evidence. Two consequences, fixed now:

* a claim rests on the **effect size, its interval, and the per-seed pattern**, not on crossing a
  threshold;
* where a reader would want a corrected number, the Holm-adjusted p-values across the eight comparisons
  are reported alongside the raw ones, explicitly labelled as a descriptive aid.

No comparison is added to the primary list after seeing a result. If an unplanned comparison turns out to
be interesting, it is reported in a clearly labelled **exploratory** section and never in the headline.

## 5. Failed and missing runs

Runs are **never silently dropped**. Every table reports `n started` and `n completed`, taken from
`results/run_manifest.json`, whose `status` field distinguishes `completed`, `failed`, `aborted` and
`running`. A run that failed is named together with its `abort_reason`.

If any run is missing, the affected comparison is reported **twice**: over the completed pairs, and — as a
sensitivity analysis — over the subset of seeds where *both* architectures completed. A pair is broken if
either side is missing; a broken pair is dropped from the paired statistic and counted in `n_dropped`.

## 6. Timing comparisons and the discrete evaluation grid

Train accuracy is evaluated every 10 steps and test accuracy every 25, so a crossing is known only to lie
in `(previous evaluated step, first crossing step]`.

The paired timing difference is therefore reported in **two forms**:

1. the difference of the reported first crossings — the point estimate;
2. the **interval-consistent bounds**: the smallest and largest difference compatible with both crossings
   lying anywhere inside their intervals, i.e. `[gen_a.prev − gen_b.first, gen_a.first − gen_b.prev]`.

A timing claim is made only if it survives the interval-consistent bounds. Given that the transformer's
crossings span 5,625–10,275 and the MLP's 8,150–10,075 across seeds, the ±25-step grid uncertainty is
small relative to the seed spread — but it is propagated rather than assumed negligible, and the
evaluation frequency is stated with every reported crossing.

**The phrase "exact transition" is not used.**

## 7. The confound matrix

Cells: {no Grokfast, Grokfast} × {`train_frac` 0.3, 0.5}, both architectures, 3 paired seeds per cell.

Analysis is a **descriptive 2 × 2 factor decomposition per architecture**, computed on paired seeds:

* main effect of Grokfast = mean over training fractions of (Grokfast − no Grokfast), per seed, with a
  bootstrap CI;
* main effect of the training fraction = the same with the roles exchanged;
* interaction = the difference of the two simple effects.

With 3 seeds per cell the intervals are wide and are reported as such. **No effect is attributed to
Grokfast or to the training fraction alone unless the decomposition separates them**, and the historical
observation that the legacy speed ratio moved from ~2.9× to ~1.3× between two settings that differed in
*both* knobs is explicitly not used as evidence about either.

## 8. The parameter-matched and two-hot controls

Each is a **separate paired comparison**, never merged with the primary one:

| control | what it can support | what it cannot |
|---|---|---|
| parameter-matched MLP (`d_mlp = 572`, +0.02 % of the transformer's count) | that a difference survives, or does not survive, equalizing the parameter count under a weight decay that acts on every parameter | it does not equalize the *shape* of the parameter budget, so it cannot rule out capacity-allocation effects |
| two-hot MLP (3 seeds) | that a difference is, or is not, attributable to the input parametrization rather than to the architecture | with 3 seeds it can only show a large effect; a null here is weak evidence |

"Same hyperparameter dimensions" and "matched parameter count" are reported as two distinct comparisons
with their own tables.

## 9. Small-sample discipline

With `n = 10`:

* no result is reported to more significant figures than the spread justifies;
* an interval that spans zero is reported as spanning zero, in words, in the same sentence as the point
  estimate;
* no claim of the form "architecture X is fundamentally faster/better" is made — only "under the
  conditions examined";
* a direction that holds in every seed is reported as such (`all_positive` / `all_negative`), because with
  `n = 10` a unanimous sign is itself informative (exact two-sided sign-test p = 1/512 ≈ 0.002) even when
  the magnitude is uncertain;
* the number of *successful* runs and the number of *started* runs are always both given.

## 10. Mapping hypotheses to statistics

| hypothesis | data | statistic | decision rule |
|---|---|---|---|
| **H1** | phase errors of structured neurons, per architecture per seed | resultant length `R` vs a permutation null (neuron identity shuffled between operand and output side), bootstrap CI on `R` | refuted for an architecture if `R` sits at or below the null's 95th percentile in the majority of seeds |
| **H2** | per-neuron best-by-AIC model | paired difference in the fraction best fit by square/odd-harmonic; median `α₃/α₁` against the ideal 1/3 | refuted if the paired difference's 95 % CI contains 0 |
| **H3b** | structured-neuron fraction under the top-1 and the family definitions | paired difference of the two architecture gaps, with the cardinality-matched top-4 control and the random-4 null | refuted if that difference's CI contains 0, or if the MLP's family-based fraction is still lower with a CI excluding 0 |
| **H4** | structure metrics at `init` and `pre_generalization` | difference exceeding three bootstrap standard errors; first checkpoint exceeding `init + 3σ` of the first two checkpoints | refuted if no metric separates before the crossing |
| **H5** | ablation damage, structured vs size-matched random control (50 seeded draws) | `z = (observed − control_mean)/control_std`, plus the paired difference in damage | refuted if the paired difference's CI contains 0 |
| **gate G1–G4** | per seed, per architecture | the numeric rules of `docs/PREREGISTRATION.md` §5 | an architecture passes if ≥ 8 of 10 seeds pass all four |

## 11. Reproducibility of the statistics themselves

Every bootstrap, permutation and random control takes an explicit seed (default 0) and echoes it in the
output JSON. Re-running `analysis/statistics.py` on the same inputs reproduces the same intervals
bit-for-bit; this is asserted in `tests/test_statistics.py`. All figures are generated from the stored
aggregate JSON, never from a recomputation, so a figure and its table cannot disagree.
