# Limitations

AI-drafted (Claude); §A 2026-09-03, §B 2026-09-04 — not yet human-reviewed. Two kinds of limitation are separated:
**design limitations**, which are fixed by the study's construction and are known before any result, and
**result limitations**, which can only be written once the analyses have run. Section B was
deliberately left empty until then; it was filled in on 2026-09-04, after the analyses.

---

## A. Design limitations (known now, not removable by more analysis)

### A1. One point in hyperparameter space

The primary comparison is `p = 113`, addition, `train_frac = 0.3`, `weight_decay = 1.0`, AdamW at
`lr = 1e-3`, full batch, no Grokfast. It is not a sweep. Manir & Rupa 2026 (arXiv:2603.25009) show that
the transformer-vs-MLP relationship *moves* with regularization and depth — and report the timing gap in
the **opposite** direction under their protocol (per-architecture optimizers, depth-4 MLP, `p = 97`,
20 % training fraction). No statement of the form "architecture X is faster/better" is licensed by this
study; only "under the conditions tested".

### A2. Ten seeds is a small sample

With 10 paired seeds, confidence intervals on timing quantities are wide — the legacy 3-seed transformer
spread was ±1196 steps. Effect sizes and intervals are reported for everything, p-values never alone, and
no spurious precision is quoted. A paired difference whose interval spans zero is reported as such.

### A3. Two architectures, one task

Only a 1-layer ReLU transformer without LayerNorm and a 2-layer shared-embedding ReLU MLP (plus a two-hot
MLP control) on modular **addition**. Nothing here transfers to deeper models, other activations, other
group operations, or LayerNorm'd transformers without new evidence. Khanh 2026 specifically finds
LayerNorm changes compression timing, and our transformer has none.

### A4. The transformer's effective operand curves are an approximation

The MLP is exactly additive in `(a, b)` up to the ReLU; the transformer is not, because attention depends
on the input. The comparable object is built with *mean* attention, and the share of the pre-activation it
fails to capture is measured per neuron as `1 − additivity_r2`
(`docs/TRANSFORMER_MECHANISM_DERIVATION.md` §4.1). Where that share is large, the effective-curve
comparison explains correspondingly less of the transformer, and this bound is carried with every
cross-architecture number derived from those curves.

### A5. Architecture-specific objects cannot be compared

`W_pos`, `W_Q/W_K/W_V/W_O`, the residual stream, the attention heads and the direct (non-MLP) logit path
exist only in the transformer. They are analysed and reported, but excluded from the headline comparison,
because the MLP has nothing to compare them against.

### A6. "Final" is a budget, not a proof of convergence

Every run stops at a fixed 25 000 steps. Whether a given metric has actually settled is *tested* per
metric and per seed (change below 5 % between the step-20 000 and step-25 000 checkpoints) and reported;
a metric that fails that test is labelled "final (budget), not converged". Khanh 2026 measures compression
lags of order 10 000 steps or more, and for our MLP the post-crossing window is only ~15 000 steps, so
censoring at the strict tolerance is a real possibility that must be reported rather than hidden.

### A7. Transitions live on a discrete evaluation grid

Train accuracy is evaluated every 10 steps and test accuracy every 25. Every crossing is therefore an
interval `(previous evaluated step, first crossing]`, and the phrase "exact transition" is not used.

### A8. The key-frequency threshold is ours, not the paper's

Nanda et al. determine key frequencies from the neuron→logit map and keep those with "nontrivial"
coefficients, but never publish the numeric cut. The 0.25-of-maximum threshold is a GROKVERSE choice,
declared in advance with sensitivity at 0.10 and 0.50.

### A9. `nanda_exact` may remain partly unreconstructable

The restricted-loss data split is not stated in the paper (the released notebook computes it on all and on
train; the TransformerLens demo on test). Where an item cannot be reconstructed from the primary source,
the implementation raises rather than guesses, and the documentation says so.

### A10. The two-hot control is ours, not Swaroop's model

It keeps our width, modulus, split and budget so that only the input parametrization differs from the
shared-embedding MLP. It therefore tests *our* question and is not a replication of arXiv:2603.23784.

### A11. Ablation is causal about *this* checkpoint, not about learning

All ablations run on unmodified trained checkpoints with no retraining. They can show that a component is
necessary or sufficient for the trained model's behaviour; they cannot show that the component *had to*
form, or how it formed.

### A12. Compute and environment

CPU-only, single-threaded per run, float32 training. Bit-exactness holds per environment: torch reductions
differ across thread counts, which over thousands of steps moves final accuracies by ~0.5 %. Thread count
is pinned and recorded so the matrix is reproducible; results from a different machine may differ in the
last digits, though transition step indices have reproduced exactly in every re-train tried.

### A13. Review is AI-only

Code, derivations, source notes and audits were written by AI agents and reviewed by *other AI agents*
with different instructions. That caught real defects — including one wrong pre-registered prediction
(`docs/LABBOOK.md` entry 15) — but it is not independent human review, and this document must not be read
as if it were. See `AI_DISCLOSURE.md`.

### A14. The pre-registration is not yet human-approved

Every threshold, definition and pass rule other than the step budget and the matrix scope is
`[AI-PROPOSED]`. `docs/HUMAN_DECISIONS.md` still reads `STATUS: NOT YET APPROVED BY HUMAN AUTHORS`.
Results computed under these settings must be reported as such until that changes.

### A15. Bounded, not exhaustive, alternative-mechanism search

If an architecture fails the evidence gate, the follow-up is a *bounded* set of tests (decodability of
`(a+b) mod p` from hidden activations, low-rank and symmetry structure, cross-seed consistency, causal
removal). The study does not promise to discover every possible alternative algorithm; a well-founded
negative result stands even if the bounded analysis finds nothing.

---

## B. Result limitations — measured, 2026-09-04

Written after the analyses, from the stored artifacts. AI-drafted, not yet human-reviewed. Each item
answers the question the design left open; none of them is a hedge added after the fact.

### B1. Which architectures passed the evidence gate, and which criterion failed where

Neither. G1, G2 and G3 hold on **10 of 10 seeds for both** architectures at the final checkpoint; **G4
fails 0/10 in both**. Within G4, `keep_structured` is *sufficient* 10/10 in both, and the failure is
entirely in the necessity conditions: `remove_structured` is not necessary in either architecture,
because the size-matched random control does **0.873** (MLP) and **0.916** (transformer) of the same
damage. `remove_key_freqs` is necessary 10/10 for the MLP and 0/10 for the transformer — but only
because the gate was wired to `remove_key_subspace_from_residual`; the embedding-level ablation gives
10/10 (see B9 and **D5**). Source: `training/results/decision_tree_final.json`.

### B2. Which decision-tree branch was taken, and what it permits

**`neither_passes`** at the final checkpoint; **`undetermined`** at the memorization crossing, because
G3's random-frequency control degenerates there on all 10 MLP seeds. The branch forbids every mechanism
reading, including the study's own conditional main claim. It permits: reporting the measured structure
as structure, reporting the gate failure, and running the bounded alternative-mechanism analysis —
which was done for both architectures.

### B3. Convergence at the budget, per metric and per architecture

A6 promised this test; here it is. Relative change between the step-20,000 and step-25,000 checkpoints,
computed from the stored trajectories; "settled" means below 5 %.

| metric | transformer | MLP | note |
|---|---|---|---|
| `structured_fraction_of_live` | **10/10 settled** (median 0.41 %) | **2/10 settled** (median 7.09 %, max 9.00 %) | the MLP is **not settled at the budget** |
| `embedding_top8_concentration` | 10/10 (0.59 %) | 8/10 (2.80 %, max 8.15 %) | |
| `phase_relation_R` | 10/10 (0.37 %) | 10/10 (0.03 %) | |
| `median_family_fraction` | 10/10 (0.15 %) | 10/10 (0.09 %) | |
| `logit_key_subspace_share` | 9/10 (1.25 %) | 10/10 (0.40 %) | |
| `median_odd_minus_even_u_a` | 0/10 | 0/10 | the relative test is meaningless here: the value sits at ~1e-3, so **maximum absolute change is 0.003**. Reported as unsettled *in relative terms only*. |
| `fraction_best_aic_sinusoid` | **0/10** (median 17.1 %) | 2/10 (15.3 %) | **largest absolute movement of any metric here: 0.255** in the transformer. Genuinely unsettled. |
| `fraction_best_aic_odd_harmonics` | 1/10 (27.6 %) | 3/10 (8.2 %) | max absolute change 0.255 (txf), 0.103 (MLP). Genuinely unsettled. |
| `fraction_best_aic_square` | 9/10 (0.0 %) | 1/10 (41.1 %) | asymmetric for a reason: the transformer's share has decayed to ≈0 and stays there (max absolute change 0.005), while the MLP's is small but still moving — the 41 % is relative to a small value, and the **absolute** change is only 0.022. |

The waveform rows were added on 2026-09-04 after the defect of labbook 101 was fixed; before that
these three series were constant zero and no convergence statement about them was possible.

**This is the most consequential entry in this document.** `structured_fraction_of_live` is the metric
behind G1 and behind the headline architecture gap (+0.085), and for the MLP it is **still rising at
the budget** in 8 of 10 seeds — from 0.861 at step 20,000 to 0.881 at 25,000 in seed 0, having climbed
from 0.223 at step 8,000. The transformer's has flattened (0.959 to 0.965). A longer budget would
therefore be expected to **shrink** the gap this study reports, and possibly to change G1's margin for
the MLP. This is precisely the censoring Khanh 2026 warns about (A6), now measured rather than
anticipated. No claim in `RESULTS.md` may be read as a converged-state comparison; every one of them is
a comparison **at a fixed 25,000-step budget**.

The waveform shares are unsettled too, and by more: the transformer's sinusoid share moves **0.255 in
absolute terms** between steps 20,000 and 25,000. The §8.1 trajectory result (the waveform composition
shifting during the plateau) is unaffected, because it is measured from initialization to the last
checkpoint *before* generalization — an interval that ends long before the budget. What is affected is
any reading of the *final* waveform split as an endpoint: it is not one.

**Measured at 100,000 steps (2026-09-08, external S1 block; `RESULTS.md` §16).** The external reviewer's
20-run block at 4× the budget, on the frozen configuration, gives median `structured_fraction_of_live`
**1.0000** for both architectures (25k: 0.9824 / 0.8848), gap **0.0000** (25k: +0.0977 as a difference of
medians; +0.0850 as the paired median of `RESULTS.md` §7); MLP settled **10/10** between steps 90,000 and
100,000, transformer 9/10 (seed 2 at 8.24 %, falling from 0.877 to 0.805). So the direction predicted above is
what happened: the gap this study reports is a budget-fixed gap that vanishes at convergence, and the sentence
"no claim may be read as a converged-state comparison" stays — now with the converged state measured for this
one metric. The gate does not change (G4 0/10 in both at 100k, because the structured set is then ~100 % of
the live neurons; D6). The block was run externally, its integrity and reports were verified here, and it has
**not** been reproduced on the authors' hardware; see `docs/HUMAN_DECISIONS.md` D9.

### B4. The measured `additivity_r2` distribution for the transformer

Median **0.922**, range 0.871–0.961 across the 10 seeds (final checkpoint). The mean-attention
effective curves therefore capture roughly 92 % of the transformer's pre-activation variance, and the
remaining ~8 % is not represented in any effective-curve comparison. The direct (non-MLP) logit path
carries a **0.002** share, so the approximation error is in the attention, not in a bypassed path. Two
independent checks bound this: the exact forward decomposition reproduces the model's logits to 2.8e-13
against a float64 reference, and fixing attention to its mean costs **0.336** test accuracy — so the
input-dependent part of attention is real, and is exactly what the effective curves omit.

### B5. Whether the confound matrix separated Grokfast from the training fraction

Yes, in direction and magnitude, but with no interval. With 3 paired seeds per cell the 2 × 2
decomposition is computed but **no bootstrap CI is reported**, deliberately. The training fraction
moves the generalization step by −6,367 (transformer) and −8,217 (MLP) steps; Grokfast by −350 and
+892; the interaction by +883 and −933. The Grokfast main effect is the same order as the interaction,
and for the transformer indistinguishable from zero at this resolution. That is enough to retire the
earlier repository claim that attributed a change of speed ratio to Grokfast, and **not** enough to
make any positive claim about Grokfast.

### B6. Whether the parameter-matched and two-hot controls changed a conclusion

**Parameter-matched: no.** At 226,217 vs 226,176 parameters (0.02 %), every structure difference
survives with the same sign and an interval excluding zero (structured fraction +0.0969
[+0.0863, +0.1240]; phase `R` −0.0076; square/harmonic fraction −0.2230; generalization step −2,212).

**Two-hot: yes.** The two-hot MLP is **more** square-wave-like than the shared-embedding MLP by
**+0.1836** [+0.1133, +0.2109] — roughly half the size of the transformer-vs-MLP waveform difference
itself (−0.324). The waveform result is therefore at least partly a property of the input
parametrization rather than of the architecture, and `RESULTS.md` says so. With 3 seeds this control
can only detect a large effect; it found one.

### B7. Ablations that did not damage the model

These are findings, and are listed rather than omitted. `keep_structured` / `keep_structured_neurons`:
drop 0.000 in both — but the size-matched random control also drops 0.000, so the sufficiency is not
informative about the *structured* set specifically. `keep_key_freqs_in_embedding` (transformer): drop
0.004 while its control drops 0.990. `restricted_circuit_only` (transformer): drop **−0.000** — test
accuracy 1.0000, marginally above the unablated model. `replace_with_sinusoid_fit` and
`replace_with_square_fit` (MLP): drop 0.000 each, test accuracy **1.0000** — replacing every neuron's
curves with a fitted waveform costs nothing. `remove_key_subspace_from_residual` (transformer): drop
0.235, far below the 0.5 necessity threshold. Singular-direction ablation on the **MLP**: drop **0.0000
at every rank up to 16**, exceeding no control on any seed.

### B8. Hypotheses refuted, with the number that refuted them

**H3**, the study's proposed primary contribution, by its own criterion 2: the architecture gap under
the family definition is **+0.0850**, CI95 [+0.0600, +0.1004], unanimous across 10 seeds. The family
definition closes **+0.0049** of a +0.0898 gap — **5.4 %**. The postulated artifact is real and
measured at +0.1893 on synthetic populations differing only in waveform; it is simply an order of
magnitude too small to be the explanation.

### B9. Runs that failed or were excluded

**None.** 51 of 51 completed; no run was dropped, so no sensitivity analysis for an excluded run is
required. Where a *value* is missing it is `null` with a stated reason: transformer seed 5's structured
set is all 512 neurons, so no size-matched control exists for it; `error_jaccard` is undefined in 4 of
10 function-agreement pairs because neither model errs; G3 is not evaluable at the crossing point for
the MLP; two H4 metrics never produce a defined onset. Denominators are always stated.

### B10. Analyses that changed after the code freeze

Five defects were found and fixed after the freeze at `0b55e1d`; each is in the post-freeze table of
`docs/PREREGISTRATION.md` §12 with its labbook entry. Three of them were **silent**: they produced
well-formed output containing no information, rather than an error.

| defect | labbook | what it would have changed |
|---|---|---|
| `key_frequencies` / `progress_measures` rejected `seed` | 56–57 | 60 driver calls failed loudly — caught immediately |
| `wave_fitting` `IndexError`, and a wrong architecture declaration | 58, 60 | 24 calls failed loudly |
| `aggregate._wave_fitting` guessed flat keys | 58 | **silent**: the H2 figure would have been blank |
| `aggregate._progress_measures` read one level too shallow | 99 | **silent**: every restricted/excluded column was `null` |
| `structure_over_time` compared a model **index** to a model **name** | 101 | **silent, and it produced 0.0 rather than `null`** — two H4 waveform trajectories were flat zero for the whole study |

The last is the reason this section exists in the form it does. A defect that returns `null` announces
itself as an absence; one that returns `0.0` does not. The countermeasure adopted is an invariant the
empty case cannot satisfy (the four waveform shares must sum to exactly 1) plus a cross-module
agreement check at a shared checkpoint. **It is likelier than not that further defects of this class
remain in extractors that have neither.**

### B11. What this study measured that it did not set out to measure

Two things, both reported because they bound what the other numbers mean.

- **Cross-seed CKA does not identify same-architecture seeds as similar** (median 0.0017 transformer,
  0.0973 MLP). Representation similarity is therefore unusable here as evidence about mechanism
  sameness *or* difference, in either direction.
- **The two architectures agree on 99.98 % of inputs while their logits correlate at 0.083** and their
  top-2 predictions agree 0.6 % of the time. Behavioural agreement and representational agreement come
  apart completely on this task, which is worth stating before anyone reads §4 of `RESULTS.md` as a
  similarity result.
