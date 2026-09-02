# Pre-registration brief — decisions fixed on 2026-09-02, before any new run

This brief is the input to `docs/PREREGISTRATION.md`, `docs/STATISTICAL_ANALYSIS_PLAN.md`,
`docs/CAUSAL_ABLATION_PLAN.md` and `docs/METHODS.md`. Every value below is **[AI-PROPOSED — pending
human approval]** unless marked **[HUMAN 2026-09-02]** (answers given by the human author in this
session). Nothing here was tuned on any result of the new runs; the only data seen are the 16 legacy runs
(all early-stopped at their crossing) and the literature notes under `docs/sources/`.

## Research question (verbatim, never to be rephrased)

> Both models generalize on modular addition. **Do they use a Fourier-based phase-addition mechanism,
> and if so, do they represent and compute it differently?**

Until Phase 6 has been measured, the only permitted wording is "both architectures generalize on the
same task".

## Positioning

* Fourier structure, phase addition, and square-wave-like ReLU-MLP weights are **not** new. H1 and the MLP
  half of H2 are a **replication** in the GROKVERSE architectures (1-layer ReLU transformer without
  LayerNorm; 2-layer ReLU MLP with a *shared, learned* embedding and operand concatenation) and training
  setting.
* Extensions: the same mechanism tests applied to **both** architectures; the transformer side of the
  sinusoid-vs-harmonics comparison; the shared-embedding concatenation MLP (vs. Swaroop's direct two-hot
  MLP); causal necessity **and** sufficiency; and the metric-validity question H3.
* Manir & Rupa (2603.25009) already report the Transformer-vs-MLP timing gap under matched conditions and
  its sensitivity to hyperparameters; a timing gap or a concentration gap alone is **not** a contribution.

## Setting **[HUMAN 2026-09-02]**

| item | value |
|---|---|
| task | `(a + b) mod 113`, tokens `[a, b, =]`, full-batch |
| split | `train_frac = 0.3`, permutation keyed on the seed; paired seeds share the split (asserted by `split_hash`) |
| optimizer | AdamW, `lr = 1e-3`, `betas = (0.9, 0.98)`, `weight_decay = 1.0`, no Grokfast |
| budget | **fixed `steps = 25000` for every primary run**, no early stop |
| seeds | `0..9`, identical for transformer and MLP; analysis paired by seed |
| threads | 1 (pinned) |
| models | transformer: `d_model 128, 4 heads × 32, d_mlp 512, n_ctx 3, no LayerNorm`; MLP: `d_model 128, d_mlp 512`, shared `W_E [113, 128]`, concat → ReLU → `W_out` |
| evaluation | train acc/loss every 10 steps, test every 25 steps; log-schedule curves + embeddings kept |
| thresholds | primary: memorization train acc ≥ 0.99, generalization test acc ≥ 0.95; sensitivity: (0.98, 0.90) and (1.00, 0.99); every crossing reported as `(previous evaluated step, first crossing step]` |
| checkpoints | grid `{0, 500, 1000, 2000, …, 10000, 12000, …, 20000, 22500, 25000}` + event checkpoints at the primary crossings; roles by the rule in `RUN_FORMAT_V2.md` §3 |
| measurement points | every structure metric is reported at **two declared points**: the generalization-crossing checkpoint (event-matched) and the final step-25000 checkpoint (budget-matched). They are never mixed in one comparison. A metric is called "converged at budget" only if its change between the step-20000 and step-25000 checkpoints is below 5 % of its value (Khanh 2607.06639); otherwise "final (budget), not converged" |

## Controls **[HUMAN 2026-09-02: full matrix]**

| block | design | seeds |
|---|---|---|
| confound matrix | {no Grokfast, Grokfast(α=0.98, λ=2)} × {`train_frac` 0.3, 0.5}, both architectures, same 25k budget | 0–2 per cell (the no-Grokfast/0.3 cell is the primary block) |
| parameter-matched | MLP with `d_mlp = 572` (226 217 params, +0.02 % vs the transformer's 226 176) | 0–9 |
| input parametrization | two-hot MLP (`concat(onehot(a), onehot(b)) → ReLU(512) → 113`), our variant of Swaroop's setup, literature control only | 0–2 |

No effect is attributed to Grokfast or to `train_frac` alone until the matrix separates them; no
difference is called an architecture effect if the two-hot control shows it is an input-parametrization
effect; "same hyperparameter dimensions" and "similar parameter count" are two separately reported
comparisons.

## Hypotheses, nulls, refutation criteria

**H1 — Fourier phase-addition mechanism (replication, per architecture).** Each architecture, tested on its
own, shows (1) periodic internal structure, (2) the phase-addition relation, (3) an end-to-end logit fit,
(4) causal necessity and sufficiency. These four are the **evidence gate** (below). H1₀: no criterion
exceeds its control. Refuted for an architecture if any gate criterion fails in the majority of seeds.

**H2 — Different waveform (MLP half: replication of Swaroop; transformer half: extension).** The MLP's
effective operand curves are better described by odd-harmonic / square-wave models than by a single
sinusoid for a larger fraction of neurons than the transformer's effective curves. Statistic: paired
per-seed difference in `fraction_neurons_best_aic ∈ {square, odd_harmonics}` (transformer − MLP) and in the
median odd-harmonic amplitude ratio `α₃/α₁` (ideal square wave: 1/3). H2₀: paired difference 0. Refuted if
the 95 % bootstrap CI of the paired difference contains 0.

**H3 — Metric validity (primary contribution).** *Does top-k Fourier concentration misclassify a
structured harmonic representation as less structured?* Sub-claims, each with a pre-specified test:

* H3a (mechanism of the artifact, synthetic): on idealized models built from the fitted per-neuron
  parameters (frequencies, phases, amplitudes) with (i) pure sinusoids and (ii) pure discrete square waves,
  top-8 concentration of the *same* frequency content is lower for (ii); reported as the metric's
  waveform sensitivity with the exact numbers.
* H3b (structure measured harmonic-aware): per neuron, the **family fraction** (share of a curve's power in
  its own dominant frequency k plus its aliased odd harmonics 3k, 5k, 7k) versus the **top-1 fraction**.
  Controls: the cardinality-matched **top-4** share of the same curve (an upper bound by construction,
  reported as the ratio family/top-4) and the **random-4** null (50 draws). H3b predicts: family/top-4 ≈ 1
  for MLP neurons (the extra power *is* the harmonics), and the architecture gap in the **structured-neuron
  fraction** closes under the family-based definition but not under the top-1 definition.
* H3c (the legacy number): the top-8 concentration of `W_E` is reported at both measurement points with
  its cardinality-matched and random controls, and the harmonic-family concentration of `W_E` alongside —
  descriptively; `W_E` alone decides nothing for the MLP.
* H3₀: the architecture gap in structured-neuron fraction is the same under top-1 and family definitions.
  Refuted if the paired difference (gap_top1 − gap_family) has a 95 % CI containing 0, **or** if the
  MLP's family-based structured fraction is still lower than the transformer's with a CI excluding 0 (then
  the deficit is real structure loss, not an artifact).
* **Dependence on the gate:** H3 does not presuppose that either architecture uses a Fourier circuit.
  Harmonic-family concentration may be interpreted as the same Fourier principle in a different form
  only for an architecture that passes the evidence gate.
* Permitted conclusion if supported: *"The results are consistent with the same Fourier principle being
  expressed through different internal representations."* Banned: "the same circuit in different bases".

**H4 — Structure precedes generalization.** Under the primary structured-neuron definition, the structured
fraction and the phase-relation R at the `pre_generalization` checkpoint exceed their `init` values by
more than 3 bootstrap standard errors, in both architectures. H4₀: no metric moves before the crossing.

**H5 — Causal relevance.** Removing the structured components damages test accuracy far more than a
size-matched random control (z ≥ 3, absolute drop ≥ 0.5 vs control drop < 0.1) and keeping only the
structured components retains test accuracy ≥ 0.9 — in both architectures. H5₀: structured and random
ablations do equal damage. Ablations that do **not** damage the model are reported as findings.

## Structured neuron — primary definition (fixed now)

A hidden neuron is *structured* iff
1. it is alive (max activation over the grid > 0),
2. `u_a` and `u_b` share the dominant frequency k,
3. the family fraction (k + aliased odd harmonics ≤ 7) of `u_a` **and** of `u_b` is ≥ **0.50**,
4. the output curve's dominant frequency is k.

Sensitivity (mandatory, always reported next to the primary): family threshold 0.30 and 0.70; the top-1
variant (dominant fraction ≥ 0.50, i.e. RESEARCH_SPEC's proposal); Doshi-style IPR and Swaroop-style
periodicity definitions with the thresholds stated in `docs/sources/` (or `[ours]` if the source has none).
The same definition is applied to the transformer's effective operand curves (INTERFACES §6); where the
transformer's effective curves are not additive (`additivity_r2 < 0.9`) this is reported as a fairness
limitation and the transformer's `hidden` activations are analysed directly as well.

## Key-frequency rule (primary) — conditional, fixed now

Primary rule: **`neuron_clusters`** (frequencies that are the dominant frequency of ≥ 5 structured neurons
and whose neurons carry ≥ 2 % of the total hidden-activation variance), because Nanda et al. read their key
frequencies off neuron clusters *(to be confirmed in `docs/sources/nanda2023_progress_measures.md`)*.
**If the source note establishes a different published rule, that rule (`nanda`) becomes primary and
`neuron_clusters` secondary — this substitution is declared here, before any run.** `embedding_threshold`
(uncapped 90 %) and `logit_sum_directions` are always reported; `embedding_top8` only as the legacy number.
Jaccard overlap between rules is reported.

## Evidence gate — pass rules per architecture (fixed now)

Evaluated per seed at the **final** checkpoint (and reported at the crossing checkpoint):

| criterion | pass rule |
|---|---|
| G1 periodic structure | structured-neuron fraction (primary definition) ≥ 0.25 of live neurons **and** median family fraction of `u_a`, `u_b` ≥ 0.50 |
| G2 phase addition | resultant length R of `φ_out − (φ_a + φ_b)` over structured neurons > null 95th percentile **and** R ≥ 0.5 |
| G3 end-to-end fit | `sparse_sinusoid` **or** `odd_harmonics` formula (key set from the primary rule): argmax accuracy of the fitted logits ≥ 0.90 on the test cells, R² above the random-frequency control's 95th percentile; the `control_difference` formula must not reach 0.5 of that R² |
| G4 causal | `remove_structured` and `remove_key_freqs` each drop test accuracy by ≥ 0.5 with the random control dropping < 0.1 (z ≥ 3); `keep_structured` retains test accuracy ≥ 0.9 |

An architecture **passes the gate** if ≥ 8 of 10 seeds pass all four criteria. Sensitivity: the gate is
re-evaluated with the 0.30/0.70 definitions and reported.

## Decision tree (fixed now)

* both pass → investigate H3 and compare how the Fourier principle is represented and computed;
* only one passes → bounded investigation of the other architecture's representation
  (`bounded_alternative.py`: `(a+b) mod p` decodability of hidden activations by linear / 1-hidden-layer
  probes vs shuffled labels; low-rank / symmetry structure of activation and logit tensors; cross-seed
  consistency (CKA); causal removal of the found structure vs random directions);
* neither passes → the Fourier evidence tested does not identify the learned mechanisms; the validity of
  the existing metrics is investigated; a well-founded negative result is the deliverable.

## Statistics (fixed now)

Paired by seed: differences in generalization step (with intervals), grokking gap, every structure metric,
every ablation damage. Percentile bootstrap CI (10 000 resamples, seed 0), exact sign test and exact
Wilcoxon signed-rank test, Cohen's d_z and Cliff's delta, median/MAD/IQR, all seeds plotted. p-values never
alone. n started / n completed always reported. No "fundamentally faster/better" claims from one
hyperparameter region; the honest magnitude of the timing gap comes with its interval.

## Freeze and pilot

* Pilot = seed 0 pair, used **only** to validate that the pipeline runs end to end and produces valid
  artifacts; no threshold is tuned on it.
* Analysis code is frozen at a named git commit after the pilot analyses execute; any later change is
  logged in the labbook with its reason, and every affected analysis is re-run on all seeds.
* The explorer stays frozen until the analyses are frozen and human-reviewed.

## Reporting rules

Every claim: observation → quantitative evidence → alternative explanation → causal test → limitation →
permissible conclusion. Graded language only. Banned without the gate: "proves", "clearly shows",
"Transformers fundamentally learn…", "MLPs are worse…", "the same circuit", "the same function".
