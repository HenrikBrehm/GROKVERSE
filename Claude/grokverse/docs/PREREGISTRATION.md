# Pre-registration — mechanistic comparison of a Transformer and an MLP on modular addition

**AI-drafted (Claude), 2026-09-02/03 — not yet human-reviewed. The human authors must revise before
submission.** Every value here other than those marked `[HUMAN 2026-09-02]` is `[AI-PROPOSED]` and is
collected for approval in `docs/HUMAN_DECISIONS.md`, which currently reads
`STATUS: NOT YET APPROVED BY HUMAN AUTHORS`.

Binding source: `docs/dev/PREREG_BRIEF.md` and its two addenda. Where this document and the brief
disagree, **the brief wins** and this document is wrong and must be fixed — it is an expansion, never a
revision.

---

## 0. How to read this document, and its honest status

This is a **plan**. At the time of writing, the training runs of the primary block have been executed but
**no structure metric, no comparison and no statistic has been computed from them**. Tags used below:

| tag | meaning |
|---|---|
| `[HUMAN 2026-09-02]` | decided by the human author, not up for approval here |
| `[AI-PROPOSED]` | proposed by the AI, pending human approval |
| `[VERIFIED]` | checked numerically or against a primary source in this repository |
| `[NOT FOUND IN SOURCE]` | the primary source does not state it; our choice, declared as ours |

### 0.1 Order deviation, disclosed

The master prompt's working order puts the pre-registration document before the full runs. In this project
the *values* were fixed first, in the committed `docs/dev/PREREG_BRIEF.md`, and the runs were launched
against them; this document expands that brief and was written while the runs executed. The expansion is
forbidden from changing any value, and no run output had been analysed when it was written. The labbook
records this (`docs/LABBOOK.md` entry 8). Anyone auditing the study should compare this file against the
brief's commit `d53cf52` to confirm no value moved.

---

## 1. Research question and positioning

### 1.1 The question, verbatim and never to be rephrased

> Both models generalize on modular addition. **Do they use a Fourier-based phase-addition mechanism, and
> if so, do they represent and compute it differently?**

This deliberately leaves open that one or both architectures use a different mechanism. It is never to be
restated in a form that presupposes a shared Fourier algorithm.

Until the full-domain function comparison of §6.2 has been measured, the only permitted wording is
**"both architectures generalize on the same task"** — never "both learn the same function".

### 1.2 What is replication and what is extension

Fourier structure, phase addition, and square-wave-like ReLU-MLP weights are **not new**.

| claim | already established by | our status |
|---|---|---|
| A grokked network on modular addition uses Fourier features and the angle-addition identity | Nanda et al. 2023, arXiv:2301.05217 | replication |
| ReLU MLPs show a `φ_out = φ_a + φ_b` phase-sum relation | Swaroop 2026, arXiv:2603.23784 §1, §3.1 — who attributes it in turn to Nanda et al. 2023 and Gromov 2023 | replication |
| ReLU MLP input weights are near-binary square waves | Swaroop 2026 §3 (two-hot input, 256 hidden, `p = 97`) | replication, in a different architecture |
| A grokked transformer's embedding is more Fourier-concentrated than an MLP's | Manir & Rupa 2026, arXiv:2603.25009 §5.7, Table 8 (98.5 % vs ≈75 % top-5, `p = 97`, one seed, DC included) | replication of the direction; ours is the first multi-seed estimate |
| The Transformer-vs-MLP timing gap is protocol-dependent | Manir & Rupa 2026, abstract and §5.5 — with the **opposite** direction under their protocol | not our contribution; we must agree with them, not claim novelty |
| Concatenation MLPs implement the same family of mechanisms; a trainable embedding reduces the number of learned frequencies | McCracken et al., NeurIPS 2025, arXiv:2505.18266 §3, §4.4 | context; our MLP is their "trainable embedding, 1 hidden layer" cell |
| Metrics read at the grokking transition overstate the converged value | Khanh 2026, arXiv:2607.06639 | method we adopt, not a finding of ours |

**A concentration gap or a timing gap alone is therefore not a contribution of this study.**

What is extended, if the evidence supports it:

1. the same mechanism battery applied to **both** architectures, including the transformer side of the
   sinusoid-versus-harmonics comparison, which no consulted source runs;
2. the **shared-embedding concatenation** MLP rather than a direct two-hot MLP, with a two-hot control to
   separate architecture from input parametrization;
3. **causal necessity and sufficiency** with size-matched random controls, in both architectures;
4. **H3** — whether top-k Fourier concentration misclassifies a harmonic representation as less
   structured.

---

## 2. Setting, controls, and measurement points

### 2.1 Primary setting `[HUMAN 2026-09-02]`

| item | value |
|---|---|
| task | `(a + b) mod 113`, tokens `[a, b, =]`, full batch |
| split | `train_frac = 0.3` from a seeded permutation; paired seeds share the split, asserted by a SHA256 split hash |
| optimizer | AdamW, `lr = 1e-3`, `betas = (0.9, 0.98)`, `weight_decay = 1.0`, no Grokfast |
| budget | **fixed `steps = 25000` for every run, no early stop** |
| seeds | `0..9`, identical for both architectures; analysis paired by seed |
| threads | 1, pinned; the matrix is parallelized across processes |
| transformer | `d_model 128`, 4 heads × 32, `d_mlp 512`, `n_ctx 3`, **no LayerNorm**; 226,176 parameters |
| MLP | shared `W_E [113, 128]`, operand concatenation, `d_mlp 512`; 204,017 parameters |
| evaluation | train accuracy and loss every 10 steps, test every 25 |
| checkpoints | 21 per run: a pre-specified grid plus event checkpoints at the primary crossings |

### 2.2 Thresholds and transition reporting

Primary: memorization at train accuracy ≥ 0.99, generalization at test accuracy ≥ 0.95.
Sensitivity, always reported alongside: (0.98, 0.90) and (1.00, 0.99).

Every crossing is reported as the interval `(previous evaluated step, first crossing step]` together with
its evaluation frequency. The grokking gap is reported as `[gen.prev − mem.first, gen.first − mem.prev]`.
**The phrase "exact transition" is not used anywhere in this study.**

### 2.3 Measurement points `[AI-PROPOSED]`

Every structure metric is reported at **two declared points**, and they are never mixed in one comparison:

* **crossing** — the generalization event checkpoint (event-matched across architectures);
* **final** — the step-25 000 checkpoint (budget-matched).

A metric may be called **converged at budget** only if it changes by less than 5 % between the step-20 000
and step-25 000 checkpoints; otherwise it is reported as "final (budget), not converged". This follows
Khanh 2026, which measures 3–5× (MLP) and 1.3–1.5× (transformer) overstatement for metrics read at the
transition, with compression lags of order 10⁴ steps. Our MLP's post-crossing window is only ~15 000
steps, so censoring at a strict tolerance is a real possibility and must be reported, not hidden.

### 2.4 Controls `[HUMAN 2026-09-02: run the full matrix]`

| block | design | seeds |
|---|---|---|
| confound matrix | {no Grokfast, Grokfast α=0.98 λ=2} × {`train_frac` 0.3, 0.5}, both architectures, same budget | 0–2 per cell; the (no Grokfast, 0.3) cell is the primary block |
| parameter-matched | MLP with `d_mlp = 572` → 226,217 parameters, +0.02 % of the transformer | 0–9 |
| input parametrization | two-hot MLP, `concat(onehot(a), onehot(b)) → ReLU(512) → 113` | 0–2 |

No effect is attributed to Grokfast or to the training fraction alone until the matrix separates them. No
difference is called an architecture effect if the two-hot control shows it is an input-parametrization
effect. "Same hyperparameter dimensions" and "matched parameter count" are two separately reported
comparisons, never merged.

The two-hot control keeps our width, modulus, split and budget so that it differs from the
shared-embedding MLP in the **input parametrization only**. It is therefore *our* two-hot variant of
Swaroop's setup (whose model is 194 → 256 → 97 with a stratified split and early stopping), never a
replication of that paper's model.

---

## 3. Hypotheses

Each has a prediction, a null, a pre-specified statistic, a refutation criterion, and a statement of what
is reported if it is refuted. **We expect some of these to fail. A refuted hypothesis is a result.**

### H1 — a Fourier phase-addition mechanism, tested per architecture (replication)

**Prediction.** For structured neurons, the dominant frequency of the `a`-curve, the `b`-curve and the
output curve agree, and the phases satisfy `φ_out ≈ φ_a + φ_b (mod 2π)`.

**Why this and not something else.** In the MLP the pre-activation is exactly `u_a(a) + u_b(b) + β`, so
the ReLU is provably the only possible source of the product the angle-addition identity needs.
Rectifying two same-frequency sinusoids produces an `(a+b)` term of amplitude `8/(3π²) ≈ 0.2702` and phase
`φ_a + φ_b` `[VERIFIED numerically: 0.2698, phase error 2.1e-4]`
(`docs/MLP_MECHANISM_DERIVATION.md` §4).

**Statistic.** Mean resultant length `R` of the circular error `φ_out − (φ_a + φ_b)` over structured
neurons, with a bootstrap CI, against a **permutation null** that shuffles neuron identity between the
operand side and the output side, leaving both marginal phase distributions intact.

**H1₀.** The phase errors are indistinguishable from that null.

**Refuted for an architecture if** `R` lies inside the null's bulk (at or below its 95th percentile) in
the majority of seeds.

**Correction recorded, 2026-09-03.** An earlier version of this prediction also asserted that a single
circuit neuron shows much more `(a+b)` than `(a−b)` structure in its own activation map. That is **false**:
`|cos s|·|cos t|` is symmetric in `s = (u+v)/2` and `t = (u−v)/2`, so the `(a−b)` term has the *same*
amplitude and phase `φ_a − φ_b` `[VERIFIED: 0.2698 vs 0.2705]`. Addition is selected by the **population
together with the readout** — with `φ_out = φ_a + φ_b` the `(a+b)` contributions add coherently across
neurons while the `(a−b)` ones cancel `[VERIFIED on 400 synthetic neurons: R² 0.938 versus 0.003]`.
Consequently the sum-over-difference contrast is tested on the **logits** (§3 H3 and §6.5's
`control_difference`) and on the population, never as a per-neuron pass criterion; per-neuron sum and
difference shares are reported descriptively. No result had been computed under the wrong version
(`docs/LABBOOK.md` entry 15).

### H2 — different waveform (MLP half replication, transformer half extension)

**Prediction.** The MLP's effective operand curves are better described by odd-harmonic or square-wave
models than by a single sinusoid, for a larger fraction of neurons than the transformer's.

**Statistic.** The paired per-seed difference (transformer − MLP) in
`fraction_neurons_best_aic ∈ {square, odd_harmonics}`, and the median fitted amplitude ratio `α₃/α₁`
(an ideal square wave gives 1/3 `[VERIFIED]`).

**H2₀.** The paired difference is zero.

**Refuted if** the 95 % bootstrap CI of the paired difference contains 0.

**Note on provenance.** The odd-harmonic and `1/j` facts are textbook Fourier series, derived in
`docs/MLP_MECHANISM_DERIVATION.md` §5 and verified at `p = 113`. Swaroop's paper never mentions harmonics;
attributing them to it would be wrong.

### H3 — metric validity (**the proposed primary contribution**)

> **Does top-k Fourier concentration misclassify a structured harmonic representation as less structured?**

A square wave at fundamental `k` puts its power at the odd harmonics `3k, 5k, 7k`, which in `Z_113` alias
to scattered indices. A metric that keeps the top-8 *individual* frequencies therefore scores a perfectly
clean square-wave circuit as unstructured. That is the mechanism H3 tests.

**H3a — the artifact, on synthetic models.** Build idealized populations from the fitted per-neuron
parameters with (i) pure sinusoids and (ii) discrete square waves at the *same* frequencies, phases and
amplitudes. Prediction: top-8 concentration is lower for (ii) although the frequency content is identical.
Reported as the metric's waveform sensitivity, with the numbers.

**H3b — structure measured harmonic-aware.** Per neuron, the **family fraction** (its own dominant
frequency plus its aliased odd harmonics up to 7) versus the **top-1 fraction**. Controls: the
cardinality-matched **top-4** share of the same curve, and a **random-4** null over 50 seeded draws.
Prediction: the architecture gap in the *structured-neuron fraction* closes under the family-based
definition but not under the top-1 definition.

**H3c — the legacy number.** Top-8 concentration of `W_E` at both measurement points with its
cardinality-matched and random controls, plus the harmonic-family concentration, reported
**descriptively**. `W_E` alone decides nothing for the MLP, which reads it through two halves of `W_in`.

**H3₀.** The architecture gap in structured-neuron fraction is the same under the top-1 and the
family definitions.

**Refuted if** the paired difference (gap under top-1 minus gap under family) has a 95 % CI containing 0,
**or** if the MLP's family-based structured fraction is still lower than the transformer's with a CI
excluding 0 — in which case the deficit is a real loss of structure, not a measurement artifact.

**A criterion that is forbidden.** "The harmonic family beats the matched top-m concentration" is
**unsatisfiable by construction**: top-m is the argmax over sets of size m, so the family can never
exceed it. It is pinned as an inequality in `test_core.py` and must never be used as support for H3. The
discriminating statistic is the odd-versus-even harmonic **shape**, not the concentration.

**Dependence on the gate (verbatim from the master prompt §12).**

> **Dependence of H3 on the mechanism tests:** H3 does not presuppose that either architecture uses a
> Fourier circuit. Each architecture is first tested independently for periodic internal structure, the
> predicted phase relationship, end-to-end logit fit, and causal relevance. Harmonic-family concentration
> may be interpreted as the same Fourier principle in a different form only for an architecture that
> passes these tests.

**Permitted conclusion if supported:** *"The results are consistent with the same Fourier principle being
expressed through different internal representations."*
**Banned:** "the two architectures use the same circuit in different bases."

**Pre-committed consequence.** If H3 is supported, the current README and `RESULTS.md` claim that the
transformer "learns a sparser Fourier circuit" must be retracted and rewritten. We commit to that in
advance.

### H4 — structure precedes generalization

**Prediction.** Under the primary structured-neuron definition, the structured fraction and the phase
relation `R` at the `pre_generalization` checkpoint exceed their values at initialization by more than
three bootstrap standard errors, in both architectures.

**H4₀.** No structure metric separates from its initialization value before the generalization crossing.

**Refuted if** no metric does so.

### H5 — causal relevance

**Prediction.** Removing the identified structured components damages the model far more than a
size-matched random control: test-accuracy drop ≥ 0.5 with the control below 0.1 and `z ≥ 3`; and keeping
only the structured components retains test accuracy ≥ 0.9.

**H5₀.** Structured and random ablations do equal damage.

**Refuted if** the paired difference in damage has a CI containing 0. **Ablations that fail to damage the
model are reported as findings about H5, not omitted.** This is the hypothesis most likely to fail.

---

## 4. Metrics

### 4.1 Primary metrics

| metric | definition | applies to |
|---|---|---|
| top-k concentration | fraction of Fourier power in the k strongest individual frequencies | both, every weight object |
| family fraction | share of a curve's power in its own dominant frequency plus aliased odd harmonics ≤ 7, always next to the cardinality-matched top-m control and the random-set null | both |
| harmonic shape | odd-harmonic (j = 3,5,7) minus even-harmonic (j = 2,4,6) share relative to the fundamental, read against the **discrete** square-wave reference at the same p, not against zero | both |
| phase error | circular `φ_out − (φ_a + φ_b)`, resultant length with bootstrap CI, against the permutation null | both |
| model-comparison win rate | fraction of neurons where an odd-harmonic or square model beats the single sinusoid by AIC | both |
| causal ablation damage | Δ test loss and accuracy, structured ablation minus size-matched random control | both |
| generalization step | first evaluated step with test accuracy ≥ 0.95, reported as an interval | both |

Secondary: spectral entropy, participation ratio, Doshi's inverse participation ratio, top-1/top-3/top-8,
Swaroop's periodicity score, explained variance of the key-frequency subspace, per-class logit shift,
margin change, weight norms, `additivity_r2`.

### 4.2 Structured neuron — primary definition `[AI-PROPOSED]`

A hidden neuron is *structured* iff

1. it is **alive** (maximum activation over the grid > 0);
2. `u_a` and `u_b` share the dominant frequency `k`;
3. the **family fraction** of `u_a` **and** of `u_b` is ≥ **0.50**;
4. the output curve's dominant frequency is also `k`.

Mandatory sensitivity variants, always reported next to the primary: family threshold **0.30** and
**0.70**; the **top-1** variant (dominant fraction ≥ 0.50); **Doshi's IPR** ranking; **Swaroop's
periodicity score** with his 12 / 5 cuts — labelled as his post-hoc histogram cuts, never as principled.

The same definition is applied to the transformer's effective operand curves. Where those curves are not
additive (`additivity_r2 < 0.9`) this is reported as a fairness limitation and the transformer's hidden
activations are analysed directly as well.

### 4.3 Key-frequency rule `[AI-PROPOSED]`

**Primary: `nanda`.** The DFT of the neuron→logit map along the class axis — transformer
`W_L = W_out @ W_U[:, :p]`, MLP and two-hot `W_L = W_out` — norm taken over neurons; keep every `k` whose
norm is ≥ **0.25 × the maximum**. The count is measured, never capped.

This rule was **pre-declared as a conditional** before the sources were read: the brief stated that if the
source note established a published rule, that rule would become primary. The Nanda note established it
(App. C.2), so the substitution fired. The published rule does not state a numeric threshold
`[NOT FOUND IN SOURCE]`; **0.25 is our choice**, following the TransformerLens demo's `> max/4`
convention, with sensitivity at **0.10** and **0.50**.

Reported alongside with pairwise Jaccard overlap: `neuron_clusters`, `embedding_threshold` (uncapped
90 %), `logit_sum_directions`. `embedding_top8` is reported **only** as the legacy number — on all 16
legacy runs its cap binds and the 90 % threshold never fires, so its count was never data-determined.

### 4.4 Fairness across architectures

Every metric used to compare architectures is defined on objects that mean the same thing in both:
effective `a`/`b` curves, output curves, hidden activations over the grid, per-neuron logit
contributions, and the embedding. Objects that exist only in the transformer — `W_pos`, `W_Q/W_K/W_V/W_O`,
the residual stream, attention heads, the direct non-MLP logit path — are analysed and reported but
**excluded from the headline comparison**. The transformer's effective curves carry their
`additivity_r2` bound with every number derived from them
(`docs/TRANSFORMER_MECHANISM_DERIVATION.md` §4.1).

---

## 5. The evidence gate

Evaluated **per architecture, per seed**, at the final checkpoint, and reported at the crossing checkpoint
as well.

| criterion | pass rule `[AI-PROPOSED]` |
|---|---|
| **G1** periodic structure | structured-neuron fraction ≥ 0.25 of live neurons **and** median family fraction of `u_a`, `u_b` ≥ 0.50 |
| **G2** phase addition | resultant length `R` > the permutation null's 95th percentile **and** `R ≥ 0.5` |
| **G3** end-to-end logit fit | the sparse-sinusoid **or** odd-harmonic formula reaches argmax accuracy ≥ 0.90 on test cells and R² above the random-frequency control's 95th percentile, while the `(a−b)` control formula stays below half that R² |
| **G4** causal | `remove_structured` and `remove_key_freqs` each drop test accuracy by ≥ 0.5 with the size-matched random control below 0.1 (`z ≥ 3`), **and** `keep_structured` retains test accuracy ≥ 0.9 |

**An architecture passes the gate if ≥ 8 of 10 seeds pass all four criteria.** The gate is re-evaluated
under the 0.30 and 0.70 structured-neuron thresholds and reported as a sensitivity analysis.

---

## 6. Decision tree, fixed before the experiments

| outcome | what we do, and what we may say |
|---|---|
| **both architectures pass** | investigate H3 and compare how the Fourier principle is represented and computed |
| **only one passes** | a bounded investigation of the other architecture's representation (§6.3); the passing architecture's mechanism is reported, the other's is not assumed |
| **neither passes** | report that the Fourier evidence tested does not identify the learned mechanisms, and investigate the validity of the existing metrics. This is a publishable outcome, not a failure |

### 6.1 End-to-end logit formula fit (mandatory, master prompt §9)

Weight-level periodicity does not establish that the model implements the algorithm. Candidate formulas
are fit to the actual logits over **all 12,769 (a,b) pairs and all 113 classes**, for both architectures
and every seed: a sparse-sinusoid Fourier formula, an odd-harmonic/square formula, a cardinality-matched
top-m control, a random-frequency control (50 seeded draws), an `(a−b)` control of matched flexibility,
and the full sum-basis ceiling. R², residuals, per-class quantiles, AIC and argmax accuracy on train and
test cells are reported for every seed. **Without this fit it must not be claimed that the model
implements the proposed formula.**

### 6.2 Functional equivalence over the full domain (mandatory, master prompt §10)

Across all 12,769 input pairs: prediction agreement, the share both get correct, inputs only one gets
correct, the overlap and structure of the errors, and logit and margin similarity — with
same-architecture seed pairs as the baseline against which any cross-architecture number is read. Until
this is measured, only "both architectures generalize on the same task" may be written.

---

### 6.3 Bounded alternative-mechanism analysis

If an architecture fails the gate, one level further is investigated, not an open-ended search:

* whether hidden activations encode `(a+b) mod 113` even without obvious Fourier concentration — linear
  and one-hidden-layer probes trained on the run's own training split, evaluated on its test split,
  against a shuffled-label control;
* whether activation or logit tensors contain low-rank or symmetry-based structure (SVD spectrum,
  effective rank, with the Khanh caveat that these must not be read at the transition);
* whether the structure recurs across seeds (CKA between seeds);
* whether removing it causally damages performance more than removing random directions of the same rank.

**The project does not promise to discover every possible alternative algorithm.** A well-founded negative
result stands if the bounded analysis identifies none.

## 7. Statistics

Full plan in `docs/STATISTICAL_ANALYSIS_PLAN.md`. In summary: the unit of analysis is a **seed pair**;
percentile bootstrap CIs with 10,000 resamples at seed 0; exact sign test and exact Wilcoxon signed-rank;
Cohen's d_z and Cliff's delta; median, MAD and IQR; **every individual seed plotted**. P-values are
descriptive companions and never the sole evidence. Runs that fail are never silently dropped: n started
and n completed are always reported, with a sensitivity analysis. With n = 10 there is no spurious
precision and no claim that an architecture is "fundamentally" anything from one hyperparameter region.

## 8. Causal ablations

Full plan in `docs/CAUSAL_ABLATION_PLAN.md`. Rules: unmodified checkpoints, **no retraining**, a
size-matched random control of 50 seeded draws for every ablation, both measurement points, repeated
across seeds, and ablations that do **not** damage the model reported as findings. At least two causal
tests per architecture, each as a necessity **and** a sufficiency test.

## 9. Pilot, freeze, and what may change afterwards

The **seed-0 pair** is the pilot. It validates that the pipeline runs end to end and produces valid
artifacts. **No threshold is tuned on it.**

After the pilot the analysis code is **frozen at a named commit**, recorded in `docs/LABBOOK.md` and in
§12 below. After the freeze:

* **may change:** performance, logging, figure styling, and genuine bug fixes — each logged in the labbook
  with its reason, and every affected analysis re-run on all seeds;
* **may not change:** any threshold, definition, pass rule, control, or statistic — unless the human
  authors change it in `docs/HUMAN_DECISIONS.md`, which forces a re-run and a labbook entry naming what
  changed and why.

Changing a threshold *because* a result under the old one was unwelcome is exactly the failure mode this
document exists to prevent.

The explorer stays frozen until the analyses are frozen and human-reviewed. Visualization must not drive
the research.

## 10. Reporting rules

Every claim is structured as: **observation → quantitative evidence → alternative explanation → causal
test → limitation → permissible conclusion.**

Permitted, graded: "We observe …", "This is consistent with …", "The ablation supports the interpretation
…", "The data are not sufficient to rule out …", "Under the conditions examined …".

**Banned without the evidence gate:** "proves", "clearly shows", "Transformers fundamentally learn …",
"MLPs are worse …", "the same circuit", "the same function", "exact transition".

The main claim may be used **only** if the data and the gate support it:

> "Transformer and MLP can solve the same modular task through related Fourier-based principles, but may
> express them in different neural representations."

If the results show the mechanisms are essentially the same, exactly that is reported. If neither
architecture passes the Fourier tests, exactly that is reported.

## 11. Values requiring human approval

Every `[AI-PROPOSED]` value in this document is collected with a blank decision column in
`docs/HUMAN_DECISIONS.md` (sections A–E), together with the division-of-labour questions in its section F.
That file currently reads `STATUS: NOT YET APPROVED BY HUMAN AUTHORS`. Until it does not, every result
must be reported as computed under AI-proposed settings.

## 12. Freeze record

| item | value |
|---|---|
| brief committed (values fixed) | `d53cf52` |
| primary runs executed at | `d53cf52` |
| analysis code freeze commit | **`0b55e1d`** (2026-09-03, 20:5x UTC) — the commit at which every module `analysis/driver.py` lists exists and the suite is green (16 files, 1,657 checks at the freeze; the suite has since grown to 24 files and 1,913 checks, all additions being tests of the aggregation layer and regression tests for the post-freeze fixes below) |
| human approval | **[pending — `docs/HUMAN_DECISIONS.md`]** |

**What the freeze covers.** Every module the driver invokes: `key_frequencies`, `mlp_mechanism`,
`transformer_mechanism`, `wave_fitting`, `logit_formula_fit`, `h3_validity`, `causal_ablation`,
`structure_over_time`, `progress_measures`, and the shared `common`/`metrics`/`fourier`/`mask_protocols`
they rest on. The aggregation layer (`aggregate`, `decision_tree`, `figures_study`,
`bounded_alternative`, INTERFACES §13) is **not** frozen here: it consumes the outputs and does not
change any measured number. Its own freeze is recorded when it lands.

**Post-freeze fixes so far** (§9 permits genuine bug fixes, logged and re-run):

| commit | what | why it is not a threshold change |
|---|---|---|
| 2026-09-03 | `key_frequencies.analyse` and `progress_measures.compute_from_checkpoints` now accept `seed` | both forwarded `**kw` into a helper that rejects it, so all 60 driver calls raised `TypeError`; the seed is **recorded and unused** — neither draws a random number. Both modules must be re-run over every seed (labbook 56–57). |
| 2026-09-03 | `wave_fitting` routed through the per-`k` split, and its architecture registry corrected to MLP-only | an `IndexError` on 4 MLP runs and 20 doomed transformer calls; no fitted number changes — the split is numerically identical (labbook 58, 60). |
| 2026-09-04 | `aggregate._wave_fitting` reads `fraction_best_by_aic` instead of guessed flat keys | **silent**: it produced the right column names with `None` in all of them, so the H2 figure would have been blank. Aggregation only; no measured number changes (labbook 58). |
| 2026-09-04 | `aggregate._progress_measures` reads the split level of `protocols[p][which][split]["loss"]` | **silent**: same shape — every restricted/excluded column was `None`. Aggregation only; every downstream report re-run and byte-identical apart from timestamps (labbook 99). |
| 2026-09-04 | `structure_over_time._fraction_best` resolves the model index through `MODEL_NAMES` | **silent, and it returned `0.0` rather than `None`**: `best_by_aic` is an integer index and was compared to the string `"square"`, so both waveform trajectories were flat zero for every run. The two metrics fed only H4's onset detector, which reported "undefined"; no gate criterion, statistic, figure or claim consumed them. `structure_over_time` re-run over all 51 runs and H4 regenerated (labbook 101). |

| 2026-09-06 | `causal_ablation.resolve_structured_masks` builds the available-definitions collection as an ordered tuple instead of a **set** | **silent, and it broke determinism**: a set made the key order of `masks["sets"]` depend on per-process string hashing, and both ablation functions iterate that dict while drawing from one shared `rng`, so which 50 size-matched control draws went to which structured definition changed between processes. Observed ablation values were never affected; the control means of `remove_structured` (MLP) and `remove_structured_neurons` (transformer) moved by up to 0.0034 and 0.0015 between runs of identical code on identical inputs. The fix removes nondeterminism and changes no definition, threshold, control or statistic. `MODULE_VERSION` 1.1 → 1.2; all 40 primary checkpoints re-run; escalated as **D8** (labbook 109). |

**What may still change after this commit** (PREREGISTRATION §9): performance, logging, figure styling
and genuine bug fixes, each logged in `docs/LABBOOK.md` with its reason and forcing a re-run of every
affected analysis. **What may not:** any threshold, definition, pass rule, control or statistic.

**Pilot outputs predating the freeze must be re-run.** The seed-0 `transformer_mechanism`,
`key_frequencies`, `logit_formula_fit`, `causal_ablation` and `structure_over_time` files written
while the code was still moving are pipeline checks, not measurements, and are regenerated by the
frozen driver.

**External block on a separate branch — not a post-freeze change to this code (2026-09-08).** The
100,000-step convergence block S1 (`RESULTS.md` §16) was executed by the external reviewer at commit `42dd79a`
on branch `arch-study-convergence`, which is this branch's `fda066e` plus ten `CHECKPOINT_GRID` entries above
25,000 (`30000 … 100000`) and the runbook now stored as `docs/external/CONVERGENCE_RUNBOOK_S1.md`. Entries
≤ 25,000 are unchanged, so every 25,000-step run selects the same 19 checkpoints; no threshold, definition,
pass rule, control or statistic changes. The grid is a constant fixed before the primary runs
(`training/grokverse/config.py`), so this is a **declared deviation, fixed before that block ran**; it is
disclosed in `AI_DISCLOSURE.md` §5 and §9 and its acceptance is `docs/HUMAN_DECISIONS.md` **D9**. This
branch's `config.py` is unchanged, and the reviewer's branch is held locally, unpushed.

## 13. Deviations from `docs/RESEARCH_SPEC.md`

| item | research spec | here | reason |
|---|---|---|---|
| stopping rule | train to 1.5× each run's own crossing | fixed 25,000 steps for every run | human decision 2026-09-02; a shared budget is required for a controlled comparison and avoids an event-relative "final" state |
| key-frequency rule | `embedding_top8` with a threshold that never fires | Nanda's neuron→logit-map rule, uncapped count | the source note established the published rule; the pre-declared conditional fired |
| per-neuron sum-vs-difference test | expected sum ≫ difference | equal by derivation; tested on the logits instead | the prediction was mathematically wrong; corrected before any analysis |
| two-hot control width | follow the source | keep `d_mlp = 512` | so the control varies the input parametrization only |

## 14. Addendum — graded structured-neuron ablation

> **[AI-PROPOSED], drafted 2026-09-06.** Added **after** the analysis freeze and **after** the primary
> results were analysed, in response to an objection raised independently by an external reviewer on
> 2026-09-05. Written **before any graded result was inspected**: at the time of writing, the only
> things read from the existing artifacts were array names, shapes and dtypes — no value from any
> pruning curve or family-fraction array had been looked at. Human approval is open as
> `docs/HUMAN_DECISIONS.md` **D7**.

### 14.1 Why this is a new pre-registration and not a threshold change

`HUMAN_DECISIONS.md` **D6** records that under B1 the structured set is 86–100 % of the live neurons
under *every* definition the pre-registration offers, so a size-matched random control does comparable
damage by construction and G4 can discriminate neither necessity nor sufficiency. D6 also records the
measured conclusion that **no available threshold rescues G4**, and that what would is *a different kind
of selection*, which is a new pre-registration rather than a tweak of B1.

This section is that new pre-registration. It therefore:

* **does not change B1**, G4, any gate criterion, pass rule, control or statistic;
* **does not change any number already reported.** The block is additive; every existing output of
  `causal_ablation` is byte-identical apart from the new key;
* **does not change the gate verdict.** The gate stands as reported (`neither_passes`), and §14.5 fixes
  in advance that no outcome here may reopen it.

§9 permits post-freeze additions that are logged and do not alter a measured number. This one is logged
here, in `docs/LABBOOK.md`, and in the freeze record's post-freeze table.

### 14.2 Question

**Q5b.** At what group size, if any, does removing the *most structured* live neurons damage the model
more than removing an equally large **random** group of live neurons?

### 14.3 Ranking score, fixed before measurement

Primary score, per live neuron `i`:

```
s_i = min( u_a__family_fraction_i , u_b__family_fraction_i )
s_i = -1   if neuron i fails B1's categorical conditions
```

B1's categorical conditions are: `u_a` and `u_b` share the dominant frequency `k`, and the output curve's
dominant frequency is also `k`. Dead neurons are excluded via `mask__alive` and never ranked.

**Rationale.** `s` is exactly the quantity B1 thresholds at 0.50. Ranking by it makes "the top `n`
neurons" a **nested family of subsets of the B1 structured set**, so the sweep contains the existing G4
test as its limiting case instead of testing something unrelated to it. Demoting the categorical
failures to `-1` keeps them strictly last without inventing a continuous surrogate for a boolean.

**Sensitivity score** (always reported alongside, never used for the decision rule):
`s'_i = mean( u_a__family_fraction_i , u_b__family_fraction_i , out__family_fraction_i )`, no demotion.

### 14.4 Design

| item | value |
|---|---|
| fractions `f` | 0.01, 0.02, 0.05, 0.10, 0.25, 0.50 of the **live** neurons |
| group size | `n_f = max(1, round(f · n_alive))` |
| operations per fraction | `remove_top` (necessity direction) and `keep_only_top` (sufficiency direction) |
| control | **50 seeded, size-matched random groups** drawn from the live neurons, `seed = 0`, reproducible; the same two operations |
| checkpoints | both measurement points, as everywhere else |
| runs | the primary block: 10 MLP and 10 transformer seeds |
| model state | unmodified checkpoints, **no retraining** |

Per fraction, per operation, we report: observed test accuracy, observed drop, control mean drop,
`z = (observed_drop − control_mean_drop) / control_sd`, and whether the observed drop **exceeds all 50**
controls. Seeds are paired as everywhere else; the median and all ten seeds are reported, never a
p-value alone (§7).

### 14.5 Decision rule, fixed now

The structured ranking counts as **discriminable at fraction `f`** for an architecture and checkpoint if,
in **≥ 8 of 10 seeds**, the observed `remove_top` drop exceeds **every one** of the 50 size-matched
controls **and** `z ≥ 3`. The reported result is the **smallest such `f`**, per architecture and per
checkpoint, or "none of the tested fractions".

**What may not be concluded, fixed before the measurement:**

* A positive result **does not make G4 pass.** G4's verdict is fixed by its own pre-registered rule and
  remains `neither_passes`. This analysis is reported separately and never folded into the gate.
* A positive result says the structured **ranking** carries causal signal at small group size. It does
  **not** establish that the B1 **set** is the mechanism, and it does not license "the same circuit",
  "the same function", or any other phrase banned by §10.
* A null result across all six fractions is a **negative result** and is reported as one, in
  `RESULTS.md` §11, not quietly dropped.
* No threshold in this section may be changed after a result is seen. If one is changed, every affected
  analysis is re-run and the change is logged with its reason.

### 14.6 Zero-cost companion: the IPR sweep that already exists

`ipr_ranked_pruning` (pre-registered as **D2**, the Doshi sweep) already provides a graded, ranked
pruning curve at 21 fractions with a 50-draw control, for both architectures and both checkpoints. It
will be read at the matching fractions and reported **side by side** with no new computation.

It answers a **different** question and is labelled as such wherever it appears: its ranking is the
**IPR**, not the family fraction, and its control is a **random-order permutation** of the whole pruning
sequence, not 50 independent size-matched draws. Agreement between the two is evidence; disagreement is
a finding about the ranking, not about the model.
