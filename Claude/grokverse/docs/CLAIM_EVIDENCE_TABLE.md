# Claim–evidence table

**AI-drafted, 2026-09-04. Not yet human-reviewed.** Every claim below is structured as master prompt
§21 requires: observation → quantitative evidence → alternative explanation → causal test →
limitation → permissible conclusion. Nothing here may enter `RESULTS.md` in stronger wording than the
"permissible conclusion" line allows.

**Scope.** The primary block only: 20 runs, 10 paired seeds, mod 113, `train_frac = 0.3`,
`weight_decay = 1.0`, no Grokfast, 25,000 steps, both architectures on the same splits (paired seeds
share a split hash). Control blocks are analysed separately and are marked *pending* where they bear
on a claim.

**The precondition that governs the whole table.** The pre-registered evidence gate reports
**`neither_passes`** (`results/decision_tree_final.json`). Master prompt §12 therefore forbids reading
any harmonic-family or "same Fourier principle" interpretation into these numbers for *either*
architecture. Claims C1–C4 are about measured structure; none of them is a mechanism claim.

---

## C1 — Both architectures reach near-perfect accuracy, and compute nearly the same function

| | |
|---|---|
| **Observation** | Both architectures generalize, and their input–output behaviour over the full domain is nearly identical. |
| **Quantitative evidence** | Over all **12,769** input pairs at the final checkpoint, median agreement **0.99977** across the 10 paired seeds. The MLP is correct on **every** cell in **all 10** seeds; the transformer has 0–26 errors (median 3). In **4 of 10** seeds the two agree on *every* input. `results/function_agreement/*_arch25k_*.json`. |
| **Alternative explanation** | Both simply memorize a task small enough that any sufficiently flexible model lands on the same function; agreement need not indicate a shared mechanism. |
| **Causal test** | None applies — this is a behavioural measurement by construction. |
| **Limitation** | Agreement over inputs says nothing about *how* either model computes. `error_jaccard` is 0.000 where defined and **undefined in 4 seeds** because neither model errs (empty union), so error-overlap structure carries almost no information here. |
| **Permissible conclusion** | "Under the conditions examined, the two architectures agree on 99.98 % of all inputs (median over 10 paired seeds) and are identical on 4 of 10 seeds." Master prompt §10's restriction is now discharged by measurement: the phrase **"both learn the same function" remains unsupported** — they agree closely, they are not identical in 6 of 10 seeds. |

## C2 — Periodic structure, the phase relation and an end-to-end Fourier fit are present in both

| | |
|---|---|
| **Observation** | Gate criteria G1, G2 and G3 hold for both architectures on every seed. |
| **Quantitative evidence** | G1 (structured fraction ≥ 0.25 of live **and** median family fraction ≥ 0.50): **10/10** both. G2 (resultant length `R` above the permutation null's 95th percentile **and** ≥ 0.5): **10/10** both. G3 (a sparse-sinusoid or odd-harmonic formula reaching argmax accuracy ≥ 0.90 on test cells, above the random-frequency control, with the `(a−b)` control below half): **10/10** both. `results/decision_tree_final.json`. |
| **Alternative explanation** | The structured-neuron definition is permissive: it selects **442/512** neurons in the MLP and **501/512** in the transformer. "Most neurons are structured" is compatible with the threshold being loose rather than the network being organised. |
| **Causal test** | G4, and it **fails** — see C3. |
| **Limitation** | G3's random-frequency control is degenerate on some runs (it can draw all 56 available frequencies at `p = 113`), so that condition is not always testable as specified; at the crossing checkpoint it is untestable on all 10 MLP seeds. |
| **Permissible conclusion** | "Periodic structure, the predicted phase-addition relation and an end-to-end Fourier fit are observed in both architectures at the final checkpoint, on all 10 seeds, under the pre-registered thresholds." **Not** "the models implement a Fourier algorithm" — that requires G4. |

## C3 — Causal necessity of the *structured-neuron set* is not established, and the reason is the threshold

| | |
|---|---|
| **Observation** | G4 fails on 10/10 seeds for both architectures; the gate branch is `neither_passes`. |
| **Quantitative evidence** | MLP `remove_structured`: observed test-accuracy drop **0.991**, size-matched random control **0.873**. Transformer `remove_structured_neurons`: **0.942** vs control **0.916**. `keep_structured` retains 1.000 / 0.9997 — but so does a random set of the same size (control drop 0.000). Structured set: 442/512 (MLP), 501/512 (transformer); for transformer seed 5 it is **all 512**, so no control set exists and the value is `null`. |
| **Alternative explanation** | This is the more likely reading: removing 86–98 % of any network destroys it, so the size-matched control cannot discriminate. `CAUSAL_ABLATION_PLAN.md` §6 names this outcome in advance — "removing `C` and the control do comparable damage → `C` is **not** specifically load-bearing". |
| **Causal test** | This *is* the causal test. |
| **Limitation** | The result is a property of the **structured-neuron threshold** (`HUMAN_DECISIONS` B1, family fraction ≥ 0.50), not a measurement of the models. Changing that threshold would force a full re-run and must not be done because this outcome is inconvenient. Recorded as `HUMAN_DECISIONS` **D6**. |
| **Permissible conclusion** | "Under the pre-registered structured-neuron definition, the ablation cannot distinguish the structured set from a size-matched random set of neurons, so causal necessity is **not** established for either architecture." |

## C4 — The key frequencies *are* causally load-bearing, and this is where the two architectures differ

| | |
|---|---|
| **Observation** | Unlike the neuron sets, the key-frequency ablations separate sharply from their controls. |
| **Quantitative evidence** | MLP `remove_key_freqs_from_curves`: drop **0.989**, control **0.000** → necessary **10/10**. Transformer `remove_key_freqs_from_embedding`: **0.984** vs **0.0005** → **10/10**. Transformer `keep_key_freqs_in_embedding`: drop **0.004** while the control drops 0.990. Transformer `restricted_circuit_only`: drop **−0.000** vs control 0.991. But transformer `remove_key_subspace_from_residual`: drop **0.235**, control 0.000 → **0/10**. |
| **Alternative explanation** | The embedding-level ablations remove a large amount of signal; the residual-subspace projection removes a rank-limited direction set. The three transformer ablations act on different objects and are not interchangeable. |
| **Causal test** | These are the causal tests, each with a size-matched random control. |
| **Limitation** | **G4's second condition does not name which transformer ablation counts as "remove_key_freqs"**, and the choice decides the criterion: `remove_key_subspace_from_residual` gives 0/10, `remove_key_freqs_from_embedding` gives 10/10. The AI wired the former; it was **not** changed after the numbers were seen. Recorded as `HUMAN_DECISIONS` **D5**. Cross-architecture comparison of these numbers is not permitted (master prompt §17): they modify different objects. |
| **Permissible conclusion** | "In both architectures, removing the key-frequency components damages the model far more than removing an equal number of random frequencies. Which transformer ablation the pre-registered gate criterion refers to is a human decision that changes that criterion's verdict but not the overall gate outcome." |

## C5 — H3, the study's proposed primary contribution, is refuted by its own criterion

| | |
|---|---|
| **Observation** | The harmonic-aware definition does not close the architecture gap in the structured-neuron fraction. |
| **Quantitative evidence** | Gap under the top-1 definition **+0.0898**; under the family definition **+0.0850**. The family definition closes **+0.0049**, i.e. **5.4 %** of the gap. Criterion 2 (is the MLP still lower with a CI excluding zero?): median **+0.0850**, CI95 **[+0.0600, +0.1004]** → yes. Criterion 1 (CI of the gap difference containing zero?): **[+0.0012, +0.0076]**, excludes zero → this branch does not refute. `results/h3_report.json`. |
| **Alternative explanation** | The particular family definition (own dominant frequency plus aliased odd harmonics ≤ 7) may be the wrong harmonic-aware measure, or the effect may exist but be smaller than this design can resolve at n = 10. |
| **Causal test** | Not applicable — H3 is a claim about a metric, not about a mechanism. |
| **Limitation** | At `p = 113` the odd-harmonic families of different fundamentals alias onto one another (those of `k = 6, 19, 51` all contain 18), which bounds what any family-based measure can identify. |
| **Permissible conclusion** | "By the criterion fixed before the experiments, the MLP's lower structured-neuron fraction is **a real difference in the measured structure, not an artifact of top-k concentration**. H3 as pre-registered is refuted." |

## C6 — The metric *is* waveform-sensitive; that sensitivity just does not explain the gap

| | |
|---|---|
| **Observation** | The mechanism H3 hypothesised is present in the measuring instrument. |
| **Quantitative evidence** | Two populations built from each checkpoint's own `(k, phase, amplitude)`, differing **only** in waveform: top-1 concentration differs by **+0.1893**, identical for both architectures (as it must be — it is a property of the metric). The odd-minus-even harmonic shape separates the two populations in the opposite direction. |
| **Alternative explanation** | None needed; this is a synthetic construction with a known answer. |
| **Causal test** | Not applicable. |
| **Limitation** | It quantifies what the metric would do to an idealized square-wave population, not what the trained models contain. |
| **Permissible conclusion** | "Top-k concentration scores a square-wave population ≈ 0.19 lower at top-1 than a sinusoidal population with the same fundamentals. The artifact is real and quantified; it accounts for only 5.4 % of the observed architecture gap (C5)." |

## C7 — Timing: the transformer generalizes earlier in most but not all seeds

| | |
|---|---|
| **Observation** | The transformer's generalization crossing precedes the MLP's in 9 of 10 seeds. |
| **Quantitative evidence** | Median paired difference **−1,562 steps**, CI95 **[−2,895, −822]**. Crossings: transformer 5,625–10,275; MLP 8,150–10,075. **Seed 4 reverses the direction** (transformer 10,275 vs MLP 8,650). All 10 per-seed interval-consistent bounds exclude zero, so the ±25-step evaluation grid never flips a sign. |
| **Alternative explanation** | A single hyperparameter point; the confound block (Grokfast × `train_frac`) is analysed separately and could show the ordering is protocol-dependent. Manir & Rupa report a timing gap in the opposite direction. |
| **Causal test** | Not applicable — this is a timing observation. |
| **Limitation** | n = 10, one hyperparameter point, direction not unanimous. The evaluation grid is 10 steps (train) / 25 steps (test); **the phrase "exact transition" is not used**. |
| **Permissible conclusion** | "Under the conditions examined, the transformer's generalization crossing precedes the MLP's by a median of about 1,560 steps, in 9 of 10 seeds; one seed reverses the direction." **Not** "transformers grok faster". |

## C8 — Structure and waveform differ measurably between the architectures

| | |
|---|---|
| **Observation** | Paired differences in several structure metrics exclude zero, several unanimously. |
| **Quantitative evidence** | Structured fraction (final) **+0.085**, CI [+0.060, +0.100], unanimous. Phase-relation `R` (final) **−0.0074**, CI [−0.0098, −0.0056], unanimous. Square/odd-harmonic best-fit fraction **−0.324**, CI [−0.397, −0.120]. Best Fourier-formula R² **−0.426**, CI [−0.459, −0.336], unanimous. Family-minus-top-1 gap **−0.0025**, CI [−0.0060, **+0.0035**] — **spans zero**. `results/statistics.json`. |
| **Alternative explanation** | Parameter count differs by ~10 % between the default configurations; the parameter-matched block exists to test this and is **pending**. Several metrics are computed on objects that are not strictly comparable across architectures (the transformer's effective curves are a mean-attention approximation with `additivity_r2 < 1`; the MLP's are exact). |
| **Causal test** | C3 and C4. |
| **Limitation** | n = 10, one hyperparameter point. The transformer's `additivity_r2` is not 1, so its "effective curves" are an approximation and its structure metrics are not on identical footing with the MLP's. |
| **Permissible conclusion** | "Under the conditions examined, the MLP shows a higher fraction of neurons best fit by a square or odd-harmonic model, and the transformer a higher structured-neuron fraction, with intervals excluding zero. The family-minus-top-1 gap does not differ between the architectures — its interval spans zero." |

---

## Claims that may **not** be made from this evidence

| forbidden | why |
|---|---|
| "the two architectures use the same circuit" | master prompt §12 bans it even when the gate passes; the gate did not pass |
| "the same Fourier principle in different representations" | conditional on passing the gate — `neither_passes` |
| "both learn the same function" | measured: 99.98 % agreement, but not identical in 6 of 10 seeds (C1) |
| "the transformer learns a sparser Fourier circuit" | the concentration gap is a **replication** (Manir & Rupa 2026), and C3 shows no causal warrant for "circuit" |
| "transformers grok faster" | one hyperparameter point, direction not unanimous (C7) |
| "the MLP's structure is a measurement artifact" | refuted by C5 — it is a real difference |
| "the MLP is more square-wave-like *because it is an MLP*" | the two-hot control moves the same metric by **+0.1836** with 3 seeds, so the input parametrization carries part of it |
| any effect attributed to Grokfast alone | the 2 x 2 separates it from `train_frac`, and its main effect is the size of the interaction and, for the transformer, indistinguishable from zero |
| any of the above read as a comparison of **converged** states | `structured_fraction_of_live` is settled in 2 of 10 MLP seeds at the budget (`LIMITATIONS.md` B3) |

## Resolved since this table was first written (2026-09-04)

Every item that stood as *pending* has been measured. The results are in `RESULTS.md`; what they change
here is recorded below, so the table is not read against a state that no longer exists.

| was pending | outcome | does it change a claim above? |
|---|---|---|
| **Confound block** (Grokfast x `train_frac`, 18 runs) | `train_frac` moves the generalization step by -6,367 (txf) / -8,217 (MLP); Grokfast by -350 / +892; interaction +883 / -933. No CI at 3 seeds per cell. | **No** — C7 stands. It retires the *earlier repository* claim that read a speed-ratio change as a Grokfast effect. |
| **Parameter-matched block** (10 runs, `d_mlp = 572`, 0.02 % apart) | every structure difference survives with the same sign and an interval excluding zero: structured fraction +0.0969 [+0.0863, +0.1240], phase `R` -0.0076, square/harmonic -0.2230, generalization -2,212. | **No** — it removes C8's leading alternative explanation. |
| **Two-hot block** (3 runs) | the two-hot MLP is **more** square-wave-like than the shared-embedding MLP: **+0.1836** [+0.1133, +0.2109]. | **Yes — C8 is weakened.** The waveform difference is at least partly input parametrization, not architecture. C8's permissible conclusion must be read with that attached. |
| **H4 / structure over time** | 4 of 7 metrics reach onset before the generalization crossing in 10/10 seeds, both architectures; 2 metrics never produce a defined onset. Correlational. | **No** — it adds a claim rather than changing one. |
| **Bounded alternative-mechanism analysis** | probes at ceiling; effective rank 12.7 (txf) vs 66.6 (MLP); top-16 singular-direction removal destroys the transformer (0.964) and not the MLP (0.000); **cross-seed CKA 0.0017 / 0.0973 between seeds of the same architecture**. | **Yes, indirectly.** The CKA result means representation similarity cannot be cited for or against mechanism sameness anywhere in this study. |

## Two further limits found after this table was written

- **C2 and C8 rest on a metric that has not converged.** `structured_fraction_of_live` is settled at the
  25,000-step budget in 10/10 transformer seeds but only **2/10 MLP seeds**, and the MLP's is still
  rising. The +0.085 gap is a budget-fixed comparison whose bias has a known direction: a longer budget
  would shrink it. (`docs/LIMITATIONS.md` B3.)
- **C6's waveform trajectory was unmeasurable until 2026-09-04.** `structure_over_time` reported the
  square- and sinusoid-best-fit shares as exactly 0.0 at every checkpoint, because it compared an
  integer model index to a model name (labbook 101). The corrected series shows the transformer's
  operand curves starting at 0.770 square-best-fit and decaying to 0.000. No claim in this table
  consumed the broken values.
