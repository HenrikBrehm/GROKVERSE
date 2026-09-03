# Causal ablation plan

**AI-drafted (Claude), 2026-09-03 — not yet human-reviewed.** Written before any ablation was run.
Implements master prompt §11 and gate criterion G4 of `docs/PREREGISTRATION.md`.
Implementation: `training/grokverse/analysis/causal_ablation*.py` (interface: `docs/dev/INTERFACES.md` §9).

---

## 1. Why ablations, and what they can license

The evidence chain the study commits to:

1. the structure is present (spectra, waveform fits) — **descriptive**;
2. it has the predicted internal relations (phases add) — **mechanistic**;
3. removing it breaks the model far more than removing an equal amount of anything else — **causal**.

**Only step 3 licenses the word "algorithm".** A descriptive plot, however periodic, does not.

Equally: an ablation that removes a hypothesized structured component and **does not** damage the model is
a finding about H5 and is reported, not omitted.

## 2. Rules that apply to every ablation

| rule | why |
|---|---|
| run on **unmodified trained checkpoints**; the model is deep-copied before modification | the question is about the trained function, not about training |
| **no retraining**, ever | a retrained model answers a different question; if a retraining study is wanted it is a separate, separately declared study |
| every ablation has a **size-matched random control**: the same number of neurons, frequencies or dimensions, drawn uniformly, `n_control = 50`, seeded (default 0) | without it, "damage" only measures how much was removed |
| run at **both measurement points** (generalization crossing and final) | an ablation at the crossing and one after the budget answer different questions |
| repeated across **all seeds** of the block | a single-seed ablation is an anecdote |
| absolute **and** relative changes reported | a 0.5 drop from 1.0 and from 0.6 are different findings |
| results are measurements; the JSON key names say what was measured, never what it means | no `is_causal` keys |

## 3. What is measured after every ablation

For the ablated model and for each of the 50 controls:

* train and test **cross-entropy** and **accuracy**, absolute and relative to the unablated model;
* **mean absolute logit change** over the full `(a, b)` grid;
* **margin change**: mean correct-class logit minus the best incorrect logit;
* **per-class accuracy change** (stored in the `.npz`, summarized in the JSON);
* `n_components_removed`;
* `z = (observed_damage − control_mean) / control_std`, plus the control's mean, std and 5/95 quantiles.

## 4. MLP ablations

The MLP's forward pass is exactly `ReLU(u_a[a] + u_b[b] + b_in) @ W_out + b_out`
(`docs/MLP_MECHANISM_DERIVATION.md` §2, verified to 1.1e-7), so every ablation below is applied to that
functional form and is exact.

`S` = the set of structured neurons under the primary definition (`docs/PREREGISTRATION.md` §4.2).

| id | what is modified | test type | size-matched control |
|---|---|---|---|
| `keep_structured` | zero every neuron outside `S` | **sufficiency** of `S` | keep a random set of `|S|` neurons |
| `remove_structured` | zero the neurons in `S` | **necessity** of `S` | zero a random set of `|S|` neurons |
| `remove_structured_by_frequency` | zero the `S`-neurons of one key frequency at a time | necessity per frequency | zero an equally sized random neuron set |
| `remove_key_freqs_from_curves` | project `u_a`, `u_b` onto the complement of the key-frequency subspace | necessity of the key frequencies | project out an equal number of random frequencies |
| `keep_key_freqs_in_curves` | project `u_a`, `u_b` onto the key-frequency subspace plus the constant | sufficiency of the key frequencies | keep an equal number of random frequencies |
| `replace_with_sinusoid_fit` | substitute `u_a`, `u_b` by their best single-sinusoid fits | sufficiency of the **sinusoid** model | substitute by a phase-scrambled fit of the same amplitude |
| `replace_with_square_fit` | substitute `u_a`, `u_b` by their best square-wave fits | sufficiency of the **square** model | same scrambled control |
| `wrong_phase_vs_weak_periodicity` | zero the group "correct frequency, phase error > 45°" versus the group "correct phase, dominant fraction < 0.3", the two groups size-matched to each other | which property carries the computation | a random set of the same size |
| `ipr_ranked_pruning` | cumulative pruning of hidden neurons ranked by Doshi's IPR, **in both directions** (lowest first, highest first), over the full 0–1 range | replication of Doshi et al.'s protocol | random-order pruning, 50 seeded orders |

`ipr_ranked_pruning` follows Doshi, Das, He & Gromov (arXiv:2310.13061) §2.1 / Fig. 6: start from a trained
network, prune one neuron at a time, measure train and test loss *and* accuracy at every pruning size.
**Their protocol has no size-matched random control** — that is our addition, and it is what makes the
sweep answer H5 rather than only reproducing their curve.

## 5. Transformer ablations

Applied to the exact forward decomposition of `docs/TRANSFORMER_MECHANISM_DERIVATION.md` §2 (verified to
2.7e-7), so that each ablation modifies exactly one stage and the rest of the forward pass is untouched.

| id | what is modified | test type | size-matched control |
|---|---|---|---|
| `remove_key_freqs_from_embedding` | filter the key-frequency rows out of `W_E[:p]` in the Fourier basis | necessity | remove an equal number of random frequencies |
| `keep_key_freqs_in_embedding` | keep only those rows plus the constant | sufficiency | keep an equal number of random frequencies |
| `remove_structured_neurons` | zero the hidden neurons in `S` | necessity | zero a random set of `|S|` neurons |
| `keep_structured_neurons` | zero every neuron outside `S` | sufficiency | keep a random set of `|S|` neurons |
| `ablate_head_h` | for each head: replace `z_h` by 0 (zero-ablation) and by its grid mean (mean-ablation) | necessity of that head | ablate a random equal-rank direction of the OV output |
| `fix_attention_to_mean` | replace the attention row `A` by its grid mean `Ā` everywhere | **is attention a fixed sum?** | none needed — this is a structural test, see below |
| `remove_key_subspace_from_residual` | project the `cos/sin(ω_k(a+b))` directions of the key frequencies out of `r_pre` over the grid, then finish the forward pass | necessity of the key subspace | project out an equal number of random directions of the same rank |
| `restricted_circuit_only` | replace the logits by their `sum_directions_only` restriction to the key frequencies | sufficiency — this **is** the restricted loss, reported here as a causal test | restrict to an equal number of random frequency pairs |

`fix_attention_to_mean` is the decisive test for the ambiguity that `docs/RESEARCH_SPEC.md` §3.4 raised: a
50/50 *mean* attention is consistent both with a constant split and with input-dependent attention that
averages to 50/50. If replacing `A` by `Ā` barely changes the loss, attention is a fixed weighted sum and
the transformer is additive-then-ReLU like the MLP; if the loss moves, the transformer has a
multiplicative path the MLP does not have. Either outcome is reported; neither is assumed.

## 6. Interpretation rules, fixed in advance

For a component set `C` in one architecture, one seed, one measurement point:

| outcome | reading |
|---|---|
| removing `C` drops test accuracy by ≥ 0.5 **and** the size-matched control drops < 0.1 (`z ≥ 3`) | `C` is **necessary** |
| keeping only `C` retains test accuracy ≥ 0.9 | `C` is **sufficient** |
| both of the above | `C` carries the computation — the only case that licenses "algorithm" |
| removing `C` and the control do comparable damage | `C` is **not** specifically load-bearing; H5 fails for that component, and this is reported |
| removing `C` does no damage at all | `C` is present but unused — a finding about the structure metric, not about the model |

**Gate criterion G4** (`docs/PREREGISTRATION.md` §5) requires, per seed: `remove_structured` **and**
`remove_key_freqs` each necessary by the rule above, **and** `keep_structured` sufficient. An architecture
passes the gate criterion if that holds in ≥ 8 of 10 seeds.

The thresholds 0.5, 0.1, `z ≥ 3` and 0.9 are `[AI-PROPOSED]` and are listed for approval in
`docs/HUMAN_DECISIONS.md` (D1). They are **not** tuned after seeing an ablation result; if a threshold is
changed, the change is logged in the labbook with its reason and every ablation is re-run.

## 7. Fairness across architectures

| ablation | exists in | in the headline comparison? |
|---|---|---|
| structured-neuron keep/remove | both | **yes** |
| key-frequency keep/remove on the operand curves | both | **yes** |
| waveform substitution | both | **yes** |
| head ablation, `fix_attention_to_mean`, residual-subspace projection, `restricted_circuit_only` | transformer only | **no** — reported as transformer-specific evidence |
| IPR-ranked pruning | both, but Doshi's causal evidence is for a 2-layer quadratic MLP only | reported for both; the claim that it is causal in the transformer is **new** and needs its own control |

Where an ablation exists in only one architecture it can support a statement about that architecture, never
a comparison between them (master prompt §17).

## 8. Compute and ordering

Ablations require no training. Each one is a forward pass over the 12,769-point grid plus 50 control
forward passes; at `p = 113` a single grid evaluation is milliseconds, so the full table over 20 runs × 2
measurement points is minutes, not hours.

Order: the two gate-critical ablations (`remove_structured`, `remove_key_freqs`, `keep_structured`) run
first on the pilot seed to validate the machinery, then the full table over all seeds after the analysis
code is frozen.

## 9. Known ways this can mislead, stated in advance

* **Removing more removes more.** Every number is read against its size-matched control; a raw damage
  figure without its control is not reported.
* **Zeroing a neuron is not "deleting its computation"** if downstream weights compensate; that is why
  sufficiency (`keep_only`) is tested as well as necessity.
* **Ablation is about this checkpoint.** It cannot show that the structure *had to* form, or how it formed;
  H4's structure-over-time analysis addresses that question separately and is correlational.
* **A structured set chosen by a threshold inherits that threshold.** Every ablation is repeated under the
  0.30 and 0.70 structured-neuron definitions as a sensitivity analysis.
* **The projections are only exact because there is no LayerNorm** in this transformer. That is stated in
  the derivation and must not be silently relied on if the architecture ever changes.
