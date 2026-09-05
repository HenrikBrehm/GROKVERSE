# HUMAN_DECISIONS.md — approval gate for the architecture study

```
STATUS: NOT YET APPROVED BY HUMAN AUTHORS
```

Every value below was **proposed by the AI** and is in force only provisionally. Fill in the *Human
decision* column, change anything you disagree with, then set the status line above to
`STATUS: APPROVED BY HUMAN AUTHORS` with your names and the date. Until then, every result produced under
these values must be reported as computed under AI-proposed settings.

Two things are already **human decisions**, taken on 2026-09-02, and are not up for approval here: the
fixed 25,000-step budget, and the scope of the run matrix (ten paired seeds plus the full control matrix).
They are recorded in `docs/dev/PREREG_BRIEF.md` as `[HUMAN 2026-09-02]`.

**Changing a value after runs have been analysed is not a free action.** Anything changed here forces a
re-run of every affected analysis and a labbook entry naming what changed and why. Changing a threshold
*because* a result under the old one was unwelcome is exactly the failure mode the pre-registration
exists to prevent.

---

## A. Framing

| # | Item | AI proposal | Human decision |
|---|---|---|---|
| A1 | Research question wording | Verbatim as in the master prompt §0; never rephrased to presuppose a shared Fourier algorithm | |
| A2 | Primary hypothesis | H3 (does top-k Fourier concentration misclassify a harmonic representation as less structured?) | |
| A3 | Positioning of H1 and the MLP half of H2 | **Replication** in these architectures, not discovery | |
| A4 | What may never be claimed without the gate | "the same circuit", "the same function", "proves", "clearly shows", "Transformers fundamentally…" | |

## B. Metrics and definitions

| # | Item | AI proposal | Human decision |
|---|---|---|---|
| B1 | Structured neuron | alive; `u_a` and `u_b` share the dominant frequency `k`; the family fraction (k plus its aliased odd harmonics up to 7) of each is ≥ **0.50**; the output curve's dominant frequency is also `k` | |
| B2 | Sensitivity variants (always reported alongside) | family threshold 0.30 and 0.70; the top-1 variant; Doshi-style IPR ranking; Swaroop's periodicity score with his 12 / 5 cuts | |
| B3 | Key-frequency rule (primary) | Nanda's: DFT of the neuron→logit map, keep `k` with norm ≥ **0.25 × max** | |
| B4 | Key-frequency sensitivity thresholds | 0.10 and 0.50, both reported | |
| B5 | Secondary key-frequency rules reported alongside | `neuron_clusters`, `embedding_threshold` (uncapped 90 %), `logit_sum_directions`; `embedding_top8` only as the legacy number | |
| B6 | Harmonic cap | odd harmonics `j ≤ 7` (uncapped, the family covers every frequency and the metric is degenerate) | |
| B7 | H3 controls | cardinality-matched top-m **and** a random-set null with 50 seeded draws; the discriminating statistic is the odd-vs-even harmonic *shape*, never the concentration | |
| B8 | Transition thresholds | primary train ≥ 0.99 / test ≥ 0.95; sensitivity (0.98, 0.90) and (1.00, 0.99) | |
| B9 | Convergence-at-budget rule | a metric counts as converged only if it changes by < 5 % between the step-20 000 and step-25 000 checkpoints; otherwise it is "final (budget), not converged" | |

## C. Evidence gate and decision tree

| # | Item | AI proposal | Human decision |
|---|---|---|---|
| C1 | Gate criteria per architecture | G1 periodic structure, G2 phase addition beating the permutation null, G3 end-to-end logit fit, G4 causal necessity **and** sufficiency | |
| C2 | Gate pass rule | an architecture passes if **≥ 8 of 10 seeds** pass all four criteria | |
| C3 | G1 numeric rule | structured fraction ≥ 0.25 of live neurons **and** median family fraction ≥ 0.50 | |
| C4 | G2 numeric rule | resultant length R > the permutation null's 95th percentile **and** R ≥ 0.5 | |
| C5 | G3 numeric rule | fitted-logit argmax accuracy ≥ 0.90 on test cells, R² above the random-frequency control's 95th percentile, and the `(a−b)` control formula below half that R² | |
| C6 | G4 numeric rule | structured and key-frequency ablations each drop test accuracy by ≥ 0.5 with the size-matched random control below 0.1 (z ≥ 3); keeping only the structured set retains ≥ 0.9 | |
| C7 | Decision tree | both pass → investigate H3; one passes → bounded analysis of the other; neither passes → report that the Fourier evidence tested does not identify the mechanisms and investigate metric validity | |
| C8 | Permitted H3 conclusion | "consistent with the same Fourier principle expressed through different internal representations" — never "the same circuit in different bases" | |

## D. Ablations and statistics

| # | Item | AI proposal | Human decision |
|---|---|---|---|
| D1 | Ablation scope | the MLP and transformer tables of `docs/CAUSAL_ABLATION_PLAN.md`, each with a size-matched random control of 50 seeded draws, on unmodified checkpoints, at both measurement points | |
| D2 | Additional causal sweep | Doshi's cumulative IPR-ranked pruning in both directions, plus the random control the paper does not have | |
| D3 | Statistics | paired by seed; percentile bootstrap CI with 10 000 resamples, seed 0; exact sign test and exact Wilcoxon; Cohen's d_z and Cliff's delta; all seeds plotted; p-values never alone | |
| D4 | Failed runs | never silently dropped; n started and n completed always reported, with a sensitivity analysis | |
| D5 | **Which transformer ablation is G4's "remove_key_freqs"** | `PREREGISTRATION.md` §5 names the criterion but not the transformer's id, and the transformer has three candidates. The AI wired `remove_key_subspace_from_residual`. **This choice decides the criterion:** at the final checkpoint, over 10 seeds, `remove_key_subspace_from_residual` gives a median drop of 0.235 (control 0.000) → necessary **0/10**, while `remove_key_freqs_from_embedding` gives 0.984 (control 0.0005) → necessary **10/10**. The overall gate verdict is unchanged either way (both architectures fail on `remove_structured`), but the reported reason is not. Deliberately **not** switched after seeing the numbers (§3.9). | |
| D6 | **The structured-neuron threshold makes G4 untestable — and no available definition fixes it** | Under the primary definition (B1, family fraction ≥ 0.50) the structured set is 442/512 neurons in the MLP and 501/512 in the transformer — all 512 for transformer seed 5. At that size the size-matched control does comparable damage by construction, so neither necessity nor sufficiency can discriminate, and G4 reads 0/10 for both architectures. This is a property of the threshold, not a measurement of the models. Changing B1 would force a re-run of every ablation and a labbook entry; it must not be changed *because* this result is inconvenient. **Measured 2026-09-04: tightening the threshold does not help.** Median structured fraction of live neurons at the final checkpoint, all 10 primary seeds:

| definition | transformer | MLP |
|---|---|---|
| family ≥ 0.30 (sensitivity) | 0.984 (~504/512) | 0.920 (~471/512) |
| **family ≥ 0.50 (primary)** | **0.982 (~503)** | **0.885 (~453)** |
| family ≥ 0.70 (sensitivity) | 0.979 (~501) | 0.863 (~442) |
| top-1 ≥ 0.50 | 0.981 (~502) | 0.880 (~450) |
| Doshi IPR, rank-matched | 0.982 (~503) | 0.885 (~453) |
| Swaroop periodicity > 12 | 0.998 (~511) | 0.888 (~454) |

**Every** definition the pre-registration offers — including both sensitivity variants and both published alternatives — selects 86–100 % of the live neurons. Moving from 0.50 to 0.70 shifts the MLP from ~453 to ~442 of 512. So G4's failure is not an artifact of *this* threshold: at `p = 113` with `d_mlp = 512`, no available structured-neuron definition yields a set small enough for a size-matched ablation to discriminate. A decision to change B1 would therefore not rescue G4; what would is a different *kind* of definition (for example a fixed small cardinality, or selection by causal contribution rather than by spectral shape), which is a new pre-registration, not a threshold tweak. **One further fact bears on this decision (measured 2026-09-04):** the MLP's `structured_fraction_of_live` has **not converged at the 25,000-step budget** — it is settled (relative change < 5 % between steps 20,000 and 25,000) in only **2 of 10** MLP seeds and is still rising, while the transformer's is settled in 10 of 10. Every fraction in the table above is therefore a budget-fixed value, and a longer budget would be expected to move the MLP's *upward*, i.e. to make the sets even larger and G4 even less testable, not more. See `docs/LIMITATIONS.md` B3. | |

| D7 | **Graded structured-neuron ablation — a new pre-registration, not a threshold change** | D6 concludes that no available B1 threshold makes G4 testable, and that only *a different kind of selection* could. An external reviewer raised the identical objection independently on 2026-09-05: the G4 ablation removes 86-100 % of the live neurons, so it cannot separate "the structured neurons matter" from "too much was removed". The AI therefore pre-registered `PREREGISTRATION.md` **§14**, written before any graded result was inspected: rank the live neurons by `s = min(u_a family fraction, u_b family fraction)` with B1's categorical failures demoted, ablate the top 1/2/5/10/25/50 % against **50 seeded size-matched random groups**, in both the remove and keep-only directions, over the 10 primary seeds and both checkpoints. Discriminable at `f` iff in ≥ 8/10 seeds the observed drop exceeds all 50 controls and `z ≥ 3`. **This changes no existing number, no gate criterion and no verdict** — the gate stands at `neither_passes`, and §14.5 fixes in advance that no outcome here may reopen G4. It is additive and separately reported. The human authors decide whether the addendum is accepted as pre-registered, whether the score and the 8/10 rule are the right ones, and how the result is positioned in the write-up. | |

## E. Process

| # | Item | AI proposal | Human decision |
|---|---|---|---|
| E1 | Pilot | the seed-0 pair validates the pipeline only; no threshold is tuned on it | |
| E2 | Freeze | analysis code frozen at a named commit after the pilot; later changes logged and force a re-run | |
| E3 | Explorer | stays frozen until the analyses are frozen and human-reviewed | |
| E4 | Two-hot control | keeps `d_mlp = 512`, p = 113, our split and budget, so only the input parametrization differs from the shared-embedding MLP; labelled "our two-hot variant of Swaroop's setup", never a replication | |
| E5 | Legacy broad-mask numbers | kept in the documentation as a named variant, never described as a reproduction of Nanda et al. | |
| E6 | The multiplication runs | remain reserved for the human author's own evaluation; the AI does not analyse them | |

## F. Division of labour and own work — **[HUMAN AUTHORS MUST COMPLETE]**

| # | Item | To be filled in by the human authors |
|---|---|---|
| F1 | Which code each author wrote, read or changed | |
| F2 | Which derivations each author checked independently | |
| F3 | Which primary sources each author read | |
| F4 | Which experiment each author designed, ran and evaluated themselves | |
| F5 | Who writes the final interpretation, and confirmation it is in their own words | |

---

## Sign-off

| | |
|---|---|
| Name | |
| Date | |
| Items changed from the AI proposal | |
| Reason for each change | |
