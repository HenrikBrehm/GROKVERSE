# Graded structured-neuron ablation

Run 2026-09-06. Pre-registered in `docs/PREREGISTRATION.md` §14; human approval open as `docs/HUMAN_DECISIONS.md` **D7**.

## Question

The evidence gate's criterion G4 ablates the structured-neuron set, which under definition B1 is 86–100 % of the live neurons. Removing that much of any network destroys it, so a size-matched random control does comparable damage by construction and G4 can discriminate nothing. That is recorded as **D6**, and an external reviewer raised the identical objection independently on 2026-09-05.

**Is there a *small* group of structured neurons that is causally load-bearing — and if so, how small?**

## Hypothesis

D6 concluded that no available B1 threshold rescues G4, because every definition the pre-registration offers selects 86–100 % of the live neurons. What could rescue it is a different *kind* of selection: not a threshold but a **ranking**, cut at small cardinalities. If the structured neurons carry the mechanism, a small top-ranked group should beat equally large random groups.

## Configuration

- runs: the 20 primary runs — `mlp_add_p113_wd1.0_frac0.3_seed{0..9}_arch25k` and `txf_add_p113_wd1.0_frac0.3_seed{0..9}_arch25k`
- checkpoints: both pre-registered measurement points (crossing, final)
- task: `(a + b) mod 113`, `d_mlp = 512`, train fraction 0.3, weight decay 1.0, 25,000 steps
- ranking score: `s = min(u_a family fraction, u_b family fraction)`, with B1's categorical conditions (`same_ab_out`) demoted to −1; dead neurons never ranked
- sensitivity score: mean of the three family fractions, no demotion
- fractions: 0.01, 0.02, 0.05, 0.10, 0.25, 0.50 of the live neurons
- directions: `remove_top` (necessity) and `keep_only_top` (sufficiency)
- control: 50 seeded size-matched random groups of live neurons, seed 0, shared between the two directions so the readings are paired
- decision rule, fixed before measurement: discriminable at *f* iff in ≥ 8 of 10 seeds the observed drop exceeds **every** control and z ≥ 3
- no retraining; unmodified checkpoints

## Procedure

An additive block `graded_structured_ablation` in `training/grokverse/analysis/causal_ablation.py` (`MODULE_VERSION` 1.2). Verified additive against a backup of the previous artifacts: every pre-existing observed value is unchanged. All 40 run-checkpoints re-run through the driver, **40 ok / 0 failed**. Aggregated by `training/analyse_graded.py`.

Because the score puts every B1 member at ≥ 0.50 and every non-member strictly below, the top-*n* sets are **nested subsets of the B1 structured set**, so the sweep contains the existing G4 test as its limiting case. This is recorded per fraction as `subset_of_b1_structured` rather than assumed, and held at every fraction in every run.

## Results

Measured, from `training/results/graded_ablation.json` and `training/results/GRADED_ABLATION.md`.

| arch | checkpoint | smallest discriminating fraction | drop vs control at that fraction | seeds |
|---|---|---|---|---|
| MLP | crossing | **1 %** (5 neurons) | 0.187 vs 0.010 | 10/10 |
| transformer | crossing | **1 %** (5 neurons) | 0.031 vs 0.004 | 9/10 |
| MLP | final | **50 %** (256 neurons) | 0.412 vs 0.052 | 9/10 |
| transformer | final | **5 %** (26 neurons) | 0.096 vs 0.0002 | 9/10 |

At the crossing checkpoint the baseline test accuracy is already ≈ 0.95 and the B1 set is far smaller (median 142 of 512 neurons in the MLP, 296 in the transformer) than at the end of training (453 and 503).

At the final checkpoint the MLP's top 1, 2, 5 and 10 % groups change test accuracy by **exactly 0.0000**.

The pre-registered sensitivity score reproduces the pattern. The IPR-ranked sweep that already existed (D2, Doshi arXiv:2310.13061), read at matching fractions with **no new computation**, agrees qualitatively: discriminating from 5 % at crossing in both architectures, and at the final checkpoint from 25 % in the MLP against 5 % in the transformer.

## Interpretation

*Measured fact:* a nested subset of the B1 structured set — five neurons — beats every one of fifty random five-neuron groups at the transition, in both architectures.

*Interpretation:* G4's failure was a property of the **set size**, as D6 suspected, and not evidence that the structured neurons are causally inert. The two architectures diverge at convergence: the MLP's final state is redundant enough that no group below half the network is distinguishable from random, while the transformer retains a small load-bearing core.

*What this does not show:* it does **not** reopen the gate. G4's verdict is fixed by its own pre-registered rule and stands at 0/10 in both architectures; §14.5 fixed that before the numbers were seen. It also does not show that the B1 *set* is the mechanism — only that the *ranking* carries causal signal at small group size.

## Limitations

- **Statistical separation is not importance.** The MLP's 25 % group at the final checkpoint separates from its control in 7 of 10 seeds while costing 0.002 of accuracy. The pre-registered rule asks only whether the observed damage beats every control.
- The crossing checkpoint is a single time point per run, not a trajectory; "at the transition" here means "at that checkpoint".
- The MLP's structured fraction has not converged at the 25,000-step budget (settled in 2 of 10 seeds), so its final-checkpoint reading is budget-fixed. See `docs/LIMITATIONS.md` B3.
- Human approval of the addendum is open as **D7**.

## Related

[[Primary architecture run matrix]] · [[Nondeterministic control draws from a Python set]] · [[Grokking measurement pitfalls]]
