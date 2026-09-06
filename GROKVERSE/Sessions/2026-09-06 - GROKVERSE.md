# GROKVERSE Session

2026-09-06, overnight. Scheduled to start at midnight; ran to ~03:00.

## What we worked on

The follow-up an external reviewer proposed on 2026-09-05 after reading branch `GROKVERSE-MP`: a **graded neuron ablation** on the existing results, ablating small ranked groups instead of the whole structured set. Their objection was that the G4 ablation removes 88–98 % of the network and therefore cannot separate "the structured neurons matter" from "too much was removed" — the same point `HUMAN_DECISIONS.md` D6 already records from inside.

## Changes

- `docs/PREREGISTRATION.md` **§14** — the addendum, written and committed (`7161842`) **before any graded number was read**, with decision **D7** opened and left blank.
- `training/grokverse/analysis/causal_ablation.py` — additive block `graded_structured_ablation`, plus `graded_group_size`, `graded_top_indices`, `resolve_structured_scores`. `MODULE_VERSION` 1.0 → 1.2.
- `training/analyse_graded.py` — aggregation over the primary block, and the zero-cost read of the existing IPR sweep.
- `training/tests/test_causal_ablation.py` — six new checks, written first and red first.
- `RESULTS.md` **§6.1**, `docs/LABBOOK.md` 109, `docs/dev/HANDOFF_PROGRESS.md` stage I.
- Determinism fix and two updated control numbers (below), `HUMAN_DECISIONS.md` **D8**, `PREREGISTRATION.md` §12 post-freeze row.

## Experiments

See [[Graded structured-neuron ablation]] for the full record. Summary:

- purpose: is there a *small* group of structured neurons that is causally load-bearing?
- configuration: 20 primary runs, both checkpoints, fractions 0.01–0.50, 50 seeded size-matched random control groups, seed 0, no retraining
- ranking: `s = min(u_a family fraction, u_b family fraction)` with B1's categorical failures demoted, so top-*n* sets are nested subsets of the B1 set
- result: discriminates from **1 % (5 neurons)** at the crossing point in **both** architectures; at the final checkpoint from **5 %** in the transformer but only **50 %** in the MLP
- driver: 40 ok / 0 failed. Files: `training/results/graded_ablation.json`, `training/results/GRADED_ABLATION.md`

## Learnings

**Check what already exists before building.** Most of what the reviewer proposed was already on disk as `ipr_ranked_pruning` (D2, the Doshi sweep): 21 fractions, both directions, a 50-draw control. Only the ranking *by the structure criterion* and the 1 % grid point were missing. The night's new compute was one additive block, not a new experiment — and the existing IPR curves then served as an independent second ranking for free.

**A control that is not reproducible is not a control.** See [[Nondeterministic control draws from a Python set]].

**Design the score so the new test contains the old one.** Ranking by the exact quantity B1 thresholds means the top-*n* sets are nested subsets of the B1 set, so the graded sweep is a refinement of G4 rather than a different test wearing its name. That property was recorded per fraction, not assumed.

## Problems

1. Re-running `causal_ablation` and diffing against the artifacts on disk showed **560 of 660** pre-existing ablation blocks differing — all in the control statistics, none in the observed values.
2. The IPR extraction silently dropped the 0.05 and 0.10 fractions: the stored grid is cardinality-derived (26/512 = 0.05078), so an exact-equality match found nothing and still produced a well-formed table.
3. A guard written that night, `float(np.nanstd(a)) >= 0.0`, is vacuously true and checked nothing.

## Solutions

1. Cause: `resolve_structured_masks` built the definitions as a Python **set**, whose iteration order depends on per-process string hashing, while both ablation functions consume **one shared rng** in that order. Fixed by iterating the already-ordered `wanted` tuple. Proven across six `PYTHONHASHSEED` values; regression test added.
2. Match the nearest grid point within half a grid step and report the grid fraction actually used.
3. Replaced with "more than one distinct finite value".

## Decisions

- **The graded ablation is a new pre-registration, not a threshold change.** D6 already establishes that no B1 threshold rescues G4; only a different *kind* of selection could. Changing B1 was therefore never on the table — §14 is a new, separately reported analysis that feeds no gate criterion.
- **The determinism fix is a §9 bug fix, not a control change.** Determinism is a stated non-negotiable of the project constitution, and the drift (≈ ±0.005) exceeded the ±0.001 the verifier asserts, so noting it was not an option. Escalated as **D8** so the human authors can disagree.
- **Reported the null half honestly.** The MLP at convergence shows nothing below 50 %, and the 25 % separation costs 0.002 of accuracy. Both are in `RESULTS.md` §6.1 next to the positive result.

## Open Tasks

- [ ] **D7** — do the authors accept §14 as pre-registered, and is the 8/10 rule the right one?
- [ ] **D8** — is the determinism fix accepted as a §9 bug fix rather than a control change, and are the two updated numbers re-checked independently?
- [ ] The collaborator's longer convergence run (their AWS account, branch `arch-study-convergence`) was due 2026-09-06 morning; compare their MLP structured fraction against ours once it lands.
- [ ] Everything still human-only: the final interpretation, `AI_DISCLOSURE.md`'s 11 placeholders, the `txf_mul_*` runs, the explorer update.

## Related Notes

[[Graded structured-neuron ablation]] · [[Nondeterministic control draws from a Python set]] · [[Primary architecture run matrix]] · [[Silent extractors that return a plausible number]] · [[Grokking measurement pitfalls]]
