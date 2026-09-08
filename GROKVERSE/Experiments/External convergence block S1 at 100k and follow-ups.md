# External convergence block S1 at 100k and follow-ups

Run 2026-09-05 (S1) and 2026-09-07 (two-hot ×10) by the external reviewer on the reviewer's own AWS account; received 2026-09-08; verified and reported here as `RESULTS.md` §16. Acceptance open as `docs/HUMAN_DECISIONS.md` **D9**.

## Question

`docs/LIMITATIONS.md` B3: the MLP's `structured_fraction_of_live` was settled in only 2 of 10 seeds at the 25,000-step budget and still rising, so the headline architecture gap (+0.0977 as a difference of medians, +0.0850 as the paired median of `RESULTS.md` §7) was a budget-fixed number with a predicted direction: **a longer budget should shrink it.** Does it?

## Configuration

- runs: 20 — `mlp_add_p113_wd1.0_frac0.3_seed{0..9}_conv100k`, `txf_add_p113_wd1.0_frac0.3_seed{0..9}_conv100k`
- the frozen `arch25k` flags, except `--steps 100000` and `--study conv100k`
- code: commit `42dd79a` on branch `arch-study-convergence` = our `fda066e` + ten `CHECKPOINT_GRID` entries above 25,000 (30k, 35k, 40k, 45k, 50k, 60k, 70k, 80k, 90k, 100k) + the runbook; verified with `git diff --stat` on the bundled source tree — no other change
- stopping rule fixed before the run: 100,000 steps, no early stopping on any metric; convergence assessed afterwards with B9 (< 5 % change between the last two checkpoints)
- compute: AWS EC2 `c7i.16xlarge`, 20 single-threaded runs in parallel, ~10.75 h; Linux x86_64, Python 3.12.14, torch 2.12.1+cpu, numpy 2.4.6
- two-hot control: 10 runs `m2h_…_seed{0..9}_arch25k`, flags byte-identical to our `matrix_twohot_20260904T022540Z.json` invocation

## Procedure

Bundle `grokverse_run_data.zip` (1.33 GB, SHA-256 `c6e532ef…add44`) unpacked into the gitignored `Claude/grokverse/training/external/`; small outputs and the reviewer's four documents copied unmodified into tracked folders (`training/results/external/`, `docs/external/`). Checks run here:

| check | result |
|---|---|
| manifests (30) | all `git_commit=42dd79a`, `status=completed`, Linux, torch 2.12.1+cpu |
| split hashes vs ours, per seed (23) | 0 mismatches |
| checkpoint SHA-256 vs `checkpoints.json` (829 files) | 829 match, 0 missing |
| frozen `decision_tree`, `statistics_report`, `h3_report`, `h4_report` re-run on the bundle's `aggregate_conv100k` | byte-identical to the reviewer's reports apart from timestamps and paths |
| per-seed structured fractions at 25k / 90k / 100k recomputed from `structure_over_time` | every value agrees with the reviewer's table |
| key-frequency counts at 100k from the frozen `key_frequencies` artifacts | agree with the reviewer's S3 counts |
| two-hot seeds 0–2 vs our Windows runs | same split hashes; memorization 230/230/230 vs 230/240/230; generalization 12,575/14,100/12,125 vs 12,775/13,300/12,500 |

**Not checkable:** the reviewer's S2, S2b and S3 analyses — their scripts are not in the bundle. Reported only.

## Results

Measured, from `training/results/external/conv100k/` and the runs.

| quantity | 25,000 steps (ours) | 100,000 steps (S1) |
|---|---|---|
| `structured_fraction_of_live`, median, transformer | 0.9824 | **1.0000** |
| same, MLP | 0.8848 | **1.0000** |
| gap (difference of medians) | +0.0977 | **0.0000** |
| B9 settled, MLP | 2/10 (20k→25k) | **10/10** (90k→100k) |
| B9 settled, transformer | 10/10 | 9/10 — seed 2 fell 0.877 → 0.805 and is still moving |
| gate | `neither_passes` (G4 0/10 both) | `neither_passes` (G4 0/10 both) |
| key frequencies, `nanda`, transformer / MLP | 4.5 / 12 | 4.5 (3–5) / 11 (9–12) |
| `fix_attention_to_mean`, median Δ test acc | −0.336 | −0.415 |
| min final test accuracy, all 20 runs | — | 0.9992 |

Two-hot at 10 seeds: `nanda` selects all 56 frequencies in every seed (flat spectrum; our three did the same); the reviewer's S2 finds a top-192 group that is both necessary and sufficient and beats its random control — unlike either shared-embedding model.

The reviewer's S2/S2b/S3 (reported): MLP LOO k\* necessity 320 vs transformer 128; keep-only-top-k never sufficient in the transformer below the full set; key-frequency-ranked removal harms the transformer from 5 % and the MLP only from 50 %, with disjoint key/non-key neuron sets in the MLP and near-identical ones in the transformer; frequency-set overlap at the size-matched null in both architectures.

## Interpretation

*Measured fact:* at four times the budget both architectures reach a structured fraction of 1.0 and the 25k gap disappears; the gate verdict does not move.

*What it does not show:* nothing about the mechanism. G4 stays at 0/10 for the reason D6 gave — the structured set is now the whole live population, so the size-matched control cannot discriminate. The count difference in key frequencies (≈ 4–5 vs ≈ 11) is what survives the longer budget, plus the reviewer's ranked-ablation asymmetry, which matches our §6.1 under a different ranking.

*Open:* the reviewer's runbook §0.3 requires at least one seed pair reproduced on our hardware before any S1 number is used (MLP ≈ 3.6 h, transformer ≈ 16 h); §0.4 leaves the competition-rules question on external compute to us. Both are in D9.

## Limitations

- Platform drift: the S1 runs' own step-25,000 medians (0.876 MLP, 0.989 transformer, Linux) differ from ours (0.885, 0.982, Windows) at the third decimal with the same seeds.
- The `CHECKPOINT_GRID` extension is a declared deviation from a pre-registered constant; it is on the reviewer's branch only, not merged here.
- S2/S2b/S3 are unreviewed, unreproduced code from outside; requested.
- `run_manifest.csv` in the bundle does not index the S1 runs (`aggregate_conv100k/index.json` does).

## Related

[[Graded structured-neuron ablation]] · [[Primary architecture run matrix]] · [[GROKVERSE Project Log]]
