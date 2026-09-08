# S1 convergence-extension block — measured findings

**Status: measurements only.** Interpretation, RESULTS.md and LIMITATIONS.md updates remain with
the student authors per `docs/HUMAN_DECISIONS.md` §F and `CONVERGENCE_RUNBOOK_S1.md` §0.2.

## 1. What ran

| item | value |
|---|---|
| block | S1 (`--study conv100k`), 20 runs: seeds 0–9 × {transformer, mlp} |
| steps | 100,000 (4× the primary block's 25,000) |
| config | frozen `arch25k` configuration; differing fields vs the primary block: `steps`, `study`, and the extended `checkpoint_grid` (see §5) |
| code | branch `arch-study-convergence`, commit `42dd79a49225abe33f1b4f144acb98ec26d47f69` (local only, not pushed) |
| compute | AWS EC2 `c7i.16xlarge` (64 vCPU), eu-central-1, 20 single-threaded runs in parallel |
| environment | Linux x86_64 (AL2023), Python 3.12.14, torch 2.12.1+cpu, numpy 2.4.6 |
| primary-block environment, for comparison | Windows 11 AMD64, Python 3.12.10, torch 2.12.1+cpu, numpy 2.4.6 |
| wall clock | ~10.75 h (2026-09-05 ≈10:07 → 20:50 UTC) |
| outcome | 20/20 completed, `failed_runs=0`, zero analysis warnings/tracebacks |
| artifacts | `s3://<bucket>/grokverse/conv100k/` — runs (full checkpoint history), results, logs, status |
| gate | `test_core.py` ALL CHECKS PASSED on the instance before training |

All 20 manifests record `git_commit=42dd79a…`, `steps_completed=100000`, `status=completed`.

## 2. Headline measurement — LIMITATIONS.md B3

`structured_fraction_of_live` (primary definition), median over 10 seeds, at each block's final
checkpoint:

| architecture | 25k block (final) | S1 @ 100k (final) | change |
|---|---|---|---|
| transformer | 0.9824 | 1.0000 | +0.0176 |
| MLP | 0.8848 | 1.0000 | +0.1152 |
| **gap (txf − mlp)** | **+0.0977** | **+0.0000** | **−0.0977** |

The 25k column is the existing `decision_tree_final.json` (commit `bb4c7c0`, point=final).
The 100k column is `decision_tree.json` from `aggregate_conv100k` (commit `42dd79a`, point=final).
B3 predicted a longer budget would shrink the gap; at 100k the median gap is zero.

Note: the runbook quotes the 25k gap as +0.085; the medians above give +0.0977. The +0.085
figure should be reconciled against its original source (likely a mean or a different
measurement point) before either number is quoted.

## 3. Per-seed values

`structured_fraction_of_live` at steps 25,000 and 100,000 *within the S1 runs*, plus test
accuracy at 100k. (The sf@25k column here is measured on the S1 runs' own step-25000
checkpoints; it differs from the primary block's values at the third decimal — see §6,
platform drift.)

| run | sf@25k | sf@100k | test acc @100k | n_structured/n_live @100k |
|---|---|---|---|---|
| mlp seed0 | 0.8691 | 1.0000 | 1.0000 | 512/512 |
| mlp seed1 | 0.8828 | 0.9941 | 1.0000 | 509/512 |
| mlp seed2 | 0.8887 | 0.9141 | 1.0000 | 468/512 |
| mlp seed3 | 0.9551 | 0.9922 | 1.0000 | 508/512 |
| mlp seed4 | 0.9336 | 1.0000 | 1.0000 | 512/512 |
| mlp seed5 | 0.8633 | 0.9902 | 1.0000 | 507/512 |
| mlp seed6 | 0.8574 | 1.0000 | 1.0000 | 512/512 |
| mlp seed7 | 0.8672 | 1.0000 | 1.0000 | 512/512 |
| mlp seed8 | 0.8828 | 1.0000 | 1.0000 | 512/512 |
| mlp seed9 | 0.8496 | 1.0000 | 1.0000 | 512/512 |
| txf seed0 | 1.0000 | 1.0000 | 1.0000 | 512/512 |
| txf seed1 | 0.9570 | 1.0000 | 1.0000 | 512/512 |
| txf seed2 | 0.9844 | **0.8047** | 0.9998 | 412/512 |
| txf seed3 | 0.9707 | 0.9824 | 1.0000 | 503/512 |
| txf seed4 | 0.9941 | 1.0000 | 1.0000 | 512/512 |
| txf seed5 | 0.9980 | 1.0000 | 0.9998 | 512/512 |
| txf seed6 | 1.0000 | 0.9980 | 1.0000 | 511/512 |
| txf seed7 | 0.9746 | 1.0000 | 1.0000 | 512/512 |
| txf seed8 | 0.9766 | 0.9980 | 0.9992 | 511/512 |
| txf seed9 | 1.0000 | 1.0000 | 0.9998 | 512/512 |

All 20 runs remain generalized at 100k (min test acc 0.9992). No de-grokking observed.

The transformer's cross-seed spread *widened* at 100k: its minimum fell from 0.9648 (25k
block) to 0.8047 (txf seed2, whose fraction moved 0.877→0.805 between steps 90k and 100k and
is still moving). The MLP's floor rose from 0.8672 to 0.9141.

## 4. Convergence assessment (HUMAN_DECISIONS.md B9)

Rule as pre-specified: a metric counts as settled if it changes < 5 % between the last two
checkpoints (here 90,000 → 100,000). Stopping was fixed at 100,000 steps before the run;
no early stopping on any metric.

| metric | architecture | settled at 25k (B3, as documented) | settled at 100k (S1) |
|---|---|---|---|
| structured_fraction_of_live | MLP | 2/10 | **10/10** |
| structured_fraction_of_live | transformer | 10/10 | 9/10 (txf seed2 at 8.24 %) |
| phase_relation_R | MLP | — | 10/10 |
| phase_relation_R | transformer | — | 10/10 |

The MLP, unsettled in 8/10 seeds at 25k, is settled in all 10 at 100k. The single unsettled
metric-seed pair in the whole block is now on the transformer side (seed 2).

## 5. Declared deviations

1. **`CHECKPOINT_GRID` extended** (commit `42dd79a`). The pre-registered grid ended at 25,000;
   entries `30000…100000` were added, fixed before the block ran. Additive only: entries
   ≤ 25000 are unchanged, so a 25,000-step run still selects exactly the same 19 steps.
   Without this the block could not measure anything past 25k: checkpoints drive
   `structure_over_time`, and the B9 rule reads the last two checkpoints. This is a deviation
   from a pre-registered constant and belongs in `AI_DISCLOSURE.md` alongside the external-compute
   row (the runbook's §6a row says `--steps` was the only change; that is superseded — the
   changes are `steps`, `study`, and the grid).
2. **Decision-tree verdicts at 100k** (context for G-gates): G1/G2/G3 pass 10/10 for both
   architectures, G4 passes 0/10 for both — same pass/fail pattern as the 25k block.

## 6. Caveats and bookkeeping

* **Platform drift.** S1 ran on Linux x86_64; the primary block on Windows AMD64 (same torch
  2.12.1+cpu, numpy 2.4.6). The S1 runs' own step-25000 values differ from the primary block's
  at the third decimal (e.g. txf median 0.9893 vs 0.9824). Per runbook §0.3 the students should
  reproduce at least one seed pair on their own hardware before any S1 number is published.
* **Results-file collision, repaired.** The analysis chain writes reports to fixed filenames in
  `results/`; on the instance this overwrote five top-level files that in the repo denote the
  25k block (`analysis_driver.json`, `decision_tree.json`, `h3_report.json`, `h4_report.json`,
  `statistics.json`). In the S3 bundle the S1 versions were moved to
  `results/conv100k_reports/` and the originals restored at the top level (verified by
  `analysis_git_commit`). The local repo was never affected.
* **`run_manifest.csv` does not index the S1 runs.** It was not regenerated; the 20-run S1
  index is `results/aggregate_conv100k/index.json`.
* Bucket has versioning enabled and all public access blocked. No secrets in the repo or the
  runner; the instance used a scoped IAM role (single bucket, tag-conditioned self-terminate).
* The instance self-terminated after upload; verified nothing left running.

## 7. What S1 does not settle (per runbook §7)

G4's size-matched ablation is *less* discriminating at 100k than at 25k: the structured set is
now ~100 % of live neurons for both architectures, so there is no unstructured complement to
contrast against. The ranked graded ablation (S2) and frequency-family comparison (S3) are the
specified follow-ups; S1 only closes the convergence question.
