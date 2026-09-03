# Primary architecture run matrix (arch25k)

## Question

Under an identical protocol, how do a 1-layer ReLU transformer and a 2-layer shared-embedding ReLU MLP
learn `(a + b) mod 113` — do they use a Fourier-based phase-addition mechanism, and if so do they
represent and compute it differently?

The runs recorded here are the **substrate** for that question. No comparison is drawn in this note: the
paired statistics and every structure metric come only after the analysis code is frozen
(`Claude/grokverse/docs/dev/PREREG_BRIEF.md`).

## Hypothesis

Pre-registered in `docs/PREREGISTRATION.md` (H1–H5) with nulls and refutation criteria, before these runs
were analysed. See [[2026-09-02 Architecture study design decisions]].

## Configuration

- task `(a + b) mod 113`, tokens `[a, b, =]`, full batch
- `train_frac = 0.3`, split from a seeded permutation; paired seeds share the split (verified by hash)
- AdamW, `lr = 1e-3`, `betas = (0.9, 0.98)`, `weight_decay = 1.0`, no Grokfast
- **fixed budget `steps = 25000`, no early stop**
- transformer: `d_model 128`, 4 heads x 32, `d_mlp 512`, `n_ctx 3`, no LayerNorm (226,176 parameters)
- MLP: shared `W_E [113, 128]`, concatenation, `d_mlp 512` (204,017 parameters)
- seeds 0–9 for each architecture; `threads = 1` pinned per run, 8 runs in parallel
- dense evaluation: train every 10 steps, test every 25 steps
- 21 checkpoints per run (pre-specified grid plus event checkpoints at the crossings)
- code commit `d53cf52`; environment Python 3.12.10, torch 2.12.1+cpu, numpy 2.4.6, CPU only

## Procedure

Launched 2026-09-02 16:43 UTC via `python -m grokverse.matrix --block primary --workers 8`, followed by
the `confound`, `param_matched` and `twohot` blocks in the same chained process. Each run writes
`run.json` (dense curves, transitions for three threshold sets), `checkpoints.json`, `manifest.json`
(split hash, versions, platform, parameter counts, weight norms), `embeddings.npy`, `model_final.pt` and
`train.log`.

## Results — raw run outcomes only

**The primary block completed on 2026-09-03 at 06:53 UTC: 20 of 20 runs, none failed.** All 20 manifests
validate; every run has 21 checkpoints with all seven roles assigned; **every paired seed shares its
training split hash**, which is what makes the paired design valid.

Memorization and generalization crossings under the primary thresholds (train accuracy >= 0.99, test
accuracy >= 0.95), from each run's dense evaluation. **Each crossing is the first evaluated step at or
above the threshold; the true crossing lies in the interval since the previous evaluation — 10 steps for
train, 25 for test.**

| seed | txf memorize | txf generalize | txf final test acc | mlp memorize | mlp generalize | mlp final test acc |
|---|---|---|---|---|---|---|
| 0 | 140 | 7975 | 0.9977 | 160 | 9125 | 1.0000 |
| 1 | 140 | 6250 | 0.9994 | 160 | 10075 | 1.0000 |
| 2 | 140 | 8125 | 0.9971 | 160 | 9175 | 1.0000 |
| 3 | 140 | 7425 | 0.9996 | 160 | 8150 | 1.0000 |
| 4 | 150 | 10275 | 1.0000 | 160 | 8650 | 1.0000 |
| 5 | 140 | 5625 | 1.0000 | 160 | 9375 | 1.0000 |
| 6 | 140 | 6325 | 1.0000 | 160 | 9900 | 1.0000 |
| 7 | 140 | 7750 | 1.0000 | 160 | 9200 | 1.0000 |
| 8 | 140 | 8125 | 0.9993 | 160 | 9800 | 1.0000 |
| 9 | 140 | 5825 | 0.9998 | 160 | 9300 | 1.0000 |

Ranges: transformer generalization 5,625–10,275, final test accuracy 0.9971–1.0000; MLP generalization
8,150–10,075, final test accuracy 1.0000 in every seed. Every run reached its full 25,000-step budget.

Wall clock: MLP runs 0.82–0.95 h each. Transformer runs 4.04 h each under 8-way contention, except seeds 8
and 9, whose recorded 10.12 h **spans an overnight machine suspension** and is therefore not a valid
compute measurement (`Claude/grokverse/docs/LABBOOK.md` entry 18).

The `confound` block (18 runs) started immediately afterwards; `param_matched` (10) and `twohot` (3)
are queued behind it.

Artifacts: `Claude/grokverse/training/runs/txf_add_p113_wd1.0_frac0.3_seed*_arch25k/` and
`mlp_add_p113_wd1.0_frac0.3_seed*_arch25k/`; block summary
`training/results/matrix_primary_20260902T164330Z.json`; aggregate manifest via
`python -m grokverse.manifest`.

## Interpretation

**None yet, deliberately.** The paired timing comparison, every structure metric, the evidence gate and
the decision tree are computed only after the analysis code is frozen, and are reported with confidence
intervals and all individual seeds. Note in advance that a timing gap alone is not a contribution —
Manir & Rupa 2026 already report that this gap is protocol-dependent, with the opposite direction under
their protocol.

## Limitations

- One hyperparameter point (p=113, train fraction 0.3, weight decay 1.0), not a sweep.
- Crossings sit on a discrete evaluation grid and are reported as intervals.
- "Final" means the 25,000-step budget; whether a metric has converged there is tested per metric and per
  seed, not assumed.
- The confound, parameter-matched and two-hot blocks were still queued when this note was written.

## Related

[[2026-09-02 Architecture study design decisions]] · [[Grokking measurement pitfalls]] ·
[[ReLU as the multiplier in modular addition]] · [[2026-09-03 - GROKVERSE]]
