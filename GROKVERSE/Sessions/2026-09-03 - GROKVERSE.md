# GROKVERSE Session

Date: 2026-09-02 into 2026-09-03. Branch `arch-study`. All work by Claude Code unless marked otherwise.

## What we worked on

Executing `GROKVERSE_MASTER_PROMPT_EN.md`: turning the reproduction-plus-explorer project into a
methodologically sound mechanistic comparison of the transformer against the MLP. This session covered
the master prompt's steps 1–8 of its working order — secure the baseline, audit, pre-registration,
literature, pipeline, derivations — and launched the run matrix.

## Changes

- Baseline secured: tag `baseline/pre-arch-study` at `d9434c1`, working branch `arch-study`, byte-copy
  archive of all 16 legacy runs and 21 figures with a SHA256 manifest. The uncommitted move of
  `grokverse/` and `PREP/` into `Claude/` was recorded as renames rather than deletions.
- New specification documents before any code: `docs/dev/PREREG_BRIEF.md` (hypotheses, nulls, gate,
  definitions, decision tree, statistics), `docs/dev/RUN_FORMAT_V2.md`, `docs/dev/INTERFACES.md`,
  `docs/BASELINE.md`, `docs/LABBOOK.md`.
- Training pipeline extended: dense evaluation, transition intervals for three threshold sets, 21
  pre-specified checkpoints with assigned roles, per-run manifest, aggregator, paired-seed matrix
  launcher, two-hot MLP control. Legacy behaviour and every legacy run id unchanged.
- Seven primary-source notes under `docs/sources/`; three audit documents; both mechanism derivations;
  `AI_DISCLOSURE.md` rewritten.
- New analysis modules (in progress): `common`, `metrics`, `wave_fitting`, `statistics`,
  `function_agreement`, `logit_formula_fit`.

## Experiments

[[Primary architecture run matrix]] — 20 runs at p=113, train fraction 0.3, weight decay 1.0, no
Grokfast, fixed 25,000-step budget, paired seeds 0–9, `threads=1`, commit `d53cf52`. 18 completed, two
transformer seeds still training at the end of the session. Raw crossings and final accuracies are in
that note. **No comparison, no structure metric and no statistic has been computed** — the analysis code
is not yet frozen, which is the pre-registered order.

## Learnings

- [[ReLU as the multiplier in modular addition]] — rectification supplies the product the angle-addition
  identity needs, with amplitude `8/(3 pi^2)` and phase `phi_a + phi_b`; but the `(a-b)` term is exactly
  as large, so only the population readout selects addition.
- [[Grokking measurement pitfalls]] — eight ways a plausible measurement goes wrong here, including
  reading structure at the transition instead of at convergence, a cap that makes a "measured" frequency
  count constant, top-k concentration punishing square waves, an unsatisfiable "family beats top-m"
  criterion, a mean without a variance, a coarse evaluation grid inventing precision, a Fourier mask
  wider than the hypothesis, and publishing a replication as a discovery.

## Problems

1. The AI session hit a usage limit mid-session while two multi-agent workflows were running; 13 agents
   died with that error.
2. One pre-registered prediction was mathematically wrong: that a single rectified circuit neuron shows
   much more `(a+b)` than `(a-b)` structure in its activation map.
3. Under 8-way parallelism a transformer run took ~4 h rather than the ~1.75 h estimated from a
   single-thread benchmark.

## Solutions

1. Nothing already written was rerun. The workflows were resumed so completed agents replayed from cache,
   and only the missing verifications, reviews and unfinished modules were re-launched. The training
   matrix is a local background process and was unaffected; no run was restarted or overwritten.
2. Corrected in the derivation, the interface contract, the implementation brief and a new test, and
   recorded in the labbook — before any run was analysed. The sum-over-difference contrast is now tested
   where it is actually predicted, on the logits and the population.
3. Accepted; the wall-clock cost is recorded in the labbook so future estimates use the contended figure.

## Decisions

[[2026-09-02 Architecture study design decisions]] — fixed 25,000-step budget, ten paired seeds plus the
full control matrix, key frequencies from the neuron-to-logit map, the two-hot control varying only the
input parametrization, the evidence gate, and the analysis-code freeze after a pilot.

## Open Tasks

- [ ] Finish transformer seeds 8–9, then the confound, parameter-matched and two-hot blocks.
- [ ] Complete and review the analysis modules; run the seed-0 pilot; freeze the analysis code.
- [ ] Run all analyses, statistics, the evidence gate and the decision tree on the full matrix.
- [ ] Rewrite `RESULTS.md` and `README.md` to match the actual evidence level.
- [ ] **Human**: approve or change the `[AI-PROPOSED]` pre-registration values in
      `Claude/grokverse/docs/HUMAN_DECISIONS.md`; check the derivations; read the primary sources; write
      the final interpretation in your own words.

## Related Notes

[[Primary architecture run matrix]] · [[2026-09-02 Architecture study design decisions]] ·
[[ReLU as the multiplier in modular addition]] · [[Grokking measurement pitfalls]]


---

## Update — evening of 2026-09-03 (editor crash, recovery, transformer mechanism, handoff)

*Appended, not rewritten. Model: Fable 5.1 for the last part; Opus 5 before the switch. All work by Claude Code.*

### What we worked on

Recovery after a Visual Studio crash mid-wave-2, then the transformer mechanism module (master prompt §8,
INTERFACES §6), then — at the author's request — preparing everything so the next session can continue
unattended. `Claude/grokverse/docs/LABBOOK.md` entries 31–44 are the primary record.

### Changes

- `tests/test_mask_protocols.py`: two defects fixed (a check asserting `acc == 1/p` on annihilated logits,
  and a masked `IndexError` behind it). 197 checks.
- New `analysis/transformer_mechanism.py` + `tests/test_transformer_mechanism.py` (67 checks): exact forward
  decomposition (2.8e-13 vs the float64 model), mean-attention effective curves with `additivity_r2`,
  attention report, causal head ablation, key-subspace variance with a size-matched random null,
  direction spectra. Every §5 battery reused, none re-implemented.
- `mlp_mechanism` / `mlp_mechanism_activation`: optional `act=` and a `[p,p,p]` bias term so the
  transformer's *true* hidden layer runs through the same code paths as the MLP.
- `key_frequencies.operand_curves`: transformer placeholder replaced by the real effective curves.
- `training/run_analysis_chain.ps1` (detached driver launcher), `docs/dev/HANDOFF_2026-09-04.md`,
  INTERFACES §6 amendment, `HUMAN_DECISIONS.md` restored after an accidental delete.
- Machine: standby/hibernate disabled (`powercfg`) — the cause of the 7.6 h overnight stall.

### Experiments

None. One pipeline check only: `transformer_mechanism.analyse` on transformer seed 0, step 25 000
(29 s; forward check exact; logit-variance shares sum to 1). Produced by unfrozen code — **not a result**.
Training: primary block 20/20 complete; `confound` 8/18 running since 17:17 UTC; `param_matched`,
`twohot` queued. ETA all blocks ≈ midday 2026-09-04.

### Learnings

- The transformer's logits split exactly as `direct_path + Σ_f h_f (W_out W_U)[f]` — the same additive
  law as the MLP, which is why one `logit_contributions` serves both. But its pre-activation is only
  *approximately* additive in the operand curves; the MLP's is exactly so. `additivity_r2` is the
  asymmetry, and it must never be compared as if symmetric.
- When a test fails, ask first whether the *check* is wrong: three times this week the module was right
  (entries 24, 26, 33).

### Problems / Solutions

- Crash killed the training chain → restarted; zero completed runs lost, ~10 h CPU lost; verified by
  manifest and parent PIDs (the doubled process list was a venv shim).
- `h3_validity` is listed by the driver but has no INTERFACES spec → handoff says: write the spec first.

### Decisions

- Store no `[p,p,d_mlp]` tensors in analysis npz files (INTERFACES §6 amendment; master prompt §20).
- Work directly, not via subagents, after repeated orphaned work (labbook 19, 23).
- `HUMAN_DECISIONS.md` restored with status unchanged (option B): the gate labels results, it does not
  block work.

### Open Tasks

- [ ] Stage A–C: `causal_ablation.py`, `h3_validity.py` (spec first), `structure_over_time.py` + tests.
- [ ] Stage D: update `tests/test_driver.py` 89–92 when A/C land.
- [ ] Stage E: freeze (fill PREREGISTRATION §12), launch `run_analysis_chain.ps1` for the primary block,
      then for everything after training ends; re-run the seed-0 pilot outputs.
- [ ] Stage F: aggregate, decision tree, statistics, `CLAIM_EVIDENCE_TABLE.md`, `RESULTS.md`/`README.md`.
- [ ] Stage G: explorer, only after human review.
- [ ] Commit `training/results/` once the training chain has finished.
- [ ] **Human**: `HUMAN_DECISIONS.md` A–E + F + sign-off; `AI_DISCLOSURE.md` placeholders; final
      interpretation in own words; the multiplication runs.

### Related Notes

[[HANDOFF_2026-09-04]] (in `Claude/grokverse/docs/dev/`) · [[Primary architecture run matrix]] ·
[[ReLU as the multiplier in modular addition]] · [[Mechanistic Interpretability]]
