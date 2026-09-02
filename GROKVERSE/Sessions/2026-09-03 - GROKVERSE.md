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
