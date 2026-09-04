# GROKVERSE Session

**2026-09-04.** Autonomous continuation of the architecture study on branch `arch-study`. All analyses
completed, the evidence gate evaluated, and every document rewritten against the measurements.

## What we worked on

Finished the master prompt's §23 Definition of Done for everything that does not require the human
authors: the aggregation and reporting layer, the evidence gate, H3/H4, the bounded
alternative-mechanism analysis, all three control blocks, and then the full documentation rewrite
(`RESULTS.md`, `README.md`, `docs/LIMITATIONS.md` §B, `PROGRESS.md`, the claim–evidence table).

Two silent defects were found and fixed along the way, and one of them turned out to have been hiding a
real result.

## Changes

- `analysis/aggregate.py` — `_progress_measures` fixed to read the split level (it had produced every
  restricted/excluded-loss column as `null`).
- `analysis/structure_over_time.py` — `_fraction_best` resolves the model index through `MODEL_NAMES`;
  all four waveform shares now reported so their sum-to-1 is a checkable invariant.
- `tests/check_results_numbers.py` — **new**: re-derives every number quoted in `RESULTS.md` from
  `results/` and fails on disagreement (~110 checks).
- `RESULTS.md`, `README.md`, `docs/LIMITATIONS.md`, `PROGRESS.md`, `docs/CLAIM_EVIDENCE_TABLE.md`,
  `docs/NOVELTY_AND_RELATED_WORK.md`, `AI_DISCLOSURE.md`, `docs/PREREGISTRATION.md` §12 — rewritten or
  brought up to date against the measurements.
- Test suite: **24 files, 1,913 checks**, green before every commit.

## Experiments

No new training. All 51 runs of the matrix were already complete (primary 20, confound 18,
parameter-matched 10, two-hot 3; none failed) — see [[Primary architecture run matrix]].

`structure_over_time` was re-run over all 51 runs at the corrected code (**51 ok, 0 failed**) after the
index-vs-name defect was fixed. Every other report was re-run into a scratch directory and diffed:
byte-identical apart from timestamps, confirming nothing outside H4 consumed the broken metric.

### Measured results

**The pre-registered evidence gate reports `neither_passes`.** G1, G2 and G3 hold on 10 of 10 seeds for
both architectures; **G4 fails 0/10 in both**, because removing the "structured" neurons is not
separable from removing an equal number of random ones — the definition selects 88–98 % of the network,
so the size-matched control does 0.873 (MLP) and 0.916 (transformer) of the same damage.

**H3, the study's own primary hypothesis, is refuted by its own criterion.** The harmonic-family
definition closes **+0.0049** of a **+0.0898** architecture gap — **5.4 %**. The artifact it postulated
is real (+0.1893 on synthetic populations differing only in waveform) and an order of magnitude too
small to be the explanation.

**Key frequencies are causally load-bearing in both architectures**, unlike the neuron sets: removing
them costs ~0.99 accuracy where removing an equal number of random frequencies costs ~0.000.

**Behaviour and representation come apart completely.** The two architectures agree on **99.98 %** of
all 12,769 inputs (identical in 4 of 10 seeds) while their logits correlate at **0.083** and their top-2
predictions agree 0.6 % of the time. Cross-seed CKA between two seeds of the *same* architecture is
0.0017 (transformer) and 0.0973 (MLP), so representation similarity cannot be cited as evidence about
mechanism sameness in either direction.

**H4 holds, and gained a result when the defect was fixed.** Six structure metrics reach their onset
before the generalization crossing in 10/10 seeds in both architectures, at 5.4 %–39.5 % of the way
there. The recovered waveform trajectory shows the two architectures **starting indistinguishable** and
separating during the plateau: the square-wave-best-fit share falls 0.787 → 0.116 (transformer) and
0.789 → 0.421 (MLP), while the sinusoid share rises to 0.646 and 0.333 respectively — unanimous in sign
across all 10 seeds in both architectures.

**Controls.** `train_frac` dominates Grokfast by an order of magnitude; every structure difference
survives matching parameter count to 0.02 %; but the **two-hot MLP is more square-wave-like (+0.1836)**
than the shared-embedding MLP, so the waveform result is at least partly about input parametrization
rather than architecture.

## Learnings

Recorded in [[Grokking measurement pitfalls]] as a new section: **the silent-extraction defect class**,
where an extractor returns well-formed output containing no information. What catches it is not more
unit tests but (i) an invariant the empty case cannot satisfy, (ii) cross-module agreement at a shared
checkpoint, (iii) suspicion of exact zeros across independent seeds, and (iv) cross-checking the prose
against the artifacts.

Also measured there: Khanh's at-transition overstatement, confirmed on our own data — the legacy
0.73-vs-0.44 concentration gap becomes **0.961 vs 0.891** when read at convergence instead of at the
crossing.

## Problems

1. `aggregate._progress_measures` produced every restricted/excluded-loss column as `null`.
2. `structure_over_time` returned **0.0** — not `null` — for both waveform trajectories, for every run,
   for the whole study.
3. The convergence check that `LIMITATIONS.md` §A6 promised had never actually been executed.
4. `docs/NOVELTY_AND_RELATED_WORK.md` still quoted retired numbers from the superseded `RESULTS.md`.

## Solutions

1. Read the split level; added a regression test pinning the nesting depth and the split labels.
2. Resolve the model index through `MODEL_NAMES`, accept both conventions, report all four shares so
   their sum-to-1 is testable. Re-ran all 51 runs.
3. Executed it. It found the MLP's headline metric unsettled at the budget in 8 of 10 seeds — now
   `LIMITATIONS.md` §B3 and `RESULTS.md` §5.1.
4. Re-measured at convergence and replaced the numbers in both documents.

## Decisions

- **Do not change the onset rule** even though it is rise-only and therefore blind to the square share's
  fall. It is pre-registered; the limitation is reported and the pre-registered `change` statistic is
  used instead.
- **Do not change the structured-neuron threshold** that makes G4 untestable. Escalated to the human
  authors as **D6**, with a measured demonstration that no available definition fixes it.
- **Do not switch which transformer ablation G4 refers to**, though the choice flips that criterion from
  0/10 to 10/10. Escalated as **D5**; deliberately not changed after seeing the numbers.
- **Report the gate failure and the refutation as the results**, per master prompt §21, rather than
  reframing them.

## Open Tasks

- [ ] Human authors: the final scientific interpretation in their own words
      (`docs/HUMAN_INTERPRETATION_TEMPLATE.md`)
- [ ] Human authors: decisions A–F in `docs/HUMAN_DECISIONS.md`, including **D5** and **D6**
- [ ] Human authors: the 11 `[HUMAN AUTHORS MUST COMPLETE]` placeholders in `AI_DISCLOSURE.md`
- [ ] Human authors: the `txf_mul_*` runs (**E6**, reserved)
- [ ] Explorer update per `docs/dev/EXPLORER_UPDATE_PLAN.md` (**E3**, `web/` deliberately untouched)

## Related Notes

[[Primary architecture run matrix]] · [[Grokking measurement pitfalls]] ·
[[2026-09-02 Architecture study design decisions]] · [[ReLU as the multiplier in modular addition]] ·
[[2026-09-03 - GROKVERSE]]
