# Silent extractors that return a plausible number

Solved 2026-09-04 during the GROKVERSE architecture study. Three defects of one shape, found weeks
apart, none of which crashed or failed a test. Source of truth:
`Claude/grokverse/docs/LABBOOK.md` entries 58, 99, 101.

## Problem

An extractor produces **well-formed output containing no information**. The pipeline runs green, the
tables have the right column names and the right number of rows, and every consumer downstream
faithfully reports the emptiness as an absence of evidence.

## Symptoms

Three instances, in increasing order of danger:

| # | where | what it returned | how visible |
|---|---|---|---|
| 1 | `aggregate._wave_fitting` guessed flat key names that did not exist | every waveform column `null` | the H2 figure would have been blank |
| 2 | `aggregate._progress_measures` read `protocols[p]["restricted"]["loss"]` when the payload nests a **split** in between | every restricted/excluded-loss column `null` | a table printed with headers and no rows |
| 3 | `structure_over_time` compared `best_by_aic` (an **integer index**) to the string `"square"` | both waveform trajectories constant **0.0** | *invisible* — 0.0 is a plausible waveform share |

The third is the one worth remembering. In a codebase where `null` legitimately means "not evaluable",
a silent `null` is invisible. A silent `0.0` is worse than invisible, because it is quietly
**persuasive**: downstream, an onset detector reporting "no onset" for a flat-zero series looks like a
finding, and it was reported as one for the whole study.

## Cause

- #1 and #2: the extractor guessed the shape of a payload it did not own, and a wrong guess returns
  `None` rather than raising.
- #3: `wave_fitting` uses **two conventions for the same field**. The batched `fit_matrix` stores
  `aic.argmin(axis=0)` — an integer index into `MODEL_NAMES` — while the single-curve `fit_curve` path
  stores the model *name*. Comparing an integer array to a string is not an error in numpy; it is
  elementwise-`False`.

## Solution

Fix the reads, and then add the two things that actually catch this class:

1. **An invariant the empty case cannot satisfy.** The four waveform shares must sum to exactly 1. No
   table of zeros can pass. The pre-existing test asserted only `square + sinusoid <= 1`, which zeros
   pass comfortably — a test can cover a line and still not constrain it.
2. **Cross-module agreement at a shared checkpoint.** Two modules computing the same quantity at step
   25,000 must agree. This is what exposed 0.0 against 0.4023.

Also added: `training/tests/check_results_numbers.py`, which re-derives every number quoted in
`RESULTS.md` from the stored artifacts, so the prose gets the same treatment as the code.

## How to prevent it in the future

- **Never let one field carry two conventions.** If it must, resolve through the shared vocabulary
  (`MODEL_NAMES.index(...)`) and accept both, rather than assuming one.
- **Prefer an invariant over an assertion about a value.** "Sums to 1", "denominators match", "these two
  modules agree" survive refactors that value-assertions do not.
- **Treat exact agreement across independent seeds as a bug signal.** The trigger for finding #3 was a
  convergence table showing *exactly* 0.0000 % change on all 20 runs. Independent seeds do not agree
  exactly.
- **Go looking for a specific number.** All three were found because someone wanted to quote a
  particular value and it was not there. Nothing else found them.
- **Check the checker on a case whose answer you already know.** The first version of the related
  figure-provenance audit reported 11 spurious failures because it resolved recorded paths against the
  wrong base directory.

## Consequence for the study

None of the three changed a reported result — verified, not assumed: after fixing #2 and #3 every
downstream report was re-run into a scratch directory and diffed byte-for-byte. But #3 was *hiding* a
real result. The recovered trajectory shows the two architectures starting indistinguishable in waveform
composition and separating during the memorization plateau (square-best-fit share 0.787 → 0.116 in the
transformer, 0.789 → 0.421 in the MLP), which is now `RESULTS.md` §8.1.

## Related

[[Grokking measurement pitfalls]] · [[Primary architecture run matrix]] · [[2026-09-04 - GROKVERSE]]
