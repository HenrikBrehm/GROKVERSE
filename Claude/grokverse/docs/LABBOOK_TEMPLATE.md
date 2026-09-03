# Labbook entry — template

The labbook (`docs/LABBOOK.md`) is the study's chronological record. It is **append-only**: entries are
never rewritten. A factual error in an old entry is corrected by a *new* entry that names the old one and
says what was wrong — the original stays.

Numbered entries, newest at the bottom, grouped under a `## YYYY-MM-DD` heading.

---

## The entry structure

```markdown
N. **<Actor>, <what happened in four or five words>.** <Body.>
```

| field | rule |
|---|---|
| **Actor** | `AI` or `Human`, always named. If a decision was the human's, write `Human decision` and the date it was given. |
| **Action** | What was actually done, in the past tense. Not what was intended. |
| **Config / seed / commit** | Whenever a run, an analysis or a code change is involved: the config or preset, the seeds, and the git commit. A number without a commit is not traceable. |
| **Files produced** | Paths, so a reader can check. |
| **Observed vs interpreted** | Measurements and interpretations in separate sentences. An entry may record a measurement; it may not silently promote one to a conclusion. |
| **Decisions** | What was decided and *why*, including alternatives rejected. |
| **Open risks** | What could still be wrong because of this entry. |

## Rules

1. **Never fabricate progress.** If something was not run, it was not run.
2. **Never invent a number.** Every number traces to a run, a file and a function.
3. **Label the status**: planned / implemented / tested / empirically confirmed / not confirmed / refuted.
4. **Record interruptions and failures**, not only successes. A stalled agent, a machine that slept, a
   test that failed — these belong in the record.
5. **Record corrections to the plan**, including who or what caught them and whether any result had
   already been computed under the old version.
6. **Do not modify an old entry.** Correct it with a new one that says "corrects entry N".
7. Nothing in `PREP/` is touched.

---

## Blank entry

```markdown
N. **<AI|Human>, <short action>.** <What was done.> Config: <preset / seeds / any deviation>.
   Commit: `<sha>`. Files: `<path>`, `<path>`.
   Observed: <measurement, with its provenance>.
   Interpretation: <if any — kept explicitly separate, or "none drawn">.
   Decision: <what was decided and why; alternatives rejected>.
   Open risk: <what could still be wrong>.
```

---

## Worked example (entry 15 of `docs/LABBOOK.md`, abridged)

```markdown
15. **A pre-registered prediction was found to be WRONG and was corrected before any run was analysed.**
    The brief and `INTERFACES.md` §5 predicted that an ideal rectified circuit neuron shows
    `sum_direction_share ≫ diff_direction_share` in its own activation map. It does not: because
    `|cos s|·|cos t|` is symmetric in `s = (u+v)/2` and `t = (u−v)/2`, rectification produces the `(a+b)`
    term (phase `φ_a + φ_b`) and the `(a−b)` term (phase `φ_a − φ_b`) with the **same** amplitude
    `8/(3π²) ≈ 0.2702`. Measured: 0.2698 vs 0.2705. What selects addition is the population plus the
    readout — with `φ_out = φ_a + φ_b` the `(a+b)` contributions add coherently while the `(a−b)` ones
    cancel (400 synthetic neurons: R² 0.938 vs 0.003; a scrambled readout gives 0.113).
    Consequence, fixed now: the sum-over-difference contrast is a prediction about the logits and the
    population, never a per-neuron pass criterion. Corrected in `MLP_MECHANISM_DERIVATION.md` §4.1/§6,
    `INTERFACES.md` §5, the wave-2 implementation brief, and pinned by
    `test_derivations.py::check_population_selects_sum`. No result had been computed under the wrong
    prediction — the correction is a pre-analysis change, and it is recorded rather than silently applied.
```

What makes it a good entry: the actor and the action are in the first line; the *reason* the old
prediction was wrong is given, not just that it was; the measured numbers and their controls are there;
the files changed are listed; and the last sentence states the one thing a sceptical reader most wants to
know — whether any result had already been produced under the wrong version.

---

## Where each kind of record belongs

| record | file |
|---|---|
| chronological account of what happened | `docs/LABBOOK.md` (this template) |
| phase-level audit trail in the project's own format | `PROGRESS.md` |
| long-term reusable knowledge, experiments, decisions, bugs, ideas | the Obsidian vault under `GROKVERSE/` |
| what a human decided | `docs/HUMAN_DECISIONS.md` |
| what the AI did versus what a human did | `AI_DISCLOSURE.md` |
| measurements | `RESULTS.md` and the per-run `analysis/` JSON |
| what the measurements mean | `docs/HUMAN_INTERPRETATION_TEMPLATE.md`, written by a human |
