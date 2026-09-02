# GROKVERSE Project Log

Continuous development log for the GROKVERSE BWKI 2026 project.

Rules for this file are defined in `Claude/grokverse/CLAUDE.md`:
append one dated entry per meaningful work session, newest entries at the bottom,
never fabricate progress, never invent experiment results, and mark any correction
of an older entry explicitly.

---

## 2026-09-01

### What was done
- Set up the Obsidian knowledge structure for GROKVERSE at the vault root.
- Created the folders `GROKVERSE/Project Log/`, `GROKVERSE/Sessions/`, `GROKVERSE/Experiments/`, `GROKVERSE/Decisions/`, `GROKVERSE/Bugs/`, `GROKVERSE/Learnings/` and `GROKVERSE/Ideas/`. All seven did not exist before and were newly created.
- Created this log file as the single main project log.
- No existing project file was modified, moved, renamed or deleted. `Claude/PREP/` was left untouched.

### Results
- No experiments were run in this session. No measured results to report.
- Repository state observed at the start of the session (not produced by this session):
  - The project code and documentation live in `Claude/grokverse/` (`training/`, `web/`, `pitch/`, `docs/`, plus `PLAN.md`, `PROMPT.md`, `PROGRESS.md`, `RESULTS.md`, `AI_DISCLOSURE.md`, `THIRD_PARTY.md`, `IDEAS_BACKLOG.md`, `README.md`).
  - `Claude/grokverse/training/figures/` contains 21 figure files.
  - `Claude/grokverse/training/runs/` exists.
  - Last commit on `main`: `d9434c1 feat: grokverse-Projekt direkt ins Repo aufnehmen`, dated 2026-08-19.

### Decisions
- Knowledge base placed at `GROKVERSE/` in the vault root, separate from the code folder `Claude/grokverse/`.
  - Reason: chosen explicitly by the user after the path conflict below was raised.
  - Alternatives considered: `Claude/` (the layout written in `CLAUDE.md`), and `Claude/grokverse/` (notes next to the code).
  - Consequence: the folder paths listed in `Claude/grokverse/CLAUDE.md` (`Claude/Sessions/`, `Claude/Learnings/`, `Claude/Project Log/`, ...) no longer describe the actual layout and need to be updated.
- Folders were created empty. No placeholder or example notes were written, to avoid seeding the vault with content that has no verified source.

### Problems
- Path conflict: the requested paths (`GROKVERSE/...`), the paths documented in `Claude/grokverse/CLAUDE.md` (`Claude/...`) and the layout on disk (`Claude/grokverse/`) disagreed with each other.
- `git status` reports every file under `grokverse/` and `PREP/` as deleted. Cause: both folders were moved on disk into `Claude/`, and that move is not committed. The files are present at their new location; nothing is lost.

### Solutions
- The path conflict was resolved by asking rather than guessing; the user selected the vault root location.
- The git deletion report was diagnosed but not fixed in this session — see open tasks.

### Open tasks
- [ ] Update the vault paths in `Claude/grokverse/CLAUDE.md` so they match the actual `GROKVERSE/` layout.
- [ ] Stage and commit the move of `grokverse/` and `PREP/` into `Claude/` so git stops reporting the old paths as deleted.
- [ ] Decide whether `Claude/grokverse/docs.zip` and `Claude/grokverse/.venv/` should be ignored or removed from the vault.

### Next step
- Update the Obsidian path section in `Claude/grokverse/CLAUDE.md` to point at `GROKVERSE/`, so future sessions write notes to the correct folders.

### Evidence
- Rules source: `Claude/grokverse/CLAUDE.md`
- Created folders: `GROKVERSE/Project Log/`, `GROKVERSE/Sessions/`, `GROKVERSE/Experiments/`, `GROKVERSE/Decisions/`, `GROKVERSE/Bugs/`, `GROKVERSE/Learnings/`, `GROKVERSE/Ideas/`
- Created file: `GROKVERSE/Project Log/GROKVERSE Project Log.md`
- Project constitution and plan: `Claude/grokverse/PROMPT.md`, `Claude/grokverse/PLAN.md`
- Existing project documentation, not written or verified in this session: `Claude/grokverse/PROGRESS.md`, `Claude/grokverse/RESULTS.md`
- Git: `d9434c1` (2026-08-19)

---

## 2026-09-01 (follow-up)

### What was done
- Verified the Obsidian knowledge paths in `Claude/grokverse/CLAUDE.md`. They already point at `GROKVERSE/` and required no change; no edit was made to the file.

### Results
- All seven knowledge folders are declared as `GROKVERSE/Project Log/`, `GROKVERSE/Sessions/`, `GROKVERSE/Learnings/`, `GROKVERSE/Experiments/`, `GROKVERSE/Decisions/`, `GROKVERSE/Bugs/`, `GROKVERSE/Ideas/`.
- The main log path is declared as `GROKVERSE/Project Log/GROKVERSE Project Log.md`.
- No remaining knowledge path beginning with `Claude/` was found in the file.

### Open tasks
- [x] Update the vault paths in `Claude/grokverse/CLAUDE.md` — resolved; the paths were already correct when checked.
- [ ] Stage and commit the move of `grokverse/` and `PREP/` into `Claude/` (still open, see entry above).
- [ ] Decide whether `Claude/grokverse/docs.zip` and `Claude/grokverse/.venv/` should be ignored or removed from the vault (still open).

### Evidence
- `Claude/grokverse/CLAUDE.md`

---

## 2026-09-02 / 2026-09-03 — Architecture study: baseline, pipeline, run matrix

### What was done

- Turned GROKVERSE from a reproduction-plus-explorer project into a controlled mechanistic comparison of
  the transformer against the MLP, following the master prompt `GROKVERSE_MASTER_PROMPT_EN.md`.
- Secured the baseline: tag `baseline/pre-arch-study` at `d9434c1`, branch `arch-study`, and a byte copy
  of all 16 legacy runs, 21 figures and the result documents under
  `Claude/grokverse/archive/pre_arch_study_2026-09-02/` with a SHA256 manifest of all 75 files. Nothing
  existing was deleted or overwritten. The uncommitted move of `grokverse/` and `PREP/` into `Claude/`
  was recorded as renames.
- Wrote the pre-registration brief (hypotheses, nulls, refutation criteria, evidence gate, structured-neuron
  definition, key-frequency rule, decision tree, statistics) **before** launching any run, and the
  interface contract every analysis module is written against.
- Implemented run format v2: dense evaluation (train every 10, test every 25 steps), transition detection
  with evaluation intervals for three threshold sets, 21 pre-specified checkpoints per run with assigned
  roles, a per-run manifest (split hash, versions, platform, parameter counts, weight norms), a CSV/JSON
  aggregator, a paired-seed matrix launcher, and a two-hot MLP input-parametrization control.
- Fetched and quoted seven primary sources into `Claude/grokverse/docs/sources/`, four of them
  adversarially re-verified by a second agent; wrote three audit documents (evidence audit,
  capacity/confound report, legacy-metric audit).
- Wrote and numerically verified both mechanism derivations
  (`docs/MLP_MECHANISM_DERIVATION.md`, `docs/TRANSFORMER_MECHANISM_DERIVATION.md`).
- Rewrote `AI_DISCLOSURE.md` to separate what the AI did from what a human has actually done.

### Results

- Test suites: `tests/test_run_format_v2.py` 115 checks pass; `tests/test_derivations.py` 41 checks pass;
  `test_core.py` unchanged at 84 pass with the one pre-existing failure documented in `docs/BASELINE.md`.
  Dense evaluation leaves the parameter trajectory bit-identical (asserted by test).
- Run matrix (commit `d53cf52`, 8 workers, `threads=1` each): 18 of 20 primary runs completed, transformer
  seeds 8–9 still training. Raw crossings per seed are recorded in [[Primary architecture run matrix]].
  Final test accuracy at step 25,000: transformer 0.9971–1.0000 (n=8), MLP 1.0000 (n=10).
- **No comparison and no structure metric has been computed** — the analysis code is not yet frozen.

### Decisions

Recorded in [[2026-09-02 Architecture study design decisions]]: fixed 25,000-step budget instead of
stopping at the crossing, ten paired seeds plus the full control matrix, key frequencies from the
neuron-to-logit map instead of a fixed top-8, the two-hot control varying only the input parametrization,
an evidence gate before any cross-architecture interpretation, and a code freeze after a pilot.

### Problems

- The AI session hit a usage limit mid-session on 2026-09-02 while two multi-agent workflows were running;
  13 agents died with that error. The local training matrix was unaffected and kept running.
- One pre-registered prediction turned out to be mathematically wrong.

### Solutions

- Everything the interrupted agents had written was kept and completed rather than rerun; the workflows
  were resumed so finished agents replayed from cache. The interruption, what continued locally, and the
  recovery are recorded in `Claude/grokverse/docs/LABBOOK.md` entries 9–13.
- The wrong prediction (that a single rectified neuron favours `(a+b)` over `(a-b)`) was corrected in the
  derivation, the interface contract and the implementation brief, and pinned by a test — before any run
  was analysed. See [[ReLU as the multiplier in modular addition]].

### Open tasks

- [ ] Finish transformer seeds 8–9 and the confound, parameter-matched and two-hot blocks.
- [ ] Complete and review the analysis modules; freeze the analysis code after the seed-0 pilot.
- [ ] Run the analyses, statistics, evidence gate and decision tree on the full matrix.
- [ ] Rewrite `RESULTS.md` and `README.md` to match the actual evidence.
- [ ] **Human**: approve or change the `[AI-PROPOSED]` pre-registration values; check the derivations;
      read the primary sources; write the final interpretation in your own words.

### Next step

Complete the wave-1 analysis modules and their reviews, then implement the mechanism, ablation and
aggregation modules against the frozen interface contract.

### Evidence

`Claude/grokverse/docs/LABBOOK.md` · `docs/dev/PREREG_BRIEF.md` · `docs/BASELINE.md` ·
`docs/CURRENT_EVIDENCE_AUDIT.md` · `docs/sources/` · `training/runs/*_arch25k/` ·
commits `25890dc`, `84d449d`, `d53cf52`, `d7b74bc`, `aa536d2`, `aab1ee4`, `6a409a0`, `908222f`
