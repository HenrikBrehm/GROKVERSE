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


## 2026-09-03 (evening)

### What was done

- Recovered from an editor crash: the three interrupted wave-2 agents' work was complete on disk; the one
  failing test file (`test_mask_protocols.py`) had two defects in the *tests*, both fixed.
- `analysis/transformer_mechanism.py` written and tested (67 checks): the transformer now undergoes the
  same mechanism batteries as the MLP through the same functions (master prompt §8, §17).
- Shared code generalized (`act=`, `[p,p,p]` bias term) instead of duplicated; `key_frequencies`
  transformer placeholder closed.
- Handoff prepared for the next session: `docs/dev/HANDOFF_2026-09-04.md`, `run_analysis_chain.ps1`,
  standby disabled, `HUMAN_DECISIONS.md` restored, memory note, everything committed on `arch-study`.

### Results

- None of the study. Test suite 13/13 files, 1,438 checks. Primary training block 20/20 complete;
  control blocks training overnight (`confound` 8/18 at 17:17 UTC start).
- Pipeline check only: `transformer_mechanism.analyse` on seed 0 / step 25 000 — forward decomposition
  exact, variance shares sum to 1. Unfrozen code; not a result.

### Decisions

- No large tensors in analysis outputs (INTERFACES §6 amendment).
- No subagents for the remaining modules; commit after every green stage.
- Restore rather than drop the approval gate; it labels results and never blocked work.

### Problems

- Machine standby after 15 min idle was the cause of the earlier 7.6 h stall — now disabled.
- `h3_validity` has no interface specification; the §13 modules (`aggregate`, `figures_study`,
  `decision_tree`) were not counted as missing before tonight.

### Solutions

- Everything is measured and written into the handoff and labbook entries 31–44 so the next session
  starts from facts, not recollection.

### Open tasks

- [ ] `causal_ablation.py`, `h3_validity.py` (spec first), `structure_over_time.py`, driver-test update.
- [ ] Freeze the analysis code (PREREGISTRATION §12), run the driver over the primary block, then all.
- [ ] Aggregate, decision tree, statistics, `CLAIM_EVIDENCE_TABLE.md`, `RESULTS.md`, `README.md`.
- [ ] **Human**: approve `HUMAN_DECISIONS.md`, fill section F and the `AI_DISCLOSURE.md` placeholders,
      write the interpretation.

### Next step

Stage A of the handoff: `analysis/causal_ablation.py` with its tests, then commit.

### Evidence

`Claude/grokverse/docs/LABBOOK.md` entries 31–44 · `Claude/grokverse/docs/dev/HANDOFF_2026-09-04.md` ·
`training/tests/run_all.py` · `training/results/matrix_launch.log` · `training/runs/*_arch25k/manifest.json`

---

## 2026-09-04

### What was done

- Completed every remaining item of the master prompt's §23 Definition of Done that does not require
  the human authors: stages A–H of the handoff.
- Aggregation and reporting layer written test-first and run: `aggregate`, `decision_tree`,
  `figures_study`, `statistics_report`, `h3_report`, `h4_report`, `controls_report`,
  `bounded_alternative`.
- Analysed all 51 runs; evaluated the pre-registered evidence gate at both measurement points; applied
  H3's refutation criteria literally; ran the bounded alternative-mechanism analysis and all three
  control blocks.
- Rewrote `RESULTS.md`, `README.md`, `docs/LIMITATIONS.md` §B and `PROGRESS.md` against the
  measurements, under master prompt §21's graded wording; updated the claim–evidence table, the
  novelty delineation, `AI_DISCLOSURE.md` and `PREREGISTRATION.md` §12.
- Added `tests/check_results_numbers.py`, which re-derives every number quoted in `RESULTS.md` from
  `results/` and fails on disagreement.

### Results

- **Evidence gate: `neither_passes`.** G1/G2/G3 hold 10/10 for both architectures; **G4 fails 0/10 for
  both**. The structured-neuron definition selects 88–98 % of the network, so the size-matched random
  control does 0.873 (MLP) / 0.916 (transformer) of the same damage and the criterion cannot
  discriminate.
- **H3 refuted by its own criterion.** The family definition closes +0.0049 of a +0.0898 gap (5.4 %);
  the family gap remains +0.0850, CI95 [+0.0600, +0.1004], unanimous over 10 seeds.
- **Key frequencies are causally load-bearing in both** (removal costs ~0.99 accuracy vs ~0.000 for a
  size-matched random set), unlike the neuron sets.
- **Function agreement 0.99977** over all 12,769 inputs, identical in 4 of 10 seeds, while logits
  correlate at 0.083 and top-2 predictions agree 0.6 % of the time.
- **H4 holds**: six structure metrics onset before the generalization crossing in 10/10 seeds, both
  architectures, at 5.4–39.5 % of the way there. The waveform composition shifts during the plateau —
  square-best-fit share 0.787 → 0.116 (txf) and 0.789 → 0.421 (MLP), sinusoid rising to 0.646 and
  0.333 — unanimous in sign across all seeds.
- **Bounded alternative**: effective rank 12.7 (txf) vs 66.6 (MLP); removing the top-16 singular
  directions destroys the transformer (0.964) and leaves the MLP untouched (0.000). Cross-seed CKA is
  0.0017 / 0.0973 between seeds of the *same* architecture, so representation similarity is unusable
  here as evidence in either direction.
- **Controls**: `train_frac` dominates Grokfast by an order of magnitude; every structure difference
  survives parameter matching to 0.02 %; the two-hot MLP is **more** square-wave-like (+0.1836) than
  the shared-embedding MLP.
- **Convergence**: the MLP's `structured_fraction_of_live` — the metric behind the headline gap — is
  settled at the 25,000-step budget in only **2 of 10 seeds** and still rising.
- Suite: 24 files, 1,913 checks. Freeze audit: all 696 analysis artifacts across all 51 runs postdate
  the freeze `0b55e1d`. All 11 figures regenerate byte-identically.

### Decisions

- Report the gate failure and the refutation as the results (master prompt §21), not reframed.
- Do not change the onset rule, the structured-neuron threshold, or which transformer ablation G4 means
  — all pre-registered. The last two are escalated as **D5** and **D6** with measured sensitivity.
- Withdraw four headline claims from the 2026-09-03 `RESULTS.md`, each with the number that retires it.

### Problems

- Two further silent-extraction defects: `aggregate._progress_measures` (every restricted/excluded
  column `null`) and `structure_over_time` comparing a model **index** to a model **name**, which
  returned **0.0** and flattened both waveform trajectories for the entire study.
- The convergence check promised by `LIMITATIONS.md` §A6 had never been executed.
- `NOVELTY_AND_RELATED_WORK.md` still quoted retired numbers.

### Solutions

- Both defects fixed with regression tests; `structure_over_time` re-run over all 51 runs (51 ok, 0
  failed) and every downstream report re-run and diffed to confirm nothing else was affected.
- The convergence check was executed and its finding written into `LIMITATIONS.md` §B3 and
  `RESULTS.md` §5.1.
- The retired numbers were re-measured at convergence: 0.961 vs 0.891, not 0.73 vs 0.44.

### Open tasks

- [ ] **Human**: the final scientific interpretation in the authors' own words.
- [ ] **Human**: `HUMAN_DECISIONS.md` A–F including **D5** and **D6**, plus the status line and sign-off.
- [ ] **Human**: the 11 `[HUMAN AUTHORS MUST COMPLETE]` placeholders in `AI_DISCLOSURE.md`.
- [ ] **Human**: the `txf_mul_*` runs (**E6**); the explorer update (**E3**).

### Next step

Human review of `docs/HUMAN_DECISIONS.md`. Nothing further can be decided by the AI without it.

### Evidence

`Claude/grokverse/RESULTS.md` · `docs/LIMITATIONS.md` §B · `docs/CLAIM_EVIDENCE_TABLE.md` ·
`docs/LABBOOK.md` entries 99–105 · `training/results/decision_tree_final.json` ·
`training/results/{statistics,h3_report,h4_report,controls_report}.json` ·
`training/results/aggregate/` · `training/results/figures/` ·
`training/tests/check_results_numbers.py`

## 2026-09-06

### What was done

- Executed the graded neuron-ablation follow-up an external reviewer proposed on 2026-09-05 after
  reading branch `GROKVERSE-MP`.
- Pre-registered it first as `docs/PREREGISTRATION.md` §14 with decision **D7** left blank, and
  committed that (`7161842`) **before any graded number was read**.
- Added `graded_structured_ablation` as an additive block in `causal_ablation`
  (`MODULE_VERSION` 1.0 → 1.2), tests first and red first, and re-ran all 40 primary run-checkpoints.
- Read the already-existing IPR sweep (D2) at matching fractions with no new computation, as an
  independent second ranking.
- Found and fixed a determinism defect in the frozen analysis code on the way.

### Results

Real measured results, 10 primary seeds per architecture, both checkpoints
(`training/results/graded_ablation.json`, `training/results/GRADED_ABLATION.md`):

| arch | checkpoint | smallest discriminating fraction | drop vs control | seeds |
|---|---|---|---|---|
| MLP | crossing | 1 % (5 neurons) | 0.187 vs 0.010 | 10/10 |
| transformer | crossing | 1 % (5 neurons) | 0.031 vs 0.004 | 9/10 |
| MLP | final | 50 % (256 neurons) | 0.412 vs 0.052 | 9/10 |
| transformer | final | 5 % (26 neurons) | 0.096 vs 0.0002 | 9/10 |

At the transition, removing the five most structured neurons beats every one of fifty random
five-neuron groups in both architectures. At convergence the MLP shows nothing below half the
network — its top 1/2/5/10 % cost exactly 0.0000 — while the transformer keeps a small load-bearing
core. G4's failure was therefore set size, not causal inertness. The gate is untouched at
`neither_passes`; §14.5 fixed in advance that no outcome here reopens it.

Driver 40 ok / 0 failed. Suite 24/24 files. `check_results_numbers.py` 117/117.

### Decisions

- The graded ablation is a **new pre-registration**, not a change to threshold B1 — D6 already
  established that no B1 threshold rescues G4, so only a different kind of selection could.
- The determinism fix is treated as a `PREREGISTRATION.md` §9 **bug fix**, not a control change,
  because determinism is a stated non-negotiable and the drift exceeded the verifier's tolerance.
  Escalated as **D8** so the human authors can disagree.

### Problems

- Re-running `causal_ablation` showed 560 of 660 pre-existing ablation blocks differing — every
  observed value identical, every control statistic moved.
- The IPR extraction silently dropped two of its four fractions and still produced a well-formed table.

### Solutions

- `resolve_structured_masks` built the definitions as a Python **set**, whose order depends on
  per-process string hashing, while both ablation functions draw from **one shared rng** in that
  order. Fixed by iterating the already-ordered tuple; proven across six `PYTHONHASHSEED` values;
  regression test added. Zero verdict flips; two published control numbers updated
  (MLP 0.873 → 0.878, transformer 0.916 → 0.910).
- The pruning grid is cardinality-derived (26/512 = 0.05078, not 0.05), so exact-equality matching
  found nothing. Now matches the nearest grid point within half a step and reports the value used.

### Open tasks

- [ ] **D7** and **D8** — both need a human decision.
- [ ] Compare the collaborator's longer convergence run (branch `arch-study-convergence`, their AWS
      account, due 2026-09-06 morning) against our MLP structured fraction.
- [ ] Unchanged and still human-only: the final interpretation, the 11 `AI_DISCLOSURE.md`
      placeholders, the `txf_mul_*` runs, the explorer update.

### Next step

Human review of `docs/HUMAN_DECISIONS.md`, now including **D7** and **D8**.

### Evidence

`Claude/grokverse/RESULTS.md` §6.1 · `docs/PREREGISTRATION.md` §14 and §12 ·
`docs/HUMAN_DECISIONS.md` D7, D8 · `docs/LABBOOK.md` 109 ·
`training/results/graded_ablation.json` · `training/results/GRADED_ABLATION.md` ·
`training/results/analysis_driver_graded.json` · `training/analyse_graded.py` ·
[[Graded structured-neuron ablation]] · [[Nondeterministic control draws from a Python set]]

---

## 2026-09-07

### What was done
- Archived the complete LinkedIn mentoring chat (2026-08-14 to 2026-09-07) between the authors and the external reviewer of 2026-09-05 in `GROKVERSE/Mentoring/` (local only, see Decisions), with a table of the reviewer's contributions and where each landed in the project, and a list of open items from the chat.
- Added `GROKVERSE/Mentoring/` to `.gitignore`.
- Re-checked the reviewer's proposed submission framing (chat of 2026-09-07, 17:09) against `RESULTS.md`, `training/results/aggregate/TABLES.md`, the per-run `key_frequencies` artifacts and the git remote. No code or result file was changed.

### Results
- No experiments were run.
- The 11 study figures (`training/results/figures/`) and the aggregate tables are on `origin/GROKVERSE-MP` (commits `1076dee`, `75d92ce`), but `README.md` links neither `training/results/figures/` nor `TABLES.md`. This is the likely reason the reviewer lists "plots and reports comparing the architectures" as still to do.
- The four graded-ablation commits (`7161842`, `f8fc430`, `b97d7b7`, `e7dcf82`) are local only; the branch is 4 ahead of origin. The reviewer asked to be told when these runs are done.
- The reviewer's convergence branch `arch-study-convergence` (100k steps, her AWS account) is not on origin. Her 100k numbers (about 11 vs about 4.5 key frequencies; nearly all live neurons structured in both architectures) cannot be verified from this repository yet. At 25k steps the same counts here are median 12 vs 4.5 (`nanda`) and 9 vs 4 (`neuron_clusters`), and the key-frequency sets differ between every pair of seeds in both architectures (`analysis/key_frequencies/step025000.json`, 20 primary runs).
- Her sentence "MLP broader and more redundant, transformer more compact and more sensitive to removing important neurons" matches `RESULTS.md` 6.1 at the final checkpoint (MLP discriminates from its control only from 50 %, transformer from 5 %, two independent rankings). At the crossing checkpoint both architectures discriminate from 1 % (5 neurons), so the sentence needs the qualifier "at the end of training".
- Her sentence "both models learn the same mathematical principle" is not licensed by the pre-registered gate (`neither_passes`, `RESULTS.md` 3 and 11) and would not become so at 100k steps: a larger structured set makes the size-matched control discriminate less, not more. What is licensed: key frequencies are causally load-bearing in both architectures (`RESULTS.md` 6, 10/10 necessary and 10/10 sufficient), phase relation and end-to-end Fourier fit hold in both (G2, G3 10/10).
- Her earlier hypothesis that attention explains the difference has a partial test on disk already: fixing attention to its mean costs 0.336 accuracy, each head drop is 0.44 to 0.49 with z 19 to 27, no single head sufficient (`RESULTS.md` 6).
- No `bwki-evaluation-criteria` skill exists yet (promised to the reviewer on 2026-09-02).

### Decisions
- The mentoring folder is kept out of git rather than redacted, so the vault holds the complete record: this vault is the public GitHub repository and the chat contains a private e-mail address and personal details. Reversible by removing the `.gitignore` line.
- This public log keeps the existing anonymisation ("external reviewer"); the name is in the local note. Whether the reviewer is named in the submission is the authors' decision (Eigenstaendigkeit disclosure).

### Open tasks
- [ ] Push the four local commits and tell the reviewer the graded ablation is done.
- [ ] Ask the reviewer for the `arch-study-convergence` branch or fork and for the analysis commit used for the 100k numbers (frozen code or not).
- [x] Link `training/results/figures/` and `training/results/aggregate/TABLES.md` from `README.md` (done 2026-09-07, also links `GRADED_ABLATION.md`).
- [ ] `AI_DISCLOSURE.md` has no section for external input; the reviewer's contributions (framing question, mask question, graded ablation, convergence run, reframing text) belong there. Human authors.
- [ ] File the reviewer's e-mail of 2026-09-02 (suggested edits to the spec) in `GROKVERSE/Mentoring/`.
- [ ] Decide the wording of the "same principle" sentence before it goes into the Projektdoku (see Results above).

### Evidence
- `git branch -r --contains 1076dee` -> `origin/GROKVERSE-MP`; `git log --oneline origin/GROKVERSE-MP..HEAD` -> 4 commits; `git branch -r` -> no `arch-study-convergence`.
- `Claude/grokverse/RESULTS.md` 3, 5, 6, 6.1, 11; `training/results/aggregate/TABLES.md` (`key_frequencies` block); `training/runs/*_frac0.3_seed?_arch25k/analysis/key_frequencies/step025000.json`.
- `.gitignore` (line 2); `GROKVERSE/Mentoring/` (local).

---

## 2026-09-08

### What was done
- Received the external reviewer's data bundle (`grokverse_run_data.zip`, 1.33 GB, SHA-256 `c6e532ef…add44`, via a presigned S3 link valid 24 h): the 100,000-step convergence block S1 (20 runs, run 2026-09-05 on the reviewer's AWS EC2 `c7i.16xlarge`), the two-hot control at 10 seeds (2026-09-07), the executed source tree with git history, and four reviewer-authored documents. Unpacked into the gitignored `Claude/grokverse/training/external/`; small outputs copied to `training/results/external/`, documents to `docs/external/` (verbatim); the reviewer's branch `arch-study-convergence` fetched from the bundled tree as a local branch, not pushed.
- Verified the bundle against this repository: 30/30 manifests at commit `42dd79a` (= our `fda066e` + `CHECKPOINT_GRID` entries above 25,000 + runbook, nothing else), 23/23 split hashes identical to ours per seed, 829/829 checkpoint SHA-256 match; the frozen `decision_tree`, `statistics_report`, `h3_report`, `h4_report` re-run on the bundle's aggregate reproduce the reviewer's reports byte-for-byte apart from timestamps and paths; the reviewer's per-seed table recomputed from raw trajectories, every value agrees.
- Wrote `RESULTS.md` §16 (separate, labelled; (v)/(r) marks), appended the measured endpoint to `LIMITATIONS.md` B3, noted the external block in `PREREGISTRATION.md` §12, added disclosure rows and a new §9 "External input" to `AI_DISCLOSURE.md`, escalated **D9**, updated `README.md`, labbook entry 110, `PROGRESS.md`.
- Archived Henrik's reply and today's chat messages in `GROKVERSE/Mentoring/` (file renamed to the 2026-09-08 range; contributions table and open items updated). No result file above §16 and no code was changed.

### Results
- At 100,000 steps the MLP's structured fraction settles (10/10 seeds; 2/10 at 25k) at a median of 1.0000, as does the transformer's; the 25k gap of +0.0977 closes to 0.0000. The one unsettled seed is now transformer seed 2 (0.877 → 0.805 between 90k and 100k, still moving). All 20 runs stay generalized (min test acc 0.9992).
- The gate at 100k is `neither_passes` with the same G1–G3 10/10, G4 0/10 pattern — D6's prediction (a larger structured set makes the size-matched control less discriminating) confirmed.
- Key-frequency counts at 100k: 4.5 (3–5) vs 11 (9–12) under `nanda`, 4 vs 8 under `neuron_clusters`; the count difference holds at both budgets under all five rules.
- Attention at 100k: fixing attention to its mean costs 0.415 (25k: 0.336); single heads 0.49–0.55.
- The reviewer's own S2/S2b/S3 (reported only): the transformer's top quarter of LOO-ranked neurons is collectively necessary but not sufficient; the MLP's ranking buys nothing over random; key-frequency-ranked percentile removal harms the transformer from 5 % and the MLP from 50 %, matching our §6.1 asymmetry under a different ranking; frequency sets sit at the size-matched null in both architectures.
- Two-hot at 10 seeds: `nanda` selects all 56 frequencies in every seed (our three: the same); seeds 0–2 reproduce our runs behaviourally.
- The "+0.085 vs +0.0977" note in the reviewer's findings is not a discrepancy: paired per-seed median under the family definition vs difference of the per-architecture medians, same metric, same block.
- **Missing:** the three scripts behind S2/S2b/S3 (`training/run_s2_ranked_ablation.py`, `run_s2b_freq_ranked_ablation.py`, `run_s3_frequency_families.py`) are neither in the bundle nor in the bundled source tree; those numbers cannot be re-derived here.

### Decisions
- Bulk data (1.5 GB) stays out of git (`training/external/` gitignored, like `training/runs/`); the reviewer's small outputs and documents are tracked as unmodified copies so `RESULTS.md` §16 can cite them.
- The reviewer's `CHECKPOINT_GRID` extension is not merged; it is disclosed as a declared deviation on the reviewer's branch and left to D9.
- The reviewer's documents are stored verbatim, including the runbook that names the reviewer; this repository's own documents keep "external reviewer". The bundled git history (work e-mail as commit author) stays out of git. Naming is the authors' call (D9).
- The reviewer's branch is held locally and **not pushed**; nothing was pushed this session.
- The local 100k reproduction (runbook §0.3, ≈ 20 CPU-hours) was not started: compute is a human decision (`AI_DISCLOSURE.md` §1).

### Open tasks
- [ ] **D9** — accept the external block on the runbook's terms? competition rules on external compute? name the reviewer? merge the grid?
- [ ] Ask the reviewer for the three S2/S2b/S3 scripts (draft reply in `GROKVERSE/Mentoring/Antwort-Entwurf 2026-09-08.md`).
- [ ] Decide whether to add the reviewer (`CatiaV5`) as a collaborator on the public repository — her question of 2026-09-08.
- [ ] Reproduce ≥ 1 seed pair at 100,000 steps locally before any S1 number is used (MLP ≈ 3.6 h, transformer ≈ 16 h) from a worktree of `arch-study-convergence`.
- [ ] Unchanged and still human-only: the final interpretation, the `AI_DISCLOSURE.md` placeholders (now also §9), the `txf_mul_*` runs, the explorer update.

### Evidence
- `Claude/grokverse/docs/external/README.md` (provenance and every check) · `RESULTS.md` §16 · `docs/LIMITATIONS.md` B3 · `docs/PREREGISTRATION.md` §12 · `AI_DISCLOSURE.md` §5, §9 · `docs/HUMAN_DECISIONS.md` D9 · `docs/LABBOOK.md` 110 · `PROGRESS.md` 2026-09-08
- `training/results/external/` (tracked copies) · `training/external/` (bundle, gitignored) · local branch `arch-study-convergence` at `42dd79a`
- [[External convergence block S1 at 100k and follow-ups]] · [[Graded structured-neuron ablation]]

---

## 2026-09-20 — Abgabetag

### What was done
- Received from Ali Kandora, a colleague of the author, an AI-generated four-phase audit of GROKVERSE (`bwki/grokverse_scientific_audit.md`) plus an add-vs-mul experiment script. Verified with three subagents: the audit assessed the **archived pre-2026-09-02 snapshot** (its K1/K2/K3/K5/K8 are already closed in the current study); its "proof" step 3 is wrong (L2 on a fixed orthonormal basis is not LASSO-like; the delta target of its own step 1 has an exactly flat spectrum, top-8 = 8/56 = 0.1429 — the very number its own §2.2 quotes as a criticism) and step 1 overclaims a finite CE optimum; its Fourier basis over ℤ₁₁₂ was overcomplete (114 rows; degenerate sine row, duplicated Nyquist row). Script corrected (`bwki/experiment_add_vs_mult.py`, `--self-test`: orthonormality 7.6e-15, Parseval 6e-16), `--analyze-run` mode added and verified on `add` runs only; the original kept as `experiment_add_vs_mult (1).py`. Disclosed in `AI_DISCLOSURE.md` §9b.
- Submission readiness: test suite 24/24 green; `web/node_modules` found incomplete (`styled-jsx` missing), reinstalled with pnpm 10 (pnpm 9 rejects `pnpm-workspace.yaml`, which has no `packages` field) — `pnpm build` clean, nothing under `web/` changed (E3). Grokfast author list corrected (`docs/NOVELTY_AND_RELATED_WORK.md`), Grokfast attribution added to `THIRD_PARTY.md`, pnpm note in `README.md`, `__pycache__/` ignored.
- `docs/ADD_VS_MUL_EIGENLEISTUNG.md` created as a headings-only template for the author's own evaluation of the `txf_mul_*` runs (E6). The AI did not read those runs' metrics.
- A draft of the written submission assembled by a subagent from the repository's records (`bwki/PROJEKTDOKUMENTATION_ENTWURF.md`, outside the repo folder), labelled AI-drafted, with the own-work and interpretation sections left to the author.

### Results
- Confirmed BWKI deadline: **2026-09-20, 23:59** (bw-ki.de); submission = written documentation + executable code + video (2–4 min). No written documentation existed before today.

### Decisions
- None taken by the AI. Escalated to the author: whether `Claude/PREP/` stays tracked (marked "not for submission", but public); the one-sentence E6 departure in `docs/sources/power2022_and_grokfast2024.md` item 8; whether the reviewer is named (D9; `docs/RESEARCH_SPEC.md` names her, the rest does not); the E6 rule missing from `CLAUDE.md`; an explorer notice; `git push` (one commit ahead plus today's changes).

### Open tasks
- [ ] `docs/HUMAN_DECISIONS.md` A–F, D5–D9, status line and sign-off — dated today, i.e. post hoc.
- [ ] `txf_mul_*` evaluation by the author (`--analyze-run`, then `docs/ADD_VS_MUL_EIGENLEISTUNG.md`).
- [ ] `AI_DISCLOSURE.md`: the 12 `[HUMAN AUTHORS MUST COMPLETE]` placeholders.
- [ ] Projektdokumentation: author's revision of the draft, then PDF.
- [ ] Push; upload at competition.bw-ki.de.

### Evidence
- `Claude/grokverse/AI_DISCLOSURE.md` §6, §9b, changelog 2026-09-20 · `Claude/grokverse/PROGRESS.md` 2026-09-20 · `bwki/experiment_add_vs_mult.py --self-test` · `bwki/analyze_run_results.json` (add runs only)
