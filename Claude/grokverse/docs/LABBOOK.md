# Labbook — architecture study

Chronological, append-only. Times are UTC dates with local order preserved. "AI" = Claude (this session,
`https://claude.ai/code/session_01H37s34kZVUTNfZtMtSZnyi`); "human" = the project author. Every entry
names what was done, by whom, and where the evidence is. Nothing here is a result.

## 2026-09-02

1. **AI, scouting.** Read the master prompt, all repo docs, all code, all 16 `run.json` files. Found the
   tree moved to `Claude/grokverse/` without a commit, untracked pre-study audit work (RESEARCH_SPEC,
   `mask_protocols.py`, `mlp_mechanism.py`, harmonic-family functions), and one failing check in
   `test_core.py` (diagnosed in `docs/BASELINE.md`). Benchmarked pinned single-thread step cost
   (transformer 254 ms, MLP 59 ms). Confirmed all seven primary sources resolve on arXiv.
2. **Human decisions (asked upfront, answered):** (a) branch `arch-study`, commit, no push; (b) full run
   matrix, 10 paired seeds; (c) proceed with AI-proposed pre-registration values, labelled as such;
   (d) fixed 25 000-step budget for every primary run instead of an event-relative stopping rule.
3. **AI, baseline secured.** Tag `baseline/pre-arch-study` = `d9434c1`; archive
   `archive/pre_arch_study_2026-09-02/` (75 files, SHA256 manifest); commit `25890dc` records the move and
   the WIP; commit `84d449d` adds `docs/BASELINE.md`, `docs/dev/RUN_FORMAT_V2.md`, `docs/dev/INTERFACES.md`.
4. **AI, specs fixed before code.** `docs/dev/PREREG_BRIEF.md` (hypotheses, nulls, gate pass rules,
   structured-neuron definition, key-frequency rule with its declared conditional substitution, decision
   tree, statistics, freeze rule) written from the master prompt and RESEARCH_SPEC — before any new run
   and before any analysis module existed.
5. **AI, launched in parallel (background):** (i) implementation of the v2 training pipeline per
   `RUN_FORMAT_V2.md`; (ii) literature verification (7 source notes, each adversarially re-verified) plus
   the evidence audit, capacity/confound report and legacy-metric audit (each reviewed by two adversarial
   reviewers); (iii) foundation analysis modules `metrics`, `wave_fitting`, `statistics`,
   `function_agreement`, `logit_formula_fit` with tests, each adversarially reviewed. Wrote
   `analysis/common.py` (shared loader; verified against legacy MLP seed 0: train acc 1.000, test acc
   0.9942 — equals `run.json`).
6. **Planned order from here** (master prompt §24): pilot pair (seed 0) as soon as the pipeline passes its
   tests → PREREGISTRATION.md and plans committed → full matrix launched → second-wave modules
   (mechanisms, ablations, structure-over-time, mask protocols with `nanda_exact`) → freeze commit →
   analyses on all runs → statistics → decision tree → documentation → human interpretation.
