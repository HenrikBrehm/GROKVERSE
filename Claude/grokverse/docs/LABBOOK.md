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
7. **AI, pipeline verified and committed.** v2 training pipeline (fork report): 115 checks pass;
   `test_core.py` unchanged (84 pass, the one pre-existing failure); determinism test passes; dense
   evaluation leaves the parameter trajectory bit-identical; 300-step smoke runs of all three
   architectures produced valid `run.json`/`checkpoints.json`/`manifest.json`. Commit `d53cf52`
   (pipeline + `PREREG_BRIEF.md` + `LABBOOK.md` + `analysis/common.py`).
8. **AI, run matrix launched** (`training/results/matrix_launch.log`, commit `d53cf52`, 8 workers,
   `threads=1` per run), blocks in order: `primary` (20 runs: transformer seeds 0–9 first, then MLP 0–9),
   `confound` (18), `param_matched` (10), `twohot` (3). The seed-0 pair doubles as the pipeline pilot.
   **Order note:** the runs start before `docs/PREREGISTRATION.md` exists as a document; every value it
   will contain is fixed in the committed `PREREG_BRIEF.md` (hypotheses, nulls, gate, definitions,
   decision tree, statistics) and the expansion is forbidden from changing any value. No run output is
   analysed before the analysis code is frozen (rule in the brief).

## 2026-09-02 (late) / 2026-09-03 — session-limit interruption and recovery

9. **Interruption.** The AI session hit its usage limit at about 18:50 UTC on 2026-09-02, while the
   sources-and-audit workflow (23 agents) and the wave-1 analysis workflow (5 agents) were running.
   The training matrix is a local background process and **kept running unaffected**; no run was
   restarted and no output was overwritten. Resumed 22:31 UTC.
10. **Matrix state at resume** (`runs/*_arch25k/manifest.json`): 18 of 20 primary runs `completed`
    (transformer seeds 0–7, MLP seeds 0–9; MLP seeds 6–9 finished during the interruption), transformer
    seeds 8–9 still running (step ~12 700 of 25 000). Under 8-way contention a transformer run took
    ~4.05 h (single-thread estimate was 1.75 h), an MLP run 0.82–0.95 h. Every completed manifest
    validates; every run has 21 checkpoints and all seven checkpoint roles assigned; the seed-0 pair
    therefore serves as the pipeline pilot and passed. `confound`, `param_matched`, `twohot` remain queued
    in the same chained process (`results/matrix_launch.log`). Disk used by v2 runs: 503 MB.
11. **Workflow outcomes.** Every one of the 13 failed agents failed with the same error
    ("session limit"), none with an implementation or research error:
    * sources-and-audit: all 7 source notes and all 3 audit documents were written; the notes for
      Swaroop, Manir & Rupa, Khanh and Power/Grokfast were adversarially re-verified (verifier edits
      present); the verifications of the Nanda, Doshi and NeurIPS-2025 notes and all six audit reviews
      did not run. A partial reviewer artifact `docs/data/reviewer_A_preactivation_check.json` exists;
      no reviewer edit reached any audit document.
    * wave 1: `statistics.py` + tests (pass), `function_agreement.py` + tests (pass, agent died before
      reporting), `logit_formula_fit.py` + tests (test fails with an IndexError in the square-wave phase
      fit — agent died mid-work), `metrics.py` and `wave_fitting.py` written but **no tests** (agent died
      before writing them). No review ran.
12. **Recovery.** (a) Nothing rerun that exists: notes, audits and modules are kept as they are and only
    verified/reviewed/completed. (b) The three missing note verifications and six audit reviews are
    re-launched by resuming the same workflow (completed agents replay from cache). (c) Wave 1 is completed
    by a targeted workflow: finish + test metrics/wave_fitting, fix + test logit_formula_fit, review all
    five modules. (d) Wave 2 does not start until every wave-1 test passes and every review returns.
13. **Pre-declared substitution triggered.** `docs/sources/nanda2023_progress_measures.md` (§3, App. C.2
    quote) establishes that Nanda et al. determine key frequencies from the DFT of the neuron→logit map
    `W_L = W_U W_out` with a measured (uncapped) count and an unpublished threshold. Per the conditional
    fixed in `PREREG_BRIEF.md`, the `nanda` rule becomes the **primary** key-frequency rule and
    `neuron_clusters` secondary; the threshold is a GROKVERSE choice declared now: keep k whose per-frequency
    norm of `W_L` is ≥ 0.25 × the maximum (sensitivity 0.10 and 0.50). Recorded in the brief's addendum
    before any analysis of a new run.
