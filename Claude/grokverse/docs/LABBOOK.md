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
14. **AI, mechanism derivations written and verified (2026-09-03).** `docs/MLP_MECHANISM_DERIVATION.md`
    and `docs/TRANSFORMER_MECHANISM_DERIVATION.md`, with every algebraic step checked numerically on
    **random** models (no trained run read) in the new `training/tests/test_derivations.py` (41 checks,
    all pass): the transformer forward decomposition reproduces the model output to 2.7e-7; the logits
    split exactly into a direct and an MLP path; per-neuron contributions through `W_out @ W_U` sum to the
    MLP path to 2.2e-16; the mean-attention effective curves are exact when attention is uniform
    (`W_Q = W_K = 0`, error 4.4e-16) and `additivity_r2` drops below 1 as soon as attention is
    input-dependent; the MLP effective curves reproduce its forward pass to 1.1e-7; a discrete square wave
    at `p = 113` puts 0.8107 of its non-constant power in the fundamental and 0.1717 of that in the odd
    harmonics 3/5/7 (continuous ideal 0.8106 / 0.1715) with a small nonzero even share 5.8e-4.
15. **A pre-registered prediction was found to be WRONG and was corrected before any run was analysed.**
    The brief and `INTERFACES.md` §5 predicted that an ideal rectified circuit neuron shows
    `sum_direction_share ≫ diff_direction_share` in its own activation map. It does not: because
    `|cos s|·|cos t|` is symmetric in `s = (u+v)/2` and `t = (u−v)/2`, rectification produces the `(a+b)`
    term (phase `φ_a + φ_b`) and the `(a−b)` term (phase `φ_a − φ_b`) with the **same** amplitude
    `8/(3π²) ≈ 0.2702`. Measured: 0.2698 vs 0.2705. What selects addition is the **population plus the
    readout** — with `φ_out = φ_a + φ_b` the `(a+b)` contributions add coherently across neurons while the
    `(a−b)` ones cancel (400 synthetic neurons: R² 0.938 on `cos(ω(a+b−c))` vs 0.003 on `cos(ω(a−b−c))`;
    a scrambled readout gives 0.113; a difference readout selects `(a−b)` instead, R² 0.909).
    Consequence, fixed now: the sum-over-difference contrast is a prediction about the **logits**
    (`logit_formula_fit`'s `control_difference`) and the neuron population, never a per-neuron pass
    criterion; per-neuron `sum`/`diff` shares are reported descriptively. Corrected in
    `MLP_MECHANISM_DERIVATION.md` §4.1/§6, `INTERFACES.md` §5, the wave-2 implementation brief, and pinned
    by `test_derivations.py::check_population_selects_sum`. No result had been computed under the wrong
    prediction — the correction is a pre-analysis change, and it is recorded rather than silently applied.
16. **AI, checkpoint pipeline validated on real v2 runs (2026-09-03).** For the seed-0 pair, every one of
    the seven assigned checkpoint roles loads through `analysis/common.load_model_at` and reproduces a
    logit grid; the test accuracy recomputed from the `generalization` checkpoint equals the value the
    training loop logged at that step to 1e-6 in both architectures (transformer 0.950324, MLP 0.951331).
    Checkpoint SHA256 values are carried into the analysis provenance envelope. This is a pipeline check,
    not a result.
17. **AI, further documents written while the analysis agents ran:** `AI_DISCLOSURE.md` rewritten with
    `[HUMAN AUTHORS MUST COMPLETE]` placeholders wherever a human has not acted; `docs/HUMAN_DECISIONS.md`
    created as the approval gate (status: NOT YET APPROVED), collecting every AI-proposed value with a
    blank decision column and a sign-off block; Obsidian session note, project-log entry, run-matrix
    experiment note (raw outcomes only), design-decision note and two learning notes.

## 2026-09-03 — overnight machine sleep, and a second recovery

18. **The machine slept overnight; nothing was lost.** Between roughly 23:05 UTC on 2026-09-02 and
    06:40 UTC on 2026-09-03 the host suspended. Consequences, all benign: the two remaining transformer
    runs advanced only ~3 000 steps in 7.6 hours (6.6 steps/min) instead of the ~370 steps/min they had
    been managing, and both multi-agent workflows stopped writing at 22:35 and 22:42 UTC and never
    resumed. On wake the training processes (PIDs 7880 and 10080, started 20:45 UTC) resumed at the
    normal rate for two concurrent runs (500 steps/min, 0.12 s/step) and reached step 23 000 of 25 000.
    No run was restarted, no checkpoint or log was overwritten, and the completed 18 runs were untouched.
19. **State of the interrupted analysis work, measured rather than assumed** (all five test files run):
    * `test_derivations` 47 checks pass, `test_driver` 19 pass, `test_statistics` 94 pass,
      `test_logit_formula_fit` **122 pass** — so the completion agent *had* fixed the `IndexError` in the
      square-wave phase fit before it died. That work is real and was kept.
    * `test_function_agreement` fails at **import**: a reviewer had upgraded
      `analysis/function_agreement.py` to v1.1 (adding `ModelSide`, `compare_sides`, `error_structure`,
      `symmetric_share_control`, `compare_all_checkpoints`, and representing an undefined ratio as `None`
      instead of the constant `EMPTY_UNION_JACCARD`) and was killed before updating the test. The module
      is the intended state; the test lags it.
    * `tests/test_metrics.py` and `tests/test_wave_fitting.py` were never written, although both modules
      exist and are substantial (483 and 603 lines; `metrics.py` has already resolved the IPR definition
      from the Doshi source note).
    * The three audit documents carry **no** reviewer marks and the Nanda and Doshi source notes carry no
      verifier marks: all six audit reviewers and both verifiers died before doing anything.
20. **Recovery, second round.** Four agents launched directly (not as a workflow): write the missing
    `test_metrics.py` and `test_wave_fitting.py`; repair `test_function_agreement.py` against the module's
    v1.1 API; review all three audit documents under both lenses (evidence grading and numeric
    traceability, recomputing at least ten numbers from the raw artifacts); and verify the Nanda and Doshi
    notes against the re-fetched sources, with the restricted-loss split and the IPR object and formula
    named as the load-bearing items. Nothing already written is being rerun.
21. **Primary block complete, 2026-09-03 06:53 UTC — 20 of 20 runs, none failed.** Verified: all 20
    manifests validate; each run has 21 checkpoints with all seven roles assigned; **every paired seed
    shares its training split hash**, which the paired design depends on. Transformer generalization
    crossings span 5,625–10,275 with final test accuracy 0.9971–1.0000; MLP crossings span 8,150–10,075
    with final test accuracy 1.0000 in all ten. Memorization is 140 in nine transformer seeds and 150 in
    one; 160 in every MLP seed. Every run reached the full 25,000-step budget. Block summary:
    `training/results/matrix_primary_20260902T164330Z.json`. The `confound` block (18 runs) started
    immediately; `param_matched` (10) and `twohot` (3) are queued. **No comparison, no structure metric
    and no statistic has been computed from these runs** — that waits for the analysis-code freeze.
    Caveat recorded: the elapsed time of transformer seeds 8 and 9 (10.12 h) spans the overnight
    suspension of entry 18 and is not a compute measurement.
22. **Documents completed while the agents ran:** `docs/PREREGISTRATION.md` (expansion of the committed
    brief, with the order deviation disclosed), `docs/STATISTICAL_ANALYSIS_PLAN.md` (eight pre-specified
    primary comparisons, multiplicity handled without leaning on p-values, interval-consistent timing
    bounds), `docs/CAUSAL_ABLATION_PLAN.md` (per-architecture tables, size-matched controls, and the
    interpretation rule for a non-damaging ablation), `docs/LIMITATIONS.md`,
    `docs/HUMAN_INTERPRETATION_TEMPLATE.md`, `docs/LABBOOK_TEMPLATE.md`, and the `PROGRESS.md` entry in
    the project's own format.

## 2026-09-03 (later) — wave 1 complete, two defects found and fixed

23. **All four recovery agents died** — two stalled on a watchdog, two on API 529 overload — but
    substantially more of their work had landed than their final messages suggested. Measured rather
    than assumed: `test_metrics.py` existed and passed 242 checks, `test_function_agreement.py` had been
    repaired to the module's v1.1 API and passed 127, `logit_formula_fit` passed 122. Only
    `tests/test_wave_fitting.py` was genuinely missing, so it was written directly rather than by a
    fifth agent. One agent left a useful diagnosis before dying: a test failure it had hit was a Windows
    MAX_PATH limit in the scratch directory, not a module bug.
24. **Defect found in `wave_fitting.odd_harmonics_1_over_j` and fixed.** The model documented as "the
    ideal-square-wave Fourier truncation" omitted the alternating sign `(-1)^((j-1)/2)`, so it was not
    one. It fitted a square wave with R² = 0.571 — *worse than a plain sinusoid* (0.811), which is
    impossible for a correct `1/j` truncation. With the sign restored it reaches **0.9499** against the
    analytic `j ≤ 7` ceiling `(8/π²)(1 + 1/9 + 1/25 + 1/49) = 0.9496`, and matches the free
    nine-parameter fit to four decimals. Found by writing the test, not by reading the code.
25. **A caveat pinned rather than avoided.** At `p = 113` the odd-harmonic family of `k = 6, 19` and
    `51` aliases onto the fundamental 18 (`3·6`, `5·19 → 18`, `7·51 → 18`), so those *wrong*
    fundamentals fit a `k = 18` sinusoid perfectly. "The harmonic model fits at `k`" therefore does not
    identify `k` as the fundamental — a caveat that bears directly on `frequency_sensitivity` and on H3,
    now asserted in `tests/test_wave_fitting.py`.
26. **The long-standing `test_core.py` failure is resolved, and a second defect it had masked.**
    `fourier.harmonic_shape` is set-based, and six harmonics of the canonical fundamental set alias onto
    other members of that set; those are subtracted from both shares, so a clean square wave reports an
    even share of 0.083 against a true 0.0005 (≈165x) and a separation of 0.092 against the 0.10 the old
    check demanded. The check asserted a property the statistic cannot have. It is replaced by checks on
    the diagnosed behaviour plus checks that the replacement, `metrics.harmonic_shares` (per curve, own
    fundamental, 0 collisions at prime `p`), reports even 0.0005 and separation 0.139. Because
    `test_core.py` exits at its first failure, everything after that check had never run; fixing it
    exposed a masked defect in the `u_a` reference check, which computed its reference in the stored
    float32 while `effective_curves` works in float64 — a 1.9e-9 gap against a 1e-9 tolerance. The module
    was right, the check was not.
27. **Test suite state: 9 of 9 files pass, 1,006 checks** — derivations 47, driver 19,
    function_agreement 127, logit_formula_fit 122, metrics 242, run_format_v2 114, statistics 94,
    wave_fitting 116, core 125. This is the first point in the study at which the whole suite is green.
    Wave 1 is complete; wave 2 may now begin.
28. **Wave 2 started (2026-09-03).** With wave 1 green, three agents were launched for the first wave-2
    layer: the `mlp_mechanism` extension, the new `key_frequencies` module, and the `mask_protocols` /
    `progress_measures` extension. To unblock the last two, `docs/MASK_PROTOCOL_AUDIT.md` was written
    first: it answers all nine questions of master prompt §6 for the published protocol and for each
    implemented variant, with the computed counts — the legacy outer-product mask keeps **289**
    components where the released Nanda operator keeps **17**, and deletes **3,360** of 12,769 cells
    (26.3 %) where the published one deletes **16**. That is 17× too much freedom and 210× too much
    deletion, both biased toward the reported conclusion. `same_frequency_block` is renamed
    `paper_literal_2x2_block` in reports; `sum_directions_only` matches the released operator.
    `nanda_exact` may now be implemented because the only unresolved item is the *split* of the paper's
    restricted-loss figure, which is handled by reporting all three splits rather than guessing one.
29. **Documentation state.** Twelve of the thirteen documents master prompt §18 requires now exist; only
    `CLAIM_EVIDENCE_TABLE.md` is missing, and it cannot be written before there are results. Added since
    the last entry: `PREREGISTRATION.md`, `STATISTICAL_ANALYSIS_PLAN.md`, `CAUSAL_ABLATION_PLAN.md`,
    `METHODS.md` (with a verification-status column, so a method resting on an unverified quote is
    visible), `NOVELTY_AND_RELATED_WORK.md`, `MASK_PROTOCOL_AUDIT.md`, `LIMITATIONS.md`,
    `HUMAN_INTERPRETATION_TEMPLATE.md`, `LABBOOK_TEMPLATE.md`, `HUMAN_DECISIONS.md`.
30. **Still open and honestly outstanding:** the three audit documents remain unreviewed and the Nanda
    and Doshi source notes only partly verified (all six reviewers and both verifiers died to usage
    limits or API overload); `METHODS.md` marks every row that rests on them. The confound block was
    still running at 16:51 UTC, ten hours in on its first eight runs.
