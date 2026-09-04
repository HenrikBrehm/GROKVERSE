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

## 2026-09-03 (evening) — editor crash, recovery, and the transformer mechanism module

31. **The editor crashed; the measured state, not the assumed one.** Visual Studio died mid-wave-2.
    Nothing in the repository was lost: the three wave-2 agents of entry 28 had all landed their work
    (1,463 changed lines across `mask_protocols.py`, `mlp_mechanism.py`, `progress_measures.py` plus
    the new `key_frequencies.py`, `mlp_mechanism_activation.py` and three test files). Running the
    suite rather than trusting the agents' last messages: **11 of 12 test files passed**, and the one
    failure was in `test_mask_protocols.py` — the file belonging to the agent that was interrupted.
32. **The training chain was restarted, and it lost compute but no data.** The crash killed the
    `run_matrix_chain.ps1` chain that entry 30 left running. It was restarted at 17:17 UTC and is
    running the `confound` block's first eight runs again. The manifest is the evidence that nothing
    was corrupted or duplicated: **20 completed** (the primary block), **16 legacy**, **8 running**,
    and the eight running processes are eight *distinct* configurations under one matrix driver. The
    earlier confound attempt had completed **zero** runs in its ten hours, so the restart re-does work
    that had produced no artifact; roughly ten hours of CPU is the whole loss. (The doubled process
    list is a venv shim re-executing the base interpreter, not a second chain — checked by parent PID
    before concluding anything.)
33. **Two defects in `test_mask_protocols.py`, the second one masked by the first.** The failing check
    asserted that excluding an ideal circuit's own key frequencies leaves accuracy at exactly `1/p`.
    The exclusion is in fact perfect — it annihilates the circuit to `max|logit| = 1.7e-13` against an
    amplitude-40 field, and the loss equals `log p` to `1.5e-14` — but the *argmax* of that residue is
    driven entirely by rounding noise: it spreads over all 23 classes and landed at `2/529`, not
    `1/23`. The check asserted a property the quantity cannot have — the same failure mode as entry 26,
    and it is now replaced by checks on what the operator does guarantee (loss `= log p`, annihilation
    relative to the circuit's own scale, and accuracy carrying no signal). Because the file exits at
    its first failure, everything after it had never run; fixing it exposed a second defect, an
    `IndexError` from feeding a deliberately cheap **3-channel** grid to `masked_ce_and_acc`, which
    needs the class axis to be the label axis. The projection identities stay on the cheap grid; the
    loss check got its own `[p, p, p]` grid. **197 checks now pass** in that file.
34. **`analysis/transformer_mechanism.py` written (INTERFACES §6, master prompt §8) — 67 checks.**
    This is the module that makes the comparison symmetric: master prompt §8 forbids assuming the
    Fourier circuit for the transformer and testing only the MLP. Every §5 battery is applied to the
    transformer by *calling the same functions*, so no threshold or metric definition can drift
    between architectures.
    * The forward decomposition is exact: `logits = r_pre @ W_U + sum_f h_f (W_out @ W_U)[f]`
      reproduces the model to **2.8e-13** on logits of scale 270 against the same weights in float64
      (the 1.5e-4 against the stored float32 checkpoint is float32 accumulation in the reference, and
      both bounds are asserted so a regression cannot hide in the loose one).
    * That split has the *same additive form* as the MLP's, with the neuron→logit map `W_out @ W_U` in
      place of `W_out` and the direct attention-only path in place of the output bias. So
      `logit_contributions` applies unchanged — and it does: the variance shares plus the bias share
      sum to **1.0000000000000** on a real checkpoint, which is the identity that proves the
      generalization correct rather than merely type-compatible.
    * **Where the architectures genuinely differ is recorded, not smoothed over.** The MLP's
      pre-activation is exactly `u_a + u_b + b_in`; the transformer's is not, because attention depends
      on the input. The comparable curves are built at the grid-mean attention and `additivity_r2`
      reports per neuron how much of the true pre-activation that captures. **The MLP's value is
      identically 1 by construction, so the two numbers are not symmetric evidence** and the module
      says so. The activation batteries therefore run on the model's **true** `hidden`, never on the
      rectified reconstruction.
35. **The enabling refactor was made in the shared code, not duplicated into the new module.**
    `sum_dependence`, `logit_contributions` and `activation_analysis` each rebuilt the activation from
    the curves, which is right for the MLP and wrong for the transformer. All three now take an
    optional `act=`, and `logit_contributions` additionally accepts a `[p, p, p]` bias term. Both
    default to the previous behaviour, and the whole suite stayed green across the change — the point
    of doing it this way is that the MLP and the transformer are now measured by literally the same
    code paths.
36. **A documented placeholder was closed.** `key_frequencies.operand_curves` had been returning
    `W_E[:p] @ W_in` for the transformer — no attention, no `W_pos`, and the *same* curve for both
    operands — labelled in its own docstring as a placeholder for this module. It now calls
    `transformer_mechanism.effective_curves` and records that in the provenance string. On a real
    checkpoint the `neuron_clusters` rule changes its key set from the placeholder's to `[11, 13, 15,
    18]`, so this was not a cosmetic substitution.
37. **Test suite: 13 of 13 files, 1,438 checks.** Up from the 1,006 of entry 27 and the 1,371 wave-2
    layer 1 reached; `transformer_mechanism` adds 67.
38. **What was run on real data, and what it is NOT.** `transformer_mechanism.analyse` was executed on
    one real checkpoint (transformer seed 0, step 25,000) purely to confirm the module runs on a
    trained model and that its internal identities hold there. It does — 29 s, forward check exact,
    shares summing to 1. **These numbers are a pipeline check, not a result.** The analysis code is
    not frozen, the run is one seed of twenty, and entry 21's rule still stands: no comparison and no
    structure statistic enters the study until the freeze. The seed-0 output file exists on disk and
    was produced by unfrozen code, which is recorded here so it cannot later be mistaken for a
    measurement of record.
39. **Still open.** `structure_over_time.py`, `h3_validity.py` and `causal_ablation.py` are the three
    modules the driver still reports as missing; `CLAIM_EVIDENCE_TABLE.md` still cannot be written
    before there are results. The three audit documents remain unreviewed and the Nanda and Doshi
    source notes only partly verified (entry 30) — unchanged, and `METHODS.md` still marks every row
    that rests on them.

## 2026-09-03 (night) — handoff prepared for the next session

40. **Model switch and a change of task.** The human author switched the session to a different model
    and asked for everything to be *made ready* for a following session that will do the remaining
    work unattended by them, without doing that work now. This section is the record of that
    preparation; nothing scientific was computed.
41. **The overnight sleep of entry 18 had a cause, and it is fixed.** `powercfg` showed
    `standby-timeout-ac = 0x384` — the machine suspends after 15 minutes of idle input regardless of
    CPU load, which is exactly what stalled two transformer seeds for 7.6 hours. Standby and hibernate
    timeouts are now 0 on AC and DC (`powercfg /change ...`, verified by re-query). Recorded here
    because the manifest's `elapsed_seconds` for seeds 8 and 9 (entry 21's caveat) is explained by it.
42. **`docs/HUMAN_DECISIONS.md` was deleted by the human author and restored unchanged from commit
    `6841a99`.** The deletion cost nothing scientific — every threshold and gate rule is also in
    `PREREGISTRATION.md` §4.2/§4.3/§5 and in code constants — but ten files reference the document,
    including the `status` strings of both mechanism modules, and its sections F and sign-off are the
    only place the division of labour for `AI_DISCLOSURE.md` can be recorded. Its status remains
    `NOT YET APPROVED`. Clarified in the conversation and repeated here: the approval gate labels
    results, it never blocked the analysis work.
43. **Preparation, measured.** The driver plans cleanly over all 20 primary runs at both measurement
    points (`--dry-run`): 60 outcomes are "missing" — `causal_ablation`, `h3_validity`,
    `structure_over_time` × 20 runs — and 20 are the expected cross-architecture skips. Found while
    preparing: (i) **`h3_validity` has no INTERFACES section** although the driver lists it; the
    handoff instructs the next session to write the spec (from PREREGISTRATION §3 H3) as INTERFACES
    §14 *before* implementing; (ii) INTERFACES §13 specifies four further modules (`aggregate`,
    `figures_study`, `decision_tree`, conditional `bounded_alternative`) that no earlier entry counted
    among the missing ones; (iii) `tests/test_driver.py` lines 89–92 pin `causal_ablation` and
    `structure_over_time` as *missing* and will fail by design when they land. Written:
    `training/run_analysis_chain.ps1` (detached driver launcher that logs the launch commit as the
    freeze evidence, waits for the training chain unless `-SkipWait`, runs `aggregate` if present);
    `docs/dev/HANDOFF_2026-09-04.md` (state, environment, staged work plan A–G with the spec pointers
    and reuse map for each module, the traps of this week, session-end duties, and the list of what
    only the human may fill in). The primary-block glob `*_frac0.3_seed*_arch25k` resolves to exactly
    the 20 primary runs (checked).
44. **Committed.** The full suite was run once more (13/13, 1,438) and the working tree — the three
    interrupted agents' wave-2 work, the transformer mechanism module, the test repairs, the launchers
    and the documents — was committed on `arch-study`. `training/results/` stays untracked until the
    training chain has finished rewriting the manifests. Training at the time of the commit: confound
    block 8/18 running, `param_matched` and `twohot` queued.

## 2026-09-03 (night) — the remaining modules, and the analysis-code freeze

45. **Stage A: `analysis/causal_ablation.py` (INTERFACES §9, master prompt §11), 91 checks.** Every
    ablation id of `CAUSAL_ABLATION_PLAN.md` §4 (nine, MLP) and §5 (eight, transformer). Every one
    carries a size-matched random control matched on the number of **live** neurons kept, which makes
    the same control size-matched in both directions at once. Two design points worth recording: `z`
    is written as JSON `null` with its reason when the control has zero spread — `common._jsonable`
    forbids non-finite values by design — and the z-condition then falls back to *exceeding every
    control* rather than being waved through as infinitely significant; and gate criterion G4 is
    `null`, never a silent `False`, when an ablation id is absent.
46. **The plan's §6 rule visibly did its job on the first real checkpoint.** On MLP seed 0 at step
    25,000, `remove_structured` drops test accuracy by 0.992 — but the size-matched control drops it
    by 0.866, because the structured set is 451 of 512 neurons and removing 88 % of any network
    destroys it. The rule therefore reads *not necessary*, exactly as intended. `keep_structured`
    retains accuracy 1.000, and so does a random 451-neuron control. This is a **pipeline check on
    unfrozen code**, not a result; it is recorded because it shows the "removing more removes more"
    guard working rather than being asserted.
47. **Stage B: `analysis/h3_validity.py` — and the INTERFACES §14 spec it never had.** The driver
    already listed `h3_validity`, but no interface section existed for it. The spec was written
    first, derived from `PREREGISTRATION.md` §3 and master prompt §12 (both of which predate any
    run), and the deviation is disclosed inside the section. Two things the implementation only got
    right by measuring: `metrics.harmonic_shares` defines `family_share` as fundamental + **odd**
    harmonics — cardinality **4**, not the 7 that `harmonic_indices` returns — so the matched control
    sits at top-4, exactly as the pre-registration says; and the honest noise control is what the odd
    harmonics add *on top of* the fundamental (measured 0.0547 against 3/56 = 0.0536 for three random
    frequencies), because the family always contains the dominant frequency and would otherwise
    flatter itself even on noise. The forbidden criterion is asserted as an inequality on real
    populations and real weights, not trusted.
48. **Stage C: `analysis/structure_over_time.py` (H4), 56 checks.** Walks all 21 checkpoints of a v2
    run in 2 m 53 s. The onset rule of INTERFACES §12 takes its baseline std over **two** checkpoints,
    which is a very weak estimate, so the module reports `init_value`, `baseline_std` and
    `n_baseline` next to the step and calls it an onset indicator, not a transition. `phase_relation`
    short-circuits when the structured set is empty — the normal state at `init` — so the missing
    resultant length travels as `null` with `insufficient_neurons` set, never as a fabricated 0. The
    whole analysis is labelled CORRELATIONAL in its own output.
49. **A latent bug in four modules, found by the fifth.** `causal_ablation`, `h3_validity` and
    `transformer_mechanism` all read `checkpoints.json` as `{"checkpoints": [...]}` when it is a
    plain list. It only affected the `--all-checkpoints` CLI path, which the driver never takes, so
    nothing had failed; `structure_over_time` reads the file in its normal path and exposed it.
50. **Stage D: the driver test no longer names unimplemented modules.** `tests/test_driver.py` lines
    89–92 asserted that `causal_ablation` and `structure_over_time` are reported *missing*, and so
    failed the moment one of them landed. The property worth pinning is the mechanism — an
    unimplemented module is surfaced, never dropped — so it is now asserted against a name that can
    never exist plus the driver's own list resolved dynamically. **The driver now reports NO missing
    modules.**
51. **Analysis code FROZEN at `0b55e1d`, recorded in `PREREGISTRATION.md` §12 (2026-09-03).** The
    freeze covers every module the driver invokes and the shared code they rest on; it explicitly does
    **not** cover the aggregation layer of INTERFACES §13, which consumes outputs and changes no
    measured number. Suite at the freeze: 16 files, 1,657 checks. Pilot outputs written earlier while
    the code was still moving are named in §12 as pipeline checks to be regenerated by the frozen
    driver.
52. **Analysis driver launched detached, 2026-09-03T20:34:04Z**, over the 20 primary runs at both
    pre-registered measurement points: **280 calls**, 4 workers, `results/analysis_chain.log` recording
    commit `1e3f646` and a clean tree. The chain pins `OMP/MKL/OPENBLAS/NUMEXPR` to one thread per
    process, because the training chain holds 8 of the machine's 12 logical CPUs and an unpinned BLAS
    pool per worker would thrash it. Training was unaffected and continued at 8 concurrent runs.

## 2026-09-03 (late) — the aggregation layer, and two modules that were failing silently in the driver

53. **Stage F complete: `aggregate`, `decision_tree`, `figures_study`, `bounded_alternative`.**
    `aggregate` walks every per-run JSON into one table per module; it extracts and never recomputes,
    never averages a seed away, and distinguishes **missing** from **not applicable to this
    architecture** (mixing the two inflates the missing count and makes completeness look worse than
    it is). `decision_tree` applies the PREREGISTRATION §5 gate and names the §6 branch, with every
    constant copied from the pre-registration and echoed into the output. `figures_study` draws
    exactly the eight paired comparisons `STATISTICAL_ANALYSIS_PLAN.md` §3 fixes in advance — a test
    asserts the count is eight, because a ninth headline figure would quietly widen the study's
    claims. `bounded_alternative` implements §6.3's four pre-committed questions and no fifth.
54. **Three-valued logic in the gate, and why it is not a detail.** Every criterion returns `True`,
    `False` or `None` = *not evaluable*, and `None` is never coerced to `False`. Counting an untested
    criterion as a failure manufactures evidence against the hypothesis exactly as surely as the
    reverse manufactures evidence for it. Two consequences the first implementation got wrong and the
    live data exposed: with fewer than ten seeds analysed the gate must read **undetermined**, not
    `neither_passes` — that would be a claim about the architectures when the only fact is that the
    driver has not finished; and the measurement point must be resolved against each run's own
    metadata rather than the file tag, because both checkpoints carry a `step` tag and one seed was
    entering the gate arithmetic twice.
55. **A degenerate control found in the real outputs, and reported rather than passed.** G3's second
    condition asks that a formula's `r2_test` exceed the random-frequency control's 95th percentile.
    In some `logit_formula_fit` outputs that control draws `set_size = 56` of the 56 frequencies
    available at `p = 113` — the odd-harmonic family of a large key set aliases onto the whole
    spectrum — so the "random" set is the same set every time, its standard deviation is ~1e-16, and
    it cannot discriminate anything. `decision_tree` detects this and reports the condition **not
    evaluable**. It is a limitation of the pre-registered G3 control at this prime and key-set size,
    and it belongs in `LIMITATIONS.md`.
56. **Two modules were failing on every driver call, and the file counts are what showed it.**
    `key_frequencies` had written only the two pre-freeze pilot files while `logit_formula_fit` had
    24 — yet `key_frequencies` runs *first* in the driver order. Cause: `analysis/driver.py` passes
    `seed` to every module it invokes, and both `key_frequencies.analyse` and
    `progress_measures.compute_from_checkpoints` accepted `**kw` and forwarded it into a helper that
    rejects `seed`. Every one of their **60 calls** (40 + 20) raised `TypeError`. The driver recorded
    them as failed and correctly carried on, so nothing was corrupted — but nothing was loud either.
    The 12 `progress_measures` files on disk had been written by `structure_over_time`, which calls
    that function internally without a seed.
57. **Both fixed as post-freeze bug fixes under PREREGISTRATION §9**, which permits "genuine bug
    fixes — each logged in the labbook with its reason, and every affected analysis re-run on all
    seeds". Both now accept `seed`, **record it, and do not use it**: no §4 selection rule and no
    restricted/excluded protocol draws a random number, so no result depends on it. No threshold,
    definition, pass rule, control or statistic changed. **The affected analyses must be re-run**:
    the driver process running since 20:34 UTC holds the old code in memory, so its
    `key_frequencies` and `progress_measures` calls will still fail and both modules need a re-run
    over every seed once it exits.
58. **A contract test that catches this class of bug, added to `test_driver.py`.**
    `inspect.signature` cannot catch it — both modules accepted `**kw`, so the signature bound
    cleanly and the failure happened at runtime inside the call. The new check therefore *invokes*
    every module the driver lists, on a small synthetic v2 run of each architecture, with exactly the
    kwargs the driver passes, and asserts the failure is never a `TypeError` about those arguments.
    A module may legitimately raise something else on a tiny synthetic run; that is allowed and
    labelled. Runtime 22 s.
59. **Suite: 20 of 20 files, 1,810 checks.**

## 2026-09-04 — the primary block is analysed, and the evidence gate is evaluated

60. **The driver's first full pass exited 1: 196 ok, 84 failed, 20 skipped.** All 84 were module
    defects, not data problems, and all are fixed: `key_frequencies` (40) and `progress_measures`
    (20) rejected the driver's `seed` argument (entries 56–57); `wave_fitting` (24) failed two
    different ways — an `IndexError` on 4 MLP runs at the crossing checkpoint, and 20 calls on
    transformer runs it was never able to serve. The re-run at commit `eeeef82` was **80 ok, 0
    failed**, and the primary block is now complete: 40 rows each for `key_frequencies`,
    `logit_formula_fit`, `h3_validity`, `causal_ablation`; 20 each for the two mechanism modules,
    `wave_fitting` (MLP-only), `structure_over_time` and `progress_measures`. **0 missing.**
61. **The `wave_fitting` IndexError had been predicted in writing and left unfixed.**
    `mlp_mechanism_activation.fit_curve_matrix` exists solely to avoid it, and its docstring names
    this exact case — "reachable at `p = 113` whenever the smallest dominant frequency is shared by
    exactly four neurons" — but closes with "``wave_fitting.py`` belongs to another work package and
    is not edited from here; the defect is reported instead." Every other caller was protected; the
    one that was not lost 4 of 20 runs. `analyse` now goes through the per-k split.
62. **Evidence gate at the final checkpoint (`results/decision_tree_final.json`): neither
    architecture passes.** Per criterion, over 10 paired seeds:

    | criterion | transformer | MLP |
    |---|---|---|
    | G1 periodic structure | 10/10 | 10/10 |
    | G2 phase addition | 10/10 | 10/10 |
    | G3 end-to-end logit fit | 10/10 | 10/10 |
    | **G4 causal** | **0/10** | **0/10** |

    Decision-tree branch, per PREREGISTRATION §6: **`neither_passes`** — "report that the Fourier
    evidence tested does not identify the learned mechanisms, and investigate the validity of the
    existing metrics". That branch was fixed before the experiments and is taken as written.
63. **Why G4 fails is a fact about the structured-neuron definition, not about the models.** Under
    the pre-registered primary definition the structured set is **442 of 512 neurons (86 %) in the
    MLP and 501 of 512 (98 %) in the transformer** — for transformer seed 5 it is all 512. At that
    size neither necessity nor sufficiency can discriminate, and the size-matched control says so
    directly (medians over 10 seeds, final checkpoint):

    * MLP `remove_structured`: observed drop **0.991**, size-matched control drop **0.873**;
    * transformer `remove_structured_neurons`: observed **0.942**, control **0.916**;
    * both `keep_structured`: retained accuracy **1.000** / **0.9997** — but a random set of the same
      size also retains it (control drop 0.000).

    This is `CAUSAL_ABLATION_PLAN.md` §6's "removing `C` and the control do comparable damage → `C`
    is **not** specifically load-bearing" outcome, and §9's "removing more removes more" guard doing
    exactly its job. For transformer seed 5 no control set exists at all (the kept set is empty), and
    that is recorded as `null` rather than fabricated.
64. **The key-frequency ablations, by contrast, are sharply discriminating — and this is where an
    AI decision changed a verdict.** Medians over 10 seeds at the final checkpoint:

    | ablation | drop | control | necessary |
    |---|---|---|---|
    | MLP `remove_key_freqs_from_curves` | 0.989 | 0.000 | **10/10** |
    | transformer `remove_key_freqs_from_embedding` | 0.984 | 0.0005 | **10/10** |
    | transformer `remove_key_subspace_from_residual` | 0.235 | 0.000 | **0/10** |
    | transformer `keep_key_freqs_in_embedding` | 0.004 | 0.990 | (sufficiency) |
    | transformer `restricted_circuit_only` | −0.000 | 0.991 | (sufficiency) |

    `PREREGISTRATION.md` §5 names G4's second condition "`remove_key_freqs`" but does **not** say
    which transformer ablation that is — the transformer has three candidates. I wired
    `remove_key_subspace_from_residual` into `causal_ablation._gate_g4`; had I wired
    `remove_key_freqs_from_embedding`, which is arguably the closer analogue of the MLP's
    filter-the-input-representation ablation, that condition would read 10/10 instead of 0/10. **The
    overall gate verdict does not change** — both architectures still fail on `remove_structured` —
    but the reported reason does. The mapping is an AI choice that materially affects a criterion, so
    it is added to `docs/HUMAN_DECISIONS.md` as D5 rather than left implicit, and it is **not**
    changed here: swapping it after seeing the numbers is exactly the post-hoc tuning §3.9 forbids.
65. **At the crossing checkpoint the gate is undetermined**
    (`results/decision_tree_crossing.json`): the transformer fails (G1 9/10, G2 10/10, G3 10/10,
    G4 0/10) while the MLP is **not evaluable** — its G3 is untestable on all 10 seeds because the
    random-frequency control draws all 56 available frequencies at `p = 113` (entry 55), and its G1
    passes 0/10. Per the module's own rule an unevaluable criterion is not a failure, so no branch is
    taken at that point.
66. **Aggregation and figures.** `results/aggregate/` holds one table per module with **every seed
    shown** and `TABLES.md` beside it; `results/figures/` holds all **11** figures — the eight
    pre-specified comparisons of `STATISTICAL_ANALYSIS_PLAN.md` §3 plus the gate summary — each
    naming its source files in its caption, drawn from the aggregate tables only. Two wiring gaps
    were found and fixed while drawing them: the H2 comparison had no transformer side (its wave
    fits live in `transformer_mechanism.wave_fits`, not in `wave_fitting`), and the H5 figure read
    only the MLP's ablation id. Both had produced "no seed has both architectures", which reads like
    absent data rather than a naming difference.
67. **Statistics computed (`results/statistics.json`, `analysis/statistics_report.py`, 47 checks).**
    Exactly the eight pre-specified comparisons of `STATISTICAL_ANALYSIS_PLAN.md` §3 — ten rows,
    because two are reported at both measurement points — each with all ten per-seed differences
    listed, a 10,000-draw percentile bootstrap CI, exact sign and Wilcoxon tests, Cohen's `d_z`,
    Cliff's delta and the robust spread. The module reuses `figures_study`'s specifications and
    extractors, so a number in the statistics file and the number on the matching figure cannot
    disagree. Holm-adjusted p-values are reported and explicitly labelled a descriptive aid; §4 makes
    p-values companions, never evidence.
68. **The timing comparison separates two things that an earlier version of my own code conflated.**
    §6 asks for the bounds implied by test accuracy being evaluated only every 25 steps. That is a
    different question from whether the seeds agree. On the real data the answers differ, and the
    difference is the finding: **all 10 seed intervals exclude zero** — the evaluation grid never
    flips a seed's sign — but the direction is **not unanimous**. Nine seeds have the transformer
    crossing earlier (median −1,562 steps), and **seed 4 reverses it** (transformer 10,275 vs MLP
    8,650, +1,625). The first version reported per-seed unanimity under the name "survives interval
    bounds", which would have described a seed-spread limitation as a grid limitation. The two are
    now reported separately, and the output says in words that the claim is about the typical
    difference under the conditions examined, not a universal one.
69. **The eight comparisons, as measured** (transformer minus MLP, final checkpoint unless noted;
    medians with 95 % bootstrap CIs, all ten seeds behind each):

    | # | quantity | median difference | CI 95 % | note |
    |---|---|---|---|---|
    | 1 | generalization step | −1,562 | [−2,895, −822] | excludes 0; **not** unanimous (seed 4 reverses) |
    | 2 | grokking gap | −1,542 | [−2,875, −806] | excludes 0 |
    | 3 | structured fraction (crossing) | +0.303 | [+0.257, +0.337] | excludes 0, unanimous |
    | 3 | structured fraction (final) | +0.085 | [+0.060, +0.100] | excludes 0, unanimous |
    | 4 | phase-relation `R` (crossing) | −0.0113 | [−0.0132, −0.0092] | excludes 0, unanimous |
    | 4 | phase-relation `R` (final) | −0.0074 | [−0.0098, −0.0056] | excludes 0, unanimous |
    | 5 | square/odd-harmonic best-fit fraction | −0.324 | [−0.397, −0.120] | excludes 0 |
    | 6 | family minus top-1 fraction (H3b) | −0.0025 | [−0.0060, +0.0035] | **spans 0** |
    | 7 | ablation damage above control (H5) | −0.079 | [−0.093, −0.039] | excludes 0; n = 9 |
    | 8 | best Fourier-formula R² | −0.426 | [−0.459, −0.336] | excludes 0, unanimous |

    These are measurements, not conclusions. Nothing here may be read as H3 evidence: the evidence
    gate reports `neither_passes` (entry 62), and master prompt §12 makes harmonic-family
    interpretation conditional on passing it. Comparison 7 has nine pairs because transformer seed 5
    has no size-matched control at all — every one of its 512 neurons is "structured" (entry 63).
70. **H3 evaluated at study level (`results/h3_report.json`, `analysis/h3_report.py`, 26 checks) —
    and it is REFUTED by its own pre-registered criterion.** `PREREGISTRATION.md` §3 fixes two
    refutation conditions; both were applied literally, in both directions:

    * **Criterion 1** — (architecture gap under top-1) minus (gap under the family definition):
      median **+0.0049**, CI95 **[+0.0012, +0.0076]**, which *excludes* zero. This branch does **not**
      refute H3: the two definitions do give different gaps.
    * **Criterion 2** — is the MLP's family-based structured fraction still lower than the
      transformer's with a CI excluding zero? Median **+0.0850**, CI95 **[+0.0600, +0.1004]**. Yes.
      **This refutes H3**: by the criterion fixed before the experiments, the deficit is "a real loss
      of structure, not a measurement artifact".

    The substantive number behind that verdict: the architecture gap is **+0.0898** under the top-1
    definition and **+0.0850** under the harmonic-aware family definition. The harmonic-aware
    definition closes **5.4 %** of the gap. H3 predicted it would close.
71. **The metric's waveform sensitivity is real, and it is not enough.** H3a measures exactly what H3
    proposed: two populations built from each checkpoint's own `(k, phase, amplitude)`, differing only
    in waveform, give a top-1 concentration difference of **+0.1893** — identical for both
    architectures, as it must be, since it is a property of the *metric* and not of the model. So the
    mechanism H3 hypothesised is genuinely present in the measuring instrument; it simply does not
    account for the architecture gap that the study set out to explain.
72. **None of this may be read as mechanism evidence, and the report says so itself.** The evidence
    gate reports `neither_passes` (entry 62), so under master prompt §12 no architecture is cleared
    for a harmonic-family mechanism reading. `h3_report` reads `decision_tree_final.json` and states
    at the top of its own output that the architectures cleared are **NONE** and that every number
    below is a measurement about the metric. A missing gate file is treated as the most restrictive
    case, not a neutral one.
73. **A labelling error caught before it reached a report.** The first version of `h3_report` named
    its verdict block `h3_0_verdict` and attached a note claiming that refuting H3₀ is what H3 needs.
    The pre-registration lists those criteria as conditions under which **H3** is refuted; H3₀ is the
    null. Naming it that way would have inverted the study's headline. The block is now `h3_verdict`,
    it states H3's prediction and what H3₀ is, and `tests/test_h3_report.py` pins the direction of
    both criteria against inputs whose answer is known by construction.
74. **The forbidden criterion audited across all 40 rows.** `family − matched_top_m` never exceeds
    **1.11e-16**, i.e. zero to floating point, exactly as the construction requires. Reported as an
    implementation audit; never as support.
75. **Definition-of-Done item 8 was genuinely undone, and is now measured.** The full-domain function
    comparison had exactly one artifact on disk, and it was for the *pre-study* runs (ids without
    `_arch25k`). `function_agreement` is not in `analysis/driver.py`'s module list — it compares two
    runs and does not fit the per-run pattern — so the driver never covered it. Run over the 10
    paired seeds at the final checkpoint: median agreement over all **12,769** inputs **0.99977**;
    the **MLP is correct on every cell in all 10 seeds**; the transformer has 0–26 errors (median 3);
    **4 of 10 seed pairs agree on every single input**. `error_jaccard` is 0.000 where defined and
    **undefined in 4 seeds** because neither model errs — the module returns `None` for an empty
    union rather than a fabricated constant, which is what makes that visible.
76. **Master prompt §10's restriction is now discharged by measurement, and it does not license the
    stronger phrase.** §10 permits only "both architectures generalize on the same task" until the
    full-domain comparison exists. It now exists, and it says the two agree on 99.98 % of inputs but
    are **not** identical in 6 of 10 seeds. So "both learn the same function" stays unsupported; what
    is supported is the agreement rate with its seed range.
77. **`docs/CLAIM_EVIDENCE_TABLE.md` written** — the last of the thirteen documents master prompt §18
    requires. Eight claims (C1–C8), each in the §21 six-part structure, plus a table of claims that
    may **not** be made from this evidence and why, and a list of what is still pending with what each
    pending item would settle. Every number traces to a file under `results/`. The table opens with
    the precondition that governs all of it: the gate reports `neither_passes`, so no mechanism
    reading is licensed for either architecture.
78. **Confound block complete and under analysis.** The training chain finished the 18 confound runs
    at 00:40:48 UTC (exit 0) and moved to `param_matched`. The analysis driver was launched over the
    confound runs at 01:0x UTC; `param_matched` trains concurrently.
79. **H4 evaluated across seeds (`results/h4_report.json`, `analysis/h4_report.py`, 20 checks) — and
    it holds.** The ordering H4 asserts is memorization → structure onset → generalization. Medians
    over 10 seeds per architecture:

    | | transformer | MLP |
    |---|---|---|
    | memorization | 140 | 160 |
    | structured-fraction onset | 1,000 | 1,000 |
    | key-subspace-share onset | 500 | 500 |
    | generalization | 7,588 | 9,250 |

    `structured_fraction_of_live`, `embedding_top8_concentration`, `logit_key_subspace_share` and
    `median_family_fraction` each place their onset **after memorization and before generalization in
    10 of 10 seeds, in both architectures**. The change from `init` to `pre_generalization` is
    +0.450 (transformer) and +0.260 (MLP) for the structured fraction.
80. **Two metrics honestly report nothing rather than something.** `phase_relation_R` has **no**
    defined onset on any seed, because at `init` the structured set is empty and the resultant length
    is `null` rather than a fabricated 0 (entry 48) — so the onset rule has no baseline to work from.
    `fraction_best_aic_square` is flat, so no onset fires. Both are reported as `0/0` with the
    denominator visible; `median_odd_minus_even_u_a` is defined on 8/10 MLP and only **2/10**
    transformer seeds, and "2 of 2" is printed next to its denominator precisely so it cannot be read
    as "2 of 10 failed".
81. **What H4 does and does not license.** Structure demonstrably precedes generalization here, on
    every seed, for four metrics. The analysis is nevertheless **correlational** and says so in its
    own output: it cannot show the structure caused the jump, and the onset indicator's baseline
    standard deviation is taken over two checkpoints, so the step is sensitive to the checkpoint grid.
    Causal claims come from `causal_ablation` alone — where, for the structured-neuron set, they
    **fail** (entry 63). H4 and G4 are therefore not in tension: structure forms early *and* the
    pre-registered threshold selects too many neurons for an ablation to identify it.
82. **The `neither_passes` branch's obligation is running.** `PREREGISTRATION.md` §6.3 pre-commits the
    bounded alternative-mechanism analysis when no architecture passes the gate, for **both**
    architectures. `run_bounded_alternative.py` was launched over the 20 primary runs. Early results
    are uniform: the linear probe decodes `(a+b) mod 113` from the hidden layer at **1.000** while its
    shuffled-label control sits at **0.008–0.010** against a chance level of 0.009, and the hidden
    layer's entropy effective rank is ~59–66 of 512 dimensions.
83. **The bounded alternative-mechanism analysis is complete: 20 runs, 0 failed.** It answers the four
    questions `PREREGISTRATION.md` §6.3 pre-committed, and no fifth. Medians over 10 seeds per
    architecture at the final checkpoint:

    | | transformer | MLP |
    |---|---|---|
    | linear probe for `(a+b) mod 113` | **1.0000** | 0.9999 |
    | its shuffled-label control | 0.0079 | 0.0091 (chance 0.0088) |
    | one-hidden-layer probe | 0.9999 | 1.0000 |
    | effective rank of `hidden` (of 512) | **12.7** | **66.6** |
    | components for 90 % of the variance | 12 | 56 |
    | effective rank of the logits | 5.0 | 17.4 |

    The sum is decodable from both hidden layers at ceiling while the shuffled-label control sits at
    chance, so the representation genuinely carries `(a+b) mod p` in both.
84. **The two architectures differ sharply in how compactly they carry it, and the causal test
    separates them.** Removing the top-`r` singular directions of `hidden` against `r` random
    directions of the same rank:

    | r | transformer drop | MLP drop | random-direction control |
    |---|---|---|---|
    | 4 | +0.0002 | 0.0000 | 0.0000 |
    | 8 | **+0.4616** | 0.0000 | 0.0000 |
    | 16 | **+0.9644** (exceeds every control in 10/10 seeds) | **0.0000** (0/10) | 0.0001 |

    The transformer's computation lives in a subspace of roughly 16 dimensions and is destroyed by
    removing it; the MLP is **completely unaffected** by removing its top 16 directions. That is
    consistent with the effective ranks (12.7 vs 66.6) and is a genuine architectural difference, not
    a metric artifact — the random-direction control does nothing in either case.
85. **A limitation of that ablation, stated rather than papered over.** `REMOVE_RANKS` was fixed in
    advance at `(1, 2, 4, 8, 16)`, and 16 is **below the MLP's effective rank of 66.6**. So the MLP
    result is "not damaged by removing up to 16 directions", *not* "has no low-rank structure": the
    pre-committed grid simply does not reach far enough to find where it breaks. Extending the grid
    now, after seeing that the MLP survives it, would be exactly the post-hoc tuning §3.9 forbids;
    the honest statement is the bounded one, and the gap is recorded here for the human authors.
86. **Cross-seed CKA is low in both architectures — an honest negative for question 3.** Linear CKA
    between the hidden representations of all 45 seed pairs: transformer median **0.0017** (range
    0.0003–0.5404), MLP median **0.0973** (0.0020–0.2683). The structure each model finds does **not**
    align across seeds. This is consistent with what the study already knew — the key frequencies
    differ by seed (PROGRESS.md, Phase 3) — and it means the low-rank subspace the transformer uses is
    seed-specific, not a shared basis. Low CKA here says "not the same subspace", not "no structure";
    both readings are recorded so the number is not over-read in either direction.

## 2026-09-04 — the run matrix is complete

87. **The training chain finished at 02:46:36 UTC.** All four blocks, **51 runs, every one
    `completed`, none failed**: primary 20, confound 18, parameter-matched 10, two-hot 3.
    `results/matrix_launch.log` ends with `matrix chain finished`.
88. **A naming assumption of mine was wrong, and the check that caught it was looking at the wrong
    thing.** I reported "twohot: 0 directories" from a glob on `*twohot*`; the two-hot runs are named
    `m2h_add_p113_wd1.0_frac0.3_seed{0,1,2}_arch25k`. All three had completed in ~1,255 s each with
    exit code 0. The block summary in `matrix_launch.log` was right and my glob was wrong — recorded
    because "0 directories" would have looked like a failed block to anyone reading only my summary.
89. **Confound block analysed: 234 ok, 0 failed, 27 skipped** (the skips are the architecture-specific
    modules on the other architecture, as designed). The analysis of the parameter-matched and two-hot
    blocks was launched immediately afterwards at 8 workers, the CPU now being free of training.
90. **The Grokfast × `train_frac` confound is separated (`results/controls_report.json`,
    `analysis/controls_report.py`, 22 checks).** Master prompt §14 forbids attributing a change to
    either knob while the two are confounded — the project's own history contains that mistake, a
    speed ratio moving from ~2.9× to ~1.3× between two settings that differed in **both**. The 2 × 2
    decomposition on 3 paired seeds per cell, generalization step:

    | | Grokfast main effect | `train_frac` main effect | interaction |
    |---|---|---|---|
    | transformer | **−350** [−888, +188] — spans 0 | **−6,367** [−6,950, −5,638] | +883 — spans 0 |
    | MLP | **+892** [+713, +1,063] | **−8,217** [−8,850, −7,713] | −933 |

    The training fraction moves the crossing by roughly 6,400 (transformer) and 8,200 (MLP) steps.
    **Grokfast's effect on the transformer is not distinguishable from zero**, and on the MLP it goes
    the *other* way — Grokfast makes the MLP cross ~890 steps **later**. So the historical shift is
    attributable to the training fraction, not to acceleration, and the legacy speed ratio must not be
    read as a statement about Grokfast.
91. **The caveat that governs those numbers.** Three paired seeds per cell. A percentile bootstrap
    over three values is a weak interval however tidy it looks, and "CI excludes zero" at n = 3 is not
    strong evidence. What the decomposition does establish is the *ordering of magnitudes* — a
    `train_frac` effect an order of magnitude larger than the Grokfast effect, consistently in both
    architectures — and that is what licenses the negative statement about Grokfast rather than any
    positive one. The module carries this caveat in its own `reading` field.
92. **The parameter-matched and two-hot comparisons are computed but not yet populated.** Their
    analysis was still running when this was written: `param_matched` is 10 MLP runs at `d_mlp = 572`
    to be paired against the primary transformer, `twohot` is 3 `m2h_*` runs against the
    shared-embedding MLP. `controls_report` reports them as `NOT COMPUTED — no seed has both sides`
    rather than as an empty or zero result, and re-running it after the analysis completes is all
    that is needed.
93. **The G4 failure is not an artifact of the chosen threshold — no available definition fixes it.**
    Measured across all six structured-neuron definitions the pre-registration provides, median over
    the 10 primary seeds at the final checkpoint: the transformer's structured fraction is
    **0.979–0.998** and the MLP's **0.863–0.920**, whichever definition is used. Tightening the family
    threshold from 0.50 to 0.70 moves the MLP from ~453 to ~442 of 512 neurons; Doshi's rank-matched
    IPR and Swaroop's periodicity score land in the same band. So at `p = 113` with `d_mlp = 512`
    **no definition available to this study yields a set small enough for a size-matched ablation to
    discriminate**, and a decision to change B1 would not rescue G4. What would is a different *kind*
    of definition — a fixed small cardinality, or selection by causal contribution rather than by
    spectral shape — which is a new pre-registration, not a threshold tweak. Recorded as a
    sensitivity note under `HUMAN_DECISIONS` D6 so the human authors can see what changing the value
    would and would not buy.
94. **`AI_DISCLOSURE.md` updated for the completed measurements (§22).** The status line now says the
    measurements are complete and that **no scientific conclusion has been written or reviewed by a
    human**; §5 records that the AI executed the analyses over all 51 runs and that the two AI
    decisions materially affecting a gate criterion are escalated as D5 and D6; §6 adds the
    claim–evidence table as AI-drafted and marks the `RESULTS.md` rewrite as now due. A changelog
    entry for 2026-09-04 states plainly that the session produced **two negative results the AI
    reported rather than avoided** — the gate's `neither_passes` and H3's refutation by its own
    criterion. All eleven `[HUMAN AUTHORS MUST COMPLETE]` placeholders remain untouched.
