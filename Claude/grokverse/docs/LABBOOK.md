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
