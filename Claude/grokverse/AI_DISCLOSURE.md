# AI_DISCLOSURE.md — GROKVERSE

This project discloses, per BWKI rules and in the spirit of the EU AI Act, all AI/tooling assistance used in its creation. It is maintained from day one and updated as the build proceeds — not reconstructed at the end. BWKI requires both a *clearly identifiable, documented portion of own work* (Eigenleistung) **and** full transparency about AI use; this file serves the transparency half and points to the own-work half.

> **Status of this file, 2026-09-04.** The architecture study's **measurements are complete** (51 runs, all four blocks; the analysis code frozen at commit `0b55e1d`), and **not one of its scientific conclusions has been written or reviewed by a human**. The sections below describe what has actually happened, separated by actor. Everything marked **[HUMAN AUTHORS MUST COMPLETE]** has *not* been done by a human yet and must not be presented as if it had. Nothing in this file is filled in from conjecture.

## Summary

- **The software in this repository was built with an autonomous agentic coding workflow** using **Claude Code** (Anthropic). The human authors set the intent and the constitution (`PROMPT.md`, `PLAN.md`, and for the architecture study a written brief), choose the research question, and are responsible for the final scientific interpretation; the AI agent writes code, runs training and analysis, and drafts documentation under that direction.
- **No scientific result is AI-fabricated.** Every number reported in `RESULTS.md` and shown in the web explorer is produced by a real, seeded training run that lives in `training/runs/` and is regenerable from source. The agent is bound by a hard anti-fabrication rule (`PROMPT.md` §6), and the load-bearing analyses carry self-checks and automated tests.
- **The AI has also corrected its own earlier claims.** The audit documents in `docs/` (written by the AI, adversarially reviewed by further AI agents) classify several existing repository claims as over-interpreted or methodologically problematic, and one pre-registered prediction was found to be mathematically wrong and corrected *before* any new run was analysed (`docs/LABBOOK.md` entry 15). These corrections are part of the record, not hidden.

---

## 1. Who decided what (architecture study, 2026-09-02/03)

| Decision | Actor | Where recorded |
|---|---|---|
| Research question ("Do both use a Fourier-based phase-addition mechanism, and if so do they represent and compute it differently?") | **Human** (given to the AI in the master prompt) | `GROKVERSE_MASTER_PROMPT_EN.md` §0 |
| Whether to commit/push, how much compute to spend, the stopping rule (fixed 25 000-step budget), the number of paired seeds (10), and whether to proceed on AI-proposed pre-registration values | **Human** (answered explicitly on 2026-09-02) | `docs/dev/PREREG_BRIEF.md` rows marked `[HUMAN 2026-09-02]`, `docs/LABBOOK.md` entry 2 |
| Every other pre-registration value: hypotheses and nulls, gate pass rules, structured-neuron definition and thresholds, key-frequency rule and its threshold, decision tree, statistics | **AI-proposed**, explicitly labelled `[AI-PROPOSED — pending human approval]`, collected in one table for approval | `docs/PREREGISTRATION.md` §11, `docs/dev/PREREG_BRIEF.md` |
| Research options and study design proposed | **AI** | `docs/RESEARCH_SPEC.md` (AI-drafted from a human brief), `docs/dev/PREREG_BRIEF.md` |

**[HUMAN AUTHORS MUST COMPLETE]** — the `[AI-PROPOSED]` values above have not yet been approved, changed or rejected by a human. Until `docs/HUMAN_DECISIONS.md` records that decision, the pre-registration is an AI proposal that a human has not signed.

## 2. Code

| Area | AI-generated | Human |
|---|---|---|
| Original pipeline (config, seeding, data, transformer, MLP, training loop, Fourier/PCA/attention/progress-measure analysis, export, explorer) | Yes, under the constitution | Reviewed; **[HUMAN AUTHORS MUST COMPLETE: which parts were read, changed or extended by a human]** |
| Architecture-study pipeline (dense evaluation, transition intervals, checkpoint grid and roles, run manifest, matrix launcher, two-hot control model) | Yes | **[HUMAN AUTHORS MUST COMPLETE]** |
| Analysis modules of the study (`metrics`, `wave_fitting`, `statistics`, `function_agreement`, `logit_formula_fit`, `key_frequencies`, mechanism, ablation, aggregation modules) | Yes, each written by one AI agent and then adversarially reviewed by a second AI agent | **[HUMAN AUTHORS MUST COMPLETE]** |
| Tests (`test_core.py`, `tests/test_*.py`) | Yes | **[HUMAN AUTHORS MUST COMPLETE]** |

Code review in this project means **AI reviewing AI**: a second agent with a different instruction set re-derives the formulas, checks edge cases and adds tests. That is a real check and it caught real defects, but it is **not** an independent human review and must not be described as one.

## 3. Mathematics

| Item | Status |
|---|---|
| Derivation that a Fourier representation solves `(a+b) mod p`, the angle-addition identities, the readout sum | AI-written from Nanda et al. 2023; verified numerically by the AI (`training/tests/test_derivations.py`) |
| Derivation that the ReLU supplies the multiplication, with the cross-term amplitude `8/(3π²)` and phase `φ_a + φ_b` | AI-derived and AI-verified numerically |
| Correction that the `(a−b)` cross term is exactly as large, so only the population readout selects addition | AI-derived and AI-verified numerically, 2026-09-03 |
| Transformer forward decomposition, neuron→logit map `W_out @ W_U`, mean-attention effective curves, additivity R² | AI-derived and AI-verified numerically |
| Square-wave odd-harmonic and aliasing facts | Textbook Fourier series; AI-verified numerically at `p = 113`. **Not** attributed to any consulted paper |
| **Independent human check of these derivations** | **[HUMAN AUTHORS MUST COMPLETE]** — no human has yet re-derived or signed off on any of the above |

## 4. Literature

Seven primary-source notes under `docs/sources/` were produced by AI agents that fetched the papers (and, where reachable, the authors' code) and quoted them verbatim with section numbers; four of the notes were then re-verified by a second AI agent that re-fetched the sources and corrected quotes, numbers and labels. Statements that could not be confirmed in the fetched text carry `[NOT FOUND IN SOURCE]`; statements taken from model memory carry `[FROM MEMORY — UNVERIFIED]`.

**[HUMAN AUTHORS MUST COMPLETE]** — no human has yet read the primary sources and confirmed the notes.

## 5. Experiments

| Item | Actor |
|---|---|
| Run matrix designed (primary 10 paired seeds, Grokfast × train-fraction confound cells, parameter-matched control, two-hot control) | AI, from the master prompt's requirements; scope approved by a human |
| Runs launched, monitored and logged | **AI** (a local background process on the human's machine). The human authorised the compute budget but did not start individual runs |
| Run metadata (config, seeds, split hash, versions, platform, weight norms, checkpoints) | Recorded automatically by the pipeline |
| Analyses executed | **AI**, over all 51 runs: 280 per-checkpoint calls on the primary block, 234 on the confound block, plus the parameter-matched and two-hot blocks |
| Evidence gate, decision tree, statistics, H3/H4 reports, bounded alternative analysis | **AI**, applying thresholds fixed in `docs/PREREGISTRATION.md` before the runs. No threshold was changed after a result was seen; two AI decisions that materially affect a criterion are escalated to the human authors as `docs/HUMAN_DECISIONS.md` D5 and D6 |
| **Externally executed blocks**: the 100,000-step convergence block S1 (20 runs) and the two-hot control at 10 seeds | **External** — announced by the study's external reviewer in the mentoring chat on 2026-09-05 and run 2026-09-05/07 on the reviewer's initiative and the reviewer's own AWS EC2 account (`c7i.16xlarge`, Linux x86_64, torch 2.12.1+cpu), on the students' frozen configuration and code at `42dd79a`; the students did not design, launch or pay for these runs. Handed back unmodified on 2026-09-08. The AI verified the bundle's integrity (30 manifests, 829 checkpoint hashes, 23 split hashes) and re-derived the gate, statistics, H3 and H4 reports with the frozen code (byte-identical). Reported separately in `RESULTS.md` §16; interpretation by the human authors (D9) |
| The reviewer's follow-up analyses S2, S2b, S3 on the S1 checkpoints, and S2 on the two-hot runs | **External** — the reviewer's own scripts, which are not in this repository; their outputs are stored under `training/results/external/` and quoted in `RESULTS.md` §16.6–16.7 as *reported*, not re-derived |
| `CHECKPOINT_GRID` extended above 25,000 on the reviewer's branch `arch-study-convergence` (commit `42dd79a`) | **External**, a declared deviation from a pre-registered constant, fixed before S1 ran, additive (entries ≤ 25,000 unchanged), **not merged** into this branch (`docs/PREREGISTRATION.md` §12) |
| **The `txf_mul_*` multiplication runs (formerly the reserved Eigenleistung)** | **Changed on 2026-09-20.** E6 had reserved these three runs for the author's own evaluation, and until that day the AI had neither run the tool on them nor read their metrics. On 2026-09-20 the author explicitly lifted the reservation. The **measurement** was then performed by the **AI** — `experiment_add_vs_mult.py --analyze-run` over the three multiplication runs and their three addition counterparts, written up as `docs/ADD_VS_MUL_EIGENLEISTUNG.md` §0–§2 — and must be counted as AI work, not as own work. The **interpretation** (§3, §4 and the signature there, and the corresponding paragraph in the submission draft) was deliberately **not** written by the AI and remains **[HUMAN AUTHORS MUST COMPLETE]**: presenting an AI-written interpretation as the author's own work would make this disclosure false. |
| **Human-run experiment (Eigenleistung)** | **[HUMAN AUTHORS MUST COMPLETE]** — after the change above, no experiment in this project is documented as designed, executed *and* evaluated by a human author. If the submission claims an own-work experiment, it must name one that actually is. |

## 6. Writing

| Document | Status |
|---|---|
| `docs/` audit, plan, derivation, pre-registration and source documents | **AI-drafted raw material**, each headed "AI-drafted — not yet human-reviewed" |
| `RESULTS.md`, `README.md` | **AI-drafted, rewritten 2026-09-04** against the completed study, under master prompt §21. Both now report the gate outcome (`neither_passes`) as the study's central result, list four headline claims withdrawn from the 2026-09-03 text, and carry an explicit "what this repository does not claim" section. No human has reviewed either. The numbers are measurements; the *reading* of them is AI-drafted and remains provisional until §6's last two rows are filled in |
| `docs/LIMITATIONS.md` | §A AI-drafted 2026-09-03 (design limits, written before results); **§B AI-drafted 2026-09-04 from measurement**, including the convergence test the design promised — which found the MLP's headline metric unsettled at the step budget in 8 of 10 seeds |
| `docs/CLAIM_EVIDENCE_TABLE.md` | **AI-drafted**, 2026-09-04: eight claims in the master prompt's six-part structure, each traced to a file under `results/`, plus an explicit table of claims that may **not** be made from this evidence. Not human-reviewed |
| Preparation material `Claude/PREP/` (study guide, jury Q&A, own-work plan) | **AI-written 2026-06-21** for the author's preparation; marked there as not part of the submission. Found on 2026-09-20 to be tracked in this repository; removed from the tracked tree the same day on the author's decision (it remains in the git history). |
| Final scientific interpretation and conclusions | **[HUMAN AUTHORS MUST COMPLETE]** — `docs/HUMAN_INTERPRETATION_TEMPLATE.md` exists for exactly this and is empty of human text. AI-drafted interpretation must remain labelled as AI-drafted until the human authors have independently revised it in their own words |
| Draft of the written submission (`bwki/PROJEKTDOKUMENTATION_ENTWURF.md`, outside this folder) | **AI-drafted 2026-09-20** by a subagent from this repository's records, every number cited to its file; headed with a status block saying so; the own-work section (3.3), the interpretation (6.5) and the add-vs-mul result are left as `[HENRIK]` slots and must be written by the author. The draft is not the submitted document; the author's revision is. |
| Submission write-up and video pitch | **[HUMAN AUTHORS MUST COMPLETE]** |

## 7. Models and tools used

- **Claude Code** (Anthropic) — autonomous coding agent driving the build. Models used in the architecture-study session: Claude Fable 5.1 and Claude Opus 5, with parallel subagents for implementation, adversarial review and source verification.
- All software dependencies and their licenses are tracked in `THIRD_PARTY.md`.

## 8. EU AI Act note

The artifact produced here is a research/educational visualization of a well-studied ML phenomenon (grokking). It is not a high-risk AI system under the EU AI Act and makes no automated decisions about people. AI assistance in *building* the project is disclosed here for transparency. Any model trained in this project is a tiny network on a synthetic modular-arithmetic task, used solely for interpretability research.

## 9. External input — the mentor (since 2026-08-14)

Since 2026-08-14 the authors have been mentored, in a LinkedIn chat, by Dr. Jana Stucke, an external reviewer (a PhD-level
solutions architect contacted through the authors' mathematics teacher; named in `docs/RESEARCH_SPEC.md` and in
the reviewer's own `docs/external/CONVERGENCE_RUNBOOK_S1.md`, anonymised elsewhere in this repository). The
following inputs came from outside and are disclosed here for the *Eigenständigkeit* criterion. The list is
AI-drafted from the archived chat and the reviewer's bundle; *confirming the list, deciding whether the reviewer
is named consistently, and checking the competition rules on externally executed compute (runbook §0.4) are the
human authors' tasks under `docs/HUMAN_DECISIONS.md` D9 — this section is not one of the 11 placeholders above.*

| date | input | where it landed |
|---|---|---|
| 2026-08-22 | question: where is the originality stated, and the motivation in the authors' own words? | README / write-up — open |
| 2026-08-29 | why two architectures at all; pointer to arXiv:2603.23784; questions about the MLP mechanism; question whether the broad masking was intended | trigger of the architecture study (`GROKVERSE_MASTER_PROMPT_EN.md`); `docs/MASK_PROTOCOL_AUDIT.md`, `RESULTS.md` §5.2 |
| 2026-08-31 | the research question in its final wording ("Do a transformer and an MLP learn the same modular-addition rule in different ways?") and the advice to write a spec first, in English, reviewed jointly | `docs/RESEARCH_SPEC.md`, master prompt |
| 2026-09-02 | e-mail with suggested edits to the spec | not archived in the repository |
| 2026-09-05 | objection that G4 removes 86–100 % of the network; proposal of a graded ablation (top 1/5/10/25/50 % against equal-size random groups) | `docs/PREREGISTRATION.md` §14, `RESULTS.md` §6.1, D7 |
| 2026-09-05/07 | S1 convergence block (100,000 steps, 20 runs), two-hot control ×10, analyses S2/S2b/S3 — external compute and the reviewer's own scripts | `RESULTS.md` §16, `docs/external/`, `training/results/external/`, D9 |
| 2026-09-07 | a proposed framing of problem / solution / "why care" for the submission, including the sentence "both learn the same mathematical principle" | not adopted as written: the pre-registered gate is `neither_passes` (`RESULTS.md` §3); on 2026-09-08 the reviewer agreed with the authors' sharper wording |

None of the above changed a threshold, definition or pass rule of the frozen study; the decisions listed in §1
remain as stated.

### 9b. A second input — the co-author's audit document (received 2026-09-20)

Ali Kandora, **co-author of this project** (`docs/RESEARCH_SPEC.md` lists him under Authors; the role was confirmed by the author on 2026-09-20), sent an AI-generated "four-phase
audit" of GROKVERSE (`bwki/grokverse_scientific_audit.md`) and an add-vs-mul experiment script, both produced
outside this repository. The AI checked them on 2026-09-20: the audit assessed the **archived pre-2026-09-02
snapshot** (its numbers come from `archive/…/PROGRESS.md`), so its critique items K1, K2, K3, K5 and K8 describe
gaps the current study had already closed; its "proof" of the Fourier ring contains one incorrect step (weight
decay on a fixed orthonormal basis is presented as sparsity-inducing "like LASSO" — it is not, and the delta
target derived in its own step 1 has an exactly flat spectrum, top-8 share 8/56) and one overclaimed step (a
finite cross-entropy optimum that does not exist); its trigonometric factorisation is the Nanda et al. 2023
argument that `docs/MLP_MECHANISM_DERIVATION.md` already reproduces with attribution; and its discrete-log
ordering idea is Power et al. 2022 §3.2 (`docs/sources/power2022_and_grokfast2024.md` item 8). **Taken over:**
the script, after the AI corrected its Fourier basis over ℤ₁₁₂ (114 rows for a 112-dimensional space: a
degenerate sine row normalised to noise and a duplicated Nyquist row; now 112 rows, orthonormal to 7.6e-15,
Parseval to 6e-16, checked by `--self-test`) and added an `--analyze-run` mode that reads a run's
`embeddings.npy` without retraining (`bwki/experiment_add_vs_mult.py`; the co-author's original kept unmodified
as `experiment_add_vs_mult (1).py`). **Not taken over:** the audit's "proof" and its recommendation table. The AI
did **not** run the tool on, or read the metrics of, the `txf_mul_*` runs (E6); `docs/ADD_VS_MUL_EIGENLEISTUNG.md`
is an empty template for the author's own evaluation.

## Changelog

- **2026-06-20** — Initial disclosure created at Phase 0 (project scaffolding). Build executed with Claude Code under the GROKVERSE constitution.
- **2026-06-21** — Autonomous Phase 0–1 build: pinned CPU-only environment (torch 2.12.1+cpu, numpy 2.4.6, matplotlib 3.11.0); `grokverse` training/analysis package implemented; determinism test passed; grokking reproduced and empirically verified, using Grokfast (`PROMPT.md` §7) to keep the transition reproducible on CPU, with the un-accelerated memorization plateau also observed; Next.js + React Three Fiber explorer building and serving real exported run data.
- **2026-06-21** — Phases 2–5 plus the Phase 4 original experiment, AI-implemented under human direction: mechanistic analysis, reproducibility harness, 6-run grid, export pipeline, cross-architecture comparison across 3 seeds.
- **2026-06-21** — Un-accelerated canonical run harvested (memorize step 145 → generalize step 8367).
- **2026-06-21** — Restricted and excluded loss progress measures added.
- **2026-08-18** — Autonomous audit and extension session: 30 confirmed findings fixed or logged; progress measures extended to the canonical run; un-accelerated cross-architecture rerun.
- **2026-08-18** — Multiplication runs completed; their evaluation deliberately reserved for the human author as Eigenleistung. The AI has not analysed them.
- **2026-09-01** — `docs/RESEARCH_SPEC.md` drafted by Claude Opus 5 from a human-written brief, then re-verified against the code and all 16 run artifacts. Not human-reworded.
- **2026-09-03** — **Architecture study started** (this session, Claude Code with Claude Fable 5.1 and Opus 5). Baseline tagged and archived; the pre-study audit committed; a run-format-v2 pipeline (dense evaluation, transition intervals, checkpoint grid, manifests, matrix launcher, two-hot control) implemented and tested; seven primary-source notes fetched and quoted, four of them adversarially re-verified; three audit documents written (evidence audit, capacity/confound report, legacy-metric audit); the pre-registration brief fixed before the runs; the run matrix launched (20 primary runs, 18 complete at the time of writing); foundation analysis modules implemented with tests; both mechanism derivations written and numerically verified, in the course of which **one pre-registered prediction was found to be wrong and corrected before any run was analysed** (`docs/LABBOOK.md` entry 15). A session usage limit interrupted the AI mid-session; the local training runs continued unaffected and no output was overwritten (`docs/LABBOOK.md` entries 9–13). Everything in this entry is AI work; the human contributions listed in §1 are the scope and compute decisions of 2026-09-02.
- **2026-09-04** — **The study's measurements were completed by the AI, unattended, in an overnight self-paced loop.** Written and tested this session: the causal-ablation module (nine MLP and eight transformer ablation ids, each with a size-matched random control), `h3_validity` together with the interface specification it had never been given, `structure_over_time`, and the aggregation layer (`aggregate`, `decision_tree`, `figures_study`, `bounded_alternative`, `statistics_report`, `h3_report`, `h4_report`, `controls_report`). The analysis code was **frozen at commit `0b55e1d`** and recorded in `docs/PREREGISTRATION.md` §12; three post-freeze fixes were genuine bugs, each logged in `docs/LABBOOK.md` with its reason and followed by a re-run, and none of them touched a threshold, definition, pass rule, control or statistic. Test suite at the end of the session: 24 files. The measurements produced include **two negative results that the AI reported rather than avoided** — the pre-registered evidence gate returns `neither_passes` for both architectures, and H3, the study's own proposed primary contribution, is **refuted by its own pre-registered criterion**. All of this is AI work. **No human has reviewed any of these numbers, and the scientific interpretation of them remains entirely outstanding** (`docs/HUMAN_INTERPRETATION_TEMPLATE.md` is still empty of human text).
- **2026-09-04** — Architecture study analyses completed and the documentation rewritten against them. AI-executed: all 51 runs, every analysis module, the aggregation and reporting layer, `RESULTS.md`, `README.md`, `docs/LIMITATIONS.md` §B, `docs/CLAIM_EVIDENCE_TABLE.md`, `PROGRESS.md` and the numbered labbook. The pre-registered evidence gate returned `neither_passes` and the study's own primary hypothesis (H3) was refuted by its own criterion; both are reported as the results rather than reframed. Five post-freeze defects were found and fixed, three of them silent — the last returned `0.0` rather than `null` and had flattened two H4 trajectories for the whole study (labbook 101). **No scientific conclusion in any of these documents has been written or reviewed by a human**, and two choices that turned out to decide a pre-registered criterion (`D5`, `D6`) were escalated to the human authors rather than resolved by the AI.
- **2026-09-08** — **External data received.** The reviewer's bundle (S1 at 100,000 steps, two-hot ×10, the executed source tree at `42dd79a`, four documents) was unpacked into the gitignored `training/external/`, its small outputs and documents copied unmodified into `training/results/external/` and `docs/external/`, its integrity verified (30 manifests, 829 checkpoint hashes, 23 split hashes) and the gate/statistics/H3/H4 reports re-derived with the frozen code (byte-identical). Reported as `RESULTS.md` §16 without changing any existing number; `docs/LIMITATIONS.md` B3 gains its measured endpoint; escalated as **D9**. This entry is AI work; the runs themselves are external (§5, §9). The reviewer's S2/S2b/S3 scripts are missing from the bundle and were requested.
- **2026-09-20** — **Submission-day session** (Claude Code with Claude Fable 5.1; three Sonnet subagents for review, verification and the script fix). All items are AI work; none changed a number, threshold, verdict, figure or anything under `web/` (E3). (a) The colleague's audit and script received and assessed — see §9b. (b) `docs/NOVELTY_AND_RELATED_WORK.md`: Grokfast author list corrected to Lee, Kang, Kim & Lee (checked against arXiv:2405.20233). (c) `THIRD_PARTY.md`: Grokfast attribution added; it had been missing (`docs/sources/power2022_and_grokfast2024.md` item 9). (d) `README.md`: pnpm 10 requirement noted; `web/node_modules` was found incomplete (`styled-jsx` missing) and reinstalled from the lockfile with pnpm 10.34.5, after which `pnpm build` is clean (Next.js 14.2.15); `python tests/run_all.py` 24/24 green. (e) `docs/ADD_VS_MUL_EIGENLEISTUNG.md` created as a headings-only template. (f) Found and reported to the author, not changed: `Claude/PREP/` is tracked in git although marked "not for submission"; `docs/sources/power2022_and_grokfast2024.md` item 8 quotes the multiplication runs' generalization steps, a one-sentence departure from E6; `docs/RESEARCH_SPEC.md` names the external reviewer while the rest of the repository does not (D9); `docs/HUMAN_DECISIONS.md` is still `NOT YET APPROVED`, so any approval recorded today is post hoc and must be dated as such. (g) `docs/CLAIM_EVIDENCE_TABLE.md` C3 and `docs/LIMITATIONS.md` B1 still carried the pre-D8 control values 0.873 / 0.916; aligned to the 0.878 / 0.910 that `RESULTS.md` §6 and labbook 1175 record. No other number touched. (h) On the author's verbal instruction (2026-09-20) the AI recorded two names: the colleague Ali Kandora (§9b) and the external reviewer Dr. Jana Stucke (§9, README, RESULTS §6/§16, HUMAN_DECISIONS D9). (i) In the submission draft outside this folder, the AI produced a documented-facts list for the own-work section and a raw draft of the interpretation, both labelled as AI text that the author must confirm or rewrite; neither is the submitted text. (k) On the author's decision, `Claude/PREP/` was removed from the tracked tree (`git rm --cached`, now ignored). (l) `docs/sources/power2022_and_grokfast2024.md` item 8: the parenthetical quoting the multiplication runs' generalization steps was removed as an E6 departure; the algebraic note stays. (m) `CLAUDE.md`: the E6 rule added so that a fresh agent session sees it. (n) The author approved the AI proposals in `docs/HUMAN_DECISIONS.md` verbally on 2026-09-20; the AI did not write that approval into the file itself — it prepared `bwki/apply_human_decisions.py`, which the author runs and commits under his own name, so the approval carries his hand. (o) On the author's approval, one scope notice was added to `web/components/Explorer.tsx` ("reproduction-phase runs; the study is in RESULTS.md; this view not yet updated"); no run data, panel, caption or tour text was changed, the `docs/dev/EXPLORER_UPDATE_PLAN.md` approval box stays unchecked, and `pnpm build` is clean. (p) **E6 lifted by the author** ("ignore the e6 rule", 2026-09-20): the AI ran `experiment_add_vs_mult.py --analyze-run` on the three `txf_mul_*` runs and their three `txf_add_*` counterparts and filled `docs/ADD_VS_MUL_EIGENLEISTUNG.md` §0–§2 and the measured paragraph of the submission draft with the output. Measured, not interpreted: the interpretation sections there stay open, and §5 above now records the multiplication analysis as AI work rather than as the author's own. No threshold, verdict or number elsewhere in the repository changed. (q) The branch was pushed to `origin/GROKVERSE-MP` on the author's instruction (six commits, `82abdfe..e5bb51b`).
