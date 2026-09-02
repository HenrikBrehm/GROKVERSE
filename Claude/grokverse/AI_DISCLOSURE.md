# AI_DISCLOSURE.md — GROKVERSE

This project discloses, per BWKI rules and in the spirit of the EU AI Act, all AI/tooling assistance used in its creation. It is maintained from day one and updated as the build proceeds — not reconstructed at the end. BWKI requires both a *clearly identifiable, documented portion of own work* (Eigenleistung) **and** full transparency about AI use; this file serves the transparency half and points to the own-work half.

> **Status of this file, 2026-09-03.** The architecture study (`docs/PREREGISTRATION.md`) is in progress. The sections below describe what has actually happened, separated by actor. Everything marked **[HUMAN AUTHORS MUST COMPLETE]** has *not* been done by a human yet and must not be presented as if it had. Nothing in this file is filled in from conjecture.

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
| Analyses executed | AI |
| **Human-run experiment (Eigenleistung)** | **[HUMAN AUTHORS MUST COMPLETE]** — the `txf_mul_*` multiplication runs remain reserved for the human author's own evaluation (`PROGRESS.md`, 2026-08-18); the AI has not analysed them and must not |

## 6. Writing

| Document | Status |
|---|---|
| `docs/` audit, plan, derivation, pre-registration and source documents | **AI-drafted raw material**, each headed "AI-drafted — not yet human-reviewed" |
| `RESULTS.md`, `README.md` (current text) | AI-drafted; several claims in them are classified as over-interpreted by `docs/CURRENT_EVIDENCE_AUDIT.md` and are scheduled for rewriting once the study's analyses are complete |
| Final scientific interpretation and conclusions | **[HUMAN AUTHORS MUST COMPLETE]** — `docs/HUMAN_INTERPRETATION_TEMPLATE.md` exists for exactly this and is empty of human text. AI-drafted interpretation must remain labelled as AI-drafted until the human authors have independently revised it in their own words |
| Submission write-up and video pitch | **[HUMAN AUTHORS MUST COMPLETE]** |

## 7. Models and tools used

- **Claude Code** (Anthropic) — autonomous coding agent driving the build. Models used in the architecture-study session: Claude Fable 5.1 and Claude Opus 5, with parallel subagents for implementation, adversarial review and source verification.
- All software dependencies and their licenses are tracked in `THIRD_PARTY.md`.

## 8. EU AI Act note

The artifact produced here is a research/educational visualization of a well-studied ML phenomenon (grokking). It is not a high-risk AI system under the EU AI Act and makes no automated decisions about people. AI assistance in *building* the project is disclosed here for transparency. Any model trained in this project is a tiny network on a synthetic modular-arithmetic task, used solely for interpretability research.

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
