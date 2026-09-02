# AI_DISCLOSURE.md — GROKVERSE

This project discloses, per BWKI rules and in the spirit of the EU AI Act, all AI/tooling assistance used in its creation. It is maintained from day one and updated as the build proceeds — not reconstructed at the end. BWKI requires both a *clearly identifiable, documented portion of own work* (Eigenleistung) **and** full transparency about AI use; this file serves the transparency half and points to the own-work half.

## Summary

- **The software in this repository was built with an autonomous agentic coding workflow** using **Claude Code** (Anthropic). The human author (Henrik) sets the intent and constitution (`PROMPT.md`/`PLAN.md`), selects the experiments, reviews outputs, and verifies every scientific claim against the logged curves; the AI agent writes code, runs training/analysis, and drafts documentation under that direction.
- **No scientific result is AI-fabricated.** Every number reported in `RESULTS.md` and shown in the web explorer is produced by a real, seeded training run that lives in `training/runs/` and is regenerable from source. The agent is bound by a hard anti-fabrication rule (`PROMPT.md` §6), and key analyses carry self-checks (e.g. the Fourier inverse-transform reconstructs the logits to ~1e-12; an injected known frequency is recovered by the analysis).

## What was AI-generated vs. authored by the human

This split is honest and deliberately specific. The right-hand column is what makes the Eigenleistung *clearly identifiable* (BWKI rule); it is being expanded as the human-authored contributions are completed and is the part the author can defend in detail at the finals.

| Area | AI-generated (Claude Code) | Authored / done / verified by Henrik |
|---|---|---|
| Project constitution (`PROMPT.md`, `PLAN.md`) | Drafted to spec | Authored the intent and success criteria; reviewed; made binding |
| Repo scaffolding & tooling | Yes | Approved |
| Training/analysis/web code | Yes | Reviewed; *[einzutragen: welche Teile selbst gelesen/verändert/erweitert]* |
| Hyperparameters | Taken from cited literature (Nanda et al.) | Approved; understands why each matters |
| Mechanistic analysis (Fourier, PCA, attention, restricted/excluded loss) | Implemented | Verified outputs against figures/curves; *[einzutragen: eigene Herleitung der Trig-Identität, Anhang]* |
| Original experiment (cross-architecture) | Proposed & implemented | Selected & approved |
| **Own extension experiment** | *(assist on request)* | *[einzutragen: z. B. add-vs-mul — Frage, Durchführung und Auswertung durch Henrik]* |
| Scientific interpretation of results | Drafted | **Verified against the actual logged curves**; *[einzutragen: Re-Formulierung in eigenen Worten]* |
| Submission write-up & video pitch | Draft material only | *[einzutragen: eigenständig verfasst / selbst gesprochen]* |

> Items marked *[einzutragen…]* are placeholders for the author's own contributions, to be filled in concretely (with pointers to the relevant file/section/appendix) as they are completed. They are intentionally not claimed before they are done.

## Models / tools used

- **Claude Code** (Anthropic) — autonomous coding agent driving the build.
- All software dependencies and their licenses are tracked in `THIRD_PARTY.md`.

## EU AI Act note

The artifact produced here is a research/educational visualization of a well-studied ML phenomenon (grokking). It is not a high-risk AI system under the EU AI Act and makes no automated decisions about people. AI assistance in *building* the project is disclosed here for transparency. Any model trained in this project is a tiny network on a synthetic modular-arithmetic task, used solely for interpretability research.

## Changelog

- **2026-06-20** — Initial disclosure created at Phase 0 (project scaffolding). Build executed with Claude Code under the GROKVERSE constitution.
- **2026-06-21** — Autonomous Phase 0–1 build completed with Claude Code: pinned CPU-only environment (torch 2.12.1+cpu, numpy 2.4.6, matplotlib 3.11.0); `grokverse` training/analysis package implemented; determinism test PASS (bit-identical first-batch logit sum across two seeded builds); grokking reproduced and empirically verified, using Grokfast (PROMPT.md §7) to keep the transition reproducible on CPU, with the un-accelerated memorization plateau also observed; Next.js + React Three Fiber explorer building and serving real exported run data.
- **2026-06-21** — Phases 2–5 + Phase 4 original experiment, AI-implemented under human direction: mechanistic analysis (embedding Fourier sparsification after grokking; PCA periodic ring; read-out attention ≈50/50 to both operands); reproducibility harness + 6-run grid; export pipeline. Original cross-architecture finding (transformer groks ~2.9× faster and sparser than a 2-layer MLP, 3 seeds each). Every reported number traces to a seeded run; `test_core.py` 11/11 correctness checks pass.
- **2026-06-21** — Un-accelerated canonical run harvested: faithful no-Grokfast reproduction groks (memorize step 145 → generalize step 8367; 8,222-step gap; final test acc 0.981; 76% top-8 embedding Fourier power).
- **2026-08-18** — Autonomous audit + extension session with Claude Code: adversarially-verified 5-dimension self-audit (30 confirmed findings fixed or logged — stale claims corrected incl. the environment-dependent determinism scalar and the 0.99-crossing labeling; config-collision guard; guided-tour rebuilt data-driven after it was found narrating text contradicted by the on-screen data; LiveLab "grokked" badge now requires a measured delay). Restricted/excluded-loss progress measures extended to the **canonical un-accelerated run** (re-train recovers the recorded transition exactly, 145/8367; restricted 0.0020 vs excluded 10.42) and surfaced as a new explorer panel that appears only for runs where the measure was actually computed. Un-accelerated cross-architecture rerun (3 seeds/arch) launched. Every number verified against seeded runs; `test_core.py` grown to 23 checks.
- **2026-06-21** — **Restricted & excluded loss progress measures (Nanda 2023) added** (`analysis/progress_measures.py`), AI-implemented under the anti-fabrication constitution and empirically verified: 2D Fourier transform of the logits over the (a,b) grid with reconstruction self-check ~1e-12; on the grokked transformer the restricted loss (key frequencies only) collapses below the full loss while the excluded loss (key frequencies removed) rises far above chance — the held-out-independent mechanistic signature of circuit formation. Closes the previously-documented "progress measures not implemented" caveat. Figure + per-run JSON saved.
