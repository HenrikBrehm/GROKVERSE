# PROMPT.md — GROKVERSE Autonomous Build (ultracode / `/goal`)

> Operating manual for the autonomous coding session. Read this **fully** before touching `PLAN.md`. `PLAN.md` is *what* to build; this file is *how to behave* while building it.

---

## 0. Kickoff

Run this once the repo is initialized and both `PROMPT.md` and `PLAN.md` are present at the root:

```
/goal Execute PLAN.md end to end. Work phase by phase in order (Phase 0 → Phase 7).
Do not skip phases. After each phase, verify every acceptance criterion empirically,
commit, and write a short progress note to PROGRESS.md before starting the next phase.
Obey every rule in PROMPT.md. If a hard blocker appears, stop and write the blocker
to PROGRESS.md with the exact error and 2-3 proposed options instead of guessing.
```

---

## 1. Mission

Build **GROKVERSE**: a scientifically rigorous reproduction of *grokking* in tiny neural networks, a mechanistic-interpretability analysis of *how* the network suddenly generalizes, **one original experiment** that produces genuine new insight, and a **polished interactive 3D web explorer** that lets anyone watch a network grok in real time.

This is a **BWKI 2026 Hauptpreis** entry. Two non-negotiable success conditions:
1. **The science is real and honestly reported** (no fabricated or cherry-picked results; negative results are valid results).
2. **The artifact is genuinely impressive to look at and use** (the visualization is the differentiator, not an afterthought).

You are optimizing for a jury of ML researchers (Tübingen AI Center / MPI) who reward: *Eigenständigkeit, Originalität, Schwierigkeitsgrad, wissenschaftliches Arbeiten (Fehleranalyse + Einordnung in den Stand der Forschung + selbstkritische Bewertung), Erkenntnisgewinn, praktische Relevanz, und Lesbarkeit/Struktur des Codes.* Every decision is in service of those.

---

## 2. Operating principles (in priority order)

1. **Scientific honesty above all.** Never claim grokking happened without the loss/accuracy curve that proves it. Never report a Fourier "ring" without the actual measured frequencies. If an experiment fails or is inconclusive, say so plainly in `RESULTS.md` and `PROGRESS.md`. Fabricating, hard-coding, or visually faking a result is a **catastrophic failure** of this project — the entire BWKI thesis is "honest interpretability."
2. **Determinism and reproducibility.** Every training run is seeded (PCG32 / fixed torch + numpy + python seeds). Identical config + seed ⇒ identical curves, bit-for-bit where the hardware allows. Every figure and every checkpoint set must be regenerable from a single command. Log the seed, the full config, library versions, and git commit into each run's metadata.
3. **Incremental, verifiable progress.** Small steps that each end in a *verified* state. Prefer a working narrow slice over a broad broken one. Reproduce the known result before attempting anything original.
4. **Readable, structured code.** This is an explicit BWKI criterion. Clear module boundaries, typed interfaces, docstrings on the science-bearing functions, a `README.md` that a competent stranger can run in <10 minutes. No dead code, no commented-out experiments left in main.
5. **Ground every scientific claim in the literature.** Reproduction targets and method choices must cite the canonical sources (see §7). Do not invent recipes when a published one exists.

---

## 3. The autonomous loop

For each task in the current phase:

1. **Plan** — restate the task and its acceptance criteria in one or two lines.
2. **Implement** — write the smallest correct thing that satisfies it.
3. **Verify empirically** — *run it*. For ML: run the training/analysis and read the actual numbers/curves. For frontend: run the dev server and confirm the view renders and behaves. Do not mark a task done from code inspection alone.
4. **Record** — save the artifact (checkpoint, figure, metric) to the right directory; note the measured outcome.
5. **Commit** — one logical commit per task with a clear message (`feat:`, `exp:`, `viz:`, `docs:`, `fix:`).
6. **Advance** — only move on when the acceptance criterion is objectively met.

At the **end of each phase**: run the phase's full acceptance checklist, append a dated entry to `PROGRESS.md` (what was done, measured results, what's next, open questions), commit, then start the next phase.

---

## 4. ML verification discipline (read twice)

The most likely way this project fails is *believing* an ML result that isn't there. Guard against it:

- **Grokking is only "reproduced" when** test accuracy stays near chance for a long memorization phase and then rises sharply to >95% **well after** train accuracy saturated — and you have the logged curve to show it. State the exact step indices of (a) train saturation and (b) test generalization.
- **A Fourier/periodic structure is only "found" when** you can name the dominant frequencies, show their magnitudes vs. the rest, and the structure is robust to re-seeding (check ≥3 seeds).
- **A progress measure is only valid when** it moves monotonically/meaningfully across the phase transition on held-out data, not just on the training trajectory you tuned on.
- **The original experiment's conclusion must survive ≥3 seeds.** A single-seed difference between architectures is noise, not a finding. Report mean ± spread.
- **If a result is fragile or noisy, that is the finding** — document the fragility honestly. The Omnigrok-style weight-decay scaling fallback (see `PLAN.md` Phase 4) exists precisely because it produces a robust, guaranteed signal if the cross-architecture comparison turns out inconclusive.

---

## 5. Definition of Done

**Per task:** acceptance criterion in `PLAN.md` is empirically met, artifact saved, committed.

**Global (the project is done when):**
- The known grokking result is reproduced, analyzed, and the analysis is saved as figures + numbers.
- The original experiment is run across ≥3 seeds and its honest conclusion is written up.
- The web explorer runs, loads real exported data, scrubs through training smoothly, shows the 3D embedding reorganization + loss phase transition + at least one mechanistic view, and has a working interactive control surface (selectable pre-computed runs **plus** the in-browser tiny-MLP "live lab").
- `README.md` (run instructions), `RESULTS.md` (findings + figures + self-critique), and `AI_DISCLOSURE.md` (see §6) are complete.
- The video-pitch material in `pitch/` is drafted (script + shot list + which views to show).
- The repo is clean, typed where it matters, and open-source ready.

---

## 6. Hard rules (never violate)

- **Never fabricate, hard-code, mock, or visually fake any scientific result.** Every number in the UI and the writeup traces to a real run.
- **Never claim a result you did not measure.** No "the network learns trig identities" unless you show the evidence in *this* repo's runs.
- **Maintain `AI_DISCLOSURE.md` from day one.** BWKI requires disclosing all AI/tooling assistance and EU-AI-Act compliance. Log: this build was executed with Claude Code (autonomous agentic workflow); list which parts were AI-generated vs. human-directed; note all libraries/models used and their licenses. Update it as you go — do not reconstruct it at the end.
- **Respect licenses.** Check and record the license of every dependency, dataset, and any reference code you adapt. Do not copy code from papers'/blogs' repos without attribution in `THIRD_PARTY.md`.
- **No secrets, no keys, no large binaries in git.** Exported checkpoint blobs go in a sized-down `web/public/data/` (keep total artifact size sane; downsample logged steps) or are gitignored with a regeneration script.
- **Stay in scope.** `PLAN.md` §"Out of scope" is binding. New ideas go into `IDEAS_BACKLOG.md`, not into main.
- **When genuinely blocked** (a result won't reproduce after honest effort, a dependency is broken, a decision needs Henrik): stop, write the blocker + the exact error + 2-3 concrete options to `PROGRESS.md`, and do not paper over it.

---

## 7. Canonical references (ground the science here)

- **Power et al. 2022** — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (arXiv:2201.02177). The original phenomenon; generalization long after memorization.
- **Nanda et al. 2023** — "Progress Measures for Grokking via Mechanistic Interpretability" (arXiv:2301.05217, ICLR 2023). **Primary reproduction target.** 1-layer transformer on modular addition mod 113; the Fourier/trig-identity circuit; restricted & excluded loss; phases of memorization → circuit formation → cleanup. Use the authors' annotated recipe as the ground truth for the reproduction.
- **Liu et al. 2022 (Omnigrok)** — arXiv:2210.01117. Grokking controlled by weight norm / initialization scale; the robust weight-decay scaling fallback experiment is based on this.
- **Grokfast 2024** — arXiv:2405.20233. Amplify slow-gradient components to accelerate grokking; use to keep run times manageable if needed (but reproduce the *un*-accelerated phenomenon first, so the phase transition is faithful).

If a recipe detail is ambiguous, prefer the Nanda et al. configuration and record the exact hyperparameters you used in the run metadata.

---

## 8. Environment & compute notes

- Target free/cheap compute: a single GPU (Colab T4) or even CPU for the smallest configs. The Nanda recipe is minutes-to-~1h per run; keep the hyperparameter grid modest and logged-step schedule logarithmic.
- Pin versions (`requirements.txt` / lockfile). Record exact versions in run metadata for reproducibility.
- The frontend must run with a single `pnpm dev` (or `npm`) after data is exported; it must not require a GPU.

---

## 9. Progress reporting format (`PROGRESS.md`)

Append one block per task-batch / phase:

```
## [YYYY-MM-DD] Phase N — <name>
Done: <bullet list of what was built/run>
Measured results: <the actual numbers/curves, e.g. "train acc >0.99 by step ~500;
  test acc <0.05 until ~step 9k, then →0.98 by ~step 14k. Dominant Fourier freqs: k=...">
Verification: <which acceptance criteria passed, how confirmed>
Open questions / risks: <honest list>
Next: <the next task>
```

Be terse and factual. This file is the audit trail the human reads — and a draft source for `RESULTS.md` and the pitch.

---

## 10. Reminder

The visualization is the part the jury and audience will *remember*, and the honesty + rigor is the part that wins the Hauptpreis. Build both. Reproduce first, get something real, then make it beautiful.