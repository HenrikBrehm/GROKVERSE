# PLAN.md — GROKVERSE

**Mission (one line):** Reproduce grokking in tiny neural nets, explain *how* the network suddenly generalizes, run one original experiment, and ship an interactive 3D web explorer that lets anyone watch it happen — as a BWKI 2026 Hauptpreis entry.

**Read `PROMPT.md` first.** That file governs *how* to work (honesty, determinism, verification discipline, hard rules). This file is the *what* and the *order*. Execute phases 0 → 7 in sequence; do not skip.

---

## North Star — what "done" looks like

A visitor opens the web app and:
1. picks a training run, hits play, and **watches the loss phase-transition** (memorization → sudden generalization) on a scrubber;
2. simultaneously sees the **token embeddings reorganize in 3D** from a blob into the periodic/Fourier structure;
3. opens a **mechanistic view** showing the dominant Fourier frequencies / the circuit;
4. opens the **live lab** and trains a tiny MLP in-browser, changing weight decay / train-fraction and watching grokking appear, vanish, or delay;
5. reads a short, honest writeup of one **original finding**.

Behind it: a clean, seeded, reproducible Python pipeline; figures and numbers that all trace to real runs; documentation embedding the literature with a self-critical evaluation; AI-assistance disclosure; and a drafted video pitch.

---

## Scientific spine (precise)

**Reproduction target (Phase 1-2):** Nanda et al. 2023 configuration — a **1-layer transformer** trained on **modular addition mod p (p = 113)**: input `a`, `b`, predict `(a + b) mod p`. Full-batch AdamW, high weight decay (~1.0), train fraction ~0.3, train for enough steps to pass the generalization transition. Reference d_model ≈ 128, ~4 heads (record exact values used). Log loss/accuracy (train + test) and weight checkpoints on a **logarithmic step schedule**.

**Analysis (Phase 3):**
- **Fourier decomposition of the embedding matrix** — identify the sparse set of dominant frequencies; show their magnitude vs. the rest; confirm robustness across ≥3 seeds.
- **PCA of embeddings** — for the 3D projection and to show the ring/periodic geometry.
- **Progress measures** — implement *restricted loss* and *excluded loss* (Nanda et al.); show they track the three phases (memorization, circuit formation, cleanup).
- **Attention / neuron inspection** — at least one view that connects the structure to the computation.

**Original experiment (Phase 4) — pick ONE as primary, the other is the documented fallback:**
- **(A) Cross-architecture circuit formation (primary).** Train a 1-layer transformer, a 2-layer MLP, and (stretch) a KAN on modular addition mod p. For each: measure the grokking delay, whether a periodic/Fourier structure emerges in the learned representation, and a progress measure. **Question:** do different architectures converge to the *same* trig-identity-style algorithm, a different periodic solution, or fail to grok? Report mean ± spread over ≥3 seeds, with comparative visualizations. *(A clean variant: add vs. multiply mod p on the same architecture — does multiplication grok the same way addition does, and what structure does it find?)*
- **(B) Omnigrok weight-decay / init-scale scaling law (fallback — guaranteed signal).** Sweep weight decay (and/or init scale) and measure how the generalization step scales. Reproduce the qualitative Omnigrok result that grokking can be induced/eliminated/delayed by controlling weight norm. Use this if (A) is inconclusive after honest effort; it always produces a presentable, real result.

**Honesty clause:** whichever experiment runs, its conclusion must survive ≥3 seeds, and fragility is itself reported (see `PROMPT.md` §4).

---

## Repository architecture

```
grokverse/
├─ PROMPT.md  PLAN.md  README.md  RESULTS.md  PROGRESS.md
├─ AI_DISCLOSURE.md  THIRD_PARTY.md  IDEAS_BACKLOG.md
├─ training/                  # Python: the science
│  ├─ pyproject.toml / requirements.txt
│  ├─ grokverse/
│  │  ├─ config.py            # dataclass configs; seeds; p; arch; optim; logging schedule
│  │  ├─ seed.py              # deterministic seeding (torch/numpy/python)
│  │  ├─ data.py              # modular arithmetic dataset + train/test split
│  │  ├─ models/              # transformer.py, mlp.py, kan.py (stretch)
│  │  ├─ train.py             # training loop + logarithmic checkpoint logging
│  │  ├─ analysis/            # fourier.py, pca.py, progress_measures.py, attention.py
│  │  ├─ experiment.py        # runs the grid / cross-arch / scaling sweeps
│  │  └─ export.py            # writes the web data contract (see below)
│  ├─ runs/                   # per-run outputs (gitignored; regenerable)
│  └─ figures/                # publication-quality plots for RESULTS.md
├─ web/                       # Next.js + React Three Fiber: the explorer
│  ├─ app/ or src/
│  │  ├─ components/          # LossView, EmbeddingView3D, FourierView, ControlPanel, GuidedTour
│  │  ├─ lib/                 # data loading, run schema types, projection helpers
│  │  └─ livelab/             # in-browser tiny-MLP training (tfjs / onnxruntime-web)
│  └─ public/data/            # exported, size-bounded run artifacts
└─ pitch/                     # script.md, shot_list.md, storyboard notes
```

**Tech stack (locked — do not relitigate):** Python + PyTorch + NumPy for training/analysis; Next.js + React Three Fiber + TypeScript for the explorer; a lightweight state layer (Zustand or React context); TensorFlow.js (or onnxruntime-web) for the live lab. Package manager: pnpm. Keep deps minimal.

---

## Data contract (Python `export.py` → `web/public/data/`)

One artifact per run. **Metadata JSON** + a **compact numeric blob**.

`run_<id>.meta.json`:
```jsonc
{
  "id": "txf_p113_wd1.0_seed0",
  "arch": "transformer",          // transformer | mlp | kan
  "task": "add",                  // add | mul
  "p": 113,
  "hyperparams": { "d_model": 128, "weight_decay": 1.0, "train_frac": 0.3, "lr": 1e-3, "...": "..." },
  "seed": 0,
  "lib_versions": { "torch": "x.y.z", "numpy": "x.y.z" },
  "git_commit": "…",
  "logged_steps": [0, 1, 2, 5, 10, 20, ...],   // logarithmic
  "curves": { "train_loss": [...], "test_loss": [...], "train_acc": [...], "test_acc": [...],
              "restricted_loss": [...], "excluded_loss": [...] },
  "dominant_frequencies": [k1, k2, ...],
  "transition": { "train_saturated_step": 500, "test_generalized_step": 14000 }
}
```

`run_<id>.embeddings.bin` (or `.json` if small): per logged step, the **3D-projected** token coordinates `[num_tokens × 3]` (project offline via PCA for determinism), plus optionally the raw embedding for one reference step. **Bound total size** (downsample logged steps to ~100-150; ~114 tokens × 3 floats × 150 steps is tiny). If you keep raw embeddings, gzip and check the total `public/data/` size stays modest.

The frontend must treat this contract as the single source of truth and never invent values not present in it.

---

## Phases

### Phase 0 — Foundations (½ day)
**Goal:** runnable skeleton, both stacks bootstrapped, discipline files seeded.
- Init git; create `README.md`, `PROGRESS.md`, `RESULTS.md`, `AI_DISCLOSURE.md`, `THIRD_PARTY.md`, `IDEAS_BACKLOG.md` (stubs).
- Bootstrap `training/` (PyTorch, pinned versions) and `web/` (Next.js + R3F, renders a placeholder 3D scene).
- Implement `seed.py` and a 5-line determinism test (same seed ⇒ identical first-batch tensor).
**Acceptance:** `python -m grokverse.train --smoke` runs one tiny step without error; `pnpm dev` serves a page with a rotating placeholder mesh; determinism test passes twice with identical output; `AI_DISCLOSURE.md` has its first entry.

### Phase 1 — Reproduce grokking (core risk — do this before anything pretty)
**Goal:** the 1-layer transformer on modular addition mod 113 demonstrably groks, seeded and logged.
- `data.py`: modular addition dataset + deterministic train/test split at `train_frac`.
- `models/transformer.py`: 1-layer transformer per the Nanda recipe (record exact dims).
- `train.py`: full-batch AdamW, weight decay, logarithmic checkpoint + curve logging to `runs/`.
- Run to past the generalization transition; save curves and checkpoints.
**Acceptance (empirical, from `PROMPT.md` §4):** logged curves show train acc >0.99 early while test acc stays near chance, then test acc rises to >0.95 distinctly later. Report the two step indices in `PROGRESS.md`. Re-run with a second seed and confirm the phenomenon repeats (timing may differ).

### Phase 2 — Reproducibility harness + run grid
**Goal:** any run regenerable from one command; a small grid exists for the explorer.
- `config.py` dataclasses; `experiment.py` to launch a parameterized grid (a handful of weight-decay / train-frac / seed combos for transformer-add).
- Persist full config + lib versions + git commit in each run's metadata.
**Acceptance:** deleting `runs/` and re-running the grid reproduces equivalent curves (same seed ⇒ same curve). Grid covers enough variety for a meaningful ControlPanel later (≥4-6 runs).

### Phase 3 — Mechanistic analysis
**Goal:** explain the grokked solution with real measurements.
- `analysis/fourier.py`: dominant-frequency extraction from embeddings; magnitude spectrum.
- `analysis/pca.py`: 3D projection for the viz; confirm ring/periodic geometry.
- `analysis/progress_measures.py`: restricted & excluded loss; verify they track the three phases.
- `analysis/attention.py`: at least one view linking structure to computation.
- Save `figures/` (spectrum, PCA ring over training, progress measures vs. step).
**Acceptance:** dominant frequencies named and shown to dominate; robust across ≥3 seeds; progress measures move meaningfully across the transition; figures saved and referenced in `RESULTS.md`.

### Phase 4 — Original experiment (the Erkenntnisgewinn)
**Goal:** one honest, novel-to-this-repo finding.
- Implement primary experiment (A): add `models/mlp.py` (and stretch `kan.py`); run transformer vs. MLP (vs. KAN) on modular addition across ≥3 seeds each; measure grokking delay, emergent structure, progress measure; produce comparative figures.
- If (A) is inconclusive after honest effort, run fallback (B): weight-decay / init-scale scaling sweep (Omnigrok-style) and report the scaling relationship.
**Acceptance:** conclusion survives ≥3 seeds (mean ± spread reported); comparative figures saved; an honest paragraph in `RESULTS.md` stating the finding *and its limits*. Negative/fragile outcomes are documented as the result, not hidden.

### Phase 5 — Export pipeline
**Goal:** real data flowing to the frontend via the data contract.
- `export.py`: write `*.meta.json` + `*.embeddings.bin` per run into `web/public/data/`; bound total size; downsample logged steps.
- Add a `data/index.json` listing available runs for the ControlPanel.
**Acceptance:** every run in the grid + the experiment has a valid exported artifact; a tiny TS loader reads one and logs its curves; total `public/data/` size is modest.

### Phase 6 — The explorer (the differentiator)
**Goal:** a polished, fast, genuinely impressive web app on real data.
- `LossView`: train/test curves with a play/scrub control over logged steps; clearly marks the transition.
- `EmbeddingView3D` (R3F): tokens as points; animate their movement from blob → periodic structure as the user scrubs; smooth interpolation between logged steps; orbit controls; color by token / by frequency phase.
- `FourierView`: the dominant-frequency spectrum / circuit view, linked to the current step.
- `ControlPanel`: select among pre-computed runs (weight decay, train-frac, arch, task); switching updates all views.
- `livelab/`: train a **tiny MLP** on modular addition **in-browser** (tfjs/onnxruntime-web); live loss curve; sliders for weight decay / train-frac / steps so the user induces or removes grokking themselves. Keep it small enough to run smoothly on a laptop.
- `GuidedTour`: a scripted walkthrough (the pitch path) a first-time viewer can click through.
- Design quality bar: intentional, non-default visual design; legible typography; coherent palette; no template smell. The 3D reorganization must read clearly as "structure emerging," not as visual noise.
**Acceptance:** `pnpm dev` runs with no console errors; all views render from real exported data; scrubbing is smooth; switching runs updates everything consistently; the live lab actually trains and visibly groks/doesn't-grok as sliders change; guided tour runs start to finish; no GPU required.

### Phase 7 — BWKI deliverables & polish
**Goal:** competition-ready.
- `README.md`: what it is, how to reproduce the science (one command), how to run the app.
- `RESULTS.md`: findings, every figure, literature embedding (Power/Nanda/Omnigrok/Grokfast), and a **self-critical evaluation** (what's solid, what's fragile, what you'd do next) — this directly serves the "wissenschaftliches Arbeiten" and "Ausblick" criteria.
- `AI_DISCLOSURE.md`: finalized (built with Claude Code autonomous workflow; AI-vs-human breakdown; libraries/models + licenses; EU-AI-Act note).
- `THIRD_PARTY.md`: dependency + adapted-code attributions and licenses.
- `pitch/script.md` + `pitch/shot_list.md`: a 3-5 min pitch that lets a non-expert grasp the goal *and* lets an expert gauge the depth (mirror the DEversAI winning formula); specify which views/screens to show and the one-sentence "aha" hook ("watch the exact moment an AI 'gets it'").
- Final code cleanup: types where they matter, docstrings on science functions, remove dead code, ensure clean `pnpm build` and a clean training run from scratch.
**Acceptance:** a competent stranger can clone, reproduce the core result, and run the app from the README in <10 min; all docs complete; `pnpm build` succeeds; pitch material drafted; repo open-source ready.

---

## Risk register

| Risk | Trigger | Mitigation |
|---|---|---|
| Reproduction won't grok | Phase 1 no transition after honest tuning | Fall back exactly to Nanda's published hyperparameters; verify split/seed; only then consider Grokfast to manage runtime (reproduce un-accelerated first). |
| "Just a visualization of known results" | Reviewer/self-critique | Phase 4 original experiment is mandatory; report a real, seed-robust finding. |
| Original experiment inconclusive | Phase 4 cross-arch result is noise across seeds | Switch to fallback (B) Omnigrok scaling — guaranteed presentable signal — and document why. |
| 3D UI scope creep | Phase 6 dragging on | Freeze a minimal-but-stunning view set early; ship LossView + EmbeddingView3D + FourierView first, live lab next, extras to backlog. |
| Data artifacts bloat the repo | `public/data/` large | Downsample logged steps; project to 3D offline; gzip; gitignore raw with regeneration script. |
| Honesty drift | Any "result" without evidence | `PROMPT.md` §4 + §6 hard rules; every UI number traces to a run. |

---

## Out of scope (binding)

- Large language models, multi-layer/large transformers, anything needing more than a single modest GPU.
- Training transformers live in the browser (only the tiny MLP live lab runs client-side).
- Any feature not serving the North Star. New ideas → `IDEAS_BACKLOG.md`.
- Deployment/hosting infra beyond a working local `pnpm dev` / `pnpm build` (a static deploy is a nice-to-have, not required for submission).

---

## First action

Execute **Phase 0**, then **Phase 1** (reproduce grokking) before any visual work. Reproduce reality first; make it beautiful second.