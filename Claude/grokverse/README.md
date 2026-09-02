# GROKVERSE

> Watch the exact moment a neural network *gets it*.

GROKVERSE is a scientifically rigorous reproduction of **grokking** — the phenomenon where a tiny network keeps memorizing for thousands of steps and then, long after its training loss has flatlined, *suddenly generalizes*. We reproduce it, dissect **how** it happens with mechanistic interpretability, run **one original experiment**, and ship an **interactive 3D web explorer** that lets anyone scrub through training and watch the structure emerge in real time.

**BWKI 2026 Hauptpreis entry.** Two non-negotiables: the science is real and honestly reported, and the artifact is genuinely impressive to use.

---

## Status

Active build. See [`PROGRESS.md`](PROGRESS.md) for the dated audit trail and [`RESULTS.md`](RESULTS.md) for findings. The governing documents are [`PROMPT.md`](PROMPT.md) (how we work) and [`PLAN.md`](PLAN.md) (what we build, phases 0→7).

## Repository layout

```
grokverse/
├─ PROMPT.md  PLAN.md            # operating manual + plan (binding)
├─ README.md  RESULTS.md  PROGRESS.md
├─ AI_DISCLOSURE.md  THIRD_PARTY.md  IDEAS_BACKLOG.md
├─ training/                     # Python + PyTorch: the science
│  ├─ grokverse/                 # config, seed, data, models/, train, analysis/, export
│  ├─ runs/                      # per-run outputs (gitignored; regenerable)
│  └─ figures/                   # publication-quality plots
└─ web/                          # Next.js + React Three Fiber: the explorer
```

## What we found

A 1-layer ReLU transformer on modular addition (mod p=113), at the **canonical un-accelerated config** (`frac=0.3`, `wd=1.0`), **memorizes the training set by step 145** (train acc crosses 0.99; exactly 1.000 by step 156) while test accuracy sits at chance through an **~8,000-step plateau**, then **suddenly generalizes**, crossing 0.95 at **step 8367** — a **grok gap of 8,222 steps** (final test acc 0.981). Its embedding concentrates **76%** of its Fourier power into the top-8 frequencies (vs 32% non-grokked) — the periodic "trig-identity" circuit, with the read-out `=` position attending ~50/50 to both operands. Seed-robust **Grokfast** runs (arXiv:2405.20233; [`PROMPT.md`](PROMPT.md) §7) reproduce the same transition faster for in-session iteration, and the **original experiment** — run in both the accelerated and the honest un-accelerated setting, 3 seeds each — finds the transformer converges to a **markedly sparser Fourier circuit** than a 2-layer MLP in both settings (0.73 vs 0.44 top-8 power un-accelerated), and generalizes earlier in every seed, though the speed gap shrinks from ~2.9× (accelerated) to ~1.3× (un-accelerated) — reported as such. Full numbers + figures in [`RESULTS.md`](RESULTS.md).

## Reproduce the science

```bash
# from training/  (Python 3.12; CPU-only — no GPU needed)
python -m venv .venv
# activate:  .venv\Scripts\activate  (Windows)  |  source .venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m grokverse.train --determinism-test          # same seed => identical first-batch tensor
python -m grokverse.train --config grokfast --train-frac 0.5 --steps 8000 --early-stop-acc 0.98
python -m grokverse.export --all                       # -> web/public/data (the explorer's source of truth)
python -m grokverse.analysis.figures runs/txf_add_p113_wd1.0_frac0.5_gf2.0_seed0
python -m grokverse.analysis.compare                   # cross-architecture finding (add --frac 0.3 --unaccelerated for the faithful setting)
python -m grokverse.analysis.progress_measures --config grokfast --train-frac 0.5 --steps 8000 --seed 0 --early-stop-acc 0.98   # restricted/excluded loss
python test_core.py                                    # 23 correctness checks
```

## Run the explorer

```bash
# from web/
pnpm install
pnpm dev   # open http://localhost:3000
```

The explorer renders the **real exported run data**: scrub training to watch the 113 token embeddings reorganize from a blob into a periodic **ring** in 3D, see the loss/accuracy **phase transition** and the sparse **Fourier spectrum**, switch between the transformer and MLP runs, take the **Guided Tour**, or open the **Live Lab** to train a tiny MLP on `(a+b) mod 23` in your browser and induce grokking yourself by raising the weight decay. End-to-end browser check (with the dev server running): `node verify.mjs`.

## How it's built

This repository is built with **Claude Code** in an autonomous agentic workflow under a strict anti-fabrication constitution. See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) for the full AI-vs-human breakdown and EU-AI-Act note, and [`THIRD_PARTY.md`](THIRD_PARTY.md) for dependency licenses.
