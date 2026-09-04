# GROKVERSE

> Watch the exact moment a neural network *gets it* — and then find out how little that picture
> actually licenses you to say.

GROKVERSE is a scientifically rigorous reproduction of **grokking** — the phenomenon where a tiny
network keeps memorizing for thousands of steps and then, long after its training loss has flatlined,
*suddenly generalizes*. We reproduce it, we run a **pre-registered mechanistic study** comparing a
1-layer transformer against a 2-layer MLP on the same task and the same data splits, and we ship an
**interactive 3D web explorer** that lets anyone scrub through training and watch the structure emerge.

**BWKI 2026 Hauptpreis entry.** Two non-negotiables: the science is real and honestly reported, and the
artifact is genuinely impressive to use.

---

## What we found

**Grokking reproduces cleanly, un-accelerated, in both architectures** — 20 of 20 runs at the canonical
config (`p = 113`, `train_frac = 0.3`, `wd = 1.0`, 25,000 steps, no early stopping). The transformer
memorizes at a median step of 140 and generalizes at 7,588; the MLP at 160 and 9,250. Final test
accuracy: 0.99966 (transformer, median over 10 seeds) and 1.000000 (MLP, all 10 seeds).

**The pre-registered evidence gate then fails — in both architectures.** Three of its four criteria
hold on 10 of 10 seeds for both: periodic structure, the predicted phase-addition relation, and an
end-to-end Fourier formula that reaches argmax accuracy 1.000 on held-out cells. The fourth, the
*causal* one, fails 0 of 10 in both — because removing the "structured" neurons is not distinguishable
from removing an equal number of random ones, once you notice that the definition selects 88–98 % of
the network. The gate branch is **`neither_passes`**, and the study reports exactly that: **the Fourier
evidence tested does not, by the pre-registered standard, identify the learned mechanism in either
architecture.**

**The study's own primary hypothesis is refuted by its own criterion.** H3 predicted that the MLP's
lower structured-neuron fraction was an artifact of top-*k* concentration missing square-wave harmonics.
The artifact is real — synthetic populations differing *only* in waveform separate by 0.189 — but it
closes only **5.4 %** of the observed gap. The difference is a real difference in the measured
structure.

**What did survive.** The two architectures agree on **99.98 %** of all 12,769 inputs (identical on 4 of
10 seeds) while their logits correlate at only 0.083. Structure metrics rise **before** the
generalization jump in 10 of 10 seeds in both. Every structure difference survives matching the
parameter count to 0.02 %. And the key *frequencies* — unlike the neuron sets — are causally
load-bearing in both: removing them costs ~0.99 accuracy where removing an equal number of random
frequencies costs ~0.000.

**What changed our mind mid-study.** A 3-seed two-hot control showed the two-hot MLP is *more*
square-wave-like than the shared-embedding MLP (+0.184), so the waveform difference is at least partly
about the input parametrization, not the architecture.

Full numbers, the six-part claim structure, the negative results and the withdrawn claims are in
[`RESULTS.md`](RESULTS.md). **The final scientific interpretation has not yet been written by the human
authors** — see [`docs/HUMAN_DECISIONS.md`](docs/HUMAN_DECISIONS.md).

### What this repository does not claim

Stated explicitly, because the earlier version of `RESULTS.md` did claim some of them:

- not "the two architectures use the same circuit", nor "the same Fourier principle in different
  representations" — both were conditional on a gate that did not pass;
- not "both learn the same function" — 99.98 % agreement, but not identical in 6 of 10 seeds;
- not "transformers grok faster" — one hyperparameter point, and seed 4 reverses the direction;
- not "the transformer learns a sparser Fourier circuit" as a discovery — the direction is a
  replication of Manir & Rupa 2026, and there is no causal warrant for "circuit".

---

## Status

The architecture study on branch `GROKVERSE-MP` (local name until 2026-09-04: `arch-study`) is complete: **51 runs, all completed, none failed**; 24
test files and 1,913 automated checks; every threshold frozen before the runs at commit `0b55e1d`.

Open, and reserved for the human authors: the final interpretation, the decisions in
[`docs/HUMAN_DECISIONS.md`](docs/HUMAN_DECISIONS.md) (including **D5** and **D6**, two choices that turned
out to be load-bearing), the `[HUMAN AUTHORS MUST COMPLETE]` placeholders in
[`AI_DISCLOSURE.md`](AI_DISCLOSURE.md), the multiplication runs, and the explorer update.

See [`PROGRESS.md`](PROGRESS.md) for the dated audit trail and [`docs/LABBOOK.md`](docs/LABBOOK.md) for
the numbered lab record, including every defect found and what it would have changed.

## Repository layout

```
grokverse/
├─ PROMPT.md  PLAN.md            # operating manual + plan (binding)
├─ README.md  RESULTS.md  PROGRESS.md
├─ AI_DISCLOSURE.md  THIRD_PARTY.md  IDEAS_BACKLOG.md
├─ docs/                         # pre-registration, plans, audits, labbook, claim–evidence table
├─ training/                     # Python + PyTorch: the science
│  ├─ grokverse/                 # config, seed, data, models/, train, analysis/, export
│  ├─ tests/                     # 24 test files, 1,913 checks
│  ├─ runs/                      # per-run outputs (gitignored; regenerable)
│  └─ results/                   # aggregates, reports, figures (committed)
├─ web/                          # Next.js + React Three Fiber: the explorer
└─ archive/                      # logs, pre-study figures, session scaffolding, raw dumps — nothing load-bearing (see archive/README.md)
```

## Reproduce the science

```bash
# from training/  (Python 3.12; CPU-only — no GPU needed)
python -m venv .venv
# activate:  .venv\Scripts\activate  (Windows)  |  source .venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python tests/run_all.py                      # 24 test files, 1,913 checks
python -m grokverse.train --determinism-test # same seed => identical first-batch tensor
```

A single canonical run, end to end (~25 min on one CPU thread):

```bash
python -m grokverse.train --arch transformer --p 113 --train-frac 0.3 --weight-decay 1.0 --steps 25000 --seed 0
```

The full study — 51 runs and the whole analysis chain — is `pwsh ./run_matrix_chain.ps1` followed by
the commands listed in [`RESULTS.md`](RESULTS.md) §15. Every figure is regenerated from the stored
aggregate tables alone, so no figure can drift from the numbers.

## Run the explorer

```bash
# from web/
pnpm install
pnpm dev   # open http://localhost:3000
```

Scrub training to watch the 113 token embeddings reorganize from a blob into a periodic **ring** in 3D,
see the loss/accuracy **phase transition** and the Fourier spectrum, switch between runs, take the
**Guided Tour**, or open the **Live Lab** to train a tiny MLP on `(a+b) mod 23` in your browser and
induce grokking yourself by raising the weight decay.

> **The explorer currently shows the pre-study runs.** It is deliberately frozen until the human
> authors review it, because several of the claims in its captions are among those withdrawn above. The
> required changes are specified in
> [`docs/dev/EXPLORER_UPDATE_PLAN.md`](docs/dev/EXPLORER_UPDATE_PLAN.md).

## How it's built

This repository is built with **Claude Code** in an autonomous agentic workflow under a strict
anti-fabrication constitution. See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) for the AI-vs-human breakdown
and the EU-AI-Act note, and [`THIRD_PARTY.md`](THIRD_PARTY.md) for dependency licenses. The scientific
methods and their sources are mapped one-to-one in [`docs/METHODS.md`](docs/METHODS.md); the delineation
against related work is in
[`docs/NOVELTY_AND_RELATED_WORK.md`](docs/NOVELTY_AND_RELATED_WORK.md).
