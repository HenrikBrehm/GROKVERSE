# THIRD_PARTY.md — dependencies, datasets, and adapted code

All third-party software, data, and any adapted reference code are listed here with their licenses, per `PROMPT.md` §6 (Respect licenses). Exact pinned versions are recorded in `training/requirements.txt` and `web/package.json`; this file records the license and role of each.

## Datasets

- **None external.** The only dataset is a *synthetically generated* modular-arithmetic table (`(a + b) mod p`), produced deterministically in `training/grokverse/data.py`. No third-party data is used.

## Python dependencies (training/)

| Package | Version | Role | License |
|---|---|---|---|
| PyTorch (`torch`) | 2.12.1+cpu | model definition, training, autograd | BSD-3-Clause |
| NumPy | 2.4.6 | numerics, Fourier/PCA analysis | BSD-3-Clause |
| matplotlib | 3.11.0 | publication figures | Matplotlib License (BSD-style, PSF-based) |

(Exact versions pinned in `training/requirements.txt`.)

## Web dependencies (web/) — finalized in Phase 6

| Package | Version | Role | License |
|---|---|---|---|
| Next.js (`next`) | 14.2.15 | app framework | MIT |
| React (`react`) | 18.3.1 | UI | MIT |
| React DOM (`react-dom`) | 18.3.1 | UI rendering | MIT |
| three.js (`three`) | ^0.169.0 | 3D rendering | MIT |
| @react-three/fiber | ^8.17.10 | React renderer for three.js | MIT |
| @react-three/drei | ^9.114.0 | R3F helpers | MIT |
| @tensorflow/tfjs | ^4.22.0 | in-browser LiveLab MLP training | Apache-2.0 |
| Playwright (dev) | ^1.49.0 | headless e2e verification (`verify.mjs`) | Apache-2.0 |

State management uses React hooks/context — no extra dependency.

## Adapted reference code

- **Method/recipe** for the grokking reproduction follows **Nanda et al. 2023** (arXiv:2301.05217) and **Power et al. 2022** (arXiv:2201.02177). Any code adapted from their public repositories will be attributed here with the specific file and commit; as of Phase 0, the implementation is written from the papers' descriptions rather than copied.

- **Grokfast** — **Lee, Kang, Kim & Lee 2024**, *Grokfast: Accelerated Grokking by Amplifying Slow Gradients* (arXiv:2405.20233). `training/grokverse/train.py::apply_grokfast` implements the paper's EMA gradient filter (Algorithm 2, lines 7–8; the official `gradfilter_ema` initialisation μ₁ = g₁) with the README defaults `alpha = 0.98`, `lambda = 2.0`. Written from the paper's description; no code copied from the official repository. Used only for the accelerated corroboration runs, never for the un-accelerated reference results (`RESULTS.md`).

_Last updated: 2026-09-20 (Grokfast attribution added; see `docs/sources/power2022_and_grokfast2024.md` item 9)._
