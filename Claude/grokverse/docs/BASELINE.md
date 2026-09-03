# Baseline state before the architecture study

Recorded 2026-09-02 before any new experiment or analysis change (master prompt §3 rule 6).

## Git

| item | value |
|---|---|
| baseline commit | `d9434c1` ("feat: grokverse-Projekt direkt ins Repo aufnehmen", 2026-08-19) |
| tag | `baseline/pre-arch-study` → `d9434c1` |
| working branch | `arch-study` (branched from `main` at `d9434c1`) |
| state of the tree at session start | `grokverse/` and `PREP/` had been moved on disk into `Claude/` without a commit; recorded as renames in the first commit on `arch-study` |
| uncommitted work found in the tree | `docs/RESEARCH_SPEC.md`, `analysis/mask_protocols.py`, `analysis/mlp_mechanism.py`, harmonic-family functions in `analysis/fourier.py`, `threads`/`n_params`/`config_diff` additions, extended `test_core.py` — all AI-drafted in a previous session, none yet used for any reported number |

## Archive

`archive/pre_arch_study_2026-09-02/` (gitignored; 103 MB) holds a byte copy of `training/runs/` (all 16
run directories: `run.json`, `embeddings.npy`, `model_final.pt`, two `progress_measures.json`),
`training/figures/` (21 PNGs), `RESULTS.md`, `README.md`, `PROGRESS.md`, `AI_DISCLOSURE.md`.
`MANIFEST.sha256` lists the SHA256 of all 75 files. No existing figure or run is deleted or overwritten by
the study; new runs use a distinct `run_id` suffix (`docs/dev/RUN_FORMAT_V2.md`).

## Existing runs (from each `run.json`, unchanged)

| run_id | steps budget | threads | commit | mem step (train ≥ .99) | gen step (test ≥ .95) | gap | final test acc | last logged step |
|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0 | 40000 | 6 | 1a35a9c | 156 | 9646 | 9490 | 0.9942 | 9646 |
| mlp_add_p113_wd1.0_frac0.3_seed1 | 40000 | 6 | 1a35a9c | 156 | 10357 | 10201 | 0.9751 | 10357 |
| mlp_add_p113_wd1.0_frac0.3_seed2 | 40000 | 6 | 1a35a9c | 167 | 9646 | 9479 | 0.9830 | 9646 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0 | 5000 | – | a22c5d5 | 204 | 2121 | 1917 | 0.9848 | 2246 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1 | 5000 | – | a22c5d5 | 204 | 2121 | 1917 | 0.9865 | 2246 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2 | 5000 | – | a22c5d5 | 204 | 2246 | 2042 | 0.9945 | 2378 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0 | 4000 | – | ad5f594 | 150 | never | – | 0.1064 | 4000 |
| txf_add_p113_wd1.0_frac0.3_seed0 | 40000 | – | 4ccac23 | 145 | 8367 | 8222 | 0.9808 | 8367 |
| txf_add_p113_wd1.0_frac0.3_seed1 | 40000 | – | ead5166 | 145 | 6295 | 6150 | 0.9653 | 6295 |
| txf_add_p113_wd1.0_frac0.3_seed2 | 40000 | – | ead5166 | 145 | 8367 | 8222 | 0.9937 | 8367 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0 | 8000 | – | ad5f594 | 179 | 675 | 496 | 0.9834 | 717 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1 | 8000 | – | ad5f594 | 202 | 859 | 657 | 0.9904 | 912 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2 | 5000 | – | a22c5d5 | 192 | 716 | 524 | 0.9850 | 953 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed0 | 8000 | 6 | 26024f0 | 190 | 912 | 722 | 0.9850 | 969 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed1 | 8000 | 6 | 26024f0 | 202 | 912 | 710 | 0.9887 | 969 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed2 | 8000 | 6 | 26024f0 | 202 | 1029 | 827 | 0.9826 | 1093 |

Facts visible in this table that constrain every later claim:

* All six un-accelerated `frac0.3` runs **stop at their own generalization crossing** (last logged step =
  generalization step): every legacy "final" structure number is an at-transition number.
* Step indices sit on the logarithmic log grid (`n_logged_steps=150`): two transformer seeds report the
  identical crossing 8367 and two MLP seeds the identical 9646, so the reported crossings carry a
  grid-spacing uncertainty of several hundred steps at that point of the schedule.
* The thread count (6 vs unrecorded) and the git commit differ between runs of the "same" setting.
* The `mul` runs are reserved for the human author's own evaluation (PROGRESS.md 2026-08-18) and are not
  touched by this study.

## Test suite at baseline

`python test_core.py` (venv Python 3.12.10, torch 2.12.1+cpu, numpy 2.4.6): all checks pass up to
**one failure**, `odd-minus-even separates square from sinusoid`. Measured values on the synthetic
embeddings of that check: square wave `odd_minus_even = 0.0918`, sinusoid `0.0`; the check demands a
margin of 0.10. Cause (diagnosed, not yet fixed): with the canonical fundamental set
{18, 15, 11, 1, 13, 22, 56, 36} in Z_113, six harmonics alias onto *other fundamentals*
(2·18→36, 7·18→13, 2·11→22, 7·11→36, 7·13→22, 2·56→1) and are excluded from both shares, and a
discrete square wave sampled at odd `p` is not exactly antisymmetric, so it carries even-harmonic energy
(`even_share = 0.083`). This is treated as a defect of the WIP harmonic-shape statistic, to be resolved
in the metric-validity work (H3), not by loosening the threshold.

**RESOLVED 2026-09-03.** `fourier.harmonic_shape` is superseded by
`metrics.harmonic_shares`, which is defined per curve on that curve's **own** dominant
frequency and is therefore collision-free at prime `p` (asserted). Measured on the same
synthetic embeddings: the set-based statistic reports an even-harmonic share of **0.083**
for a clean square wave, where the truth is **0.0005** — inflated about 165x by the six
in-set collisions — and its separation is 0.092, below the 0.10 the old check demanded.
The per-curve statistic reports even 0.0005, separation **0.139**, and 0 collisions in
every case. The stale check was replaced by checks on the diagnosed behaviour of the old
statistic plus a check that the replacement separates; the numbers above are pinned in
`test_core.py`. Fixing it also revealed a **second, masked** defect: the `u_a` reference
check computed its hand reference in the stored float32 while `effective_curves` works in
float64, leaving a 1.9e-9 gap against a 1e-9 tolerance. The module was right; the check
was not. `test_core.py` now passes in full (125 checks).

## Environment

Windows 11, 12 logical CPUs, no GPU. Pinned single-thread step cost (40-step mean after 5 warm-up
steps, `frac=0.3`): transformer 254 ms, MLP 59 ms; at 6 threads 96 ms / 22 ms. Disk free 168 GB.
`scipy`, `pytest`, `pandas`, `sklearn` are **not** installed; all new analysis code is numpy/torch only.
