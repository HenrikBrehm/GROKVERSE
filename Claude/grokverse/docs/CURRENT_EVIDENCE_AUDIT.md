# CURRENT_EVIDENCE_AUDIT — Phase-1 evidence audit of the GROKVERSE claims

> **AI-drafted audit (Claude), 2026-09-02 — not yet human-reviewed**
>
> This document only *classifies* the evidence behind the claims in `README.md` and `RESULTS.md` (master prompt `GROKVERSE_MASTER_PROMPT_EN.md` §5). It proposes no fixes. Every number below was recomputed on 2026-09-02 from the artifacts in `training/runs/` and carries its provenance (run_id, file, function). Statements taken from the primary literature are quoted verbatim from the page text returned by the fetch tool on 2026-09-02 and carry the section numbers as reported there; anything not found in that text is marked `[NOT FOUND IN SOURCE]`; anything from memory is marked `[FROM MEMORY - UNVERIFIED]`.

## 0. Scope, method, conventions

**Repository state audited.** `HenrikBrehm/GROKVERSE`, local copy `Claude/grokverse`, HEAD `25890dc` on branch `arch-study` (baseline tag `baseline/pre-arch-study` = `d9434c1`, `docs/BASELINE.md`). `git status` at audit time: `training/grokverse/config.py` and `training/grokverse/data.py` modified, `training/grokverse/models/mlp_twohot.py`, `docs/BASELINE.md`, `docs/dev/` untracked. The code read for this audit is the working tree.

**Files read.** `README.md`, `RESULTS.md`, `PLAN.md`, `PROGRESS.md`, `AI_DISCLOSURE.md`, `PROMPT.md`, `docs/RESEARCH_SPEC.md`, `docs/BASELINE.md`, `docs/data/capacity_report.json`, `training/grokverse/{config,data,train}.py`, `models/{mlp,transformer}.py`, `analysis/{fourier,progress_measures,mask_protocols,mlp_mechanism,compare,attention}.py`, `training/test_core.py`, all 16 legacy `training/runs/*/run.json`, the two `progress_measures.json` files, and `training/figures/` (22 PNGs).

**Environment used for the recomputation.** `training/.venv` Python 3.12.10, torch 2.12.1+cpu, numpy 2.4.6, `torch.set_num_threads(1)`, cwd `training/`. The recomputation script lives outside the repository (audit scratchpad `recompute.py`); Appendix C lists the exact library calls so every number is reproducible from the repository's own functions.

**Run inventory.** 16 legacy run directories with `run.json` + `embeddings.npy` + `model_final.pt` (byte copy in `archive/pre_arch_study_2026-09-02/`, `docs/BASELINE.md`). A 17th directory, `runs/txf_add_p113_wd1.0_frac0.3_seed0_smoketest/` (run-format-v2 artifacts: `run.json`, `manifest.json`, `checkpoints.json`, `ckpt_step000000.pt`; no `embeddings.npy`), appeared during the audit from a parallel process; it is **not** a legacy run and is excluded here. The three `txf_mul_*` runs are listed in Appendix A for completeness only; their scientific evaluation is reserved for the human author (`PROGRESS.md`, 2026-08-18, RESOLVED) and is not interpreted in this audit.

**Evidence levels** (master prompt §5): *descriptive* = a measured quantity reported without a comparison that isolates a cause; *correlational* = covariation between an internal quantity and a condition or outcome; *mechanistic* = a test of an internal relation predicted by a hypothesized algorithm; *causal* = an intervention on the trained model, with an appropriate control, that changes behaviour.
**Status:** *solid* / *preliminary* / *overinterpreted* / *methodologically problematic*, as defined in master prompt §5.

**One convention that affects several rows.** All transition steps in this repository are *first logged step at or above threshold* (`train.detect_transition`: train ≥ 0.99, test ≥ 0.95) on the logarithmic evaluation grid `utils.log_step_schedule(steps, 150)`. The audit therefore reports every crossing together with the previous evaluated step; the true crossing lies in the half-open interval (previous, reported].

---

## 1. Recomputed numbers

### 1.1 Per-run: transitions, final accuracy, top-8 Fourier concentration of `W_E` at the last logged step

Provenance: `run.json` → `config`, `git_commit`, `torch_num_threads`, `logged_steps`, `transition`; `embeddings.npy[-1]` → `analysis.fourier.dominant_frequencies(W_E, p=113, threshold=0.9, max_k=8)` → `dominant_fraction`, `dominant`, `n_freqs_for_threshold`, `cap_binding`. "prev" = previous logged step before the crossing.

| run_id | arch / setting | commit (run.json) | threads | mem step (prev) | gen step (prev) | gap | final test acc | last logged / budget | top-8 fraction | n_freqs for 90 % | cap binding | dominant (top-8) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| txf_add_p113_wd1.0_frac0.3_seed0 | txf, un-acc. frac 0.3 | 4ccac23 | not recorded | 145 (135) | 8367 (7792) | 8222 | 0.9808 | 8367 / 40000 | **0.7571** | 27 | True | 18, 15, 1, 11, 13, 22, 56, 36 |
| txf_add_p113_wd1.0_frac0.3_seed1 | txf, un-acc. frac 0.3 | ead5166 | not recorded | 145 (135) | 6295 (5863) | 6150 | 0.9653 | 6295 / 40000 | 0.6590 | 36 | True | 16, 30, 17, 9, 50, 32, 45, 26 |
| txf_add_p113_wd1.0_frac0.3_seed2 | txf, un-acc. frac 0.3 | ead5166 | not recorded | 145 (135) | 8367 (7792) | 8222 | 0.9937 | 8367 / 40000 | 0.7853 | 23 | True | 46, 53, 55, 52, 4, 3, 21, 43 |
| mlp_add_p113_wd1.0_frac0.3_seed0 | mlp, un-acc. frac 0.3 | 1a35a9c | 6 | 156 (145) | 9646 (8984) | 9490 | 0.9942 | 9646 / 40000 | **0.4308** | 39 | True | 21, 37, 3, 33, 19, 47, 1, 13 |
| mlp_add_p113_wd1.0_frac0.3_seed1 | mlp, un-acc. frac 0.3 | 1a35a9c | 6 | 156 (145) | 10357 (9646) | 10201 | 0.9751 | 10357 / 40000 | 0.4456 | 40 | True | 18, 6, 1, 26, 37, 29, 47, 16 |
| mlp_add_p113_wd1.0_frac0.3_seed2 | mlp, un-acc. frac 0.3 | 1a35a9c | 6 | 167 (156) | 9646 (8984) | 9479 | 0.9830 | 9646 / 40000 | 0.4311 | 38 | True | 24, 40, 53, 4, 8, 31, 3, 10 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0 | txf, Grokfast frac 0.5 | ad5f594 | not recorded | 179 (168) | 675 (635) | 496 | 0.9834 | 717 / 8000 | 0.5206 | 43 | True | 16, 39, 17, 35, 8, 44, 21, 53 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1 | txf, Grokfast frac 0.5 | ad5f594 | not recorded | 202 (190) | 859 (808) | 657 | 0.9904 | 912 / 8000 | 0.6050 | 40 | True | 27, 9, 16, 54, 33, 38, 45, 40 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2 | txf, Grokfast frac 0.5 | a22c5d5 | not recorded | 192 (182) | 716 (676) | 524 | 0.9850 | 953 / **5000** | 0.6432 | 38 | True | 4, 29, 15, 8, 55, 38, 47, 41 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0 | mlp, Grokfast frac 0.5 | a22c5d5 | not recorded | 204 (192) | 2121 (2003) | 1917 | 0.9848 | 2246 / 5000 | 0.3542 | 43 | True | 13, 38, 33, 4, 48, 11, 1, 21 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1 | mlp, Grokfast frac 0.5 | a22c5d5 | not recorded | 204 (192) | 2121 (2003) | 1917 | 0.9865 | 2246 / 5000 | 0.3486 | 43 | True | 20, 47, 1, 38, 31, 3, 26, 16 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2 | mlp, Grokfast frac 0.5 | a22c5d5 | not recorded | 204 (192) | 2246 (2121) | 2042 | 0.9945 | 2378 / 5000 | 0.3543 | 42 | True | 44, 24, 51, 32, 10, 19, 27, 13 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0 | txf, Grokfast frac 0.3 — never generalized ("non-grokked control") | ad5f594 | not recorded | 150 (142) | none | none | 0.1064 | 4000 / 4000 | **0.3197** | 46 | True | 21, 18, 11, 8, 15, 47, 17, 44 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed0 | txf, mul, Grokfast frac 0.5 (reserved, not interpreted) | 26024f0 | 6 | 190 (179) | 912 (859) | 722 | 0.9850 | 969 / 8000 | 0.1592 | 50 | True | 44, 48, 30, 29, 39, 31, 9, 24 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed1 | txf, mul, Grokfast frac 0.5 (reserved) | 26024f0 | 6 | 202 (190) | 912 (859) | 710 | 0.9887 | 969 / 8000 | 0.1581 | 50 | True | 16, 11, 24, 23, 49, 19, 7, 20 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed2 | txf, mul, Grokfast frac 0.5 (reserved) | 26024f0 | 6 | 202 (190) | 1029 (969) | 827 | 0.9826 | 1093 / 8000 | 0.1659 | 50 | True | 4, 6, 32, 13, 12, 40, 16, 23 |

Facts visible in this table:

* `cap_binding` is `True` for all 16 runs and `n_keep == 8` always. The number of frequencies needed to reach the 90 % threshold ranges from 23 (txf un-acc. seed 2) to 50 (mul runs); even the sparsest transformer needs 23 of the 56 available frequencies. "Top-8" is therefore a fixed cap, never a data-determined key-frequency count (`analysis/fourier.py` lines 77–86 already document this).
* All six un-accelerated `frac0.3` runs have `last logged step == generalization step` (`--early-stop-acc 0.95`, `PROGRESS.md` 2026-08-18): every "final" structure number for these runs is an at-crossing number.
* Two transformer seeds report the identical crossing 8367 and two MLP seeds the identical 9646 (grid snapping, §1.3).
* Embedding tensors: transformer `[T, 114, 128]` (includes the `=` token row; `dominant_frequencies` slices `[:113]`), MLP `[T, 113, 128]` (`models/mlp.py` allocates `W_E` as `[p, d]`).
* The Grokfast transformer runs do not share a step budget (seeds 0–1: 8000, seed 2: 5000; `run.json → config.steps`), so seed 2 sits on a different logarithmic evaluation grid than seeds 0–1 (§1.3). Thread counts are recorded only for the August runs (6) and absent for the June runs; commits differ within the "same" setting (txf un-acc.: `4ccac23` vs `ead5166`).
* The canonical run's dominant set `{18, 15, 1, 11, 13, 22, 56, 36}` (`embeddings.npy[-1]`) equals the set `[18, 15, 11, 1, 13, 56, 22, 36]` recorded in `progress_measures.json → key_frequencies` for the deterministic re-train (same set, different rank order).

### 1.2 Per-architecture aggregates (`analysis.compare.agg`: mean, sample std with ddof=1, n = 3 each)

| setting | arch | generalization step | grok gap | top-8 fraction | final test acc | per-seed values (gen / top-8) |
|---|---|---|---|---|---|---|
| Grokfast, frac 0.5 | transformer | 750.0 ± 96.6 | 559.0 ± 86.0 | 0.5896 ± 0.0627 | 0.9863 ± 0.0037 | 675 / 859 / 716 ; 0.5206 / 0.6050 / 0.6432 |
| Grokfast, frac 0.5 | MLP | 2162.7 ± 72.2 | 1958.7 ± 72.2 | 0.3524 ± 0.0033 | 0.9886 ± 0.0052 | 2121 / 2121 / 2246 ; 0.3542 / 0.3486 / 0.3543 |
| un-accelerated, frac 0.3 | transformer | 7676.3 ± 1196.3 | 7531.3 ± 1196.3 | 0.7338 ± 0.0663 | 0.9799 ± 0.0142 | 8367 / 6295 / 8367 ; 0.7571 / 0.6590 / 0.7853 |
| un-accelerated, frac 0.3 | MLP | 9883.0 ± 410.5 | 9723.3 ± 413.7 | 0.4358 ± 0.0085 | 0.9841 ± 0.0096 | 9646 / 10357 / 9646 ; 0.4308 / 0.4456 / 0.4311 |

* Ratio of mean generalization steps MLP/transformer: **2.884** (Grokfast) and **1.287** (un-accelerated). Paired by seed (same seed ⇒ identical train/test split; verified with `data.split_hash`, §1.7): (675, 2121), (859, 2121), (716, 2246) and (8367, 9646), (6295, 10357), (8367, 9646) — the MLP crossing is later in all 6 pairs.
* Every value in `RESULTS.md` §3's two tables is reproduced to the printed precision. The sentence "Both architectures grok to ≥98% test accuracy in both settings" (`RESULTS.md` §3, "Finding") is **not** reproduced: transformer un-acc. seed 1 = 0.9653, MLP un-acc. seed 1 = 0.9751 (`run.json → transition.final_test_acc`); `docs/RESEARCH_SPEC.md` §14 item 1 already records this.
* Non-grokked control: 0.3197 (`README.md` "32%").

### 1.3 Logarithmic evaluation grid at the crossings (`utils.log_step_schedule(steps, 150)`)

`log_step_schedule(40000, 150)` yields 127 distinct steps (not 150, because early integer rounding collapses duplicates).

| budget | crossing (first logged step ≥ threshold) | previous logged step | interval containing the true crossing | grid spacing at that point |
|---|---|---|---|---|
| 40000 | 145 (train ≥ 0.99, all 3 txf un-acc. seeds) | 135 | (135, 145] | 10 |
| 40000 | 156 / 167 (MLP un-acc. memorization) | 145 / 156 | (145, 156] / (156, 167] | 11 |
| 40000 | 6295 (txf seed 1) | 5863 | (5863, 6295] | 432 |
| 40000 | 8367 (txf seeds 0, 2) | 7792 | (7792, 8367] | 575 |
| 40000 | 9646 (MLP seeds 0, 2) | 8984 | (8984, 9646] | 662 |
| 40000 | 10357 (MLP seed 1) | 9646 | (9646, 10357] | 711 |
| 8000 | 675 / 859 (txf gf seeds 0, 1) | 635 / 808 | (635, 675] / (808, 859] | 40 / 51 |
| 5000 | 716 (txf gf seed 2) ; 2121, 2246 (MLP gf) | 676 ; 2003, 2121 | (676, 716] ; (2003, 2121], (2121, 2246] | 40 ; 118, 125 |

Canonical run curve values (`run.json → curves`, run `txf_add_p113_wd1.0_frac0.3_seed0`): train acc 0.9893 at step 135, **0.9992 at 145** (the ≥ 0.99 crossing), **1.0000 at 156**; test loss 4.746 at step 0, maximum **26.00 at step 1226**, 0.0597 at 8367. `README.md`'s "145 / exactly 1.000 by 156 / 8367 / 8,222 / 0.981 / 4.7 → 0.06 / ~26" are all reproduced.

### 1.4 Attention of the `=` position (canonical run and Grokfast seed 0)

Provenance: `models/transformer.py → OneLayerTransformer.attention_pattern(x)` on `data.make_dataset(cfg)["all_x"]` (all 12,769 inputs), slice `[:, :, POS_EQ=2, :]`; mean over inputs (this is what `analysis/attention.py` reports) **and** per-input standard deviation over inputs (ddof=1; computed for this audit — `attention.py` computes no dispersion statistic). Model: `model_final.pt`.

| run | head | mean to a / b / = | per-input std to a / b / = | to-a quantiles min / 5 % / 50 % / 95 % / max | share of inputs with to-a in (0.4, 0.6) |
|---|---|---|---|---|---|
| txf_add_p113_wd1.0_frac0.3_seed0 | 0 | 0.5011 / 0.4985 / 0.0005 | 0.2907 / 0.2907 / 0.0005 | 0.009 / 0.064 / 0.502 / 0.937 / 0.991 | 0.220 |
| | 1 | 0.4903 / 0.5090 / 0.0007 | 0.2760 / 0.2760 / 0.0007 | 0.015 / 0.071 / 0.486 / 0.921 / 0.983 | 0.211 |
| | 2 | 0.4795 / 0.5199 / 0.0006 | 0.2769 / 0.2769 / 0.0006 | 0.017 / 0.068 / 0.471 / 0.916 / 0.979 | 0.209 |
| | 3 | 0.5316 / 0.4676 / 0.0008 | 0.2486 / 0.2485 / 0.0006 | 0.020 / 0.123 / 0.542 / 0.909 / 0.985 | 0.237 |
| | mean over heads | 0.5006 / 0.4987 / 0.0007 | 0.2730 (to a) | | |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0 | 0 | 0.5018 / 0.4977 / 0.0005 | 0.3155 / 0.3155 / 0.0006 | 0.003 / 0.038 / 0.503 / 0.963 / 0.997 | 0.158 |
| | 1 | 0.4988 / 0.5000 / 0.0012 | 0.2776 / 0.2776 / 0.0011 | 0.002 / 0.063 / 0.499 / 0.935 / 0.997 | 0.218 |
| | 2 | 0.4953 / 0.5035 / 0.0013 | 0.2475 / 0.2475 / 0.0011 | 0.008 / 0.100 / 0.494 / 0.895 / 0.991 | 0.259 |
| | 3 | 0.5003 / 0.4988 / 0.0008 | 0.3174 / 0.3174 / 0.0011 | 0.005 / 0.037 / 0.501 / 0.963 / 0.994 | 0.160 |
| | mean over heads | 0.4991 / 0.5000 / 0.0009 | 0.2895 (to a) | | |

* The means reproduce `RESULTS.md` §2 (canonical "0.501/0.499/0.001"; Grokfast per head "0.502/0.498/0.001, 0.499/0.500/0.001, 0.495/0.504/0.001, 0.500/0.499/0.001"). One printed range is slightly off: canonical head 3 attends to b with mean 0.4676, marginally outside the stated "0.47–0.53".
* The per-input dispersion is large: std ≈ 0.25–0.32 around a mean of 0.50; the central 90 % of inputs span roughly 0.04–0.96; only 16–26 % of inputs lie within (0.4, 0.6). Additional descriptive decomposition (audit only): for every head, the a-token explains ≈ 0.48–0.49 of `var(to-a)` and the b-token independently ≈ 0.48–0.49 (variance of the per-token conditional means divided by total variance), i.e. the attention weight is close to additively separable in a and b.

### 1.5 The two `progress_measures.json` files (restricted / excluded loss)

Provenance: `training/runs/<run>/progress_measures.json` (written by `analysis/progress_measures.py` on 2026-08-18, `PROGRESS.md`).

| item | txf_add_p113_wd1.0_frac0.3_seed0 | txf_add_p113_wd1.0_frac0.5_gf2.0_seed0 |
|---|---|---|
| top-level keys | run_id, config, key_frequencies, transition, measured_steps, full_loss, restricted_loss, excluded_loss, test_acc_at_measured, train_acc_at_measured, sanity | same + torch_num_threads |
| `variants` block present | **no** | **no** |
| `top_level_protocol` key present | **no** | **no** |
| `sanity.key_freq_cap_binding` present | no | no |
| key_frequencies | 18, 15, 11, 1, 13, 56, 22, 36 (n = 8) | 16, 39, 17, 35, 8, 44, 21, 53 (n = 8) |
| measured steps | 105, last = 8367 | 81, last = 761 |
| transition (re-train) | 145 / 8367 / gap 8222; final test acc 0.9869 | 179 / 675 / gap 496; final test acc 0.9894 |
| `sanity.matches_recorded_run` | True (recorded 145 / 8367; recorded final test acc 0.9808) | True (recorded 179 / 675; recorded final test acc 0.9834) |
| final full / restricted / excluded loss | 0.03053 / 0.002035 / 10.4239 | 0.01647 / 0.000203 / 12.6227 |
| `sanity.reconstruction_max_abs_err` | 5.94e-12 | 2.34e-12 |
| restricted < full | at **every** measured step, including step 0 (4.7436 vs 4.7447) | at every measured step, including step 0 |
| excluded vs full during the plateau (steps 200–7000) | excluded **< full at 37 of 50** measured steps | 0 of 23 (run ends at 761) |
| max full loss / max excluded loss | 18.28 at step 1414 / 18.87 at step 7792 | 4.86 at step 72 / 21.18 at step 308 |
| ln(113) | 4.7274 | 4.7274 |

Both files predate `analysis/mask_protocols.py`: the `variants` block, `top_level_protocol` and `key_freq_cap_binding` that the current `compute()` writes are absent. Consequently **every** restricted/excluded number in `RESULTS.md` was produced by the outer-product mask now named `legacy_broad_mask` (`progress_measures._mode_indices`; bit-identical reproduction asserted by `test_core.py` "legacy protocol == original _mode_indices masks"). No `same_frequency_block`, `sum_directions_only` or Nanda-exact number exists anywhere in the repository.

### 1.6 Parameter counts and mask cardinalities

Provenance: `config.Config.n_params` and `sum(p.numel() for p in build_model(cfg).parameters())` (identical); `analysis.mask_protocols.build_protocol(name, 113, keys).n_kept / n_removed`.

| model | total params | per module |
|---|---|---|
| transformer (d_model 128, 4×32 heads, d_mlp 512) | **226,176** | W_E 14,592; W_pos 384; W_Q/W_K/W_V 16,384 each; W_O 16,384; W_in 65,536; W_out 65,536; W_U 14,592 |
| MLP (d_mlp 512) | **204,017** (9.80 % fewer) | W_E 14,464; W_in 131,072; W_out 57,856; b_in 512; b_out 113 |
| MLP (d_mlp 572, `mlp_param_matched` preset) | 226,217 (+0.018 %) | W_in 146,432; W_out 64,636; b_in 572 |

`docs/data/capacity_report.json` (AI-drafted 2026-09-02, unreviewed) additionally records expected total L2 norm at initialization 37.19 (transformer) vs 27.17 (MLP d_mlp 512), and final/initial total-L2 ratios ≈ 2.90–2.93 for the three un-accelerated MLP runs; these are reported here as existing artifacts, not as audited claims.

| mask protocol (p = 113, keys {18, 15, 1, 11, 13, 22, 56, 36}) | components kept by restricted | components removed by excluded | keeps cross-frequency blocks |
|---|---|---|---|
| `legacy_broad_mask` (the one behind all RESULTS.md numbers) | 289 of 12,769 | 3,360 of 12,769 (26.3 %) | yes |
| `same_frequency_block` | 33 | 32 | no |
| `sum_directions_only` | 17 (constant + 2 directions per frequency) | 16 | no |
| `sum_directions_only` with Nanda's five key frequencies {14, 35, 41, 42, 52} | 11 directions | 10 | no |

### 1.7 Data split pairing

`data.make_dataset(cfg)` seeds the permutation with `cfg.seed` only; `data.split_hash(train_idx)` is identical for the transformer and the MLP at each of seeds 0, 1, 2 (`train_frac = 0.3`: 3,831 train / 8,938 test pairs). The legacy cross-architecture runs are therefore split-paired by seed.

### 1.8 `python test_core.py` on 2026-09-02

Exit code 1. Output: **84 `[PASS]` lines, then `[FAIL] odd-minus-even separates square from sinusoid`** (`test_core.py` line 298–300), after which `check()` raises `SystemExit(1)`. The sections `check_mlp_mechanism()` (line 322 ff.) and `check_config_additions()` (line 400 ff.) are **never executed** — their headers do not appear in the output — so the "effective curves reproduce the model's forward pass", "synthetic circuit beats the permutation null", and "transformer has 226,176 parameters" assertions are currently unverified by the suite. `docs/BASELINE.md` gives the measured values behind the failure (square-wave `odd_minus_even` = 0.0918, sinusoid 0.0, required margin 0.10) and its diagnosed cause (harmonic aliasing onto other fundamentals and even-harmonic energy of a discrete square wave). `README.md` ("23 correctness checks") and `RESULTS.md` §5 ("23/23 core correctness checks pass") describe the June/August suite, not the current one.

---

## 2. Claim table

One row per central claim of `README.md` and `RESULTS.md`; the twelve statements the master prompt singles out have their own rows (marked ★) and a paragraph in §3.

| Claim | Existing evidence | Seeds | Metric | Alternative explanation | Methodological limitation | Evidence level | Status |
|---|---|---|---|---|---|---|---|
| Grokking is reproduced un-accelerated: train ≥ 0.99 at step 145 (1.000 at 156), test ≥ 0.95 at 8367, gap 8,222, final test acc 0.981 (README; RESULTS §1) | `runs/txf_add_p113_wd1.0_frac0.3_seed0/run.json` (commit 4ccac23): curves + `transition`; recomputed §1.1, §1.3 | 1 canonical (+2 seeds: 6295, 8367) | `train.detect_transition` thresholds 0.99 / 0.95 on the log grid | None for the phenomenon itself | Crossing is the first logged step: true crossing in (7792, 8367]; run stopped at the crossing (`--early-stop-acc 0.95`); memorization crossing resolution 10 steps | descriptive | solid (phenomenon); step indices are grid-resolved, not "exact" — see ★ log-grid row |
| Grokking is seed-robust (Grokfast frac 0.5: generalize 750 ± 97, final 0.986 ± 0.004; RESULTS §1) | 3 gf transformer runs; recomputed 750.0 ± 96.6, 0.9863 ± 0.0037 (§1.2) | 3 | as above; `compare.agg` ddof = 1 | — | Grokfast on; seed 2 trained with a 5000-step budget (seeds 0–1: 8000) → different evaluation grid; early stop at 0.98 | descriptive | solid |
| Deterministic re-train recovers the recorded transition "exactly" (145/8367; 179/675) (RESULTS §1–2, PROGRESS) | `progress_measures.json → sanity.matches_recorded_run = True` for both runs; final test acc differs (0.9869 vs 0.9808; 0.9894 vs 0.9834) | 2 runs (both seed 0) | equality of the two grid crossings | With a 575-step grid cell at 8367, two trajectories that differ in the last float digits will land on the same logged step unless the true crossing sits near a cell boundary | Equality of a grid index, not of the trajectory; final accuracies differ by 0.5–0.6 pp (thread-count effect, RESULTS §5) | descriptive | solid as "same grid crossing"; "exactly" overstates the resolution |
| Grokfast "compresses the timing of the same phase transition — it does not manufacture it" (RESULTS §1) | Both settings show plateau then jump (curves figures); top-8 of gf runs 0.52–0.64 vs un-acc. 0.66–0.79 | 3 + 3 | qualitative curve shape | Grokfast may change the learned solution, not only its timing (the measured concentration differs); Grokfast and `train_frac` change together | No mechanism comparison between gf and un-acc. models exists; two knobs differ (RESULTS §3 discloses this) | descriptive | preliminary |
| Grokked embedding concentrates 76 % of Fourier power in the top-8 frequencies vs 32 % non-grokked; "the periodic trig-identity circuit" (README; RESULTS §2) | canonical 0.7571; control `txf_add_p113_wd1.0_frac0.3_gf2.0_seed0` 0.3197; all 6 grokked transformer runs 0.52–0.79 (§1.1) | 1 vs 1 (contrast supported by 6 grokked runs) | `fourier.dominant_frequencies` top-8 of `W_E[:113]`; cap binding (n for 90 %: 27 vs 46) | Control is a Grokfast run with a 4000-step budget that never generalized — not a training-length-matched non-grokked model; concentration also rises with training length | Single control; W_E only; the word "circuit" is not supported by any mechanism or ablation test | correlational | preliminary (the concentration contrast is a solid descriptive fact; "trig-identity circuit" is not established) |
| "Sparsity tracks convergence": un-acc. 0.76 > Grokfast 0.52–0.64 "exactly as expected" (RESULTS §2) | 1 un-acc. run vs 3 gf runs (§1.1) | 1 vs 3 | top-8 of W_E | Runs differ in Grokfast, `train_frac` (0.3 vs 0.5), and steps trained (8367 vs 717–953) | Three confounds; n = 1 on one side; no time series of the metric across settings | correlational | overinterpreted |
| PCA of final embeddings shows a periodic ring (RESULTS §2) | `figures/*_pca_ring.png` (canonical, 3 gf txf seeds, 1 gf MLP seed) | 5 figures | visual | A 2D/3D PCA of any set of vectors dominated by two Fourier frequencies looks ring-like; no quantification | No numeric circularity statistic exists in any artifact | descriptive | preliminary |
| Key-frequency concentration over time rises across the transition — a held-out-independent progress signal (RESULTS §2) | curves figure panel 3 (`fourier.frequency_concentration_over_time`); not stored numerically in `run.json` | figures for canonical + gf runs | fraction of W_E power in the final run's top-8 frequencies, applied retroactively | Any metric defined by the final state rises toward the final state by construction; rising concentration during the plateau is consistent with but does not demonstrate circuit formation | Key set fixed post hoc (top-8 cap); no numeric export; no MLP series reported | correlational | preliminary |
| ★ Restricted and excluded loss reproduce Nanda et al.'s method (RESULTS §2 "We implement the Nanda et al. 2023 restricted and excluded loss"; §5 "reproduced and self-checked") | `analysis/progress_measures.py`; both `progress_measures.json` files carry only the `legacy_broad_mask` numbers (§1.5); Nanda §5.1 quoted in §4 | 2 runs (both transformer seed 0) | legacy mask: 289 kept / 3,360 removed on the full (a,b) grid; key set = top-8 of W_E | — | Protocol differs from the source in (i) mask width (source: constant + 20 terms for 5 frequencies = the sum directions; repo: 289 components incl. cross-frequency blocks), (ii) key-frequency selection and count (source: 5 key frequencies "appear in later parts of the network"; repo: top-8 cap on W_E), (iii) evaluation split (source: excluded loss "on the training data"; repo: full grid), (iv) restricted/excluded are not complements in the legacy mask; the repo's own `mask_protocols.NandaExact` raises "NOT reconstructed yet" | mechanistic (intended) | methodologically problematic |
| "The eight key frequencies alone solve the task (restricted ≤ full), while removing them collapses the network far below chance" (RESULTS §2, §5) | canonical: full 0.0305 / restricted 0.0020 / excluded 10.42; gf: 0.0165 / 0.0002 / 12.62 (§1.5) | 2 runs (seed 0 only) | legacy-mask restricted/excluded CE over all 12,769 inputs | Restricted keeps 289 of 12,769 coefficients (~17× the sum-direction circuit): a large low-pass reconstruction can beat the full logits without the kept set being "the circuit"; excluded deletes 26.3 % of the coefficient tensor of a confident model — a size-matched random deletion has never been run, so "far below chance" has no control | No random-control ablation; single seed; full-grid split; excluded < full at 37 of 50 plateau steps in the canonical run, which the "destroys the solution" reading does not accommodate | mechanistic (no control → not causal) | methodologically problematic |
| Restricted loss "separates from the full loss around memorization and stays below it through the entire plateau — the Fourier circuit is forming quietly beneath the memorized solution" (RESULTS §2) | canonical `progress_measures.json`: restricted < full at every step incl. step 0 (4.7436 vs 4.7447); plateau values restricted 6.7–7.9 vs full 11–18 (both ≫ ln 113 = 4.73); restricted reaches ≈ 3.1 near step 5000 while test acc ≈ 0.15–0.19 | 1 | legacy-mask restricted vs full CE, full grid | "restricted ≤ full" holds trivially even at initialization under this mask; a lower-but-still-far-above-chance restricted loss on a grid that is 70 % test points does not show a working circuit | Legacy mask; full grid; no split separation; n = 1 | correlational | overinterpreted |
| ★ 50/50 attention of the `=` position "directly links the learned structure to the computation" / "attention signature of a circuit that must combine both operands" (RESULTS §2; README) | `analysis/attention.py` means (§1.4): 0.5006 / 0.4987 / 0.0007 over heads; per-head means 0.47–0.53; Grokfast run similar | 2 runs (seed 0, canonical + gf) | mean over 12,769 inputs of the softmax weight from `=` to a / b / = | Per-input std 0.25–0.32 with 5–95 % range ≈ 0.04–0.96: the attention is strongly input-dependent and only its mean is 0.5; Nanda App. C.1.3's form "0.5 ± C(cos(w(a+θ)) − cos(w(b+θ)))" averages to exactly 0.5 over the full grid for any C, so the mean carries no information about the mechanism; with a causal mask over 3 positions, near-zero self-attention plus non-zero weight on both operands is required of any 1-layer transformer that uses both inputs | Mean only (no dispersion, no per-input analysis, no head ablation, no phase-resolved measurement) | descriptive | overinterpreted |
| ★ The attention structure explains the earlier generalization ("The transformer's attention bias … steers it toward a cleaner trig-identity-style circuit", RESULTS §3) | None beyond the mean-attention numbers and the timing gap | 2 (attention) / 3 + 3 (timing) | — | Parameter count (204,017 vs 226,176), initial weight norm (27.2 vs 37.2), effective weight-decay pressure, input parametrization (concatenation vs mixing), hyperparameters tuned for the transformer recipe (Manir & Rupa: gap "largely disappears (1.11×) under matched hyperparameters"; at optimal λ each their MLP is *earlier*: 26,800 vs 50,800) | No intervention on attention; no confound control; no mechanism measured in the MLP at all | descriptive (not even correlational: attention is not measured in the MLP nor across seeds) | overinterpreted |
| MLP generalizes later than the transformer in every seed (RESULTS §3) | 6 split-paired seed pairs, all MLP-later (§1.2) | 3 + 3 per setting | first logged step with test ≥ 0.95 | Confounds as in the row above; grid snapping makes two of the three un-acc. transformer crossings identical | n = 3; un-acc. crossings resolved to 432–711 steps; gf budgets differ across seeds | descriptive | solid in direction under these exact hyperparameters; preliminary as a general architecture statement |
| Speed ratio ~2.9× (Grokfast/0.5) vs ~1.3× (un-acc./0.3), "setting-dependent" (RESULTS §3) | means 2162.7/750.0 = 2.884; 9883.0/7676.3 = 1.287 (§1.2) | 3 + 3 per setting | ratio of mean crossings | Either Grokfast or `train_frac` or both; not separable (RESULTS §3 says so) | Two knobs change at once; n = 3; transformer spread ± 1196 | descriptive | preliminary (the repo already labels it so) |
| ★ 0.73 vs 0.44 shows different algorithms / "sparsity difference — robust across settings … per-seed gaps never overlap" (RESULTS §3) | top-8 of W_E: 0.7338 ± 0.0663 vs 0.4358 ± 0.0085 (un-acc.), 0.5896 vs 0.3524 (gf); per-seed ranges disjoint (§1.1–1.2) | 3 + 3 per setting | top-8 concentration of W_E at the crossing | (a) metric artifact: a harmonic/square-wave representation spreads power over aliased odd harmonics (RESEARCH_SPEC H3; Swaroop reports "near-binary square wave input weights" in ReLU MLPs); (b) at-crossing measurement (Khanh: at-grok metrics overstate/misstate converged values, "3-5x on an MLP, 1.3-1.5x on a transformer"); (c) W_E plays different roles in the two models; (d) parameter/regularization confound; (e) the gap is already reported by Manir & Rupa (98.5 % vs 74.7–75.6 % top-5) *while* "all three models converge to highly sparse Fourier representations" — a concentration gap coexists with the same algorithm family | Wrong object for the MLP (see ★ W_E row); cap k = 8; early stop; no mechanism test on either side | correlational | overinterpreted |
| ★ "The Transformer learns a sparser Fourier circuit" (README "markedly sparser Fourier circuit"; RESULTS §3) | Same numbers as the row above | 3 + 3 | top-8 of W_E | As above; a sparser *embedding spectrum* is not a sparser *circuit* without evidence that the embedding's frequencies are the ones the network computes with (Nanda derives key frequencies from later layers) | "Circuit" requires mechanism/causal evidence; none exists; "converges to" is contradicted by early stop at the crossing | correlational | overinterpreted |
| ★ "The MLP uses a more distributed solution" / "solves the same task with a more distributed frequency representation" (RESULTS §3) | None beyond top-8 of W_E (0.44 / 0.35) | 3 + 3 | top-8 of W_E | The neuron input in `models/mlp.py` is `W_E @ W_in[:d]` and `W_E @ W_in[d:]`; W_in can select/rotate so that per-neuron curves are sharply periodic while W_E's spectrum is diffuse; harmonic spreading (H3) | No effective-weight, activation, or output-weight analysis of any MLP run exists (`mlp_mechanism.py` has never been run on a run directory; no `mlp_mechanism.json` in any run dir) | descriptive (of W_E), none for the "solution" | methodologically problematic |
| "The MLP's lower sparsity is partly architectural (concatenated operand embeddings vs attention-combined)" (RESULTS §3, §5) | None (assertion) | — | — | Could be metric artifact, stopping-time artifact, or parametrization; untested | Stated as a partial explanation without any test | none | preliminary (hypothesis presented as partial fact) |
| "Both architectures grok to ≥98 % test accuracy in both settings" (RESULTS §3) | Contradicted: 0.9653 (txf un-acc. seed 1), 0.9751 (MLP un-acc. seed 1) (§1.1) | 12 | final test acc at last logged step | — | Factually wrong for 2 of 12 runs (RESEARCH_SPEC §14 item 1) | descriptive | overinterpreted (as written, false) |
| ★ "Both models learn the same function" (RESEARCH_SPEC research question "learn the *same* rule … in *different* ways"; RESULTS §3 "solves the same task"; PLAN Phase 4 "converge to the same trig-identity-style algorithm") | Only per-architecture test accuracies 0.965–0.994 | 3 + 3 | test accuracy | Two models with 0.6–3.5 % test error can disagree on hundreds of inputs; identical accuracy does not imply identical function or identical algorithm | No prediction-agreement, error-overlap, or logit-similarity measurement over the 12,769 inputs exists (master prompt §10) | descriptive | overinterpreted — permitted formulation: "both architectures generalize on the same task" |
| "Within-setting comparisons are controlled" (RESULTS §3, "Honest limits") | Same seeds/splits (§1.7), same optimizer, thresholds, p, task | 3 + 3 | — | Uncontrolled: parameter count (−9.80 % for the MLP), init norm, effective regularization, input parametrization, thread count (6 vs unrecorded), code commit (4ccac23 / ead5166 / 1a35a9c), Grokfast step budgets | Controlled only for split/seed/optimizer | descriptive | overinterpreted |
| ★ Early stop at the generalization crossing: reported "final"/"converged" structure numbers are at-transition numbers (README "converges to a markedly sparser Fourier circuit") | All six un-acc. runs: last logged step == generalization step (§1.1); gf runs stop 1–3 grid steps after the crossing (test ≥ 0.98) | 12 | — | Cleanup-phase sharpening after the crossing (Nanda §5.2: cleanup phase 9.4k–14k epochs in their run) may differ between architectures; Khanh: at-grok metrics differ from converged ones by architecture-dependent factors | Every legacy structure number, including 0.73/0.44 and 0.76, is measured before any cleanup; no post-crossing checkpoint exists | descriptive | methodologically problematic (for any "converged"/"final circuit" wording) |
| ★ Log-grid transition-step resolution ("exact step indices", RESULTS §5; "generalizes at step 8367") | §1.3: spacing 432–711 steps at the un-acc. crossings; two txf seeds share 8367, two MLP seeds share 9646 | 12 | first logged step ≥ threshold | The seed-to-seed spread of the transformer (± 1196) is 2–3 grid cells; identical crossings across seeds are a grid artifact, not a real coincidence | No dense evaluation exists for the legacy runs; the phrase "exact" is not supported by a discrete grid (master prompt §15) | descriptive | methodologically problematic (for "exact"); the intervals themselves are solid |
| ★ Fixed `max_k = 8` cap: "the eight key frequencies", "top-8" | `fourier.dominant_frequencies`: `cap_binding = True` in all 16 runs; frequencies needed for 90 %: 23–50 (§1.1); Nanda §4.1: W_E has "significant nonnegligible norm at 6 frequencies", five key frequencies | 16 | — | The number 8 is neither Nanda's 5 nor measured; the count that a 90 % criterion would give (23–50) says the legacy models are far less Fourier-sparse in W_E than Nanda's converged model (consistent with early stop) | Key-frequency set for the progress measures inherits the cap; cross-architecture comparison at a fixed k penalizes any representation whose power is spread over more than 8 lines by construction (H3) | descriptive | methodologically problematic |
| ★ W_E-only measurement for the MLP (all structural MLP claims) | `compare.summarize_run` → `dominant_frequencies(embeds[-1])`, i.e. `W_E` only; transformer W_E has 114 rows incl. `=`, MLP 113 | 3 + 3 | top-8 of W_E | In the MLP, W_E feeds two different halves of W_in; in the transformer, W_E feeds Q/K/V and the residual stream; the two matrices are not the same object functionally | No `u_a = W_E @ W_in[:d]`, `u_b`, activation, or `W_out` analysis has been run; no transformer W_in/W_out/W_U analysis either | descriptive | methodologically problematic |
| ★ Failing `test_core.py` check (`docs/BASELINE.md`); "23/23 core correctness checks pass" (README, RESULTS §5) | 2026-09-02 run: 84 PASS, 1 FAIL (`odd-minus-even separates square from sinusoid`), exit 1; `check_mlp_mechanism` and `check_config_additions` never executed (§1.8) | — | — | The failing statistic is a WIP H3 metric, not one behind any published number; but the abort also leaves the effective-weight and parameter-count checks unrun | Suite aborts on first failure; README/RESULTS test counts are stale | — | methodologically problematic (test state); "23/23 pass" overinterpreted (stale) |

---

## 3. The statements to be treated especially critically

### 3.1 "The Transformer learns a sparser Fourier circuit."

What is measured: the fraction of the Fourier power of the token-embedding matrix `W_E[:113]` that lies in its 8 strongest frequencies at the last logged step — 0.7338 ± 0.0663 (transformer) vs 0.4358 ± 0.0085 (MLP) un-accelerated, 0.5896 ± 0.0627 vs 0.3524 ± 0.0033 with Grokfast (`fourier.dominant_frequencies`, §1.2). The per-seed ranges are disjoint in both settings. That statistic is solid as a description of `W_E`. The claim, however, contains three words that the statistic does not cover. *Sparser*: the count 8 is a cap that binds in every run (§1.1), and by the same function's 90 % criterion the sparsest transformer needs 23 frequencies — the transformer is "sparser" only relative to a cutoff chosen in advance. *Fourier*: the fraction is measured in the Fourier basis of `W_E` alone; whether the network computes with those frequencies (Nanda §4.1 derives the key frequencies from where they "appear in later parts of the network") is not measured. *Circuit*: no restricted/excluded measurement exists for the MLP, and the transformer's exists only under the legacy broad mask without a control (§3.4). The runs are also early-stopped at the crossing, so "learns"/"converges to" describes a state before any cleanup. Manir & Rupa report a comparable embedding-concentration gap (98.5 % vs 74.7–75.6 % top-5) in a setting where "All three models converge to highly sparse Fourier representations" (§4.3); a concentration gap therefore does not, on its own, distinguish circuits. Classification: correlational; overinterpreted.

### 3.2 "The MLP uses a more distributed solution."

The only evidence is the same `W_E` statistic. For the MLP, `W_E` is not the input a hidden neuron sees: `models/mlp.py` concatenates `W_E[a]` and `W_E[b]` and multiplies by `W_in`, so neuron *i* sees `u_a[a,i] = W_E[a] · W_in[:128, i]` and `u_b[b,i] = W_E[b] · W_in[128:, i]` (`analysis/mlp_mechanism.effective_curves`, verified against the forward pass in `test_core.check_mlp_mechanism` — a check that currently never runs, §1.8). A diffuse `W_E` spectrum is compatible with sharply periodic per-neuron curves because `W_in` is free to select and rotate. No effective-curve, activation-map or output-weight measurement has been run on any MLP run directory (no `mlp_mechanism.json` exists). The statement therefore rests on a quantity that is not the MLP's solution. A second alternative is the H3 measurement artifact: a square-wave-like curve at fundamental *k* spreads power over the aliased odd harmonics 3k, 5k, 7k (e.g. 18 → 54, 23, 13 in ℤ₁₁₃; `fourier.alias_frequency`), which a top-8 metric scores as "distributed"; Swaroop reports exactly such "near-binary square wave input weights" for ReLU MLPs (§4.5). Classification: descriptive of `W_E`, no evidence about the solution; methodologically problematic.

### 3.3 "The attention structure explains the earlier generalization."

`RESULTS.md` §3 writes that the transformer's attention "steers it toward a cleaner trig-identity-style circuit". No experiment connects attention to timing: attention is measured on two transformer runs (seed 0 of each setting) as a mean, there is no head ablation, no attention measurement over training, no MLP-side counterpart, and no confound control. The timing difference itself is confounded by parameter count (204,017 vs 226,176, §1.6), initial weight norm (27.2 vs 37.2, `capacity_report.json`), effective weight-decay pressure, input parametrization, and hyperparameters taken from the transformer recipe. Manir & Rupa, in a controlled study on mod 97, report that "the apparent gap between Transformers and MLPs largely disappears (1.11× delay) under matched hyperparameters" and that at optimal λ for each architecture their MLP grokked *earlier* (26,800 ± 6,419 vs 50,800 ± 38,745, §4.3) — the direction of the gap is itself hyperparameter-dependent in the literature. Classification: not even correlational (attention is not measured across the compared conditions); overinterpreted.

### 3.4 "Restricted and excluded loss reproduce Nanda's method."

The source defines the restricted loss (Section 5.1, quoted in §4.1) as keeping "the constant term and the 20 terms corresponding to cos(wk(a+b)) and sin(wk(a+b)) for the five key frequencies" — the sum directions only — and the excluded loss as removing "only those key frequencies from the logits but keep the rest", measured "on the training data". The repository's numbers (§1.5) come from `legacy_broad_mask`: the outer product of a 1-D key-row mask, which keeps 289 coefficients including cross-frequency products such as cos(ω₁₈a)cos(ω₁₅b) and, for the excluded loss, removes 3,360 coefficients (26.3 % of the tensor), evaluated over the full grid of 12,769 inputs (70 % of which are test points). The key set is the top-8 of `W_E` under the cap, whereas the source identifies five key frequencies from "later parts of the network". The repository's own `mask_protocols.NandaExact` raises `NotImplementedError("nanda_exact is not implemented … Guessing it would manufacture the false reproduction claim")`, and the current `progress_measures.compute()` prints that the top-level keys "are NOT a reproduction of Nanda et al." — but the two stored JSON files predate that code and carry only the legacy numbers with no `variants` block. `RESULTS.md` discloses one deviation (full grid instead of per-split) and none of the others. The split on which the source evaluates the *restricted* loss is `[NOT FOUND IN SOURCE]` in the fetched text (§4.1). Classification: mechanistic in intent; methodologically problematic; the numbers are "not directly comparable to Nanda" (master prompt §6 item 6).

### 3.5 "0.73 versus 0.44 shows different algorithms."

The numbers are reproduced (0.7338 ± 0.0663 vs 0.4358 ± 0.0085, n = 3 each, paired seeds, split-matched). They are a difference in one descriptive statistic of one weight matrix at one training-time point (the crossing). Five alternative explanations remain open, none tested: (a) the H3 metric artifact (harmonic spreading under a fixed k = 8); (b) the stopping-time artifact — both architectures are measured at their own crossing, and Khanh reports that at-grok representation metrics differ from converged values by architecture-dependent factors ("3-5x on an MLP … 1.3-1.5x on a transformer", §4.2); (c) `W_E` is functionally a different object in the two models (§3.7); (d) parameter-count and regularization confounds; (e) the same-family-different-concentration outcome already documented by Manir & Rupa. "Different algorithms" would require, at minimum, a mechanism measurement in both models and a causal test; neither exists. Classification: correlational; overinterpreted.

### 3.6 "50/50 attention is proof of the addition mechanism."

`README.md` states that the `=` position attends "~50/50 to both operands", and `RESULTS.md` §2 that this "directly links the learned structure to the computation". The mean is reproduced (0.5006 / 0.4987 / 0.0007 over heads, canonical run; §1.4). The recomputation adds what `attention.py` does not report: the per-input standard deviation is 0.25–0.29 (canonical) and 0.25–0.32 (Grokfast) around the mean of 0.50; the central 90 % of inputs span roughly 0.04–0.96; only 21–24 % of inputs attend to *a* with a weight in (0.4, 0.6). The attention is strongly input-dependent, and 50/50 is a property of the average only. Nanda's own description (App. C.1.3, §4.1) writes the attention as "0.5±Cⱼ(cos(wₖⱼ(a+θⱼ))−cos(wₖⱼ(b+θⱼ)))"; the cosine terms average to zero over the full input grid, so that form yields a mean of exactly 0.5 for *any* amplitude C — the mean cannot discriminate between a constant 50/50 split, Nanda's frequency-dependent form, or any other zero-mean input dependence. (The audit's variance decomposition — a-token and b-token each explaining ≈ 0.49 of `var(to-a)` — is *consistent* with an additively separable form but is itself only descriptive.) Two further points: the criterion `attends_both_operands` in `attention.py` is a > 0.1 threshold on the mean; and with a causal mask over three positions, any 1-layer transformer that uses both operands must give both non-zero weight and, if the model is confident, near-zero self-weight — a necessity, not a signature of *this* algorithm. No head ablation exists. Classification: descriptive; overinterpreted. The master prompt's rule applies: a symmetric attention distribution alone is not causal evidence.

### 3.7 "Both models learn the same function."

`RESULTS.md` uses "solves the same task"; `docs/RESEARCH_SPEC.md` frames the study as whether the two "learn the *same* rule … in *different* ways", and `PLAN.md` Phase 4 asks whether they "converge to the *same* trig-identity-style algorithm". The only shared measurement is test accuracy (0.965–0.994 per run). No prediction-agreement, shared-error, or logit-similarity comparison over the 12,769 inputs exists in any artifact; with 0.6–3.5 % test error per model, the two could disagree on hundreds of held-out inputs. Until such a measurement exists, the only permitted formulation is **"both architectures generalize on the same task"** (master prompt §10). Classification: descriptive; overinterpreted whenever "same function" or "same algorithm" is used.

### 3.8 Early stop at the generalization crossing

All six un-accelerated runs were launched with `--early-stop-acc 0.95` (`PROGRESS.md` 2026-08-18) and their last logged step equals the detected generalization step (§1.1); the Grokfast runs stop at the first logged step with test ≥ 0.98, one to three grid steps after the 0.95 crossing. Consequently every "final" structure number in `README.md`/`RESULTS.md` — 0.76, 0.73 vs 0.44, 0.59 vs 0.35, the PCA rings, the attention means, and the final restricted/excluded losses — is measured at the transition, before the cleanup phase that Nanda describes for their run as a distinct third phase (§5.2: "network is becoming sparser in the Fourier basis"; cleanup at "Epochs 9.4k–14k" in their training, quoted §4.1). Two different architectures may be at different distances from convergence at their respective crossings; Khanh (§4.2) reports architecture-dependent discrepancies between at-grok and converged metric values. Any wording of the form "converges to", "final circuit", or "the grokked solution" is therefore not supported by the artifacts; the numbers remain valid as at-crossing numbers. Status: methodologically problematic for converged-state claims.

### 3.9 Log-grid transition-step resolution

Every crossing is the first logged step at or above threshold on `log_step_schedule(steps, 150)`. At the un-accelerated crossings the grid spacing is 432–711 steps (§1.3): the canonical "8367" means "in (7792, 8367]", the MLP "9646" means "in (8984, 9646]". The artifact is visible in the data: two transformer seeds report the identical crossing 8367 and two MLP seeds the identical 9646. `RESULTS.md` §5 speaks of "exact step indices"; the master prompt §15 disallows "exact transition" on a discrete grid. The memorization crossings are better resolved (10–11 steps). The direction of the cross-architecture timing difference survives the resolution (the smallest un-acc. MLP interval, (8984, 9646], lies entirely above the largest transformer interval, (7792, 8367]), but the magnitude (1.29×) carries an unstated uncertainty of roughly one grid cell per run. Status: solid as intervals; methodologically problematic as "exact" steps.

### 3.10 The fixed `max_k = 8` cap

`fourier.dominant_frequencies(threshold=0.9, max_k=8)` returns `min(n_wanted, 8)` frequencies. In all 16 runs `cap_binding` is `True` and `n_freqs_for_threshold` is 23–50 (§1.1); the 90 % criterion never fires. The "eight key frequencies" of the progress measures and the "top-8" of every concentration number are therefore a fixed constant, not a measured property of any model, and the same constant is applied to a transformer and an MLP whose spectra may differ in *shape* (H3), which a fixed-k concentration cannot see. The source model uses six non-negligible embedding frequencies and five key frequencies identified downstream (§4.1); the 23+ frequencies the legacy models need for 90 % says that they are, in `W_E`, far from that state at the crossing — consistent with §3.8. Status: methodologically problematic (as the basis of "key frequencies" and of the cross-architecture comparison).

### 3.11 W_E-only measurement for the MLP

`analysis/compare.summarize_run` computes `dominant_frequencies(embeds[-1])` — `W_E` at the last logged step — for both architectures and nothing else. For the transformer, `W_E` (114 rows incl. `=`) feeds the residual stream, Q/K/V and, via attention, the MLP; for the MLP, `W_E` (113 rows) is read through two separate halves of `W_in`. No analysis of `W_in`, `b_in`, hidden activations or `W_out` exists for the MLP, and none of `W_Q/W_K/W_V/W_O/W_in/W_out/W_U`, the residual stream, or MLP neurons for the transformer. Every structural cross-architecture claim in the repository is thus a comparison of one matrix whose role differs between the two models. `analysis/mlp_mechanism.py` implements the effective-curve measurement and labels itself "MEASUREMENT ONLY"; it has produced no artifact. Status: methodologically problematic.

### 3.12 The failing `test_core.py` check (`docs/BASELINE.md`)

Re-run on 2026-09-02: 84 checks pass, then `[FAIL] odd-minus-even separates square from sinusoid` (`test_core.py` lines 298–300) and the process exits with code 1 (§1.8). `docs/BASELINE.md` records the measured values (square wave 0.0918, sinusoid 0.0, required margin 0.10) and the diagnosed cause (six harmonics of the canonical fundamental set alias onto other fundamentals — 2·18→36, 7·18→13, 2·11→22, 7·11→36, 7·13→22, 2·56→1 — and a discrete square wave sampled at odd p carries even-harmonic energy, `even_share = 0.083`). Two consequences for the evidence classification: (i) the `harmonic_shape` statistic, which `fourier.py` designates as "the statistic that actually decides H3", does not currently separate its two synthetic reference cases at the declared margin, so no H3-type inference could be drawn from it today; (ii) because `check()` raises `SystemExit` on the first failure, `check_mlp_mechanism()` and `check_config_additions()` are never reached, leaving the effective-curve/forward-pass equivalence, the synthetic phase-relation test, and the 226,176/204,017/226,217 parameter assertions unverified by the suite (they were verified independently for this audit, §1.6, but not by `test_core.py`). The counts "23 correctness checks" (README) and "23/23 … pass" (RESULTS §5) describe an earlier suite. Status: methodologically problematic (test state); the numbers behind `RESULTS.md` do not depend on the failing statistic.

---

## 4. Primary-source passages used (verbatim from the fetched page text, 2026-09-02)

The text below is what the fetch tool returned from each URL; section and figure numbers are as reported in that text. Nothing here is from memory.

### 4.1 Nanda et al., arXiv:2301.05217 (fetched `https://arxiv.org/html/2301.05217`)

* Section 3 (setup): "one-layer ReLU transformer, token embeddings with d=128, learned positional embeddings, 4 attention heads of dimension d/4=32, and n=512 hidden units"; "30% of the entire set of possible inputs...full batch gradient descent using the AdamW optimizer with learning rate γ=0.001 and weight decay parameter λ=1. We perform 40,000 epochs of training." — matches `config.Config` defaults (`d_model=128`, `n_heads=4`, `d_head=32`, `d_mlp=512`, `train_frac=0.3`, `lr=1e-3`, `weight_decay=1.0`; `steps` default 30000, legacy un-acc. runs 40000).
* Section 4.1 (key frequencies): "We apply a Fourier transform along the input dimension of the embedding matrix W_E then compute the ℓ₂-norm along the other dimension; results are shown in Figure 3. The embedding matrix W_E is sparse in the Fourier basis–it only has significant nonnegligible norm at 6 frequencies." and "Of the six non-zero frequencies, five 'key frequencies' appear in later parts of the network, corresponding to k∈{14,35,41,42,52}."
* Section 5.1 (restricted loss): "we perform a 2D DFT on the logits to write them as a linear combination of waves in a and b, and set all terms besides the constant term and the 20 terms corresponding to cos(wk(a+b)) and sin(wk(a+b)) for the five key frequencies to 0. We then measure the loss of the ablated network." Evaluation split for the restricted loss: `[NOT FOUND IN SOURCE]` (fetch result: "No sentence specifies the split for restricted loss").
* Section 5.1 (excluded loss): "Instead of keeping the important frequencies wk, we next remove only those key frequencies from the logits but keep the rest. We measure this on the training data to track how much of the performance comes from Fourier multiplication versus memorization."
* Section 5.2 (phases, Figure 7): "We plot the excluded loss, restricted loss, Gini coefficient of the matrices W_U and W_L, and sum of squared weights in Figure 7. We find that training splits into three phases, which we call the memorization, circuit formation, and cleanup phases." Memorization ("Epochs 0k–1.4k"): "model memorizes the data, and the frequencies wₖ used by the final model are unused"; circuit formation ("Epochs 1.4k–9.4k"): "model's behavior on the train set transitions smoothly from the memorizing solution to the Fourier multiplication algorithm...circuit is formed well before grokking occurs"; cleanup ("Epochs 9.4k–14k"): "network is becoming sparser in the Fourier basis". Progress measures "improve continuously prior to when grokking occurs."
* Section 4.3 / Appendix C.1.3 (attention): "the attention patterns of each head will be well approximated by 0.5±Cⱼ(cos(wₖⱼ(a+θⱼ))−cos(wₖⱼ(b+θⱼ)))"; Appendix C.1.3: "As argued in Appendix A.1, the attention paid = to = is negligible and can be ignored. So the softmax reduces to a softmax over two elements, which is a sigmoid on their difference."

### 4.2 Khanh, arXiv:2607.06639 (fetched `https://arxiv.org/abs/2607.06639`)

Title as fetched: "At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics", author Truong Xuan Khanh. Abstract sentence: "Reading effective rank at the grokking transition overstates the converged value by 3-5x on an MLP, and by 1.3-1.5x on a transformer trained to convergence." (Only the abstract was fetched; the metric quoted is effective rank, not top-k concentration.)

### 4.3 Manir & Rupa, arXiv:2603.25009 (fetched `https://arxiv.org/abs/2603.25009` and `https://arxiv.org/html/2603.25009`)

Title as fetched: "A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization", authors Shalima Binta Manir, Anamika Paul Rupa. Abstract: "We present a controlled study that systematically disentangles these factors on modular addition (mod 97), with matched and carefully tuned training regimes across models." and "the apparent gap between Transformers and MLPs largely disappears (1.11× delay) under matched hyperparameters, indicating that previously reported differences are largely due to optimizer and regularization confounds." Section 5.7, Table 8 caption: "Fourier concentration of learned embeddings post-grokking. Top-5 concentration = fraction of total Fourier energy in the five highest-energy frequency components." Section 5.7 results: "All three models converge to highly sparse Fourier representations...the Transformer concentrates 98.5% of its embedding energy in just 5 frequencies, compared to 74.7% for MLP-GELU and 75.6% for MLP-ReLU." Section 6.1: "At optimal λ each (Part C of H4), the gap is 1.90× (MLP: 26,800±6,419; Transformer: 50,800±38,745)" (hyperparameters reported for that comparison: MLP λ=10⁻³, SGD, lr=3×10⁻²; Transformer λ=5.0, AdamW, lr=10⁻³). Note: their setting (mod 97, different hyperparameters, GELU/ReLU MLP variants, post-grokking measurement) is not the GROKVERSE setting; the quotes are used only as documented alternative explanations.

### 4.4 Swaroop, arXiv:2603.23784 (fetched `https://arxiv.org/abs/2603.23784`)

Title as fetched: "Latent Algorithmic Structure Precedes Grokking: A Mechanistic Study of ReLU MLPs on Modular Arithmetic", author Anand Swaroop. Abstract: "We find empirically that ReLU MLPs in our experimental setting instead learn near-binary square wave input weights, where intermediate-valued weights appear exclusively near sign-change boundaries, alongside output weight distributions whose dominant Fourier phases satisfy a phase-sum relation φ_out = φ_a + φ_b; this relation holds even when the model is trained on noisy data and fails to grok." (Abstract only; the methods sections have not been recorded, so the `[UNVERIFIED]` label in `docs/RESEARCH_SPEC.md` H2 for the *methods* remains; the abstract-level claim itself is now verified as quoted.)

### 4.5 Not fetched

Power et al. arXiv:2201.02177, Lee et al. (Grokfast) arXiv:2405.20233, Liu et al. (Omnigrok) arXiv:2210.01117, Doshi et al. arXiv:2310.13061 were not fetched for this audit; the repository's citations of them are not evaluated here.

---

## 5. Not classified / out of scope

* Web-explorer, LiveLab and guided-tour statements in `README.md` (e.g. "LiveLab groks in-browser") are software claims, not scientific claims about the study; not classified.
* The three `txf_mul_*` runs: only the raw per-run numbers are listed (§1.1); no interpretation (reserved for the human author, `PROGRESS.md` 2026-08-18).
* `docs/data/capacity_report.json` and the `_smoketest` run directory are AI-drafted, unreviewed artifacts created on 2026-09-02; their contents are cited only where stated and are not themselves audited.
* Literature-level positioning (what Manir & Rupa, Swaroop, Doshi, Khanh establish and leave open) belongs to `docs/NOVELTY_AND_RELATED_WORK.md`, not to this audit.

---

## Appendix A — Summary of the recomputed headline numbers vs. the published ones

| published (README / RESULTS) | recomputed (this audit) | provenance |
|---|---|---|
| 145 / 156 / 8367 / 8,222 / 0.981 (canonical) | 145 (prev 135) / 156 / 8367 (prev 7792) / 8222 / 0.9808 | `run.json` txf_add_p113_wd1.0_frac0.3_seed0 → `transition`, `curves` |
| test loss 4.7 → 0.06, "~26" during the plateau | 4.746 → 0.0597; max 26.00 at step 1226 | same, `curves.test_loss` |
| 76 % vs 32 % non-grokked | 0.7571 vs 0.3197 | `embeddings.npy[-1]` → `fourier.dominant_frequencies` |
| 0.52–0.64 (Grokfast txf seeds) | 0.5206 / 0.6050 / 0.6432 | same |
| 750 ± 97, 559 ± 86, 0.59 ± 0.06, 0.986 ± 0.004 | 750.0 ± 96.6, 559.0 ± 86.0, 0.5896 ± 0.0627, 0.9863 ± 0.0037 | `compare.agg` (ddof = 1) |
| 2163 ± 72, 1959 ± 72, 0.35 ± 0.003, 0.989 ± 0.005 | 2162.7 ± 72.2, 1958.7 ± 72.2, 0.3524 ± 0.0033, 0.9886 ± 0.0052 | same |
| 7676 ± 1196, 7531 ± 1196, 0.73 ± 0.07, 0.980 ± 0.014 | 7676.3 ± 1196.3, 7531.3 ± 1196.3, 0.7338 ± 0.0663, 0.9799 ± 0.0142 | same |
| 9883 ± 410, 9723 ± 414, 0.44 ± 0.01, 0.984 ± 0.010 | 9883.0 ± 410.5, 9723.3 ± 413.7, 0.4358 ± 0.0085, 0.9841 ± 0.0096 | same |
| ~2.9× / ~1.3× | 2.884 / 1.287 | ratio of the means above |
| "≥98 % in both settings" | **not reproduced**: 0.9653, 0.9751 | `run.json` → `transition.final_test_acc` |
| attention 0.501 / 0.499 / 0.001 (canonical, mean over heads); per-head 0.47–0.53 | 0.5006 / 0.4987 / 0.0007; head 3 to-b = 0.4676; **per-input std 0.25–0.29** | `attention_pattern`, all 12,769 inputs |
| full 0.031 / restricted 0.0020 / excluded 10.42; 0.017 / 0.0002 / 12.62 | 0.03053 / 0.002035 / 10.4239; 0.01647 / 0.000203 / 12.6227 (legacy mask only) | `progress_measures.json → sanity.final_*` |
| reconstruction ~1e-12 | 5.94e-12 / 2.34e-12 | `sanity.reconstruction_max_abs_err` |
| key freqs {18, 15, 11, 1, 13, 56, 22, 36} = same set as spectrum | confirmed (same set, different order) | `progress_measures.json → key_frequencies`; `dominant` |
| 226,176 / 204,017 / 226,217 params; −9.8 %; +0.02 % (RESEARCH_SPEC §3.6) | 226,176 / 204,017 / 226,217; −9.80 %; +0.018 % | `Config.n_params`, model numel |
| 289 / 33 / 17 kept; 3,360 / 32 / 16 removed (RESEARCH_SPEC §3.1) | confirmed | `mask_protocols.build_protocol` |
| "23/23 checks pass" | 84 PASS + 1 FAIL, exit 1; two sections not executed | `python test_core.py` |

## Appendix B — Where each central statement is written

* README "What we found": memorizes by 145, exactly 1.000 by 156, ~8,000-step plateau, 8367, gap 8,222, 0.981; 76 % vs 32 %; "periodic 'trig-identity' circuit"; "attending ~50/50"; "markedly sparser Fourier circuit"; "0.73 vs 0.44"; "generalizes earlier in every seed"; "~2.9× → ~1.3×"; "23 correctness checks".
* RESULTS §1: canonical table; determinism; Grokfast "compresses the timing … does not manufacture it".
* RESULTS §2: 76 % / 32 % / 0.52–0.64; "Sparsity tracks convergence"; PCA ring; progress measure (embedding); restricted/excluded "VERIFIED"; "the eight key frequencies alone solve the task"; "Fourier circuit is forming quietly beneath the memorized solution"; attention 50/50 "directly links the learned structure to the computation".
* RESULTS §3: two tables; "≥98% test accuracy in both settings"; "Sparsity difference — robust across settings"; "attention bias … steers it toward a cleaner trig-identity-style circuit"; "more distributed frequency representation"; "Speed difference — direction robust, magnitude setting-dependent"; "Honest limits".
* RESULTS §5: "exact step indices"; "23/23 core correctness checks pass"; "restricted/excluded-loss progress measures reproduced"; "MLP's lower sparsity is partly architectural".

## Appendix C — Reproduction of the recomputed numbers (repository functions only)

From `training/` with the venv Python and `PYTHONPATH=.`:

* Top-8 per run: `dominant_frequencies(np.load(run/"embeddings.npy")[-1], p=113)` → `dominant_fraction`, `dominant`, `n_freqs_for_threshold`, `cap_binding` (`grokverse.analysis.fourier`).
* Aggregates: `agg([...])` from `grokverse.analysis.compare` over the three runs of each `(arch, setting)` selected by `compare.discover(arch, frac, grokfast)`.
* Grid: `log_step_schedule(40000, 150)` / `(8000, 150)` / `(5000, 150)` from `grokverse.utils`; previous step = the schedule entry before the recorded crossing.
* Attention: `build_model(cfg)`, `load_state_dict(torch.load(run/"model_final.pt"))`, `model.attention_pattern(make_dataset(cfg)["all_x"])[:, :, 2, :]`; `.mean(0)` and `.std(0, unbiased=True)` over the input axis; `cfg` rebuilt from `run.json["config"]` restricted to `dataclasses.fields(Config)`.
* Progress measures: read `run/"progress_measures.json"` directly (no re-train was performed for this audit).
* Parameters: `get_config("nanda").n_params`, `get_config("nanda", arch="mlp").n_params`, `get_config("mlp_param_matched").n_params`; `sum(p.numel() for p in build_model(cfg).parameters())`.
* Masks: `build_protocol(name, 113, [18, 15, 1, 11, 13, 22, 56, 36])` for `name in IMPLEMENTED` (`grokverse.analysis.mask_protocols`) → `n_kept`, `n_removed`, `keeps_cross_frequency()`.
* Split pairing: `split_hash(make_dataset(get_config("nanda", train_frac=0.3, seed=s))["train_idx"])` for both `arch` values (`grokverse.data`).
* Tests: `python test_core.py` (exit code and `[PASS]`/`[FAIL]` lines).
