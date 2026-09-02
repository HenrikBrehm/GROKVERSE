# Parameter and capacity confounds

> **AI-drafted (Claude), 2026-09-02 — not yet human-reviewed.**
> Addresses master prompt §14 "Parameter and capacity confounds" (the master prompt is not stored in this
> repository; it is cited through `docs/dev/RUN_FORMAT_V2.md`) and `docs/RESEARCH_SPEC.md` §3.6, WP-5
> ("Parameter control" / "Input-parametrization control") and acceptance criterion 8.
> Every number below was computed by `training/grokverse/analysis/capacity_report.py` and is stored at full
> precision in `docs/data/capacity_report.json`; the tables are the script's own rounded rendering
> (`python -m grokverse.analysis.capacity_report --tables`). Text outside tables and quote blocks is the
> drafter's reasoning and is labelled as such.

## 0. Provenance

| item | value |
|---|---|
| generator | `training/grokverse/analysis/capacity_report.py`, run as `python -m grokverse.analysis.capacity_report` with cwd `training/` |
| data file | `docs/data/capacity_report.json` (generated 2026-09-02T16:34:29 UTC) |
| environment | venv Python 3.12.10, torch 2.12.1+cpu, numpy 2.4.6, `torch.set_num_threads(1)`, Windows 11, CPU only |
| git | HEAD `84d449d` on branch `arch-study`; working tree **dirty** (uncommitted: `config.py`, `data.py`, `models/__init__.py`, `train.py`, new `models/mlp_twohot.py`, `analysis/capacity_report.py`, …). The two-hot model is buildable only through the uncommitted `models/__init__.py` |
| model definitions | `training/grokverse/models/transformer.py`, `mlp.py`, `mlp_twohot.py` (tensor shapes and `_init_weights` standard deviations) |
| count formula | `Config.n_params` in `training/grokverse/config.py`; asserted against the built modules in `test_core.py` ("n_params matches the built …") |
| optimizer | `training/grokverse/train.py`: `torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))`; `Config` defaults `lr=1e-3`, `weight_decay=1.0`, `beta1=0.9`, `beta2=0.98` |
| run artifacts | `training/runs/<run_id>/{run.json, model_final.pt, embeddings.npy}` for the 16 legacy runs listed in `docs/BASELINE.md` |
| excluded | three `*_smoketest` directories (`study="smoketest"`, 300 steps; `m2h_…`, `mlp_…`, `txf_…`) created on 2026-09-02 by the run-format-v2 work were present at generation time and are **not** among the 16; they are listed under `excluded_runs` in the JSON |
| sources | fetched 2026-09-02: Omnigrok arXiv:2210.01117 (abstract page and the ar5iv HTML rendering — no `arxiv.org/html` version exists for this 2022 paper and the PDF could not be text-extracted), PyTorch `torch.optim.AdamW` documentation. Verbatim quotes in §9 |

## 1. What is compared

| name in JSON | class | how built | fixed dims |
|---|---|---|---|
| `transformer_d512` | `OneLayerTransformer` | `get_config("nanda", arch="transformer")` | `p=113`, vocab 114 (`p` numbers + `=`), `d_model=128`, 4 heads × `d_head=32`, `d_mlp=512`, `n_ctx=3`, no LayerNorm, no biases |
| `mlp_d512` | `TwoLayerMLP` (shared embedding) | `get_config("nanda", arch="mlp")` | `d_model=128`, `d_mlp=512`, `W_E` has `p=113` rows (no `=` token) |
| `mlp_param_matched_d572` | `TwoLayerMLP` | preset `mlp_param_matched` | as above with `d_mlp=572` |
| `mlp_twohot_d512` | `TwoHotMLP` (no embedding) | preset `arch25k_twohot` | input `concat(onehot(a), onehot(b)) ∈ R^226`, `d_mlp=512` |

The two-hot model is not part of the master-prompt §14 count comparison; it is included because it is the
input-parametrization control (§7) and its count is needed to interpret that control.

## 2. Parameter counts

### T1 — parameter counts

| architecture | preset / override | total (built) | trainable | `Config.n_params` formula | vs transformer |
|---|---|---:|---:|---:|---:|
| transformer_d512 | `nanda`, `arch=transformer` | 226,176 | 226,176 | 226,176 | +0 (+0.000 %) |
| mlp_d512 | `nanda`, `arch=mlp` | 204,017 | 204,017 | 204,017 | -22,159 (-9.797 %) |
| mlp_param_matched_d572 | `mlp_param_matched` (`arch=mlp`, `d_mlp=572`) | 226,217 | 226,217 | 226,217 | +41 (+0.018 %) |
| mlp_twohot_d512 | `arch25k_twohot` (`arch=mlp_twohot`) | 174,193 | 174,193 | 174,193 | -51,983 (-22.983 %) |

"total (built)" is `sum(p.numel() for p in model.parameters())`, "trainable" counts only `requires_grad`
tensors. All three columns agree for every model: nothing is frozen, and the formula in `Config.n_params`
tracks the real modules.

### T2-transformer_d512 — per-module counts and analytic initial norm

| parameter | shape | numel | init std (`_init_weights`) | expected ‖·‖₂ at init = √numel·std |
|---|---|---:|---:|---:|
| `W_E` | [114, 128] | 14,592 | 0.088388 | 10.6771 |
| `W_pos` | [3, 128] | 384 | 0.088388 | 1.7321 |
| `W_Q` | [4, 128, 32] | 16,384 | 0.088388 | 11.3137 |
| `W_K` | [4, 128, 32] | 16,384 | 0.088388 | 11.3137 |
| `W_V` | [4, 128, 32] | 16,384 | 0.088388 | 11.3137 |
| `W_O` | [4, 32, 128] | 16,384 | 0.088388 | 11.3137 |
| `W_in` | [128, 512] | 65,536 | 0.088388 | 22.6274 |
| `W_out` | [512, 128] | 65,536 | 0.044194 | 11.3137 |
| `W_U` | [128, 114] | 14,592 | 0.088388 | 10.6771 |
| **total** | | **226,176** | | **37.1887** |

### T2-mlp_d512 — per-module counts and analytic initial norm

| parameter | shape | numel | init std (`_init_weights`) | expected ‖·‖₂ at init = √numel·std |
|---|---|---:|---:|---:|
| `W_E` | [113, 128] | 14,464 | 0.088388 | 10.6301 |
| `W_in` | [256, 512] | 131,072 | 0.062500 | 22.6274 |
| `W_out` | [512, 113] | 57,856 | 0.044194 | 10.6301 |
| `b_in` | [512] | 512 | 0.000000 | 0.0000 |
| `b_out` | [113] | 113 | 0.000000 | 0.0000 |
| **total** | | **204,017** | | **27.1662** |

### T2-mlp_param_matched_d572 — per-module counts and analytic initial norm

| parameter | shape | numel | init std (`_init_weights`) | expected ‖·‖₂ at init = √numel·std |
|---|---|---:|---:|---:|
| `W_E` | [113, 128] | 14,464 | 0.088388 | 10.6301 |
| `W_in` | [256, 572] | 146,432 | 0.062500 | 23.9165 |
| `W_out` | [572, 113] | 64,636 | 0.041812 | 10.6301 |
| `b_in` | [572] | 572 | 0.000000 | 0.0000 |
| `b_out` | [113] | 113 | 0.000000 | 0.0000 |
| **total** | | **226,217** | | **28.2489** |

### T2-mlp_twohot_d512 — per-module counts and analytic initial norm

| parameter | shape | numel | init std (`_init_weights`) | expected ‖·‖₂ at init = √numel·std |
|---|---|---:|---:|---:|
| `W_in` | [226, 512] | 115,712 | 0.066519 | 22.6274 |
| `W_out` | [512, 113] | 57,856 | 0.044194 | 10.6301 |
| `b_in` | [512] | 512 | 0.000000 | 0.0000 |
| `b_out` | [113] | 113 | 0.000000 | 0.0000 |
| **total** | | **174,193** | | **25.0000** |

Init std values are `init_scale / sqrt(fan)` with `init_scale = 1.0`, transcribed from each model's
`_init_weights`: `1/√128 = 0.088388` (`d_model`), `1/√256 = 0.0625` (`2·d_model`), `1/√226 = 0.066519`
(`2p`), `1/√512 = 0.044194`, `1/√572 = 0.041812` (`d_mlp`). Biases start at exactly zero.

**Where the parameters sit** (sums of T2 rows; drafter's arithmetic on the table):

| block | transformer | MLP `d_mlp=512` |
|---|---:|---:|
| embedding / unembedding / positions (`W_E`, `W_U`, `W_pos`) | 29,568 | 14,464 (`W_E` only) |
| attention (`W_Q`, `W_K`, `W_V`, `W_O`) | 65,536 | – |
| MLP / hidden layer (`W_in`, `W_out`, biases) | 131,072 | 189,553 |
| total | 226,176 | 204,017 |

So "the same `d_mlp=512`" does not mean the same hidden layer: the MLP's `W_in` reads a `2·d_model = 256`-wide
concatenation and its `W_out` writes `p = 113` logits directly, giving a hidden layer 45 % larger than the
transformer's MLP sub-layer, while the transformer spends 65,536 parameters on attention that the MLP does
not have. The −9.8 % total difference is the net of these two opposite effects.

## 3. Initial L2 norms

### T3 — measured initial L2 norms after `set_seed(seed)` + `build_model(cfg)` (threads=1)

**transformer_d512** (expected total 37.1887)

| seed | `W_E` | `W_pos` | `W_Q` | `W_K` | `W_V` | `W_O` | `W_in` | `W_out` | `W_U` | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.7445 | 1.7205 | 11.2621 | 11.2999 | 11.2644 | 11.3616 | 22.6308 | 11.3277 | 10.7209 | **37.2063** |
| 1 | 10.7230 | 1.7140 | 11.3648 | 11.3381 | 11.3532 | 11.2660 | 22.6637 | 11.3554 | 10.6023 | **37.2350** |
| 2 | 10.6351 | 1.7019 | 11.2960 | 11.3016 | 11.1921 | 11.3018 | 22.7045 | 11.2998 | 10.7003 | **37.1752** |

**mlp_d512** (expected total 27.1662)

| seed | `W_E` | `W_in` | `W_out` | `b_in` | `b_out` | **total** |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 10.6955 | 22.6129 | 10.6403 | 0.0000 | 0.0000 | **27.1837** |
| 1 | 10.6724 | 22.6604 | 10.6758 | 0.0000 | 0.0000 | **27.2280** |
| 2 | 10.5809 | 22.6297 | 10.6097 | 0.0000 | 0.0000 | **27.1408** |

**mlp_param_matched_d572** (expected total 28.2489)

| seed | `W_E` | `W_in` | `W_out` | `b_in` | `b_out` | **total** |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 10.6955 | 23.9091 | 10.6447 | 0.0000 | 0.0000 | **28.2727** |
| 1 | 10.6724 | 23.9595 | 10.6435 | 0.0000 | 0.0000 | **28.3062** |
| 2 | 10.5809 | 23.9088 | 10.6276 | 0.0000 | 0.0000 | **28.2229** |

**mlp_twohot_d512** (expected total 25.0000)

| seed | `W_in` | `W_out` | `b_in` | `b_out` | **total** |
|---|---:|---:|---:|---:|---:|
| 0 | 22.6537 | 10.6326 | 0.0000 | 0.0000 | **25.0249** |
| 1 | 22.6752 | 10.6361 | 0.0000 | 0.0000 | **25.0458** |
| 2 | 22.5994 | 10.6234 | 0.0000 | 0.0000 | **24.9718** |

Observations (drafter's reading of T2/T3):

* Every measured total is within 0.25 % of its analytic expectation (largest deviation +0.228 %, MLP seed 1),
  so the initial norm is a deterministic function of the shapes and init stds, not of the seed.
* **The transformer starts with a 1.37× larger weight norm than the MLP** (expected 37.19 vs 27.17; the ratio
  of expectations is 1.3689). Matching the parameter count (`d_mlp=572`) raises the MLP's initial norm by
  only 4.0 % (to 28.25); it stays 24 % below the transformer's.
* For a given seed the two MLPs share the identical `W_E` draw (it is the first tensor sampled after
  `set_seed`), so their initial norms differ only through `W_in`/`W_out`.
* The `mul` runs and the `add` runs share the initial weights for equal seeds (task does not enter the model
  build), which is why `W_E` seed-0 norms coincide across all seed-0 transformer runs in T4/T5.

**Init-code consistency check.** The 16 legacy runs were produced at six different commits (`1a35a9c`,
`a22c5d5`, `ad5f594`, `4ccac23`, `ead5166`, `26024f0`). For every run, slice 0 of the stored
`embeddings.npy` (the `W_E` snapshot at step 0) was compared element-wise with the `W_E` produced by the
current code for the same seed: the maximum absolute difference is **0.0 in all 16 runs** (T4, last column).
The `W_E` initialization therefore has not changed across those commits and the "initial norm" columns for
`W_E` are exact for every run. The other modules are not snapshotted in any legacy artifact, so their initial
norms are current-code values assumed, not verified, to equal the ones the run actually started from
(open issue §10).

## 4. Final L2 norms of the 16 existing runs

All norms are read from `model_final.pt`. Reminder from `docs/RESEARCH_SPEC.md` §3.9 and `docs/BASELINE.md`:
for the six un-accelerated `frac0.3` runs the "final" state **is the generalization crossing** (the run was
early-stopped at test acc ≥ 0.95), so these are at-crossing norms, not converged norms. The Grokfast runs
also stopped shortly after their crossing. Run 7 never generalized within its 4,000-step budget.

### T4 — the 16 existing runs: total L2 norm, initial vs final

| # | run_id | run commit | threads | steps budget | last logged step | n_params (state_dict) | ‖θ₀‖₂ | ‖θ_final‖₂ | final/init | (1−lr·wd)^last | init check max\|Δ\| |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `mlp_add_p113_wd1.0_frac0.3_seed0` | `1a35a9c` | 6 | 40,000 | 9,646 | 204,017 | 27.1837 | 78.8712 | 2.9014 | 6.44e-05 | 0.0 |
| 2 | `mlp_add_p113_wd1.0_frac0.3_seed1` | `1a35a9c` | 6 | 40,000 | 10,357 | 204,017 | 27.2280 | 79.8140 | 2.9313 | 3.16e-05 | 0.0 |
| 3 | `mlp_add_p113_wd1.0_frac0.3_seed2` | `1a35a9c` | 6 | 40,000 | 9,646 | 204,017 | 27.1408 | 78.9207 | 2.9078 | 6.44e-05 | 0.0 |
| 4 | `mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0` | `a22c5d5` | – | 5,000 | 2,246 | 204,017 | 27.1837 | 92.9474 | 3.4192 | 1.06e-01 | 0.0 |
| 5 | `mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1` | `a22c5d5` | – | 5,000 | 2,246 | 204,017 | 27.2280 | 92.7537 | 3.4066 | 1.06e-01 | 0.0 |
| 6 | `mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2` | `a22c5d5` | – | 5,000 | 2,378 | 204,017 | 27.1408 | 91.6883 | 3.3782 | 9.26e-02 | 0.0 |
| 7 | `txf_add_p113_wd1.0_frac0.3_gf2.0_seed0` | `ad5f594` | – | 4,000 | 4,000 | 226,176 | 37.2063 | 58.3579 | 1.5685 | 1.83e-02 | 0.0 |
| 8 | `txf_add_p113_wd1.0_frac0.3_seed0` | `4ccac23` | – | 40,000 | 8,367 | 226,176 | 37.2063 | 47.7030 | 1.2821 | 2.31e-04 | 0.0 |
| 9 | `txf_add_p113_wd1.0_frac0.3_seed1` | `ead5166` | – | 40,000 | 6,295 | 226,176 | 37.2350 | 50.5945 | 1.3588 | 1.84e-03 | 0.0 |
| 10 | `txf_add_p113_wd1.0_frac0.3_seed2` | `ead5166` | – | 40,000 | 8,367 | 226,176 | 37.1752 | 45.7743 | 1.2313 | 2.31e-04 | 0.0 |
| 11 | `txf_add_p113_wd1.0_frac0.5_gf2.0_seed0` | `ad5f594` | – | 8,000 | 717 | 226,176 | 37.2063 | 52.3607 | 1.4073 | 4.88e-01 | 0.0 |
| 12 | `txf_add_p113_wd1.0_frac0.5_gf2.0_seed1` | `ad5f594` | – | 8,000 | 912 | 226,176 | 37.2350 | 52.5024 | 1.4100 | 4.02e-01 | 0.0 |
| 13 | `txf_add_p113_wd1.0_frac0.5_gf2.0_seed2` | `a22c5d5` | – | 5,000 | 953 | 226,176 | 37.1752 | 48.9947 | 1.3179 | 3.85e-01 | 0.0 |
| 14 | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed0` | `26024f0` | 6 | 8,000 | 969 | 226,176 | 37.2063 | 51.3828 | 1.3810 | 3.79e-01 | 0.0 |
| 15 | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed1` | `26024f0` | 6 | 8,000 | 969 | 226,176 | 37.2350 | 51.5281 | 1.3839 | 3.79e-01 | 0.0 |
| 16 | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed2` | `26024f0` | 6 | 8,000 | 1,093 | 226,176 | 37.1752 | 51.3179 | 1.3804 | 3.35e-01 | 0.0 |

Columns: "threads" is `torch_num_threads` from `run.json` ("–" = not recorded, the field did not exist yet);
"‖θ₀‖₂" is the current-code initial norm for that seed (T3); "(1−lr·wd)^last" is the factor pure weight decay
alone would have applied over the logged steps (§5); the last column is the `W_E` init check of §3.

### T5-transformer — per-module final L2 norm (final/init ratio in parentheses; '–' = init norm 0)

| # | run_id | `W_E` | `W_pos` | `W_Q` | `W_K` | `W_V` | `W_O` | `W_in` | `W_out` | `W_U` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | `txf_add_p113_wd1.0_frac0.3_gf2.0_seed0` | 23.43 (2.18) | 2.13 (1.24) | 20.73 (1.84) | 4.47 (0.40) | 27.50 (2.44) | 24.10 (2.12) | 26.98 (1.19) | 16.73 (1.48) | 7.57 (0.71) |
| 8 | `txf_add_p113_wd1.0_frac0.3_seed0` | 18.22 (1.70) | 1.97 (1.14) | 14.88 (1.32) | 2.53 (0.22) | 24.44 (2.17) | 21.72 (1.91) | 21.29 (0.94) | 12.79 (1.13) | 5.12 (0.48) |
| 9 | `txf_add_p113_wd1.0_frac0.3_seed1` | 21.10 (1.97) | 1.77 (1.03) | 15.35 (1.35) | 3.00 (0.26) | 25.17 (2.22) | 21.54 (1.91) | 23.56 (1.04) | 12.86 (1.13) | 7.06 (0.67) |
| 10 | `txf_add_p113_wd1.0_frac0.3_seed2` | 18.38 (1.73) | 1.91 (1.12) | 12.43 (1.10) | 2.32 (0.21) | 23.25 (2.08) | 20.11 (1.78) | 21.43 (0.94) | 12.45 (1.10) | 5.96 (0.56) |
| 11 | `txf_add_p113_wd1.0_frac0.5_gf2.0_seed0` | 16.70 (1.55) | 1.34 (0.78) | 13.58 (1.21) | 9.19 (0.81) | 17.48 (1.55) | 16.96 (1.49) | 28.09 (1.24) | 24.09 (2.13) | 15.13 (1.41) |
| 12 | `txf_add_p113_wd1.0_frac0.5_gf2.0_seed1` | 17.67 (1.65) | 1.48 (0.86) | 12.67 (1.12) | 7.68 (0.68) | 18.56 (1.63) | 18.23 (1.62) | 29.55 (1.30) | 22.63 (1.99) | 12.66 (1.19) |
| 13 | `txf_add_p113_wd1.0_frac0.5_gf2.0_seed2` | 16.89 (1.59) | 1.22 (0.72) | 12.11 (1.07) | 6.94 (0.61) | 17.74 (1.59) | 17.03 (1.51) | 25.75 (1.13) | 22.19 (1.96) | 12.60 (1.18) |
| 14 | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed0` | 16.62 (1.55) | 1.26 (0.73) | 13.66 (1.21) | 8.43 (0.75) | 17.19 (1.53) | 16.92 (1.49) | 26.52 (1.17) | 24.23 (2.14) | 15.26 (1.42) |
| 15 | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed1` | 16.78 (1.56) | 1.26 (0.74) | 14.50 (1.28) | 8.45 (0.75) | 17.52 (1.54) | 16.88 (1.50) | 26.65 (1.18) | 23.48 (2.07) | 15.40 (1.45) |
| 16 | `txf_mul_p113_wd1.0_frac0.5_gf2.0_seed2` | 16.46 (1.55) | 1.29 (0.76) | 14.52 (1.29) | 8.47 (0.75) | 17.24 (1.54) | 16.79 (1.49) | 26.58 (1.17) | 23.78 (2.10) | 15.08 (1.41) |

### T5-mlp — per-module final L2 norm (final/init ratio in parentheses; '–' = init norm 0)

| # | run_id | `W_E` | `W_in` | `W_out` | `b_in` | `b_out` |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `mlp_add_p113_wd1.0_frac0.3_seed0` | 42.47 (3.97) | 48.11 (2.13) | 45.76 (4.30) | 2.88 (–) | 0.53 (–) |
| 2 | `mlp_add_p113_wd1.0_frac0.3_seed1` | 43.07 (4.04) | 48.70 (2.15) | 46.20 (4.33) | 2.98 (–) | 0.61 (–) |
| 3 | `mlp_add_p113_wd1.0_frac0.3_seed2` | 42.62 (4.03) | 48.15 (2.13) | 45.65 (4.30) | 3.03 (–) | 0.57 (–) |
| 4 | `mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0` | 43.78 (4.09) | 61.40 (2.72) | 54.25 (5.10) | 2.86 (–) | 0.94 (–) |
| 5 | `mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1` | 43.67 (4.09) | 61.29 (2.70) | 54.14 (5.07) | 2.81 (–) | 0.91 (–) |
| 6 | `mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2` | 44.13 (4.17) | 59.81 (2.64) | 53.59 (5.05) | 3.05 (–) | 0.96 (–) |

### Group summary (drafter's arithmetic on T4; `n` = runs per cell)

| cell | runs | n | ‖θ_final‖₂ mean (min–max) | final/init mean (min–max) |
|---|---|---:|---|---|
| MLP, `add`, `frac0.3`, no Grokfast (at crossing) | 1–3 | 3 | 79.20 (78.87–79.81) | 2.914 (2.901–2.931) |
| transformer, `add`, `frac0.3`, no Grokfast (at crossing) | 8–10 | 3 | 48.02 (45.77–50.59) | 1.291 (1.231–1.359) |
| MLP, `add`, `frac0.5`, Grokfast | 4–6 | 3 | 92.46 (91.69–92.95) | 3.401 (3.378–3.419) |
| transformer, `add`, `frac0.5`, Grokfast | 11–13 | 3 | 51.29 (48.99–52.50) | 1.378 (1.318–1.410) |
| transformer, `mul`, `frac0.5`, Grokfast | 14–16 | 3 | 51.41 (51.32–51.53) | 1.382 (1.380–1.384) |
| transformer, `add`, `frac0.3`, Grokfast, **never generalized** | 7 | 1 | 58.36 | 1.568 |

What the stored runs already show (descriptive only; no significance claim is made on n = 3, and the
`mul` runs are reserved for the human author's own evaluation per `docs/BASELINE.md` — they are listed,
not interpreted):

1. **The norm ordering flips between initialization and the crossing.** At init the transformer's norm is
   1.37× the MLP's; at the generalization crossing in the paired un-accelerated setting the MLP's norm is
   1.65× the transformer's (79.20 vs 48.02). Relative to its own start the MLP grew ×2.90–2.93, the
   transformer ×1.23–1.36.
2. **The two architectures spend the decay budget on different modules.** In the un-accelerated transformer
   runs `W_K` shrinks to 0.21–0.26× its initial norm and `W_U` to 0.48–0.67×, `W_in` stays at 0.94–1.04×,
   while `W_V`, `W_O` (≈1.8–2.2×) and `W_E` (1.70–1.97×) grow. In the MLP every matrix grows:
   `W_E` ×3.97–4.04, `W_in` ×2.13–2.15, `W_out` ×4.30–4.33; the biases grow from 0 to ≈3 (`b_in`) and
   ≈0.6 (`b_out`).
3. **Pure decay would have shrunk everything by 10³–10⁴; instead the norms grew**, so at every stored end
   point the gradient term, not the decay term, sets the norm — the final norm is a *balance* between the
   two, and that balance is architecture-specific (§6).
4. The Grokfast MLP runs end with a higher norm (≈92.5) than the un-accelerated ones (≈79.2); the
   Grokfast transformer runs end at ≈51 vs ≈48. Whether this is Grokfast, `frac=0.5`, or the shorter
   run is not separable in these data (`docs/RESEARCH_SPEC.md` §3.5).

## 5. Effective per-step weight-decay shrink factor of AdamW

PyTorch's AdamW applies decoupled weight decay as a separate multiplicative step on the parameters before
the Adam update. From the `torch.optim.AdamW` documentation (fetched 2026-09-02; algorithm box):

> "θt←θt−1−γλθt−1"

> "Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance."

> "weight_decay (float, optional) – weight decay coefficient (default: 1e-2)"

With `γ = lr = 1e-3` and `λ = weight_decay = 1.0` (this project's `Config` defaults; the PyTorch default
would be `1e-2`):

| quantity | value | provenance |
|---|---|---|
| `lr · weight_decay` | 1e-3 × 1.0 = **1e-3** | `Config.lr`, `Config.weight_decay` |
| per-step multiplicative factor `1 − lr·wd` | **0.999** | AdamW decay step above |
| applied to | every tensor in `model.parameters()` — one param group, **including the MLP's `b_in`, `b_out`** (the transformer has no biases) | `train.py` constructs AdamW with `model.parameters()` and no per-group settings |

Pure-decay factor `0.999^N` over `N` steps (what the decay step alone would do to every weight if the
gradient contribution were zero; `N` = the stored runs' last logged steps plus the planned 25k budget):

| N (steps) | 717 | 953 | 2,246 | 4,000 | 6,295 | 8,367 | 9,646 | 10,357 | 25,000 |
|---|---|---|---|---|---|---|---|---|---|
| `0.999^N` | 4.88e-1 | 3.85e-1 | 1.06e-1 | 1.83e-2 | 1.84e-3 | 2.31e-4 | 6.44e-5 | 3.16e-5 | 1.37e-11 |

Since `ln(1/0.999) ≈ 1.0005e-3`, the decay-only e-folding time is ≈1,000 steps for both architectures. The
identical per-step factor is exactly what makes the comparison look "controlled" — and §6 explains why it
is not.

## 6. Why the identical decay coefficient is a confound

**In two sentences (drafter's argument, built on the Omnigrok quotes below):**

1. In the Omnigrok picture grokking time is the time weight decay needs to carry the weight norm from its
   initial value w₀ to the generalizing "Goldilocks" shell w_c — "t≈ln(w₀/w_c)/γ" — and both w₀ (37.2 for
   the transformer vs 27.2 for the MLP, fixed by the parameter count and per-module init std, T2/T3) and
   w_c (a property of each architecture's loss landscape, unknown here) differ between the two models, so
   the same γ = 1.0 prescribes different generalization delays even if both models implemented the identical
   algorithm.
2. In addition, the decoupled decay is a *relative* shrink (×0.999 per step on every parameter) that competes
   with gradient-driven growth whose magnitude depends on how many parameters carry gradient and how they
   are wired — and the stored runs show that this tug-of-war settles in very different places (at the crossing
   the MLP's norm has grown ×2.9, the transformer's ×1.3; T4) — so "`weight_decay = 1.0`" names the same
   coefficient in both configs but not the same regularization pressure.

**Source support** (Omnigrok, arXiv:2210.01117v2, fetched 2026-09-02 via the arXiv abstract page and the
ar5iv HTML rendering; section numbers as rendered there):

> §2 "The LU mechanism for grokking": "There is a spherical shell in the weight space (the "Goldilocks" zone),
> where generalization is better than outside this zone. We illustrate the Goldilocks zone as the green area
> with average radius w_c in Figure 1(a)"

> §2: "Fortunately, there are usually explicit and/or implicit regularizations that can drive the weight vector
> towards the Goldilocks zone w≈w_c. When the regularization magnitude is non-zero but small, the radial
> motion can be (arbitrarily) slow. If weight decay is the only source of regularization, and training loss
> is negligible after overfitting, then weight decay γ causes w(t)≈exp(−γt)w₀, when w₀>w_c, so it takes
> time t≈ln(w₀/w_c)/γ∝γ⁻¹ to generalize."

> §2: "our model is initialized by multiplying a factor α≡w/w₀ to the standard initialization"

> §3: "large initialization (α=2.0) can demonstrate no generalization (no reg), grokking (small reg) and
> fast generalization (large reg)."

> §5.1 (Figure 7(b)): "constraining optimization to hold model weight norm constant over training brings
> train accuracy and test accuracy learning curves together, almost eliminating grokking"

What the source does **not** say: no sentence relating the *number of parameters*, width or model size to
grokking was found in the fetched text — **[NOT FOUND IN SOURCE]**. The step from "w₀ differs" to "the
parameter count is (part of) the reason w₀ differs" is the drafter's, and rests only on T2: w₀² is the sum
over modules of `numel · std²`, so it is set jointly by the counts and the init stds. Note also that
Omnigrok's `w(t)≈exp(−γt)w₀` is the decay-only limit; T4 shows the stored runs are far from that limit (norms
grew), so the formula is used here as the qualitative mechanism, not as a quantitative prediction.

Consequence for the study: a grokking-time or structure difference between the two architectures can be
produced by (a) a different w₀, (b) a different w_c, (c) a different gradient/decay balance, or (d) a genuine
mechanistic difference, and the existing 16 runs cannot separate these. `docs/RESEARCH_SPEC.md` WP-5 already
schedules the parameter-matched control (b/c partly); §8 and §10 note that neither existing control equalizes
w₀.

## 7. The input-parametrization confound

The three models do not merely differ in "architecture"; they present the pair `(a, b)` to their first
nonlinearity in three different parametrizations. The shared-embedding MLP (`models/mlp.py`) reads both
operands through one learned table and concatenates: `x = concat(W_E[a], W_E[b]) ∈ R^256`, then
`h = ReLU(x·W_in + b_in)` with `W_in ∈ R^{256×512}`, so the effective per-neuron operand curves are
`u_a = W_E·W_in[:128]` and `u_b = W_E·W_in[128:]` (`docs/RESEARCH_SPEC.md` §3.3) — both curves are linear
images of the *same* table, decayed once but used twice. The two-hot MLP (`models/mlp_twohot.py`) has no
table at all: `x = concat(onehot(a), onehot(b)) ∈ R^226`, `W_in ∈ R^{226×512}`, and the operand curves are
simply `u_a = W_in[:113]`, `u_b = W_in[113:]`, each an unconstrained, independently decayed block. The
transformer reads both operands (and the `=` token) through `W_E ∈ R^{114×128}` plus a position vector, and
lets them interact bilinearly through attention (`W_Q`, `W_K`) *before* its ReLU, whereas in both MLPs the
operands are exactly additive until the ReLU (§3.3 of the spec). Because `d_model = 128 ≥ p = 113`, the shared
table is not an expressivity restriction — `W_E` can have full row rank 113, so `u_a = W_E·W_in[:128]` can
realise any 113×512 matrix (drafter's derivation from the shapes) — but it is a different parametrization
with different gradient dynamics, different weight-decay pressure, a different parameter count (204,017 vs
174,193, i.e. the two-hot model has 14.6 % fewer parameters than the shared-embedding MLP and 23.0 % fewer
than the transformer, T1), and a different *measured object*: the legacy Fourier/PCA statistics are computed
on `W_E`, which the two-hot model does not have (`docs/dev/RUN_FORMAT_V2.md` §2 logs `W_in[:p]` in its place
and records `embedding_object`). The two-hot control therefore isolates one thing only — whether the
structure found in the shared-embedding MLP survives removing the shared table — and it is silent about the
transformer's attention-mediated parametrization; the attribution of the two-hot setup to arXiv:2603.23784
is still **[UNVERIFIED]** in the spec (H2 provenance note) and is not repeated here as a claim.

## 8. "Same hyperparameter dimensions" vs "similar parameter count"

`docs/RESEARCH_SPEC.md` WP-5 requires both to be reported as *two distinct comparisons*. They control
different things:

| comparison | transformer | MLP | what is equal | what is not |
|---|---|---|---|---|
| **A — same hyperparameter dimensions** (legacy runs, `d_mlp=512`) | 226,176 params; w₀ ≈ 37.19 | 204,017 params (−9.8 %); w₀ ≈ 27.17 (−27 %) | `d_model=128`, `d_mlp=512`, `p`, optimizer, `lr`, `wd`, seeds/splits | parameter count, initial norm, hidden-layer size (131,072 vs 189,553), presence of attention (65,536) and biases (625) |
| **B — matched parameter count** (`mlp_param_matched`, `d_mlp=572`) | 226,176; w₀ ≈ 37.19 | 226,217 (+0.018 %); w₀ ≈ 28.25 (−24 %) | total count (to 41 parameters), everything in A except `d_mlp` | hidden width (572 vs 512 → `W_out` init std 0.0418 vs 0.0442), initial norm (only +4.0 % vs A), same structural differences as A |

Reading (drafter's):

* Comparison A answers "same recipe, different architecture" — it is what every legacy number was measured
  under. Comparison B answers "same budget of parameters, different architecture". A result that holds under
  both is robust to the *count* confound; it is **not** thereby robust to the *initial-norm* confound of §6,
  because B moves w₀ by only 4 % of a 27 % gap.
* "Same `d_mlp`" is itself ambiguous (§2): the transformer's `d_mlp` sizes a 128→512→128 sub-layer, the MLP's
  a 256→512→113 layer. Matching the count with `d_mlp=572` therefore makes the MLP's hidden layer *wider*
  than the transformer's, not equal to it. Neither A nor B is "the fair comparison"; they bracket it.
* An initial-norm-matched control is not defined anywhere in the repo. Because every init std is
  proportional to `init_scale`, `init_scale = 37.1887 / 27.1662 = 1.369` would give the `d_mlp=512` MLP the
  transformer's expected w₀ (drafter's arithmetic on T2; this is Omnigrok's α rescaling) — recorded as an
  option for the human authors in §10, not as a plan.
* For the two-hot control no parameter-matched variant exists either; its count is `d_mlp·(3p+1) + p =
  340·d_mlp + 113`, so `d_mlp = 665` would give 226,213 (+0.016 % vs the transformer) — drafter's arithmetic
  from `Config.n_params`, verified numerically in §10.

## 9. Sources (verbatim, fetched 2026-09-02)

**Omnigrok: Grokking Beyond Algorithmic Data** — Ziming Liu, Eric J. Michaud, Max Tegmark, arXiv:2210.01117,
v2 (last revised 23 Mar 2023; submitted 3 Oct 2022). Fetched from `https://arxiv.org/abs/2210.01117` (title,
authors, abstract, version) and `https://ar5iv.labs.arxiv.org/html/2210.01117` (body text; section numbers
are ar5iv's rendering). `https://arxiv.org/html/2210.01117` does not exist for this paper and
`https://arxiv.org/pdf/2210.01117` returned a binary stream that could not be text-extracted in this
environment.

> Abstract: "Grokking, the unusual phenomenon for algorithmic datasets where generalization happens long after
> overfitting the training data, has remained elusive. We aim to understand grokking by analyzing the loss
> landscapes of neural networks, identifying the mismatch between training and test losses as the cause for
> grokking. We refer to this as the 'LU mechanism' because training and test losses (against model weight
> norm) typically resemble 'L' and 'U', respectively. This simple mechanism can nicely explain many aspects of
> grokking: data size dependence, weight decay dependence, the emergence of representations, etc. Guided by
> the intuitive picture, we are able to induce grokking on tasks involving images, language and molecules. In
> the reverse direction, we are able to eliminate grokking for algorithmic datasets. We attribute the dramatic
> nature of grokking for algorithmic datasets to representation learning."

> §2 (Eq. 1 context): "Letting w denote the weights of a model, any function f(w) (e.g, train/test
> loss/accuracy) depends on both the weight norm w≡‖w‖₂ and the angular direction ŵ≡w/w. Similar to [Fort and
> Scherlis 2019], we define a reduced function f̃(w) by minimizing training loss l_train(w) over angular
> directions"

> §2: "In practice, we perform the constrained minimization by rescaling the model weights back to their
> original norm after each unconstrained optimization step."

> §5.1 setup (the modular-addition model of that paper, quoted so it is not confused with ours): "The MLP has
> two hidden layers, with neurons 1-200-200-30 in each layer and ReLU activations." and "Each input digit
> 0≤i≤p−1 is embedded as a vector 𝐄i"

(The §2 Goldilocks / weight-decay-time / α quotes and the §3 and §5.1 quotes are given in §6.)

**PyTorch, `torch.optim.AdamW`** — `https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html`
(the `stable` URL redirects there).

> "θt←θt−1−γλθt−1"

> "Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance."

> "weight_decay (float, optional) – weight decay coefficient (default: 1e-2)"

> "For further details regarding the algorithm we refer to Decoupled Weight Decay Regularization"

**Repository sources**: `docs/RESEARCH_SPEC.md` §3.3, §3.5, §3.6, §3.9, H2, WP-5, §12; `docs/BASELINE.md`;
`docs/dev/RUN_FORMAT_V2.md` §2, §4; `training/grokverse/config.py`; `training/grokverse/models/*.py`;
`training/grokverse/train.py`; `training/test_core.py` (§3.6 checks).

## 10. Open issues and caveats

1. **At-crossing, not converged.** Every "final" norm of runs 1–3 and 8–10 is the norm at the early-stop
   crossing (`docs/RESEARCH_SPEC.md` §3.9). The fixed-budget `arch25k` runs (`docs/dev/RUN_FORMAT_V2.md`) will
   provide norms at the crossing *and* at 25,000 steps; this document should be regenerated then
   (`python -m grokverse.analysis.capacity_report`), and `manifest.json` already specifies
   `weight_norms_init`, `weight_norms_final` and `effective_weight_decay_per_step` for every new run.
2. **Only `W_E` init is verified across commits.** The other modules' initial norms are current-code values;
   nothing in the legacy artifacts can confirm them for the runs recorded at `1a35a9c` … `26024f0`. New runs
   store a step-0 checkpoint, which closes this gap.
3. **Heterogeneous legacy provenance.** Thread count (6 vs unrecorded) and git commit differ across the runs
   of one setting (`docs/BASELINE.md`); the norms in T4/T5 inherit that heterogeneity.
4. **Neither existing control equalizes the initial norm** (§8). Whether to add an `init_scale`-matched
   MLP (`init_scale ≈ 1.369`) is a human decision (`docs/RESEARCH_SPEC.md` §9); it would add one more cell
   to the WP-5 matrix.
5. **No parameter-matched two-hot variant** exists; `d_mlp = 665` would match the transformer to +0.016 %
   (§8). Whether the two-hot control needs one is likewise a human decision.
6. **`mul` runs** are listed for completeness only; per `docs/BASELINE.md` they are reserved for the human
   author's evaluation.
7. **Omnigrok's parameter-count silence.** The argument that the count is a confound is the drafter's
   inference (§6), not a statement of the source; the source's mechanism is stated in terms of weight norm
   only.
8. The generator and this document were produced while other uncommitted work (run format v2, smoke-test
   runs) was landing in the same tree; the `excluded_runs` list in the JSON and the dirty-file list in its
   `environment` block record what was present.
