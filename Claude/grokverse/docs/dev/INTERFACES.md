# Analysis-module interface contract

Status: specification (2026-09-02). Every analysis module of the architecture study is written against
this document so that modules written in parallel compose. Numbers produced by any of these modules are
**measurements**, never conclusions; conclusions live only in `docs/` after the decision tree is evaluated.

Companion documents: `docs/dev/RUN_FORMAT_V2.md` (run directories, checkpoints, manifest),
`docs/PREREGISTRATION.md` (hypotheses, thresholds, evidence gate — written before any full run).

## 0. Conventions shared by every module

* **Python**: `training/.venv` (3.12, torch 2.12.1+cpu, numpy 2.4.6, matplotlib 3.11). **No new
  dependencies** (no scipy/pandas/sklearn). All statistics are implemented with numpy.
* **Determinism**: every random control takes `seed: int` and uses `np.random.default_rng(seed)`; default
  `seed=0`; the seed is echoed in the output JSON.
* **Fourier basis**: `analysis.fourier.fourier_basis(p)` — row 0 constant, rows `1+2(k-1)`, `2+2(k-1)` =
  `cos_k, sin_k` normalized to unit norm (each has norm `sqrt(p/2)` before normalization). The 2D transform
  over `(a, b)` is `analysis.progress_measures.fwd2d/inv2d`. A frequency index is `k in 1..(p-1)//2`;
  `alias_frequency(x, p)` folds any integer into that range.
* **Phase convention**: a curve `y[n]` is written `A cos(2πkn/p − φ)`; `φ = atan2(coeff_sin, coeff_cos)`
  on the *normalized* basis rows, exactly as `analysis.mlp_mechanism.curve_spectra` does. Angles are
  wrapped to `(−π, π]`. The phase-addition prediction is `φ_out ≈ φ_a + φ_b (mod 2π)`.
* **Loading a model**: `analysis.common.load_model_at(run_dir, step=None) -> (cfg, model, state, meta)`.
  `step=None` → the legacy `model_final.pt`; an integer → `ckpt_step{step:06d}.pt` via
  `grokverse.checkpoints.load_checkpoint`. `meta` carries `run_id`, `step` (or `"final"`),
  `checkpoint_sha256` (or `null`), `arch`, `p`. `analysis.common.grid_logits(model, cfg) -> np.ndarray
  [p, p, p]` (a outer, b inner, class last; float64; `logits[:, :p]`) and
  `analysis.common.split_masks(cfg) -> (train_mask [p,p] bool, test_mask)` from `data.make_dataset`.
* **Output location**: `run_dir/analysis/<module>/<tag>.json` where `<tag>` is `step{N:06d}` or `final`
  for legacy runs; large arrays go next to it as `<tag>.npz` (float32 unless precision matters). A module
  never overwrites a file written by another module. Re-running overwrites its own file.
* **Output envelope** (every JSON):
  ```json
  {"module": "wave_fitting", "module_version": "1.0", "run_id": "...", "arch": "mlp", "p": 113,
   "step": 25000, "checkpoint_sha256": "...", "analysis_git_commit": "...", "created_utc": "...",
   "params": {every argument incl. seeds and thresholds}, "results": {...}}
  ```
* **Arrays vs summaries**: per-neuron / per-class arrays go to the `.npz`; the JSON holds summaries
  (median, mean, quantiles 5/25/75/95, min, max, n) and never a silently truncated list.
* **Random controls**: default 50 draws (`n_control=50`); report `control_mean`, `control_std`,
  `control_q05`, `control_q95`, `control_values` (in npz), and `z = (observed − control_mean)/control_std`.
* **No interpretation in code**: keys are named for what is measured (`fraction_neurons_best_aic_square`),
  not for a conclusion (`is_square_wave`). Threshold-dependent classifications always report the
  threshold in the key or in `params`.
* **Errors**: a module raises `ValueError` with a message naming the run and the missing artifact; it never
  falls back to a default run or a made-up value.
* **CLI**: every module has `python -m grokverse.analysis.<module> <run_dir> [--step N] [--seed S] ...`
  and prints the JSON summary. `--all-checkpoints` runs on every entry of `checkpoints.json`.
* **Tests**: `training/tests/test_<module>.py` with the `check(name, cond)` convention; synthetic inputs
  with known answers; a scrambled control that must NOT pass. `python tests/run_all.py` runs every test
  file and `test_core.py` and exits non-zero on any failure.

## 1. `analysis/common.py` — loading and shared helpers (small, written first)

```python
load_model_at(run_dir, step=None)          # see §0
grid_logits(model, cfg) -> np.ndarray      # [p, p, p] float64
split_masks(cfg) -> (train_mask, test_mask)
center_logits(L) -> L - L.mean(axis=-1, keepdims=True)
envelope(module, version, meta, params) -> dict   # the output envelope skeleton
write_result(run_dir, module, tag, payload, arrays: dict | None)
key_frequency_set(run_dir, step, rule)     # §4 — delegates to key_frequencies.select
```

## 2. `analysis/metrics.py` — structure metrics on 1-D curves (shared by both architectures)

All functions take `power [half, n]` (per-frequency power per column, from `curve_spectra`) unless noted.

| function | definition |
|---|---|
| `topk_concentration(power, k)` | Σ of the k largest / Σ all, per column |
| `spectral_entropy(power)` | `−Σ q log q / log(half)`, `q = power/Σpower`, in [0,1] |
| `participation_ratio(power)` | `(Σ q_k)² / Σ q_k²` on normalized power — number of "effective" frequencies |
| `inverse_participation_ratio(...)` | as defined in Doshi et al. 2310.13061 — implement exactly the object and formula in `docs/sources/doshi2023_grok_or_not.md`; if the source defines it on a different object (weights vs Fourier coefficients), implement that and add `_ours` variants explicitly |
| `dominant_frequency(power)`, `dominant_fraction(power)` | argmax and its share |
| `harmonic_shares(power, k, p)` | per column with its **own** fundamental `k`: `fundamental_share = q_k`; `odd_share = Σ_{j∈{3,5,7}} q_{alias(jk)}`; `even_share = Σ_{j∈{2,4,6}} q_{alias(jk)}`; `odd_minus_even`; `n_collisions` (harmonics aliasing onto each other or onto k — zero for prime p, asserted in a test); `ideal_square_odd_share`, `ideal_square_even_share` from `discrete_square_reference(p, k)` |
| `discrete_square_reference(p, k, phi)` | spectrum shares of `sign(cos(2πkn/p − φ))` sampled at n=0..p−1 (the discrete square wave has small even-harmonic power at odd p; the reference is computed, not assumed) |
| `periodicity_score(curve)` | Swaroop 2603.23784's per-neuron score **if the source note defines one**; otherwise `dominant_fraction` under the name `periodicity_score_ours` with a docstring saying the source has no named score |
| `binarization_score(curve)` | fraction of `|y|/max|y|` above 0.8 (near-binary weights, Swaroop) — labelled ours unless the source defines it |

Tests: injected sinusoid → entropy≈0, PR≈1, topk=1; white noise → entropy≈1, PR≈half; ideal continuous
square wave → odd shares ≈ 1/9, 1/25, 1/49 of fundamental; the discrete reference reproduces itself.

## 3. `analysis/wave_fitting.py` — waveform model comparison

Input: one curve `y [p]` (or a matrix `[p, n]`), a fundamental `k`.

| model | form | free params |
|---|---|---|
| `sinusoid` | `α cos(2πkn/p + φ) + β` | 3 (linear in `(α cos φ, −α sin φ, β)`) |
| `square` | `α sign(cos(2πkn/p + φ)) + β` | 3 (`φ` by grid search over 4p points then golden-section refine; `α, β` by LSQ) |
| `odd_harmonics` | `β + Σ_{j∈{1,3,5,7}} α_j cos(2π(jk)n/p + φ_j)` | 9 (linear in cos/sin pairs) |
| `odd_harmonics_1_over_j` | same but amplitudes constrained `α_j = α/j`, shared phase → 3 params (the ideal-square-wave Fourier truncation) |

Per fit: `pred`, `rss`, `nmse = rss / Σ(y−ȳ)²`, `r2 = 1 − nmse`, `aic = p·ln(rss/p) + 2·n_params`,
`aicc`, and the parameters. `compare_models(y, k)` returns all four plus `best_by_aic`, `delta_aic`
(each model minus the best), and `best_by_cv`. `cross_validate(y, k, folds=5, seed)` splits the `p`
token positions into folds, fits on the rest, reports held-out NMSE per model (mean over folds).
`frequency_sensitivity(y, ks)` refits at each `k` in `ks` (default: the top-3 spectral peaks and `k±1`)
and reports the best model per `k`. `fit_matrix(Y [p, n], k_per_column) -> dict of arrays` and
`summarize(...)` → distribution over columns: fraction best-by-AIC per model, quantiles of `r2` per model,
quantiles of `delta_aic(sinusoid − square)`, quantiles of the fitted `α_j/α_1` ratios for j=3,5,7 (to be
compared with 1/j).

Tests: synthetic sinusoid → sinusoid wins by AIC with r2>0.999 and square loses by >10 AIC; synthetic
discrete square wave → square wins; odd-harmonic ratios ≈ 1/3, 1/5, 1/7 within 5 %; noise robustness at
SNR 3; phase recovery within 2°; CV agrees with AIC on both synthetic cases; a wrong `k` gives r2<0.2.

## 4. `analysis/key_frequencies.py` — key-frequency selection rules (pre-registered)

`select(run_dir, step, rule, **kw) -> {"rule", "key_frequencies", "n", "scores", "cap_binding", ...}` with rules:

| rule | definition |
|---|---|
| `embedding_top8` | legacy: top-8 of the `W_E[:p]` power spectrum (cap always binds — audit §3.2) |
| `embedding_threshold` | smallest set reaching 90 % of `W_E` power, **uncapped**; count is measured |
| `logit_sum_directions` | 2D Fourier of the centered full-grid logits; power in the `cos/sin(w_k(a+b))` directions per k (via `mask_protocols.SumDirectionsOnly` per single k); keep k whose share exceeds `1/half · 3` (three times uniform) — report the share curve |
| `neuron_clusters` | per hidden neuron (effective curves, §5/§6) dominant frequency; key set = frequencies that are the dominant frequency of at least `min_neurons` (default 5) neurons **and** whose neurons jointly carry ≥ 2 % of total activation variance |
| `nanda` | the rule reconstructed in `docs/sources/nanda2023_progress_measures.md`; if that note says it is not reconstructable, this rule raises `NotImplementedError` with the reason |

The **pre-registered primary rule** is written in `docs/PREREGISTRATION.md` §Metrics; every module that
needs a key set takes `rule` as a parameter and records it. `agreement(rules...)` reports the Jaccard
overlap between the sets from different rules.

## 5. `analysis/mlp_mechanism.py` — per-neuron mechanism of the shared-embedding MLP (extend, keep API)

Keep `effective_curves`, `curve_spectra`, `classify_neurons`, `phase_relation`, `sum_dependence`,
`PROPOSED`, `SENSITIVITY`. Add:

* `neuron_tables(curves, p)` → per neuron and per curve (`u_a`, `u_b`, `out`): dominant frequency,
  dominant fraction, every §2 metric, harmonic shares, phases of the top-3 frequencies; frequency-agreement
  flags `same_ab`, `same_ab_out`; contingency counts.
* `activation_analysis(curves, p, key_freqs)` → per neuron, on `act[a,b] = ReLU(u_a[a]+u_b[b]+b_in)`:
  2D power spectrum (via `fwd2d`) and its top-5 `(i, j)` mode pairs; `same_frequency_block_share`
  (power in same-frequency 2×2 blocks / total non-constant power); `sum_direction_share` and
  `diff_direction_share` (per-k `SumDirectionsOnly` projection and its (a−b) counterpart — add a
  `DiffDirectionsOnly` operator to `mask_protocols` for this); `ev_sum` / `ev_diff` (existing
  `sum_dependence`); `swap_symmetry = corr(act[a,b], act[b,a])`; `ev_hypothesized`: R² of `act` against
  `ReLU(fitted_a[a] + fitted_b[b] + b_in)` for the sinusoid fits and for the square fits of `u_a`, `u_b`
  (from §3); `dead` flag; store the 2D spectra of up to 16 exemplar neurons and all per-neuron scalars.
* `logit_contributions(curves, W_out, W_b_out, p)` → per neuron: `act[a,b,i]·W_out[i,c]` summarized as
  (i) mean contribution to the correct-class logit over the grid, (ii) share of total logit variance,
  (iii) share of the correct-class margin; the full `[p,p,p]` per-neuron tensor is **not** stored.
* `structured_neuron_definitions(tables, activation)` → masks for: `proposed_0.50` (+ sensitivity 0.30 /
  0.70), `ipr_doshi` (threshold from the source note, else `[ours]`), `periodicity_swaroop` (same), and the
  pre-registered **primary** definition (in `PREREGISTRATION.md`); pairwise Jaccard overlap between
  definitions; counts per frequency.
* `analyse(run_dir, step, key_rule, seed)` runs everything and writes
  `analysis/mlp_mechanism/<tag>.json` + `.npz` (per-neuron arrays: `u_a`, `u_b`, `out` spectra, masks,
  phases, fit results).
* Works unchanged for `arch == "mlp_twohot"` where `u_a = W_in[:p]`, `u_b = W_in[p:]`.

Tests (extend the existing ones): hand-built circuit → tables/masks/phase-relation pass; scrambled
control fails; activation analysis on an ideal `ReLU(cos(w a − φa) + cos(w b − φb))` neuron shows
`sum_direction_share` ≫ `diff_direction_share` and `swap_symmetry` ≈ 1 when φa = φb.

## 6. `analysis/transformer_mechanism.py` — the same tests for the transformer

The model (`models/transformer.py`, **no LayerNorm**, so every map below is exact):

```
x_s = W_E[tok_s] + W_pos[s]                          s ∈ {0: a, 1: b, 2: '='}
z_h = Σ_s attn[h, 2, s] · (x_s @ W_V[h])             attention output of head h at the read-out position
r_pre = x_2 + Σ_h z_h @ W_O[h]                       residual before the MLP, [p, p, d_model] over (a,b)
h_f  = ReLU(r_pre @ W_in[:, f])                      hidden activation of neuron f, [p, p]
r_post = r_pre + Σ_f h_f · W_out[f]
logits = r_post @ W_U[:, :p]
```

Provide `forward_decomposition(model, cfg) -> dict` returning, over the full grid, `attn [p,p,h,3]`
(row 2 only), `z [p,p,h,d]`, `r_pre`, `hidden [p,p,d_mlp]`, `r_post`, `logits`, `direct_path_logits =
r_pre @ W_U`, `mlp_path_logits`, all float32 npz-able (≈ 113²·(4·128+128+512+128+113·3)·4 B ≈ 80 MB; store
only `hidden`, `attn`, and the two logit paths; recompute the rest on demand).

* **Effective operand curves** (the object comparable to the MLP's `u_a`, `u_b`): with the *mean*
  attention weights `ā_h,s` over the grid,
  `u_a[a, f] = ā_{h,0} Σ_h ((W_E[a] + W_pos[0]) @ W_V[h] @ W_O[h]) @ W_in[:, f]`, likewise `u_b` with
  position 1, and a constant `c_f` from position 2 and the mean of the remaining terms. Report
  `additivity_r2[f]` = R² of the true pre-activation `r_pre @ W_in[:, f]` against `u_a + u_b + c`;
  `1 − additivity_r2` is the share carried by input-dependent attention — the quantity RESEARCH_SPEC §3.4
  asks for. The MLP has `additivity_r2 ≡ 1` by construction (state this in the docstring).
* **Output curves**: `neuron_logit_map = W_out @ W_U[:, :p]` (`[d_mlp, p]`), the transformer's `out`
  curve; also `direct_path_share` of logit variance.
* Apply **exactly** §2, §3, §5 (`neuron_tables`, `activation_analysis` on the true `hidden`,
  `logit_contributions`, `structured_neuron_definitions`, `phase_relation`) to `u_a`, `u_b`, `out` —
  reuse the functions from `mlp_mechanism`; do not re-implement them.
* **Input/output direction spectra**: spectrum of `W_E[:p] @ W_in[:, f]` (embedding→neuron direct
  direction, ignoring attention) and of `W_E[:p] @ W_OV_h @ W_in[:, f]` per head; spectrum of the
  neuron-logit map rows; spectrum of `W_U[:, :p]` columns; `W_pos` norms.
* **Key-frequency subspace variance**: for `r_pre`, `r_post` and `logits` over `(a,b)`: fraction of
  variance in the `cos/sin(w_k(a+b))` directions of the key set (per k and total) vs. `n_control` random
  same-size frequency sets (via `mask_protocols` projections applied to the `[p,p,d]` tensor).
* **Attention, differentiated**: per head: mean, std, min, max, 5/95 quantiles of `attn[h,2,0]` over all
  inputs; dependence on `a`, on `b`, on `(a+b) mod p` (variance explained); over checkpoints (with
  `--all-checkpoints`); **causal head ablation**: zero-ablate and mean-ablate each head (replace `z_h` by
  0 / by its grid mean) and report train/test loss/acc changes; also "fix attention to its mean" (replace
  `attn` by `ā`) — if loss barely changes, attention is a fixed sum (§3.4 case (i)).
* **Trig-identity agreement**: for each key k, the share of `hidden` variance and of `logits` variance in
  the `cos/sin(w_k(a+b))` subspace, and the phase relation on structured neurons (from §5 reuse).

Writes `analysis/transformer_mechanism/<tag>.json` + `.npz`. Tests: on a hand-built transformer with
`W_Q = W_K = 0` (uniform attention) the effective curves reproduce the true pre-activation exactly
(`additivity_r2 = 1`); `forward_decomposition` logits equal `model.logits_last` to 1e-4; head ablation of
an all-zero head changes nothing; the key-subspace variance of a synthetic `cos(w_k(a+b))` field is 1 for
k and 0 for other k.

## 7. `analysis/logit_formula_fit.py` — end-to-end fit to the full logit tensor (mandatory, Phase 5)

Input: centered logits `L [p,p,p]` (`center_logits`), a frequency set `K` (from `key_frequencies`),
split masks. Candidate formulas, all fit by linear least squares on the full grid unless noted:

| id | features | params |
|---|---|---|
| `sparse_sinusoid` | `{cos(w_k(a+b−c)), sin(w_k(a+b−c)) : k ∈ K}` + const | `2|K|+1` |
| `odd_harmonics` | same features over the aliased odd-harmonic family `{alias(jk): k∈K, j∈{1,3,5,7}}` | `2|F|+1` |
| `ideal_square` | `{sq(w_k(a+b−c) + φ_k) : k ∈ K}`, `φ_k` by grid search per k, then joint LSQ on amplitudes | `2|K|+1` |
| `control_top_m` | `sparse_sinusoid` with the top-`|F|` frequencies of the logit sum-direction power (matched cardinality to `odd_harmonics`) | `2|F|+1` |
| `control_random_m` | `sparse_sinusoid` with `n_control` random frequency sets of size `|F|` | `2|F|+1` |
| `control_difference` | `sparse_sinusoid` in `(a−b−c)` (wrong symmetry, same flexibility) | `2|K|+1` |
| `full_sum_basis` | all `(p−1)/2` frequencies in `(a+b−c)` — the ceiling of any "function of a+b−c" model | `p` |

Report per formula: `r2` on all/train/test cells, per-class `r2` quantiles, residual RMS, `aic`,
`argmax_accuracy` of the fitted logits on train/test, and for the random control the distribution.
`analyse(run_dir, step, key_rule, seed)` writes `analysis/logit_formula_fit/<tag>.json` (+ npz with the
fitted coefficients and the residual tensor's per-(a,b) RMS map). Tests: a synthetic model whose logits
are exactly `Σ_k cos(w_k(a+b−c))` → `sparse_sinusoid` r2 > 0.999 and `control_difference` r2 < 0.05; a
synthetic square-wave logit tensor → `ideal_square`/`odd_harmonics` beat `sparse_sinusoid` by AIC; random
logits → every r2 ≈ params/p³.

## 8. `analysis/function_agreement.py` — functional equivalence over all p² inputs (Phase 6)

`compare(run_dir_a, run_dir_b, step_a, step_b)` → over all `p²` pairs: `agreement_rate`, `both_correct`,
`only_a_correct`, `only_b_correct`, `both_wrong`, `error_jaccard`, the same four on the train and on the
test split of **each** model's own split (paired seeds share the split; assert and record
`split_hash` equality), `logit_pearson` (flattened centered logits), `margin_pearson`, `top2_agreement`,
error structure: error counts by residue `(a+b) mod p`, by `a`, by `b`, by `|a−b|`, symmetric-error
share (`(a,b)` and `(b,a)` both wrong), and the list of disagreement pairs (npz). `baseline(run_dirs)`
computes the same for every same-architecture seed pair so cross-architecture numbers can be read against
within-architecture variability. Writes `results/function_agreement/<id_a>__vs__<id_b>.json`. Tests: two
identical synthetic models → agreement 1, jaccard 1; two models differing on a known set of 10 inputs →
agreement `1 − 10/p²` and the disagreement list equals that set.

## 9. `analysis/causal_ablation.py` — necessity and sufficiency tests (Phase 7)

Common: `evaluate(model_fn, cfg) -> {train_loss, test_loss, train_acc, test_acc, logits}` on the full grid
using split masks; `report(base, ablated)` → absolute and relative changes of the four metrics, `mean_abs_logit_change`,
`margin_change` (mean correct-class margin), per-class accuracy change (npz), `n_components_removed`.
Every ablation returns `{observed: report, control: {n_control reports summarized}, z}` where the control is
**size-matched random** (same number of neurons / frequencies / dimensions, `n_control=50`, seeded). Nothing
is retrained; models are copied before modification.

MLP (`effective` forward `ReLU(u_a[a] + u_b[b] + b_in) @ W_out + b_out`, identical to the model):

| id | what | test type |
|---|---|---|
| `keep_structured` | zero all neurons outside the structured set S | sufficiency of S |
| `remove_structured` | zero S | necessity of S |
| `keep_unstructured` | = `remove_structured` (reported once, named both ways) | |
| `remove_structured_by_frequency` | zero the S-neurons of one key frequency at a time | necessity per frequency |
| `remove_key_freqs_from_curves` | project `u_a`, `u_b` onto the complement of the key-frequency subspace | necessity of key freqs |
| `keep_key_freqs_in_curves` | project onto the key-frequency subspace (+ constant) | sufficiency |
| `replace_with_sinusoid_fit` / `replace_with_square_fit` | substitute `u_a`, `u_b` by their §3 fits | sufficiency of the waveform model |
| `wrong_phase_vs_weak_periodicity` | zero the group "right frequency, phase error > 45°" vs the group "right phase, dominant fraction < 0.3", size-matched | comparison |

Transformer (functional forward from §6 `forward_decomposition` pieces):

| id | what |
|---|---|
| `remove_key_freqs_from_embedding` / `keep_key_freqs_in_embedding` | filter `W_E[:p]` rows in the Fourier basis |
| `remove_structured_neurons` / `keep_structured_neurons` | zero hidden neurons (S from §6 definitions) |
| `ablate_head_h` (zero, mean) and `fix_attention_to_mean` | attention |
| `remove_key_subspace_from_residual` | project the `(a+b)`-key-frequency directions out of `r_pre` over the grid, then finish the forward |
| `restricted_circuit_only` | replace logits by their key-frequency `sum_directions_only` restriction (this **is** the restricted loss, reported as a sufficiency test) |

`analyse(run_dir, step, structured_definition, key_rule, seed)` writes `analysis/causal_ablation/<tag>.json`.
Tests on a hand-built 2-frequency ideal MLP circuit: `remove_structured` destroys accuracy (≤ 0.1) while
the size-matched random control keeps > 0.9; `keep_structured` keeps accuracy; `remove_key_freqs` destroys
it; a neuron set that the model does not use changes nothing (the "no damage" case must be representable).

## 10. `analysis/mask_protocols.py` + `progress_measures.py` — restricted/excluded variants

Add, named exactly as the master prompt requires: `nanda_exact_restricted_loss`,
`nanda_exact_excluded_loss` (implemented from `docs/sources/nanda2023_progress_measures.md`; if the note
says an item is not reconstructable the function raises with that reason and the doc records it),
`full_grid_extension_restricted_loss`, `full_grid_extension_excluded_loss` (our variant: the exact
component rule but evaluated on the full grid), `legacy_broad_mask_variant` (alias of the existing one).
Each takes `(logits [p,p,p], key_freqs, cfg)` and returns `{loss, split, n_components, protocol}`. Add a
`DiffDirectionsOnly` operator (see §5) and a `per_frequency` option that reports the share of logit
variance per key k in the sum directions. `progress_measures.compute_from_checkpoints(run_dir)` evaluates
every variant on every checkpoint of a v2 run (no re-train) and writes
`analysis/progress_measures/all_checkpoints.json`. The existing re-train path stays for legacy runs.

## 11. `analysis/statistics.py` — paired-by-seed statistics

`paired(values_a, values_b, seeds_a, seeds_b)` (aligns on seed, drops unpaired with a count) →
`n_pairs`, `n_total_a/b`, per-seed differences (listed), `mean_diff`, `median_diff`, `bootstrap_ci95`
(percentile, `n_boot=10000`, seeded), `sign_test_p` (exact binomial), `wilcoxon_signed_rank_p` (exact
enumeration for n ≤ 20, numpy only), `cohens_dz`, `cliffs_delta`, `direction`, `all_positive/negative`.
`robust(values)` → median, MAD, IQR, min, max, n, n_missing. `bootstrap_ci(fn, x, n_boot, seed)`.
`plot_paired(values_a, values_b, seeds, labels, path)` draws every seed as a connected pair. Tests:
known paired shift recovers sign and CI excluding 0; identical samples give CI containing 0 and p ≈ 1;
Wilcoxon exact for n=6 matches a hand-enumerated value; unpaired seeds are dropped and counted.

## 12. `analysis/structure_over_time.py` — H4 (structure precedes generalization)

For a v2 run: for every checkpoint compute a compact metric set (embedding top-k and `embedding_threshold`
count; per-neuron structured fraction under the primary definition; phase-relation R; fraction of neurons
best-fit by square vs sinusoid; median harmonic `odd_minus_even`; restricted/excluded losses for every
protocol; key-subspace variance of the logits; weight norms; train/test acc at that step) →
`analysis/structure_over_time.json` + a figure. H4 statistic: for each metric, value at
`pre_generalization` minus value at `init`, with a bootstrap CI over neurons, and the first checkpoint at
which the metric exceeds `init + 3·(its std over the first two checkpoints)`.

## 13. `analysis/aggregate.py`, `analysis/figures_study.py`, `analysis/decision_tree.py`

* `aggregate` collects every per-run JSON of the matrix into `results/aggregate/<module>.json` and
  markdown tables with all individual seeds shown; `figures_study` draws every figure of the study from
  those files only (no recomputation), naming the source file in the figure's metadata caption.
* `decision_tree.evaluate(results_dir)` applies the **pre-registered** evidence gate per architecture
  (four criteria, pass rules in `PREREGISTRATION.md`) and prints the branch taken; it never changes a
  threshold. If an architecture fails, `bounded_alternative.py` runs the bounded follow-up: linear and
  1-hidden-layer probes decoding `(a+b) mod p` from `hidden` (train on the model's train split, test on
  its test split, vs shuffled-label control), effective rank / SVD spectrum of `hidden` and of the logits
  (with the Khanh caveat), cross-seed consistency of the found structure (CKA between seeds), and the
  causal test of removing the top-r singular directions vs random directions.
