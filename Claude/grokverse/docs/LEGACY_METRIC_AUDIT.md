# Legacy metric audit — structure metrics on the 16 pre-study runs

**AI-drafted (Claude), 2026-09-02 — not yet human-reviewed.**

Phase 1 (audit) document: it *describes* what the structure metrics used so far measure on the
existing runs and what the work-in-progress harmonic-shape statistic does on synthetic data. It
interprets no hypothesis (H1–H5 of `docs/RESEARCH_SPEC.md`) and implements nothing.

All numbers below are produced by `training/audit_legacy_metrics.py` and stored in
`docs/data/legacy_metrics_all_runs.json` (the JSON holds more decimals and more fields than the
tables here). Anything not traceable to that file is marked.

## 0. Inputs, code, environment

| item | value |
|---|---|
| runs | the 16 run directories listed in `docs/BASELINE.md` (pinned by run_id in the script). The 20 `*_arch25k` run-format-v2 directories of the running study matrix that share `training/runs/` are excluded; the excluded names are listed in the JSON's `meta.non_legacy_dirs_skipped` |
| weight object | `training/runs/<run_id>/embeddings.npy[-1][:p]` — the last logged embedding snapshot, rows 0..112 (number tokens; the transformer's `=` row 113 is dropped by `[:p]`); float32 on disk, cast to float64 |
| snapshot step | `logged_steps[-1]` of each `run.json` (column "step" in Table A). For the six un-accelerated `frac0.3` runs this is the generalization crossing (`docs/BASELINE.md`: "every legacy 'final' structure number is an at-transition number") |
| spectrum | `grokverse.analysis.fourier.embedding_power_spectrum`: orthonormal real Fourier basis over Z_113, `power[k] = cos_k^2 + sin_k^2` summed over `d_model`, k = 1..56; the constant mode is excluded from every denominator (its share of total power is 0.03–0.6 % on these runs, column `const_power` in the JSON) |
| legacy metric | `grokverse.analysis.fourier.dominant_frequencies(W_E, p, threshold=0.9, max_k=8)` |
| family metric | `grokverse.analysis.fourier.harmonic_family_concentration(W_E, p, n_f, max_harmonic=7, n_null=2000, seed=0)` |
| environment | Python 3.12.10, numpy 2.4.6, Windows 11, branch `arch-study`. First run 2026-09-02 at git HEAD `84d449d`; re-run 2026-09-03 (JSON `meta.generated_utc` = 2026-09-02T22:36Z) at HEAD `aa536d2`. Every value outside `meta` is identical between the two JSON files (compared field by field, relative tolerance 10⁻¹²); only `git_head`, the timestamp and the skipped-directory list differ |
| script | `training/audit_legacy_metrics.py` (run from `training/`; deterministic, all random draws seeded) |

### 0.1 Definitions used in the tables

| column | definition |
|---|---|
| top-K | sum of the K largest `power[k]` divided by the sum over all 56 |
| n90 | `dominant_frequencies(...)["n_freqs_for_threshold"]`: the smallest K whose cumulative fraction reaches 0.9, **uncapped** |
| cap binds | `dominant_frequencies(...)["cap_binding"]` = (n90 > 8). When true the reported "dominant" set is the fixed top-8, not a measured count |
| n50 | same as n90 for a 0.5 threshold (descriptive extra; n75 and n95 are in the JSON) |
| H_norm | spectral entropy `-Σ q_k ln q_k / ln 56`, `q_k = power_k / Σ power`; 0 = one frequency, 1 = flat |
| PR | participation ratio `(Σ power_k)^2 / Σ power_k^2`, in [1, 56]; 56 = flat |
| family | `family_fraction`: power in `{alias(j·k): k ∈ fundamentals, j ∈ {1,3,5,7}}` / total |
| top-m | `matched_top_m_fraction`: top-m concentration at the same cardinality m = |family| |
| null mean / q95 | concentration of 2000 uniformly random size-m frequency sets (seed 0) |
| odd / even | `harmonic_shape`: power at aliased j = 3,5,7 (resp. 2,4,6) harmonics of the fundamentals, fundamentals removed, **divided by the power at the fundamentals** |
| o−e | `odd_minus_even` |
| fpf | `fundamental_power_fraction`: power at the fundamentals / total |
| coll | number of pairs (k, j), j ∈ {2..7}, k in the selected set, with `alias(j·k)` equal to *another* selected fundamental |
| ovl | number of indices that lie in **both** the odd and the even index set of `harmonic_shape` |

The legacy metric's own description, verbatim from `analysis/fourier.py` (`dominant_frequencies`):

> """Smallest set of frequencies capturing `threshold` of the embedding power."""

and the in-code provenance comment added in the previous session:

> On all 16 existing runs the max_k cap binds and the 90% threshold never fires, so "top-8" is a FIXED k, not one measured from data

`docs/RESEARCH_SPEC.md` §3.2 states the same:

> Checked on all 16 existing runs: the cap binds in **every single one** — `n_keep == 8` always, for Transformer and MLP alike, including the multiplication runs at 0.16 concentration. The 90% threshold never fires.

## 1. Table A — concentration, frequency count, entropy, participation ratio

| run_id | step | test acc | top-1 | top-3 | top-8 | n90 | cap binds | n50 | H_norm | PR |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_add_p113_wd1.0_frac0.3_seed0 | 9646 | 0.9942 | 0.0805 | 0.2059 | 0.4308 | 39 | yes | 10 | 0.9038 | 28.43 |
| mlp_add_p113_wd1.0_frac0.3_seed1 | 10357 | 0.9751 | 0.0788 | 0.2209 | 0.4456 | 40 | yes | 10 | 0.9092 | 28.20 |
| mlp_add_p113_wd1.0_frac0.3_seed2 | 9646 | 0.9830 | 0.0747 | 0.2114 | 0.4311 | 38 | yes | 10 | 0.9019 | 28.15 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0 | 2246 | 0.9848 | 0.0670 | 0.1710 | 0.3542 | 43 | yes | 14 | 0.9439 | 36.19 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1 | 2246 | 0.9865 | 0.0601 | 0.1540 | 0.3486 | 43 | yes | 14 | 0.9434 | 36.68 |
| mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2 | 2378 | 0.9945 | 0.0669 | 0.1662 | 0.3543 | 42 | yes | 14 | 0.9425 | 36.14 |
| txf_add_p113_wd1.0_frac0.3_gf2.0_seed0 (never generalized) | 4000 | 0.1064 | 0.0631 | 0.1743 | 0.3197 | 46 | yes | 17 | 0.9627 | 40.09 |
| txf_add_p113_wd1.0_frac0.3_seed0 | 8367 | 0.9808 | 0.1895 | 0.4698 | 0.7571 | 27 | yes | 4 | 0.7130 | 9.86 |
| txf_add_p113_wd1.0_frac0.3_seed1 | 6295 | 0.9653 | 0.1468 | 0.4098 | 0.6590 | 36 | yes | 4 | 0.7845 | 12.67 |
| txf_add_p113_wd1.0_frac0.3_seed2 | 8367 | 0.9937 | 0.2107 | 0.4897 | 0.7853 | 23 | yes | 4 | 0.6952 | 9.32 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed0 | 717 | 0.9834 | 0.1291 | 0.3500 | 0.5206 | 43 | yes | 7 | 0.8687 | 18.54 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed1 | 912 | 0.9904 | 0.2610 | 0.4634 | 0.6050 | 40 | yes | 4 | 0.7914 | 10.32 |
| txf_add_p113_wd1.0_frac0.5_gf2.0_seed2 | 953 | 0.9850 | 0.2950 | 0.5297 | 0.6432 | 38 | yes | 3 | 0.7349 | 7.48 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed0 | 969 | 0.9850 | 0.0208 | 0.0609 | 0.1592 | 50 | yes | 27 | 0.9991 | 55.63 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed1 | 969 | 0.9887 | 0.0209 | 0.0616 | 0.1581 | 50 | yes | 27 | 0.9994 | 55.75 |
| txf_mul_p113_wd1.0_frac0.5_gf2.0_seed2 | 1093 | 0.9826 | 0.0221 | 0.0648 | 0.1659 | 50 | yes | 26 | 0.9989 | 55.51 |

Descriptive facts visible in Table A:

* `cap_binding` is true for all 16 runs; the uncapped count n90 lies between 23 and 50. The recorded
  "dominant" set is therefore the top-8 by construction on every run, as §3.2 states.
* The number of frequencies needed for **half** of the power is 4 for each of the three un-accelerated
  transformer `add` runs and 3–7 for the three Grokfast ones, 10 for the un-accelerated MLPs, 14 for the
  Grokfast MLPs, 17 for the non-generalized transformer, 26–27 for the `mul` runs.
* The three `mul` runs have PR 55.5–55.7 of a maximum 56 and H_norm 0.999: their embedding spectra
  are flat to within the resolution of these statistics. (These runs are reserved for the human author,
  `docs/BASELINE.md`; they are tabulated, not discussed.)

### 1.1 Cross-check against the numbers already reported

Mean ± sample std (ddof = 1) of top-8 over seeds, recomputed here, next to the reported figure:

| group | recomputed top-8 | reported | where |
|---|---|---|---|
| transformer, frac 0.3, un-accelerated (3 seeds) | 0.7338 ± 0.0663 | 0.73 ± 0.07 | `RESULTS.md` §3 |
| MLP, frac 0.3, un-accelerated (3) | 0.4358 ± 0.0085 | 0.44 ± 0.01 | `RESULTS.md` §3 |
| transformer, frac 0.5, Grokfast (3) | 0.5896 ± 0.0627 | 0.59 ± 0.06 | `RESULTS.md` §3 |
| MLP, frac 0.5, Grokfast (3) | 0.3524 ± 0.0032 | 0.35 ± 0.003 | `RESULTS.md` §3 |
| canonical run `txf_add_p113_wd1.0_frac0.3_seed0` | 0.7571 | 76 % | `README.md`, `RESULTS.md` |
| non-generalized `txf_add_p113_wd1.0_frac0.3_gf2.0_seed0` | 0.3197 | 32 % | `README.md` |
| transformer `mul` (3) | 0.1611 ± 0.0042 | 0.16 | `docs/RESEARCH_SPEC.md` §3.2 |
| transformer Grokfast seed0 / seed1 | 0.5206 / 0.6050 | 52 % / 60 % | `PROGRESS.md` |

Every reported legacy number is reproduced from the archived embeddings.

Secondary statistics per group (mean ± sample std; the non-generalized run and the `mul` runs shown for
completeness):

| group | top-1 | top-3 | H_norm | PR | n90 |
|---|---|---|---|---|---|
| txf add frac0.3 plain (3) | 0.182 ± 0.033 | 0.456 ± 0.042 | 0.731 ± 0.047 | 10.6 ± 1.8 | 28.7 ± 6.7 |
| mlp add frac0.3 plain (3) | 0.078 ± 0.003 | 0.213 ± 0.008 | 0.905 ± 0.004 | 28.3 ± 0.2 | 39.0 ± 1.0 |
| txf add frac0.5 gf (3) | 0.228 ± 0.088 | 0.448 ± 0.091 | 0.798 ± 0.067 | 12.1 ± 5.7 | 40.3 ± 2.5 |
| mlp add frac0.5 gf (3) | 0.065 ± 0.004 | 0.164 ± 0.009 | 0.943 ± 0.001 | 36.3 ± 0.3 | 42.7 ± 0.6 |
| txf add frac0.3 gf, never generalized (1) | 0.063 | 0.174 | 0.963 | 40.1 | 46 |
| txf mul frac0.5 gf (3) | 0.021 ± 0.001 | 0.062 ± 0.002 | 0.999 ± 0.000 | 55.6 ± 0.1 | 50.0 ± 0.0 |

## 2. Table B — the legacy top-8 sets and their aliasing collisions

For each run, the recorded top-8 set (`dominant_frequencies(...)["dominant"]`, strongest first) and every
harmonic j·k with j ∈ {2,…,7} of a member k that folds (`alias_frequency`) onto *another* member.

| run_id | legacy top-8 | coll (odd j / even j) | colliding pairs `j·k → k'` |
|---|---|---|---|
| mlp_add … frac0.3_seed0 | 21, 37, 3, 33, 19, 47, 1, 13 | 9 (3 / 6) | 6·21→13, 7·37→33, 7·3→21, 2·33→47, 4·33→19, 4·19→37, 6·19→1, 2·47→19, 3·1→3 |
| mlp_add … frac0.3_seed1 | 18, 6, 1, 26, 37, 29, 47, 16 | 4 (3 / 1) | 3·6→18, 6·1→6, 3·29→26, 7·16→1 |
| mlp_add … frac0.3_seed2 | 24, 40, 53, 4, 8, 31, 3, 10 | 8 (2 / 6) | 6·24→31, 2·4→8, 6·4→24, 3·8→24, 5·8→40, 6·31→40, 4·10→40, 6·10→53 |
| mlp_add … frac0.5_gf2.0_seed0 | 13, 38, 33, 4, 48, 11, 1, 21 | 5 (3 / 2) | 5·13→48, 3·38→1, 3·11→33, 4·1→4, 6·21→13 |
| mlp_add … frac0.5_gf2.0_seed1 | 20, 47, 1, 38, 31, 3, 26, 16 | 5 (4 / 1) | 4·47→38, 3·1→3, 3·38→1, 3·31→20, 7·16→1 |
| mlp_add … frac0.5_gf2.0_seed2 | 44, 24, 51, 32, 10, 19, 27, 13 | 2 (2 / 0) | 3·44→19, 3·27→32 |
| txf_add … frac0.3_gf2.0_seed0 | 21, 18, 11, 8, 15, 47, 17, 44 | 5 (2 / 3) | 5·21→8, 4·11→44, 6·11→47, 7·15→8, 6·17→11 |
| txf_add … frac0.3_seed0 (canonical) | 18, 15, 1, 11, 13, 22, 56, 36 | 6 (3 / 3) | 2·18→36, 7·18→13, 2·11→22, 7·11→36, 7·13→22, 2·56→1 |
| txf_add … frac0.3_seed1 | 16, 30, 17, 9, 50, 32, 45, 26 | 10 (5 / 5) | 2·16→32, 6·16→17, 7·30→16, 4·17→45, 5·9→45, 7·9→50, 4·50→26, 3·32→17, 4·26→9, 5·26→17 |
| txf_add … frac0.3_seed2 | 46, 53, 55, 52, 4, 3, 21, 43 | 9 (6 / 3) | 2·46→21, 5·46→4, 3·53→46, 6·53→21, 2·55→3, 3·55→52, 7·55→46, 3·52→43, 7·3→21 |
| txf_add … frac0.5_gf2.0_seed0 | 16, 39, 17, 35, 8, 44, 21, 53 | 9 (3 / 6) | 6·16→17, 2·39→35, 6·39→8, 3·35→8, 6·35→16, 2·8→16, 5·21→8, 5·53→39, 6·53→21 |
| txf_add … frac0.5_gf2.0_seed1 | 27, 9, 16, 54, 33, 38, 45, 40 | 8 (5 / 3) | 2·27→54, 3·9→27, 5·9→45, 6·9→54, 5·16→33, 7·38→40, 2·40→33, 7·40→54 |
| txf_add … frac0.5_gf2.0_seed2 | 4, 29, 15, 8, 55, 38, 47, 41 | 5 (2 / 3) | 2·4→8, 2·29→55, 5·15→38, 7·15→8, 4·47→38 |
| txf_mul … frac0.5_gf2.0_seed0 | 44, 48, 30, 29, 39, 31, 9, 24 | 6 (4 / 2) | 7·44→31, 3·48→31, 5·39→31, 7·31→9, 2·24→48, 6·24→31 |
| txf_mul … frac0.5_gf2.0_seed1 | 16, 11, 24, 23, 49, 19, 7, 20 | 6 (4 / 2) | 4·16→49, 5·24→7, 5·49→19, 7·19→20, 7·7→49, 6·20→7 |
| txf_mul … frac0.5_gf2.0_seed2 | 4, 6, 32, 13, 12, 40, 16, 23 | 4 (1 / 3) | 3·4→12, 4·4→16, 2·6→12, 2·16→32 |

Every legacy top-8 set has between 2 and 10 such collisions (101 in total over 16 sets, 6.3 per set). The
canonical run's six collisions are the ones listed in `docs/BASELINE.md`. Note on order only: the
canonical set recorded in `progress_measures.json` is `[18, 15, 11, 1, 13, 56, 22, 36]`; the recomputation
ranks it `[18, 15, 1, 11, 13, 22, 56, 36]` — the same eight frequencies, two adjacent pairs swapped
(the JSON's `ranked_fraction_top8` shows how close those pairs are). Not investigated further here.

## 3. Harmonic-family concentration with its controls and the shape statistic

Computed exactly as `harmonic_family_concentration` does: greedy fundamentals
(`select_fundamentals`), capped odd family j ≤ 7, cardinality-matched top-m control, random-set null
(2000 draws, seed 0), and `harmonic_shape` on the selected fundamentals. Row order as in Table A.

### 3.1 n_f = 2

| run (abbrev.) | fund. | m | family | top-m | null mean / q95 | odd | even | o−e | fpf | coll | ovl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp 0.3 s0 | 3, 37 | 8 | 0.3196 | 0.4308 | 0.143 / 0.228 | 1.548 | 0.559 | +0.989 | 0.125 | 0 | 0 |
| mlp 0.3 s1 | 6, 16 | 8 | 0.3291 | 0.4456 | 0.142 / 0.224 | 2.171 | 0.516 | +1.655 | 0.104 | 0 | 0 |
| mlp 0.3 s2 | 8, 53 | 8 | 0.3488 | 0.4311 | 0.141 / 0.225 | 2.127 | 0.725 | +1.402 | 0.112 | 0 | 1 |
| mlp gf s0 | 13, 43 | 8 | 0.2546 | 0.3542 | 0.141 / 0.203 | 2.129 | 0.884 | +1.245 | 0.081 | 0 | 0 |
| mlp gf s1 | 30, 31 | 8 | 0.2419 | 0.3486 | 0.144 / 0.206 | 2.766 | 1.509 | +1.257 | 0.064 | 0 | 0 |
| mlp gf s2 | 44, 8 | 8 | 0.2647 | 0.3543 | 0.142 / 0.205 | 2.028 | 0.909 | +1.119 | 0.087 | 0 | 0 |
| txf 0.3 gf s0 (never gen.) | 3, 19 | 8 | 0.2360 | 0.3197 | 0.143 / 0.195 | 4.814 | 3.243 | +1.570 | 0.041 | 0 | 1 |
| txf 0.3 s0 | 18, 3 | 8 | 0.4580 | 0.7571 | 0.142 / 0.342 | 1.342 | 0.193 | +1.148 | 0.196 | 1 | 0 |
| txf 0.3 s1 | 30, 9 | 8 | 0.4938 | 0.6590 | 0.143 / 0.318 | 0.940 | 0.184 | +0.756 | 0.255 | 0 | 0 |
| txf 0.3 s2 | 55, 20 | 8 | 0.6333 | 0.7853 | 0.141 / 0.347 | 3.390 | 0.579 | +2.811 | 0.144 | 0 | 0 |
| txf gf s0 | 26, 42 | 8 | 0.3578 | 0.5206 | 0.144 / 0.276 | 14.456 | 2.110 | +12.346 | 0.023 | 1 | 0 |
| txf gf s1 | 9, 16 | 8 | 0.5452 | 0.6050 | 0.145 / 0.356 | 1.694 | 0.419 | +1.275 | 0.202 | 0 | 0 |
| txf gf s2 | 39, 28 | 8 | 0.5498 | 0.6432 | 0.141 / 0.370 | 27.394 | 4.412 | +22.982 | 0.019 | 0 | 0 |
| mul s0 | 13, 44 | 8 | 0.1540 | 0.1592 | 0.143 / 0.149 | 2.901 | 2.621 | +0.280 | 0.040 | 0 | 0 |
| mul s1 | 24, 16 | 8 | 0.1536 | 0.1581 | 0.143 / 0.149 | 2.742 | 2.175 | +0.567 | 0.041 | 0 | 1 |
| mul s2 | 39, 17 | 8 | 0.1536 | 0.1659 | 0.143 / 0.150 | 2.945 | 2.822 | +0.123 | 0.039 | 0 | 0 |

### 3.2 n_f = 4

| run (abbrev.) | fund. | m | family | top-m | null mean / q95 | odd | even | o−e | fpf | coll | ovl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp 0.3 s0 | 3, 37, 44, 47 | 15 | 0.5309 | 0.6461 | 0.269 / 0.373 | 1.599 | 0.924 | +0.675 | 0.204 | 0 | 2 |
| mlp 0.3 s1 | 6, 16, 29, 50 | 16 | 0.5524 | 0.6333 | 0.286 / 0.395 | 2.238 | 0.898 | +1.341 | 0.171 | 0 | 3 |
| mlp 0.3 s2 | 8, 53, 44, 49 | 15 | 0.5627 | 0.6459 | 0.269 / 0.368 | 3.025 | 0.894 | +2.131 | 0.140 | 0 | 1 |
| mlp gf s0 | 13, 43, 37, 49 | 16 | 0.4445 | 0.5572 | 0.284 / 0.360 | 2.999 | 1.692 | +1.308 | 0.111 | 0 | 2 |
| mlp gf s1 | 30, 31, 38, 22 | 16 | 0.4525 | 0.5698 | 0.287 / 0.360 | 2.853 | 1.714 | +1.140 | 0.117 | 0 | 2 |
| mlp gf s2 | 44, 8, 29, 20 | 16 | 0.4633 | 0.5593 | 0.286 / 0.362 | 3.133 | 1.396 | +1.737 | 0.112 | 0 | 2 |
| txf 0.3 gf s0 (never gen.) | 3, 19, 43, 17 | 16 | 0.4170 | 0.4907 | 0.286 / 0.350 | 4.187 | 3.013 | +1.174 | 0.080 | 0 | 4 |
| txf 0.3 s0 | 18, 3, 45, 11 | 16 | 0.7805 | 0.8421 | 0.285 / 0.518 | 1.429 | 0.328 | +1.101 | 0.321 | 1 | 3 |
| txf 0.3 s1 | 30, 9, 32, 56 | 16 | 0.7164 | 0.7596 | 0.286 / 0.494 | 1.346 | 0.321 | +1.024 | 0.305 | 0 | 3 |
| txf 0.3 s2 | 55, 20, 3, 39 | 16 | 0.8239 | 0.8633 | 0.287 / 0.547 | 3.021 | 0.339 | +2.681 | 0.205 | 1 | 1 |
| txf gf s0 | 26, 42, 53, 21 | 16 | 0.6202 | 0.6408 | 0.287 / 0.438 | 9.228 | 1.305 | +7.923 | 0.061 | 3 | 1 |
| txf gf s1 | 9, 16, 40, 25 | 16 | 0.6763 | 0.7021 | 0.287 / 0.513 | 2.043 | 0.713 | +1.329 | 0.222 | 0 | 3 |
| txf gf s2 | 39, 28, 15, 24 | 16 | 0.6998 | 0.7346 | 0.283 / 0.612 | 10.555 | 2.097 | +8.459 | 0.061 | 0 | 4 |
| mul s0 | 13, 44, 12, 43 | 16 | 0.3032 | 0.3106 | 0.285 / 0.294 | 2.980 | 2.760 | +0.220 | 0.076 | 0 | 2 |
| mul s1 | 24, 16, 19, 3 | 16 | 0.3004 | 0.3089 | 0.286 / 0.293 | 2.788 | 2.458 | +0.330 | 0.079 | 0 | 3 |
| mul s2 | 39, 17, 29, 10 | 16 | 0.3038 | 0.3194 | 0.286 / 0.295 | 3.018 | 2.793 | +0.225 | 0.076 | 0 | 1 |

### 3.3 n_f = 8

| run (abbrev.) | fund. | m | family | top-m | null mean / q95 | odd | even | o−e | fpf | coll | ovl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp 0.3 s0 | 3, 37, 44, 47, 38, 34, 13, 26 | 30 | 0.8009 | 0.8394 | 0.536 / 0.648 | 1.745 | 0.803 | +0.942 | 0.292 | 5 | 9 |
| mlp 0.3 s1 | 6, 16, 29, 50, 39, 20, 2, 3 | 31 | 0.7775 | 0.8316 | 0.553 / 0.660 | 2.227 | 1.233 | +0.994 | 0.241 | 4 | 7 |
| mlp 0.3 s2 | 8, 53, 44, 49, 1, 10, 26, 25 | 30 | 0.8095 | 0.8469 | 0.538 / 0.646 | 2.807 | 1.614 | +1.193 | 0.213 | 3 | 9 |
| mlp gf s0 | 13, 43, 37, 49, 45, 7, 6, 29 | 30 | 0.7206 | 0.7825 | 0.534 / 0.618 | 2.728 | 1.635 | +1.094 | 0.193 | 2 | 9 |
| mlp gf s1 | 30, 31, 38, 22, 26, 53, 52, 51 | 31 | 0.7371 | 0.7962 | 0.553 / 0.632 | 2.465 | 1.216 | +1.249 | 0.213 | 4 | 9 |
| mlp gf s2 | 44, 8, 29, 20, 47, 46, 25, 11 | 31 | 0.7478 | 0.8031 | 0.553 / 0.637 | 3.061 | 1.503 | +1.557 | 0.184 | 4 | 8 |
| txf 0.3 gf s0 (never gen.) | 3, 19, 43, 17, 39, 23, 45, 33 | 32 | 0.6816 | 0.7421 | 0.571 / 0.638 | 3.631 | 2.959 | +0.672 | 0.147 | 4 | 9 |
| txf 0.3 s0 | 18, 3, 45, 11, 47, 8, 44, 53 | 30 | 0.8949 | 0.9153 | 0.536 / 0.759 | 1.531 | 0.433 | +1.098 | 0.354 | 4 | 9 |
| txf 0.3 s1 | 30, 9, 32, 56, 52, 26, 38, 13 | 31 | 0.8527 | 0.8698 | 0.554 / 0.772 | 1.476 | 0.402 | +1.074 | 0.344 | 5 | 11 |
| txf 0.3 s2 | 55, 20, 3, 39, 26, 10, 40, 34 | 31 | 0.9149 | 0.9319 | 0.548 / 0.780 | 3.196 | 1.582 | +1.614 | 0.218 | 5 | 8 |
| txf gf s0 | 26, 42, 53, 21, 2, 1, 19, 39 | 31 | 0.7828 | 0.8007 | 0.552 / 0.717 | 2.602 | 1.053 | +1.550 | 0.217 | 6 | 7 |
| txf gf s1 | 9, 16, 40, 25, 19, 22, 53, 2 | 32 | 0.8196 | 0.8472 | 0.576 / 0.775 | 2.242 | 0.963 | +1.280 | 0.253 | 2 | 11 |
| txf gf s2 | 39, 28, 15, 24, 36, 2, 52, 16 | 32 | 0.8348 | 0.8621 | 0.565 / 0.799 | 8.533 | 5.544 | +2.989 | 0.088 | 0 | 11 |
| mul s0 | 13, 44, 12, 43, 47, 33, 24, 46 | 32 | 0.5867 | 0.6030 | 0.571 / 0.580 | 2.967 | 2.100 | +0.866 | 0.148 | 2 | 9 |
| mul s1 | 24, 16, 19, 3, 29, 46, 10, 54 | 32 | 0.5828 | 0.5989 | 0.572 / 0.579 | 2.913 | 2.526 | +0.387 | 0.149 | 2 | 11 |
| mul s2 | 39, 17, 29, 10, 42, 8, 7, 11 | 32 | 0.5875 | 0.6064 | 0.571 / 0.582 | 3.040 | 2.310 | +0.730 | 0.145 | 4 | 10 |

### 3.4 Group means (mean ± sample std over seeds)

| group | n_f | family | top-m | o−e |
|---|---|---|---|---|
| txf add 0.3 plain (3) | 2 | 0.528 ± 0.093 | 0.734 ± 0.066 | +1.57 ± 1.09 |
| mlp add 0.3 plain (3) | 2 | 0.333 ± 0.015 | 0.436 ± 0.009 | +1.35 ± 0.34 |
| txf add 0.5 gf (3) | 2 | 0.484 ± 0.110 | 0.590 ± 0.063 | +12.2 ± 10.9 |
| mlp add 0.5 gf (3) | 2 | 0.254 ± 0.011 | 0.352 ± 0.003 | +1.21 ± 0.08 |
| txf mul 0.5 gf (3) | 2 | 0.154 ± 0.000 | 0.161 ± 0.004 | +0.32 ± 0.22 |
| txf add 0.3 plain (3) | 4 | 0.774 ± 0.054 | 0.822 ± 0.055 | +1.60 ± 0.94 |
| mlp add 0.3 plain (3) | 4 | 0.549 ± 0.016 | 0.642 ± 0.007 | +1.38 ± 0.73 |
| txf add 0.5 gf (3) | 4 | 0.665 ± 0.041 | 0.693 ± 0.048 | +5.90 ± 3.97 |
| mlp add 0.5 gf (3) | 4 | 0.453 ± 0.009 | 0.562 ± 0.007 | +1.39 ± 0.31 |
| txf mul 0.5 gf (3) | 4 | 0.303 ± 0.002 | 0.313 ± 0.006 | +0.26 ± 0.06 |
| txf add 0.3 plain (3) | 8 | 0.888 ± 0.032 | 0.906 ± 0.032 | +1.26 ± 0.30 |
| mlp add 0.3 plain (3) | 8 | 0.796 ± 0.017 | 0.839 ± 0.008 | +1.04 ± 0.13 |
| txf add 0.5 gf (3) | 8 | 0.812 ± 0.027 | 0.837 ± 0.032 | +1.94 ± 0.92 |
| mlp add 0.5 gf (3) | 8 | 0.735 ± 0.014 | 0.794 ± 0.011 | +1.30 ± 0.24 |
| txf mul 0.5 gf (3) | 8 | 0.586 ± 0.003 | 0.603 ± 0.004 | +0.66 ± 0.25 |

### 3.5 Descriptive properties of the family/shape numbers as measured

These are statements about the numbers in §3.1–3.4, not about H3.

1. `family_fraction ≤ matched_top_m_fraction` on all 48 (run, n_f) cells, as the code asserts by
   construction; the difference ranges from −0.299 (canonical run, n_f = 2) to −0.004 (`mul`, n_f = 2).
2. `odd_share` is **greater than 1** in 47 of 48 cells (minimum 0.940, `txf 0.3 s1`, n_f = 2; maximum
   27.39, `txf gf s2`, n_f = 2): the "odd harmonics" carry more power than the fundamentals they are
   referred to. `even_share` ranges 0.18–5.54. `fundamental_power_fraction` ranges 0.019–0.354.
3. `odd_minus_even` is positive in all 48 cells, including all nine `mul` cells whose spectra are flat
   (PR ≥ 55.5 / 56): minimum +0.123 (`mul s2`, n_f = 2). The sign of the statistic therefore does not
   separate a near-uniform spectrum from a concentrated one on these runs.
4. The greedy rule frequently selects a fundamental that is itself weak because one of its aliased
   harmonics is strong. From the JSON field `fundamental_ranks` (rank = position among the 56 by power):

   | run | selected k | rank of k (share) | strong family member |
   |---|---|---|---|
   | txf 0.3 s0 (canonical) | 3 | 19 (0.6 %) | alias(5·3) = 15, rank 2 (15.4 %) |
   | txf gf s2 | 39 | 47 (0.6 %) | alias(3·39) = 4, rank 1 (29.5 %) |
   | txf gf s2 | 28 | 9 (1.4 %) | alias(3·28) = 29, rank 2 (19.9 %) |
   | txf gf s0 | 26 | 28 (1.0 %) | alias(5·26) = 17, rank 3 (9.3 %) |
   | txf gf s0 | 42 | 16 (1.4 %) | alias(5·42) = 16, rank 1 (12.9 %) |
   | mlp gf s0 | 43 | 26 (1.4 %) | alias(7·43) = 38, rank 2 (5.7 %) |
   | txf 0.3 gf s0 | 19 | 24 (1.6 %) | alias(5·19) = 18, rank 2 (5.7 %) |

   The arithmetic behind this is verified numerically in the JSON (`alias_bijection_p113_j2to7`):
   for p = 113 and every j ∈ {2,…,7}, k ↦ alias(j·k) is a permutation of {1,…,56}. Hence **every**
   frequency f is the aliased 3rd, 5th and 7th harmonic of exactly one k each; "f is a harmonic of
   some fundamental" is always true in Z_113, and which k is called the fundamental is decided by the
   greedy gain, not by anything in the spectrum at f itself. Item 2 above (shares ≫ 1) is the direct
   consequence: the shares are ratios to the power at those chosen k.
5. Collisions among *greedily* selected fundamentals (column coll) are 0–1 at n_f = 2, 0–3 at n_f = 4
   and 0–6 at n_f = 8; the overlap between the odd and the even index sets (column ovl) grows to 7–11
   indices at n_f = 8, where the odd set has 22–24 indices and the even set 15–21 (JSON `index_sets`):
   a third to a half of the odd set is also in the even set, and the two sets differ in size on every
   run at n_f = 8.

## 4. The failing `test_core.py` check, reproduced

`python test_core.py` (from `training/`, same venv) stops at

```
[PASS] sinusoids put ~no energy at harmonics
[FAIL] odd-minus-even separates square from sinusoid
```

(`check` raises `SystemExit(1)` at the first failure, so the remaining harmonic-family assertions and
the whole `check_mlp_mechanism` / `check_config_additions` groups do not run.) Re-run on 2026-09-03 at
HEAD `aa536d2`: the same three `[PASS]` lines precede the same `[FAIL]` line, exit code 1.

The synthetic embeddings of `check_harmonic_families` were rebuilt with the identical generator and draw
order (`np.random.default_rng(0)`; fundamentals `FUND = [18, 15, 11, 1, 13, 22, 56, 36]`; four random
phases per fundamental; `sign(sin(·) + 1e-12)` for the square wave; the noise array is drawn even at
noise = 0). Measured `harmonic_shape` outputs:

| waveform | noise | odd_share | even_share | odd − even | fundamental_power_fraction |
|---|---|---|---|---|---|
| sinusoid | 0.0 | 0.0000 | 0.0000 | +0.0000 (2·10⁻²⁹) | 1.0000 |
| sinusoid | 0.6 | 0.2036 | 0.2220 | −0.0184 | 0.6368 |
| square | 0.0 | 0.1745 | 0.0827 | +0.0918 | 0.8250 |
| square | 0.6 | 0.2886 | 0.2071 | +0.0815 | 0.6392 |

The square/noise-0 line reproduces `docs/BASELINE.md` (`odd_minus_even = 0.0918`, `even_share = 0.083`).
Status of each assertion in that block against these values:

| assertion (test_core.py) | value | passes |
|---|---|---|
| square odd_share > 0.15 | 0.1745 | yes |
| \|square odd_share − 0.1715\| < 0.12 | 0.0030 | yes |
| \|sinusoid odd−even\| < 0.02 | 0.0000 | yes |
| square odd−even > sinusoid odd−even + 0.10 (noise 0) | 0.0918 vs 0.1000 | **no** |
| square odd−even > sinusoid odd−even + 0.10 (noise 0.6) | 0.0815 vs −0.0184 + 0.10 = 0.0816 (margin 0.0999) | **no** (by 1·10⁻⁴; not printed because the run stops earlier) |
| sinusoid even_share rises with noise | 0.2220 > 0.0000 | yes |

The `harmonic_shape` docstring says of the difference: "Measured on synthetic embeddings it stays ~+0.16
for square waves and ~0 for sinusoids across noise levels that swamp the raw shares." On the test's own
synthetic data the measured values are +0.092 and +0.082, not ~+0.16.

## 5. Diagnosis (descriptive)

### 5.1 A discrete ±1 square wave at odd p carries even-harmonic energy — but very little

Spectrum of `sign(sin(2π·1·n/113) + 1e-12)`, n = 0..112, in the project basis (JSON key
`square_wave_k1_p113`):

| quantity | value |
|---|---|
| number of +1 / −1 samples | 57 / 56 (sample mean 0.00885) |
| constant-mode share of all power | 7.8·10⁻⁵ |
| fundamental (k = 1) share of frequency power | 0.8107 |
| all odd harmonics j = 3..55, share of frequency power | 0.1837 |
| all even harmonics j = 2..56, share of frequency power | 0.0056 |
| j = 3, 5, 7 relative to the fundamental | 0.11117, 0.04006, 0.02047 (sum **0.17170**; continuous 1/j² law: 0.11111, 0.04000, 0.02041, sum 0.17152) |
| j = 2, 4, 6 relative to the fundamental | 0.000193, 0.000194, 0.000195 (sum **0.00058**; continuous: 0) |
| `harmonic_shape(power, [1], 113)` | odd 0.17170, even 0.00058, odd − even 0.17112 |

Why the even harmonics are non-zero: a two-level sequence has zero even harmonics iff it is half-wave
antisymmetric, x[n + p/2] = −x[n], which needs p/2 to be an integer. At odd p the +1 and −1 runs have
lengths 57 and 56, and the DFT power of a two-level ±1 sequence with a +1 run of length L is
|X[m]|² = 4 sin²(πmL/p) / sin²(πm/p) (checked against the measured spectrum: maximum deviation
1.3·10⁻¹⁶ in fraction units). With L = 57 = (p+1)/2 this gives ≈ 1 for even m and ≈ (2p/πm)² for odd m,
i.e. each even harmonic carries ≈ (π/2p)² ≈ 1.9·10⁻⁴ of the fundamental — the values in the table.
Contrast: the same construction at even p = 112 with balanced 56/56 runs (numpy `rfft`, JSON key
`even_p_contrast_fft`) puts 5·10⁻³³ of the power at even m.

Two further measured facts: the shares are the same for all eight test fundamentals in isolation
(18, 15, 11, 1, 13, 22, 56, 36 each give odd 0.17170 / even 0.00058 — because n ↦ k·n mod p permutes
the sample grid, the spectrum of a k-wave is a permutation of the 1-wave's), and they do not depend on
the phase (four phases in 0..3.3 rad, 56 or 57 positive samples: identical to 15 digits).

So the intrinsic discrete-grid even-harmonic energy is ~0.06 % of the fundamental. It cannot produce
`even_share = 0.083`.

### 5.2 Where the measured even_share = 0.0827 comes from (exact attribution)

Power adds exactly over embedding columns, each column of the noise-free square test is one square
wave at a known fundamental, and for prime p every index is the aliased j-th harmonic of that column's
fundamental for exactly one j ∈ 1..56. The JSON field `square_noise0.0_attribution` therefore splits
every share by harmonic order of its source (denominator = power at the eight fundamentals, as in
`harmonic_shape`):

| index set | intrinsic even (j even) | odd j ≤ 7 of *another* fundamental | odd tail j ≥ 9 | total |
|---|---|---|---|---|
| even set (18 indices) → `even_share` | 0.0022 | **0.0633** | 0.0172 | 0.0827 |
| odd set (17 indices) → `odd_share` | 0.0017 | 0.1612 | 0.0116 | 0.1745 |
| overlap (7 indices in both sets) | 0.0007 | 0.0633 | 0.0044 | 0.0684 |
| even-only (11 indices) | 0.0015 | 0.0000 | 0.0128 | 0.0143 |
| odd-only (10 indices) | 0.0010 | 0.0979 | 0.0072 | 0.1061 |
| the 3 fundamentals 13, 36, 22 hit by 7·18, 7·11, 7·13 | 0.0003 | **0.0075** | 0.0028 | (own power 0.3685) |

Reading the first row: of the 0.0827 "even" share, 0.0633 (77 %) is the *odd* 3rd/5th/7th-harmonic
power of other fundamentals landing on the same index, 0.0172 (21 %) is the j ≥ 9 odd tail, and
0.0022 (3 %) is genuine even-harmonic energy of the discrete wave. The seven overlap indices with
their generating pairs (from `index_sources_j1to7`):

| index | in odd set as | in even set as |
|---|---|---|
| 3 | 3·1, 5·22 | 6·56 |
| 5 | 5·1, 3·36 | 6·18 |
| 23 | 5·18 | 6·15 |
| 26 | 7·36 | 2·13 |
| 41 | 7·22 | 4·18, 2·36 |
| 47 | 3·22 | 6·11 |
| 53 | 7·56 | 4·15 |

The j ≥ 9 tail is unavoidable under aliasing: `test_core.py` itself asserts that the uncapped odd
family of a single fundamental covers all 56 frequencies, so the 9th, 11th, … harmonics
(1/81, 1/121, … of a fundamental each) of eight fundamentals fall on every index, including the
"even" ones.

### 5.3 Why `odd_share` is depressed and why the difference collapses

The eight square waves put 0.1717 of their fundamentals' power into their own j = 3, 5, 7 harmonics
(§5.1). Relative to the shape statistic's denominator this is 0.1687, because that denominator (power
*at the fundamental indices*) also absorbs 1.7 % foreign harmonic power (e.g. 7·18 on index 13). Of the
0.1687:

* **0.0075 is discarded**: the harmonics 7·18 → 13, 7·11 → 36, 7·13 → 22 land on other fundamentals and
  `harmonic_shape` removes fundamentals from both index sets (`- base_idx`).
* **0.0633 is cancelled**: it sits on the seven overlap indices, is counted in `odd_share` *and* in
  `even_share`, and therefore contributes zero to `odd_minus_even`.
* 0.0979 survives as odd-only power.

Together with the tails: `odd_minus_even` = (0.1061 + 0.0684) − (0.0143 + 0.0684) = 0.0918. Had the
overlap not been double-counted, the same spectrum would give 0.1745 − 0.0143 = 0.160; had the three
harmonics on fundamentals not been discarded, a further 0.0075. The gap to the assertion's 0.10 margin
is thus produced by index-set bookkeeping, not by the waveform.

### 5.4 Noise does not cancel exactly in the difference

For the sinusoid at noise 0.6 the odd set has 17 indices and the even set 18. Measured per-index share:
0.01198 (odd) vs 0.01233 (even), i.e. equal within 3 %, and the difference −0.0184 is ≈ one index's
worth of noise. The claim that broadband noise "cancels" in `odd_minus_even` holds only when the two
index sets have equal size, which collisions break.

### 5.5 Correction to the one-line diagnosis in `docs/BASELINE.md`

`docs/BASELINE.md` ("Test suite at baseline") attributes the failure to two causes and writes of the
second: "a discrete square wave sampled at odd `p` is not exactly antisymmetric, so it carries
even-harmonic energy (`even_share = 0.083`)". The attribution in §5.2 shows that this effect
contributes 0.0022 of the 0.0827; the remainder is aliasing of odd-harmonic power (0.0633 from j ≤ 7 of
other fundamentals, 0.0172 from the j ≥ 9 tail). The first cause named there (harmonics aliasing onto
other fundamentals, 0.0075) is confirmed but is the smaller of the aliasing effects; the overlap
double-counting (0.0633) was not named.

## 6. Properties a valid harmonic-shape statistic must have

Listed as requirements only; nothing is designed or implemented here. Each follows from a measured
fact above.

1. **Symmetric, single-count handling of collisions.** An index that is an odd harmonic of one selected
   fundamental and an even harmonic of another must be assigned to exactly one side (or to neither, or
   weighted by an explicit rule) — never to both. Seven such indices carry 0.0633 of the ideal signal
   and cancel it (§5.2–5.3). Harmonics that land on another fundamental must be treated by the same
   declared rule, not silently discarded (0.0075, §5.3). The declared rule must give the same answer
   whichever of two colliding fundamentals is listed first.
2. **Invariance to which fundamentals are selected.** For prime p every frequency is the 3rd, 5th and
   7th aliased harmonic of some k (`alias_bijection_p113_j2to7`), so the greedy choice of "fundamental"
   is a modelling decision with no counterpart in the spectrum. A valid statistic must be a function of
   the spectrum and a frequency set fixed by a declared rule, must not change when a family is re-labelled
   by a different member, and must not be a ratio to the power at the selected fundamentals (§3.5, items
   2 and 4: shares of 27.4 with the "fundamentals" holding 1.9 % of the power).
3. **A known value on an ideal discrete square wave at the same p.** The reference must be computed for
   the discrete grid, the index set and the cap actually used — for p = 113, j = 3, 5, 7: 0.17170 and
   j = 2, 4, 6: 0.00058 (§5.1) — not the continuous constant 0.1715 with even energy 0, and it must be
   reproduced by the statistic on a collision-free single wave *and* on a set of waves with the
   collision pattern of the real fundamental set.
4. **Exact noise cancellation.** The two compared index sets must have equal cardinality or be normalized
   per index, so that white noise contributes zero to the difference (§5.4: 17 vs 18 indices gives
   −0.018 at noise 0.6, the size of the failing margin).
5. **A declared home for the j ≥ 9 tail.** The uncapped odd family covers all 56 indices; the tail
   (0.0116 + 0.0172 of the fundamentals' power here) must be assigned by rule, not left to fall on
   whichever side it aliases to.
6. **Zero on sinusoids under every collision pattern**, which the current bookkeeping does achieve
   (2·10⁻²⁹ at noise 0) and which must be preserved.
7. **A null that a flat spectrum cannot pass.** On the three `mul` runs (PR ≥ 55.5 / 56) the current
   difference is +0.12 to +0.87 (§3.5, item 3). The statistic needs a null distribution (frequency
   permutation, phase randomization, or the random-set null already used for concentration) against
   which a near-uniform spectrum scores as null, plus the degenerate-case guard
   (`fundamental_power_fraction` threshold) declared before use.
8. **Full reporting.** The odd set, even set, discarded and overlapping indices, and the selection path
   must be emitted with every value, as the JSON of this audit does, so that any number can be
   re-derived from the spectrum.

## 7. Open issues and limits of this audit

* Snapshots differ in meaning across runs (at-transition for the six un-accelerated `frac0.3` runs,
  a few log-grid steps past it for the Grokfast runs, step 4000 for the never-generalized run); no
  number here is corrected for that.
* Only `W_E` is analysed (`docs/RESEARCH_SPEC.md` §3.3 flags this as a defect of the legacy analysis);
  effective MLP weights are out of scope for this audit.
* The `mul` runs are tabulated because the task asked for all 16 runs; they are reserved for the human
  author and are not discussed.
* The `harmonic_shape` docstring's "~+0.16" claim is not reproduced (§4); whether it holds for a
  collision-free eight-fundamental set was not tested, since that would be a new experiment.
* The order-only discrepancy in the canonical key set (§2) was noted, not investigated.
* `training/audit_legacy_metrics.py` is a new file in the tree; it duplicates nothing in
  `grokverse/` but is not covered by `test_core.py`.
* This document and its JSON were committed as work in progress (`d7b74bc`, "unreviewed") after the
  session-limit interruption recorded in `docs/LABBOOK.md`; the adversarial reviews scheduled there
  have not run on it. The one reviewer artifact in `docs/data/`
  (`reviewer_A_preactivation_check.json`) concerns `docs/CAPACITY_AND_CONFOUNDS.md`, not this audit.

## 8. Verification record (2026-09-03, AI, not a human review)

What was re-checked when this document was finalized, so a reviewer knows what has and has not been
looked at:

* `training/audit_legacy_metrics.py` re-executed at HEAD `aa536d2`; the regenerated JSON was compared
  field by field with the 2026-09-02 file: no difference outside `meta` (§0).
* Every aggregate statement in §1–§3 (cap binds on 16/16, n90 ∈ [23, 50], 101 collisions over the 16
  legacy sets, `family ≤ top-m` on 48/48 cells with differences in [−0.299, −0.004], `odd_share > 1`
  on 47/48, `even_share` ∈ [0.18, 5.54], `fpf` ∈ [0.019, 0.354], `odd_minus_even > 0` on 48/48 with
  minimum +0.123, the per-n_f ranges of `coll`, `ovl` and index-set sizes, and every group mean ± std
  in §1.1 and §3.4) was recomputed from the JSON and matches the text.
* The §4 table, the §5.1 square-wave numbers, the §5.2 attribution table, the §5.4 per-index shares
  and the phase / isolated-fundamental / bijection / even-p entries were read back from the JSON and
  match the text. The §5.3 figures 0.1687 and "1.7 % foreign power" (not stored in the JSON) were
  recomputed directly: own j = 1 power / denominator = 0.98268, own j = 3,5,7 power / denominator =
  0.16873.
* Quotations were checked against their sources: the `dominant_frequencies` docstring and provenance
  comment (`analysis/fourier.py`), `docs/RESEARCH_SPEC.md` §3.2, the two `docs/BASELINE.md` passages,
  and the "~+0.16" sentence of the `harmonic_shape` docstring (`analysis/fourier.py` line 182).
* The reported legacy numbers in §1.1 were located verbatim in `RESULTS.md` (§3 table and the
  "~76 % / ~32 %" sentence), `README.md` (76 % / 32 %, 0.73 vs 0.44) and `PROGRESS.md`
  (52 % / 60 %); the canonical key set `[18, 15, 11, 1, 13, 56, 22, 36]` was read from
  `training/runs/txf_add_p113_wd1.0_frac0.3_seed0/progress_measures.json` (`key_frequencies`).
* Not checked: nothing outside `W_E`; no run other than the 16 legacy runs; no new synthetic
  configuration beyond those the script already computes.
