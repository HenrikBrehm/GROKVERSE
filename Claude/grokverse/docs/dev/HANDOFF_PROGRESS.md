# Handoff progress — tick as you go, with the evidence path

Companion to `HANDOFF_2026-09-04.md`. Every wake-up or new session reads this first. A tick without a
path (commit hash, file, log line) is not a tick. Three states only: `[x] done — <evidence>`,
`[~] blocked, compute pending — <job, ETA>`, `[!] blocked, human decision — <what>`.

## Stages

- [x] A — `analysis/causal_ablation.py` + tests green; committed — 91 checks; suite 14/14, 1,531; commit below
- [x] B — INTERFACES §14 spec written first, then `analysis/h3_validity.py` + tests — 70 checks; suite 15/15, 1,601
- [x] C — `analysis/structure_over_time.py` + tests — 56 checks; suite 16/16, 1,657; driver reports NO missing modules
- [x] D — `tests/test_driver.py` 89–92 made durable (asserts the mechanism, not module names); suite green — done with A
- [x] E1 — analysis code frozen at `0b55e1d`; recorded in `PREREGISTRATION.md` §12 (commit `1e3f646`)
- [x] E2 — finished 2026-09-03T23:00:43Z, exit 1: **196 ok, 84 failed, 20 skipped** (`results/analysis_driver.json`). Failures were 3 module bugs, all now fixed (labbook 56–57, 60): key_frequencies 40 + progress_measures 20 (seed arg), wave_fitting 24 (IndexError ×4, wrong arch declaration ×20)
- [x] E3 — training chain finished 2026-09-04T02:46:36Z; **51 runs, all completed, none failed** (primary 20, confound 18, param_matched 10, two-hot 3); `training/results/` committed
- [x] E4 — every block analysed: primary 280, confound 234 ok/0 failed, controls 181 ok/1 failed (a race, fixed and re-run); aggregate over all 51 runs shows **0 missing**
- [ ] E5 — seed-0 pilot outputs re-run under the freeze commit —
- [x] E6 — re-run at commit `eeeef82`: **80 ok, 0 failed**; primary block now complete, 0 missing rows in every module
- [x] F — `aggregate`, `decision_tree`, `figures_study`, `bounded_alternative` + tests — suite 20/20, 1,794 checks
- [x] G1 — aggregate (0 missing) + gate at both points; **branch: `neither_passes` at final** (G1/G2/G3 10/10 both, G4 0/10 both), `undetermined` at crossing — labbook 62–65, `results/decision_tree_{final,crossing}.json`
- [x] G2 — `analysis/statistics_report.py` + tests; the 8 pre-specified comparisons (10 rows) with per-seed differences, bootstrap CIs, exact tests, effect sizes, Holm aid and the §6 timing bounds — `results/statistics.json`, suite 21/21
- [x] G3 — all three control comparisons done: Grokfast/`train_frac` separated, parameter-matched (differences survive), two-hot (waveform result is partly input parametrization) — `results/controls_report.json`, labbook 90–91, 97–98
- [x] G4b — `analysis/h4_report.py` + tests: structure onset precedes generalization in 10/10 seeds, both architectures — `results/h4_report.json`, labbook 79–81
- [x] G4c — bounded alternative-mechanism analysis: **20 ok, 0 failed**; probes at ceiling vs chance controls, effective rank 12.7 (txf) vs 66.6 (MLP), transformer destroyed by removing its top-16 directions while the MLP is unaffected, cross-seed CKA low in both — `results/aggregate/bounded_alternative.json`, `results/cross_seed_cka.json`, labbook 83–86
- [x] G4 — `analysis/h3_report.py` + tests; both refutation criteria applied literally. **H3 is REFUTED** by criterion 2 (family gap +0.0850, CI [+0.0600, +0.1004]); the family definition closes only 5.4 % of the gap — `results/h3_report.json`, labbook 70–74
- [x] G5 — all 11 figures drawn from the aggregate tables only, captions naming sources — `results/figures/index.json`
- [x] G6 — `docs/CLAIM_EVIDENCE_TABLE.md`: 8 claims in the §21 six-part structure, a forbidden-claims table, and what each pending item would settle
- [ ] G7 — `RESULTS.md`, `README.md`, `docs/LIMITATIONS.md`, `PROGRESS.md` —
- [ ] G8 — `AI_DISCLOSURE.md` per master prompt §22, human parts as placeholders —
- [ ] G9 — `docs/HUMAN_DECISIONS.md` sensitivity notes added, status untouched —
- [~] H — `docs/dev/EXPLORER_UPDATE_PLAN.md` written (web/ untouched; approval box open); labbook, Obsidian, §25 block still to come

## Master prompt §23 — Definition of Done (19 items)

| # | item | state | evidence |
|---|---|---|---|
| 1 | mask methodology checked against the primary source | partly: audit done, Nanda/Doshi source notes only partly verified (labbook 30) | `docs/MASK_PROTOCOL_AUDIT.md` |
| 2 | old/new restricted-excluded variants separated | done | `mask_protocols.py`, 197 checks |
| 3 | tests for every load-bearing function | every driver module done; the §13 set (aggregate/decision_tree/figures) still to come | `tests/run_all.py` (16 files, 1,657) |
| 4 | effective MLP weights analysed, not only `W_E` | code done, not run | `mlp_mechanism.py` |
| 5 | hidden activations + output weights of both architectures | code done, not run | `transformer_mechanism.py` |
| 6 | ≥ 2 causal ablations per architecture | module done (9 MLP + 8 transformer ids), not yet run over the matrix | `analysis/causal_ablation.py` |
| 7 | end-to-end logit fit, both architectures, all seeds | module done, run on 2/28 (stage E) | |
| 8 | full-domain comparison over 12,769 pairs | **done**: 10 paired seeds, median agreement 0.99977 | `results/function_agreement/*_arch25k_*` |
| 9 | evidence gate evaluated, branch named | **done**: `neither_passes` at final, `undetermined` at crossing | `results/decision_tree_*.json` |
| 10 | primary setting with multiple paired seeds | done | 20/20, shared split hashes |
| 11 | main effects with confidence intervals | **done**: 8 comparisons, bootstrap CIs, all seeds shown | `results/statistics.json` |
| 12 | parameter count + input parametrization as confounds | **done**: both measured; differences survive matching, waveform result is partly parametrization | `results/controls_report.json` |
| 13 | grokking times with measurement interval | **done**: interval-consistent bounds, 10/10 seed intervals exclude zero | `results/statistics.json` |
| 14 | delineation vs Manir/Rupa, Swaroop, Doshi, Khanh; H1/H2 as replication | done | `docs/NOVELTY_AND_RELATED_WORK.md` |
| 15 | every figure reproducible from stored raw data | **done**: 11 figures from the aggregate tables only | `results/figures/index.json` |
| 16 | negative results documented | **two so far**: gate `neither_passes` and H3 refuted by its own criterion | labbook 62–63, 70 |
| 17 | README claims match the evidence | old numbers `[AUDIT]`-marked; rewrite in stage G7 | |
| 18 | `AI_DISCLOSURE.md` describes the actual process | done for now; update in stage G8 | |
| 19 | final conclusions checked and written by the human authors | `[!]` **human only** | `docs/HUMAN_DECISIONS.md` F + sign-off; `AI_DISCLOSURE.md` placeholders |

## Human-only items (never fill in)

- `docs/HUMAN_DECISIONS.md`: decision columns A–E, status line, section F, sign-off.
- `AI_DISCLOSURE.md`: every `[HUMAN AUTHORS MUST COMPLETE]`.
- The final interpretation in the authors' own words; any threshold change; the `txf_mul_*` runs;
  the explorer update (after review).
