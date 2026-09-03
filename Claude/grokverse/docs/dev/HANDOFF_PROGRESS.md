# Handoff progress — tick as you go, with the evidence path

Companion to `HANDOFF_2026-09-04.md`. Every wake-up or new session reads this first. A tick without a
path (commit hash, file, log line) is not a tick. Three states only: `[x] done — <evidence>`,
`[~] blocked, compute pending — <job, ETA>`, `[!] blocked, human decision — <what>`.

## Stages

- [x] A — `analysis/causal_ablation.py` + tests green; committed — 91 checks; suite 14/14, 1,531; commit below
- [ ] B — INTERFACES §14 spec for `h3_validity` written; module + tests green; committed —
- [ ] C — `analysis/structure_over_time.py` + tests green; committed —
- [x] D — `tests/test_driver.py` 89–92 made durable (asserts the mechanism, not module names); suite green — done with A
- [ ] E1 — freeze commit named in `PREREGISTRATION.md` §12 and the labbook —
- [ ] E2 — `run_analysis_chain.ps1 -SkipWait` launched over the primary block; `results/analysis_chain.log` shows the commit —
- [ ] E3 — training chain finished (`matrix_launch.log`: `matrix chain finished`); `training/results/` committed —
- [ ] E4 — full analysis chain launched over every block; finished —
- [ ] E5 — seed-0 pilot outputs re-run under the freeze commit —
- [ ] F — `aggregate`, `decision_tree`, `figures_study`, `bounded_alternative` + tests green; committed —
- [ ] G1 — aggregate + decision tree run; branch named per architecture in the labbook —
- [ ] G2 — statistics per `STATISTICAL_ANALYSIS_PLAN.md` computed and stored —
- [ ] G3 — control-block comparisons (factorial, parameter-matched, two-hot) —
- [ ] G4 — H3a/H3b/H3c with both controls, refutation criteria applied —
- [ ] G5 — figures from stored files only —
- [ ] G6 — `docs/CLAIM_EVIDENCE_TABLE.md` —
- [ ] G7 — `RESULTS.md`, `README.md`, `docs/LIMITATIONS.md`, `PROGRESS.md` —
- [ ] G8 — `AI_DISCLOSURE.md` per master prompt §22, human parts as placeholders —
- [ ] G9 — `docs/HUMAN_DECISIONS.md` sensitivity notes added, status untouched —
- [ ] H — `docs/dev/EXPLORER_UPDATE_PLAN.md`; labbook, Obsidian, §25 block; final commit —

## Master prompt §23 — Definition of Done (19 items)

| # | item | state | evidence |
|---|---|---|---|
| 1 | mask methodology checked against the primary source | partly: audit done, Nanda/Doshi source notes only partly verified (labbook 30) | `docs/MASK_PROTOCOL_AUDIT.md` |
| 2 | old/new restricted-excluded variants separated | done | `mask_protocols.py`, 197 checks |
| 3 | tests for every load-bearing function | partly: 3 modules + §13 set missing | `tests/run_all.py` |
| 4 | effective MLP weights analysed, not only `W_E` | code done, not run | `mlp_mechanism.py` |
| 5 | hidden activations + output weights of both architectures | code done, not run | `transformer_mechanism.py` |
| 6 | ≥ 2 causal ablations per architecture | module done (9 MLP + 8 transformer ids), not yet run over the matrix | `analysis/causal_ablation.py` |
| 7 | end-to-end logit fit, both architectures, all seeds | module done, run on 2/28 (stage E) | |
| 8 | full-domain comparison over 12,769 pairs | module done, 1 pair of 10 (stage E/G) | |
| 9 | evidence gate evaluated, branch named | **missing** (stage G1) | |
| 10 | primary setting with multiple paired seeds | done | 20/20, shared split hashes |
| 11 | main effects with confidence intervals | **missing** (stage G2) | |
| 12 | parameter count + input parametrization as confounds | docs done; `param_matched`/`twohot` blocks training (stage G3) | |
| 13 | grokking times with measurement interval | in manifests, not reported (stage G2) | |
| 14 | delineation vs Manir/Rupa, Swaroop, Doshi, Khanh; H1/H2 as replication | done | `docs/NOVELTY_AND_RELATED_WORK.md` |
| 15 | every figure reproducible from stored raw data | **missing** (stage G5) | |
| 16 | negative results documented | no results yet (stage G7) | |
| 17 | README claims match the evidence | old numbers `[AUDIT]`-marked; rewrite in stage G7 | |
| 18 | `AI_DISCLOSURE.md` describes the actual process | done for now; update in stage G8 | |
| 19 | final conclusions checked and written by the human authors | `[!]` **human only** | `docs/HUMAN_DECISIONS.md` F + sign-off; `AI_DISCLOSURE.md` placeholders |

## Human-only items (never fill in)

- `docs/HUMAN_DECISIONS.md`: decision columns A–E, status line, section F, sign-off.
- `AI_DISCLOSURE.md`: every `[HUMAN AUTHORS MUST COMPLETE]`.
- The final interpretation in the authors' own words; any threshold change; the `txf_mul_*` runs;
  the explorer update (after review).
