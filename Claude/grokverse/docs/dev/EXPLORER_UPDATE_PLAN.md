# Explorer update plan — written, not executed

**Status: PLAN ONLY. Nothing in `web/` has been changed.** Written 2026-09-04 by the AI while the
analysis driver was still running, so that the update is one reviewed step later rather than an
improvisation at the end.

Two rules gate this work, and both are still closed:

* `docs/HUMAN_DECISIONS.md` **E3**: *"The explorer stays frozen until the analyses are frozen and
  human-reviewed."* The analysis code is frozen (`PREREGISTRATION.md` §12, commit `0b55e1d`), but the
  **human review has not happened**.
* Master prompt §12 / §24 step 17: the explorer is updated **last**, and *"visualization must not
  drive the research"*.

So this file describes what to do, in what order, and what must not be shown. It ends with the
approval checkbox that unlocks it.

---

## 1. Why the explorer needs changing at all

The explorer today serves **13 runs from the reproduction phase** (`web/public/data/index.json`,
2.1 MB total) — ids like `txf_add_p113_wd1.0_frac0.3_seed0`, without the `_arch25k` suffix. Those are
not the architecture study's runs, and the claims they illustrate are exactly the ones
`docs/CURRENT_EVIDENCE_AUDIT.md` marked. Three specific problems:

1. **It shows the wrong runs.** The study's evidence is the 20 paired-seed `*_arch25k` runs. The
   explorer shows an older, differently-configured set.
2. **It shows an audited claim as a headline.** The "0.73 vs 0.44 top-8 concentration" framing is
   (i) a **replication**, not a discovery — Manir & Rupa 2026 publish the concentration gap
   (`docs/NOVELTY_AND_RELATED_WORK.md`) — and (ii) measured on `W_E` alone, which is the wrong object
   for an MLP that reads it through two halves of `W_in`.
3. **Its progress-measure panel uses the legacy mask.** `docs/MASK_PROTOCOL_AUDIT.md` established that
   the legacy outer-product mask keeps **289** components where the released Nanda operator keeps
   **17**, and deletes 26.3 % of the grid where the published one deletes 16 cells. Whatever the panel
   shows must be labelled with the protocol it used, and the legacy variant must never be called a
   reproduction of Nanda.

## 2. Which runs to export

| block | runs | export? | why |
|---|---|---|---|
| primary `*_frac0.3_seed{0..9}_arch25k`, both architectures | 20 | **yes** | the study's evidence; paired seeds |
| `confound` (Grokfast × `train_frac`) | 18 | only if a panel uses it | supports the §14 factorial, not the main narrative |
| `param_matched` | 10 | no | a control, better shown as a number than a scene |
| `twohot` | 3 | no | ditto |
| `txf_mul_*` | 3 | **never** | reserved for the human author (`HUMAN_DECISIONS` E6) |
| the 13 legacy runs | 13 | **remove** | superseded; keeping them invites the audited comparison |

All 22 completed `*_arch25k` runs already carry `embeddings.npy`, so `training/grokverse/export.py`
runs on them unchanged. Size: `coords.bin` is ~140 KB per run, so 20 runs ≈ **2.8 MB** — the same
order as today's 2.1 MB and well inside the "modest" budget `PLAN.md` sets.

## 3. Which analysis file feeds which panel

Every panel number must trace to a file, exactly as the aggregate tables do. Nothing in the explorer
may be computed in the browser from raw weights.

| panel | today | after the update | source |
|---|---|---|---|
| `LossView` | `run.json` curves | unchanged, plus the seven checkpoint **roles** as markers | `run.json`, `checkpoints.json` |
| `EmbeddingView3D` | PCA coords over time | unchanged (it is honest and it is the best thing in the app) | `*.coords.bin` |
| `FourierView` | top-8 of `W_E` | the **pre-registered key set** and its rule, with the legacy top-8 shown *as* the legacy number | `analysis/key_frequencies/<tag>.json` |
| `ProgressMeasuresView` | legacy-mask restricted/excluded | one line **per named protocol**, `nanda_exact` and `legacy_broad_mask_variant` labelled | `analysis/progress_measures/all_checkpoints.json` |
| *(new)* structure-over-time | — | structured-neuron fraction, phase `R`, key-subspace share vs step | `analysis/structure_over_time/all_checkpoints.json` |
| *(new)* causal panel | — | ablation damage **next to its size-matched control**, never alone | `analysis/causal_ablation/<tag>.json` |
| *(new)* gate panel | — | the four criteria per architecture and the branch taken | `results/decision_tree.json` |
| `ControlPanel` | 13 legacy runs | the 20 primary runs, selectable **as seed pairs** | `web/public/data/index.json` |
| `GuidedTour` | narrates the old claims | rewritten from `RESULTS.md` **after** the branch is known | `docs/CLAIM_EVIDENCE_TABLE.md` |
| `LiveLab` | in-browser tiny MLP | unchanged — it is a teaching device, not evidence, and already says so | — |

## 4. Rules the update must obey

1. **No number the evidence gate has not licensed.** If `decision_tree` reports `undetermined` or
   `neither_passes`, the explorer may not present a Fourier-circuit narrative. The gate panel exists
   partly so that an honest negative result is *visible* rather than hidden by an unchanged UI.
2. **Every ablation damage figure appears with its control.** A bar showing "accuracy drops 0.99"
   without the size-matched control next to it is the single most misleading thing this project could
   ship; `CAUSAL_ABLATION_PLAN.md` §2 forbids it in the analysis and it is forbidden here too.
3. **The legacy numbers keep their labels.** Top-8 on `W_E` is "the legacy metric"; the broad mask is
   "our legacy variant, not a reproduction of Nanda".
4. **Replication is labelled as replication.** H1 and the MLP half of H2 are replications in these
   architectures (`NOVELTY_AND_RELATED_WORK.md`); the tour must not present them as discoveries.
5. **`export.py` keeps `allow_nan=False`.** A diverged run must fail the export loudly rather than
   ship JSON the browser rejects at runtime.
6. **No re-styling before the content is right.** The visual quality bar in `PLAN.md` Phase 6 stands,
   but content correctness comes first; a prettier panel showing an unlicensed claim is worse than an
   ugly one showing a correct one.

## 5. Order of work, once unblocked

1. Human review of the frozen analyses and of `RESULTS.md` → `HUMAN_DECISIONS.md` E3 signed.
2. `python -m grokverse.export --runs "*_frac0.3_seed*_arch25k"`; delete the 13 legacy artifacts;
   rebuild `index.json`. Verify total size and that every id carries `_arch25k`.
3. Extend `web/lib/types.ts` for the new source files (types first — the loader is the contract).
4. `ProgressMeasuresView` and `FourierView` relabelling. These are corrections, not features, and
   should land before any new panel.
5. New panels: structure-over-time, causal, gate.
6. `ControlPanel` seed-pair selection.
7. `GuidedTour` rewritten from the finished `RESULTS.md`.
8. `pnpm build` clean, `node verify.mjs` passes, and a fresh check that every visible number appears
   in a file under `training/runs/*/analysis/` or `training/results/`.

## 6. What this plan deliberately does not decide

* Whether to show the `confound` block at all — that depends on whether §14's factorial result is
  worth a panel, which is a judgement to make once the numbers exist.
* The visual design of the new panels.
* Whether the tour keeps its current eight-step shape.

---

## Approval

The explorer stays frozen until a human author checks this box:

```
[ ] The frozen analyses and RESULTS.md have been reviewed, and the explorer update above is approved.
    Name:                        Date:
    Changes to the plan:
```
