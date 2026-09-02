# PROGRESS.md — GROKVERSE audit trail

Append-only. Newest entry at the top. One block per task-batch / phase, per the format in `PROMPT.md` §9. This is the human-readable audit trail and a draft source for `RESULTS.md` and the pitch.

---

## [2026-08-18] add-vs-mul runs completed — evaluation DELIBERATELY ON HOLD (decision needed: Henrik)
Done: after the wrap-up entry below was written, the backlog's add-vs-mul stretch was started after all
(compute had freed up): 3 seeded Grokfast runs of `(a*b) mod 113` (`--config grokfast --train-frac 0.5
--steps 8000 --task mul --early-stop-acc 0.98`; `--task` CLI flag added to train.py). All 3 grokked.
Measured (raw transitions only): seed 0 memorize 190 / generalize 912 / gap 722 / final test 0.985;
seed 1: 202 / 912 / 710 / 0.989; seed 2: 202 / 1029 / 827 / 0.983. Multiplication groks as reliably and
on a similar timescale as addition in this setting.
**Blocker (per PROMPT §6 — a decision needs Henrik, not a technical failure):** AI_DISCLOSURE.md reserves
the "own extension experiment" Eigenleistung slot with add-vs-mul as its named example. Running the
training was AI-side; the scientifically interesting part — Fourier-sparsity of the mul embeddings,
whether mul finds an additive-Fourier-sparse circuit or a different representation, the comparison
writeup — is exactly the part that could be Henrik's documented own work. Options:
1. Henrik takes over the full Auswertung/Interpretation of these runs himself (question, analysis,
   RESULTS section in his words) → the runs become the substrate of his Eigenleistung.
2. The AI completes the analysis as a disclosed AI contribution, and Henrik picks a different
   experiment as his own (wd/frac sweep or multi-prime, both in IDEAS_BACKLOG / RESULTS §5 Next).
Until decided: no analysis, no RESULTS/§3 changes, no export of the mul runs into the explorer.
Raw artifacts: `training/runs/txf_mul_p113_wd1.0_frac0.5_gf2.0_seed{0,1,2}/` (run.json + embeddings.npy).
Next: Henrik's call between option 1 and 2.

**RESOLVED [2026-08-18, later]:** Option 1 — Henrik evaluates the mul runs himself as his
Eigenleistung (question refinement, spectra/sparsity evaluation, interpretation, writeup).
Disclosure notes: Henrik delegated this choice to the AI ("entscheide du für mich"), and the
methodological hint (additive-Fourier basis vs. the multiplicative group / discrete-log structure)
originated from the AI — both to be named as support in AI_DISCLOSURE/the submission, like a
mentor's hint. The AI continues NOT to touch the mul analysis/RESULTS/export until Henrik's
evaluation exists. The wd/frac sweep and multi-prime remain in IDEAS_BACKLOG, unclaimed.

## [2026-08-18] Un-accelerated cross-architecture experiment + batch wrap-up (FINAL for this session)
Done:
- **Item 2 (closes the other RESULTS §5 caveat):** reran the Phase 4 cross-architecture experiment UN-ACCELERATED at `frac=0.3`, `steps=40000`, early-stop 0.95, seeds 0–2 per architecture (canonical seed-0 transformer reused; 5 new runs, ~4 h CPU total). `compare.py --frac 0.3 --unaccelerated` aggregation + new figure `crossarch_comparison_unaccelerated.png`.
- RESULTS §3 rewritten as a two-setting comparison; §5 Solid/Fragile/Next updated; README updated.
- **Pitch rewritten** around the canonical run (script.md + shot_list.md): headline numbers are now the un-accelerated 145/8367/8222; the stale "un-accelerated run pending" self-critique replaced by the real two-setting cross-arch critique; the environment-dependent determinism scalar retired as a credibility beat in favor of "re-train recovers 145/8367 exactly".
- Reviewer pass (python-reviewer + react-reviewer) on the whole diff before committing; all CRITICAL/HIGH findings fixed: export now refuses progress measures whose re-train mismatched the recorded run; experiment.py grids skip-and-continue on config collisions (--force added); fetches aborted on run switch; LiveLab test-bar comment corrected (0.9 is a deliberate relaxation for the tiny mod-23 model, offline bar is 0.95); chart paths memoized; contract validation extended.
Measured results (mean ± sample std, 3 seeds, all runs grokked):
- **Un-accelerated frac=0.3:** transformer generalizes **7676 ± 1196** (seeds: 8367/6295/8367), MLP **9883 ± 410** (9646/10357/9646) — MLP later in EVERY seed (zero overlap) but only **~1.3×**, vs ~2.9× in the Grokfast/frac=0.5 setting → the speed-gap *direction* is robust, its *magnitude* is setting-dependent (reported as such in RESULTS §3).
- **Sparsity is the robust finding:** top-8 embedding Fourier power **0.73 ± 0.07 (transformer) vs 0.44 ± 0.01 (MLP)** un-accelerated — same direction as accelerated (0.59 vs 0.35), gap never overlapping across seeds.
- Explorer now serves **13 runs** (2.1 MB data): 3 un-accelerated txf + 3 un-accelerated mlp + 6 Grokfast + 1 labeled non-grokked control; guided tour computes its architecture-comparison copy live from index.json (currently "~1.2× slower under identical conditions", 9,646 vs 8,367 — matched un-accelerated runs).
Verification: `pnpm build` clean; `node verify.mjs` ALL WEB CHECKS PASSED (13 runs, panel logic, LiveLab grok gap 2100, tour, zero console errors); `test_core.py` 23/23; determinism PASS; every RESULTS number re-derived from run.json/progress_measures.json this session.
Audit disposition (fixed vs logged): 30 confirmed findings fixed in-session (see previous entry); logged to IDEAS_BACKLOG instead of fixed: per-step Fourier spectrum view, per-split (Nanda-exact) restricted/excluded losses, steps-in-run_id rename. Left parked per instructions: KAN, multi-prime, audio. Backlog item "restricted/excluded live overlay" promoted and shipped. Stretch item 4 (add-vs-mul) NOT started — the un-accelerated cross-arch rerun consumed the compute budget; it remains the top backlog candidate.
Open questions / risks: n=3 keeps the un-accelerated timing intervals wide (±1196); the two settings differ in both acceleration and train_frac, so the ratio change is not attributed to a single knob (honest limit in §3). (react-reviewer's remaining HIGH — no ESLint — was fixed too: eslint + eslint-config-next added, `pnpm lint` passes with zero warnings, hook rules now enforced.)
Next: RESULTS §5 "Next" now points to: a wd/frac sweep to locate where the cross-arch speed gap grows; add-vs-mul; static deploy; per-split losses.

## [2026-08-18] Improvement audit + canonical progress measures + explorer overlay
Done:
- **Step 0:** committed the pending restricted/excluded-loss work after spot-checking its numbers against `progress_measures.json` (all matched).
- **Improvement audit (30 confirmed findings; 5-dimension adversarially-verified review of claims/python/react/UX/robustness).** Fixed directly:
  - *Claims:* "train acc → 1.000 at step 145" was the ≥0.99 crossing (acc 0.9992; exact 1.0 at step 156) — relabeled in RESULTS/README. Stale determinism scalar (documented `2449.995445…` vs measured `2449.9951895352006`) — re-measured, reworded as per-environment bit-exactness, `torch_num_threads` now logged in run metadata. Grokfast sparsity range corrected to 0.52–0.64 (seed 2 = 0.643). Cross-arch spreads switched to sample std (ddof=1): 750±97 / 2163±72 etc. Attention numbers now attributed to the gf run.
  - *Python:* config-collision guard (run_id doesn't encode `steps`; the gf run dir held a 4000-step measure beside an 8000-step run.json — regenerated at matched `--steps 8000`, transitions now recover the recorded run **exactly**, stored as `matches_recorded_run`). The docstring's "built-in honesty check" is now actually implemented. `train_frac`/odd-p validation; PYTHONHASHSEED no-op removed; dead code dropped; `compare.py` never renders a never-grokked arch as a zero bar; +12 checks in `test_core.py` (23/23 PASS) covering detect_transition, log_step_schedule, fwd2d/inv2d, mode masks, agg.
  - *Explorer:* stale-fetch race on run switch fixed (last-selected wins); tour re-built data-driven (phases resolved from each run's own curves — the old fraction-based tour showed train acc 0.022 while narrating "memorized", and paired the un-accelerated transformer against a Grokfast MLP while claiming "~3× slower"); FourierView retitled "final spectrum" + live key-frequency meter from the exported per-step `progress_measure`; run selector shows honest labels (arch/frac/wd/seed/Grokfast/un-accelerated/did-not-grok); LiveLab: Adam optimizer leak fixed, frac slider no longer silently inert, plot history thinned (plateau stays visible), "grokked" badge now requires a measured DELAY; `verify.mjs` now fails on console errors and asserts the grok gap; data.ts validates coords length vs meta; reduced-motion respected; error state recoverable.
  - *Pipeline:* `export --all` no longer crashes on an in-progress run dir (skips loudly); index ordering keeps the canonical un-accelerated run as default (a never-exported non-grokked frac-0.3 Grokfast run had slipped in front — it is now exported honestly as a labeled negative control); NaN-free JSON enforced.
- **Item 1 (closes RESULTS §5 caveat):** restricted/excluded loss on the CANONICAL un-accelerated run (`--config nanda --steps 40000 --seed 0 --early-stop-acc 0.95`, matching the recorded protocol exactly).
- **Item 3 (promoted from IDEAS_BACKLOG):** progress-measure panel in the explorer — export.py embeds the measured curves into meta.json (only for runs where the measure exists), new `ProgressMeasuresView` (gray/gold/violet, CVD-validated palette; chance-line ln p; scrubber-synced cursor), verify.mjs asserts the panel appears exactly for runs with the measure and never otherwise.
Measured results:
- Canonical progress measures: re-train recovered transition **exactly 145/8367** (built-in honesty check); reconstruction err 5.9e-12; at convergence full 0.031 / restricted **0.0020** / excluded **10.42** (≫ ln 113 ≈ 4.73); key freqs {18,15,11,1,13,22,56,36} = same set as the recorded final embedding. Restricted loss separates from full at ~memorization and stays below through the whole 8k plateau — circuit forms beneath the memorized solution.
- Regenerated gf measure (steps=8000): transition exactly 179/675; full 0.017 / restricted 0.0002 / excluded 12.62.
- Environment note: re-trains cross the same grid steps but final accs wobble ~0.5% vs the recorded runs (thread-count-level FP divergence over thousands of steps) — documented as a fragile bullet in RESULTS §5.
Verification: `test_core.py` 23/23 PASS; determinism test PASS; `pnpm build` clean; `node verify.mjs` ALL WEB CHECKS PASSED (9 runs, panel shown for canonical + gf, absent elsewhere; LiveLab grokked with measured gap 1300–2075 across runs; zero console errors).
Open questions / risks: un-accelerated cross-arch runs (txf seed 2, mlp seeds 0–2) still training; pitch rewrite pending until those land.
Next: harvest item 2 (un-accelerated cross-arch, 3 seeds/arch), update RESULTS §3/§5, rewrite pitch around the canonical run.

## [2026-06-21] Phase 0 — Foundations (DONE)
Done:
- Verified environment: Python 3.12.10; Node 24 / npm 11 / pnpm 11.5; **no NVIDIA GPU** → CPU-only.
- Scaffolded `grokverse/` per PLAN; git initialized; foundation commit (`ad5f594`).
- `training/`: `.venv` with pinned **torch 2.12.1+cpu, numpy 2.4.6, matplotlib 3.11.0** (requirements.txt frozen).
  Package: config, seed, data, models/transformer (Nanda 1-layer recipe), train, utils, analysis/{fourier,pca,figures}, export, experiment.
- `web/`: hand-rolled **Next.js 14 + React Three Fiber 8**; data layer (types, loader) + Explorer
  (EmbeddingView3D, LossView, FourierView, ControlPanel).

Measured results / verification:
- **Determinism test PASS**: identical first-batch logit sum (`2449.995445449221`) across two seeded builds.
- `--smoke` trains 1 step cleanly; initial loss 4.74 ≈ ln(113) (correct 113-way uniform baseline).
- `pnpm build` **succeeds** (Compiled successfully, types valid) for the real Explorer.

Open questions / risks:
- CPU step rate ~**172 ms/step** (full-batch, d_mlp=512); benchmarked, not thread-bound → 30k un-accelerated steps ≈ 86 min.

Next: Phase 1 — verify the full grokking transition.

## [2026-06-21] Phase 1 — Reproduce grokking (DONE, 2 seeds)
Done / measured (Grokfast-accelerated, train_frac=0.5, canonical p=113 / d_model=128 dims):
- seed 0 (`txf_add_p113_wd1.0_frac0.5_gf2.0_seed0`): train acc → 1.000 by **step 179**; test crosses 0.95 at
  **step 675**; grok_gap **496**; final test acc **0.983**; test loss 5.9 → 0.05.
- seed 1: train → 1.000 by **step 202**; test crosses 0.95 at **step 859**; grok_gap **657**; final test acc **0.990**.
- Same phenomenon, different timing across seeds — PROMPT §4 acceptance met. Determinism test PASS.
- Un-accelerated `frac=0.3` reference reproduced the memorization plateau directly (train → 1.0 by ~step 156, test
  stuck ~0.05, test loss climbing). Grokfast (arXiv:2405.20233; PROMPT §7) used for in-session runtime; the full
  un-accelerated transition (~10–25k steps, ~30–70 min CPU) is queued for the record.
Verification: logged curves `training/figures/*_curves.png`; step indices above; phenomenon repeats on seed 1.

## [2026-06-21] Phase 3 / 5 / 6 — analysis, export, explorer (partial, verified)
Done / measured:
- **Fourier** (`analysis/fourier.py`): grokked-embedding top-8 frequencies hold **52% (seed0) / 60% (seed1)** of
  power vs **32%** for a non-grokked run — grokking measurably sparsifies the embedding. The *specific* key
  frequencies differ by seed (expected; the sparse structure is robust, the exact frequencies are not). `*_fourier.png`.
- **PCA** (`analysis/pca.py`): final embeddings lie on a periodic **ring** (`*_pca_ring.png`); the same offline
  projection drives the 3D explorer.
- **Attention** (`analysis/attention.py`): the read-out `=` position attends **~50/50 to both operands** a, b
  (per head ≈0.50/0.50, ~0.001 self) over all 12,769 inputs — the `(a+b) mod p` circuit signature. Independently
  re-verified. `*_attention.png`.
- **Progress measure**: key-frequency concentration rises across the transition (curves figure, panel 3).
- **Export** (`export.py`): 2 grokked runs → `web/public/data/` (`meta.json` + `coords.bin` Float32 [T,113,3]) +
  `index.json`; **264 KB** total.
- **Explorer** (`web/`): EmbeddingView3D + LossView + FourierView + ControlPanel — `pnpm build` clean; serves real
  data over HTTP (verified 200 for `/`, `index.json`, `coords.bin`, `meta.json`). Pixel render unverified (no headless browser here).
Open questions / risks:
- Embedding only moderately sparse (early-stop). Restricted/excluded loss (Nanda) not yet implemented (needs per-step checkpoints).
- LiveLab + GuidedTour (Phase 6) and the Phase 4 cross-architecture experiment remain; ≥3-seed Fourier robustness pending.
Next: Phase 2 grid → Phase 4 original experiment (transformer vs MLP) → Phase 6 LiveLab/GuidedTour → un-accelerated canonical run.

## [2026-06-21] Phase 4 — Original experiment: cross-architecture grokking (DONE, 3 seeds each)
Done / measured (1-layer transformer vs 2-layer MLP; mod 113, wd=1.0, frac=0.5, Grokfast; 3 seeds each, all grokked):
- Transformer: generalize **750 ± 79**, gap 559 ± 70, top-8 freq power **0.59 ± 0.05**, final test acc 0.986 ± 0.003.
- MLP: generalize **2163 ± 59**, gap 1959 ± 59, top-8 freq power **0.35 ± 0.003**, final test acc 0.989 ± 0.004.
- **Finding:** the transformer groks **~2.9× faster** AND with a **much sparser/periodic embedding** than the MLP.
  The gap (750 vs 2163) far exceeds per-seed spread (≤80) → robust, not seed noise. Figure: `crossarch_comparison.png`.
- Core correctness suite `test_core.py`: **11/11 PASS** (add+mul labels, determinism, MLP wiring, Fourier
  orthonormality + injected-frequency recovery, PCA shapes).
- Explorer now serves all 6 runs (txf + mlp × 3 seeds), transformer-first; 884 KB total.
Verification: `compare.py` aggregation over 3 seeds/arch (mean ± std); figure saved; finding direction consistent across all seeds.
Open questions / risks: single hyperparameter point (not a sweep); Grokfast-accelerated; restricted/excluded loss pending.
Next: Phase 6 LiveLab + GuidedTour (needs a browser to verify); un-accelerated canonical run to completion; Phase 7 polish.

## [2026-06-21] Phase 6 / 7 — Explorer complete + deliverables (DONE)
Done / measured:
- **LiveLab** (in-browser MLP via TensorFlow.js on `(a+b) mod 23`, decoupled AdamW-style weight decay + sliders):
  empirically **VERIFIED to grok in a real headless browser** (Playwright): train 1.000, **test 0.912 @ step ~2175**.
  Default frac=0.7, wd=2 chosen via a Node weight-decay sweep (groks ~1100–1400 steps; lower frac overfits in budget).
- **GuidedTour** (8-step scripted walkthrough) + Explorer integration (tour + Live-Lab toggle over the 3D scene).
- **verify.mjs** headless e2e: 3D render + 6 runs load + LiveLab groks + tour steps → "ALL WEB CHECKS PASSED".
- `pnpm build` + `pnpm dev` work out of the box (`pnpm-workspace.yaml verifyDepsBeforeRun:false` to clear the
  tfjs/core-js build-script block).
- Docs finalized: README (env setup + reproduce + explorer features), RESULTS (findings + self-critique),
  AI_DISCLOSURE, THIRD_PARTY (tfjs/playwright pinned), pitch/ (script + shot_list). Removed dead code (`seeded_generator`).
Verification: `pnpm build` PASS; `node verify.mjs` PASS (in-browser grok); determinism PASS; `test_core.py` 11/11 PASS.
Open questions / risks: un-accelerated canonical frac=0.3 run launched for the record (early-stop 0.95) — harvest when done.
  Restricted/excluded loss (Nanda) still deferred (needs per-step full-model checkpoints).
Next: harvest the un-accelerated run into RESULTS if/when it groks; optional add-vs-mul variant; static deploy (nice-to-have).

## [2026-06-21] Un-accelerated canonical run — harvested (DONE)
The faithful un-accelerated run (`txf_add_p113_wd1.0_frac0.3_seed0`; **no Grokfast**) **GROKKED**: train → 1.0 by
**step 145**; test crosses 0.95 at **step 8367** — an **8,222-step** memorization plateau, final test acc **0.981**,
in ~24 min on CPU. Its embedding is the sparsest yet: **76%** of Fourier power in the top-8 frequencies
(vs 0.52–0.60 accelerated, 0.32 non-grokked). Exported + figures generated; it is now the explorer's **default** run.
RESULTS §1/§2 and README updated — the main honesty caveats (un-accelerated transition; embedding sparsity) are resolved.
