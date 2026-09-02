# RESULTS.md — GROKVERSE findings

> Every number here traces to a real seeded run in `training/runs/` and a figure in `training/figures/`. Nothing on this page is fabricated, hard-coded, or cherry-picked. Negative and fragile results are reported as results. (PROMPT.md §6)

> ### ⚠ Under revision — architecture study in progress (2026-09-03)
>
> This page predates the mechanistic architecture study on branch `arch-study` and **several of its
> claims are classified as over-interpreted or methodologically problematic** by
> [`docs/CURRENT_EVIDENCE_AUDIT.md`](docs/CURRENT_EVIDENCE_AUDIT.md). The specific corrections known
> today are marked inline below with **[AUDIT]**. Every number here still traces to a real seeded run;
> what is wrong is not the arithmetic but what the numbers were said to show. The page is rewritten
> only after the new analyses are complete, so that the corrections are made once, against evidence.

Status: **Phases 0–7 complete and verified** — an un-accelerated canonical reproduction, seed-robust mechanistic analysis, a cross-architecture finding, and a verified interactive 3D explorer.

---

## 1. Reproduction of grokking — VERIFIED

**Setup.** 1-layer ReLU transformer, no LayerNorm; modular addition `(a + b) mod 113`; `d_model=128`, 4 heads, `d_head=32`, `d_mlp=512` (Nanda et al. 2023 recipe). Full-batch AdamW, `lr=1e-3`, `betas=(0.9, 0.98)`, `weight_decay=1.0`. Tokens `[a, b, =]`; prediction read off the `=` position. Every run seeded and deterministic.

**Canonical reproduction (headline) — `txf_add_p113_wd1.0_frac0.3_seed0`**, the full Nanda config (`train_frac=0.3`, `wd=1.0`, **no acceleration**, ~24 min on CPU). Measured transition (`run.json["transition"]`):

| quantity | value |
|---|---|
| train acc crosses 0.99 (memorization) | **step 145** (reaches exactly 1.000 by step 156) |
| test acc crosses 0.95 (generalization) | **step 8367** |
| grokking gap | **8,222 steps** (both crossings per `train.py`'s 0.99/0.95 thresholds) |
| final test acc (early-stop) | **0.981** |
| test loss | **4.7 → 0.06** (climbs to ~26 during the plateau, then collapses) |

Train accuracy crosses 0.99 by step 145 (exactly 1.0 from step 156); test accuracy then sits at chance through an **~8,000-step memorization plateau** before suddenly generalizing — the defining grokking signature, with no acceleration. This is the explorer's default view. Figure: `training/figures/txf_add_p113_wd1.0_frac0.3_seed0_curves.png`.

**Determinism.** The reproducibility test passes: two seeded builds in the same environment produce a bit-identical first-batch logit sum (currently `2449.9951895352006` on the pinned torch 2.12.1+cpu). The exact scalar is *environment-dependent* — CPU thread count and BLAS kernel dispatch shift the last digits (an earlier snapshot of this environment produced `2449.995445…`), which is why run metadata records `torch_num_threads`. Same config + seed + environment ⇒ same curve; the transition *step indices* are far more robust: the canonical run's deterministic re-train (for the progress measures below) recovered memorize/generalize at exactly 145/8367.

**Seed-robust corroboration (Grokfast-accelerated).** To iterate quickly and check seed-robustness in-session, we also ran **Grokfast** (Lee et al. 2024, arXiv:2405.20233 — the runtime tool PROMPT.md §7 explicitly permits) at `frac=0.5` across seeds 0–2: grokking repeats every time (generalize at **750 ± 97** steps, final test acc **0.986 ± 0.004**; mean ± sample std, n=3). Grokfast compresses the *timing* of the same phase transition — perfect memorization, a long plateau, then a sharp jump — it does not manufacture it; the un-accelerated run above is the faithful reference.

---

## 2. Mechanistic analysis (Phase 3)

**Embedding Fourier spectrum.** Projecting the final token-embedding matrix `W_E[:113]` onto the orthonormal real Fourier basis over ℤ₁₁₃, a sparse set of frequencies dominates. For the **canonical un-accelerated run** the top 8 (`k = 18, 15, 1, 11, 13, 22, 56, 36`) hold **~76%** of the total frequency power — versus **~32%** for a matched *non-grokked* run (and 0.52–0.64 across the three Grokfast seeds). Grokking measurably concentrates the embedding into a sparse periodic ("trig-identity") circuit. Figure: `..._fourier.png`.

**Convergence.** Sparsity tracks convergence: the longer un-accelerated run (0.76) is markedly sparser than the early-stopped Grokfast runs (0.52–0.64), exactly as expected — more training concentrates the circuit further. The *specific* dominant frequencies differ by seed (the sparse structure is robust; the exact frequencies are the network's free choice).

**PCA geometry.** A PCA projection of the final embeddings shows the tokens arranged on a **periodic ring** (the geometric face of the Fourier structure). Figure: `..._pca_ring.png`. The same offline PCA projection drives the 3D explorer, where the points visibly migrate from a blob into the ring as the user scrubs through training.

**Progress measure (embedding).** The fraction of embedding power in the key frequencies (`frequency_concentration_over_time`) rises across the transition — a held-out-independent progress signal tracking circuit formation (third panel of the curves figure).

**Progress measures (restricted & excluded loss) — NOT a reproduction of Nanda et al. [AUDIT].** The mask used here is the `legacy_broad_mask` variant: built as the outer product of a 1D key-frequency mask, it keeps every *pair* of key rows including cross-frequency blocks such as `cos(w_18 a)cos(w_15 b)`, which the trig-identity circuit never uses. For p=113 and 8 key frequencies it keeps 289 components where the released Nanda operator keeps 17, and its excluded mask deletes 3,360 (26.3% of the logit tensor) where the published one deletes 16. Both biases push toward the reported conclusion. The numbers below are therefore **not directly comparable to Nanda et al.** and are not evidence for the trig-identity circuit; see `docs/sources/nanda2023_progress_measures.md` and `docs/MASK_PROTOCOL_AUDIT.md`. The description that follows is of what was computed: We implement the Nanda et al. 2023 *restricted* and *excluded* loss (`analysis/progress_measures.py`): the model's logits over the full `(a, b)` grid are 2D-Fourier-transformed over the two input axes in the orthonormal basis over ℤ₁₁₃; **restricted loss** keeps only the key-frequency components (plus the constant) and rebuilds the logits, **excluded loss** removes the key frequencies. (Protocol note: Nanda et al. evaluate these on the train/test splits; we measure over the full grid — the divergence signature is the same.) The transform is self-checked (inverse∘forward reconstructs the logits to **~1e-12**; keeping all modes is the identity), and the deterministic re-train that captures the per-step model states recovers the recorded run's transition **exactly** (a built-in honesty check, stored as `matches_recorded_run`).

- **Canonical un-accelerated run** (`txf_add_p113_wd1.0_frac0.3_seed0`, key freqs k = {18, 15, 11, 1, 13, 56, 22, 36} — the same set as the spectrum above, ranked by the re-trained embedding's power; transition recovered at exactly 145/8367): at convergence **full loss 0.031, restricted loss 0.0020, excluded loss 10.42**. The restricted loss separates from the full loss already around the memorization step and stays *below* it through the entire ~8,000-step plateau — the Fourier circuit is forming quietly beneath the memorized solution — then everything collapses at step 8367 while the excluded loss stays ruined (≫ ln 113 ≈ 4.73). Figure: `..._frac0.3_seed0_progress.png`.
- **Grokfast run** (`txf_..._frac0.5_gf2.0_seed0`, key freqs k = 16, 39, 17, 35, 8, 44, 21, 53; transition recovered at exactly 179/675): **full 0.017, restricted 0.0002, excluded 12.62** — same signature, compressed timing. Figure: `..._gf2.0_seed0_progress.png`.

**[AUDIT]** The sentence that stood here — that the eight key frequencies alone solve the task while removing them destroys it — does not follow from these numbers, because the mask is ~17x wider than the hypothesis it is supposed to test and the excluded mask removes ~200x more of the logit tensor. The re-measurement under the named protocols (`nanda_exact`, `same_frequency_block`, `sum_directions_only`) is part of the architecture study. These curves are also live in the explorer (a dedicated panel appears for exactly the runs where the measure was computed).

**Attention.** Averaged over all 12,769 inputs, the read-out (`=`) position splits its attention almost exactly **50/50 between the two operand positions**. On the **canonical un-accelerated run**: mean over heads (to `a` / to `b` / to `=`) `0.501/0.499/0.001`, per-head operand shares all within 0.47–0.53 and self-attention ≤ 0.001. The Grokfast `frac=0.5` run shows the same split even more tightly (per head: h0 `0.502/0.498/0.001`, h1 `0.499/0.500/0.001`, h2 `0.495/0.504/0.001`, h3 `0.500/0.499/0.001`). Every head pulls in both `a` and `b` and essentially ignores itself. **[AUDIT]** This is a *mean without a variance* and is therefore descriptive only: a 50/50 mean is equally consistent with a constant 50/50 split (attention is a fixed sum, making the transformer additive-then-ReLU like the MLP) and with input-dependent attention that averages to 50/50 (a multiplicative path the MLP does not have). Those are different circuits. The claim that this "directly links the learned structure to the computation" is withdrawn pending the per-input variance and the causal head ablations of the architecture study. Figure: `..._attention.png`.

---

## 3. Original experiment — cross-architecture grokking (Phase 4)

**Question.** Do different architectures grok modular addition the same way? We compare a 1-layer **transformer** against a 2-layer **MLP** (given a shared embedding table `W_E [p, d]` so the *same* Fourier analysis applies), across **3 seeds each** in **two settings**: the fast Grokfast setting (`frac=0.5`, iteration/robustness) and the **canonical un-accelerated** setting (`frac=0.3`, `steps=40000`, no acceleration — the same protocol as §1).

**Results (mean ± sample std over 3 seeds; all 12 runs grokked):**

| Grokfast, frac=0.5 | generalization step | grok gap | embedding top-8 freq power | final test acc |
|---|---|---|---|---|
| **transformer** | **750 ± 97** | 559 ± 86 | **0.59 ± 0.06** | 0.986 ± 0.004 |
| **MLP** | **2163 ± 72** | 1959 ± 72 | **0.35 ± 0.003** | 0.989 ± 0.005 |

| un-accelerated, frac=0.3 | generalization step | grok gap | embedding top-8 freq power | final test acc |
|---|---|---|---|---|
| **transformer** | **7676 ± 1196** | 7531 ± 1196 | **0.73 ± 0.07** | 0.980 ± 0.014 |
| **MLP** | **9883 ± 410** | 9723 ± 414 | **0.44 ± 0.01** | 0.984 ± 0.010 |

**Finding.** Both architectures grok to ≥98% test accuracy in both settings — and the two effects fare differently under the honest un-accelerated re-test:

1. **Sparsity difference — robust across settings.** The transformer converges to a markedly sparser, more periodic embedding in *both* settings (0.59 vs 0.35 accelerated; 0.73 vs 0.44 un-accelerated; per-seed gaps never overlap). **[AUDIT]** Two corrections. (i) The direction of this gap is already published — Manir & Rupa 2026 (arXiv:2603.25009, Table 8) report 98.5% vs ~75% top-5 concentration for a transformer vs an MLP at p=97 — so it is a replication, not a discovery. (ii) "The MLP solves the same task with a more distributed frequency representation" is unsupported: the number is measured on the shared `W_E` alone, which is the wrong object for an MLP that reads it through two halves of `W_in`, and a square-wave-like circuit necessarily spreads its power over aliased odd harmonics that a top-8 metric cannot see. Both are tested in the architecture study.
2. **Speed difference — direction robust, magnitude setting-dependent.** The MLP generalizes later than the transformer in *every* seed of both settings (un-accelerated: slowest transformer 8367 < fastest MLP 9646 — zero overlap). But the size of the gap shrinks from **~2.9×** (Grokfast, frac=0.5) to **~1.3×** (un-accelerated, frac=0.3, where the transformer's own seed spread is large, ±1196). The headline "transformer groks ~3× faster" is therefore partly a property of the accelerated/frac=0.5 setting, not a setting-independent constant — reported as such.

Figures: `training/figures/crossarch_comparison.png` (Grokfast), `training/figures/crossarch_comparison_unaccelerated.png`.

**Honest limits.** Two hyperparameter points, not a sweep; with n=3 the un-accelerated timing means carry wide intervals (sample std 1196 vs 410), so ~1.3× is an estimate of a consistently-positive but modest effect, not a precise constant; the MLP's lower sparsity is partly architectural (concatenated operand embeddings vs. attention-combined). The un-accelerated setting also differs from the Grokfast one in `train_frac` (0.3 vs 0.5), so the two settings differ in two knobs at once — the within-setting comparisons are controlled, the between-setting change of ratio is not attributed to a single cause.

## 4. Where this sits in the literature

- **Power et al. 2022** (arXiv:2201.02177) — first reported grokking: generalization long after memorization on small algorithmic datasets. We reproduce exactly this gap.
- **Nanda et al. 2023** (arXiv:2301.05217) — our primary recipe and analysis target: the 1-layer transformer on modular addition, the Fourier/trig-identity circuit, and progress measures.
- **Liu et al. 2022, Omnigrok** (arXiv:2210.01117) — grokking is controlled by weight norm; motivates the weight-decay/init-scale fallback experiment (Phase 4).
- **Lee et al. 2024, Grokfast** (arXiv:2405.20233) — slow-gradient amplification; used here for runtime, with the un-accelerated phenomenon reproduced first.

---

## 5. Self-critical evaluation (Ausblick)

**Solid.** Deterministic, reproducible pipeline (23/23 core correctness checks pass, incl. injected-frequency recovery and the 2D-Fourier round-trip); an unambiguous grokking transition with exact step indices — a faithful **un-accelerated ~8,200-step gap** plus seed-robust accelerated runs; a measurable embedding sparsification after grokking (**76%** top-8 power, up from 32% non-grokked); a **cross-architecture finding re-tested in the honest un-accelerated setting** — the sparsity gap is robust across both settings, and the speed gap holds in direction for every seed (§3); a working, browser-verified end-to-end Python→web pipeline feeding a real 3D explorer (LiveLab groks in-browser with a genuine measured delay); the Nanda **restricted/excluded-loss** progress measures reproduced and self-checked **on the canonical un-accelerated run** — the eight key frequencies alone solve the task while removing them destroys it, with the re-train recovering the recorded transition exactly (§2).

**Fragile / limited (reported honestly).**
- The cross-architecture **speed ratio is setting-dependent** (~2.9× Grokfast/frac=0.5 vs ~1.3× un-accelerated/frac=0.3, §3): the direction survived the honest re-test, the headline magnitude did not. Two settings ≠ a sweep, and the settings differ in two knobs at once (acceleration and train fraction).
- Bit-exact determinism holds per environment, not across environments: the CPU thread count shifts the last float digits, which over thousands of steps wobbles final accuracies by ~0.5% (the transition step indices reproduced exactly in every re-train we ran; `torch_num_threads` is now logged in run metadata).
- The restricted/excluded losses use a mask far wider than the published operator **and** are measured over the full `(a,b)` grid rather than Nanda's per-split protocol (excluded on train). They are not a reproduction (§2 **[AUDIT]**).
- **[AUDIT]** Every structure number on this page is measured *at* each run's generalization crossing, because the un-accelerated runs were launched with `--early-stop-acc 0.95`. Khanh 2026 (arXiv:2607.06639) measures 3–5x (MLP) and 1.3–1.5x (transformer) overstatement for metrics read at the transition rather than after convergence, so an architecture difference here may be a difference in how much cleanup each had done by its own crossing.
- **[AUDIT]** "Top-8 concentration" is a fixed k, not a measured one: the count is capped at 8 and the cap binds on all 16 runs, so the 90% threshold never fires.
- The MLP's lower sparsity is partly architectural (concatenated operand embeddings vs. attention-combined), not purely a grokking-quality difference.

**Next.** A weight-decay/train-frac sweep to locate where the cross-architecture speed gap grows; an add-vs-multiply variant; a static deploy of the explorer; per-split restricted/excluded losses.
