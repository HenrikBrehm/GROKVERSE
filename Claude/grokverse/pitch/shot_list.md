> ## ⚠ VERALTET — NICHT DARAUS FILMEN (Stand 2026-09-20)
>
> Diese Shot-Liste ist vom 2026-08-18 und stammt aus der Reproduktionsphase. Es enthält mindestens drei
> Aussagen, die die Architekturstudie seither **zurückgezogen** hat: die 76-%-Top-8-Schlagzeile, den
> „sparser circuit"-Schluss und die 50/50-Attention als Nachweis der Schaltung. Die aktuelle Fassung ist
> **`pitch/script_2026-09-20.md`** (Skript und Shot-Liste in einer Datei). Diese Datei bleibt als Beleg der Projektgeschichte stehen.

# GROKVERSE — Shot List

Maps each script beat to the exact explorer view and on-screen action.
Components referenced are the live ones in `web/`: **EmbeddingView3D**, **LossView**,
**ProgressMeasuresView**, **FourierView**, **ControlPanel**, **LiveLab**.
Run shown: `txf_add_p113_wd1.0_frac0.3_seed0` — the canonical **un-accelerated**
run and the explorer's default. All numbers verified against its `run.json` /
`progress_measures.json`.

| # | Timecode | Script beat | View / component | On-screen action |
|---|----------|-------------|------------------|------------------|
| 1 | 0:00–0:08 | Hook: "Watch the exact moment an AI 'gets it'." | Black → EmbeddingView3D | Fade from black onto the 3D embedding scene, paused at the **blob** (step 0). Title card overlays the hook line, then clears. |
| 2 | 0:08–0:50 | What is grokking? | EmbeddingView3D (idle) + LossView (small inset) | Slow auto-rotate of the blob. LossView inset previews the curve shape (train high, test flat) but no annotations yet. Keep it ambient under VO. |
| 3 | 0:50–1:18 | Reproduction setup + memorization | LossView (full) | Cut to full LossView. Scrub/play to **step 145**; highlight train acc crossing **0.99** (perfect by 156) while test acc sits near random. Callout chip: "memorized @ step 145". |
| 4 | 1:18–1:45 | The plateau → generalization numbers | LossView (full) with transition marker | Continue playback across the **8,000-step plateau** (test loss visibly climbing to ~26) to the **gold marker at step 8,367**. Callouts as they hit: "test acc crosses 0.95 @ 8,367", "grok gap = 8,222 steps", "final test acc 0.981", "test loss ~26 → 0.06". The credibility beat: "deterministic re-train recovers 145 / 8,367 exactly." |
| 5 | 1:45–2:10 | Mechanistic story — embeddings reorganize | EmbeddingView3D (scrub) | Cut back to EmbeddingView3D. Scrub slowly from the plateau; hold mid-transition so the structure is half-formed. |
| 6 | 2:10–2:25 | Fourier / key frequencies + attention | FourierView | Cut to FourierView. Highlight the gold dominant bars (**k = 18, 15, 1, 11, 13, 22, 56, 36**). Callouts: "top 8 = 76% of frequency power (32% when it never groks)", "read-out attends 50/50 to both operands". |
| 7 | 2:25–2:40 | Restricted / excluded loss | ProgressMeasuresView | Scrub across the transition in the progress-measures panel: gold (restricted) collapsing, violet (excluded) climbing past the chance line. Callouts: "keep only 8 frequencies → loss 0.002", "delete them → 10.4, far worse than chance". |
| 8 | 2:40–3:05 | Explorer — scrub the blob into a ring | EmbeddingView3D (full scrub) + ControlPanel (scrubber) | Center EmbeddingView3D. Drag the ControlPanel scrubber across the grok point; the **blob visibly pulls into a ring**. Cursor/drag visible to sell interactivity. |
| 9 | 3:05–3:20 | Explorer — switch runs + Live Lab | ControlPanel, then LiveLab | Open the run selector: honest labels ("un-accelerated", "Grokfast", "**did not grok**"). Briefly select the never-grokked run (structure absent), back to canonical. Then open LiveLab: train, weight decay up, the **measured grok gap** appears in the status line. |
| 10 | 3:20–3:42 | Self-critique: cross-architecture re-test | both crossarch figures side by side | Show `figures/crossarch_comparison.png` then `crossarch_comparison_unaccelerated.png`. VO: ~3× faster under Grokfast; re-tested un-accelerated the transformer still wins every seed but only ~1.3× — sparsity gap (0.73 vs 0.44) is the robust finding. Honesty as a feature. |
| 11 | 3:42–4:00 | Self-critique: determinism scope | LossView or plain overlay | VO over the curves: bit-exactness is per-environment (thread count recorded per run); the transition steps are the robust, exactly-reproducing quantity. No fake data shown. |
| 12 | 4:00–4:15 | Close | EmbeddingView3D (ring) | Return to EmbeddingView3D holding on the fully-formed **ring**. Hook line returns as end card. Fade out. |

## Notes for the editor
- The three "wow" visual moments: (a) blob → ring in EmbeddingView3D (shots 5/8/12), (b) test-loss cliff hitting the gold **step-8,367** marker after an 8,000-step plateau (shot 4), (c) the restricted/excluded split in ProgressMeasuresView (shot 7). Give all three room to breathe.
- Never display a number not in the verified set. Canonical run: memorize 145 (0.99-crossing; 1.000 @ 156), generalize 8,367, gap 8,222, final test acc 0.981, test loss ~26 → 0.06, key freqs [18, 15, 1, 11, 13, 22, 56, 36], top-8 = 76% (vs 32% non-grokked), attention ≈ 0.501/0.499/0.001, restricted 0.0020 / full 0.031 / excluded 10.42. Cross-arch, Grokfast frac 0.5 (3 seeds each): 750 ± 97 vs 2,163 ± 72 steps; top-8 power 0.59 ± 0.06 vs 0.35 ± 0.003. Cross-arch, un-accelerated frac 0.3 (3 seeds each): 7,676 ± 1,196 vs 9,883 ± 410 steps (~1.3×, transformer first in every seed); top-8 power 0.73 ± 0.07 vs 0.44 ± 0.01.
- The old "identical logit sum" determinism beat is retired — the scalar is environment-dependent. If a credibility beat is needed, use "deterministic re-train recovers the transition at exactly 145 / 8,367" (shot 4).
