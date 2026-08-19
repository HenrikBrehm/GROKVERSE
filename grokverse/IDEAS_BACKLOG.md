# IDEAS_BACKLOG.md

Out-of-scope ideas park here instead of leaking into `main` (per `PROMPT.md` §6 "Stay in scope"). Nothing here is a commitment; promote an item only after the North Star is met.

## Backlog

- KAN (Kolmogorov–Arnold Network) as a third architecture in the cross-arch experiment (stretch in PLAN Phase 4).
- `add` vs `mul` mod p comparison on a fixed architecture (clean variant of the original experiment).
- Grokfast acceleration view: show the same run with/without slow-gradient amplification.
- Static deploy of the explorer (Vercel/GitHub Pages) — nice-to-have, not required for submission.
- Multiple-prime sweep (p ∈ {59, 97, 113}) to show frequency structure scales with p.
- Audio: sonify the loss phase transition for the pitch video.
- Per-step Fourier spectrum in the explorer (FourierView currently shows the final spectrum plus a live key-frequency-concentration meter; a fully live spectrum needs per-step spectra in the export).
- Per-split restricted/excluded losses (Nanda's exact protocol: restricted on test, excluded on train; ours is full-grid, disclosed in RESULTS §2).
- Encode `steps` in `run_id` (would rename every existing run dir/artifact; the config-collision guard in train.py/progress_measures.py covers the failure mode for now).

## Promoted

- ~~Restricted/excluded-loss live overlay in the 3D view.~~ → shipped 2026-08-18 as the explorer's progress-measures panel (per-run, only where the measure was actually computed).

_Last updated: 2026-08-18._
