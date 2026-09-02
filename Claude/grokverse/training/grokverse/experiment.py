"""Run a parameterized grid of training runs (PLAN Phase 2 reproducibility harness).

Each run is fully determined by its config + seed and saved to runs/<run_id>/,
so deleting runs/ and re-running reproduces equivalent curves.

Usage (from training/):
    python -m grokverse.experiment --grid main
    python -m grokverse.experiment --grid seeds --steps 20000
"""
from __future__ import annotations

import argparse
import itertools
import json

from .config import get_config
from .train import train
from .utils import runs_dir

GRIDS: dict[str, dict] = {
    # Variety for the explorer ControlPanel: weight decay x train fraction.
    "main": {
        "weight_decay": [1.0, 0.5],
        "train_frac": [0.3, 0.5],
        "seed": [0],
        "task": ["add"],
    },
    # Seed-robustness of the canonical reproduction.
    "seeds": {
        "weight_decay": [1.0],
        "train_frac": [0.3],
        "seed": [0, 1, 2],
        "task": ["add"],
    },
}


def run_grid(name: str, steps: int | None = None, force: bool = False) -> list:
    grid = GRIDS[name]
    keys = list(grid)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        overrides = dict(zip(keys, combo))
        if steps is not None:
            overrides["steps"] = steps
        cfg = get_config("nanda", **overrides)
        print(f"=== {cfg.run_id} ===", flush=True)
        try:
            res, _, _ = train(cfg, out_dir=runs_dir() / cfg.run_id, allow_overwrite=force)
        except RuntimeError as e:
            # config collision with an existing run dir: report and continue the
            # grid instead of aborting the whole batch
            print(f"[skip] {cfg.run_id}: {e}", flush=True)
            results.append({"run_id": cfg.run_id, "skipped": str(e)})
            continue
        results.append({"run_id": cfg.run_id, "transition": res["transition"]})
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main", choices=list(GRIDS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing run dirs even if their config differs")
    args = ap.parse_args()
    run_grid(args.grid, args.steps, args.force)


if __name__ == "__main__":
    main()
