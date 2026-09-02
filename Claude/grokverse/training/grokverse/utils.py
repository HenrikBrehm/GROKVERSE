"""Run metadata + small IO helpers (PROMPT.md §2: log versions, seed, git commit)."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def lib_versions() -> dict:
    import numpy
    import torch
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
    }


def git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def config_diff(recorded: dict, ours: dict) -> tuple[dict, list[str]]:
    """Compare a config against one recorded in an existing ``run.json``.

    Returns ``(differs, added_since)``:

    * ``differs``     — fields present in BOTH that disagree; ``{field: (recorded, ours)}``.
                        These are real protocol mismatches and must block.
    * ``added_since`` — fields the current ``Config`` has that the recorded run
                        predates (e.g. ``threads``, added in the §3.8 fix).
                        A missing key cannot be a disagreement: the recorded run
                        simply ran before the field existed. Reported, not fatal
                        — otherwise adding any Config field would retroactively
                        invalidate every stored run.
    """
    differs, added_since = {}, []
    for k, v in ours.items():
        if k not in recorded:
            added_since.append(k)
        elif recorded[k] != v:
            differs[k] = (recorded[k], v)
    return differs, sorted(added_since)


def log_step_schedule(steps: int, n: int) -> list[int]:
    """A logarithmic checkpoint schedule including step 0 and the final step."""
    if steps <= 0:
        return [0]
    pts = np.unique(np.round(np.geomspace(1, steps, max(n, 2))).astype(int))
    pts = np.unique(np.clip(np.concatenate([[0], pts, [steps]]), 0, steps))
    return [int(x) for x in pts]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "runs"
