"""Full-state checkpoints and their pre-specified roles (RUN_FORMAT_V2.md §3).

A v2 run directory holds ``ckpt_step{step:06d}.pt`` files listed in
``checkpoints.json``. Two kinds exist:

* **grid** — saved at the steps of ``Config.checkpoint_grid`` (fixed before the
  runs), plus the final step;
* **event** — saved at the FIRST dense evaluation that crosses the PRIMARY
  thresholds (memorization: train acc >= 0.99; generalization: test acc >= 0.95).

``assign_roles`` maps these to the seven analysis positions of master prompt
§15 by a RULE fixed in advance (table in RUN_FORMAT_V2.md §3), so no checkpoint
is ever picked after the fact to suit a narrative. The final checkpoint is
called ``final`` (fixed budget), never "converged".
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from .config import Config

CKPT_FMT = "ckpt_step{step:06d}.pt"
INDEX_NAME = "checkpoints.json"
ROLES = ("init", "pre_memorization", "memorization", "mid_plateau",
         "pre_generalization", "generalization", "final")


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return Path(run_dir) / CKPT_FMT.format(step=int(step))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(run_dir: Path, model: torch.nn.Module, step: int,
                    kind: str, entries: list[dict]) -> dict:
    """Save ``model.state_dict()`` at ``step`` and record it in ``entries``.

    If an entry for ``step`` already exists (an event coinciding with a grid
    step) no second file is written; ``kind`` is appended to the existing
    entry's comma-separated kind string instead.
    """
    for e in entries:
        if e["step"] == int(step):
            kinds = e["kind"].split(",")
            if kind not in kinds:
                e["kind"] = ",".join(kinds + [kind])
            return e
    path = checkpoint_path(run_dir, step)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    entry = {"step": int(step), "path": path.name, "kind": kind, "sha256": _sha256(path)}
    entries.append(entry)
    entries.sort(key=lambda e: e["step"])
    return entry


def write_checkpoint_index(run_dir: Path, entries: list[dict]) -> None:
    (Path(run_dir) / INDEX_NAME).write_text(json.dumps(entries, indent=2))


def list_checkpoints(run_dir: Path) -> list[dict]:
    idx = Path(run_dir) / INDEX_NAME
    if not idx.exists():
        raise FileNotFoundError(
            f"{idx} not found — {run_dir} is not a run-format-v2 directory "
            "(legacy runs hold only model_final.pt)")
    entries = json.loads(idx.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{idx} must hold a list of checkpoint entries")
    return sorted(entries, key=lambda e: e["step"])


def load_checkpoint(run_dir: Path, step: int) -> dict[str, torch.Tensor]:
    """The saved ``state_dict`` at ``step`` (must be listed in checkpoints.json)."""
    entries = list_checkpoints(run_dir)
    match = [e for e in entries if e["step"] == int(step)]
    if not match:
        raise KeyError(f"no checkpoint at step {step} in {run_dir}; "
                       f"available: {[e['step'] for e in entries]}")
    path = Path(run_dir) / match[0]["path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} listed in {INDEX_NAME} but missing on disk")
    return torch.load(path, map_location="cpu")


def config_from_run(run: dict) -> Config:
    """Rebuild ``Config`` from a run.json ``config`` block.

    The block also carries computed properties (``vocab_size``, ``run_id``);
    keep only real dataclass fields so old and new run files both load.
    """
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in run["config"].items() if k in fields})


def load_run(run_dir: Path) -> tuple[Config, dict, dict | None]:
    """``(cfg, run.json dict, manifest.json dict or None for legacy runs)``."""
    run_dir = Path(run_dir)
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"{run_json} not found")
    run = json.loads(run_json.read_text())
    cfg = config_from_run(run)
    man_path = run_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else None
    return cfg, run, manifest


# --------------------------------------------------------------------------- #
# pre-specified roles                                                          #
# --------------------------------------------------------------------------- #
def assign_roles_from(entries: list[dict], memorization_step: int | None,
                      generalization_step: int | None, final_step: int) -> dict:
    """Pure rule (RUN_FORMAT_V2.md §3) on an in-memory checkpoint list.

    ``memorization_step`` / ``generalization_step`` are the PRIMARY-threshold
    first-crossing steps (``None`` if never crossed). Returns
    ``{role: {"step", "path", "kind"} | None}``.
    """
    by_step = {e["step"]: e for e in entries}
    grid = sorted(e["step"] for e in entries if "grid" in e["kind"].split(","))

    def pick(step):
        return None if step is None or step not in by_step else dict(by_step[step])

    def last_grid_before(step):
        cands = [s for s in grid if step is not None and s < step]
        return pick(cands[-1]) if cands else None

    def nearest_grid(target):
        if target is None or not grid:
            return None
        return pick(min(grid, key=lambda s: (abs(s - target), s)))

    mid = None
    if memorization_step is not None and generalization_step is not None:
        mid = (memorization_step + generalization_step) / 2.0
    return {
        "init": pick(0),
        "pre_memorization": last_grid_before(memorization_step),
        "memorization": pick(memorization_step),
        "mid_plateau": nearest_grid(mid),
        "pre_generalization": last_grid_before(generalization_step),
        "generalization": pick(generalization_step),
        "final": pick(final_step),
    }


def assign_roles(run_dir: Path) -> dict:
    """Apply the role rule to a v2 run directory (reads run.json + index)."""
    cfg, run, _ = load_run(run_dir)
    entries = list_checkpoints(run_dir)
    tr = run.get("transitions", {}).get("primary")
    if tr is None:
        raise ValueError(f"{run_dir}/run.json has no 'transitions.primary' block "
                         "(not a run-format-v2 run)")
    mem = tr["memorization"]["first_crossing_step"]
    gen = tr["generalization"]["first_crossing_step"]
    final_step = int(run.get("steps_completed", cfg.steps))
    return assign_roles_from(entries, mem, gen, final_step)
