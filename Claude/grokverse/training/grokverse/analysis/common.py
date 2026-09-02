"""Shared loading, logit-grid, and output helpers for every study analysis module.

Contract: docs/dev/INTERFACES.md §0–§1. Every analysis module loads models through
``load_model_at`` and writes its results through ``write_result`` so that outputs are
uniformly located (``<run_dir>/analysis/<module>/<tag>.json`` + ``.npz``) and carry the
same provenance envelope (run id, checkpoint hash, analysis git commit, parameters).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..data import make_dataset
from ..models import build_model
from ..utils import git_commit, utcnow


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def config_from_run_json(run: dict) -> Config:
    """Rebuild ``Config`` from a logged ``run.json`` block, keeping only dataclass fields.

    The logged block also carries computed properties (``vocab_size``, ``run_id``, ...)
    that are not constructor arguments.
    """
    fields = {f.name for f in dataclasses.fields(Config)}
    params = {k: v for k, v in run["config"].items() if k in fields}
    if "checkpoint_grid" in params and isinstance(params["checkpoint_grid"], list):
        params["checkpoint_grid"] = tuple(params["checkpoint_grid"])
    return Config(**params)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_at(run_dir: Path | str, step: int | None = None):
    """Load a run's config and the model state at ``step``.

    ``step=None`` loads the legacy ``model_final.pt``; an integer loads the v2 checkpoint
    ``ckpt_step{step:06d}.pt`` through ``grokverse.checkpoints``. Returns
    ``(cfg, model, state, meta)`` where ``state`` maps parameter names to float64 numpy
    arrays and ``meta`` records provenance (INTERFACES §0).
    """
    run_dir = Path(run_dir)
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise ValueError(f"{run_dir}: no run.json — not a run directory")
    run = json.loads(run_json.read_text())
    cfg = config_from_run_json(run)

    if step is None:
        ckpt = run_dir / "model_final.pt"
        if not ckpt.exists():
            raise ValueError(f"{run_dir}: model_final.pt missing (run incomplete?)")
        tag = "final"
    else:
        from ..checkpoints import checkpoint_path  # lazy: legacy runs need no checkpoints module
        ckpt = checkpoint_path(run_dir, int(step))
        if not ckpt.exists():
            raise ValueError(f"{run_dir}: no checkpoint at step {step} ({ckpt.name})")
        tag = f"step{int(step):06d}"

    raw = torch.load(ckpt, map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(raw)
    model.eval()
    state = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in raw.items()}
    meta = {
        "run_id": cfg.run_id,
        "arch": cfg.arch,
        "p": cfg.p,
        "step": (int(step) if step is not None else "final"),
        "tag": tag,
        "checkpoint_file": ckpt.name,
        "checkpoint_sha256": sha256_of_file(ckpt),
        "run_git_commit": run.get("git_commit"),
    }
    return cfg, model, state, meta


# --------------------------------------------------------------------------- #
# logits over the full grid + splits                                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def grid_logits(model: torch.nn.Module, cfg: Config) -> np.ndarray:
    """Logits over every (a, b): ``[p, p, p]`` float64, ``a`` outer, ``b`` inner, class last.

    Row order matches ``data.make_dataset`` (``a = arange(p).repeat_interleave(p)``).
    """
    x = make_dataset(cfg)["all_x"]
    lg = model.logits_last(x)[:, :cfg.p].double().cpu().numpy()
    return lg.reshape(cfg.p, cfg.p, cfg.p)


def split_masks(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``[p, p]`` masks of the training and test cells for this config's seed."""
    d = make_dataset(cfg)
    p = cfg.p
    train = np.zeros((p, p), dtype=bool)
    tx = d["train_x"].numpy()
    train[tx[:, 0], tx[:, 1]] = True
    test = ~train
    if train.sum() + test.sum() != p * p:
        raise ValueError("split masks do not partition the grid")
    return train, test


def split_hash(cfg: Config) -> str:
    """SHA256 of the sorted training index array (same definition as the run manifest)."""
    d = make_dataset(cfg)
    tx = d["train_x"].numpy()
    idx = np.sort(tx[:, 0].astype(np.int64) * cfg.p + tx[:, 1].astype(np.int64))
    return hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()


def center_logits(L: np.ndarray) -> np.ndarray:
    """Subtract the per-(a, b) mean over classes (softmax-invariant centering)."""
    return L - L.mean(axis=-1, keepdims=True)


def masked_ce_and_acc(L: np.ndarray, mask: np.ndarray, p: int) -> tuple[float, float]:
    """Cross-entropy and accuracy of a ``[p, p, p]`` logit grid on the cells where ``mask``."""
    a = np.arange(p)[:, None]
    b = np.arange(p)[None, :]
    y = ((a + b) % p)[mask]
    lg = L[mask]
    lg = lg - lg.max(axis=-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(axis=-1, keepdims=True))
    ce = float(-logp[np.arange(len(y)), y].mean())
    acc = float((lg.argmax(-1) == y).mean())
    return ce, acc


# --------------------------------------------------------------------------- #
# output envelope                                                              #
# --------------------------------------------------------------------------- #
def envelope(module: str, version: str, meta: dict, params: dict) -> dict:
    """The provenance header every analysis JSON starts with (INTERFACES §0)."""
    return {
        "module": module,
        "module_version": version,
        "run_id": meta["run_id"],
        "arch": meta["arch"],
        "p": meta["p"],
        "step": meta["step"],
        "checkpoint_file": meta.get("checkpoint_file"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "analysis_git_commit": git_commit(),
        "created_utc": utcnow(),
        "params": _jsonable(params),
        "results": {},
    }


def write_result(run_dir: Path | str, module: str, tag: str, payload: dict,
                 arrays: dict[str, np.ndarray] | None = None) -> Path:
    """Write ``<run_dir>/analysis/<module>/<tag>.json`` (+ ``.npz`` for arrays)."""
    out_dir = Path(run_dir) / "analysis" / module
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False))
    if arrays:
        np.savez_compressed(out_dir / f"{tag}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def summarize(values: np.ndarray) -> dict:
    """Distribution summary used instead of dumping arrays into JSON (INTERFACES §0)."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
        "q75": float(q[3]), "q95": float(q[4]), "max": float(v.max()),
    }


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, tuples and Paths to JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        if not np.isfinite(f):
            raise ValueError(f"non-finite value {f!r} cannot be written to JSON")
        return f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        raise ValueError(f"non-finite value {obj!r} cannot be written to JSON")
    if isinstance(obj, Path):
        return str(obj)
    return obj
