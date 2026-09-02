"""Modular-arithmetic dataset + deterministic train/test split (PLAN Phase 1).

Builds every (a, b) pair for a, b in [0, p), with target c = (a op b) mod p.
Each example is the token sequence [a, b, '='] and the label c. The split is a
fixed permutation keyed on the run seed, so it is reproducible.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .config import Config


def split_hash(train_idx: torch.Tensor | np.ndarray) -> str:
    """SHA-256 of the SORTED training-index array (int64, little-endian bytes).

    Identifies a train/test split independently of the architecture that used
    it, so paired runs can be checked to share a split (RUN_FORMAT_V2.md §4).
    """
    idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    return hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest()


def make_dataset(cfg: Config) -> dict[str, torch.Tensor]:
    p = cfg.p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    if cfg.task == "add":
        c = (a + b) % p
    elif cfg.task == "mul":
        c = (a * b) % p
    else:  # pragma: no cover
        raise ValueError(f"unknown task {cfg.task!r}")
    eq = torch.full_like(a, cfg.equals_token)
    x = torch.stack([a, b, eq], dim=1).long()
    y = c.long()

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(round(cfg.train_frac * p * p))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return {
        "train_x": x[train_idx], "train_y": y[train_idx],
        "test_x": x[test_idx], "test_y": y[test_idx],
        "all_x": x, "all_y": y,
        "train_idx": train_idx, "test_idx": test_idx,
    }
