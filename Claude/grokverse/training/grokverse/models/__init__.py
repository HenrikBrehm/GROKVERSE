"""Model factory + the per-architecture "embedding snapshot" helper."""
from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from .transformer import OneLayerTransformer


def build_model(cfg: Config):
    if cfg.arch == "transformer":
        return OneLayerTransformer(cfg)
    if cfg.arch == "mlp":
        from .mlp import TwoLayerMLP
        return TwoLayerMLP(cfg)
    if cfg.arch == "mlp_twohot":
        from .mlp_twohot import TwoHotMLP
        return TwoHotMLP(cfg)
    raise ValueError(f"arch {cfg.arch!r} not implemented")


def embedding_object(model: torch.nn.Module) -> str:
    """Name of the tensor ``embedding_snapshot`` logs for this model.

    ``W_E`` for the transformer and the shared-embedding MLP; the two-hot MLP
    has no embedding table, so its a-half input weights stand in
    (RUN_FORMAT_V2.md §2). Recorded in run.json so the legacy ``embeddings.npy``
    is never mistaken for an embedding it is not.
    """
    if hasattr(model, "W_E"):
        return "W_E"
    if hasattr(model, "embedding_snapshot"):
        return "W_in_a_half"
    raise TypeError(f"{type(model).__name__} exposes neither W_E nor embedding_snapshot")


@torch.no_grad()
def embedding_snapshot(model: torch.nn.Module) -> np.ndarray:
    """Copy of the per-step "embedding" tensor logged to ``embeddings.npy``."""
    if hasattr(model, "W_E"):
        return model.W_E.detach().cpu().numpy().copy()
    if hasattr(model, "embedding_snapshot"):
        return model.embedding_snapshot().detach().cpu().numpy().copy()
    raise TypeError(f"{type(model).__name__} exposes neither W_E nor embedding_snapshot")
