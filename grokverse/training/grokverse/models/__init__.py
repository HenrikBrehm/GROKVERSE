"""Model factory."""
from __future__ import annotations

from ..config import Config
from .transformer import OneLayerTransformer


def build_model(cfg: Config):
    if cfg.arch == "transformer":
        return OneLayerTransformer(cfg)
    if cfg.arch == "mlp":
        from .mlp import TwoLayerMLP
        return TwoLayerMLP(cfg)
    raise ValueError(f"arch {cfg.arch!r} not implemented")
