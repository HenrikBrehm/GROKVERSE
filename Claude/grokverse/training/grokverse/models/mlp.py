"""2-layer MLP for modular arithmetic (cross-architecture comparison, Phase 4).

Embeds a and b with a shared embedding table W_E [p, d_model] (so the *same*
Fourier/PCA analysis used for the transformer applies unchanged), concatenates
the two operand embeddings, and maps them through one ReLU hidden layer to the p
output classes. It receives the same [a, b, =] token tensor as the transformer
but uses only the a and b positions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import Config


class TwoLayerMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.d_model, cfg.p
        self.W_E = nn.Parameter(torch.empty(p, d))
        self.W_in = nn.Parameter(torch.empty(2 * d, cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, p))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(p))
        self._init_weights()

    def _init_weights(self) -> None:
        s, d = self.cfg.init_scale, self.cfg.d_model
        nn.init.normal_(self.W_E, std=s / math.sqrt(d))
        nn.init.normal_(self.W_in, std=s / math.sqrt(2 * d))
        nn.init.normal_(self.W_out, std=s / math.sqrt(self.cfg.d_mlp))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        a, b = tokens[:, 0], tokens[:, 1]
        x = torch.cat([self.W_E[a], self.W_E[b]], dim=-1)   # [B, 2d]
        h = torch.relu(x @ self.W_in + self.b_in)           # [B, d_mlp]
        return h @ self.W_out + self.b_out                  # [B, p]

    def logits_last(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens)                          # [B, p]
