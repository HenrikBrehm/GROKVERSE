"""1-layer ReLU transformer for modular arithmetic (Nanda et al. 2023 recipe).

A deliberately minimal, no-LayerNorm transformer so the learned circuit is easy
to read out: token + positional embeddings, one multi-head attention layer, one
ReLU MLP, and an unembedding. The prediction is read off the final ('=')
position. The embedding matrix ``W_E`` is the main object of the Fourier/PCA
analysis in Phase 3.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import Config


class OneLayerTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_heads * cfg.d_head == cfg.d_model, \
            "n_heads * d_head must equal d_model"
        self.cfg = cfg
        d, v, h, dh = cfg.d_model, cfg.vocab_size, cfg.n_heads, cfg.d_head

        self.W_E = nn.Parameter(torch.empty(v, d))
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, d))
        self.W_Q = nn.Parameter(torch.empty(h, d, dh))
        self.W_K = nn.Parameter(torch.empty(h, d, dh))
        self.W_V = nn.Parameter(torch.empty(h, d, dh))
        self.W_O = nn.Parameter(torch.empty(h, dh, d))
        self.W_in = nn.Parameter(torch.empty(d, cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, d))
        self.W_U = nn.Parameter(torch.empty(d, v))
        self._init_weights()

    def _init_weights(self) -> None:
        s, d, dh, h = self.cfg.init_scale, self.cfg.d_model, self.cfg.d_head, self.cfg.n_heads
        nn.init.normal_(self.W_E, std=s / math.sqrt(d))
        nn.init.normal_(self.W_pos, std=s / math.sqrt(d))
        for W in (self.W_Q, self.W_K, self.W_V):
            nn.init.normal_(W, std=s / math.sqrt(d))
        nn.init.normal_(self.W_O, std=s / math.sqrt(dh * h))
        nn.init.normal_(self.W_in, std=s / math.sqrt(d))
        nn.init.normal_(self.W_out, std=s / math.sqrt(self.cfg.d_mlp))
        nn.init.normal_(self.W_U, std=s / math.sqrt(d))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        T = tokens.shape[1]
        x = self.W_E[tokens] + self.W_pos[None, :T, :]

        q = torch.einsum("btd,hde->bhte", x, self.W_Q)
        k = torch.einsum("btd,hde->bhte", x, self.W_K)
        v = torch.einsum("btd,hde->bhte", x, self.W_V)
        scores = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(cfg.d_head)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        z = torch.einsum("bhts,bhse->bhte", attn, v)
        x = x + torch.einsum("bhte,hed->btd", z, self.W_O)

        hmlp = torch.relu(torch.einsum("btd,df->btf", x, self.W_in))
        x = x + torch.einsum("btf,fd->btd", hmlp, self.W_out)

        return torch.einsum("btd,dv->btv", x, self.W_U)

    @torch.no_grad()
    def attention_pattern(self, tokens: torch.Tensor) -> torch.Tensor:
        """Softmaxed attention weights, recomputed exactly as ``forward``.

        Re-runs the embedding + Q/K projection + scaled-dot-product + causal
        mask + softmax of the single attention layer (no value/output path),
        and returns the attention tensor of shape ``[B, n_heads, n_ctx, n_ctx]``
        where entry ``[b, h, t, s]`` is how much position ``t`` attends to
        position ``s`` for head ``h`` on input ``b``. Used by the Phase-3
        attention view to read off where the '=' position looks.
        """
        cfg = self.cfg
        T = tokens.shape[1]
        x = self.W_E[tokens] + self.W_pos[None, :T, :]

        q = torch.einsum("btd,hde->bhte", x, self.W_Q)
        k = torch.einsum("btd,hde->bhte", x, self.W_K)
        scores = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(cfg.d_head)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        return scores.softmax(dim=-1)

    def logits_last(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens)[:, -1, :]
