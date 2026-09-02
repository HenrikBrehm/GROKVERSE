"""Two-hot-input MLP — the input-parametrization literature control.

The shared-embedding MLP (``models/mlp.py``) reads both operands through one
learned table ``W_E``. This model has NO embedding: its input is the
concatenation of two one-hot vectors, ``[onehot_p(a), onehot_p(b)]`` in R^{2p},
mapped by one ReLU hidden layer to the ``p`` output classes::

    x      = concat(onehot_p(a), onehot_p(b))        # [B, 2p]
    h      = ReLU(x @ W_in + b_in)                    # W_in [2p, d_mlp]
    logits = h @ W_out + b_out                        # W_out [d_mlp, p]

It answers one question only (master prompt §14): *does the structure found in
the shared-embedding MLP persist when the MLP sees the raw two-hot input?* It
is the setup described for ReLU MLPs by Swaroop (arXiv:2603.23784); until the
source note confirms the exact layer sizes this is OUR two-hot variant and is
labelled as such (docs/dev/RUN_FORMAT_V2.md §2).

Because a one-hot row-select equals an index into ``W_in``, the forward pass is
computed as ``W_in[a] + W_in[p + b]`` — mathematically identical to the two-hot
matrix product, without materializing the one-hot tensor. The effective
per-neuron curves are therefore simply ``u_a = W_in[:p]`` and ``u_b = W_in[p:]``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import Config


class TwoHotMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.arch != "mlp_twohot":
            raise ValueError(f"TwoHotMLP built with arch={cfg.arch!r}")
        self.cfg = cfg
        p = cfg.p
        self.W_in = nn.Parameter(torch.empty(2 * p, cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, p))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(p))
        self._init_weights()

    def _init_weights(self) -> None:
        s, p = self.cfg.init_scale, self.cfg.p
        nn.init.normal_(self.W_in, std=s / math.sqrt(2 * p))
        nn.init.normal_(self.W_out, std=s / math.sqrt(self.cfg.d_mlp))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        a, b = tokens[:, 0], tokens[:, 1]
        # two-hot @ W_in == row a of the a-half plus row b of the b-half
        x = self.W_in[a] + self.W_in[self.cfg.p + b]        # [B, d_mlp]
        h = torch.relu(x + self.b_in)
        return h @ self.W_out + self.b_out                   # [B, p]

    def logits_last(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens)

    @torch.no_grad()
    def embedding_snapshot(self) -> torch.Tensor:
        """The a-half of ``W_in`` ([p, d_mlp]) — the closest analogue of an
        embedding table this model has. Logged as ``embeddings.npy`` with
        ``embedding_object = "W_in_a_half"`` in run.json."""
        return self.W_in[: self.cfg.p]
