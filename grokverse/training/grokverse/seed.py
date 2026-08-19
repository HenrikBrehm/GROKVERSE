"""Deterministic seeding (PROMPT.md §2: identical config + seed => identical curves)."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every source of randomness used in a run.

    Note: bit-exact reproducibility additionally depends on the environment —
    in particular the CPU thread count (torch reductions differ across thread
    counts), which is why run metadata records ``torch_num_threads``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
