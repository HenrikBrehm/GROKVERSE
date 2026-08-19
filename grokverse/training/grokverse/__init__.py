"""GROKVERSE training & analysis package.

A small, seeded, reproducible pipeline for studying *grokking* in tiny networks
on modular arithmetic (Nanda et al. 2023; Power et al. 2022). Every run is
deterministic given its config + seed, and every reported number traces back to
an artifact in ``training/runs/`` (PROMPT.md §2, §6).
"""

__version__ = "0.1.0"
