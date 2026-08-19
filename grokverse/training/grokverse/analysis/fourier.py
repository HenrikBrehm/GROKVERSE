"""Fourier analysis of the token-embedding matrix (Nanda et al. 2023, Phase 3).

A grokked modular-addition network represents each number n by a sparse set of
frequencies: its embedding is approximately a sum of cos/sin of 2*pi*k*n/p for a
few key k. We build the orthonormal real Fourier basis over Z_p and measure how
much of the embedding's power sits at each frequency; a few dominant frequencies
(robust across seeds) is the signature of the learned trig-identity circuit.
"""
from __future__ import annotations

import numpy as np


def fourier_basis(p: int) -> tuple[np.ndarray, list[str]]:
    """Orthonormal real Fourier basis of R^p (rows = modes).

    Row 0 is the constant mode; then for k = 1..(p-1)//2 a (cos_k, sin_k) pair.
    For odd p this is exactly p rows. Returns (F [p, p], labels).
    """
    if p % 2 == 0:
        raise ValueError(f"fourier_basis requires odd p (got {p}): the cos/sin "
                         "pairing yields only p-1 rows for even p")
    n = np.arange(p)
    rows = [np.ones(p) / np.sqrt(p)]
    labels = ["const"]
    for k in range(1, (p - 1) // 2 + 1):
        c = np.cos(2 * np.pi * k * n / p)
        s = np.sin(2 * np.pi * k * n / p)
        rows.append(c / np.linalg.norm(c))
        rows.append(s / np.linalg.norm(s))
        labels += [f"cos{k}", f"sin{k}"]
    return np.stack(rows), labels


def embedding_power_spectrum(W_E: np.ndarray, p: int) -> dict:
    """Power per Fourier frequency for the number-token embeddings W_E[:p].

    Returns freqs (1..(p-1)//2), power[k] (cos^2+sin^2 summed over d_model),
    fraction of total frequency power per k, plus const/total power.
    """
    W = np.asarray(W_E)[:p]                       # [p, d]
    F, _ = fourier_basis(p)
    coeffs = F @ W                                # [p, d]
    power = (coeffs ** 2).sum(axis=1)            # [p]
    half = (p - 1) // 2
    freq_power = np.array([power[1 + 2 * (k - 1)] + power[2 + 2 * (k - 1)]
                           for k in range(1, half + 1)])
    total = float(freq_power.sum())
    fractions = (freq_power / total) if total > 0 else freq_power
    return {
        "freqs": list(range(1, half + 1)),
        "power": freq_power.tolist(),
        "fraction": fractions.tolist(),
        "const_power": float(power[0]),
        "total_power": total,
    }


def dominant_frequencies(W_E: np.ndarray, p: int, threshold: float = 0.9,
                         max_k: int = 8) -> dict:
    """Smallest set of frequencies capturing `threshold` of the embedding power."""
    spec = embedding_power_spectrum(W_E, p)
    power = np.array(spec["power"])
    freqs = np.array(spec["freqs"])
    order = np.argsort(power)[::-1]
    ranked = freqs[order]
    frac_sorted = power[order] / power.sum() if power.sum() > 0 else power[order]
    cum = np.cumsum(frac_sorted)
    n_keep = max(1, min(int(np.searchsorted(cum, threshold) + 1), max_k))
    return {
        "dominant": ranked[:n_keep].tolist(),
        "dominant_fraction": float(cum[n_keep - 1]) if len(cum) else 0.0,
        "ranked_freqs": ranked.tolist(),
        "ranked_fraction": frac_sorted.tolist(),
        "spectrum": spec,
    }


def frequency_concentration_over_time(embeds: np.ndarray, p: int,
                                      key_freqs: list[int]) -> list[float]:
    """Progress measure: fraction of embedding power in the key frequencies per
    logged step. Rises as the periodic circuit forms. embeds: [T, vocab, d]."""
    key = np.asarray(key_freqs)
    out = []
    for t in range(embeds.shape[0]):
        spec = embedding_power_spectrum(embeds[t], p)
        power = np.array(spec["power"])
        freqs = np.array(spec["freqs"])
        total = power.sum()
        out.append(float(power[np.isin(freqs, key)].sum() / total) if total > 0 else 0.0)
    return out
