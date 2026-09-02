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
    n_wanted = int(np.searchsorted(cum, threshold) + 1)   # what `threshold` asks for
    n_keep = max(1, min(n_wanted, max_k))
    return {
        "dominant": ranked[:n_keep].tolist(),
        "dominant_fraction": float(cum[n_keep - 1]) if len(cum) else 0.0,
        "ranked_freqs": ranked.tolist(),
        "ranked_fraction": frac_sorted.tolist(),
        "spectrum": spec,
        # --- provenance of the count (docs/RESEARCH_SPEC.md §3.2 [AUDIT]) ---
        # On all 16 existing runs the max_k cap binds and the 90% threshold
        # never fires, so "top-8" is a FIXED k, not one measured from data —
        # exactly the assumption H3 says may be wrong. Reported explicitly so
        # no caller can mistake the cap for a measurement.
        "n_keep": n_keep,
        "threshold": threshold,
        "max_k": max_k,
        "n_freqs_for_threshold": n_wanted,
        "cap_binding": bool(n_wanted > max_k),
    }


# --------------------------------------------------------------------------- #
# Harmonic families (docs/RESEARCH_SPEC.md §4/H3)                              #
# --------------------------------------------------------------------------- #
#: Odd harmonics kept when building a family. The cap is LOAD-BEARING, not
#: cosmetic: for odd p the odd multiples of any single fundamental hit every
#: residue class of Z_p, so an uncapped family covers ALL (p-1)/2 frequencies
#: and the concentration would be identically 1 for every model — a metric that
#: cannot distinguish anything (asserted in test_core.py). j <= 7 retains ~95%
#: of an ideal square wave's power (1 + 1/9 + 1/25 + 1/49 of pi^2/8).
MAX_ODD_HARMONIC = 7


def alias_frequency(k: int, p: int) -> int:
    """Fold an arbitrary integer frequency into 1..(p-1)//2.

    ``cos`` identifies k with p-k, so ``k' = min(k mod p, p - (k mod p))``.
    Returns 0 for multiples of p (the constant mode), which is not a frequency.
    """
    r = int(k) % p
    return min(r, p - r) if r else 0


def harmonic_family(fundamentals, p: int,
                    max_harmonic: int = MAX_ODD_HARMONIC) -> list[int]:
    """Aliased odd-harmonic family {alias(j*k) : k in fundamentals, j odd <= cap}.

    A square wave at fundamental k puts its power at 3k, 5k, 7k, ... which in
    Z_p alias to scattered indices — the mechanism by which a perfectly clean
    algorithm can score low on a top-8 metric (H3).
    """
    fam: set[int] = set()
    for k in fundamentals:
        for j in range(1, int(max_harmonic) + 1, 2):
            a = alias_frequency(j * int(k), p)
            if a > 0:
                fam.add(a)
    return sorted(fam)


def select_fundamentals(power: np.ndarray, p: int, n_f: int,
                        max_harmonic: int = MAX_ODD_HARMONIC) -> list[int]:
    """Greedy fundamental selection, fixed in advance (§4/H3).

    Repeatedly adds the fundamental whose capped family contributes the most
    *not yet covered* power. ``power`` is indexed by frequency-1 (as returned by
    ``embedding_power_spectrum``). Deterministic: ties break on the lower
    frequency index.
    """
    half = (p - 1) // 2
    power = np.asarray(power, dtype=float)
    chosen: list[int] = []
    covered: set[int] = set()
    for _ in range(int(n_f)):
        best_k, best_gain = None, -np.inf
        for k in range(1, half + 1):
            if k in chosen:
                continue
            gain = sum(power[f - 1] for f in harmonic_family([k], p, max_harmonic)
                       if f not in covered)
            if gain > best_gain:
                best_k, best_gain = k, gain
        if best_k is None:
            break
        chosen.append(best_k)
        covered.update(harmonic_family([best_k], p, max_harmonic))
    return chosen


#: Power at odd harmonics j=3,5,7 relative to the fundamental, for an IDEAL
#: square wave: amplitudes decay as 1/j, so power decays as 1/j^2.
#: sum(1/9 + 1/25 + 1/49) = 0.1715. The measured statistic is compared against
#: this reference rather than against "bigger is better".
IDEAL_SQUARE_ODD_SHARE = sum(1.0 / j ** 2 for j in (3, 5, 7))


def harmonic_shape(power: np.ndarray, fundamentals, p: int) -> dict:
    """Odd- vs even-harmonic energy around a set of fundamentals.

    THIS is the statistic that discriminates H2/H3's square-wave claim, and it
    is deliberately NOT a concentration metric. Concentration cannot do the job:
    ``family_fraction <= matched_top_m_fraction`` holds by construction (top-m
    is the argmax over sets of size m), and in synthetic tests the
    cardinality-matched control closed the architecture gap just as well as the
    harmonic family — i.e. a concentration gap that closes says "we kept more
    components", not "the structure is harmonic".

    A square wave has energy at odd harmonics **only**, so:

    * ``odd_share``  (j = 3, 5, 7)  is high  — near ``IDEAL_SQUARE_ODD_SHARE``
    * ``even_share`` (j = 2, 4, 6)  is low
    * ``odd_minus_even`` is the signal; broadband noise inflates both shares
      equally, so the difference cancels it. Measured on synthetic embeddings it
      stays ~+0.16 for square waves and ~0 for sinusoids across noise levels
      that swamp the raw shares.

    ``fundamental_power_fraction`` guards the degenerate case: these are ratios
    to the fundamentals' power, so if the fundamentals carry almost nothing (a
    model with no periodic structure at all) the ratios explode and mean
    nothing. Check it before interpreting the shares.
    """
    power = np.asarray(power, dtype=float)
    total = float(power.sum())
    base_idx = {alias_frequency(k, p) for k in fundamentals} - {0}
    base = float(sum(power[i - 1] for i in base_idx))

    def share(js) -> float:
        idx: set[int] = set()
        for j in js:
            idx |= {alias_frequency(j * k, p) for k in fundamentals}
        idx = {i for i in idx if i > 0} - base_idx      # never count j=1 twice
        return float(sum(power[i - 1] for i in idx) / base) if base > 0 else float("nan")

    odd, even = share((3, 5, 7)), share((2, 4, 6))
    return {
        "odd_share": odd,
        "even_share": even,
        "odd_minus_even": odd - even,
        "ideal_square_odd_share": IDEAL_SQUARE_ODD_SHARE,
        "fundamental_power_fraction": (base / total) if total > 0 else 0.0,
    }


def harmonic_family_concentration(W_E: np.ndarray, p: int, n_f: int,
                                  max_harmonic: int = MAX_ODD_HARMONIC,
                                  n_null: int = 2000, seed: int = 0) -> dict:
    """H3's descriptive numbers, always reported WITH their controls.

    A family of ``n_f`` fundamentals holds up to ``4*n_f`` frequencies, so it
    could close the architecture gap merely by keeping more components than a
    top-8 metric. Returned alongside every family number:

    * ``matched_top_m_fraction`` — plain top-m concentration at the SAME
      cardinality ``m = len(family)``.
    * ``null_*`` — concentration of ``n_null`` uniformly random frequency sets
      of size m (seeded, reproducible).
    * ``shape`` — the ``harmonic_shape`` statistic, which is what actually
      decides H3.

    IMPORTANT: ``family_fraction <= matched_top_m_fraction`` ALWAYS (top-m is
    the maximum-power set of size m), so "the family beats top-m" is not a
    satisfiable criterion and must never be used as one. The concentration
    numbers are descriptive; ``shape`` carries the inferential load. See
    docs/RESEARCH_SPEC.md §4/H3.
    """
    spec = embedding_power_spectrum(W_E, p)
    power = np.asarray(spec["power"], dtype=float)
    total = float(power.sum())
    if total <= 0:
        raise ValueError("embedding has zero Fourier power — cannot form families")

    fundamentals = select_fundamentals(power, p, n_f, max_harmonic)
    family = harmonic_family(fundamentals, p, max_harmonic)
    m = len(family)
    fam_frac = float(power[[f - 1 for f in family]].sum() / total)

    ranked = np.argsort(power)[::-1]
    top_m = (ranked[:m] + 1).tolist()
    top_m_frac = float(power[ranked[:m]].sum() / total)

    rng = np.random.default_rng(seed)
    half = (p - 1) // 2
    draws = np.array([power[rng.choice(half, size=m, replace=False)].sum() / total
                      for _ in range(int(n_null))])
    return {
        "n_f": int(n_f),
        "max_harmonic": int(max_harmonic),
        "fundamentals": fundamentals,
        "family": family,
        "m": m,
        "family_fraction": fam_frac,
        "matched_top_m": top_m,
        "matched_top_m_fraction": top_m_frac,
        # <= 0 BY CONSTRUCTION (top-m is the argmax over size-m sets). Reported
        # as a sanity check on the implementation, never as evidence for H3.
        "family_minus_matched_top_m": fam_frac - top_m_frac,
        # the statistic that actually decides H3 (see harmonic_shape)
        "shape": harmonic_shape(power, fundamentals, p),
        "null_mean": float(draws.mean()),
        "null_std": float(draws.std(ddof=1)),
        "null_q95": float(np.quantile(draws, 0.95)),
        "null_n": int(n_null),
        "null_seed": int(seed),
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
