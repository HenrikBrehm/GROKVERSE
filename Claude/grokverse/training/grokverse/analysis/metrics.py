"""Structure metrics on 1-D periodic curves (docs/dev/INTERFACES.md §2).

Every function takes ``power [half, n]`` — per-frequency power per column, frequencies
``k = 1..half`` with ``half = (p-1)//2``, constant mode excluded — exactly the ``"power"``
array of ``mlp_mechanism.curve_spectra`` — unless the docstring says it takes a curve.
A curve matrix ``[p, n]`` (one column per neuron) can be turned into that array with
``power_spectrum`` here, which uses the same orthonormal basis and the same phase
convention (``y[n] = A cos(2*pi*k*n/p - phi)``, ``phi = atan2(coeff_sin, coeff_cos)``).

``q = power / power.sum(axis=0)`` is the normalized power distribution of a column and
is the object every concentration metric is defined on.

Conventions
-----------
* 1-D input (a single column / single curve) returns a scalar; 2-D input returns ``[n]``.
* A column with zero total power (or zero amplitude) has undefined metrics: it returns
  ``NaN`` — never a substitute value — and the wrappers count it in ``n_undefined_columns``.
* Every threshold is a parameter echoed in the output. Nothing here concludes; key names
  say what was measured.

Source status of the borrowed metrics
-------------------------------------
* ``periodicity_score``: Swaroop (2026), arXiv:2603.23784, Eq. 1 — the note
  ``docs/sources/swaroop2026_relu_mlp_square_waves.md`` §4 quotes the definition verbatim;
  implemented literally (see the docstring). Its thresholds (>12 / <5) were read off a
  histogram at ``p = 97`` and are ``p``-specific (score maximum is ``(p-1)/2``).
* ``binarization_score``: the same note records that the paper provides NO per-neuron
  binariness statistic (§2, "[NOT FOUND IN SOURCE]"), so this is OUR definition, named
  ``binarization_score_ours``.
* ``inverse_participation_ratio``: ``docs/sources/doshi2023_grok_or_not.md`` did not exist
  when this module was written (2026-09-02); the definition of Doshi et al. (arXiv:2310.13061)
  is therefore PENDING and only ``inverse_participation_ratio_ours`` is implemented
  (``sum_k q_k^2`` on the normalized half-spectrum power). Both names are exposed; the
  plain name is an alias of the ``_ours`` function until the source note lands.
"""
from __future__ import annotations

import numpy as np

from .fourier import MAX_ODD_HARMONIC, alias_frequency, fourier_basis

#: Status string echoed by the IPR functions (see module docstring).
IPR_DEFINITION_STATUS = ("ours_pending_source: docs/sources/doshi2023_grok_or_not.md absent at "
                         "implementation time (2026-09-02); IPR_ours = sum_k q_k^2 on the "
                         "normalized half-spectrum power")

#: Swaroop (2026) §3 thresholds on Eq. 1, chosen "to capture the clear gap in the bimodal
#: distribution" at p = 97. The score's maximum is (p-1)/2, so these do NOT transfer to
#: another p without re-deriving the gap (source note §4, own computation).
SWAROOP_PERIODICITY_THRESHOLDS = {"structured_gt": 12.0, "unstructured_lt": 5.0,
                                  "p_of_source": 97, "source": "arXiv:2603.23784 §3"}

#: Power ratios of the odd harmonics of an ideal CONTINUOUS square wave to its fundamental
#: (amplitudes decay as 1/j, so power as 1/j^2). Textbook Fourier series, not a source claim.
CONTINUOUS_SQUARE_POWER_RATIO = {3: 1.0 / 9.0, 5: 1.0 / 25.0, 7: 1.0 / 49.0}
CONTINUOUS_SQUARE_AMPLITUDE_RATIO = {3: 1.0 / 3.0, 5: 1.0 / 5.0, 7: 1.0 / 7.0}


# --------------------------------------------------------------------------- #
# input handling                                                               #
# --------------------------------------------------------------------------- #
def _as_matrix(x, name: str) -> tuple[np.ndarray, bool]:
    """Return ``(float64 [rows, n], was_1d)``; rejects non-finite values and >2 dims."""
    arr = np.asarray(x, dtype=np.float64)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} is empty: shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr, was_1d


def _as_power(power) -> tuple[np.ndarray, bool]:
    P, was_1d = _as_matrix(power, "power")
    if (P < 0).any():
        raise ValueError("power must be non-negative")
    return P, was_1d


def _squeeze(values: np.ndarray, was_1d: bool):
    return float(values[0]) if was_1d else values


def normalized_power(power) -> np.ndarray:
    """``q = power / sum_k power`` per column; a zero-power column becomes all-NaN."""
    P, _ = _as_power(power)
    total = P.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(total > 0, P / total, np.nan)
    return q


def power_spectrum(curves, p: int) -> dict:
    """Per-column half-spectrum power and phase of ``curves [p, n]`` (or ``[p]``).

    Same formula as ``mlp_mechanism.curve_spectra`` (asserted equal in tests): with the
    orthonormal basis rows ``cos_k, sin_k`` (``fourier.fourier_basis``),
    ``c_k = cos_k . y``, ``s_k = sin_k . y``, ``power_k = c_k^2 + s_k^2``,
    ``phase_k = atan2(s_k, c_k)`` so that ``y ~ A cos(2*pi*k*n/p - phase_k)``.
    Kept here (not imported) so that ``metrics`` stays import-free of ``mlp_mechanism``.
    """
    Y, _ = _as_matrix(curves, "curves")
    if Y.shape[0] != p:
        raise ValueError(f"curves have {Y.shape[0]} rows but p={p}")
    F, _ = fourier_basis(p)
    coeff = F @ Y
    half = (p - 1) // 2
    idx = np.arange(1, half + 1)
    c, s = coeff[2 * idx - 1], coeff[2 * idx]
    power = c ** 2 + s ** 2
    phase = np.arctan2(s, c)
    dom = power.argmax(axis=0)
    cols = np.arange(Y.shape[1])
    return {"power": power, "phase": phase, "freqs": idx, "constant_coeff": coeff[0],
            "dominant_freq": dom + 1, "dominant_phase": phase[dom, cols],
            "total_power": power.sum(axis=0)}


# --------------------------------------------------------------------------- #
# concentration metrics on the normalized power                                #
# --------------------------------------------------------------------------- #
def topk_concentration(power, k: int):
    """Share of power in the ``k`` largest frequencies: ``sum(top-k power) / sum(all power)``."""
    P, was_1d = _as_power(power)
    half = P.shape[0]
    k = int(k)
    if not 1 <= k <= half:
        raise ValueError(f"topk_concentration: k={k} must be in 1..{half}")
    q = normalized_power(P)
    top = np.sort(q, axis=0)[::-1][:k]
    return _squeeze(top.sum(axis=0), was_1d)


def spectral_entropy(power):
    """``H = -sum_k q_k log q_k / log(half)`` in [0, 1]; ``0 log 0 := 0``.

    0 for a single frequency, 1 for a perfectly flat spectrum. Note that the periodogram
    of white noise is NOT flat per realization (its bins are exponentially distributed),
    so a noise column scores below 1.
    """
    P, was_1d = _as_power(power)
    half = P.shape[0]
    if half < 2:
        raise ValueError("spectral_entropy needs at least 2 frequencies (log(half) = 0)")
    q = normalized_power(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0, q * np.log(q), 0.0)
    H = -terms.sum(axis=0) / np.log(half)
    H = np.where(np.isnan(q).any(axis=0), np.nan, H)
    return _squeeze(H, was_1d)


def participation_ratio(power):
    """``PR = (sum_k q_k)^2 / sum_k q_k^2 = 1 / sum_k q_k^2`` — effective number of frequencies.

    1 for a single frequency, ``half`` for a flat spectrum.
    """
    P, was_1d = _as_power(power)
    q = normalized_power(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = q.sum(axis=0) ** 2 / (q ** 2).sum(axis=0)
    return _squeeze(pr, was_1d)


def inverse_participation_ratio_ours(power):
    """``IPR_ours = sum_k q_k^2`` on the normalized half-spectrum power (``= 1 / PR``).

    OUR definition. The published IPR of Doshi et al. (arXiv:2310.13061) is to be taken
    from ``docs/sources/doshi2023_grok_or_not.md``, which did not exist when this was
    written — see ``IPR_DEFINITION_STATUS``. Whether the source defines it on the same
    object (half-spectrum power of a per-neuron curve) or another (full DFT, weights) is
    exactly what is pending; do not cite this function as the source's IPR.
    """
    P, was_1d = _as_power(power)
    q = normalized_power(P)
    return _squeeze((q ** 2).sum(axis=0), was_1d)


def inverse_participation_ratio(power):
    """Alias of ``inverse_participation_ratio_ours`` while the source definition is pending.

    Exposed under the plain name so callers written against INTERFACES §2 resolve; the
    value they get is ``IPR_ours`` (see ``IPR_DEFINITION_STATUS``).
    """
    return inverse_participation_ratio_ours(power)


def dominant_frequency(power):
    """1-based index ``k`` of the largest power per column (ties: the lowest ``k``)."""
    P, was_1d = _as_power(power)
    dom = P.argmax(axis=0) + 1
    dom = np.where(P.sum(axis=0) > 0, dom, 0)          # 0 = undefined (zero power)
    return int(dom[0]) if was_1d else dom


def dominant_fraction(power):
    """Share of power at the dominant frequency: ``max_k q_k``."""
    P, was_1d = _as_power(power)
    q = normalized_power(P)
    return _squeeze(q.max(axis=0), was_1d)


# --------------------------------------------------------------------------- #
# harmonic shares around each column's OWN fundamental                         #
# --------------------------------------------------------------------------- #
def harmonic_indices(k, p: int, max_harmonic: int = MAX_ODD_HARMONIC) -> np.ndarray:
    """``idx[j-1, col] = alias(j * k[col], p)`` for ``j = 1..max_harmonic`` (0 = constant)."""
    ks = np.atleast_1d(np.asarray(k, dtype=np.int64))
    half = (p - 1) // 2
    if (ks < 1).any() or (ks > half).any():
        raise ValueError(f"fundamental k must be in 1..{half} for p={p}, got {ks}")
    js = np.arange(1, int(max_harmonic) + 1)
    return np.array([[alias_frequency(int(j * kk), p) for kk in ks] for j in js])


def _harmonic_shares_core(q: np.ndarray, k: np.ndarray, p: int, max_harmonic: int) -> dict:
    """Shares of total power at each aliased harmonic ``j*k``; collisions counted, not summed."""
    half, n = q.shape
    idx = harmonic_indices(k, p, max_harmonic)                  # [J, n]
    J = idx.shape[0]
    dup = np.zeros(idx.shape, dtype=bool)
    for j in range(1, J):                                       # row j <-> harmonic j+1
        dup[j] = (idx[j] == 0) | (idx[j][None, :] == idx[:j]).any(axis=0)
    safe = np.where(idx > 0, idx, 1) - 1
    contrib = q[safe, np.arange(n)[None, :]]
    contrib = np.where(dup, 0.0, contrib)                       # a collision is not counted twice
    js = np.arange(1, J + 1)
    odd = (js % 2 == 1) & (js > 1)
    even = js % 2 == 0
    fundamental = contrib[0]
    odd_share = contrib[odd].sum(axis=0)
    even_share = contrib[even].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        odd_to_fund = np.where(fundamental > 0, odd_share / fundamental, np.nan)
        even_to_fund = np.where(fundamental > 0, even_share / fundamental, np.nan)
    return {
        "fundamental_share": fundamental,
        "odd_share": odd_share,
        "even_share": even_share,
        "family_share": fundamental + odd_share,
        "odd_minus_even": odd_share - even_share,
        "odd_to_fundamental_ratio": odd_to_fund,
        "even_to_fundamental_ratio": even_to_fund,
        "harmonic_share_by_j": contrib,                         # [J, n], row j-1 = harmonic j
        "harmonic_index_by_j": idx,                             # [J, n]
        "n_collisions": dup.sum(axis=0),
    }


def harmonic_shares(power, k, p: int, max_harmonic: int = MAX_ODD_HARMONIC,
                    reference_phi=0.0) -> dict:
    """Per column with its OWN fundamental ``k``: shares of total power at aliased harmonics.

    ``fundamental_share = q_k``; ``odd_share = sum_{j in {3,5,7}} q_{alias(jk)}``;
    ``even_share = sum_{j in {2,4,6}} q_{alias(jk)}``; ``family_share = fundamental + odd``;
    ``odd_minus_even``. Shares are fractions of the column's TOTAL (non-constant) power —
    unlike ``fourier.harmonic_shape`` whose shares are ratios to the fundamental; those
    ratios are given here as ``odd_to_fundamental_ratio`` / ``even_to_fundamental_ratio``.

    ``n_collisions`` counts harmonics ``j >= 2`` that alias onto the constant or onto an
    earlier harmonic (including ``k`` itself); a colliding harmonic contributes nothing so
    that no power is counted twice. For prime ``p > max_harmonic`` it is zero for every ``k``
    (asserted in tests).

    ``ideal_square_*`` are the same shares of ``sign(cos(2*pi*k*n/p - reference_phi))``
    sampled at ``n = 0..p-1`` (``discrete_square_reference``) — computed, not assumed.
    ``reference_phi`` is a scalar or a per-column array (e.g. the measured dominant phase).
    """
    P, was_1d = _as_power(power)
    half, n = P.shape
    if half != (p - 1) // 2:
        raise ValueError(f"power has {half} frequencies but p={p} implies {(p - 1) // 2}")
    ks = np.broadcast_to(np.atleast_1d(np.asarray(k, dtype=np.int64)), (n,)).copy()
    phis = np.broadcast_to(np.atleast_1d(np.asarray(reference_phi, dtype=np.float64)), (n,))
    q = normalized_power(P)
    out = _harmonic_shares_core(q, ks, p, int(max_harmonic))

    ref_keys = ("fundamental_share", "odd_share", "even_share", "family_share", "odd_minus_even")
    refs = {key: np.empty(n) for key in ref_keys}
    cache: dict[tuple[int, float], dict] = {}
    for col in range(n):
        key = (int(ks[col]), float(phis[col]))
        if key not in cache:
            cache[key] = discrete_square_reference(p, key[0], key[1], int(max_harmonic))
        for name in ref_keys:
            refs[name][col] = cache[key][name]
    for name in ref_keys:
        out[f"ideal_square_{name}"] = refs[name]
    out["params"] = {"p": int(p), "max_harmonic": int(max_harmonic),
                     "reference_phi": (float(reference_phi) if np.ndim(reference_phi) == 0
                                       else "per_column")}
    return _squeeze_dict(out) if was_1d else out


def _squeeze_dict(d: dict) -> dict:
    """Single-column results: ``[1]`` -> scalar, ``[J, 1]`` -> ``[J]``; other values unchanged."""
    out = {}
    for key, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == 1:
            out[key] = float(v[0]) if v.dtype.kind == "f" else int(v[0])
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
            out[key] = v[:, 0]
        else:
            out[key] = v
    return out


def discrete_square_reference(p: int, k: int, phi: float = 0.0,
                              max_harmonic: int = MAX_ODD_HARMONIC) -> dict:
    """Spectrum shares of the DISCRETE square wave ``sign(cos(2*pi*k*n/p - phi))``, ``n = 0..p-1``.

    ``cos >= 0`` maps to ``+1`` (a stated convention for the measure-zero zero crossings).
    At odd ``p`` the two levels have unequal counts, so the wave has a small constant
    component and small but nonzero even-harmonic power (aliasing) — both are reported
    rather than assumed away. ``power_ratio_to_fundamental[j] = power(alias(jk)) / power(k)``
    is to be read against the continuous ideal ``1/j^2`` (``continuous_ideal_power_ratio``).
    """
    p, k = int(p), int(k)
    if p < 3 or p % 2 == 0:
        raise ValueError(f"discrete_square_reference needs odd p >= 3, got {p}")
    n = np.arange(p)
    y = np.where(np.cos(2 * np.pi * k * n / p - float(phi)) >= 0, 1.0, -1.0)
    spec = power_spectrum(y, p)
    q = normalized_power(spec["power"])
    core = _harmonic_shares_core(q, np.array([k]), p, int(max_harmonic))
    shares = core["harmonic_share_by_j"][:, 0]
    fund = shares[0]
    js = list(range(3, int(max_harmonic) + 1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        power_ratio = {j: float(shares[j - 1] / fund) if fund > 0 else float("nan") for j in js}
    out = {name: float(core[name][0]) for name in
           ("fundamental_share", "odd_share", "even_share", "family_share", "odd_minus_even",
            "odd_to_fundamental_ratio", "even_to_fundamental_ratio")}
    out.update({
        "p": p, "k": k, "phi": float(phi), "max_harmonic": int(max_harmonic),
        "sign_convention": "cos >= 0 -> +1",
        "other_share": float(1.0 - core["family_share"][0] - core["even_share"][0]),
        "constant_share": float(y.mean() ** 2),          # share of sum(y^2) = p in the DC mode
        "n_positive_samples": int((y > 0).sum()),
        "n_collisions": int(core["n_collisions"][0]),
        "harmonic_share_by_j": shares,
        "power_ratio_to_fundamental": power_ratio,
        "amplitude_ratio_to_fundamental": {j: float(np.sqrt(v)) for j, v in power_ratio.items()},
        "continuous_ideal_power_ratio": {j: 1.0 / j ** 2 for j in js},
        "continuous_ideal_amplitude_ratio": {j: 1.0 / j for j in js},
        "power": spec["power"][:, 0],
        "curve": y,
    })
    return out


# --------------------------------------------------------------------------- #
# per-curve scores borrowed from / inspired by Swaroop (2026)                   #
# --------------------------------------------------------------------------- #
def periodicity_score(curve):
    """Swaroop (2026) Eq. 1, implemented literally on the full DFT of each column:

        per(w) = max_{k=1}^{p-1} |w_hat_k| / ( (1/(p-1)) * sum_{k=1}^{p-1} |w_hat_k| )

    with ``w_hat = fft(w)`` (numpy convention; the ratio is scale-invariant). Source:
    ``docs/sources/swaroop2026_relu_mlp_square_waves.md`` §4 (verbatim quote of Eq. 1).
    The maximum is ``(p-1)/2`` (a pure cosine), an ideal ``sign(cos)`` scores far lower
    (≈15.8 at p=97, source note own computation, re-derived in the tests), so a high score
    measures single-frequency dominance, not square-wave shape. Thresholds:
    ``SWAROOP_PERIODICITY_THRESHOLDS`` (p = 97 specific). ``NaN`` for an all-zero column.
    """
    Y, was_1d = _as_matrix(curve, "curve")
    p = Y.shape[0]
    if p < 3:
        raise ValueError(f"periodicity_score needs p >= 3, got {p}")
    mags = np.abs(np.fft.fft(Y, axis=0))[1:]                 # k = 1..p-1
    mean = mags.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(mean > 0, mags.max(axis=0) / mean, np.nan)
    return _squeeze(score, was_1d)


periodicity_score_swaroop = periodicity_score


def periodicity_score_from_power(power):
    """Eq. 1 evaluated from half-spectrum ``power``: ``max sqrt(power) / mean sqrt(power)``.

    Identical to ``periodicity_score`` for real curves at odd ``p`` because ``|w_hat_k| =
    |w_hat_{p-k}|`` and ``power_k ∝ |w_hat_k|^2`` (equality asserted in tests).
    """
    P, was_1d = _as_power(power)
    amp = np.sqrt(P)
    mean = amp.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(mean > 0, amp.max(axis=0) / mean, np.nan)
    return _squeeze(score, was_1d)


def periodicity_score_max(p: int) -> float:
    """Value of Eq. 1 for a pure single-frequency cosine at this ``p``: ``(p-1)/2``."""
    return (int(p) - 1) / 2.0


def binarization_score_ours(curve, threshold: float = 0.8):
    """OUR near-binary score: fraction of samples with ``|y| / max|y| > threshold``.

    Swaroop (2026) describes "near-binary" weights (levels ``ε ± A``) but gives no
    per-neuron statistic (source note §2), so this definition is ours and is labelled
    so. 1.0 for an ideal square wave; ≈ ``2 arccos(threshold)/pi`` (≈0.41 at 0.8) for a
    pure sinusoid. ``NaN`` for an all-zero column. The threshold is echoed by name.
    """
    Y, was_1d = _as_matrix(curve, "curve")
    thr = float(threshold)
    if not 0.0 < thr < 1.0:
        raise ValueError(f"binarization threshold must be in (0, 1), got {thr}")
    m = np.abs(Y).max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(m > 0, (np.abs(Y) / np.where(m > 0, m, 1.0) > thr).mean(axis=0), np.nan)
    return _squeeze(frac, was_1d)


binarization_score = binarization_score_ours


# --------------------------------------------------------------------------- #
# thin wrapper over a curve matrix                                             #
# --------------------------------------------------------------------------- #
def curve_metrics(curves, p: int, topk: tuple[int, ...] = (1, 4, 8),
                  max_harmonic: int = MAX_ODD_HARMONIC,
                  binarization_threshold: float = 0.8) -> dict:
    """Every §2 metric per column of ``curves [p, n]``, each at its own dominant frequency.

    The harmonic reference uses each column's measured dominant phase. Returns per-column
    arrays plus ``params`` (echoing ``topk``, ``max_harmonic``, ``binarization_threshold``)
    and ``n_undefined_columns`` (zero-power columns, whose metrics are NaN).
    """
    Y, _ = _as_matrix(curves, "curves")
    spec = power_spectrum(Y, p)
    P = spec["power"]
    defined = spec["total_power"] > 0
    if not defined.any():
        raise ValueError("every column has zero spectral power — nothing to measure")
    k = np.where(defined, spec["dominant_freq"], 1)     # placeholder k for undefined cols only
    shares = harmonic_shares(P, k, p, max_harmonic, reference_phi=spec["dominant_phase"])
    for key, v in shares.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            shares[key] = np.where(defined, v, np.nan)
    return {
        "params": {"p": int(p), "topk": [int(t) for t in topk], "max_harmonic": int(max_harmonic),
                   "binarization_threshold": float(binarization_threshold),
                   "ipr_definition": IPR_DEFINITION_STATUS},
        "n_columns": int(Y.shape[1]),
        "n_undefined_columns": int((~defined).sum()),
        "dominant_frequency": np.where(defined, spec["dominant_freq"], 0),
        "dominant_phase": np.where(defined, spec["dominant_phase"], np.nan),
        "dominant_fraction": dominant_fraction(P),
        "topk_concentration": {int(t): topk_concentration(P, t) for t in topk},
        "spectral_entropy": spectral_entropy(P),
        "participation_ratio": participation_ratio(P),
        "inverse_participation_ratio_ours": inverse_participation_ratio_ours(P),
        "periodicity_score_swaroop": periodicity_score(Y),
        "binarization_score_ours": binarization_score_ours(Y, binarization_threshold),
        "harmonic_shares": shares,
        "total_power": spec["total_power"],
    }
