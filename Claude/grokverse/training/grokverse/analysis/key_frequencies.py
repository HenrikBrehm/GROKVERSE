"""Key-frequency selection rules (docs/dev/INTERFACES.md §4).

WHAT A "KEY FREQUENCY" IS, AND WHY THERE ARE FIVE RULES
-------------------------------------------------------
Every downstream module that talks about "the circuit" — the restricted/excluded
protocols (§10), the end-to-end logit fit (§7), the causal ablations (§9) — needs a
set of frequencies ``K ⊂ {1..(p−1)/2}``. That set is *not* an observation: it is the
output of a **rule**, and different published rules disagree. This module implements
the rules side by side so that the choice is a recorded parameter instead of an
accident, and reports the pairwise Jaccard overlap between them.

``docs/sources/nanda2023_progress_measures.md`` §3 finds **three** different rules in
the primary sources: (a) Fourier norms of ``W_E`` (paper §4.1), (b) a DFT of the
neuron→logit map ``W_L`` with "nontrivial coefficients" (paper App. C.2 — the rule
actually used for the progress measures), (c) an argmax over per-neuron 2-D
same-frequency blocks (the released Colab). No source publishes a numeric threshold.

THE PRE-REGISTERED PRIMARY RULE
-------------------------------
``docs/PREREGISTRATION.md`` §4.3 fixes the primary rule as ``nanda`` — rule (b) —
with a **0.25-of-maximum** threshold and mandatory sensitivity at **0.10** and
**0.50**. The 0.25 is declared as *ours* (following the TransformerLens demo's
``> max/4`` convention); the source states no number
(``[NOT FOUND IN SOURCE]``, MASK_PROTOCOL_AUDIT §1 row 7). **The count is measured
and never capped** — that is the whole point of replacing the legacy top-8 rule,
whose cap binds on all 16 legacy runs so its count was never data-determined.

Nothing in this module tunes a threshold. Every threshold is a module constant
carrying its provenance, and every result echoes the threshold it used.

THE RULES
---------
=========================  ====================================================
``nanda`` (PRIMARY)        DFT of ``W_L`` along the CLASS axis, power summed over
                           neurons, norm = its square root; keep every ``k`` with
                           ``norm_k >= threshold_frac * max_k norm_k``. Uncapped.
                           ``W_L`` = ``W_out @ W_U[:, :p]`` (transformer) or
                           ``W_out`` (mlp / mlp_twohot), both ``[d_mlp, p]``.
``embedding_top8``         LEGACY. Top-8 of the ``W_E`` power spectrum, exactly as
                           ``fourier.dominant_frequencies`` has always computed it
                           (smallest set reaching 90 % of the power, then capped at
                           8). Reported only as the legacy number; its
                           ``cap_binding`` flag is the audit trail.
``embedding_threshold``    Smallest set reaching 90 % of the ``W_E`` power,
                           UNCAPPED — the same threshold without the cap, so the
                           count is measured.
``logit_sum_directions``   2-D Fourier of the centred full-grid logits; per ``k``
                           the share of power in the ``cos/sin(w_k(a+b))``
                           directions (``mask_protocols.SumDirectionsOnly`` applied
                           to that single ``k``); keep ``k`` whose share exceeds
                           three times uniform, ``3/half``.
``neuron_clusters``        Per hidden neuron the dominant frequency of the
                           effective operand curves; keep the frequencies that are
                           dominant for >= ``min_neurons`` neurons AND whose
                           neurons jointly carry >= 2 % of the total hidden
                           activation variance.
=========================  ====================================================

WHAT IS *NOT* CLAIMED
---------------------
``embedding_top8`` is not a reproduction of anything: it is this repository's own
legacy rule, kept runnable so old numbers stay comparable. ``nanda`` reproduces the
published *object* and *operation* (App. C.2) but not a published threshold, because
none exists. ``neuron_clusters`` is inspired by the Colab's cluster rule but is not
it (the Colab uses the 2-D 3×3 block of the neuron activations, we use the 1-D
operand curves), so it is reported under its own name.

PURITY / IMPORT DISCIPLINE
--------------------------
``select_from_arrays`` is a pure function of numpy arrays and imports nothing
heavier than numpy — the test suite exercises every rule without a run directory.
``common`` (torch), ``progress_measures`` (matplotlib) and ``mlp_mechanism`` are
imported **lazily**, inside the functions that need them, so that an unrelated
module being mid-edit cannot break this one.

Usage (from training/)::

    python -m grokverse.analysis.key_frequencies runs/<run> --step 25000 --rule all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics as _metrics
from .fourier import dominant_frequencies, embedding_power_spectrum, fourier_basis
from .mask_protocols import SumDirectionsOnly

MODULE = "key_frequencies"
MODULE_VERSION = "1.0"

#: The pre-registered PRIMARY key-frequency rule, read from
#: ``docs/PREREGISTRATION.md`` §4.3 ("**Primary: `nanda`.**") and restated in
#: ``docs/dev/INTERFACES.md`` §4. Hard-coded so that no caller can quietly select a
#: different rule; changing this string changes a pre-registered choice and requires
#: a pre-registration amendment, not a code edit.
PRIMARY_RULE = "nanda"

#: ``nanda`` threshold: keep ``k`` with ``norm_k >= NANDA_THRESHOLD_FRAC * max norm``.
#: docs/PREREGISTRATION.md §4.3: "keep every `k` whose norm is >= 0.25 x the maximum";
#: the published rule states no number, so 0.25 is OURS (TransformerLens `> max/4`).
NANDA_THRESHOLD_FRAC = 0.25
#: Mandatory sensitivity thresholds, always reported next to the primary (§4.3).
NANDA_SENSITIVITY_FRACS: tuple[float, ...] = (0.10, 0.50)

#: ``embedding_top8`` / ``embedding_threshold``: the cumulative-power target.
EMBEDDING_POWER_THRESHOLD = 0.90
#: The legacy cap. It exists to be *reported as binding*, never to be tuned.
EMBEDDING_TOP_K = 8

#: ``logit_sum_directions``: keep ``k`` whose sum-direction share exceeds
#: ``LOGIT_SHARE_MULTIPLE / half`` — three times the uniform share (INTERFACES §4).
LOGIT_SHARE_MULTIPLE = 3.0

#: ``neuron_clusters``: a frequency must be dominant for at least this many neurons
#: and its neurons must jointly carry at least this share of the hidden-activation
#: variance (INTERFACES §4).
NEURON_CLUSTER_MIN_NEURONS = 5
NEURON_CLUSTER_MIN_VARIANCE_SHARE = 0.02

#: Every rule name, primary first. Must stay equal to ``logit_formula_fit.KEY_RULES``.
RULES: tuple[str, ...] = ("nanda", "neuron_clusters", "embedding_threshold",
                          "logit_sum_directions", "embedding_top8")


# --------------------------------------------------------------------------- #
# score curves (pure numpy)                                                    #
# --------------------------------------------------------------------------- #
def neuron_logit_norms(W_L, p: int) -> np.ndarray:
    """Per-frequency Fourier norm of the neuron→logit map, over neurons.

    The published operation (Nanda et al. 2023 App. C.2, quoted in
    ``docs/sources/nanda2023_progress_measures.md`` §3): "we perform a DFT on the
    neuron-logit map W_L, then take the frequencies with nontrivial coefficients".

    With the orthonormal real basis of ``fourier.fourier_basis`` applied along the
    **class** axis, ``c_k[i] = cos_k . W_L[i]`` and ``s_k[i] = sin_k . W_L[i]``, the
    per-frequency power is ``P_k = sum_i (c_k[i]^2 + s_k[i]^2)`` and the returned
    norm is ``sqrt(P_k)`` — the norm of the [d_mlp]-vector of that frequency's
    coefficients, i.e. the "norm over neurons" of INTERFACES §4.

    Returns ``[half]`` indexed by ``k − 1`` for ``k = 1..(p−1)/2``.
    """
    W = np.asarray(W_L, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"W_L must be 2-D [d_mlp, p], got shape {W.shape}")
    if W.shape[1] != p:
        raise ValueError(f"W_L has {W.shape[1]} columns but p={p}: the DFT is taken "
                         "along the CLASS axis, which must be the last one")
    if not np.isfinite(W).all():
        raise ValueError("W_L contains non-finite entries")
    F, _ = fourier_basis(p)
    coeff = W @ F.T                                   # [d_mlp, p] in basis coordinates
    half = (p - 1) // 2
    k = np.arange(1, half + 1)
    c, s = coeff[:, 2 * k - 1], coeff[:, 2 * k]       # cos_k row 2k-1, sin_k row 2k
    return np.sqrt((c ** 2 + s ** 2).sum(axis=0))


def sum_direction_shares(logits, p: int, centre: bool = True) -> np.ndarray:
    """Share of logit power in the ``cos/sin(w_k(a+b))`` directions, per frequency.

    ``share_k = ||P_k Lhat||^2 / ||Lhat_nonconstant||^2`` where ``P_k`` is the
    orthogonal projection of ``mask_protocols.SumDirectionsOnly`` built on the single
    frequency ``k``, and ``Lhat = fwd2d(centre(L))`` is the 2-D transform over the
    two input axes (INTERFACES §0). The denominator excludes the ``(0,0)`` constant
    mode, so a spectrum whose power sat entirely in sum directions and was spread
    evenly over the ``half`` frequencies would give ``1/half`` each — the uniform
    reference the ``LOGIT_SHARE_MULTIPLE`` threshold is measured against.

    Pure ``(a−b)`` structure projects to exactly zero here (the ``(A−B)`` and
    ``(C+D)`` combinations both vanish), which is what makes this a usable negative
    control rather than a rule that fires on any periodic tensor.
    """
    L = np.asarray(logits, dtype=np.float64)
    if L.ndim != 3 or L.shape[0] != p or L.shape[1] != p:
        raise ValueError(f"logits must be [p, p, n_classes] with p={p}, got {L.shape}")
    if not np.isfinite(L).all():
        raise ValueError("logits contain non-finite entries")
    if centre:
        L = L - L.mean(axis=-1, keepdims=True)
    # lazy: progress_measures pulls in matplotlib + torch; the pure-array path stays light
    from .progress_measures import fwd2d
    F, _ = fourier_basis(p)
    Lhat = fwd2d(L, F)
    const = float((Lhat[0, 0] ** 2).sum())
    total = float((Lhat ** 2).sum()) - const
    half = (p - 1) // 2
    shares = np.zeros(half, dtype=np.float64)
    if total <= 0.0:
        return shares
    for k in range(1, half + 1):
        kept = SumDirectionsOnly(p, [k]).restrict(Lhat)   # constant + the 2-D sum subspace
        shares[k - 1] = (float((kept ** 2).sum()) - const) / total
    return np.maximum(shares, 0.0)                        # clip float round-off at zero


def activation_variance(u_a, u_b, b_in=None, chunk: int = 64) -> np.ndarray:
    """Per-neuron variance of ``ReLU(u_a[a] + u_b[b] + b_in)`` over the full grid.

    Chunked over neurons so the ``[p, p, d_mlp]`` activation tensor is never
    materialized in full. Returns ``[d_mlp]``.
    """
    A = np.asarray(u_a, dtype=np.float64)
    B = np.asarray(u_b, dtype=np.float64)
    if A.ndim != 2 or B.shape != A.shape:
        raise ValueError(f"u_a and u_b must both be [p, d_mlp]; got {A.shape} and {B.shape}")
    n = A.shape[1]
    bias = np.zeros(n) if b_in is None else np.asarray(b_in, dtype=np.float64).ravel()
    if bias.shape != (n,):
        raise ValueError(f"b_in must have one entry per neuron ({n}), got {bias.shape}")
    var = np.empty(n, dtype=np.float64)
    for start in range(0, n, int(chunk)):
        end = min(start + int(chunk), n)
        pre = A[:, None, start:end] + B[None, :, start:end] + bias[None, None, start:end]
        act = np.maximum(pre, 0.0).reshape(-1, end - start)
        var[start:end] = act.var(axis=0)
    return var


def neuron_dominant_frequencies(u_a, u_b, p: int) -> np.ndarray:
    """Per-neuron dominant frequency of the operand curves (0 = undefined).

    The two operand curves are pooled (``power_a + power_b``) before the argmax, so a
    neuron gets one frequency rather than two that may disagree; the per-curve
    agreement itself is §5's ``neuron_tables`` business, not this rule's.
    """
    spec_a = _metrics.power_spectrum(u_a, p)
    spec_b = _metrics.power_spectrum(u_b, p)
    return np.asarray(_metrics.dominant_frequency(spec_a["power"] + spec_b["power"]))


# --------------------------------------------------------------------------- #
# rule implementations — each returns the §4 result dict for one rule          #
# --------------------------------------------------------------------------- #
def _threshold_of_max(scores: np.ndarray, frac: float) -> tuple[list[int], float]:
    """Frequencies with ``score >= frac * max(score)``; an all-zero curve selects none."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return [], 0.0
    if not np.isfinite(s).all():
        raise ValueError("score curve contains non-finite entries")
    top = float(s.max())
    if top <= 0.0:                       # no structure at all: empty set, n = 0, no crash
        return [], 0.0
    cut = float(frac) * top
    return [int(k) for k in (np.nonzero(s >= cut)[0] + 1)], cut


def _rule_nanda(p: int, W_L, threshold_frac: float, sensitivity) -> dict:
    """PRIMARY rule (docs/PREREGISTRATION.md §4.3). The count is measured, never capped."""
    if W_L is None:
        raise ValueError("rule 'nanda' needs W_L, the neuron->logit map [d_mlp, p] "
                         "(transformer: W_out @ W_U[:, :p]; mlp/mlp_twohot: W_out)")
    norms = neuron_logit_norms(W_L, p)
    keys, cut = _threshold_of_max(norms, threshold_frac)
    sens = {f"{float(f):.2f}": _threshold_of_max(norms, f)[0] for f in sensitivity}
    return {
        "score_name": "fourier_norm_over_neurons_of_W_L_along_class_axis",
        "scores": norms.tolist(),
        "key_frequencies": keys,
        "threshold_frac": float(threshold_frac),
        "threshold_value": cut,
        "max_score": float(norms.max()) if norms.size else 0.0,
        "threshold_kind": "frac_of_max_norm",
        # Never capped — that is the pre-registered difference from embedding_top8.
        "cap_binding": False,
        "cap": None,
        "n_neurons": int(np.asarray(W_L).shape[0]),
        "sensitivity": sens,
        "sensitivity_n": {k: len(v) for k, v in sens.items()},
        "threshold_provenance": "0.25 is OURS (TransformerLens '> max/4'); the published "
                                "rule states no numeric threshold [NOT FOUND IN SOURCE]",
    }


def _embedding_rule(rule: str, p: int, W_E, power_threshold: float, max_k: int) -> dict:
    """Shared body of ``embedding_top8`` (capped) and ``embedding_threshold`` (uncapped)."""
    if W_E is None:
        raise ValueError(f"rule {rule!r} needs W_E, the [>=p, d] embedding matrix")
    W = np.asarray(W_E, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] < p:
        raise ValueError(f"W_E must be [>= p, d] with p={p}, got shape {W.shape}")
    spec = embedding_power_spectrum(W[:p], p)
    half = (p - 1) // 2
    if spec["total_power"] <= 0.0:        # no power anywhere: empty set, n = 0, no crash
        return {"score_name": "embedding_power_fraction", "scores": [0.0] * half,
                "key_frequencies": [], "ranked_freqs": [], "cumulative_fraction": 0.0,
                "n_freqs_for_threshold": 0, "cap_binding": False, "cap": int(max_k),
                "threshold_frac": float(power_threshold),
                "threshold_kind": "cumulative_power_fraction", "threshold_value": None}
    dom = dominant_frequencies(W[:p], p, threshold=power_threshold, max_k=max_k)
    return {
        "score_name": "embedding_power_fraction",
        "scores": [float(x) for x in spec["fraction"]],
        "key_frequencies": sorted(int(k) for k in dom["dominant"]),
        "ranked_freqs": [int(k) for k in dom["ranked_freqs"]],
        "cumulative_fraction": float(dom["dominant_fraction"]),
        "n_freqs_for_threshold": int(dom["n_freqs_for_threshold"]),
        "cap_binding": bool(dom["cap_binding"]),
        "cap": int(max_k),
        "threshold_frac": float(power_threshold),
        "threshold_kind": "cumulative_power_fraction",
        "threshold_value": None,
        "const_power_fraction": float(spec["const_power"]
                                      / (spec["const_power"] + spec["total_power"])),
    }


def _rule_logit_sum_directions(p: int, logits, share_multiple: float) -> dict:
    """2-D sum-direction share per k, against three times the uniform share."""
    if logits is None:
        raise ValueError("rule 'logit_sum_directions' needs logits [p, p, n_classes]")
    shares = sum_direction_shares(logits, p)
    half = (p - 1) // 2
    cut = float(share_multiple) / half
    keys = [int(k) for k in (np.nonzero(shares > cut)[0] + 1)]
    return {
        "score_name": "sum_direction_share_of_centred_logit_power",
        "scores": shares.tolist(),
        "key_frequencies": keys,
        "threshold_frac": cut,
        "threshold_value": cut,
        "threshold_kind": "multiple_of_uniform_share",
        "share_multiple": float(share_multiple),
        "uniform_share": 1.0 / half,
        "total_share_in_sum_directions": float(shares.sum()),
        "cap_binding": False,
        "cap": None,
    }


def _rule_neuron_clusters(p: int, curves, hidden_variance,
                          min_neurons: int, min_variance_share: float) -> dict:
    """Frequencies carried by a cluster of neurons that is both large and loud."""
    if curves is None:
        raise ValueError("rule 'neuron_clusters' needs curves={'u_a':..., 'u_b':..., "
                         "'b_in':...} (effective operand curves, [p, d_mlp])")
    u_a = np.asarray(curves["u_a"], dtype=np.float64)
    u_b = np.asarray(curves["u_b"], dtype=np.float64)
    dom = neuron_dominant_frequencies(u_a, u_b, p)
    var = (activation_variance(u_a, u_b, curves.get("b_in"))
           if hidden_variance is None else np.asarray(hidden_variance, dtype=np.float64))
    if var.shape != (u_a.shape[1],):
        raise ValueError(f"hidden_variance must be [{u_a.shape[1]}], got {var.shape}")
    total = float(var.sum())
    half = (p - 1) // 2
    counts = np.array([int((dom == k).sum()) for k in range(1, half + 1)])
    shares = np.array([float(var[dom == k].sum() / total) if total > 0 else 0.0
                       for k in range(1, half + 1)])
    keep = (counts >= int(min_neurons)) & (shares >= float(min_variance_share))
    return {
        "score_name": "hidden_activation_variance_share_per_dominant_frequency",
        "scores": shares.tolist(),
        "neuron_counts": counts.tolist(),
        "key_frequencies": [int(k) for k in (np.nonzero(keep)[0] + 1)],
        "threshold_frac": float(min_variance_share),
        "threshold_value": float(min_variance_share),
        "threshold_kind": "min_neurons_and_variance_share",
        "min_neurons": int(min_neurons),
        "n_neurons": int(u_a.shape[1]),
        "n_neurons_with_undefined_frequency": int((dom == 0).sum()),
        "total_activation_variance": total,
        "cap_binding": False,
        "cap": None,
        "_arrays": {"neuron_dominant_frequency": dom, "neuron_activation_variance": var},
    }


# --------------------------------------------------------------------------- #
# the pure-function core                                                       #
# --------------------------------------------------------------------------- #
def select_from_arrays(rule: str, p: int, *,
                       W_L=None, W_E=None, logits=None, curves=None,
                       hidden_variance=None,
                       threshold_frac: float = NANDA_THRESHOLD_FRAC,
                       power_threshold: float = EMBEDDING_POWER_THRESHOLD,
                       top_k: int = EMBEDDING_TOP_K,
                       share_multiple: float = LOGIT_SHARE_MULTIPLE,
                       min_neurons: int = NEURON_CLUSTER_MIN_NEURONS,
                       min_variance_share: float = NEURON_CLUSTER_MIN_VARIANCE_SHARE,
                       sensitivity=NANDA_SENSITIVITY_FRACS) -> dict:
    """Apply one §4 rule to already-extracted arrays — no run directory needed.

    Every rule returns ``{"rule", "key_frequencies", "n", "scores", "cap_binding",
    "threshold_frac", ...}``. ``key_frequencies`` is sorted ascending (a key set is a
    set); ``scores`` is the FULL per-frequency score curve, ``scores[k-1]`` for
    ``k = 1..(p−1)/2``, never truncated. A rule that selects nothing returns an empty
    list with ``n = 0`` — that is a measurement, not an error.

    An unknown ``rule`` raises ``ValueError``; a rule whose input array is missing
    raises ``ValueError`` naming the array it needs. ``neuron_clusters`` additionally
    returns ``_arrays`` (per-neuron dominant frequency and activation variance), which
    ``analyse`` moves into the ``.npz`` and ``select`` drops.
    """
    if int(p) < 3 or int(p) % 2 == 0:
        raise ValueError(f"p must be an odd integer >= 3 (got {p}); the cos/sin basis needs odd p")
    p = int(p)
    if rule == "nanda":
        out = _rule_nanda(p, W_L, threshold_frac, sensitivity)
    elif rule == "embedding_top8":
        out = _embedding_rule(rule, p, W_E, power_threshold, int(top_k))
    elif rule == "embedding_threshold":
        out = _embedding_rule(rule, p, W_E, power_threshold, (p - 1) // 2)
    elif rule == "logit_sum_directions":
        out = _rule_logit_sum_directions(p, logits, share_multiple)
    elif rule == "neuron_clusters":
        out = _rule_neuron_clusters(p, curves, hidden_variance, min_neurons, min_variance_share)
    else:
        raise ValueError(f"unknown key-frequency rule {rule!r}; INTERFACES §4 rules: {list(RULES)}")
    out["rule"] = rule
    out["p"] = p
    out["n"] = len(out["key_frequencies"])
    out["is_primary_rule"] = (rule == PRIMARY_RULE)
    return out


def agreement(sets) -> dict:
    """Pairwise Jaccard overlap between key-frequency sets (INTERFACES §4).

    ``sets`` is a mapping ``name -> iterable of k`` (order preserved) or a sequence of
    iterables (named by position). ``J(A, B) = |A ∩ B| / |A ∪ B|``, with the empty /
    empty pair defined as ``1.0`` so the diagonal is 1.0 for every set including the
    empty one. Returns ``{"names", "sizes", "jaccard", "intersection", "union"}``.
    """
    if isinstance(sets, dict):
        names = [str(k) for k in sets]
        values = [set(int(x) for x in sets[k]) for k in sets]
    else:
        values = [set(int(x) for x in s) for s in sets]
        names = [str(i) for i in range(len(values))]
    n = len(values)
    jac = [[0.0] * n for _ in range(n)]
    inter = [[0] * n for _ in range(n)]
    union = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a, b = values[i], values[j]
            inter[i][j] = len(a & b)
            union[i][j] = len(a | b)
            jac[i][j] = 1.0 if union[i][j] == 0 else inter[i][j] / union[i][j]
    return {"names": names, "sizes": [len(v) for v in values],
            "jaccard": jac, "intersection": inter, "union": union}


# --------------------------------------------------------------------------- #
# extracting the arrays from a run directory                                   #
# --------------------------------------------------------------------------- #
def neuron_logit_map(state: dict, cfg) -> np.ndarray:
    """``W_L [d_mlp, p]`` — the neuron→logit map of INTERFACES §4 / audit §1 row 7.

    Transformer: ``W_out @ W_U[:, :p]`` (``W_out`` is ``[d_mlp, d_model]`` and writes
    into the residual stream, so the map to logits goes through the unembedding).
    MLP and two-hot MLP: ``W_out`` itself, already ``[d_mlp, p]``.
    """
    if "W_out" not in state:
        raise ValueError(f"checkpoint has no W_out — no neuron->logit map for arch={cfg.arch!r}")
    W_out = np.asarray(state["W_out"], dtype=np.float64)
    p = int(cfg.p)
    if cfg.arch == "transformer":
        if "W_U" not in state:
            raise ValueError("transformer checkpoint has no W_U — cannot form W_out @ W_U[:, :p]")
        return W_out @ np.asarray(state["W_U"], dtype=np.float64)[:, :p]
    if cfg.arch in ("mlp", "mlp_twohot"):
        return W_out[:, :p]
    raise ValueError(f"no neuron->logit map defined for arch={cfg.arch!r}")


def embedding_matrix(state: dict, cfg) -> tuple[np.ndarray, str]:
    """The ``[p, d]`` object the embedding rules read, plus the name of that object.

    ``mlp_twohot`` has no embedding table; its closest analogue is the ``a``-half of
    ``W_in`` (``models/mlp_twohot.py``: a one-hot row-select *is* an index into
    ``W_in``), logged in ``run.json`` as ``embedding_object = "W_in_a_half"``. The name
    is returned so no reader can mistake it for a real ``W_E``.
    """
    p = int(cfg.p)
    if cfg.arch in ("transformer", "mlp"):
        if "W_E" not in state:
            raise ValueError(f"checkpoint has no W_E for arch={cfg.arch!r}")
        return np.asarray(state["W_E"], dtype=np.float64)[:p], "W_E"
    if cfg.arch == "mlp_twohot":
        if "W_in" not in state:
            raise ValueError("two-hot checkpoint has no W_in")
        return np.asarray(state["W_in"], dtype=np.float64)[:p], "W_in_a_half"
    raise ValueError(f"no embedding object defined for arch={cfg.arch!r}")


def operand_curves(state: dict, cfg, run_dir=None) -> tuple[dict, str]:
    """Effective operand curves ``u_a``, ``u_b`` (``[p, d_mlp]``) plus their source.

    * ``mlp`` / ``mlp_twohot``: from ``analysis.mlp_mechanism.effective_curves``
      (INTERFACES §5), imported **lazily** so that module being mid-edit cannot break
      this one. If it is absent the error names it; if it does not yet accept
      ``mlp_twohot`` the exact curves stated in ``models/mlp_twohot.py`` are used
      (``u_a = W_in[:p]``, ``u_b = W_in[p:]``) and the fallback is recorded in the
      returned source string.
    * ``transformer``: the proper effective operand curves from
      ``analysis.transformer_mechanism.effective_curves`` (INTERFACES §6) — the two
      operand positions weighted by the grid-mean read-out attention, carrying
      ``W_pos`` and the ``'='`` position's constant. Imported **lazily**, like the MLP
      branch, so that module being mid-edit cannot break this one. If it is absent the
      old placeholder ``W_E[:p] @ W_in`` (no attention, no ``W_pos``, the same curve for
      both operands) is used and *says so* in the returned source string; numbers from
      that path must never be read as the transformer's effective curves.
    """
    p, where = int(cfg.p), (str(run_dir) if run_dir is not None else "<state>")
    if cfg.arch == "transformer":
        for name in ("W_E", "W_in"):
            if name not in state:
                raise ValueError(f"{where}: transformer checkpoint has no {name}")
        try:
            from . import transformer_mechanism as TM   # lazy: §6 module, may be mid-edit
            attn = TM.attention_weights({k: np.asarray(v, dtype=np.float64)
                                         for k, v in state.items()}, cfg)
            abar = attn.reshape(p * p, cfg.n_heads, 3).mean(axis=0)
            c = TM.effective_curves(state, cfg, abar)
            return ({"u_a": c["u_a"], "u_b": c["u_b"], "b_in": c["b_in"]},
                    "transformer_mechanism.effective_curves (mean-attention, INTERFACES §6)")
        except ImportError:                             # pragma: no cover - sibling edit
            u = np.asarray(state["W_E"], dtype=np.float64)[:p] @ np.asarray(state["W_in"], float)
            return ({"u_a": u, "u_b": u, "b_in": None},
                    "transformer_direct_W_E_at_W_in (PLACEHOLDER — "
                    "analysis.transformer_mechanism unavailable)")
    if cfg.arch not in ("mlp", "mlp_twohot"):
        raise ValueError(f"{where}: no operand curves defined for arch={cfg.arch!r}")
    try:
        from . import mlp_mechanism            # lazy: §5 module, may be mid-edit
    except ImportError as exc:                 # pragma: no cover - depends on a sibling edit
        raise ValueError(
            f"{where}: rule 'neuron_clusters' needs grokverse.analysis.mlp_mechanism "
            f"(INTERFACES §5) for arch={cfg.arch!r}, which failed to import ({exc}). "
            "Retry once that module is importable, or pass curves=... explicitly.") from exc
    try:
        c = mlp_mechanism.effective_curves(state, cfg)
        return ({"u_a": c["u_a"], "u_b": c["u_b"], "b_in": c.get("b_in")},
                "mlp_mechanism.effective_curves")
    except ValueError as exc:
        if cfg.arch != "mlp_twohot":
            raise ValueError(f"{where}: mlp_mechanism.effective_curves failed: {exc}") from exc
    W_in = np.asarray(state["W_in"], dtype=np.float64)      # documented two-hot identity
    return ({"u_a": W_in[:p], "u_b": W_in[p:], "b_in": state.get("b_in")},
            "models/mlp_twohot.py identity u_a=W_in[:p], u_b=W_in[p:] "
            "(mlp_mechanism.effective_curves does not accept this arch yet)")


# --------------------------------------------------------------------------- #
# run-directory entry points                                                   #
# --------------------------------------------------------------------------- #
def select(run_dir, step: int | None = None, rule: str = PRIMARY_RULE, **kw) -> dict:
    """Apply one §4 rule to a run's checkpoint. See ``select_from_arrays`` for the keys.

    ``step=None`` reads the legacy ``model_final.pt``; an integer reads that v2
    checkpoint. Extra keyword arguments (``threshold_frac``, ``min_neurons``, ...) are
    forwarded to ``select_from_arrays``; ``curves=`` may supply precomputed effective
    operand curves so a caller that already has them does not recompute them.
    """
    from . import common                        # lazy: torch-heavy, not needed by the core
    if rule not in RULES:
        raise ValueError(f"unknown key-frequency rule {rule!r}; INTERFACES §4 rules: {list(RULES)}")
    cfg, model, state, meta = common.load_model_at(run_dir, step)
    curves = kw.pop("curves", None)
    inputs, provenance = _gather_inputs((rule,), cfg, model, state, curves, run_dir)
    out = select_from_arrays(rule, cfg.p, **_subset(inputs, rule), **kw)
    out.pop("_arrays", None)
    out.update({"run_id": meta["run_id"], "arch": meta["arch"], "step": meta["step"],
                "source": provenance})
    return out


def analyse(run_dir, step: int | None = None, rules=RULES, seed: int = 0, **kw) -> dict:
    """Run the requested rules on one checkpoint and write the §0 result envelope.

    Writes ``<run_dir>/analysis/key_frequencies/<tag>.json`` (every rule's key set,
    its full score curve, its ``cap_binding`` flag and the pairwise Jaccard matrix)
    plus ``<tag>.npz`` with the score curves and the per-neuron arrays.

    ``seed`` is accepted and **recorded, not used**: none of the §4 selection rules draws a random
    number, so no result depends on it. It is in the signature because `analysis/driver.py` passes
    ``seed`` to every module it invokes (INTERFACES §0), and this function previously forwarded it
    into ``select_from_arrays``, which rejects it — every driver call therefore failed with a
    TypeError. Post-freeze bug fix, 2026-09-03; see `docs/LABBOOK.md`.
    """
    from . import common
    rules = tuple(rules)
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        raise ValueError(f"unknown key-frequency rule(s) {unknown}; INTERFACES §4: {list(RULES)}")
    cfg, model, state, meta = common.load_model_at(run_dir, step)
    curves = kw.pop("curves", None)
    inputs, provenance = _gather_inputs(rules, cfg, model, state, curves, run_dir)

    results, arrays = {}, {}
    for rule in rules:
        res = select_from_arrays(rule, cfg.p, **_subset(inputs, rule), **kw)
        for name, arr in res.pop("_arrays", {}).items():
            arrays[f"{rule}__{name}"] = np.asarray(arr)
        arrays[f"{rule}__scores"] = np.asarray(res["scores"], dtype=np.float64)
        results[rule] = res
    jac = agreement({r: results[r]["key_frequencies"] for r in rules})

    params = {"step": meta["step"], "rules": list(rules), "primary_rule": PRIMARY_RULE,
              "seed": int(seed),
              "seed_is_unused": ("no §4 selection rule draws a random number; the seed is recorded "
                                 "for provenance only"),
              "threshold_frac": float(kw.get("threshold_frac", NANDA_THRESHOLD_FRAC)),
              "nanda_sensitivity_fracs": list(NANDA_SENSITIVITY_FRACS),
              "embedding_power_threshold": EMBEDDING_POWER_THRESHOLD,
              "embedding_top_k": EMBEDDING_TOP_K,
              "logit_share_multiple": LOGIT_SHARE_MULTIPLE,
              "neuron_cluster_min_neurons": kw.get("min_neurons", NEURON_CLUSTER_MIN_NEURONS),
              "neuron_cluster_min_variance_share": kw.get("min_variance_share",
                                                          NEURON_CLUSTER_MIN_VARIANCE_SHARE),
              "source": provenance,
              "preregistration": "docs/PREREGISTRATION.md §4.3 (primary rule and thresholds)"}
    payload = common.envelope(MODULE, MODULE_VERSION, meta, params)
    payload["results"] = {
        "primary_rule": PRIMARY_RULE,
        "primary_key_frequencies": results.get(PRIMARY_RULE, {}).get("key_frequencies"),
        "n_by_rule": {r: results[r]["n"] for r in rules},
        "cap_binding_by_rule": {r: results[r]["cap_binding"] for r in rules},
        "rules": results,
        "agreement": jac,
    }
    path = common.write_result(run_dir, MODULE, meta["tag"], payload, arrays)
    payload["output_path"] = str(path)
    return payload


def _gather_inputs(rules, cfg, model, state, curves, run_dir) -> tuple[dict, dict]:
    """Every array the requested rules need, each computed at most once."""
    from . import common
    inputs: dict = {}
    provenance: dict = {}
    if "nanda" in rules:
        inputs["W_L"] = neuron_logit_map(state, cfg)
        provenance["W_L"] = "W_out @ W_U[:, :p]" if cfg.arch == "transformer" else "W_out"
    if {"embedding_top8", "embedding_threshold"} & set(rules):
        inputs["W_E"], provenance["embedding_object"] = embedding_matrix(state, cfg)
    if "logit_sum_directions" in rules:
        inputs["logits"] = common.center_logits(common.grid_logits(model, cfg))
        provenance["logits"] = "common.grid_logits, centred over the class axis"
    if "neuron_clusters" in rules:
        if curves is None:
            curves, provenance["curves"] = operand_curves(state, cfg, run_dir)
        else:
            provenance["curves"] = "supplied by the caller"
        inputs["curves"] = curves
    return inputs, provenance


#: Which of the gathered arrays each rule consumes.
_RULE_INPUTS: dict[str, tuple[str, ...]] = {
    "nanda": ("W_L",),
    "embedding_top8": ("W_E",),
    "embedding_threshold": ("W_E",),
    "logit_sum_directions": ("logits",),
    "neuron_clusters": ("curves",),
}


def _subset(inputs: dict, rule: str) -> dict:
    return {k: inputs[k] for k in _RULE_INPUTS[rule] if k in inputs}


def summary_lines(payload: dict) -> list[str]:
    """Human-readable one-liner per rule plus the Jaccard matrix."""
    res = payload["results"]
    lines = [f"{payload['run_id']} ({payload['arch']}) step={payload['step']}  "
             f"primary rule = {res['primary_rule']}"]
    for rule, r in res["rules"].items():
        mark = " [PRIMARY]" if r["is_primary_rule"] else ""
        cap = "  cap_binding=TRUE" if r["cap_binding"] else ""
        lines.append(f"  {rule:<22} n={r['n']:<3} {sorted(r['key_frequencies'])}{cap}{mark}")
        if rule == PRIMARY_RULE:
            lines.append(f"    sensitivity (frac -> n): {r['sensitivity_n']}")
    names = res["agreement"]["names"]
    lines.append("  jaccard: " + "  ".join(
        f"{names[i]}|{names[j]}={res['agreement']['jaccard'][i][j]:.3f}"
        for i in range(len(names)) for j in range(i + 1, len(names))))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Key-frequency selection rules (INTERFACES §4)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: the legacy model_final.pt)")
    ap.add_argument("--rule", type=str, default="all",
                    choices=("all",) + RULES,
                    help=f"one rule or 'all' (default); the primary rule is {PRIMARY_RULE}. "
                         "A single rule writes a PARTIAL <tag>.json (that rule only) over any "
                         "previous one — re-run with 'all' to restore the full file")
    ap.add_argument("--threshold-frac", type=float, default=NANDA_THRESHOLD_FRAC,
                    help=f"'nanda' threshold as a fraction of the maximum norm "
                         f"(pre-registered default {NANDA_THRESHOLD_FRAC})")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="run on every entry of checkpoints.json (v2 runs)")
    args = ap.parse_args()

    rules = RULES if args.rule == "all" else (args.rule,)
    if args.all_checkpoints:
        from ..checkpoints import list_checkpoints
        steps = [int(e["step"]) for e in list_checkpoints(args.run_dir)]
    else:
        steps = [args.step]
    for step in steps:
        payload = analyse(args.run_dir, step, rules, threshold_frac=args.threshold_frac)
        print("\n".join(summary_lines(payload)))
        print(json.dumps({"output": payload["output_path"]}, indent=2))
        print(f"[saved] {payload['output_path']}", flush=True)


if __name__ == "__main__":
    main()
