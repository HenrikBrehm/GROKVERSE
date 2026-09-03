"""Per-neuron mechanism of the 2-layer MLP (docs/RESEARCH_SPEC.md §3.3, WP-2).

WHY THIS MODULE EXISTS
----------------------
Every structural claim in the repo so far rests on the shared embedding matrix
``W_E`` alone. For the MLP that is the wrong object. ``models/mlp.py``
concatenates the two operand embeddings *before* ``W_in``, so what hidden neuron
``i`` actually sees is::

    u_a[a, i] = W_E[a] . W_in[:d_model, i]        # effective a-curve of neuron i
    u_b[b, i] = W_E[b] . W_in[d_model:, i]        # effective b-curve of neuron i
    pre[a, b, i] = u_a[a, i] + u_b[b, i] + b_in[i]
    act[a, b, i] = ReLU(pre[a, b, i])
    out[c, i]    = W_out[i, c]                    # effective output curve of neuron i

``W_E`` is a *shared* table that both operand paths read through different halves
of ``W_in``. A per-neuron circuit can be perfectly periodic while the raw ``W_E``
spectrum looks diffuse — ``W_in`` is free to select and rotate. Measuring only
``W_E`` and concluding "distributed solution" is a non-sequitur.

Note the structural fact this makes visible: **the MLP is exactly additive in
(a, b) up to the ReLU.** No path lets a and b interact before the nonlinearity,
so in this architecture the ReLU is provably the only possible source of the
multiplication in the trig identity — which makes the MLP the cleaner test case
of the mechanism, not the messier one.

WHAT H1 PREDICTS
----------------
Writing each curve as ``A cos(w_k n - phi)``, rectifying a sum of two same-
frequency sinusoids produces a cross term in (a+b) with phase ``phi_a + phi_b``.
For the readout to peak at ``c = a + b`` the output curve must carry that same
phase, giving the per-neuron, falsifiable prediction::

    phi_out ~= phi_a + phi_b   (mod 2*pi)

``phase_relation`` measures it against a permutation null (neuron identities
shuffled), which is H1's stated null hypothesis.

STATUS: measurement code only. It reports distributions; it does not decide
whether H1 holds — and the "structured neuron" thresholds remain a [HUMAN]
decision (§9 item 6), so nothing here silently blesses a threshold.

WHAT THE MODULE PROVIDES (docs/dev/INTERFACES.md §5)
----------------------------------------------------
``effective_curves`` (both MLP architectures) · ``curve_spectra`` ·
``neuron_tables`` (per neuron and per curve: dominant frequency and fraction,
every ``analysis.metrics`` statistic, harmonic shares, the phases of the top-3
frequencies, the frequency-agreement flags and their contingency table) ·
``activation_analysis`` (re-exported from ``mlp_mechanism_activation``) ·
``logit_contributions`` · ``structured_neuron_definitions`` (the pre-registered
primary definition of ``docs/PREREGISTRATION.md`` §4.2 and every sensitivity
variant listed there) · ``classify_neurons`` / ``phase_relation`` (unchanged) ·
``analyse`` (runs everything and writes ``analysis/mlp_mechanism/<tag>.json``
plus ``.npz``).

Usage (from training/):
    python -m grokverse.analysis.mlp_mechanism runs/mlp_add_p113_wd1.0_frac0.3_seed0
    python -m grokverse.analysis.mlp_mechanism <run_dir> --step 25000 --key-rule nanda
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from . import metrics as M
from .common import envelope, load_model_at, summarize as dist_summary, write_result
from .fourier import dominant_frequencies, fourier_basis
from .mlp_mechanism_activation import (CHUNK, _act_builder, activation_analysis,  # noqa: F401
                                       fit_curve_matrix)
from .wave_fitting import MODEL_NAMES
from .wave_fitting import summarize as summarize_fits

MODULE = "mlp_mechanism"
MODULE_VERSION = "2.0"
#: The three curves a neuron's circuit is made of (INTERFACES §5).
CURVE_NAMES: tuple[str, ...] = ("u_a", "u_b", "out")
#: Architectures whose pre-activation is exactly ``u_a[a] + u_b[b] + b_in``.
SUPPORTED_ARCH: tuple[str, ...] = ("mlp", "mlp_twohot")


# --------------------------------------------------------------------------- #
# "structured neuron" criteria — thresholds are a [HUMAN] decision (§5, §9.6)  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StructuredNeuronCriteria:
    """A neuron counts as *structured* iff

    1. its a-curve and b-curve share a dominant frequency k,
    2. that frequency explains >= ``min_variance_explained`` of each curve's power,
    3. its output curve's dominant frequency is also k.

    The thresholds are NOT settled science. §9 item 6 reserves them for the
    human authors, and §5 makes a sensitivity analysis over >= 2 alternative
    thresholds mandatory. ``PROPOSED`` below is the spec's proposal awaiting
    approval — it is deliberately not a module-level default that could quietly
    become the answer.
    """

    min_variance_explained: float
    require_same_output_frequency: bool = True
    label: str = "unnamed"


#: The §5 proposal, pending approval in docs/HUMAN_DECISIONS.md.
PROPOSED = StructuredNeuronCriteria(0.50, True, "proposed_0.50")
#: Mandatory sensitivity companions (§5). Never report PROPOSED alone.
SENSITIVITY = (
    StructuredNeuronCriteria(0.30, True, "sensitivity_0.30"),
    StructuredNeuronCriteria(0.70, True, "sensitivity_0.70"),
)


# --------------------------------------------------------------------------- #
# effective per-neuron curves                                                  #
# --------------------------------------------------------------------------- #
def effective_curves(state: dict, cfg: Config) -> dict[str, np.ndarray]:
    """The three curves that define a neuron's circuit, all shaped [p, d_mlp].

    Derived from the actual forward pass of ``models/mlp.py`` (verified against
    a hand-computed reference in test_core.py), not from a redrawn diagram.

    Both MLP architectures are covered (INTERFACES §5), because both have an
    exactly additive pre-activation ``u_a[a] + u_b[b] + b_in``:

    * ``mlp`` — shared embedding read through the two halves of ``W_in``:
      ``u_a = W_E[:p] @ W_in[:d_model]``, ``u_b = W_E[:p] @ W_in[d_model:]``.
    * ``mlp_twohot`` — no embedding at all (``models/mlp_twohot.py``): a one-hot
      row-select IS an index into ``W_in``, so ``u_a = W_in[:p]`` and
      ``u_b = W_in[p:]`` exactly (the model's own forward pass computes
      ``W_in[a] + W_in[p + b]``).

    ``out[c, i] = W_out[i, c]`` and ``b_in`` are identical in both.
    """
    p = cfg.p
    W_in = np.asarray(state["W_in"], dtype=np.float64)        # [2d, d_mlp] / [2p, d_mlp]
    W_out = np.asarray(state["W_out"], dtype=np.float64)      # [d_mlp, p]
    b_in = np.asarray(state["b_in"], dtype=np.float64)        # [d_mlp]
    if cfg.arch == "mlp":
        d = cfg.d_model
        W_E = np.asarray(state["W_E"], dtype=np.float64)[:p]  # [p, d]
        u_a, u_b = W_E @ W_in[:d], W_E @ W_in[d:]
    elif cfg.arch == "mlp_twohot":
        u_a, u_b = W_in[:p], W_in[p:2 * p]
    else:
        raise ValueError(f"effective_curves handles arch 'mlp'/'mlp_twohot'; got {cfg.arch!r} — "
                         "the transformer's effective curves come from "
                         "analysis.transformer_mechanism (INTERFACES §6)")
    return {
        "u_a": u_a,                   # [p, d_mlp]
        "u_b": u_b,                   # [p, d_mlp]
        "out": W_out.T,               # [p, d_mlp] — class axis first, to match
        "b_in": b_in,                 # [d_mlp]
    }


def curve_spectra(curves: np.ndarray, p: int) -> dict:
    """Per-column Fourier power, phase, and dominant frequency.

    ``curves`` is [p, n]. Phase convention: the column is written as
    ``A cos(w_k n - phi)``, so ``phi = atan2(coeff_sin, coeff_cos)`` — valid
    because the cos_k and sin_k basis rows carry identical normalization.
    """
    F, _ = fourier_basis(p)
    coeff = F @ np.asarray(curves, dtype=np.float64)          # [p, n]
    half = (p - 1) // 2
    cos_rows = 1 + 2 * (np.arange(1, half + 1) - 1)
    sin_rows = 2 + 2 * (np.arange(1, half + 1) - 1)
    c, s = coeff[cos_rows], coeff[sin_rows]                   # [half, n]
    power = c ** 2 + s ** 2
    phase = np.arctan2(s, c)
    total = power.sum(axis=0)
    dom_idx = power.argmax(axis=0)
    n = power.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        dom_frac = np.where(total > 0, power[dom_idx, np.arange(n)] / total, 0.0)
    return {
        "power": power,                                       # [half, n]
        "phase": phase,                                       # [half, n]
        "freqs": np.arange(1, half + 1),
        "dominant_freq": dom_idx + 1,                         # [n]
        "dominant_fraction": dom_frac,                        # [n]
        "dominant_phase": phase[dom_idx, np.arange(n)],       # [n]
        "total_power": total,                                 # [n]
    }


def classify_neurons(spec_a: dict, spec_b: dict, spec_out: dict,
                     crit: StructuredNeuronCriteria) -> dict:
    """Apply one criteria set. Returns boolean masks and the counts."""
    same_ab = spec_a["dominant_freq"] == spec_b["dominant_freq"]
    strong = ((spec_a["dominant_fraction"] >= crit.min_variance_explained)
              & (spec_b["dominant_fraction"] >= crit.min_variance_explained))
    structured = same_ab & strong
    if crit.require_same_output_frequency:
        structured = structured & (spec_out["dominant_freq"] == spec_a["dominant_freq"])
    return {
        "criteria": dataclasses.asdict(crit),
        "n_neurons": int(structured.size),
        "n_same_ab_frequency": int(same_ab.sum()),
        "n_above_variance_threshold": int(strong.sum()),
        "n_structured": int(structured.sum()),
        "fraction_structured": float(structured.mean()),
        "mask": structured,
    }


# --------------------------------------------------------------------------- #
# H1: phi_out ~= phi_a + phi_b, against a permutation null                     #
# --------------------------------------------------------------------------- #
def _wrap(x: np.ndarray) -> np.ndarray:
    """Wrap angles into (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(x)))


def _resultant_length(err: np.ndarray) -> float:
    """Mean resultant length R in [0, 1] — the circular concentration statistic.

    R ~ 1 means the phase errors pile up at one angle (H1's prediction);
    R ~ 0 means they are spread like the uniform null.
    """
    return float(np.abs(np.exp(1j * np.asarray(err)).mean())) if err.size else 0.0


def phase_relation(spec_a: dict, spec_b: dict, spec_out: dict,
                   mask: np.ndarray, n_boot: int = 2000, n_perm: int = 2000,
                   seed: int = 0) -> dict:
    """Circular test of ``phi_out - (phi_a + phi_b)`` on the selected neurons.

    The null shuffles neuron identity between the operand side and the output
    side, destroying the per-neuron correspondence while leaving both marginal
    phase distributions intact — H1_0 as stated in §4.
    """
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return {"n": int(idx.size), "insufficient_neurons": True}
    pa = spec_a["dominant_phase"][idx]
    pb = spec_b["dominant_phase"][idx]
    po = spec_out["dominant_phase"][idx]
    err = _wrap(po - (pa + pb))
    R = _resultant_length(err)

    rng = np.random.default_rng(seed)
    boot = np.array([_resultant_length(err[rng.integers(0, err.size, err.size)])
                     for _ in range(int(n_boot))])
    null = np.array([_resultant_length(_wrap(po[rng.permutation(idx.size)] - (pa + pb)))
                     for _ in range(int(n_perm))])
    return {
        "n": int(idx.size),
        "insufficient_neurons": False,
        "circular_mean_error": float(np.angle(np.exp(1j * err).mean())),
        "median_abs_error": float(np.median(np.abs(err))),
        "resultant_length": R,
        "resultant_ci95": [float(np.quantile(boot, 0.025)),
                           float(np.quantile(boot, 0.975))],
        "null_resultant_mean": float(null.mean()),
        "null_resultant_q95": float(np.quantile(null, 0.95)),
        # H1 is falsified if the observed R sits inside the null's bulk
        "exceeds_null_q95": bool(R > np.quantile(null, 0.95)),
        "n_boot": int(n_boot), "n_perm": int(n_perm), "seed": int(seed),
    }


# --------------------------------------------------------------------------- #
# is the activation a function of (a + b)?                                     #
# --------------------------------------------------------------------------- #
def sum_dependence(curves: dict, p: int, act: np.ndarray | None = None) -> dict:
    """Fraction of each neuron's activation variance explained by (a+b) mod p.

    A neuron implementing the circuit is (to first order) a function of a+b
    alone, so ``Var(E[act | a+b]) / Var(act)`` should be near 1. Computed
    exactly — every residue class of a+b holds exactly p of the p^2 pairs — and
    reported next to the same quantity for (a-b), which the addition circuit has
    no use for and which therefore acts as a built-in control.

    ``act`` overrides the curve-rebuilt activation with a supplied ``[p, p, n]``
    tensor; ``analysis.transformer_mechanism`` passes the transformer's true
    hidden layer, which is only approximately the rectified sum of its effective
    curves (INTERFACES 6).
    """
    u_a, u_b, b_in = curves["u_a"], curves["u_b"], curves["b_in"]
    act = (np.maximum(u_a[:, None, :] + u_b[None, :, :] + b_in, 0.0)  # [p, p, n]
           if act is None else np.asarray(act, dtype=np.float64))
    want = (p, p, int(np.shape(u_a)[1]))
    if act.shape != want:
        raise ValueError(f"act must be [p, p, n_neurons] = {want}, got {act.shape}")
    a_idx = np.arange(p)[:, None]
    b_idx = np.arange(p)[None, :]

    def explained(group: np.ndarray) -> np.ndarray:
        n = act.shape[-1]
        sums = np.zeros((p, n))
        np.add.at(sums, group.ravel(), act.reshape(-1, n))
        means = sums / p                                    # each class has p pairs
        grand = act.reshape(-1, n).mean(axis=0)
        between = ((means - grand) ** 2).sum(axis=0) * p
        total = ((act.reshape(-1, n) - grand) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total > 0, between / total, 0.0)

    ev_sum = explained((a_idx + b_idx) % p)
    ev_diff = explained((a_idx - b_idx) % p)
    return {
        "variance_explained_by_sum": ev_sum,
        "variance_explained_by_diff": ev_diff,
        "median_sum": float(np.median(ev_sum)),
        "median_diff": float(np.median(ev_diff)),
        "n_sum_above_0.9": int((ev_sum > 0.9).sum()),
        "dead_neurons": int((act.max(axis=(0, 1)) <= 0).sum()),
    }


# --------------------------------------------------------------------------- #
# per-neuron tables (INTERFACES §5, first bullet)                              #
# --------------------------------------------------------------------------- #
def _top_frequencies(spec: dict, n_top: int) -> dict:
    """The ``n_top`` strongest frequencies per column with their phases and power shares."""
    power, phase = spec["power"], spec["phase"]
    n_top = int(min(int(n_top), power.shape[0]))
    order = np.argsort(-power, axis=0, kind="stable")[:n_top]          # [n_top, n]
    total = power.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(total > 0, power / total, np.nan)
    return {"top_frequency": order + 1,
            "top_phase": np.take_along_axis(phase, order, axis=0),
            "top_fraction": np.take_along_axis(q, order, axis=0)}


def neuron_tables(curves: dict, p: int, topk: tuple[int, ...] = (1, 4, 8),
                  n_top_phases: int = 3, max_harmonic: int = M.MAX_ODD_HARMONIC,
                  binarization_threshold: float = 0.8) -> dict:
    """Per neuron and per curve (``u_a``, ``u_b``, ``out``): the full §2 metric table.

    For every curve: dominant frequency and dominant fraction, every metric of
    ``analysis.metrics`` (top-k concentration, spectral entropy, participation ratio, both
    inverse-participation-ratio definitions, Swaroop's periodicity score, our binarization
    score, the harmonic shares around the curve's OWN fundamental with the discrete
    square-wave reference), and the frequencies, phases and power shares of the ``n_top_phases``
    strongest frequencies.

    Across the three curves: the frequency-agreement flags ``same_ab``
    (``u_a`` and ``u_b`` share a dominant frequency) and ``same_ab_out`` (the output curve
    agrees as well) and the 2x2 contingency table of the two conditions. Phases follow the
    §0 convention ``y[n] = A cos(2 pi k n / p - phi)``.

    Nothing here classifies a neuron: the thresholds live in
    ``structured_neuron_definitions`` (pre-registered in ``docs/PREREGISTRATION.md`` §4.2).
    """
    missing = [c for c in CURVE_NAMES if c not in curves]
    if missing:
        raise ValueError(f"neuron_tables needs curves {list(CURVE_NAMES)}; missing {missing}")
    per_curve: dict[str, dict] = {}
    for name in CURVE_NAMES:
        Y = np.asarray(curves[name], dtype=np.float64)
        if Y.ndim != 2 or Y.shape[0] != p:
            raise ValueError(f"curve {name!r} must be [p={p}, n_neurons], got {Y.shape}")
        spec = curve_spectra(Y, p)
        met = M.curve_metrics(Y, p, tuple(topk), int(max_harmonic), float(binarization_threshold))
        per_curve[name] = {
            # curve_spectra's dominant frequency is always >= 1 (a zero-power column is
            # flagged by `zero_power`, not silently given frequency 0 as metrics does).
            "dominant_frequency": np.asarray(spec["dominant_freq"], dtype=np.int64),
            "dominant_fraction": np.asarray(spec["dominant_fraction"], dtype=np.float64),
            "dominant_phase": np.asarray(spec["dominant_phase"], dtype=np.float64),
            "family_fraction": np.asarray(met["harmonic_shares"]["family_share"], dtype=np.float64),
            "total_power": np.asarray(spec["total_power"], dtype=np.float64),
            "power": np.asarray(spec["power"], dtype=np.float64),        # [half, n]
            "zero_power": np.asarray(spec["total_power"], dtype=np.float64) <= 0.0,
            **_top_frequencies(spec, n_top_phases),
            "metrics": met,
        }
    dom = {name: per_curve[name]["dominant_frequency"] for name in CURVE_NAMES}
    same_ab = dom["u_a"] == dom["u_b"]
    same_out_as_a = dom["out"] == dom["u_a"]
    same_ab_out = same_ab & same_out_as_a
    ipr = M.ipr_doshi_neuron(curves["u_a"], curves["u_b"], curves["out"], "rfft")
    n = int(same_ab.size)
    return {
        "n_neurons": n,
        "per_curve": per_curve,
        "same_ab": same_ab,
        "same_out_as_a": same_out_as_a,
        "same_ab_out": same_ab_out,
        "ipr_doshi_neuron_rfft": np.asarray(ipr["ipr_neuron"], dtype=np.float64),
        "contingency": {
            "n_neurons": n,
            "same_ab_and_same_out": int((same_ab & same_out_as_a).sum()),
            "same_ab_not_same_out": int((same_ab & ~same_out_as_a).sum()),
            "not_same_ab_but_out_matches_a": int((~same_ab & same_out_as_a).sum()),
            "neither": int((~same_ab & ~same_out_as_a).sum()),
            "n_same_ab": int(same_ab.sum()),
            "n_same_ab_out": int(same_ab_out.sum()),
        },
        "frequency_histogram": {name: _histogram(dom[name]) for name in CURVE_NAMES},
        "params": {"p": int(p), "topk": [int(t) for t in topk],
                   "n_top_phases": int(n_top_phases), "max_harmonic": int(max_harmonic),
                   "binarization_threshold": float(binarization_threshold),
                   "phase_convention": "y[n] = A cos(2*pi*k*n/p - phi)",
                   "ipr_definition": ipr["definition"]},
    }


def _histogram(values: np.ndarray) -> dict:
    ks, counts = np.unique(np.asarray(values), return_counts=True)
    return {int(k): int(c) for k, c in zip(ks, counts)}


#: Scalar per-curve entries of ``neuron_tables`` that are summarized into the JSON.
_TABLE_SCALARS: tuple[str, ...] = (
    "dominant_fraction", "family_fraction", "spectral_entropy", "participation_ratio",
    "inverse_participation_ratio_ours", "inverse_participation_ratio_doshi_rfft",
    "inverse_participation_ratio_doshi_fft", "periodicity_score_swaroop",
    "binarization_score_ours",
)
_HARMONIC_SCALARS: tuple[str, ...] = (
    "fundamental_share", "odd_share", "even_share", "family_share", "odd_minus_even",
    "ideal_square_family_share", "ideal_square_odd_minus_even",
)


def summarize_tables(tables: dict) -> dict:
    """JSON-sized summary of ``neuron_tables`` (INTERFACES §0: no truncated arrays)."""
    out: dict = {"n_neurons": tables["n_neurons"], "contingency": tables["contingency"],
                 "frequency_histogram": tables["frequency_histogram"],
                 "ipr_doshi_neuron_rfft": dist_summary(tables["ipr_doshi_neuron_rfft"]),
                 "params": tables["params"], "per_curve": {}}
    for name in CURVE_NAMES:
        cur = tables["per_curve"][name]
        met = cur["metrics"]
        summary = {"n_undefined_columns": met["n_undefined_columns"],
                   "n_zero_power": int(cur["zero_power"].sum())}
        for key in _TABLE_SCALARS:
            src = met.get(key, cur.get(key))
            if src is not None:
                summary[key] = dist_summary(src)
        summary["topk_concentration"] = {str(k): dist_summary(v)
                                         for k, v in met["topk_concentration"].items()}
        summary["harmonic_shares"] = {key: dist_summary(met["harmonic_shares"][key])
                                      for key in _HARMONIC_SCALARS}
        summary["n_harmonic_collisions"] = int(np.asarray(
            met["harmonic_shares"]["n_collisions"]).sum())
        out["per_curve"][name] = summary
    return out


# --------------------------------------------------------------------------- #
# per-neuron logit contributions (INTERFACES §5, third bullet)                 #
# --------------------------------------------------------------------------- #
def logit_contributions(curves: dict, W_out, b_out, p: int, chunk: int = CHUNK,
                        act: np.ndarray | None = None) -> dict:
    """What each hidden neuron contributes to the logits, without storing ``[n, p, p, p]``.

    With ``act[a,b,i] = ReLU(u_a[a,i] + u_b[b,i] + b_in[i])`` the model is exactly
    ``logits[a,b,c] = sum_i act[a,b,i] W_out[i,c] + b_out[c]``. Three per-neuron numbers
    are reported, all *exactly* additive so that they can be checked against the total:

    ``mean_correct_logit_contribution[i]``
        mean over the ``p^2`` grid of ``act[a,b,i] * W_out[i, (a+b) mod p]``. Summed over
        neurons and added to ``mean(b_out)`` this is the mean correct-class logit.
    ``logit_variance_share[i]``
        ``<C_i, L> / <L, L>`` on the class-centered tensors (``center_logits``), where
        ``C_i[a,b,c] = act[a,b,i] (W_out[i,c] - mean_c W_out[i,c])``. Because
        ``L = sum_i C_i + b_centered``, the shares plus ``bias_variance_share`` sum to
        exactly 1 — a covariance decomposition, not a ratio of variances (which would not).
    ``margin_share[i]``
        share of the mean correct-class margin, the margin being the class-centered
        correct-class logit ``L[a,b,(a+b) mod p]`` (correct-class logit minus the mean over
        classes). Also exactly additive over neurons.

    The full ``[p,p,p]`` per-neuron contribution tensor is never materialized: the grid is
    walked in chunks of ``chunk`` neurons (INTERFACES §5). The centered *total* logit tensor
    is returned under ``logits_centered`` for callers that need it and is NOT written to the
    npz by ``analyse``.

    TWO ARGUMENTS ARE GENERALIZED FOR THE TRANSFORMER (INTERFACES 6). Its logits obey the
    *same* additive law, ``logits = direct_path + sum_f hidden_f (W_out W_U)[f]``, so the
    decomposition below applies unchanged once:

    * ``act`` supplies the true ``hidden`` (``[p, p, n]``) instead of rebuilding it from the
      curves -- the transformer's pre-activation is only approximately additive; and
    * ``b_out`` accepts the ``[p, p, p]`` direct (attention-only) path in place of the MLP's
      ``[p]`` output bias. It is the term that carries no hidden neuron, which is exactly the
      role ``b_out`` plays for the MLP, so ``bias_variance_share`` keeps its meaning and the
      shares still sum to 1.
    """
    u_a = np.asarray(curves["u_a"], dtype=np.float64)
    u_b = np.asarray(curves["u_b"], dtype=np.float64)
    b_in = np.asarray(curves["b_in"], dtype=np.float64).ravel()
    W = np.asarray(W_out, dtype=np.float64)
    bo = np.asarray(b_out, dtype=np.float64)
    n = u_a.shape[1]
    if W.shape != (n, p):
        raise ValueError(f"W_out must be [n_neurons={n}, p={p}], got {W.shape}")
    cls = ((np.arange(p)[:, None] + np.arange(p)[None, :]) % p).ravel()
    if bo.shape == (p,):                        # MLP output bias
        bc = (bo - bo.mean())[None, None, :]
        mean_bias_correct = float(bo.mean())    # each class occurs p times on the grid
        bias_shape = "[p] output bias"
    elif bo.shape == (p, p, p):                 # transformer direct (attention-only) path
        bc = bo - bo.mean(axis=-1, keepdims=True)
        mean_bias_correct = float(bo.reshape(-1, p)[np.arange(p * p), cls].mean())
        bias_shape = "[p, p, p] direct path"
    else:
        raise ValueError(f"b_out must be [p={p}] or [p, p, p], got {bo.shape}")
    Wc = W - W.mean(axis=1, keepdims=True)
    get_act = _act_builder(u_a, u_b, b_in, act)

    L = np.zeros((p, p, p))                       # class-centered logits, neuron part
    S = np.zeros((p, n))                          # S[m, i] = sum_{a+b=m} act[a,b,i]
    for start in range(0, n, int(chunk)):
        cols = slice(start, min(start + int(chunk), n))
        a_c = get_act(cols)
        L += np.einsum("abn,nc->abc", a_c, Wc[cols], optimize=True)
        part = np.zeros((p, a_c.shape[-1]))
        np.add.at(part, cls, a_c.reshape(-1, a_c.shape[-1]))
        S[:, cols] = part
    L += bc

    ss = float((L ** 2).sum())
    G = np.zeros((n, p))
    for start in range(0, n, int(chunk)):
        cols = slice(start, min(start + int(chunk), n))
        G[cols] = np.einsum("abn,abc->nc", get_act(cols), L, optimize=True)

    grid = float(p * p)
    mean_correct = np.einsum("mn,nm->n", S, W) / grid
    mean_margin = np.einsum("mn,nm->n", S, Wc) / grid
    total_margin = float(mean_margin.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        var_share = (Wc * G).sum(axis=1) / ss if ss > 0 else np.full(n, np.nan)
        margin_share = (mean_margin / total_margin if total_margin != 0.0
                        else np.full(n, np.nan))
    bias_share = float((bc * L).sum() / ss) if ss > 0 else float("nan")
    per_neuron = {"mean_correct_logit_contribution": mean_correct,
                  "logit_variance_share": np.asarray(var_share, dtype=np.float64),
                  "margin_share": np.asarray(margin_share, dtype=np.float64),
                  "mean_margin_contribution": mean_margin}
    identities = {
        "sum_mean_correct_logit_contribution": float(mean_correct.sum()),
        "mean_b_out": mean_bias_correct,
        "mean_correct_logit": float(mean_correct.sum() + mean_bias_correct),
        "mean_correct_class_margin": total_margin,
        "sum_logit_variance_share": float(np.nansum(var_share)),
        "bias_variance_share": bias_share,
        "total_variance_share": float(np.nansum(var_share) + bias_share),
        "total_centered_logit_power": ss,
    }
    return {"per_neuron": per_neuron, "identities": identities,
            "summary": {key: dist_summary(v) for key, v in per_neuron.items()} | identities,
            "logits_centered": L,
            "params": {"p": int(p), "n_neurons": int(n), "chunk": int(chunk),
                       "bias_term": bias_shape,
                       "activation_source": ("relu(u_a + u_b + b_in) rebuilt from the curves"
                                             if act is None else
                                             "supplied activation tensor (true hidden layer)"),
                       "margin_definition": "class-centered correct-class logit",
                       "variance_share_definition": "<C_i, L> / <L, L> on class-centered tensors"}}


# --------------------------------------------------------------------------- #
# structured-neuron definitions (docs/PREREGISTRATION.md §4.2)                 #
# --------------------------------------------------------------------------- #
#: PREREGISTRATION §4.2 primary family-fraction threshold, and its two mandatory
#: sensitivity variants. Never widened, never narrowed, never defaulted away.
FAMILY_THRESHOLD_PRIMARY = 0.50
FAMILY_THRESHOLDS_SENSITIVITY: tuple[float, ...] = (0.30, 0.70)
#: The "top-1 variant" of §4.2: dominant fraction >= 0.50 instead of the family fraction.
TOP1_THRESHOLD = 0.50
PRIMARY_DEFINITION = f"primary_family_{FAMILY_THRESHOLD_PRIMARY:.2f}"


def _family_definition(alive, tables: dict, threshold: float) -> np.ndarray:
    """§4.2 with the family fraction: alive, same dominant k on u_a/u_b/out, family >= t."""
    per = tables["per_curve"]
    return (alive & tables["same_ab_out"]
            & (per["u_a"]["family_fraction"] >= threshold)
            & (per["u_b"]["family_fraction"] >= threshold))


def _top1_definition(alive, tables: dict, threshold: float) -> np.ndarray:
    """The top-1 variant: the same conditions with the dominant fraction."""
    per = tables["per_curve"]
    return (alive & tables["same_ab_out"]
            & (per["u_a"]["dominant_fraction"] >= threshold)
            & (per["u_b"]["dominant_fraction"] >= threshold))


def _jaccard(x: np.ndarray, y: np.ndarray) -> float:
    """|x & y| / |x | y|; two empty sets are identical, so the convention is 1.0."""
    union = int((x | y).sum())
    return 1.0 if union == 0 else float(int((x & y).sum()) / union)


def structured_neuron_definitions(tables: dict, activation: dict,
                                  family_threshold: float = FAMILY_THRESHOLD_PRIMARY,
                                  family_sensitivity=FAMILY_THRESHOLDS_SENSITIVITY,
                                  top1_threshold: float = TOP1_THRESHOLD) -> dict:
    """Every structured-neuron definition of ``docs/PREREGISTRATION.md`` §4.2, side by side.

    The **primary** definition, verbatim from §4.2: a hidden neuron is structured iff
    (1) it is alive (maximum activation over the grid > 0), (2) ``u_a`` and ``u_b`` share
    the dominant frequency ``k``, (3) the **family fraction** — the curve's own dominant
    frequency plus its aliased odd harmonics up to 7 (§4.1) — of ``u_a`` **and** of ``u_b``
    is >= 0.50, and (4) the output curve's dominant frequency is also ``k``.

    Mandatory sensitivity variants, always reported next to it:

    * family threshold **0.30** and **0.70** (same definition, other threshold);
    * the **top-1** variant: dominant fraction >= 0.50 instead of the family fraction;
    * **Doshi's IPR** — the source (arXiv:2310.13061) *ranks* neurons and states no
      threshold (``metrics.IPR_DEFINITIONS``), so the mask here is the top-``N`` live
      neurons by ``ipr_doshi_neuron`` with ``N`` matched to the primary definition's count.
      The cardinality matching is OURS and is labelled so; no threshold is invented.
    * **Swaroop's periodicity score** with his 12 / 5 cuts
      (``metrics.SWAROOP_PERIODICITY_THRESHOLDS``) — read off a histogram at ``p = 97``,
      post-hoc and ``p``-specific, never principled, and reported as such. Both cuts are
      given: ``> 12`` on ``u_a`` and ``u_b`` (his "structured") and ``< 5`` (his
      "unstructured", which is *not* the complement of the first).

    Also returns the pairwise Jaccard overlap between all definitions and, per definition,
    the count of selected neurons per dominant frequency of ``u_a``.
    """
    alive = ~np.asarray(activation["per_neuron"]["dead"], dtype=bool)
    per = tables["per_curve"]
    n = int(tables["n_neurons"])
    if alive.size != n:
        raise ValueError(f"activation covers {alive.size} neurons but the tables have {n}")

    defs: dict[str, dict] = {}

    def add(name: str, mask: np.ndarray, definition: str, params: dict) -> None:
        mask = np.asarray(mask, dtype=bool)
        defs[name] = {"mask": mask, "n": int(mask.sum()),
                      "fraction_of_all_neurons": float(mask.mean()),
                      "fraction_of_live_neurons": (float(mask.sum() / alive.sum())
                                                   if alive.any() else float("nan")),
                      "definition": definition, "params": params,
                      "counts_per_frequency": _histogram(
                          per["u_a"]["dominant_frequency"][mask])}

    primary = _family_definition(alive, tables, float(family_threshold))
    primary_name = f"primary_family_{float(family_threshold):.2f}"
    add(primary_name, primary,
        "PREREGISTRATION 4.2 primary: alive AND same dominant k on u_a, u_b and out AND "
        "family fraction (own fundamental + aliased odd harmonics <= 7) >= threshold on both "
        "operand curves",
        {"family_threshold": float(family_threshold), "max_harmonic": tables["params"]["max_harmonic"],
         "source": "docs/PREREGISTRATION.md 4.2 [AI-PROPOSED]"})
    for t in (t for t in family_sensitivity if float(t) != float(family_threshold)):
        add(f"sensitivity_family_{float(t):.2f}", _family_definition(alive, tables, float(t)),
            "PREREGISTRATION 4.2 sensitivity variant: the primary definition at another "
            "family-fraction threshold",
            {"family_threshold": float(t), "source": "docs/PREREGISTRATION.md 4.2"})
    add(f"top1_{float(top1_threshold):.2f}", _top1_definition(alive, tables, float(top1_threshold)),
        "PREREGISTRATION 4.2 top-1 variant: dominant fraction >= threshold on u_a and u_b "
        "instead of the family fraction",
        {"dominant_fraction_threshold": float(top1_threshold),
         "source": "docs/PREREGISTRATION.md 4.2"})

    ipr = np.asarray(tables["ipr_doshi_neuron_rfft"], dtype=np.float64)
    n_match = int(primary.sum())
    ranked = np.zeros(n, dtype=bool)
    live_idx = np.flatnonzero(alive)
    if n_match > 0 and live_idx.size:
        order = live_idx[np.argsort(-np.nan_to_num(ipr[live_idx], nan=-np.inf), kind="stable")]
        ranked[order[:n_match]] = True
    add("ipr_doshi_rank_matched", ranked,
        "Doshi et al. 2023 (arXiv:2310.13061) Eq. 3/4 IPR ranking of live neurons, truncated "
        "to the primary definition's count [ours: the source ranks and states NO threshold]",
        {"ipr_definition": M.IPR_DEFINITIONS["doshi_rfft"], "n_matched_to": primary_name,
         "n_selected": n_match, "threshold_source": "[NOT FOUND IN SOURCE] — cardinality match "
         "to the primary definition is ours"})

    per_a = np.asarray(per["u_a"]["metrics"]["periodicity_score_swaroop"], dtype=np.float64)
    per_b = np.asarray(per["u_b"]["metrics"]["periodicity_score_swaroop"], dtype=np.float64)
    cuts = M.SWAROOP_PERIODICITY_THRESHOLDS
    swaroop_note = {"thresholds": dict(cuts), "score_maximum_at_this_p": M.periodicity_score_max(
        tables["params"]["p"]),
        "caveat": ("Swaroop's 12 / 5 cuts are post-hoc histogram cuts at p = 97; the score's "
                   "maximum is (p-1)/2, so they do not transfer to another p unchanged "
                   "(metrics.SWAROOP_PERIODICITY_THRESHOLDS)")}
    add("periodicity_swaroop_gt12", alive & (per_a > cuts["structured_gt"]) & (per_b > cuts["structured_gt"]),
        "Swaroop 2026 (arXiv:2603.23784) Eq. 1 periodicity score > 12 on u_a AND u_b, alive "
        "[his post-hoc histogram cut at p = 97, not a principled threshold]", swaroop_note)
    add("periodicity_swaroop_lt5", alive & (per_a < cuts["unstructured_lt"]) & (per_b < cuts["unstructured_lt"]),
        "Swaroop 2026 Eq. 1 periodicity score < 5 on u_a AND u_b, alive — his 'unstructured' "
        "cut, which is NOT the complement of the > 12 cut", swaroop_note)

    names = list(defs)
    jaccard = {f"{a}|{b}": _jaccard(defs[a]["mask"], defs[b]["mask"])
               for i, a in enumerate(names) for b in names[i + 1:]}
    return {"n_neurons": n, "n_live": int(alive.sum()), "alive": alive,
            "primary": primary_name, "definitions": defs, "jaccard": jaccard,
            "params": {"family_threshold_primary": float(family_threshold),
                       "family_thresholds_sensitivity": [float(t) for t in family_sensitivity],
                       "top1_threshold": float(top1_threshold),
                       "jaccard_convention_two_empty_sets": 1.0,
                       "source": "docs/PREREGISTRATION.md 4.2 (all values [AI-PROPOSED], "
                                 "pending docs/HUMAN_DECISIONS.md)"}}


# --------------------------------------------------------------------------- #
# key-frequency set (INTERFACES §4; docs/PREREGISTRATION.md §4.3)              #
# --------------------------------------------------------------------------- #
#: Pre-registered PRIMARY key-frequency rule (PREREGISTRATION §4.3 / INTERFACES §4).
PRIMARY_KEY_RULE = "nanda"
#: Used only when ``analysis.key_frequencies`` does not exist yet; the substitution is
#: recorded in ``params.key_frequency_selection`` of every output file.
FALLBACK_KEY_RULE = "embedding_top8"


def _embedding_top8(state: dict, cfg: Config) -> tuple[list[int], dict]:
    """The legacy top-8 rule on the model's embedding-like object (INTERFACES §4).

    ``mlp`` uses ``W_E[:p]``; the two-hot model has no embedding, so the a-half of ``W_in``
    is used — the object its own ``embedding_snapshot`` logs as ``W_in_a_half``. The cap
    binds on every legacy run (fourier.dominant_frequencies documents this), so the count is
    NOT data-determined; ``cap_binding`` is carried through.
    """
    p = cfg.p
    if cfg.arch == "mlp":
        obj, name = np.asarray(state["W_E"], dtype=np.float64)[:p], "W_E[:p]"
    else:
        obj, name = np.asarray(state["W_in"], dtype=np.float64)[:p], "W_in[:p] (W_in_a_half)"
    dom = dominant_frequencies(obj, p)
    return [int(k) for k in dom["dominant"]], {
        "object": name, "threshold": dom["threshold"], "max_k": dom["max_k"],
        "n_keep": dom["n_keep"], "n_freqs_for_threshold": dom["n_freqs_for_threshold"],
        "cap_binding": dom["cap_binding"], "dominant_fraction": dom["dominant_fraction"]}


def resolve_key_frequencies(run_dir, step, rule: str, state: dict, cfg: Config) -> tuple[list[int], dict]:
    """Key set for ``rule``, delegated to ``analysis.key_frequencies`` (INTERFACES §4).

    That module is written in a separate work package and may not exist yet. If it is
    absent the selection falls back to ``embedding_top8``, computed here, and the fallback
    is recorded verbatim in the result so no number can silently look like the
    pre-registered primary rule. If the module IS present, its errors are raised, never
    swallowed.
    """
    info = {"requested_key_rule": str(rule), "primary_key_rule": PRIMARY_KEY_RULE,
            "fallback_used": False}
    try:
        from . import key_frequencies                     # noqa: WPS433 — optional module
    except ImportError as exc:
        keys, extra = _embedding_top8(state, cfg)
        info.update({
            "key_rule": FALLBACK_KEY_RULE, "fallback_used": True,
            "fallback_reason": f"analysis.key_frequencies is not available ({exc})",
            "note": (f"requested rule {rule!r} could not be evaluated; the legacy "
                     f"{FALLBACK_KEY_RULE!r} rule was used instead. Its cap binds on every "
                     "legacy run, so the SIZE of this set is not data-determined "
                     "(docs/PREREGISTRATION.md 4.3)."),
            **extra})
        return keys, info
    if rule == FALLBACK_KEY_RULE and not hasattr(key_frequencies, "select"):
        keys, extra = _embedding_top8(state, cfg)
        info.update({"key_rule": FALLBACK_KEY_RULE, "fallback_used": True,
                     "fallback_reason": "key_frequencies has no select()", **extra})
        return keys, info
    sel = key_frequencies.select(run_dir, step, rule)
    keys = [int(k) for k in sel["key_frequencies"]]
    info.update({k: v for k, v in sel.items() if not isinstance(v, np.ndarray)})
    info["key_rule"] = str(rule)
    info["key_frequencies"] = keys
    return keys, info


# --------------------------------------------------------------------------- #
# run-level analysis + CLI                                                     #
# --------------------------------------------------------------------------- #
def _fit_arrays(fits: dict) -> dict[str, np.ndarray]:
    """Per-neuron fit arrays kept in the npz: best-by-AIC, r2 per model, alpha_3/alpha_1."""
    arrays: dict[str, np.ndarray] = {}
    for name, fit in fits.items():
        arrays[f"fit__{name}__k"] = fit["k"]
        arrays[f"fit__{name}__best_by_aic"] = fit["best_by_aic"]
        arrays[f"fit__{name}__best_by_aicc"] = fit["best_by_aicc"]
        arrays[f"fit__{name}__delta_aic_sinusoid_minus_square"] = \
            fit["delta_aic_sinusoid_minus_square"]
        for model in MODEL_NAMES:
            arrays[f"fit__{name}__r2__{model}"] = fit["r2"][model]
        ratios = fit["params"]["odd_harmonics"]["amplitude_ratio_to_fundamental"]
        arrays[f"fit__{name}__alpha3_over_alpha1"] = ratios[0]
        arrays[f"fit__{name}__alpha5_over_alpha1"] = ratios[1]
        arrays[f"fit__{name}__alpha7_over_alpha1"] = ratios[2]
    return arrays


def _table_arrays(tables: dict) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "same_ab": tables["same_ab"], "same_ab_out": tables["same_ab_out"],
        "same_out_as_a": tables["same_out_as_a"],
        "ipr_doshi_neuron_rfft": tables["ipr_doshi_neuron_rfft"]}
    for name in CURVE_NAMES:
        cur = tables["per_curve"][name]
        for key in ("dominant_frequency", "dominant_fraction", "dominant_phase",
                    "family_fraction", "total_power", "top_frequency", "top_phase",
                    "top_fraction"):
            arrays[f"{name}__{key}"] = cur[key]
        met = cur["metrics"]
        for key in _TABLE_SCALARS:
            if key in met:
                arrays[f"{name}__{key}"] = np.asarray(met[key])
        for key in _HARMONIC_SCALARS + ("n_collisions",):
            arrays[f"{name}__harmonic__{key}"] = np.asarray(met["harmonic_shares"][key])
        arrays[f"{name}__harmonic__share_by_j"] = np.asarray(
            met["harmonic_shares"]["harmonic_share_by_j"], dtype=np.float32)
        arrays[f"{name}__power"] = np.asarray(cur["power"], dtype=np.float32)
    return arrays


def analyse(run_dir, step: int | None = None, key_rule: str = PRIMARY_KEY_RULE, seed: int = 0,
            n_boot: int = 2000, n_perm: int = 2000, n_exemplars: int = 16,
            chunk: int = CHUNK, cv_folds: int = 0) -> tuple[dict, Path]:
    """Run every §5 measurement on one checkpoint and write the JSON + npz result.

    Works for ``arch`` ``mlp`` and ``mlp_twohot``. Waveform fits come from
    ``wave_fitting.fit_matrix`` at each neuron's OWN dominant frequency, per curve.
    """
    run_dir = Path(run_dir)
    cfg, _model, state, meta = load_model_at(run_dir, step)
    if cfg.arch not in SUPPORTED_ARCH:
        raise ValueError(f"{meta['run_id']}: analysis.mlp_mechanism handles {list(SUPPORTED_ARCH)}, "
                         f"got arch={cfg.arch!r} (the transformer has its own module, INTERFACES 6)")
    p = cfg.p
    curves = effective_curves(state, cfg)
    tables = neuron_tables(curves, p)
    fits = {name: fit_curve_matrix(curves[name], tables["per_curve"][name]["dominant_frequency"],
                                   int(cv_folds), int(seed)) for name in CURVE_NAMES}
    key_freqs, key_info = resolve_key_frequencies(run_dir, step, key_rule, state, cfg)
    sums = sum_dependence(curves, p)
    act = activation_analysis(curves, p, key_freqs, fits=fits, n_exemplars=int(n_exemplars),
                              chunk=int(chunk), sum_dependence_result=sums)
    contrib = logit_contributions(curves, np.asarray(state["W_out"], dtype=np.float64),
                                  np.asarray(state["b_out"], dtype=np.float64), p, int(chunk))
    defs = structured_neuron_definitions(tables, act)

    spec = {name: curve_spectra(curves[name], p) for name in CURVE_NAMES}
    definitions_out: dict[str, dict] = {}
    for name, entry in defs["definitions"].items():
        mask = entry["mask"]
        definitions_out[name] = {k: v for k, v in entry.items() if k != "mask"}
        definitions_out[name]["phase_relation"] = phase_relation(
            spec["u_a"], spec["u_b"], spec["out"], mask, n_boot, n_perm, seed)

    legacy = []
    for crit in (PROPOSED, *SENSITIVITY):
        cls = classify_neurons(spec["u_a"], spec["u_b"], spec["out"], crit)
        cls_mask = cls.pop("mask")
        cls["phase_relation"] = phase_relation(spec["u_a"], spec["u_b"], spec["out"],
                                               cls_mask, n_boot, n_perm, seed)
        legacy.append(cls)

    params = {"step": meta["step"], "seed": int(seed), "n_boot": int(n_boot),
              "n_perm": int(n_perm), "n_exemplars": int(n_exemplars), "chunk": int(chunk),
              "cv_folds": int(cv_folds), "curves": list(CURVE_NAMES),
              "wave_fit_k_rule": "dominant_frequency_of_each_column",
              "key_frequency_selection": key_info,
              "key_frequencies": [int(k) for k in key_freqs],
              "structured_neuron_definitions": defs["params"],
              "neuron_tables": tables["params"],
              "activation_analysis": act["params"],
              "logit_contributions": contrib["params"],
              "status": "MEASUREMENT ONLY — thresholds pending docs/HUMAN_DECISIONS.md 9.6"}
    payload = envelope(MODULE, MODULE_VERSION, meta, params)
    payload["results"] = {
        "n_neurons": tables["n_neurons"],
        "neuron_tables": summarize_tables(tables),
        "wave_fits": {name: summarize_fits(fits[name]) for name in CURVE_NAMES},
        "activation": act["summary"],
        "logit_contributions": contrib["summary"],
        "structured_neurons": {"primary": defs["primary"], "n_live": defs["n_live"],
                               "definitions": definitions_out, "jaccard": defs["jaccard"]},
        "sum_dependence": {k: v for k, v in sums.items() if not isinstance(v, np.ndarray)},
        "criteria_sweep_legacy": legacy,
    }

    arrays = _table_arrays(tables)
    arrays.update(_fit_arrays(fits))
    for key, value in act["per_neuron"].items():
        arrays[f"act__{key}"] = value
    arrays["act__exemplar_index"] = act["exemplars"]["index"]
    arrays["act__exemplar_power2d"] = act["exemplars"]["power2d"].astype(np.float32)
    for key, value in contrib["per_neuron"].items():
        arrays[f"logit__{key}"] = value
    for name, entry in defs["definitions"].items():
        arrays[f"mask__{name}"] = entry["mask"]
    arrays["mask__alive"] = defs["alive"]
    path = write_result(run_dir, MODULE, meta["tag"], payload, arrays)
    return payload, path


def main() -> None:
    ap = argparse.ArgumentParser(description="MLP per-neuron mechanism analysis (INTERFACES 5)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: the legacy model_final.pt)")
    ap.add_argument("--key-rule", default=PRIMARY_KEY_RULE,
                    help=f"key-frequency rule (INTERFACES 4; default {PRIMARY_KEY_RULE!r})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-exemplars", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--cv-folds", type=int, default=0)
    ap.add_argument("--all-checkpoints", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="ALSO copy the JSON here (the canonical location is always "
                         "<run_dir>/analysis/mlp_mechanism/<tag>.json)")
    args = ap.parse_args()

    if args.all_checkpoints:
        from ..checkpoints import list_checkpoints
        try:
            steps = [e["step"] for e in list_checkpoints(args.run_dir)]
        except FileNotFoundError as exc:
            raise ValueError(f"{args.run_dir}: --all-checkpoints needs checkpoints.json ({exc})") from exc
    else:
        steps = [args.step]

    for step in steps:
        payload, path = analyse(args.run_dir, step, args.key_rule, args.seed, args.n_boot,
                                args.n_perm, args.n_exemplars, args.chunk, args.cv_folds)
        print(json.dumps({key: payload[key] for key in
                          ("module", "module_version", "run_id", "arch", "p", "step", "params")},
                         indent=2))
        print(json.dumps(payload["results"], indent=2))
        print(f"[saved] {path}", flush=True)
        if args.out is not None:
            args.out.write_text(path.read_text())
            print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
