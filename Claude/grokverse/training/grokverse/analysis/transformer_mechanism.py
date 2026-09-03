"""Per-neuron and per-head mechanism of the 1-layer transformer (INTERFACES §6, WP-3).

WHY THIS MODULE EXISTS
----------------------
Master prompt §8 is explicit: *the transformer must undergo the same mechanism tests as
the MLP.* It is not permissible to assume the Fourier circuit for the transformer because
its ``W_E`` spectrum looks sparse, and to interrogate only the MLP. Every §5 battery
(``neuron_tables``, ``activation_analysis``, ``logit_contributions``,
``structured_neuron_definitions``, ``phase_relation``) is therefore applied here to the
transformer's own effective curves, by *calling the same functions* — not by
re-implementing them, so a threshold or a metric definition can never drift between the
two architectures (master prompt §17).

THE ARCHITECTURE (models/transformer.py, no LayerNorm — so every map below is exact)
-----------------------------------------------------------------------------------
With ``s in {0: a, 1: b, 2: '='}`` and the prediction read off position 2::

    x_s    = W_E[tok_s] + W_pos[s]
    z_h    = sum_s attn[h, 2, s] * (x_s @ W_V[h])
    r_pre  = x_2 + sum_h z_h @ W_O[h]                  # residual entering the MLP
    h_f    = ReLU(r_pre @ W_in[:, f])
    r_post = r_pre + sum_f h_f * W_out[f]
    logits = r_post @ W_U

which splits the output *exactly* into a path that contains no hidden neuron and a sum
over neurons::

    logits = r_pre @ W_U            +  sum_f h_f * (W_out @ W_U)[f]
             ^ direct path             ^ mlp path, one term per hidden neuron

That is the same additive law the MLP obeys, with ``(W_out @ W_U)`` in place of ``W_out``
and the direct path in place of the output bias — which is why
``mlp_mechanism.logit_contributions`` applies unchanged (it accepts both shapes).

WHERE THE TWO ARCHITECTURES GENUINELY DIFFER
--------------------------------------------
The MLP's pre-activation is **exactly** ``u_a[a] + u_b[b] + b_in``: no path lets a and b
interact before the ReLU. The transformer's is not, because ``attn`` depends on the input.
Fixing attention at its grid mean ``abar`` gives the comparable object::

    u_a[a, f] = sum_h abar[h,0] * ((W_E[a] + W_pos[0]) @ W_V[h] @ W_O[h]) @ W_in[:, f]
    u_b[b, f] = sum_h abar[h,1] * ((W_E[b] + W_pos[1]) @ W_V[h] @ W_O[h]) @ W_in[:, f]
    c[f]      = (x_2 + sum_h abar[h,2] * (x_2 @ W_V[h] @ W_O[h])) @ W_in[:, f]

and ``additivity_r2[f]`` reports how much of the *true* pre-activation that reconstruction
captures. ``1 - additivity_r2`` is the share carried by input-dependent attention — the
quantity RESEARCH_SPEC §3.4 asks for. **The MLP has additivity_r2 == 1 by construction**,
so the two numbers are not symmetric evidence and must never be compared as if they were.

Because of this, the activation batteries here are run on the model's **true** ``hidden``
(passed through the ``act=`` argument added to the §5 functions), never on the rectified
reconstruction. ``additivity_r2`` is reported alongside so a reader can see how far the
comparable curve object is from the thing the model actually computes.

STORAGE DEVIATION FROM INTERFACES §6 (deliberate, recorded)
-----------------------------------------------------------
§6 sizes the npz for storing ``hidden``, ``attn`` and the two logit paths. ``hidden`` alone
is ``113^2 * 512 * 4 B = 26 MB`` per checkpoint, ~1 GB over the study's checkpoints — and
it is a deterministic function of a checkpoint whose SHA256 is already in the envelope.
Following master prompt §20 ("do not save unnecessarily large files"), ``analyse`` writes
per-neuron, per-head and per-frequency arrays plus the attention tensor summaries, and NOT
the full ``hidden``/``r_pre``/``r_post`` tensors. ``forward_decomposition`` still returns
them in memory for any caller that wants them.

STATUS: measurement code only. It reports distributions and ablation deltas; it does not
decide whether H1 or H2 holds, and every threshold it applies is the pre-registered one
owned by ``docs/HUMAN_DECISIONS.md``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..config import Config
from . import mask_protocols
from . import metrics as M
from .common import (envelope, grid_logits, load_model_at, masked_ce_and_acc, split_masks,
                     summarize as dist_summary, write_result)
from .fourier import fourier_basis
from .mlp_mechanism import (CURVE_NAMES, PRIMARY_KEY_RULE, PROPOSED, SENSITIVITY,
                            activation_analysis, classify_neurons, curve_spectra,
                            fit_curve_matrix, logit_contributions, neuron_tables,
                            phase_relation, resolve_key_frequencies,
                            structured_neuron_definitions, sum_dependence, summarize_tables)
from .mlp_mechanism import _fit_arrays, _table_arrays  # noqa: F401 — shared npz layout
from .progress_measures import fwd2d
from .wave_fitting import summarize as summarize_fits

MODULE = "transformer_mechanism"
MODULE_VERSION = "1.0"
#: This module is defined for the 1-layer transformer only; the MLPs have their own (§5).
SUPPORTED_ARCH: tuple[str, ...] = ("transformer",)
#: Random same-size frequency sets drawn as the null for a key-subspace share (INTERFACES §0).
N_CONTROL = 50
#: Head-ablation modes. "zero" removes the head's contribution; "mean" replaces it by its
#: grid mean, which keeps the head's average effect and removes only its input dependence.
ABLATION_MODES: tuple[str, ...] = ("zero", "mean")


# --------------------------------------------------------------------------- #
# forward decomposition                                                        #
# --------------------------------------------------------------------------- #
def _state_of(model_or_state) -> dict[str, np.ndarray]:
    """Accept either an ``nn.Module`` (INTERFACES §6 spelling) or a loaded state dict."""
    if hasattr(model_or_state, "named_parameters"):
        return {k: v.detach().cpu().numpy().astype(np.float64)
                for k, v in model_or_state.named_parameters()}
    return {k: np.asarray(v, dtype=np.float64) for k, v in dict(model_or_state).items()}


def _token_grid(cfg: Config) -> np.ndarray:
    """``[p*p, 3]`` token sequences ``[a, b, '=']`` in the grid order a outer, b inner.

    Identical to ``data.make_dataset``'s ``all_x`` ordering, so every ``[p, p, ...]``
    reshape in this module lines up with ``common.grid_logits`` and ``common.split_masks``.
    """
    p = cfg.p
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    eq = np.full(p * p, cfg.equals_token)
    return np.stack([a, b, eq], axis=1)


def attention_weights(state: dict, cfg: Config) -> np.ndarray:
    """Read-out attention over the full grid: ``[p, p, n_heads, 3]``.

    Recomputes the model's softmax exactly (``models/transformer.py``): scaled dot product,
    causal mask, softmax over keys. Only row ``t = 2`` (the ``'='`` position the prediction
    is read from) is kept — the other rows never reach the logits in this architecture.
    """
    p, h, dh = cfg.p, cfg.n_heads, cfg.d_head
    toks = _token_grid(cfg)
    x = state["W_E"][toks] + state["W_pos"][None, :3, :]           # [B, 3, d]
    q = np.einsum("bsd,hde->bhse", x, state["W_Q"])
    k = np.einsum("bsd,hde->bhse", x, state["W_K"])
    scores = np.einsum("bhte,bhse->bhts", q, k) / math.sqrt(dh)
    scores = scores - scores.max(axis=-1, keepdims=True)           # softmax stability
    row = scores[:, :, 2, :]                                       # [B, h, 3]; row 2 sees all
    e = np.exp(row)
    return (e / e.sum(axis=-1, keepdims=True)).reshape(p, p, h, 3)


def forward_decomposition(model_or_state, cfg: Config, attn: np.ndarray | None = None) -> dict:
    """Every intermediate of the forward pass over the full ``(a, b)`` grid (INTERFACES §6).

    Returns ``attn [p,p,h,3]``, ``z [p,p,h,d_head]``, ``r_pre``, ``hidden [p,p,d_mlp]``,
    ``r_post``, ``logits [p,p,p]``, ``direct_path_logits`` (``r_pre @ W_U``) and
    ``mlp_path_logits``, all float64. ``logits == direct + mlp`` exactly, and equals the
    model's own ``logits_last`` to float precision (pinned by the test).

    ``attn`` may be supplied to evaluate a counterfactual attention pattern (the
    "fix attention to its grid mean" test of INTERFACES §6) without touching the weights.
    """
    state = _state_of(model_or_state)
    p = cfg.p
    toks = _token_grid(cfg)
    A = attention_weights(state, cfg) if attn is None else np.asarray(attn, dtype=np.float64)
    if A.shape != (p, p, cfg.n_heads, 3):
        raise ValueError(f"attn must be [p, p, n_heads, 3] = {(p, p, cfg.n_heads, 3)}, got {A.shape}")

    x = state["W_E"][toks] + state["W_pos"][None, :3, :]           # [B, 3, d]
    v = np.einsum("bsd,hde->bhse", x, state["W_V"])                # [B, h, 3, d_head]
    z = np.einsum("bhs,bhse->bhe", A.reshape(p * p, cfg.n_heads, 3), v)
    r_pre = x[:, 2, :] + np.einsum("bhe,hed->bd", z, state["W_O"])
    hidden = np.maximum(r_pre @ state["W_in"], 0.0)                # [B, d_mlp]
    r_post = r_pre + hidden @ state["W_out"]
    W_U = state["W_U"]
    direct = (r_pre @ W_U)[:, :p]
    mlp_path = (hidden @ (state["W_out"] @ W_U))[:, :p]
    d, dm = cfg.d_model, cfg.d_mlp
    return {
        "attn": A,
        "z": z.reshape(p, p, cfg.n_heads, cfg.d_head),
        "r_pre": r_pre.reshape(p, p, d),
        "hidden": hidden.reshape(p, p, dm),
        "r_post": r_post.reshape(p, p, d),
        "logits": (direct + mlp_path).reshape(p, p, p),
        "direct_path_logits": direct.reshape(p, p, p),
        "mlp_path_logits": mlp_path.reshape(p, p, p),
    }


# --------------------------------------------------------------------------- #
# effective operand curves under mean attention                                #
# --------------------------------------------------------------------------- #
def effective_curves(state: dict, cfg: Config, attn_mean: np.ndarray) -> dict[str, np.ndarray]:
    """The transformer's analogue of ``mlp_mechanism.effective_curves`` (INTERFACES §6).

    ``attn_mean`` is ``[n_heads, 3]``, the grid-mean read-out attention. The returned dict
    has exactly the MLP's keys and shapes — ``u_a``, ``u_b`` ``[p, d_mlp]``, ``out``
    ``[p, d_mlp]`` (class axis first), ``b_in`` ``[d_mlp]`` — so every §5 function accepts
    it unchanged.

    ``out[c, f] = (W_out @ W_U)[f, c]``: what neuron ``f`` adds to the logit of class ``c``,
    the transformer's output curve. ``b_in`` is the constant ``c[f]`` contributed by the
    ``'='`` position, which plays the MLP's ``b_in`` role in ``ReLU(u_a + u_b + b_in)``.

    THIS IS A MEAN-ATTENTION APPROXIMATION and the only approximation in this module;
    ``additivity`` measures its error per neuron.
    """
    p = cfg.p
    st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    abar = np.asarray(attn_mean, dtype=np.float64)
    if abar.shape != (cfg.n_heads, 3):
        raise ValueError(f"attn_mean must be [n_heads={cfg.n_heads}, 3], got {abar.shape}")
    OV = np.einsum("hde,hef->hdf", st["W_V"], st["W_O"])           # [h, d, d]
    W_in, W_E, W_pos = st["W_in"], st["W_E"], st["W_pos"]
    x0 = W_E[:p] + W_pos[0]                                        # [p, d]
    x1 = W_E[:p] + W_pos[1]
    x2 = W_E[cfg.equals_token] + W_pos[2]                          # [d]
    u_a = np.einsum("h,ad,hdf,fg->ag", abar[:, 0], x0, OV, W_in, optimize=True)
    u_b = np.einsum("h,bd,hdf,fg->bg", abar[:, 1], x1, OV, W_in, optimize=True)
    const = (x2 + np.einsum("h,d,hdf->f", abar[:, 2], x2, OV, optimize=True)) @ W_in
    return {
        "u_a": u_a,                                                # [p, d_mlp]
        "u_b": u_b,                                                # [p, d_mlp]
        "out": (st["W_out"] @ st["W_U"][:, :p]).T,                 # [p, d_mlp]
        "b_in": const,                                             # [d_mlp]
    }


def additivity(decomp: dict, curves: dict, cfg: Config, state: dict) -> dict:
    """Per neuron: how much of the TRUE pre-activation the mean-attention curves explain.

    ``additivity_r2[f] = 1 - SS_res/SS_tot`` of ``r_pre @ W_in[:, f]`` against
    ``u_a[a, f] + u_b[b, f] + b_in[f]``, over all ``p^2`` inputs. A neuron whose true
    pre-activation is constant has ``SS_tot == 0`` and is reported as ``nan`` and counted in
    ``n_constant_preactivation`` rather than being given a fabricated 1.0.

    ``1 - additivity_r2`` is the share of the pre-activation carried by *input-dependent*
    attention (RESEARCH_SPEC §3.4). For the MLP this quantity is identically 0.
    """
    p = cfg.p
    true_pre = (decomp["r_pre"].reshape(p * p, cfg.d_model)
                @ np.asarray(state["W_in"], dtype=np.float64))     # [p^2, d_mlp]
    a_idx = np.repeat(np.arange(p), p)
    b_idx = np.tile(np.arange(p), p)
    approx = curves["u_a"][a_idx] + curves["u_b"][b_idx] + curves["b_in"]
    ss_res = ((true_pre - approx) ** 2).sum(axis=0)
    ss_tot = ((true_pre - true_pre.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return {
        "additivity_r2": r2,
        "residual_share": 1.0 - r2,
        "n_constant_preactivation": int((ss_tot <= 0).sum()),
        "summary": dist_summary(r2),
        "definition": ("R^2 of the true pre-activation r_pre @ W_in against "
                       "u_a + u_b + b_in built from the GRID-MEAN attention; "
                       "1 - R^2 is the input-dependent-attention share. "
                       "The MLP's value is identically 1 by construction."),
    }


# --------------------------------------------------------------------------- #
# grouped-variance helper                                                      #
# --------------------------------------------------------------------------- #
def variance_explained_by(values: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """``Var(E[v | group]) / Var(v)`` per column, computed exactly.

    ``values`` is ``[p*p, n]``, ``group`` an integer ``[p*p]`` label array. This is the
    same between/total decomposition ``mlp_mechanism.sum_dependence`` performs for
    ``(a+b)`` and ``(a-b)``, generalized to an arbitrary grouping so the attention analysis
    can ask the same question of ``a``, of ``b`` and of ``(a+b) mod p`` (INTERFACES §6).
    Group sizes are not assumed equal.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[:, None] if v.ndim == 1 else v
    n = v.shape[1]
    counts = np.bincount(group, minlength=n_groups).astype(np.float64)
    sums = np.zeros((n_groups, n))
    np.add.at(sums, group, v)
    grand = v.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(counts[:, None] > 0, sums / np.maximum(counts[:, None], 1.0), grand)
    between = (counts[:, None] * (means - grand) ** 2).sum(axis=0)
    total = ((v - grand) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, between / total, 0.0)


# --------------------------------------------------------------------------- #
# key-frequency subspace variance                                              #
# --------------------------------------------------------------------------- #
def key_subspace_variance(T: np.ndarray, p: int, key_freqs, n_control: int = N_CONTROL,
                          seed: int = 0) -> dict:
    """Share of a ``[p, p, n]`` tensor's variance in the ``cos/sin(w_k(a+b))`` directions.

    The projection is ``mask_protocols.SumDirectionsOnly`` — the same operator the audit
    verified against the released Nanda code — applied in the 2-D Fourier coefficient plane
    over ``(a, b)``. "Variance" excludes the constant coefficient, so a tensor with a large
    mean cannot inflate the denominator.

    Reported per key frequency, for the whole key set, and against ``n_control`` random
    frequency sets of the SAME cardinality drawn from the complement of the key set — the
    mandatory size-matched null (master prompt §12). Without it a larger key set would score
    higher for no mechanistic reason.
    """
    keys = mask_protocols._validate(p, key_freqs)
    Fb, _ = fourier_basis(p)
    That = fwd2d(np.asarray(T, dtype=np.float64), Fb)
    total = float((That ** 2).sum() - (That[0, 0] ** 2).sum())     # variance := power minus DC

    def share(ks) -> float:
        if total <= 0:
            return float("nan")
        proj = mask_protocols.SumDirectionsOnly(p, list(ks))._project(That)
        return float((proj ** 2).sum() / total)

    half = (p - 1) // 2
    pool = [k for k in range(1, half + 1) if k not in set(keys)]
    rng = np.random.default_rng(int(seed))
    ctrl = np.array([share(rng.choice(pool, size=len(keys), replace=False))
                     for _ in range(int(n_control))]) if len(pool) >= len(keys) else np.zeros(0)
    observed = share(keys)
    out = {
        "key_frequencies": [int(k) for k in keys],
        "share_total": observed,
        "share_per_key": {str(int(k)): share([k]) for k in keys},
        "total_power_excluding_constant": total,
        "n_control": int(ctrl.size),
        "control_definition": (f"{int(n_control)} random size-{len(keys)} frequency sets drawn "
                               "without replacement from the complement of the key set"),
    }
    if ctrl.size:
        out.update({"control_mean": float(ctrl.mean()), "control_std": float(ctrl.std(ddof=1)),
                    "control_q05": float(np.quantile(ctrl, 0.05)),
                    "control_q95": float(np.quantile(ctrl, 0.95)),
                    "control_max": float(ctrl.max()),
                    "z": (float((observed - ctrl.mean()) / ctrl.std(ddof=1))
                          if ctrl.std(ddof=1) > 0 else float("nan")),
                    "_control_values": ctrl})
    return out


# --------------------------------------------------------------------------- #
# attention, differentiated                                                    #
# --------------------------------------------------------------------------- #
def attention_report(attn: np.ndarray, p: int) -> dict:
    """Per-head attention distribution and what it depends on (INTERFACES §6).

    For each head and each attended position: mean, std, min, max and the 5/95 quantiles
    over all ``p^2`` inputs, plus the share of the weight's variance explained by ``a``, by
    ``b`` and by ``(a+b) mod p``.

    A mean near 0.5 on both operands is a DESCRIPTIVE finding only. Master prompt §5 is
    explicit that a symmetric attention distribution is not causal evidence; the causal
    claim, if any, comes from ``head_ablation``. The spread reported here is what tells a
    reader whether "50/50" is a fixed sum or an average over an input-dependent pattern.
    """
    A = np.asarray(attn, dtype=np.float64)
    n_heads = A.shape[2]
    flat = A.reshape(p * p, n_heads, 3)
    a_idx = np.repeat(np.arange(p), p)
    b_idx = np.tile(np.arange(p), p)
    groups = {"a": a_idx, "b": b_idx, "sum": (a_idx + b_idx) % p}
    heads = []
    for h in range(n_heads):
        entry = {"head": int(h)}
        for s, label in enumerate(("a", "b", "eq")):
            w = flat[:, h, s]
            entry[label] = dist_summary(w) | {
                "variance_explained_by": {g: float(variance_explained_by(w, idx, p)[0])
                                          for g, idx in groups.items()}}
        heads.append(entry)
    return {
        "n_heads": int(n_heads),
        "per_head": heads,
        "mean_over_heads": {label: float(flat[:, :, s].mean())
                            for s, label in enumerate(("a", "b", "eq"))},
        "note": ("descriptive only — a 50/50 mean is not causal evidence for the addition "
                 "circuit (master prompt §5); see head_ablation for the causal test"),
    }


# --------------------------------------------------------------------------- #
# causal head ablation                                                         #
# --------------------------------------------------------------------------- #
def _evaluate(logits: np.ndarray, cfg: Config) -> dict:
    """Train/test loss and accuracy of a ``[p, p, p]`` logit grid on this run's own split."""
    train_mask, test_mask = split_masks(cfg)
    tr_ce, tr_acc = masked_ce_and_acc(logits, train_mask, cfg.p)
    te_ce, te_acc = masked_ce_and_acc(logits, test_mask, cfg.p)
    return {"train_loss": tr_ce, "test_loss": te_ce,
            "train_acc": tr_acc, "test_acc": te_acc}


def _delta(base: dict, ablated: dict) -> dict:
    """Absolute AND relative change of every metric (master prompt §11)."""
    out = {}
    for key, b in base.items():
        a = ablated[key]
        out[key] = a
        out[f"delta_{key}"] = a - b
        out[f"relative_{key}"] = (a - b) / b if b != 0 else float("nan")
    return out


def head_ablation(state: dict, cfg: Config, decomp: dict | None = None,
                  modes: tuple[str, ...] = ABLATION_MODES) -> dict:
    """Zero- and mean-ablate each attention head, and fix attention to its grid mean.

    Every ablation is applied to the UNMODIFIED checkpoint at evaluation time — no weights
    are changed and nothing is retrained (master prompt §11's ablation rules).

    * ``zero``: the head's ``z_h`` is set to 0 — the head contributes nothing.
    * ``mean``: ``z_h`` is replaced by its grid mean, which preserves the head's average
      contribution and removes only its input dependence. A head that matters merely as a
      constant offset therefore survives ``mean`` and fails ``zero``, and the pair separates
      those two roles.
    * ``fix_attention_to_mean``: all heads keep their weights but attention is frozen at
      ``abar``. If the loss barely moves, attention is acting as a fixed sum over the two
      operands and not as a computation — RESEARCH_SPEC §3.4 case (i).

    A size-matched random control is NOT defined for a single head (there is nothing of the
    same size to draw); the comparison across heads plays that role here, and the neuron-set
    controls live in ``analysis/causal_ablation.py``.
    """
    st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    p, n_heads = cfg.p, cfg.n_heads
    d = decomp if decomp is not None else forward_decomposition(st, cfg)
    base = _evaluate(d["logits"], cfg)

    z = d["z"].reshape(p * p, n_heads, cfg.d_head)
    toks = _token_grid(cfg)
    x2 = (st["W_E"][toks] + st["W_pos"][None, :3, :])[:, 2, :]
    W_U, W_in, W_out = st["W_U"], st["W_in"], st["W_out"]

    def logits_from_z(zz: np.ndarray) -> np.ndarray:
        r_pre = x2 + np.einsum("bhe,hed->bd", zz, st["W_O"])
        r_post = r_pre + np.maximum(r_pre @ W_in, 0.0) @ W_out
        return (r_post @ W_U)[:, :p].reshape(p, p, p)

    per_head = []
    for h in range(n_heads):
        entry = {"head": int(h)}
        for mode in modes:
            zz = z.copy()
            zz[:, h, :] = 0.0 if mode == "zero" else z[:, h, :].mean(axis=0)
            entry[mode] = _delta(base, _evaluate(logits_from_z(zz), cfg))
        per_head.append(entry)

    abar = d["attn"].reshape(p * p, n_heads, 3).mean(axis=0)
    fixed = np.broadcast_to(abar, (p * p, n_heads, 3)).reshape(p, p, n_heads, 3)
    fixed_eval = _evaluate(forward_decomposition(st, cfg, attn=fixed)["logits"], cfg)
    return {
        "baseline": base,
        "per_head": per_head,
        "fix_attention_to_mean": _delta(base, fixed_eval),
        "attention_mean": abar,
        "modes": list(modes),
        "rules": ("unmodified checkpoint, no retraining; absolute and relative changes; "
                  "evaluated on this run's own train/test split"),
    }


# --------------------------------------------------------------------------- #
# input / output direction spectra                                             #
# --------------------------------------------------------------------------- #
def _spectrum_summary(Y: np.ndarray, p: int, label: str) -> dict:
    """Dominant frequency, dominant fraction and top-k concentration of every column."""
    spec = curve_spectra(np.asarray(Y, dtype=np.float64), p)
    met = M.curve_metrics(np.asarray(Y, dtype=np.float64), p, (1, 4, 8), M.MAX_ODD_HARMONIC, 0.8)
    ks, counts = np.unique(spec["dominant_freq"], return_counts=True)
    return {
        "object": label,
        "n_columns": int(np.shape(Y)[1]),
        "dominant_fraction": dist_summary(spec["dominant_fraction"]),
        "top8_concentration": dist_summary(met["topk_concentration"][8]),
        "spectral_entropy": dist_summary(met["spectral_entropy"]),
        "dominant_frequency_histogram": {int(k): int(c) for k, c in zip(ks, counts)},
        "_arrays": {"dominant_freq": spec["dominant_freq"],
                    "dominant_fraction": spec["dominant_fraction"]},
    }


def direction_spectra(state: dict, cfg: Config) -> dict:
    """Fourier spectra of the transformer's input and output directions (INTERFACES §6).

    * ``embedding_to_neuron_direct``: ``W_E[:p] @ W_in`` — what a neuron would read if
      attention and ``W_pos`` were ignored. It is NOT the effective operand curve (that is
      ``effective_curves``); it is reported because it is the object the legacy analysis
      implicitly used, so the two can be compared.
    * ``embedding_to_neuron_head{h}``: ``W_E[:p] @ W_V[h] @ W_O[h] @ W_in`` — the path a
      single head actually opens, per head.
    * ``neuron_logit_map``: rows of ``W_out @ W_U[:, :p]``, i.e. the ``out`` curve.
    * ``unembedding``: columns of ``W_U[:, :p]`` as functions of the class index.
    * ``W_pos`` norms per position (no spectrum: 3 positions carry no frequency).
    """
    st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    p = cfg.p
    W_E, W_in = st["W_E"][:p], st["W_in"]
    out: dict[str, dict] = {"embedding_to_neuron_direct":
                            _spectrum_summary(W_E @ W_in, p, "W_E[:p] @ W_in")}
    OV = np.einsum("hde,hef->hdf", st["W_V"], st["W_O"])
    for h in range(cfg.n_heads):
        out[f"embedding_to_neuron_head{h}"] = _spectrum_summary(
            W_E @ OV[h] @ W_in, p, f"W_E[:p] @ W_V[{h}] @ W_O[{h}] @ W_in")
    out["neuron_logit_map"] = _spectrum_summary(
        (st["W_out"] @ st["W_U"][:, :p]).T, p, "(W_out @ W_U[:, :p]) rows")
    out["unembedding"] = _spectrum_summary(st["W_U"][:, :p].T, p, "W_U[:, :p] columns")
    out["W_pos_norms"] = {f"position_{s}": float(np.linalg.norm(st["W_pos"][s]))
                          for s in range(cfg.n_ctx)}
    return out


# --------------------------------------------------------------------------- #
# run-level analysis                                                           #
# --------------------------------------------------------------------------- #
def analyse(run_dir, step: int | None = None, key_rule: str = PRIMARY_KEY_RULE, seed: int = 0,
            n_boot: int = 2000, n_perm: int = 2000, n_exemplars: int = 16,
            chunk: int = 128, cv_folds: int = 0, n_control: int = N_CONTROL) -> tuple[dict, Path]:
    """Run every INTERFACES §6 measurement on one checkpoint and write the JSON + npz.

    The §5 batteries are called on the transformer's effective curves with the model's TRUE
    ``hidden`` supplied as the activation, and with the direct (attention-only) path in the
    role the MLP's output bias plays. Nothing here is re-implemented from ``mlp_mechanism``.
    """
    run_dir = Path(run_dir)
    cfg, model, state, meta = load_model_at(run_dir, step)
    if cfg.arch not in SUPPORTED_ARCH:
        raise ValueError(f"{meta['run_id']}: analysis.transformer_mechanism handles "
                         f"{list(SUPPORTED_ARCH)}, got arch={cfg.arch!r} "
                         "(the MLPs have their own module, INTERFACES §5)")
    st = _state_of(state)
    p = cfg.p
    decomp = forward_decomposition(st, cfg)

    # The forward decomposition must reproduce the model it claims to decompose. The
    # checkpoint is float32 and this module works in float64, so the residual is float32
    # accumulation in the reference, not disagreement: against the same model cast to
    # float64 the two agree to ~1e-13 (pinned by tests/test_transformer_mechanism.py).
    # The RELATIVE error is therefore what is reported alongside the absolute one.
    ref = grid_logits(model, cfg)
    max_logit_error = float(np.abs(decomp["logits"] - ref).max())
    logit_scale = float(np.abs(ref).max())

    abar = decomp["attn"].reshape(p * p, cfg.n_heads, 3).mean(axis=0)
    curves = effective_curves(st, cfg, abar)
    add = additivity(decomp, curves, cfg, st)
    hidden = decomp["hidden"]

    tables = neuron_tables(curves, p)
    fits = {name: fit_curve_matrix(curves[name], tables["per_curve"][name]["dominant_frequency"],
                                   int(cv_folds), int(seed)) for name in CURVE_NAMES}
    key_freqs, key_info = resolve_key_frequencies(run_dir, step, key_rule, state, cfg)
    sums = sum_dependence(curves, p, act=hidden)
    act = activation_analysis(curves, p, key_freqs, fits=fits, n_exemplars=int(n_exemplars),
                              chunk=int(chunk), sum_dependence_result=sums, act=hidden)
    contrib = logit_contributions(curves, (st["W_out"] @ st["W_U"][:, :p]),
                                  decomp["direct_path_logits"], p, int(chunk), act=hidden)
    defs = structured_neuron_definitions(tables, act)

    spec = {name: curve_spectra(curves[name], p) for name in CURVE_NAMES}
    definitions_out: dict[str, dict] = {}
    for name, entry in defs["definitions"].items():
        definitions_out[name] = {k: v for k, v in entry.items() if k != "mask"}
        definitions_out[name]["phase_relation"] = phase_relation(
            spec["u_a"], spec["u_b"], spec["out"], entry["mask"], n_boot, n_perm, seed)

    legacy = []
    for crit in (PROPOSED, *SENSITIVITY):
        cls = classify_neurons(spec["u_a"], spec["u_b"], spec["out"], crit)
        cls["phase_relation"] = phase_relation(spec["u_a"], spec["u_b"], spec["out"],
                                               cls.pop("mask"), n_boot, n_perm, seed)
        legacy.append(cls)

    subspace = {name: key_subspace_variance(decomp[name], p, key_freqs, int(n_control), int(seed))
                for name in ("r_pre", "r_post", "hidden", "logits")}
    ctrl_arrays = {f"subspace__{k}__control_values": v.pop("_control_values")
                   for k, v in subspace.items() if "_control_values" in v}

    attn_rep = attention_report(decomp["attn"], p)
    ablation = head_ablation(st, cfg, decomp)
    spectra = direction_spectra(st, cfg)
    spectra_arrays = {f"spectra__{name}__{k}": v
                      for name, entry in spectra.items() if isinstance(entry, dict)
                      and "_arrays" in entry
                      for k, v in entry.pop("_arrays").items()}

    # Share of the logit variance carried by the path that contains no hidden neuron.
    # Both tensors are class-centered first, because a per-(a, b) offset is invisible to
    # the softmax and must not count as explained variance.
    def _centered_power(T: np.ndarray) -> float:
        return float(((T - T.mean(axis=-1, keepdims=True)) ** 2).sum())

    direct_power = _centered_power(decomp["direct_path_logits"])
    total_power = _centered_power(decomp["logits"])

    params = {"step": meta["step"], "seed": int(seed), "n_boot": int(n_boot),
              "n_perm": int(n_perm), "n_exemplars": int(n_exemplars), "chunk": int(chunk),
              "cv_folds": int(cv_folds), "n_control": int(n_control),
              "curves": list(CURVE_NAMES),
              "wave_fit_k_rule": "dominant_frequency_of_each_column",
              "key_frequency_selection": key_info,
              "key_frequencies": [int(k) for k in key_freqs],
              "effective_curve_definition": (
                  "mean-attention operand curves; the ONLY approximation in this module, "
                  "quantified per neuron by additivity_r2"),
              "activation_source": "the model's true hidden layer (not the rectified curves)",
              "bias_term": "the direct (attention-only) logit path, [p, p, p]",
              "structured_neuron_definitions": defs["params"],
              "neuron_tables": tables["params"],
              "activation_analysis": act["params"],
              "logit_contributions": contrib["params"],
              "storage_deviation": (
                  "hidden/r_pre/r_post are NOT written to the npz (26+ MB each, and a "
                  "deterministic function of the checkpoint whose sha256 is in this "
                  "envelope) — master prompt §20"),
              "status": "MEASUREMENT ONLY — thresholds pending docs/HUMAN_DECISIONS.md §9.6"}

    payload = envelope(MODULE, MODULE_VERSION, meta, params)
    payload["results"] = {
        "n_neurons": tables["n_neurons"],
        "forward_check": {
            "max_abs_logit_error_vs_model": max_logit_error,
            "max_abs_logit": logit_scale,
            "relative_logit_error_vs_model": (max_logit_error / logit_scale
                                              if logit_scale > 0 else float("nan")),
            "direct_plus_mlp_equals_logits": float(np.abs(
                decomp["direct_path_logits"] + decomp["mlp_path_logits"]
                - decomp["logits"]).max()),
            "note": ("the decomposition is checked against the model on every run; the "
                     "residual is float32 accumulation in the stored checkpoint, not "
                     "disagreement (float64 model: ~1e-13)"),
        },
        "additivity": {k: v for k, v in add.items() if not isinstance(v, np.ndarray)},
        "direct_path_share_of_logit_variance": (direct_power / total_power
                                                if total_power > 0 else float("nan")),
        "neuron_tables": summarize_tables(tables),
        "wave_fits": {name: summarize_fits(fits[name]) for name in CURVE_NAMES},
        "activation": act["summary"],
        "logit_contributions": contrib["summary"],
        "structured_neurons": {"primary": defs["primary"], "n_live": defs["n_live"],
                               "definitions": definitions_out, "jaccard": defs["jaccard"]},
        "sum_dependence": {k: v for k, v in sums.items() if not isinstance(v, np.ndarray)},
        "criteria_sweep_legacy": legacy,
        "key_subspace_variance": subspace,
        "attention": attn_rep,
        "head_ablation": {k: v for k, v in ablation.items() if k != "attention_mean"},
        "direction_spectra": spectra,
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
    arrays["additivity_r2"] = add["additivity_r2"]
    arrays["curve__u_a"] = curves["u_a"].astype(np.float32)
    arrays["curve__u_b"] = curves["u_b"].astype(np.float32)
    arrays["curve__out"] = curves["out"].astype(np.float32)
    arrays["curve__b_in"] = curves["b_in"]
    arrays["attention_mean"] = ablation["attention_mean"]
    arrays["attention_read_out"] = decomp["attn"].astype(np.float32)
    arrays.update(ctrl_arrays)
    arrays.update(spectra_arrays)
    path = write_result(run_dir, MODULE, meta["tag"], payload, arrays)
    return payload, path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transformer per-neuron and per-head mechanism analysis (INTERFACES §6)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: the legacy model_final.pt)")
    ap.add_argument("--key-rule", default=PRIMARY_KEY_RULE,
                    help=f"key-frequency rule (INTERFACES §4; default {PRIMARY_KEY_RULE!r})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="run on every entry of the run's checkpoints.json")
    args = ap.parse_args()

    steps: list[int | None] = [args.step]
    if args.all_checkpoints:
        ck = Path(args.run_dir) / "checkpoints.json"
        if not ck.exists():
            raise SystemExit(f"{args.run_dir}: no checkpoints.json (legacy run?)")
        steps = [int(e["step"]) for e in json.loads(ck.read_text())["checkpoints"]]
    for s in steps:
        payload, path = analyse(args.run_dir, s, args.key_rule, args.seed,
                                n_control=args.n_control)
        res = payload["results"]
        print(f"{payload['run_id']} step={payload['step']}  "
              f"additivity_r2 median={res['additivity']['summary']['median']:.4f}  "
              f"direct_path_share={res['direct_path_share_of_logit_variance']:.4f}  "
              f"key_subspace(logits)={res['key_subspace_variance']['logits']['share_total']:.4f}"
              f"  -> {path}")


if __name__ == "__main__":
    main()
