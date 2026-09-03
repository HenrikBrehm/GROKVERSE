"""Causal necessity and sufficiency tests for both architectures (INTERFACES §9, WP-4).

WHY THIS MODULE EXISTS
----------------------
`docs/CAUSAL_ABLATION_PLAN.md` §1 states the evidence chain the study commits to:

1. the structure is present (spectra, waveform fits) — **descriptive**;
2. it has the predicted internal relations (phases add) — **mechanistic**;
3. removing it breaks the model far more than removing an equal amount of anything else — **causal**.

**Only step 3 licenses the word "algorithm".** Every earlier module produces steps 1 and 2. This one
produces step 3, and it is the module gate criterion G4 of `docs/PREREGISTRATION.md` §5 reads.

THE RULE THAT MAKES A NUMBER MEAN ANYTHING
-------------------------------------------
Removing more always removes more. A raw damage figure is therefore never reported alone: every ablation
carries a **size-matched random control** — the same number of neurons, frequencies or dimensions, drawn
uniformly, ``n_control`` draws, seeded — and the reported statistic is the observed damage next to that
control's distribution and the resulting ``z``. This is `CAUSAL_ABLATION_PLAN.md` §2 and master prompt
§11 ("Every ablation needs a matching size-matched random control ablation").

WHAT IS DELIBERATELY REPRESENTABLE
-----------------------------------
`CAUSAL_ABLATION_PLAN.md` §6 fixes the reading of *every* outcome in advance, including the two that are
inconvenient: an ablation whose control does comparable damage means the component is **not**
specifically load-bearing, and an ablation that does **no** damage means the structure is present but
unused — a finding about the metric, not about the model. Both are ordinary results here. Nothing in
this module suppresses them, and `tests/test_causal_ablation.py` asserts that the no-damage case is
representable.

NO RETRAINING, EVER
-------------------
Every ablation is applied to the unmodified trained checkpoint at evaluation time; arrays are copied
before modification and no optimizer is ever constructed. A retrained model answers a different
question (`CAUSAL_ABLATION_PLAN.md` §2).

EXACTNESS
---------
Both forward passes below are the models' own, not approximations:

* MLP — ``ReLU(u_a[a] + u_b[b] + b_in) @ W_out + b_out`` is exact (no path lets ``a`` and ``b`` interact
  before the ReLU), so a curve-level ablation is exact too.
* transformer — the decomposition of `analysis.transformer_mechanism` reproduces the model to ~1e-13 in
  float64. The projections are exact **only because this transformer has no LayerNorm**
  (`CAUSAL_ABLATION_PLAN.md` §9, last bullet); if the architecture ever gains one, they are not.

The one place an approximation enters is the transformer's *effective operand curves*, which are built at
the grid-mean attention. This module therefore does **not** ablate them: the transformer's
key-frequency ablations act on ``W_E`` and on the residual stream, which are exact objects
(`CAUSAL_ABLATION_PLAN.md` §5). Where an ablation exists for only one architecture it supports a
statement about that architecture and never a comparison (master prompt §17, plan §7).

STATUS: measurement code only. The interpretation thresholds below are `[AI-PROPOSED]`, listed for
approval as D1 of `docs/HUMAN_DECISIONS.md`, applied here to *label* an outcome and never to filter one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import Config
from . import mask_protocols
from . import metrics as M
from .common import (envelope, load_model_at, masked_ce_and_acc, split_masks,
                     summarize as dist_summary, write_result)
from .fourier import fourier_basis
from .mlp_mechanism import (CURVE_NAMES, PRIMARY_DEFINITION, PRIMARY_KEY_RULE, _wrap,
                            activation_analysis, curve_spectra, effective_curves,
                            fit_curve_matrix, neuron_tables, resolve_key_frequencies,
                            structured_neuron_definitions)
from .progress_measures import fwd2d, inv2d
from . import transformer_mechanism as TM

MODULE = "causal_ablation"
MODULE_VERSION = "1.0"
#: Size-matched random draws per ablation (`CAUSAL_ABLATION_PLAN.md` §2).
N_CONTROL = 50
#: Interpretation thresholds — `CAUSAL_ABLATION_PLAN.md` §6, `[AI-PROPOSED]`, HUMAN_DECISIONS D1.
#: They label an outcome; they never decide what is reported, and they are never tuned on a result.
NECESSARY_TEST_ACC_DROP = 0.5
NECESSARY_CONTROL_MAX_DROP = 0.1
NECESSARY_Z = 3.0
SUFFICIENT_TEST_ACC = 0.9
#: Structured-neuron definitions the structured ablations are repeated under (plan §9, last bullet).
SENSITIVITY_DEFINITIONS = ("sensitivity_family_0.30", "sensitivity_family_0.70")
#: Pruning-curve grid for `ipr_ranked_pruning` (fractions of neurons pruned).
PRUNING_FRACTIONS = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))
#: `wrong_phase_vs_weak_periodicity` group boundaries (plan §4).
PHASE_ERROR_CUT = np.pi / 4          # 45 degrees
WEAK_PERIODICITY_CUT = 0.3


# --------------------------------------------------------------------------- #
# evaluation and reporting                                                     #
# --------------------------------------------------------------------------- #
def evaluate(model_fn, cfg: Config) -> dict:
    """Train/test loss and accuracy of a ``[p, p, p]`` logit grid on this run's own split.

    ``model_fn`` is either a callable returning the grid or the grid itself (INTERFACES §9 spells it
    as a callable; both are accepted so a caller that already has the logits need not wrap them).
    """
    L = np.asarray(model_fn() if callable(model_fn) else model_fn, dtype=np.float64)
    p = cfg.p
    if L.shape != (p, p, p):
        raise ValueError(f"logits must be [p, p, p] = {(p, p, p)}, got {L.shape}")
    train_mask, test_mask = split_masks(cfg)
    train_loss, train_acc = masked_ce_and_acc(L, train_mask, p)
    test_loss, test_acc = masked_ce_and_acc(L, test_mask, p)
    return {"train_loss": train_loss, "test_loss": test_loss,
            "train_acc": train_acc, "test_acc": test_acc, "logits": L}


def _finite(x) -> float | None:
    """``float(x)`` or ``None``. ``common._jsonable`` rejects nan/inf by design — a guard against a
    non-finite value entering a result file unnoticed — so an undefined quantity is written as JSON
    ``null`` next to the reason it is undefined, never as a number that looks real."""
    f = float(x)
    return f if np.isfinite(f) else None


def _labels(p: int) -> np.ndarray:
    return ((np.arange(p)[:, None] + np.arange(p)[None, :]) % p).ravel()


def margin(L: np.ndarray, p: int) -> float:
    """Mean correct-class logit minus the best incorrect logit, over the full grid (plan §3)."""
    flat = np.asarray(L, dtype=np.float64).reshape(p * p, p)
    y = _labels(p)
    rows = np.arange(p * p)
    correct = flat[rows, y]
    others = flat.copy()
    others[rows, y] = -np.inf
    return float((correct - others.max(axis=1)).mean())


def _per_class_accuracy(L: np.ndarray, p: int) -> np.ndarray:
    """Accuracy per true class over the full grid (each class holds exactly p cells)."""
    flat = np.asarray(L, dtype=np.float64).reshape(p * p, p)
    y = _labels(p)
    hit = (flat.argmax(axis=1) == y).astype(np.float64)
    out = np.zeros(p)
    np.add.at(out, y, hit)
    return out / p


def report(base: dict, ablated: dict, cfg: Config, n_components_removed: int | None = None) -> dict:
    """Absolute and relative change of every measured quantity (plan §3).

    ``test_accuracy_drop`` (base minus ablated, so *positive means damage*) is the single statistic the
    size-matched control is compared against and the one gate criterion G4 reads.
    """
    p = cfg.p
    out: dict = {}
    for key in ("train_loss", "test_loss", "train_acc", "test_acc"):
        b, a = float(base[key]), float(ablated[key])
        out[key] = a
        out[f"delta_{key}"] = a - b
        out[f"relative_{key}"] = _finite((a - b) / b) if b != 0 else None
    out["test_accuracy_drop"] = float(base["test_acc"] - ablated["test_acc"])
    out["train_accuracy_drop"] = float(base["train_acc"] - ablated["train_acc"])
    out["mean_abs_logit_change"] = float(np.abs(ablated["logits"] - base["logits"]).mean())
    out["margin_change"] = float(margin(ablated["logits"], p) - margin(base["logits"], p))
    out["n_components_removed"] = (None if n_components_removed is None
                                   else int(n_components_removed))
    delta_pc = _per_class_accuracy(ablated["logits"], p) - _per_class_accuracy(base["logits"], p)
    out["_per_class_accuracy_change"] = delta_pc
    out["per_class_accuracy_change"] = dist_summary(delta_pc)
    return out


def _classify(observed: dict, control_mean_drop: float, z: float, exceeds_all_controls: bool) -> dict:
    """Apply the plan's §6 reading rules. Labels only — nothing is filtered on them.

    ``z`` is undefined when the controls have zero spread (every draw did exactly the same damage).
    The degenerate case is not waved through as "infinitely significant": the z-condition then falls
    back to whether the observed damage exceeds *every* control, which is what a z-test is a proxy for.
    """
    drop = observed["test_accuracy_drop"]
    separated = (z >= NECESSARY_Z) if np.isfinite(z) else bool(exceeds_all_controls)
    necessary = bool(drop >= NECESSARY_TEST_ACC_DROP
                     and control_mean_drop < NECESSARY_CONTROL_MAX_DROP
                     and separated)
    return {
        "necessary_by_plan_rule": necessary,
        "sufficient_by_plan_rule": bool(observed["test_acc"] >= SUFFICIENT_TEST_ACC),
        "comparable_to_control": bool(np.isfinite(z) and abs(z) < NECESSARY_Z
                                      and control_mean_drop >= NECESSARY_CONTROL_MAX_DROP),
        "no_damage": bool(abs(drop) < 1e-9),
        "separated_from_control": bool(separated),
        "thresholds": {"test_acc_drop": NECESSARY_TEST_ACC_DROP,
                       "control_max_drop": NECESSARY_CONTROL_MAX_DROP,
                       "z": NECESSARY_Z, "sufficient_test_acc": SUFFICIENT_TEST_ACC},
        "threshold_status": "[AI-PROPOSED] docs/HUMAN_DECISIONS.md D1 — labels only",
    }


def with_control(observed: dict, controls: list[dict]) -> dict:
    """Pair an observed ablation with its size-matched random control distribution."""
    entry: dict = {"observed": {k: v for k, v in observed.items() if not k.startswith("_")}}
    if not controls:
        entry["control"] = {"n": 0, "note": "no size-matched control is defined for this ablation"}
        entry["z"] = None
        entry["z_undefined_reason"] = "no control defined for this ablation"
        entry["reading"] = _classify(observed, float("nan"), float("nan"), False)
        return entry
    drops = np.array([c["test_accuracy_drop"] for c in controls], dtype=np.float64)
    std = float(drops.std(ddof=1)) if drops.size > 1 else 0.0
    mean = float(drops.mean())
    exceeds_all = bool(observed["test_accuracy_drop"] > drops.max())
    z = (observed["test_accuracy_drop"] - mean) / std if std > 0 else float("inf")
    entry["control"] = {
        "n": int(drops.size),
        "test_accuracy_drop": {"mean": mean, "std": std,
                               "q05": float(np.quantile(drops, 0.05)),
                               "q95": float(np.quantile(drops, 0.95)),
                               "min": float(drops.min()), "max": float(drops.max())},
        "test_acc": dist_summary(np.array([c["test_acc"] for c in controls])),
        "mean_abs_logit_change": dist_summary(
            np.array([c["mean_abs_logit_change"] for c in controls])),
    }
    entry["z"] = _finite(z)
    if entry["z"] is None:
        entry["z_undefined_reason"] = ("every control did exactly the same damage (zero spread); the "
                                       "z-condition falls back to exceeding every control")
    entry["exceeds_all_controls"] = exceeds_all
    entry["reading"] = _classify(observed, mean, z, exceeds_all)
    entry["_control_drops"] = drops
    return entry


# --------------------------------------------------------------------------- #
# frequency helpers                                                            #
# --------------------------------------------------------------------------- #
def _freq_rows(p: int, freqs) -> np.ndarray:
    """Fourier-basis row indices (cos_k, sin_k) of the given frequencies."""
    ks = np.asarray(sorted({int(k) for k in freqs}), dtype=np.int64)
    return np.concatenate([1 + 2 * (ks - 1), 2 + 2 * (ks - 1)]) if ks.size else np.zeros(0, np.int64)


def filter_curve_frequencies(Y: np.ndarray, p: int, freqs, mode: str) -> np.ndarray:
    """Keep or remove the given frequencies of every column of ``Y`` (``[p, n]``).

    ``mode='keep'`` retains those frequencies **plus the constant row** (plan §4: "the key-frequency
    subspace plus the constant"); ``mode='remove'`` zeroes exactly those frequencies and leaves
    everything else, the constant included.
    """
    F, _ = fourier_basis(p)
    coeff = F @ np.asarray(Y, dtype=np.float64)
    rows = _freq_rows(p, freqs)
    if mode == "keep":
        kept = np.zeros_like(coeff)
        kept[0] = coeff[0]
        if rows.size:
            kept[rows] = coeff[rows]
        coeff = kept
    elif mode == "remove":
        if rows.size:
            coeff = coeff.copy()
            coeff[rows] = 0.0
    else:
        raise ValueError(f"mode must be 'keep' or 'remove', got {mode!r}")
    return F.T @ coeff


def random_frequency_sets(p: int, keys, n_control: int, rng) -> list[list[int]]:
    """``n_control`` random frequency sets of the same size, drawn from the complement of ``keys``."""
    half = (p - 1) // 2
    pool = [k for k in range(1, half + 1) if k not in set(int(k) for k in keys)]
    size = len(set(int(k) for k in keys))
    if size == 0 or len(pool) < size:
        return []
    return [sorted(int(k) for k in rng.choice(pool, size=size, replace=False))
            for _ in range(int(n_control))]


def random_neuron_masks(n: int, size: int, n_control: int, rng) -> list[np.ndarray]:
    """``n_control`` boolean masks selecting ``size`` neurons uniformly at random."""
    if size <= 0 or size > n:
        return []
    out = []
    for _ in range(int(n_control)):
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, size=size, replace=False)] = True
        out.append(m)
    return out


def phase_scramble(Y: np.ndarray, rng) -> np.ndarray:
    """Randomly circular-shift every column — a true phase shift that preserves waveform and amplitude.

    Used as the control for `replace_with_*_fit`: substituting the fitted wave tests the waveform
    model, so its control must keep the same waveform and amplitude and destroy only the phase
    relation between ``u_a``, ``u_b`` and the output curve (plan §4).
    """
    Y = np.asarray(Y, dtype=np.float64)
    p, n = Y.shape
    shifts = rng.integers(0, p, size=n)
    idx = (np.arange(p)[:, None] - shifts[None, :]) % p
    return np.take_along_axis(Y, idx, axis=0)


# --------------------------------------------------------------------------- #
# MLP                                                                          #
# --------------------------------------------------------------------------- #
def _mlp_pieces(state: dict, cfg: Config) -> dict:
    """Everything the MLP ablations need, computed once."""
    p = cfg.p
    curves = effective_curves(state, cfg)
    W_out = np.asarray(state["W_out"], dtype=np.float64)          # [n, p]
    b_out = np.asarray(state["b_out"], dtype=np.float64)          # [p]
    act = np.maximum(curves["u_a"][:, None, :] + curves["u_b"][None, :, :] + curves["b_in"], 0.0)
    act2d = act.reshape(p * p, -1)
    base_logits = (act2d @ W_out + b_out).reshape(p, p, p)
    return {"curves": curves, "W_out": W_out, "b_out": b_out,
            "act2d": act2d, "base_logits": base_logits, "n_neurons": act2d.shape[1]}


def _logits_from_neuron_mask(pieces: dict, keep: np.ndarray, p: int) -> np.ndarray:
    """Logits with only ``keep`` neurons alive — computed from whichever side is smaller."""
    act2d, W, base = pieces["act2d"], pieces["W_out"], pieces["base_logits"]
    removed = ~keep
    if int(removed.sum()) <= int(keep.sum()):
        if not removed.any():
            return base
        return base - (act2d[:, removed] @ W[removed]).reshape(p, p, p)
    if not keep.any():
        return np.broadcast_to(pieces["b_out"], (p, p, p)).copy()
    return (act2d[:, keep] @ W[keep] + pieces["b_out"]).reshape(p, p, p)


def _logits_from_curves(u_a: np.ndarray, u_b: np.ndarray, pieces: dict, p: int) -> np.ndarray:
    act = np.maximum(u_a[:, None, :] + u_b[None, :, :] + pieces["curves"]["b_in"], 0.0)
    return (act.reshape(p * p, -1) @ pieces["W_out"] + pieces["b_out"]).reshape(p, p, p)


def mlp_ablations(state: dict, cfg: Config, masks: dict, key_freqs, spectra: dict, fits: dict,
                  seed: int, n_control: int) -> tuple[dict, dict]:
    """Every ablation of `CAUSAL_ABLATION_PLAN.md` §4. Returns (results, arrays-for-npz)."""
    p = cfg.p
    pieces = _mlp_pieces(state, cfg)
    n = pieces["n_neurons"]
    base = evaluate(pieces["base_logits"], cfg)
    curves = pieces["curves"]
    alive = masks["alive"]
    res: dict = {}
    arrays: dict = {}
    rng = np.random.default_rng(int(seed))

    alive_idx = np.flatnonzero(alive)

    def neuron_ablation(name: str, keep: np.ndarray, removed_count: int) -> None:
        """Ablate to ``keep``; the control keeps the same number of ALIVE neurons at random.

        Matching on the kept count (not the removed one) makes the control size-matched in both
        directions at once: for a `remove_*` ablation it removes as many, for a `keep_*` ablation it
        keeps as many. Dead neurons are excluded from the draw because zeroing one is a no-op and
        would silently weaken the control.
        """
        obs = report(base, evaluate(_logits_from_neuron_mask(pieces, keep, p), cfg), cfg,
                     removed_count)
        keep_size = int((keep & alive).sum())
        ctrl = []
        if 0 < keep_size <= alive_idx.size:
            for _ in range(int(n_control)):
                m = np.zeros(n, dtype=bool)
                m[rng.choice(alive_idx, size=keep_size, replace=False)] = True
                ctrl.append(report(base, evaluate(_logits_from_neuron_mask(pieces, m, p), cfg), cfg,
                                   removed_count))
        entry = with_control(obs, ctrl)
        entry["control_definition"] = (f"{keep_size} of {alive_idx.size} live neurons kept at random, "
                                       f"{n_control} draws")
        arrays[f"{name}__per_class_accuracy_change"] = obs["_per_class_accuracy_change"]
        if "_control_drops" in entry:
            arrays[f"{name}__control_drops"] = entry.pop("_control_drops")
        res[name] = entry

    # --- structured-neuron necessity and sufficiency, under every definition -----------------
    for label, S in masks["sets"].items():
        size = int(S.sum())
        suffix = "" if label == masks["primary"] else f"__{label}"
        # `keep_unstructured` IS `remove_structured` (plan §4); reported once, named both ways.
        neuron_ablation(f"remove_structured{suffix}", alive & ~S, size)
        res[f"remove_structured{suffix}"]["also_named"] = "keep_unstructured"
        neuron_ablation(f"keep_structured{suffix}", S, int((alive & ~S).sum()))

    # --- per-frequency necessity within the structured set ----------------------------------
    S = masks["sets"][masks["primary"]]
    per_freq: dict = {}
    dom = spectra["u_a"]["dominant_freq"]
    for k in sorted({int(x) for x in dom[S]}):
        group = S & (dom == k)
        size = int(group.sum())
        if size == 0:
            continue
        name = f"remove_structured_by_frequency__k{k}"
        neuron_ablation(name, alive & ~group, size)
        per_freq[str(k)] = res.pop(name)
        arrays[f"remove_structured_by_frequency__k{k}__per_class_accuracy_change"] = \
            arrays.pop(f"{name}__per_class_accuracy_change")
    res["remove_structured_by_frequency"] = {"per_frequency": per_freq,
                                             "n_frequencies": len(per_freq)}

    # --- key-frequency subspace of the effective curves --------------------------------------
    n_rows = 2 * len({int(k) for k in key_freqs})
    for mode, name in (("remove", "remove_key_freqs_from_curves"),
                       ("keep", "keep_key_freqs_in_curves")):
        obs_logits = _logits_from_curves(
            filter_curve_frequencies(curves["u_a"], p, key_freqs, mode),
            filter_curve_frequencies(curves["u_b"], p, key_freqs, mode), pieces, p)
        obs = report(base, evaluate(obs_logits, cfg), cfg, n_rows)
        ctrl = []
        for ks in random_frequency_sets(p, key_freqs, n_control, rng):
            lg = _logits_from_curves(filter_curve_frequencies(curves["u_a"], p, ks, mode),
                                     filter_curve_frequencies(curves["u_b"], p, ks, mode), pieces, p)
            ctrl.append(report(base, evaluate(lg, cfg), cfg, n_rows))
        entry = with_control(obs, ctrl)
        arrays[f"{name}__per_class_accuracy_change"] = obs["_per_class_accuracy_change"]
        if "_control_drops" in entry:
            arrays[f"{name}__control_drops"] = entry.pop("_control_drops")
        res[name] = entry

    # --- waveform substitution ---------------------------------------------------------------
    for model, name in (("sinusoid", "replace_with_sinusoid_fit"),
                        ("square", "replace_with_square_fit")):
        pa, pb = fits["u_a"]["pred"][model], fits["u_b"]["pred"][model]
        obs = report(base, evaluate(_logits_from_curves(pa, pb, pieces, p), cfg), cfg, None)
        ctrl = []
        for _ in range(int(n_control)):
            lg = _logits_from_curves(phase_scramble(pa, rng), phase_scramble(pb, rng), pieces, p)
            ctrl.append(report(base, evaluate(lg, cfg), cfg, None))
        entry = with_control(obs, ctrl)
        entry["control_definition"] = ("the same fitted waveform and amplitude with a random circular "
                                       "shift per neuron and curve (phase scrambled)")
        arrays[f"{name}__per_class_accuracy_change"] = obs["_per_class_accuracy_change"]
        if "_control_drops" in entry:
            arrays[f"{name}__control_drops"] = entry.pop("_control_drops")
        res[name] = entry

    # --- which property carries the computation: phase or periodicity ------------------------
    same_freq = ((spectra["u_a"]["dominant_freq"] == spectra["u_b"]["dominant_freq"])
                 & (spectra["out"]["dominant_freq"] == spectra["u_a"]["dominant_freq"]))
    err = np.abs(_wrap(spectra["out"]["dominant_phase"] - spectra["u_a"]["dominant_phase"]
                       - spectra["u_b"]["dominant_phase"]))
    weakness = np.minimum(spectra["u_a"]["dominant_fraction"], spectra["u_b"]["dominant_fraction"])
    g_wrong = alive & same_freq & (err > PHASE_ERROR_CUT)
    g_weak = alive & (err <= PHASE_ERROR_CUT) & (weakness < WEAK_PERIODICITY_CUT)
    m = int(min(g_wrong.sum(), g_weak.sum()))
    comparison: dict = {"n_wrong_phase": int(g_wrong.sum()), "n_weak_periodicity": int(g_weak.sum()),
                        "matched_size": m,
                        "definition": (f"wrong phase: same dominant frequency in u_a, u_b and out with "
                                       f"|phi_out - phi_a - phi_b| > {PHASE_ERROR_CUT:.4f} rad (45 deg); "
                                       f"weak periodicity: phase error within 45 deg but "
                                       f"min(dominant_fraction of u_a, u_b) < {WEAK_PERIODICITY_CUT}")}
    if m > 0:
        for label, group in (("wrong_phase", g_wrong), ("weak_periodicity", g_weak)):
            idx = rng.choice(np.flatnonzero(group), size=m, replace=False)
            sub = np.zeros(len(alive), dtype=bool)
            sub[idx] = True
            nm = f"wrong_phase_vs_weak_periodicity__{label}"
            neuron_ablation(nm, alive & ~sub, m)
            comparison[label] = res.pop(nm)
            arrays[f"{nm}__per_class_accuracy_change"] = arrays.pop(f"{nm}__per_class_accuracy_change")
    else:
        comparison["note"] = "one of the two groups is empty at this checkpoint; nothing to match"
    res["wrong_phase_vs_weak_periodicity"] = comparison

    # --- Doshi's IPR-ranked pruning, in both directions, with the control they do not have ----
    ipr = np.asarray(M.ipr_doshi_neuron(curves["u_a"], curves["u_b"], curves["out"],
                                        "rfft")["ipr_neuron"], dtype=np.float64)
    res["ipr_ranked_pruning"], pruning_arrays = _pruning_block(pieces, cfg, ipr, rng, n_control)
    arrays.update({f"ipr_ranked_pruning__{k}": v for k, v in pruning_arrays.items()})
    arrays["ipr_neuron"] = ipr
    return res, arrays


def _pruning_block(pieces: dict, cfg: Config, score: np.ndarray, rng, n_control: int) -> tuple[dict, dict]:
    """IPR-ranked cumulative pruning in both directions against random-order pruning.

    Doshi, Das, He & Gromov (arXiv:2310.13061) §2.1 / Fig. 6 prune a trained network one neuron at a
    time and track train and test loss and accuracy. **Their protocol has no random-order control** —
    that is our addition, and it is what makes the sweep answer H5 instead of only reproducing the
    curve (`CAUSAL_ABLATION_PLAN.md` §4).
    """
    lowest_first = np.argsort(score, kind="stable")                 # prune least concentrated first
    highest_first = lowest_first[::-1].copy()
    curves_out = {"lowest_ipr_first": _pruning_curve_generic(pieces, lowest_first, cfg),
                  "highest_ipr_first": _pruning_curve_generic(pieces, highest_first, cfg)}
    n = pieces["n_neurons"]
    random_curves = [_pruning_curve_generic(pieces, rng.permutation(n), cfg)
                     for _ in range(int(n_control))]
    test_acc = np.array([c["test_acc"] for c in random_curves])     # [n_control, n_points]
    block = {
        "fraction_pruned": curves_out["lowest_ipr_first"]["fraction_pruned"],
        "lowest_ipr_first": {k: v for k, v in curves_out["lowest_ipr_first"].items()},
        "highest_ipr_first": {k: v for k, v in curves_out["highest_ipr_first"].items()},
        "random_order_control": {
            "n": int(test_acc.shape[0]),
            "test_acc_mean": test_acc.mean(axis=0).tolist(),
            "test_acc_q05": np.quantile(test_acc, 0.05, axis=0).tolist(),
            "test_acc_q95": np.quantile(test_acc, 0.95, axis=0).tolist(),
        },
        "protocol": ("Doshi et al. arXiv:2310.13061 §2.1/Fig. 6; the random-order control is OURS, "
                     "their protocol has none"),
        "grid": f"{len(PRUNING_FRACTIONS)} fractions, cumulative, no retraining",
    }
    arrays = {"random_control_test_acc": test_acc,
              "lowest_ipr_first_test_acc": np.array(curves_out["lowest_ipr_first"]["test_acc"]),
              "highest_ipr_first_test_acc": np.array(curves_out["highest_ipr_first"]["test_acc"])}
    return block, arrays


# --------------------------------------------------------------------------- #
# transformer                                                                  #
# --------------------------------------------------------------------------- #
def _txf_logits_from_r_pre(r_pre: np.ndarray, state: dict, cfg: Config) -> np.ndarray:
    """Finish the forward pass from a (possibly modified) residual stream."""
    p = cfg.p
    flat = r_pre.reshape(p * p, cfg.d_model)
    hidden = np.maximum(flat @ state["W_in"], 0.0)
    return ((flat + hidden @ state["W_out"]) @ state["W_U"])[:, :p].reshape(p, p, p)


def transformer_ablations(state: dict, cfg: Config, masks: dict, key_freqs, decomp: dict,
                          seed: int, n_control: int) -> tuple[dict, dict]:
    """Every ablation of `CAUSAL_ABLATION_PLAN.md` §5. Returns (results, arrays-for-npz)."""
    p = cfg.p
    st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    base = evaluate(decomp["logits"], cfg)
    hidden2d = decomp["hidden"].reshape(p * p, cfg.d_mlp)
    NL = st["W_out"] @ st["W_U"][:, :p]                              # [d_mlp, p] neuron -> logit
    direct = decomp["direct_path_logits"]
    alive = masks["alive"]
    n = hidden2d.shape[1]
    res: dict = {}
    arrays: dict = {}
    rng = np.random.default_rng(int(seed))

    def neuron_logits(keep: np.ndarray) -> np.ndarray:
        removed = ~keep
        if int(removed.sum()) <= int(keep.sum()):
            if not removed.any():
                return decomp["logits"]
            return decomp["logits"] - (hidden2d[:, removed] @ NL[removed]).reshape(p, p, p)
        if not keep.any():
            return direct
        return direct + (hidden2d[:, keep] @ NL[keep]).reshape(p, p, p)

    def add(name: str, obs: dict, ctrl: list[dict], **extra) -> None:
        entry = with_control(obs, ctrl)
        entry.update(extra)
        arrays[f"{name}__per_class_accuracy_change"] = obs["_per_class_accuracy_change"]
        if "_control_drops" in entry:
            arrays[f"{name}__control_drops"] = entry.pop("_control_drops")
        res[name] = entry

    # --- structured hidden neurons ------------------------------------------------------------
    alive_idx = np.flatnonzero(alive)

    def neuron_ablation(name: str, keep: np.ndarray, removed_count: int) -> None:
        """As in the MLP: the control keeps the same number of ALIVE neurons at random."""
        obs = report(base, evaluate(neuron_logits(keep), cfg), cfg, removed_count)
        keep_size = int((keep & alive).sum())
        ctrl = []
        if 0 < keep_size <= alive_idx.size:
            for _ in range(int(n_control)):
                m = np.zeros(n, dtype=bool)
                m[rng.choice(alive_idx, size=keep_size, replace=False)] = True
                ctrl.append(report(base, evaluate(neuron_logits(m), cfg), cfg, removed_count))
        add(name, obs, ctrl,
            control_definition=(f"{keep_size} of {alive_idx.size} live neurons kept at random, "
                                f"{n_control} draws"))

    for label, S in masks["sets"].items():
        size = int(S.sum())
        suffix = "" if label == masks["primary"] else f"__{label}"
        neuron_ablation(f"remove_structured_neurons{suffix}", alive & ~S, size)
        neuron_ablation(f"keep_structured_neurons{suffix}", S, int((alive & ~S).sum()))

    # --- key frequencies of the embedding (a weight change: attention changes too) -------------
    n_rows = 2 * len({int(k) for k in key_freqs})
    for mode, name in (("remove", "remove_key_freqs_from_embedding"),
                       ("keep", "keep_key_freqs_in_embedding")):
        def with_embedding(ks):
            s2 = {k: v.copy() for k, v in st.items()}
            s2["W_E"] = s2["W_E"].copy()
            s2["W_E"][:p] = filter_curve_frequencies(s2["W_E"][:p], p, ks, mode)
            return TM.forward_decomposition(s2, cfg)["logits"]
        obs = report(base, evaluate(with_embedding(key_freqs), cfg), cfg, n_rows)
        ctrl = [report(base, evaluate(with_embedding(ks), cfg), cfg, n_rows)
                for ks in random_frequency_sets(p, key_freqs, n_control, rng)]
        add(name, obs, ctrl,
            note="W_E feeds Q and K as well, so attention is recomputed under this ablation")

    # --- attention -----------------------------------------------------------------------------
    z = decomp["z"].reshape(p * p, cfg.n_heads, cfg.d_head)
    toks = TM._token_grid(cfg)
    x2 = (st["W_E"][toks] + st["W_pos"][None, :3, :])[:, 2, :]

    def logits_from_z(zz: np.ndarray) -> np.ndarray:
        r_pre = x2 + np.einsum("bhe,hed->bd", zz, st["W_O"])
        return _txf_logits_from_r_pre(r_pre.reshape(p, p, cfg.d_model), st, cfg)

    ov_total = np.einsum("bhe,hed->bd", z, st["W_O"])                 # [p*p, d_model]
    for h in range(cfg.n_heads):
        for mode in ("zero", "mean"):
            zz = z.copy()
            zz[:, h, :] = 0.0 if mode == "zero" else z[:, h, :].mean(axis=0)
            obs = report(base, evaluate(logits_from_z(zz), cfg), cfg, cfg.d_head)
            ctrl = []
            for _ in range(int(n_control)):
                Q, _ = np.linalg.qr(rng.standard_normal((cfg.d_model, cfg.d_head)))
                removed = ov_total @ Q @ Q.T
                r_pre = (x2 + ov_total - removed).reshape(p, p, cfg.d_model)
                ctrl.append(report(base, evaluate(_txf_logits_from_r_pre(r_pre, st, cfg), cfg), cfg,
                                   cfg.d_head))
            add(f"ablate_head_{h}__{mode}", obs, ctrl,
                control_definition=("a random rank-d_head orthogonal subspace projected out of the "
                                    "summed OV output"))

    abar = decomp["attn"].reshape(p * p, cfg.n_heads, 3).mean(axis=0)
    fixed = np.broadcast_to(abar, (p * p, cfg.n_heads, 3)).reshape(p, p, cfg.n_heads, 3)
    obs = report(base, evaluate(TM.forward_decomposition(st, cfg, attn=fixed)["logits"], cfg), cfg, None)
    add("fix_attention_to_mean", obs, [],
        note=("structural test, no size-matched control is defined (plan §5). If the loss barely "
              "moves, attention is a fixed weighted sum and the transformer is additive-then-ReLU "
              "like the MLP; if it moves, there is a multiplicative path the MLP does not have. "
              "Either outcome is reported."))

    # --- key-frequency subspace of the residual stream ------------------------------------------
    Fb, _ = fourier_basis(p)
    r_hat = fwd2d(decomp["r_pre"], Fb)

    def residual_without(ks):
        proj = mask_protocols.SumDirectionsOnly(p, list(ks))._project(r_hat)
        return _txf_logits_from_r_pre(inv2d(r_hat - proj, Fb), st, cfg)

    obs = report(base, evaluate(residual_without(key_freqs), cfg), cfg, n_rows)
    ctrl = [report(base, evaluate(residual_without(ks), cfg), cfg, n_rows)
            for ks in random_frequency_sets(p, key_freqs, n_control, rng)]
    add("remove_key_subspace_from_residual", obs, ctrl)

    # --- the restricted loss, reported as a sufficiency test -------------------------------------
    l_hat = fwd2d(decomp["logits"], Fb)

    def restricted(ks):
        return inv2d(mask_protocols.SumDirectionsOnly(p, list(ks)).restrict(l_hat), Fb)

    obs = report(base, evaluate(restricted(key_freqs), cfg), cfg, None)
    ctrl = [report(base, evaluate(restricted(ks), cfg), cfg, None)
            for ks in random_frequency_sets(p, key_freqs, n_control, rng)]
    add("restricted_circuit_only", obs, ctrl,
        note="this IS the restricted loss of mask_protocols.sum_directions_only, reported here as a "
             "causal sufficiency test (plan §5)")

    # --- IPR-ranked pruning of the hidden neurons ------------------------------------------------
    txf_pieces = {"act2d": hidden2d, "W_out": NL, "b_out": np.zeros(p), "n_neurons": n,
                  "base_logits": decomp["logits"]}
    eff = TM.effective_curves(st, cfg, abar)
    ipr = np.asarray(M.ipr_doshi_neuron(eff["u_a"], eff["u_b"], eff["out"], "rfft")["ipr_neuron"],
                     dtype=np.float64)
    block, pruning_arrays = _pruning_block_txf(txf_pieces, direct, cfg, ipr, rng, n_control)
    res["ipr_ranked_pruning"] = block
    arrays.update({f"ipr_ranked_pruning__{k}": v for k, v in pruning_arrays.items()})
    arrays["ipr_neuron"] = ipr
    return res, arrays


def _pruning_block_txf(pieces: dict, direct: np.ndarray, cfg: Config, score: np.ndarray,
                       rng, n_control: int) -> tuple[dict, dict]:
    """Same sweep as `_pruning_block`, with the direct path in the constant term's role."""
    p = cfg.p
    shim = dict(pieces)
    shim["b_out"] = direct.reshape(p * p, p)
    lowest = np.argsort(score, kind="stable")
    highest = lowest[::-1].copy()
    lo = _pruning_curve_generic(shim, lowest, cfg)
    hi = _pruning_curve_generic(shim, highest, cfg)
    rnd = [_pruning_curve_generic(shim, rng.permutation(pieces["n_neurons"]), cfg)
           for _ in range(int(n_control))]
    test_acc = np.array([c["test_acc"] for c in rnd])
    block = {"fraction_pruned": lo["fraction_pruned"], "lowest_ipr_first": lo,
             "highest_ipr_first": hi,
             "random_order_control": {"n": int(test_acc.shape[0]),
                                      "test_acc_mean": test_acc.mean(axis=0).tolist(),
                                      "test_acc_q05": np.quantile(test_acc, 0.05, axis=0).tolist(),
                                      "test_acc_q95": np.quantile(test_acc, 0.95, axis=0).tolist()},
             "protocol": ("Doshi et al. arXiv:2310.13061 applied to the transformer's hidden layer; "
                          "their causal evidence is for a 2-layer quadratic MLP, so this is NEW and "
                          "rests on the random-order control (plan §7)")}
    return block, {"random_control_test_acc": test_acc,
                   "lowest_ipr_first_test_acc": np.array(lo["test_acc"]),
                   "highest_ipr_first_test_acc": np.array(hi["test_acc"])}


def _pruning_curve_generic(pieces: dict, order: np.ndarray, cfg: Config,
                           fractions=PRUNING_FRACTIONS) -> dict:
    """`_pruning_curve` with a ``[p*p, p]`` constant term (the transformer's direct path)."""
    p, n = cfg.p, pieces["n_neurons"]
    act2d, W, const = pieces["act2d"], pieces["W_out"], np.asarray(pieces["b_out"])
    L = np.broadcast_to(const, (p * p, p)).astype(np.float64).copy()
    cuts = sorted({int(round(f * n)) for f in fractions}, reverse=True)
    out: dict[str, list] = {"n_pruned": [], "fraction_pruned": [],
                            "train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    previous = n
    for cut in cuts:
        block = order[cut:previous]
        if block.size:
            L += act2d[:, block] @ W[block]
        previous = cut
        ev = evaluate(L.reshape(p, p, p), cfg)
        out["n_pruned"].append(int(cut))
        out["fraction_pruned"].append(float(cut / n))
        for key in ("train_acc", "test_acc", "train_loss", "test_loss"):
            out[key].append(float(ev[key]))
    for key in out:
        out[key] = out[key][::-1]
    return out


# --------------------------------------------------------------------------- #
# structured-neuron masks: read from the mechanism module, or recompute        #
# --------------------------------------------------------------------------- #
def resolve_structured_masks(run_dir: Path, tag: str, arch: str, curves: dict, cfg: Config,
                             key_freqs, definition: str, seed: int) -> dict:
    """The structured-neuron sets, preferring the mechanism module's own npz.

    Reading them guarantees that the ablated set is *exactly* the set the mechanism analysis
    reported — the alternative, recomputing, could drift if either module changed. The driver runs
    the mechanism modules before this one, so the file normally exists; when it does not (a direct
    CLI call, or a fresh run), the sets are recomputed here and ``source`` says so.
    """
    module = "transformer_mechanism" if arch == "transformer" else "mlp_mechanism"
    npz = Path(run_dir) / "analysis" / module / f"{tag}.npz"
    wanted = (definition,) + SENSITIVITY_DEFINITIONS
    if npz.exists():
        with np.load(npz) as z:
            have = {d for d in wanted if f"mask__{d}" in z.files}
            if "mask__alive" in z.files and definition in have:
                return {"sets": {d: np.asarray(z[f"mask__{d}"], dtype=bool) for d in have},
                        "alive": np.asarray(z["mask__alive"], dtype=bool),
                        "primary": definition,
                        "source": f"analysis/{module}/{tag}.npz"}
    tables = neuron_tables(curves, cfg.p)
    act = activation_analysis(curves, cfg.p, key_freqs, n_exemplars=0)
    defs = structured_neuron_definitions(tables, act)
    return {"sets": {d: defs["definitions"][d]["mask"] for d in wanted if d in defs["definitions"]},
            "alive": defs["alive"], "primary": definition,
            "source": f"recomputed here ({module} npz not found)"}


# --------------------------------------------------------------------------- #
# run-level analysis                                                           #
# --------------------------------------------------------------------------- #
def analyse(run_dir, step: int | None = None, key_rule: str = PRIMARY_KEY_RULE, seed: int = 0,
            structured_definition: str = PRIMARY_DEFINITION, n_control: int = N_CONTROL,
            cv_folds: int = 0) -> tuple[dict, Path]:
    """Run every §4 / §5 ablation on one checkpoint and write the JSON + npz result."""
    run_dir = Path(run_dir)
    cfg, model, state, meta = load_model_at(run_dir, step)
    st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    p = cfg.p
    key_freqs, key_info = resolve_key_frequencies(run_dir, step, key_rule, state, cfg)

    if cfg.arch == "transformer":
        abar_decomp = TM.forward_decomposition(st, cfg)
        abar = abar_decomp["attn"].reshape(p * p, cfg.n_heads, 3).mean(axis=0)
        curves = TM.effective_curves(st, cfg, abar)
    else:
        curves = effective_curves(st, cfg)

    masks = resolve_structured_masks(run_dir, meta["tag"], cfg.arch, curves, cfg, key_freqs,
                                     structured_definition, seed)
    if structured_definition not in masks["sets"]:
        raise ValueError(f"{meta['run_id']}: structured definition {structured_definition!r} is not "
                         f"available; have {sorted(masks['sets'])}")
    spectra = {name: curve_spectra(curves[name], p) for name in CURVE_NAMES}

    if cfg.arch == "transformer":
        results, arrays = transformer_ablations(st, cfg, masks, key_freqs, abar_decomp,
                                                int(seed), int(n_control))
        base_eval = evaluate(abar_decomp["logits"], cfg)
    else:
        tables = neuron_tables(curves, p)
        fits = {name: fit_curve_matrix(curves[name],
                                       tables["per_curve"][name]["dominant_frequency"],
                                       int(cv_folds), int(seed)) for name in ("u_a", "u_b")}
        results, arrays = mlp_ablations(st, cfg, masks, key_freqs, spectra, fits,
                                        int(seed), int(n_control))
        base_eval = evaluate(_mlp_pieces(st, cfg)["base_logits"], cfg)

    gate = _gate_g4(results, masks["primary"])
    params = {
        "step": meta["step"], "seed": int(seed), "n_control": int(n_control),
        "structured_definition": masks["primary"],
        "structured_definitions_available": sorted(masks["sets"]),
        "structured_mask_source": masks["source"],
        "sensitivity_definitions": list(SENSITIVITY_DEFINITIONS),
        "key_frequency_selection": key_info,
        "key_frequencies": [int(k) for k in key_freqs],
        "pruning_fractions": list(PRUNING_FRACTIONS),
        "interpretation_thresholds": {
            "necessary_test_acc_drop": NECESSARY_TEST_ACC_DROP,
            "necessary_control_max_drop": NECESSARY_CONTROL_MAX_DROP,
            "necessary_z": NECESSARY_Z, "sufficient_test_acc": SUFFICIENT_TEST_ACC,
            "status": "[AI-PROPOSED] docs/HUMAN_DECISIONS.md D1"},
        "rules": ("unmodified checkpoint, arrays copied before modification, no retraining; every "
                  "ablation carries a size-matched random control; absolute and relative changes"),
        "status": "MEASUREMENT ONLY — thresholds pending docs/HUMAN_DECISIONS.md D1",
    }
    payload = envelope(MODULE, MODULE_VERSION, meta, params)
    payload["results"] = {
        "baseline": {k: v for k, v in base_eval.items() if k != "logits"},
        "baseline_margin": margin(base_eval["logits"], p),
        "n_neurons": int(curves["u_a"].shape[1]),
        "n_structured": {d: int(m.sum()) for d, m in masks["sets"].items()},
        "n_alive": int(masks["alive"].sum()),
        "ablations": results,
        "gate_criterion_g4": gate,
    }
    for name, mask in masks["sets"].items():
        arrays[f"mask__{name}"] = mask
    arrays["mask__alive"] = masks["alive"]
    path = write_result(run_dir, MODULE, meta["tag"], payload, arrays)
    return payload, path


def _gate_g4(results: dict, definition: str) -> dict:
    """Gate criterion G4 of `docs/PREREGISTRATION.md` §5, evaluated for this seed and checkpoint.

    G4 asks for three things at once: ``remove_structured`` and ``remove_key_freqs`` each *necessary*
    by the plan's §6 rule, and ``keep_structured`` *sufficient*. The per-architecture ids differ, so
    both spellings are looked up; a missing entry makes the criterion ``null``, never a silent False.
    """
    def look(*names):
        for n in names:
            if n in results and isinstance(results[n], dict) and "reading" in results[n]:
                return n, results[n]
        return None, None

    id_s, remove_s = look("remove_structured", "remove_structured_neurons")
    id_ks, keep_s = look("keep_structured", "keep_structured_neurons")
    id_k, remove_k = look("remove_key_freqs_from_curves", "remove_key_subspace_from_residual")
    parts = {
        "remove_structured_necessary": None if remove_s is None else remove_s["reading"]["necessary_by_plan_rule"],
        "remove_key_freqs_necessary": None if remove_k is None else remove_k["reading"]["necessary_by_plan_rule"],
        "keep_structured_sufficient": None if keep_s is None else keep_s["reading"]["sufficient_by_plan_rule"],
    }
    known = [v for v in parts.values() if v is not None]
    parts["g4_passed"] = (all(known) if len(known) == 3 else None)
    parts["definition_used"] = definition
    parts["ids_used"] = {"remove_structured": id_s, "remove_key_freqs": id_k,
                         "keep_structured": id_ks}
    parts["note"] = ("per seed and checkpoint; an architecture passes G4 in the gate only if this "
                     "holds in >= 8 of 10 seeds (PREREGISTRATION 5) — that aggregation is not done here")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal ablations for both architectures (INTERFACES 9)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--key-rule", default=PRIMARY_KEY_RULE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--definition", default=PRIMARY_DEFINITION)
    ap.add_argument("--all-checkpoints", action="store_true")
    args = ap.parse_args()

    steps: list[int | None] = [args.step]
    if args.all_checkpoints:
        ck = Path(args.run_dir) / "checkpoints.json"
        if not ck.exists():
            raise SystemExit(f"{args.run_dir}: no checkpoints.json (legacy run?)")
        steps = [int(e["step"]) for e in json.loads(ck.read_text())]
    for s in steps:
        payload, path = analyse(args.run_dir, s, args.key_rule, args.seed,
                                args.definition, args.n_control)
        g = payload["results"]["gate_criterion_g4"]
        print(f"{payload['run_id']} step={payload['step']}  "
              f"G4 passed={g['g4_passed']}  "
              f"(necessary S={g['remove_structured_necessary']}, "
              f"key={g['remove_key_freqs_necessary']}, sufficient S={g['keep_structured_sufficient']})"
              f"  -> {path}")


if __name__ == "__main__":
    main()
