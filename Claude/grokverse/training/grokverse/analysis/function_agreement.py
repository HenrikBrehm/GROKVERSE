"""Functional agreement of two trained models over all p^2 inputs (INTERFACES.md section 8).

WHAT IS MEASURED
----------------
Two models that both generalize on ``(a op b) mod p`` may still compute different
functions on the cells they get wrong, rank the runner-up classes differently, or
place their logit mass differently. Over every one of the p^2 input pairs (a, b)
this module measures

* argmax agreement and the correctness contingency table (both correct / only a /
  only b / both wrong) on the full grid and on each model's OWN train and test
  split, plus the Jaccard overlap of the two error sets;
* Pearson correlation of the centered logits and of the correct-class margins;
* top-2 agreement (unordered set and ordered);
* the structure of each model's errors (and of the disagreement cells): counts by
  residue ``(a+b) mod p``, by the correct class, by ``a``, by ``b``, by ``|a-b|``,
  and the share of wrong cells whose mirror ``(b, a)`` is also wrong, against a
  seeded random-placement control;
* the list of disagreement cells (npz).

Every number is a measurement. Nothing here decides whether two models "compute
the same function"; ``baseline`` runs the identical comparison on every pair of
runs that share a configuration and differ only by seed, so that a
cross-architecture number can be read against within-configuration, cross-seed
variability.

Split hashes: paired seeds share the split (PREREG_BRIEF "Setting"). ``compare``
records both hashes and their equality and, by default (``require_same_split=True``,
INTERFACES section 8 "assert and record"), raises when they differ; ``baseline``
compares seeds of one configuration, whose splits differ by construction, and
passes ``require_same_split=False`` (echoed in its params). The per-split numbers
are always reported on each model's own split.

Undefined ratios (Jaccard of two empty error sets, symmetric share with no error)
are written as ``null`` with their counts next to them -- never as a convention
value that could pass as a measurement.

Output: ``results/function_agreement/<id_a>__vs__<id_b>.json`` + ``.npz`` where
``<id>`` is the run id for the legacy final model (``step=None``) and
``<run_id>_step{N:06d}`` for a v2 checkpoint.

Usage (from training/):
    python -m grokverse.analysis.function_agreement <run_dir_a> <run_dir_b> [--step N | --step-a N --step-b N]
    python -m grokverse.analysis.function_agreement <run_dir_a> <run_dir_b> --all-checkpoints
    python -m grokverse.analysis.function_agreement --baseline <run_dir> <run_dir> ... [--step N]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..utils import git_commit, utcnow
from .common import (_jsonable, center_logits, envelope, grid_logits,
                     load_model_at, split_hash, split_masks, summarize)

MODULE = "function_agreement"
MODULE_VERSION = "1.1"
DEFAULT_N_CONTROL = 50
DEFAULT_SEED = 0
SPLIT_NAMES = ("train", "test")
HISTOGRAM_NAMES = ("by_residue_sum", "by_target_class", "by_a", "by_b", "by_abs_diff")

#: Config fields allowed to differ between the runs of one baseline group. Everything
#: else must match exactly: a baseline group is "same configuration, other seed", and
#: ``run_id`` does not encode every field (``steps``, ``lr``, ...), so the full config
#: is compared (``label``/``device`` do not enter the computed function; ``run_id`` and
#: ``vocab_size`` are derived).
BASELINE_IGNORED_FIELDS = ("seed", "label", "device", "run_id", "vocab_size")
_SEED_TOKEN = re.compile(r"_seed\d+")

#: Per-pair numbers the baseline table summarizes (mean/min/max/... via ``summarize``).
BASELINE_METRICS = (
    "agreement_rate", "error_jaccard", "both_correct", "only_a_correct",
    "only_b_correct", "both_wrong", "accuracy_a", "accuracy_b",
    "logit_pearson", "margin_pearson", "top2_agreement", "top2_ordered_agreement",
    "agreement_rate_test_a_split", "agreement_rate_test_b_split",
    "error_jaccard_test_a_split", "error_jaccard_test_b_split",
)


def results_dir() -> Path:
    """``training/results/function_agreement`` (INTERFACES section 8)."""
    return Path(__file__).resolve().parents[2] / "results" / MODULE


# --------------------------------------------------------------------------- #
# targets and per-model quantities on a [p, p, p] logit grid                   #
# --------------------------------------------------------------------------- #
def target_grid(p: int, task: str) -> np.ndarray:
    """Correct class for every cell: ``(a + b) mod p`` (add) or ``(a * b) mod p`` (mul)."""
    a = np.arange(p)[:, None]
    b = np.arange(p)[None, :]
    if task == "add":
        return (a + b) % p
    if task == "mul":
        return (a * b) % p
    raise ValueError(f"unknown task {task!r}; expected 'add' or 'mul'")


def _check_grid(L: np.ndarray, p: int, name: str) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    if L.shape != (p, p, p):
        raise ValueError(f"{name}: expected a logit grid of shape {(p, p, p)}, got {L.shape}")
    if not np.isfinite(L).all():
        raise ValueError(f"{name}: logit grid contains non-finite values")
    return L


def _check_splits(splits, p: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    if len(splits) != 2:
        raise ValueError(f"{name}: splits must be (train_mask, test_mask)")
    train, test = (np.asarray(m, dtype=bool) for m in splits)
    if train.shape != (p, p) or test.shape != (p, p):
        raise ValueError(f"{name}: split masks must be [p, p] = {(p, p)}")
    if (train & test).any() or not (train | test).all():
        raise ValueError(f"{name}: train/test masks must partition the p x p grid")
    return train, test


def correct_class_margin(L: np.ndarray, p: int, task: str) -> np.ndarray:
    """``margin[a, b] = L[a, b, y] - max_{c != y} L[a, b, c]`` with ``y`` the correct class.

    Positive iff the cell is classified correctly; its magnitude is the logit gap
    to the strongest competitor.
    """
    y = target_grid(p, task)
    ai, bi = np.indices((p, p))
    correct = L[ai, bi, y]
    others = L.copy()
    others[ai, bi, y] = -np.inf
    return correct - others.max(axis=-1)


def top2_classes(L: np.ndarray) -> np.ndarray:
    """``[p, p, 2]``: best and second-best class per cell (ties: lower index first)."""
    return np.argsort(-L, axis=-1, kind="stable")[..., :2]


def pearson(x: np.ndarray, y: np.ndarray, name: str = "pearson") -> float:
    """Pearson r of the flattened arrays:
    ``sum((x - x_mean)(y - y_mean)) / sqrt(sum (x - x_mean)^2 * sum (y - y_mean)^2)``.

    Raises (naming ``name`` and the side) if either input has zero variance -- r is
    undefined there and no value is invented.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"{name}: shapes differ ({x.shape} vs {y.shape})")
    xc, yc = x - x.mean(), y - y.mean()
    nx, ny = float(np.sqrt((xc ** 2).sum())), float(np.sqrt((yc ** 2).sum()))
    if nx == 0.0 or ny == 0.0:
        side = "model a" if nx == 0.0 else "model b"
        raise ValueError(f"{name} undefined: {side}'s values have zero variance")
    return float((xc * yc).sum() / (nx * ny))


# --------------------------------------------------------------------------- #
# agreement and contingency                                                    #
# --------------------------------------------------------------------------- #
def agreement_block(pred_a: np.ndarray, pred_b: np.ndarray, correct_a: np.ndarray,
                    correct_b: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Argmax agreement and correctness contingency over the cells where ``mask`` (all if None).

    ``agreement_rate = mean(pred_a == pred_b)`` (argmax classes equal -- two models that are
    both wrong with DIFFERENT wrong classes disagree); ``both_correct = mean(correct_a & correct_b)``
    and likewise ``only_a_correct``, ``only_b_correct``, ``both_wrong`` -- fractions of the
    ``n_cells`` selected cells that sum to 1; ``error_jaccard = |E_a & E_b| / |E_a | E_b|``
    with ``E`` the set of wrong cells -- ``None`` (0/0, undefined) when neither model has an
    error on the selected cells; ``n_errors_union`` is always written next to it.
    """
    if mask is None:
        mask = np.ones(pred_a.shape, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        raise ValueError("agreement_block: the mask selects no cells")
    pa, pb, ca, cb = pred_a[mask], pred_b[mask], correct_a[mask], correct_b[mask]
    ea, eb = ~ca, ~cb
    inter, union = int((ea & eb).sum()), int((ea | eb).sum())
    return {
        "n_cells": n,
        "agreement_rate": float((pa == pb).mean()),
        "n_disagreements": int((pa != pb).sum()),
        "accuracy_a": float(ca.mean()),
        "accuracy_b": float(cb.mean()),
        "both_correct": float((ca & cb).mean()),
        "only_a_correct": float((ca & eb).mean()),
        "only_b_correct": float((ea & cb).mean()),
        "both_wrong": float((ea & eb).mean()),
        "n_errors_a": int(ea.sum()),
        "n_errors_b": int(eb.sum()),
        "n_errors_intersection": inter,
        "n_errors_union": union,
        "error_jaccard": (inter / union) if union > 0 else None,
    }


def disagreement_pairs(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    """``[n, 2]`` int64 array of the cells ``(a, b)`` whose argmax classes differ (row-major)."""
    return np.argwhere(pred_a != pred_b).astype(np.int64)


def logit_agreement(L_a: np.ndarray, L_b: np.ndarray, p: int, task: str) -> dict:
    """Logit-level agreement: ``logit_pearson`` on the flattened class-centered grids
    (``center_logits``: per-cell mean over classes removed, so a per-cell offset -- a
    softmax invariance -- cannot change it), ``margin_pearson`` on the correct-class margins,
    ``top2_agreement`` = fraction of cells whose unordered top-2 class sets coincide,
    ``top2_ordered_agreement`` = both ranks equal.
    """
    t2a, t2b = top2_classes(L_a), top2_classes(L_b)
    ordered = (t2a == t2b).all(axis=-1)
    unordered = (np.sort(t2a, axis=-1) == np.sort(t2b, axis=-1)).all(axis=-1)
    return {
        "logit_pearson": pearson(center_logits(L_a), center_logits(L_b), "logit_pearson"),
        "margin_pearson": pearson(correct_class_margin(L_a, p, task),
                                  correct_class_margin(L_b, p, task), "margin_pearson"),
        "top2_agreement": float(unordered.mean()),
        "top2_ordered_agreement": float(ordered.mean()),
    }


# --------------------------------------------------------------------------- #
# error structure                                                              #
# --------------------------------------------------------------------------- #
def error_histograms(error_mask: np.ndarray, p: int, task: str = "add") -> dict[str, np.ndarray]:
    """Counts of the wrong cells by ``(a+b) mod p``, by the correct class ``y[a, b]``
    (identical to ``by_residue_sum`` for ``task='add'``; ``(a*b) mod p`` for ``'mul'``),
    by ``a``, by ``b``, by ``|a-b|``.

    Each histogram has ``p`` bins (``|a-b|`` ranges over 0..p-1) and sums to the error count.
    ``a`` indexes the first (row) axis of the mask, ``b`` the second, as in ``grid_logits``.
    """
    E = np.asarray(error_mask, dtype=bool)
    if E.shape != (p, p):
        raise ValueError(f"error_histograms: mask must be [p, p] = {(p, p)}, got {E.shape}")
    a, b = np.nonzero(E)
    y = target_grid(p, task)
    return {
        "by_residue_sum": np.bincount((a + b) % p, minlength=p),
        "by_target_class": np.bincount(y[a, b], minlength=p),
        "by_a": np.bincount(a, minlength=p),
        "by_b": np.bincount(b, minlength=p),
        "by_abs_diff": np.bincount(np.abs(a - b), minlength=p),
    }


def histogram_summary(counts: np.ndarray) -> dict:
    """JSON summary of a count histogram: total, max, argmax bin, non-zero bins and the
    normalized entropy ``-sum q log q / log(n_bins)`` with ``q = counts / total``
    (1 = uniform over bins; ``None`` when total is 0)."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    entropy = None
    if total > 0 and counts.size > 1:
        q = counts[counts > 0] / total
        entropy = float(-(q * np.log(q)).sum() / np.log(counts.size))
    return {
        "n_bins": int(counts.size), "total": total, "max_count": int(counts.max()),
        "argmax_bin": int(counts.argmax()), "n_nonzero_bins": int((counts > 0).sum()),
        "normalized_entropy": entropy,
    }


def symmetric_error_share(error_mask: np.ndarray) -> dict:
    """Share of wrong cells ``(a, b)`` whose mirror ``(b, a)`` is also wrong.

    ``symmetric_error_share = |{(a,b) in E : (b,a) in E}| / |E|`` -- diagonal cells
    ``a == b`` are their own mirror and count as symmetric; ``symmetric_error_share_offdiag``
    restricts numerator and denominator to ``a != b``. ``None`` when a denominator is 0.
    """
    E = np.asarray(error_mask, dtype=bool)
    p = E.shape[0]
    sym = E & E.T
    off = ~np.eye(p, dtype=bool)
    n, n_sym = int(E.sum()), int(sym.sum())
    n_off, n_sym_off = int((E & off).sum()), int((sym & off).sum())
    return {
        "n_errors": n,
        "n_symmetric": n_sym,
        "symmetric_error_share": (n_sym / n) if n > 0 else None,
        "n_errors_offdiag": n_off,
        "n_symmetric_offdiag": n_sym_off,
        "symmetric_error_share_offdiag": (n_sym_off / n_off) if n_off > 0 else None,
    }


def symmetric_share_control(error_mask: np.ndarray, n_control: int, seed: int) -> dict:
    """Random-placement control for the off-diagonal symmetric-error share.

    Each of the ``n_control`` draws places the same number of wrong cells uniformly
    without replacement over the ``p^2`` cells (``np.random.default_rng(seed)``) and
    measures ``symmetric_error_share_offdiag``; draws with no off-diagonal error are
    undefined and dropped (``n_control_defined``). ``z = (observed - control_mean) / control_std``.
    Also reports the analytic expectation ``(n_errors - 1) / (p^2 - 1)``: given one wrong
    off-diagonal cell, the other ``n_errors - 1`` wrong cells are uniform over the
    remaining ``p^2 - 1`` cells.
    """
    E = np.asarray(error_mask, dtype=bool)
    p, n = E.shape[0], int(E.sum())
    observed = symmetric_error_share(E)["symmetric_error_share_offdiag"]
    rng = np.random.default_rng(seed)
    values = np.full(int(n_control), np.nan)
    for i in range(int(n_control)):
        flat = np.zeros(p * p, dtype=bool)
        flat[rng.choice(p * p, size=n, replace=False)] = True
        v = symmetric_error_share(flat.reshape(p, p))["symmetric_error_share_offdiag"]
        values[i] = np.nan if v is None else v
    s = summarize(values)
    z = None
    if observed is not None and s.get("n", 0) > 1 and s["std"] > 0:
        z = float((observed - s["mean"]) / s["std"])
    return {
        "n_control": int(n_control), "seed": int(seed), "n_control_defined": int(s.get("n", 0)),
        "control_mean": s.get("mean"), "control_std": s.get("std"),
        "control_q05": s.get("q05"), "control_q95": s.get("q95"),
        "expected_share_offdiag_random_placement": ((n - 1) / (p * p - 1)) if n > 0 else None,
        "z": z,
        "control_values": values,
    }


def error_structure(error_mask: np.ndarray, p: int, task: str, n_control: int,
                    seed: int) -> tuple[dict, dict]:
    """JSON summary and npz arrays of one cell mask (histograms + symmetric share).

    The mask is a model's wrong cells or -- under the label ``disagreement`` -- the cells
    where the two argmaxes differ; the ``n_errors`` keys then count those cells.
    """
    hists = error_histograms(error_mask, p, task)
    control = symmetric_share_control(error_mask, n_control, seed)
    control_values = control.pop("control_values")
    summary = {
        "n_errors": int(np.asarray(error_mask, dtype=bool).sum()),
        "histograms": {k: histogram_summary(v) for k, v in hists.items()},
        **symmetric_error_share(error_mask),
        "symmetric_offdiag_control": control,
    }
    arrays = {f"errors_{k}": v for k, v in hists.items()}
    arrays["symmetric_control_values"] = control_values
    return summary, arrays


# --------------------------------------------------------------------------- #
# the pure comparison of two logit grids                                       #
# --------------------------------------------------------------------------- #
def compare_logits(L_a: np.ndarray, L_b: np.ndarray, p: int, task: str, splits_a, splits_b,
                   n_control: int = DEFAULT_N_CONTROL, seed: int = DEFAULT_SEED) -> tuple[dict, dict]:
    """Every section-8 quantity from two ``[p, p, p]`` logit grids.

    ``splits_a`` / ``splits_b`` are each model's own ``(train_mask, test_mask)``. Returns
    ``(results, arrays)``: JSON-ready summaries and the npz arrays (argmax grids, correctness
    masks, split masks, disagreement pairs, error histograms, control draws).
    """
    L_a, L_b = _check_grid(L_a, p, "model a"), _check_grid(L_b, p, "model b")
    splits_a, splits_b = _check_splits(splits_a, p, "model a"), _check_splits(splits_b, p, "model b")
    y = target_grid(p, task)
    pred_a, pred_b = L_a.argmax(axis=-1), L_b.argmax(axis=-1)
    correct_a, correct_b = pred_a == y, pred_b == y

    def per_split(splits) -> dict:
        return {name: agreement_block(pred_a, pred_b, correct_a, correct_b, m)
                for name, m in zip(SPLIT_NAMES, splits)}

    results = {
        "full_grid": agreement_block(pred_a, pred_b, correct_a, correct_b),
        "model_a_split": per_split(splits_a),
        "model_b_split": per_split(splits_b),
        "logits": logit_agreement(L_a, L_b, p, task),
    }
    arrays = {
        "argmax_a": pred_a.astype(np.int16), "argmax_b": pred_b.astype(np.int16),
        "correct_a": correct_a, "correct_b": correct_b,
        "train_mask_a": splits_a[0], "train_mask_b": splits_b[0],
        "disagreement_pairs": disagreement_pairs(pred_a, pred_b),
    }
    errors = {}
    for label, mask in (("model_a", ~correct_a), ("model_b", ~correct_b),
                        ("disagreement", pred_a != pred_b)):
        summary, arrs = error_structure(mask, p, task, n_control, seed)
        errors[label] = summary
        arrays.update({f"{label}_{k}": v for k, v in arrs.items()})
    results["error_structure"] = errors
    return results, arrays


# --------------------------------------------------------------------------- #
# loading, comparing and writing run directories                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSide:
    """One loaded model: config, provenance, full-grid logits and its own split."""

    run_dir: Path
    cfg: Config
    meta: dict
    logits: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    split_hash: str

    @property
    def id(self) -> str:
        step = self.meta["step"]
        if step == "final":
            return self.meta["run_id"]
        return f"{self.meta['run_id']}_step{int(step):06d}"


def load_side(run_dir: Path | str, step: int | None = None) -> ModelSide:
    """Load a run at ``step`` (None = legacy ``model_final.pt``) with logits and split."""
    cfg, model, _state, meta = load_model_at(run_dir, step)
    train, test = split_masks(cfg)
    return ModelSide(Path(run_dir), cfg, meta, grid_logits(model, cfg), train, test, split_hash(cfg))


def _check_comparable(a: ModelSide, b: ModelSide) -> None:
    if a.cfg.p != b.cfg.p:
        raise ValueError(f"cannot compare {a.id} (p={a.cfg.p}) with {b.id} (p={b.cfg.p})")
    if a.cfg.task != b.cfg.task:
        raise ValueError(f"cannot compare {a.id} (task={a.cfg.task!r}) with {b.id} "
                         f"(task={b.cfg.task!r})")


def _write(payload: dict, arrays: dict | None, stem: str, out_dir: Path | None) -> dict:
    """Write ``<out_dir>/<stem>.json`` (+ ``.npz``); returns a new payload with the paths."""
    out_dir = Path(out_dir) if out_dir is not None else results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    out = dict(payload)
    if arrays is not None:
        npz_path = out_dir / f"{stem}.npz"
        np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in arrays.items()})
        out.update({"arrays_file": npz_path.name, "output_npz": str(npz_path)})
    json_path.write_text(json.dumps(_jsonable(out), indent=2, allow_nan=False))
    return {**out, "output_json": str(json_path)}


def _check_split(a: ModelSide, b: ModelSide) -> None:
    if a.split_hash != b.split_hash:
        raise ValueError(
            f"{a.id} and {b.id} do not share a train/test split (split_hash "
            f"{a.split_hash[:12]}... vs {b.split_hash[:12]}...); paired seeds must share the "
            "split (INTERFACES section 8). Pass require_same_split=False "
            "(CLI: --allow-different-splits) for a deliberate cross-split comparison")


def compare_sides(a: ModelSide, b: ModelSide, n_control: int = DEFAULT_N_CONTROL,
                  seed: int = DEFAULT_SEED, out_dir: Path | None = None,
                  require_same_split: bool = True) -> dict:
    """Compare two loaded models and write ``<id_a>__vs__<id_b>.json`` + ``.npz``.

    ``require_same_split`` asserts split-hash equality before anything is measured
    (INTERFACES section 8); the hashes and their equality are recorded either way.
    """
    _check_comparable(a, b)
    if require_same_split:
        _check_split(a, b)
    p, task = a.cfg.p, a.cfg.task
    params = {
        "run_dir_a": a.run_dir, "run_dir_b": b.run_dir,
        "step_a": a.meta["step"], "step_b": b.meta["step"], "task": task,
        "n_control": int(n_control), "seed": int(seed),
        "require_same_split": bool(require_same_split),
    }
    payload = envelope(MODULE, MODULE_VERSION, a.meta, params)
    payload.update({
        "run_id_b": b.meta["run_id"], "arch_b": b.meta["arch"], "p_b": b.meta["p"],
        "step_b": b.meta["step"], "checkpoint_file_b": b.meta.get("checkpoint_file"),
        "checkpoint_sha256_b": b.meta.get("checkpoint_sha256"),
    })
    results, arrays = compare_logits(a.logits, b.logits, p, task,
                                     (a.train_mask, a.test_mask), (b.train_mask, b.test_mask),
                                     n_control, seed)
    results["split"] = {
        "split_hash_a": a.split_hash, "split_hash_b": b.split_hash,
        "split_hash_equal": a.split_hash == b.split_hash,
        "n_train_a": int(a.train_mask.sum()), "n_test_a": int(a.test_mask.sum()),
        "n_train_b": int(b.train_mask.sum()), "n_test_b": int(b.test_mask.sum()),
    }
    payload["results"] = results
    return _write(payload, arrays, f"{a.id}__vs__{b.id}", out_dir)


def compare(run_dir_a: Path | str, run_dir_b: Path | str, step_a: int | None = None,
            step_b: int | None = None, n_control: int = DEFAULT_N_CONTROL,
            seed: int = DEFAULT_SEED, out_dir: Path | None = None,
            require_same_split: bool = True) -> dict:
    """INTERFACES section 8 ``compare``: load both runs and measure their functional agreement.

    ``step=None`` loads the legacy ``model_final.pt``; an integer loads the v2 checkpoint.
    Split-hash equality is asserted (``require_same_split``, default True) and recorded.
    """
    return compare_sides(load_side(run_dir_a, step_a), load_side(run_dir_b, step_b),
                         n_control, seed, out_dir, require_same_split)


def common_checkpoint_steps(run_dir_a: Path | str, run_dir_b: Path | str) -> list[int]:
    """Steps listed in BOTH runs' ``checkpoints.json`` (sorted); raises for legacy runs."""
    from ..checkpoints import list_checkpoints  # lazy: legacy runs need no checkpoints module
    listed = []
    for d in (run_dir_a, run_dir_b):
        try:
            listed.append({int(e["step"]) for e in list_checkpoints(Path(d))})
        except FileNotFoundError as exc:
            raise ValueError(f"{d}: --all-checkpoints needs checkpoints.json ({exc})") from exc
    common = sorted(listed[0] & listed[1])
    if not common:
        raise ValueError(f"no checkpoint step is listed in both {run_dir_a} and {run_dir_b} "
                         f"({sorted(listed[0])} vs {sorted(listed[1])})")
    return common


def compare_all_checkpoints(run_dir_a: Path | str, run_dir_b: Path | str,
                            n_control: int = DEFAULT_N_CONTROL, seed: int = DEFAULT_SEED,
                            out_dir: Path | None = None,
                            require_same_split: bool = True) -> list[dict]:
    """``compare`` at every step present in both runs' ``checkpoints.json`` (same step on
    both sides); one JSON/npz per step, the payloads returned in step order."""
    return [compare(run_dir_a, run_dir_b, s, s, n_control, seed, out_dir, require_same_split)
            for s in common_checkpoint_steps(run_dir_a, run_dir_b)]


# --------------------------------------------------------------------------- #
# within-architecture baseline                                                 #
# --------------------------------------------------------------------------- #
def _id_from(run_id: str, step) -> str:
    return run_id if step == "final" else f"{run_id}_step{int(step):06d}"


def headline(payload: dict) -> dict:
    """The per-pair numbers a baseline table summarizes, pulled from a ``compare`` payload."""
    r = payload["results"]
    fg, lg, sp = r["full_grid"], r["logits"], r["split"]
    grid_keys = ("agreement_rate", "error_jaccard", "both_correct", "only_a_correct",
                 "only_b_correct", "both_wrong", "accuracy_a", "accuracy_b")
    logit_keys = ("logit_pearson", "margin_pearson", "top2_agreement", "top2_ordered_agreement")
    return {
        "id_a": _id_from(payload["run_id"], payload["step"]),
        "id_b": _id_from(payload["run_id_b"], payload["step_b"]),
        "arch_a": payload["arch"], "arch_b": payload["arch_b"],
        "split_hash_equal": sp["split_hash_equal"],
        **{k: fg[k] for k in grid_keys},
        **{k: lg[k] for k in logit_keys},
        "agreement_rate_test_a_split": r["model_a_split"]["test"]["agreement_rate"],
        "agreement_rate_test_b_split": r["model_b_split"]["test"]["agreement_rate"],
        "error_jaccard_test_a_split": r["model_a_split"]["test"]["error_jaccard"],
        "error_jaccard_test_b_split": r["model_b_split"]["test"]["error_jaccard"],
    }


def config_key(cfg: Config) -> str:
    """``run_id`` with its ``_seed{N}`` token removed: the label of a baseline group.

    Two runs belong to one group iff their configs agree on every field outside
    ``BASELINE_IGNORED_FIELDS`` (checked by ``_check_group_configs``, since the label alone
    does not encode every field).
    """
    return _SEED_TOKEN.sub("", cfg.run_id)


def _config_fields(cfg: Config) -> dict:
    return {k: v for k, v in cfg.to_dict().items() if k not in BASELINE_IGNORED_FIELDS}


def _check_group_configs(key: str, group: list[ModelSide]) -> None:
    ref = _config_fields(group[0].cfg)
    for s in group[1:]:
        other = _config_fields(s.cfg)
        differing = sorted(k for k in ref if ref[k] != other.get(k))
        if differing:
            raise ValueError(f"baseline group {key!r}: {group[0].id} and {s.id} differ in config "
                             f"field(s) {differing} -- not a same-configuration seed pair")


def _metric_summary(values: list) -> dict:
    """``summarize`` over the defined values; ``n_undefined`` counts the ``None`` entries."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    return {**summarize(arr), "n_undefined": int(np.isnan(arr).sum())}


def _baseline_group(key: str, group: list[ModelSide], n_control: int, seed: int,
                    out_dir: Path | None) -> dict:
    _check_group_configs(key, group)
    pairs = [headline(compare_sides(a, b, n_control, seed, out_dir, require_same_split=False))
             for a, b in itertools.combinations(group, 2)]
    return {
        "config_key": key, "arch": group[0].cfg.arch, "n_runs": len(group),
        "run_ids": [s.id for s in group], "seeds": [int(s.cfg.seed) for s in group],
        "n_pairs": len(pairs), "pairs": pairs,
        "summary": {m: _metric_summary([pr[m] for pr in pairs]) for m in BASELINE_METRICS},
    }


def baseline(run_dirs, step: int | None = None, n_control: int = DEFAULT_N_CONTROL,
             seed: int = DEFAULT_SEED, out_dir: Path | None = None) -> dict:
    """``compare`` for every unordered pair of runs in ``run_dirs`` that share a
    configuration and differ only by seed (INTERFACES section 8 "same-architecture seed pair").

    Runs are grouped by ``config_key`` (the seedless ``run_id``); runs with the same key but a
    differing config field raise. A group with a single run yields zero pairs (reported, not
    hidden). Seeds of one configuration never share a split, so every pair is compared with
    ``require_same_split=False`` (echoed in ``params``). Every pair's JSON/npz is written as
    by ``compare``; the table is written to ``baseline__<sha256(sorted ids)[:12]>.json``.
    """
    run_dirs = [Path(d) for d in run_dirs]
    if len(run_dirs) < 2:
        raise ValueError("baseline needs at least two run directories")
    sides = [load_side(d, step) for d in run_dirs]
    ids = [s.id for s in sides]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"baseline: duplicate run ids {duplicates} (a run paired with itself)")
    groups: dict[str, list[ModelSide]] = {}
    for s in sides:
        groups.setdefault(config_key(s.cfg), []).append(s)
    payload = {
        "module": MODULE, "module_version": MODULE_VERSION, "kind": "baseline",
        "analysis_git_commit": git_commit(), "created_utc": utcnow(),
        "params": {"run_dirs": run_dirs, "step": step, "n_control": int(n_control),
                   "seed": int(seed), "require_same_split": False,
                   "ignored_config_fields": list(BASELINE_IGNORED_FIELDS)},
        "run_ids": ids,
        "by_config": {key: _baseline_group(key, group, n_control, seed, out_dir)
                      for key, group in sorted(groups.items())},
    }
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:12]
    return _write(payload, None, f"baseline__{digest}", out_dir)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Functional agreement of two models over all p^2 inputs")
    ap.add_argument("run_dirs", nargs="*", type=Path, help="two run directories to compare")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step for every run (default: legacy model_final.pt)")
    ap.add_argument("--step-a", type=int, default=None, help="checkpoint step of run a (overrides --step)")
    ap.add_argument("--step-b", type=int, default=None, help="checkpoint step of run b (overrides --step)")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="compare at every step listed in BOTH runs' checkpoints.json")
    ap.add_argument("--baseline", nargs="+", type=Path, default=None, metavar="RUN_DIR",
                    help="compare every same-configuration (seed-only) pair of these runs")
    ap.add_argument("--allow-different-splits", action="store_true",
                    help="do not raise when the two runs' split hashes differ (recorded either way)")
    ap.add_argument("--n-control", type=int, default=DEFAULT_N_CONTROL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: training/results/function_agreement)")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = _build_parser()
    args = ap.parse_args(argv)
    require = not args.allow_different_splits
    if args.baseline is not None:
        if args.run_dirs or args.all_checkpoints:
            ap.error("--baseline takes its run directories itself and excludes --all-checkpoints")
        res = baseline(args.baseline, args.step, args.n_control, args.seed, args.out_dir)
    elif len(args.run_dirs) != 2:
        ap.error("exactly two run directories are required (or use --baseline)")
    elif args.all_checkpoints:
        payloads = compare_all_checkpoints(args.run_dirs[0], args.run_dirs[1], args.n_control,
                                           args.seed, args.out_dir, require)
        res = {"steps": [pl["step"] for pl in payloads], "pairs": [headline(pl) for pl in payloads],
               "output_json": [pl["output_json"] for pl in payloads]}
    else:
        step_a = args.step_a if args.step_a is not None else args.step
        step_b = args.step_b if args.step_b is not None else args.step
        res = compare(args.run_dirs[0], args.run_dirs[1], step_a, step_b,
                      args.n_control, args.seed, args.out_dir, require)
    print(json.dumps(_jsonable(res), indent=2))


if __name__ == "__main__":
    main()
