"""End-to-end fit of closed-form candidate formulas to the full logit tensor (INTERFACES §7).

The object under test is the centred logit tensor ``L[a, b, c]`` over every input pair and every
class.  A Fourier phase-addition circuit predicts that the logit is (up to a per-(a, b) constant) a
function of the single residue ``n = (a + b - c) mod p`` built from a few frequencies::

    sparse_sinusoid     L ~ β0 + Σ_{k∈K} βc_k cos(w_k n) + βs_k sin(w_k n)          w_k = 2πk/p
    odd_harmonics       same columns over the aliased odd-harmonic family F of K (j ∈ {1,3,5,7})
    ideal_square        L ~ β0 + Σ_{k∈K} β_k sq(w_k n + φ_k),   sq(x) = +1 if cos x ≥ 0 else −1
    control_top_m       sparse_sinusoid on the |F| frequencies with the most logit power in the
                        cos/sin(w_k(a+b)) directions (cardinality matched to odd_harmonics)
    control_random_m    sparse_sinusoid on n_control random frequency sets of size |F|
    control_difference  sparse_sinusoid in the wrong symmetry m = (a − b − c) mod p
    full_sum_basis      every frequency 1..(p−1)/2 in n — the ceiling of any function of n

Every candidate is a function of one residue, so its p³-row design matrix is ``X[row, j] =
columns[residue(row), j]`` and — because each residue value occupies exactly p² of the p³ cells —
its normal equations are ``XᵀX = p²·CᵀC`` and ``Xᵀy = CᵀS`` with ``C`` the p-row column table and
``S[n]`` the sum of the logits over the cells with residue n.  ``normal_equations`` computes exactly
that (no p³ × n_feat matrix is ever materialised for p = 113); ``design_matrix`` materialises the
explicit matrix so the test can verify the identity.

Everything reported is a measurement: R² on all / train / test cells, per-class R² quantiles,
residual RMS, AIC = N ln(RSS/N) + 2·n_params, argmax accuracy of the fitted logits, and for the
random control its distribution plus z-scores.  No key here states a conclusion.

Usage (from training/):
    python -m grokverse.analysis.logit_formula_fit <run_dir> [--step N] [--key-freqs 18,15,...]
                                                  [--key-rule embedding_top8] [--seed 0]
                                                  [--n-control 50] [--all-checkpoints]
"""
from __future__ import annotations

import argparse
import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import (center_logits, envelope, grid_logits, load_model_at, split_masks,
                     summarize, write_result)
from .fourier import (MAX_ODD_HARMONIC, dominant_frequencies, fourier_basis,
                      harmonic_family)
from .mask_protocols import SumDirectionsOnly
from .progress_measures import fwd2d

MODULE = "logit_formula_fit"
MODULE_VERSION = "1.0"

#: The two residues a §7 formula may be a function of.
INDEX_VARIABLES: tuple[str, ...] = ("a+b-c", "a-b-c")
#: Phase grid density for ``ideal_square``: 4p grid points per 2π (INTERFACES §7).
PHASE_GRID_POINTS_PER_P = 4
#: Candidate formula ids, in the order of the §7 table (the random control is reported separately).
FORMULA_IDS: tuple[str, ...] = ("sparse_sinusoid", "odd_harmonics", "ideal_square",
                                "control_top_m", "control_difference", "full_sum_basis")
#: Scalar metrics compared against the random-control distribution.
CONTROL_METRICS: tuple[str, ...] = ("r2_all", "r2_train", "r2_test",
                                    "argmax_accuracy_train", "argmax_accuracy_test")
AIC_FORMULA = "N * ln(RSS / N) + 2 * n_params, N = p^3 cells, RSS over all cells"
#: Key-frequency rules of INTERFACES §4.  ``nanda`` is the pre-registered primary rule
#: (PREREG_BRIEF addendum 2026-09-03); only ``embedding_top8`` is computed in this module, every
#: other rule is delegated to ``analysis.key_frequencies.select`` (see ``resolve_key_set``).
KEY_RULES: tuple[str, ...] = ("nanda", "neuron_clusters", "embedding_threshold",
                              "logit_sum_directions", "embedding_top8")
PRIMARY_KEY_RULE = "nanda"


# --------------------------------------------------------------------------- #
# feature sets                                                                 #
# --------------------------------------------------------------------------- #
def validate_frequencies(p: int, freqs, allow_empty: bool = True) -> tuple[int, ...]:
    """Frequencies must be distinct integers in 1..(p−1)/2 for odd p ≥ 3."""
    if int(p) < 3 or int(p) % 2 == 0:
        raise ValueError(f"p must be an odd integer >= 3 (got {p}); the cos/sin basis needs odd p")
    half = (int(p) - 1) // 2
    ks = tuple(int(k) for k in freqs)
    if not ks and not allow_empty:
        raise ValueError("frequency set is empty")
    if len(set(ks)) != len(ks):
        raise ValueError(f"duplicate frequencies in {ks}")
    bad = [k for k in ks if not 1 <= k <= half]
    if bad:
        raise ValueError(f"frequencies {bad} outside 1..{half} for p={p}")
    return ks


@dataclass(frozen=True)
class FeatureSet:
    """Columns of one candidate formula tabulated over the residue ``index_variable mod p``.

    ``columns[n, j]`` is feature j evaluated at residue n; the p³-row design matrix is
    ``columns[residue(row)]``.  ``n_params`` is what AIC charges: the number of columns for linear
    formulas, plus one searched phase per frequency for the square-wave formula.
    """

    p: int
    index_variable: str
    columns: np.ndarray
    names: tuple[str, ...]
    frequencies: tuple[int, ...]
    n_params: int
    phases_rad: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        validate_frequencies(self.p, self.frequencies)
        if self.index_variable not in INDEX_VARIABLES:
            raise ValueError(f"index_variable {self.index_variable!r} not in {INDEX_VARIABLES}")
        cols = np.asarray(self.columns, dtype=np.float64)
        if cols.ndim != 2 or cols.shape[0] != self.p:
            raise ValueError(f"columns must be [p={self.p}, n_feat], got {cols.shape}")
        if len(self.names) != cols.shape[1]:
            raise ValueError(f"{len(self.names)} names for {cols.shape[1]} columns")
        if int(self.n_params) < cols.shape[1]:
            raise ValueError("n_params cannot be smaller than the number of columns")
        object.__setattr__(self, "columns", cols)

    @property
    def n_features(self) -> int:
        return int(self.columns.shape[1])


def _trig_columns(K: tuple[int, ...], p: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """[1, cos(w_k n), sin(w_k n), ...] for n = 0..p−1 (unnormalised, so β are amplitudes)."""
    n = np.arange(p)
    cols, names = [np.ones(p)], ["const"]
    for k in K:
        w = 2 * np.pi * k * n / p
        cols += [np.cos(w), np.sin(w)]
        names += [f"cos{k}", f"sin{k}"]
    return np.stack(cols, axis=1), tuple(names)


def sum_features(K, p: int) -> FeatureSet:
    """{1} ∪ {cos(w_k n), sin(w_k n) : k ∈ K} over n = (a + b − c) mod p; 2|K| + 1 parameters."""
    ks = validate_frequencies(p, K)
    cols, names = _trig_columns(ks, p)
    return FeatureSet(p, "a+b-c", cols, names, ks, 2 * len(ks) + 1)


def diff_features(K, p: int) -> FeatureSet:
    """The same columns as ``sum_features`` over m = (a − b − c) mod p (wrong symmetry control)."""
    ks = validate_frequencies(p, K)
    cols, names = _trig_columns(ks, p)
    return FeatureSet(p, "a-b-c", cols, names, ks, 2 * len(ks) + 1)


def square_wave(k: int, p: int, phi: float) -> np.ndarray:
    """sq(w_k n + φ) := +1 where cos(2πk n/p + φ) ≥ 0, else −1, for n = 0..p−1."""
    x = 2 * np.pi * int(k) * np.arange(int(p)) / int(p) + float(phi)
    return np.where(np.cos(x) >= 0.0, 1.0, -1.0)


def phase_grid(p: int) -> np.ndarray:
    """The 4p search phases φ_g = 2π(g + ½)/(4p), g = 0..4p−1.

    A sampled square wave changes only when some n crosses cos(2πkn/p + φ) = 0, i.e. at
    φ = π/2 + mπ − 2πkn/p — always an integer multiple of 2π/(4p).  Placing the grid half a step off
    those events visits every distinct sign pattern exactly once and never lands on a zero crossing
    (whose sign would depend on floating-point rounding).
    """
    g = PHASE_GRID_POINTS_PER_P * int(p)
    return 2 * np.pi * (np.arange(g) + 0.5) / g


def square_features(K, p: int, phases) -> FeatureSet:
    """{1} ∪ {sq(w_k n + φ_k) : k ∈ K}; AIC charges 2|K| + 1 (|K| amplitudes, |K| phases, const)."""
    ks = validate_frequencies(p, K)
    phis = tuple(float(x) for x in phases)
    if len(phis) != len(ks):
        raise ValueError(f"{len(phis)} phases for {len(ks)} frequencies")
    cols = [np.ones(p)] + [square_wave(k, p, phi) for k, phi in zip(ks, phis)]
    names = ("const",) + tuple(f"sq{k}" for k in ks)
    return FeatureSet(p, "a+b-c", np.stack(cols, axis=1), names, ks, 2 * len(ks) + 1, phis)


# --------------------------------------------------------------------------- #
# design matrix over the p^3 cells                                             #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def residue_index(p: int, index_variable: str) -> np.ndarray:
    """Read-only ``[p, p, p]`` int array of the residue of every cell (a outer, b, c inner)."""
    if index_variable not in INDEX_VARIABLES:
        raise ValueError(f"index_variable {index_variable!r} not in {INDEX_VARIABLES}")
    a = np.arange(p)[:, None, None]
    b = np.arange(p)[None, :, None]
    c = np.arange(p)[None, None, :]
    idx = ((a + b - c) if index_variable == "a+b-c" else (a - b - c)) % p
    idx.setflags(write=False)
    return idx


def design_matrix(features: FeatureSet) -> np.ndarray:
    """Explicit ``[p³, n_feat]`` float32 design matrix, row order = ``L.reshape(-1)``.

    Only for verification and small p — ``normal_equations`` never needs it.
    """
    idx = residue_index(features.p, features.index_variable).reshape(-1)
    return features.columns.astype(np.float32)[idx]


def normal_equations(L: np.ndarray, features: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
    """``XᵀX`` and ``Xᵀy`` of the p³-row design matrix, computed by residue aggregation.

    Each residue value occurs in exactly p² cells (for every (a, b) exactly one c hits it), so
    ``XᵀX = p²·CᵀC`` and ``Xᵀy = CᵀS`` with ``S[n] = Σ_{cells with residue n} L``.  Exact — verified
    against ``design_matrix`` in tests/test_logit_formula_fit.py.
    """
    p = features.p
    idx = residue_index(p, features.index_variable).reshape(-1)
    S = np.bincount(idx, weights=np.asarray(L, dtype=np.float64).reshape(-1), minlength=p)
    C = features.columns
    return float(p * p) * (C.T @ C), C.T @ S


# --------------------------------------------------------------------------- #
# one fit                                                                      #
# --------------------------------------------------------------------------- #
def _validate_inputs(L: np.ndarray, p: int, train_mask: np.ndarray,
                     test_mask: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    if L.shape != (p, p, p):
        raise ValueError(f"logits must be [p, p, p] = {(p, p, p)}, got {L.shape}")
    if not np.isfinite(L).all():
        raise ValueError("logits contain non-finite values")
    for name, m in (("train_mask", train_mask), ("test_mask", test_mask)):
        if not (isinstance(m, np.ndarray) and m.dtype == bool and m.shape == (p, p)):
            raise ValueError(f"{name} must be a bool array of shape {(p, p)}")
    if (train_mask & test_mask).any():
        raise ValueError("train_mask and test_mask overlap")
    return L


def _subset_stats(Lc: np.ndarray, resid: np.ndarray, cells: np.ndarray | None) -> dict:
    """RSS, TSS (about the subset mean), R² = 1 − RSS/TSS and RMS over a set of (a, b) cells."""
    y = Lc if cells is None else Lc[cells]
    r = resid if cells is None else resid[cells]
    rss = float((r ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    n = int(y.size)
    return {"r2": (1.0 - rss / tss) if tss > 0 else None, "rss": rss, "tss": tss,
            "residual_rms": float(np.sqrt(rss / n)) if n else None, "n_values": n}


def per_class_r2(Lc: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """``r2[c] = 1 − Σ_{a,b} resid[a,b,c]² / Σ_{a,b} (Lc[a,b,c] − mean_{a,b} Lc[·,·,c])²`` per class;
    NaN where the class column is constant (TSS = 0)."""
    tss = ((Lc - Lc.mean(axis=(0, 1), keepdims=True)) ** 2).sum(axis=(0, 1))
    rss = (resid ** 2).sum(axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tss > 0, 1.0 - rss / tss, np.nan)


def argmax_accuracy(fitted: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> dict:
    """Accuracy of ``argmax_c fitted[a, b, c]`` against (a + b) mod p; ties resolved by numpy argmax
    (lowest class) and counted separately so a tie-inflated number cannot pass unnoticed."""
    p = fitted.shape[0]
    target = (np.arange(p)[:, None] + np.arange(p)[None, :]) % p
    correct = fitted.argmax(axis=-1) == target
    tied = (fitted == fitted.max(axis=-1, keepdims=True)).sum(axis=-1) > 1
    return {
        "argmax_accuracy_all": float(correct.mean()),
        "argmax_accuracy_train": float(correct[train_mask].mean()) if train_mask.any() else None,
        "argmax_accuracy_test": float(correct[test_mask].mean()) if test_mask.any() else None,
        "n_cells_argmax_tied": int(tied.sum()),
    }


def _aic(rss: float, n_cells: int, n_params: int) -> dict:
    if rss <= 0.0:
        return {"aic": None, "aic_undefined_reason": "rss == 0 (ln undefined)"}
    return {"aic": float(n_cells * np.log(rss / n_cells) + 2 * n_params)}


def _coefficient_table(features: FeatureSet, beta: np.ndarray) -> dict:
    """Per-frequency amplitude/phase from the coefficients.

    Trig columns: ``βc cos(w n) + βs sin(w n) = A cos(w n − φ)`` with ``A = sqrt(βc² + βs²)`` and
    ``φ = atan2(βs, βc)`` (INTERFACES §0 phase convention).  Square columns: the signed amplitude.
    """
    names = list(features.names)
    rows = []
    for k in features.frequencies:
        if f"cos{k}" in names:
            bc, bs = float(beta[names.index(f"cos{k}")]), float(beta[names.index(f"sin{k}")])
            rows.append({"k": int(k), "amplitude": float(np.hypot(bc, bs)),
                         "phase_rad": float(np.arctan2(bs, bc))})
        else:
            rows.append({"k": int(k), "amplitude": float(beta[names.index(f"sq{k}")])})
    return {"const": float(beta[names.index("const")]), "per_frequency": rows}


def fit_formula(L: np.ndarray, features: FeatureSet, train_mask: np.ndarray,
                test_mask: np.ndarray) -> dict:
    """Linear least squares of one formula to the centred logits over all p³ cells.

    Solves ``(XᵀX) β = Xᵀy`` with ``np.linalg.lstsq`` on the normal equations (rank reported), then
    measures: ``r2_S = 1 − Σ_S (L − L̂)² / Σ_S (L − mean_S L)²`` for S = all / train / test cells
    (every class), per-class R² quantiles, residual RMS, ``AIC = N ln(RSS/N) + 2 n_params`` (N = p³),
    and argmax accuracy of L̂.  L is centred per (a, b) over classes first (idempotent).
    Returns ``{"coefficients", "fitted", "residual_rms_map", "report"}``.
    """
    p = features.p
    Lc = center_logits(_validate_inputs(L, p, train_mask, test_mask))
    if not (Lc != 0).any():
        raise ValueError("centred logits are identically zero — nothing to fit")
    XtX, Xty = normal_equations(Lc, features)
    beta, _, rank, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
    fitted = (features.columns @ beta)[residue_index(p, features.index_variable)]
    resid = Lc - fitted
    stats = {s: _subset_stats(Lc, resid, m) for s, m in
             (("all", None), ("train", train_mask), ("test", test_mask))}
    report = {
        "index_variable": features.index_variable, "frequencies": list(features.frequencies),
        "n_features": features.n_features, "n_params": int(features.n_params),
        "n_cells": int(p ** 3), "n_train_cells": int(train_mask.sum()),
        "n_test_cells": int(test_mask.sum()), "normal_equations_rank": int(rank),
        "centered_per_ab": True,
    }
    for s, st in stats.items():
        report[f"r2_{s}"] = st["r2"]
        report[f"residual_rms_{s}"] = st["residual_rms"]
        report[f"rss_{s}"] = st["rss"]
        report[f"tss_{s}"] = st["tss"]
    class_r2 = per_class_r2(Lc, resid)
    report["per_class_r2"] = summarize(class_r2)
    report["n_classes_r2_undefined"] = int((~np.isfinite(class_r2)).sum())
    report.update(_aic(stats["all"]["rss"], p ** 3, features.n_params))
    report.update(argmax_accuracy(fitted, train_mask, test_mask))
    report["coefficient_table"] = _coefficient_table(features, beta)
    if features.phases_rad:
        report["phases_rad"] = list(features.phases_rad)
    return {"coefficients": beta, "fitted": fitted, "per_class_r2": class_r2,
            "residual_rms_map": np.sqrt((resid ** 2).mean(axis=-1)), "report": report}


# --------------------------------------------------------------------------- #
# ideal_square: phase search                                                   #
# --------------------------------------------------------------------------- #
#: Upper bound on backfitting passes of the square-phase refinement (echoed in the output).
DEFAULT_PHASE_REFINE_PASSES = 10


def phase_correlation_row(k: int, resid: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Pearson correlation, over all p³ cells, of ``sq(w_k(a+b−c) + φ)`` with ``resid`` for every φ
    of ``grid``.

    The square wave is a function of the residue n, so with ``R[n]`` the mean residual of the p²
    cells with residue n, ``x̄``, ``ȳ`` the grand means and ``Var_cells(y)`` the residual variance
    over all cells, ``corr(φ) = mean_n[(x_n − x̄)(R[n] − ȳ)] / sqrt(mean_n (x_n − x̄)² ·
    Var_cells(y))`` — identical to the correlation over the p³ rows.
    """
    p = int(resid.shape[0])
    idx = residue_index(p, "a+b-c").reshape(-1)
    R = np.bincount(idx, weights=resid.reshape(-1), minlength=p) / float(p * p)
    y_mean = float(resid.mean())
    var_y = float(((resid - y_mean) ** 2).mean())
    if var_y <= 0:
        raise ValueError("residual has zero variance — no phase to search")
    n = np.arange(p)
    X = np.where(np.cos(2 * np.pi * int(k) * n[None, :] / p + grid[:, None]) >= 0.0, 1.0, -1.0)
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ (R - y_mean) / p) / np.sqrt((Xc ** 2).mean(axis=1) * var_y)


def square_partial_fitted(Lc: np.ndarray, ks, phases, train_mask: np.ndarray,
                          test_mask: np.ndarray) -> np.ndarray:
    """Fitted tensor of ``const + Σ_i α_i sq(w_{ks[i]}(a+b−c) + phases[i])`` (amplitudes by LSQ).

    ``ks`` holds only the frequencies whose phase is already fixed; with no fitted phase yet
    (``ks == []``) the model is the constant alone, which is what the first grid search of every
    frequency is compared against.
    """
    p = int(Lc.shape[0])
    ks, phases = list(ks), list(phases)
    if len(ks) != len(phases):
        raise ValueError(f"{len(phases)} phases for {len(ks)} fitted frequencies")
    fs = square_features(ks, p, phases) if ks else sum_features((), p)
    return fit_formula(Lc, fs, train_mask, test_mask)["fitted"]


def _residual_without(Lc, ks, grid_index, skip: int, grid, train_mask, test_mask) -> np.ndarray:
    """Residual of ``Lc`` after const + every square wave of ``ks`` except index ``skip``."""
    keep = [i for i in range(len(ks)) if i != skip]
    fitted = square_partial_fitted(Lc, [ks[i] for i in keep],
                                   [float(grid[grid_index[i]]) for i in keep],
                                   train_mask, test_mask)
    return Lc - fitted


def search_square_phases(L: np.ndarray, K, train_mask: np.ndarray, test_mask: np.ndarray,
                         n_refine_passes: int = DEFAULT_PHASE_REFINE_PASSES) -> dict:
    """Grid phases of the ``ideal_square`` formula, one per k ∈ K, over ``phase_grid(p)``.

    Stage 1 (INTERFACES §7): each k independently, argmax over the grid of the correlation of
    ``sq(w_k(a+b−c) + φ)`` with the residual of the constant-only model (no phase fitted yet)
    → ``phases_initial_rad``.
    Stage 2 (backfitting): sweep k ∈ K, re-search φ_k against the residual of const + the other
    |K|−1 square waves at their current phases (amplitudes by LSQ), until a full sweep changes no
    phase or ``n_refine_passes`` sweeps ran.  Square waves of different k are not orthogonal on Z_p
    (their aliased harmonics collide), so the stage-1 argmax can sit one grid step off the joint
    optimum.  Both stages are reported; ``refinement_converged`` is True only when a full sweep
    changed nothing (never with ``n_refine_passes = 0``).
    """
    p = int(np.asarray(L).shape[0])
    ks = validate_frequencies(p, K, allow_empty=False)
    if int(n_refine_passes) < 0:
        raise ValueError("n_refine_passes must be >= 0")
    Lc = center_logits(np.asarray(L, dtype=np.float64))
    grid = phase_grid(p)
    resid0 = Lc - square_partial_fitted(Lc, [], [], train_mask, test_mask)
    corr0 = np.stack([phase_correlation_row(k, resid0, grid) for k in ks])
    best = [int(g) for g in corr0.argmax(axis=1)]
    initial = list(best)
    passes_used, converged = 0, False
    for _ in range(int(n_refine_passes)):
        passes_used += 1
        before = list(best)
        for i, k in enumerate(ks):
            resid = _residual_without(Lc, ks, best, i, grid, train_mask, test_mask)
            best[i] = int(phase_correlation_row(k, resid, grid).argmax())
        if best == before:
            converged = True
            break
    final_corr = [float(phase_correlation_row(
        k, _residual_without(Lc, ks, best, i, grid, train_mask, test_mask), grid)[best[i]])
        for i, k in enumerate(ks)]
    return {"frequencies": ks, "phases_rad": tuple(float(grid[g]) for g in best),
            "grid_index": tuple(best), "correlation_max": tuple(final_corr),
            "phases_initial_rad": tuple(float(grid[g]) for g in initial),
            "grid_index_initial": tuple(initial),
            "correlation_max_initial": tuple(float(corr0[i, g]) for i, g in enumerate(initial)),
            "n_phases_changed_by_refinement": int(sum(a != b for a, b in zip(best, initial))),
            "n_refine_passes": int(n_refine_passes), "n_refine_passes_used": int(passes_used),
            "refinement_converged": bool(converged), "n_grid": int(grid.size),
            "correlation_grid": corr0}


# --------------------------------------------------------------------------- #
# logit power in the cos/sin(w_k(a+b)) directions                              #
# --------------------------------------------------------------------------- #
def sum_direction_power(L: np.ndarray) -> dict:
    """Power of the centred logits inside ``span{cos(w_k(a+b)), sin(w_k(a+b))}`` per k, summed over
    classes — ``mask_protocols.SumDirectionsOnly`` with the single key k, constant removed — as a
    share of the total non-constant 2D-Fourier power of the tensor."""
    p = int(np.asarray(L).shape[0])
    validate_frequencies(p, ())
    Fb, _ = fourier_basis(p)
    Lhat = fwd2d(center_logits(np.asarray(L, dtype=np.float64)), Fb)
    total = float((Lhat ** 2).sum() - (Lhat[0, 0] ** 2).sum())
    if total <= 0:
        raise ValueError("logits have no non-constant power over (a, b)")
    half = (p - 1) // 2
    power = np.zeros(half)
    for k in range(1, half + 1):
        proj = SumDirectionsOnly(p, [k]).restrict(Lhat)
        proj[0, 0] = 0.0
        power[k - 1] = float((proj ** 2).sum())
    return {"freqs": list(range(1, half + 1)), "power": power, "share": power / total,
            "total_nonconstant_power": total,
            "definition": "sum over classes of ||P_k Lhat||^2, P_k = SumDirectionsOnly([k])"}


# --------------------------------------------------------------------------- #
# the §7 table                                                                 #
# --------------------------------------------------------------------------- #
def candidate_feature_sets(L: np.ndarray, K, train_mask: np.ndarray, test_mask: np.ndarray,
                           sdp: dict) -> tuple[dict[str, FeatureSet], dict]:
    """Build the six candidate ``FeatureSet``s of the §7 table for key set K.  Returns them with a
    provenance dict (family, top-m frequencies, square phases)."""
    p = int(np.asarray(L).shape[0])
    ks = validate_frequencies(p, K, allow_empty=False)
    half = (p - 1) // 2
    family = harmonic_family(ks, p, MAX_ODD_HARMONIC)
    m = len(family)
    order = np.argsort(-np.asarray(sdp["power"]), kind="stable")
    top_m = sorted(int(sdp["freqs"][i]) for i in order[:m])
    phases = search_square_phases(L, ks, train_mask, test_mask)
    sets = {
        "sparse_sinusoid": sum_features(ks, p),
        "odd_harmonics": sum_features(family, p),
        "ideal_square": square_features(ks, p, phases["phases_rad"]),
        "control_top_m": sum_features(top_m, p),
        "control_difference": diff_features(ks, p),
        "full_sum_basis": sum_features(range(1, half + 1), p),
    }
    info = {"key_frequencies": list(ks), "harmonic_family": list(family), "family_size": m,
            "max_odd_harmonic": MAX_ODD_HARMONIC, "control_top_m_frequencies": top_m,
            "ideal_square_phase_search": {k: v for k, v in phases.items()
                                         if k != "correlation_grid"},
            "ideal_square_phase_correlation_grid": phases["correlation_grid"]}
    return sets, info


def random_frequency_control(L: np.ndarray, m: int, train_mask: np.ndarray, test_mask: np.ndarray,
                             n_control: int, seed: int) -> tuple[dict, dict]:
    """``sparse_sinusoid`` on ``n_control`` seeded random frequency sets of size m
    (``np.random.default_rng(seed)``, sampled without replacement from 1..(p−1)/2)."""
    p = int(np.asarray(L).shape[0])
    half = (p - 1) // 2
    if not 1 <= int(m) <= half:
        raise ValueError(f"control set size {m} outside 1..{half} for p={p}")
    if int(n_control) < 2:
        raise ValueError("n_control must be >= 2 to report a standard deviation")
    rng = np.random.default_rng(int(seed))
    sets = np.stack([np.sort(rng.choice(half, size=int(m), replace=False) + 1)
                     for _ in range(int(n_control))])
    values = {metric: [] for metric in CONTROL_METRICS}
    for freqs in sets:
        rep = fit_formula(L, sum_features(freqs, p), train_mask, test_mask)["report"]
        for metric in CONTROL_METRICS:
            values[metric].append(rep[metric])
    summary = {"n_control": int(n_control), "seed": int(seed), "set_size": int(m),
               "index_variable": "a+b-c", "n_params": 2 * int(m) + 1}
    arrays = {"control_random_frequency_sets": sets}
    for metric, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        arrays[f"control_random_{metric}"] = arr
        summary[metric] = {"control_mean": float(arr.mean()), "control_std": float(arr.std(ddof=1)),
                           "control_q05": float(np.quantile(arr, 0.05)),
                           "control_q95": float(np.quantile(arr, 0.95)),
                           "control_min": float(arr.min()), "control_max": float(arr.max())}
    return summary, arrays


def z_scores(reports: dict[str, dict], control: dict) -> dict:
    """``z = (observed − control_mean) / control_std`` per formula and metric; ``None`` (flagged)
    when the control has zero spread."""
    out = {}
    for fid, rep in reports.items():
        out[fid] = {}
        for metric in CONTROL_METRICS:
            c, obs = control[metric], rep[metric]
            if obs is None or c["control_std"] <= 0:
                out[fid][metric] = None
                out[fid][f"{metric}_z_undefined"] = ("observed undefined" if obs is None
                                                     else "control_std == 0")
            else:
                out[fid][metric] = float((obs - c["control_mean"]) / c["control_std"])
    return out


def aic_comparison(reports: dict[str, dict]) -> dict:
    """AIC ranking over the formulas with a finite AIC; ``delta_aic`` = AIC − min AIC."""
    finite = {fid: rep["aic"] for fid, rep in reports.items() if rep.get("aic") is not None}
    if not finite:
        return {"ranking": [], "delta_aic": {}, "formulas_without_aic": list(reports)}
    best = min(finite.values())
    return {"ranking": sorted(finite, key=finite.get),
            "delta_aic": {fid: float(v - best) for fid, v in finite.items()},
            "formulas_without_aic": [fid for fid in reports if fid not in finite]}


def fit_all_formulas(L: np.ndarray, K, train_mask: np.ndarray, test_mask: np.ndarray,
                     seed: int = 0, n_control: int = 50) -> tuple[dict, dict]:
    """Fit every §7 formula plus the random control.  Returns ``(results, arrays)`` where ``arrays``
    holds the coefficients and per-(a, b) residual RMS map of every formula and the control draws."""
    p = int(np.asarray(L).shape[0])
    sdp = sum_direction_power(L)
    sets, provenance = candidate_feature_sets(L, K, train_mask, test_mask, sdp)
    grid_key = "ideal_square_phase_correlation_grid"
    info = {k: v for k, v in provenance.items() if k != grid_key}
    reports = {}
    arrays = {"sum_direction_power": sdp["power"], "sum_direction_share": sdp["share"],
              grid_key: provenance[grid_key]}
    for fid in FORMULA_IDS:
        fit = fit_formula(L, sets[fid], train_mask, test_mask)
        reports[fid] = fit["report"]
        arrays[f"coefficients_{fid}"] = fit["coefficients"]
        arrays[f"per_class_r2_{fid}"] = fit["per_class_r2"]
        arrays[f"residual_rms_map_{fid}"] = fit["residual_rms_map"].astype(np.float32)
    control, control_arrays = random_frequency_control(L, info["family_size"], train_mask,
                                                       test_mask, n_control, seed)
    arrays.update(control_arrays)
    results = {
        **info, "p": p, "n_cells": int(p ** 3), "fit_cells": "all_p3",
        "aic_formula": AIC_FORMULA, "formulas": reports, "control_random_m": control,
        "z_vs_control_random_m": z_scores(reports, control), "aic_comparison": aic_comparison(reports),
        "logit_sum_direction_power": {"freqs": sdp["freqs"], "power": sdp["power"].tolist(),
                                      "share": sdp["share"].tolist(),
                                      "total_nonconstant_power": sdp["total_nonconstant_power"],
                                      "definition": sdp["definition"]},
    }
    return results, arrays


# --------------------------------------------------------------------------- #
# key set                                                                      #
# --------------------------------------------------------------------------- #
def resolve_key_set(run_dir, step, key_freqs, key_rule, state: dict, p: int) -> tuple[list[int], dict]:
    """Exactly one of ``key_freqs`` (explicit list) or ``key_rule`` (a name in ``KEY_RULES``) is given.

    ``embedding_top8`` is implemented here (``fourier.dominant_frequencies`` on ``W_E[:p]``, its
    threshold / cap echoed); every other §4 rule — including the primary ``nanda`` — is delegated
    to ``analysis.key_frequencies.select`` and raises ``NotImplementedError`` naming that module if
    it is absent.  A rule name outside ``KEY_RULES`` is a ``ValueError``.
    """
    if (key_freqs is None) == (key_rule is None):
        raise ValueError("give exactly one of key_freqs (explicit list) or key_rule")
    if key_freqs is not None:
        ks = validate_frequencies(p, key_freqs, allow_empty=False)
        return list(ks), {"key_rule": "explicit", "key_frequencies": list(ks)}
    if key_rule not in KEY_RULES:
        raise ValueError(f"unknown key rule {key_rule!r}; INTERFACES §4 rules: {list(KEY_RULES)}")
    if key_rule == "embedding_top8":
        if "W_E" not in state:
            raise ValueError("state has no W_E — embedding_top8 is undefined for this model")
        dom = dominant_frequencies(np.asarray(state["W_E"], dtype=np.float64)[:p], p)
        ks = validate_frequencies(p, dom["dominant"], allow_empty=False)
        return list(ks), {"key_rule": key_rule, "key_frequencies": list(ks),
                          "threshold": dom["threshold"], "max_k": dom["max_k"], "n_keep": dom["n_keep"],
                          "n_freqs_for_threshold": dom["n_freqs_for_threshold"],
                          "cap_binding": dom["cap_binding"],
                          "dominant_fraction": dom["dominant_fraction"]}
    try:
        from . import key_frequencies  # noqa: WPS433 — optional module (INTERFACES §4)
    except ImportError as exc:
        raise NotImplementedError(
            f"key rule {key_rule!r} needs grokverse.analysis.key_frequencies (INTERFACES §4), "
            f"which is not available ({exc}); pass --key-freqs or --key-rule embedding_top8") from exc
    sel = key_frequencies.select(run_dir, step, key_rule)
    ks = validate_frequencies(p, sel["key_frequencies"], allow_empty=False)
    info = {k: v for k, v in sel.items() if not isinstance(v, np.ndarray)}
    return list(ks), {**info, "key_rule": key_rule, "key_frequencies": list(ks)}


# --------------------------------------------------------------------------- #
# run-level driver                                                             #
# --------------------------------------------------------------------------- #
def analyse(run_dir, step: int | None = None, key_freqs=None, key_rule: str | None = None,
            seed: int = 0, n_control: int = 50) -> dict:
    """Fit the §7 table to one checkpoint and write ``analysis/logit_formula_fit/<tag>.json`` + npz."""
    cfg, model, state, meta = load_model_at(run_dir, step)
    if cfg.task != "add":
        raise ValueError(f"{meta['run_id']}: task {cfg.task!r} — the (a+b−c) formulas are defined "
                         "for modular addition only")
    p = cfg.p
    L = center_logits(grid_logits(model, cfg))
    train_mask, test_mask = split_masks(cfg)
    K, key_info = resolve_key_set(run_dir, step, key_freqs, key_rule, state, p)
    params = {"step": meta["step"], "key_frequencies": K, "key_rule": key_info["key_rule"],
              "primary_key_rule_interfaces_s4": PRIMARY_KEY_RULE,
              "key_selection": key_info, "seed": int(seed), "n_control": int(n_control),
              "max_odd_harmonic": MAX_ODD_HARMONIC,
              "phase_grid_points_per_p": PHASE_GRID_POINTS_PER_P,
              "phase_grid_points": PHASE_GRID_POINTS_PER_P * p,
              "phase_refine_passes_max": DEFAULT_PHASE_REFINE_PASSES, "fit_cells": "all_p3",
              "centering": "per_(a,b)_mean_over_classes", "aic_formula": AIC_FORMULA}
    results, arrays = fit_all_formulas(L, K, train_mask, test_mask, seed, n_control)
    payload = envelope(MODULE, MODULE_VERSION, meta, params)
    payload["results"] = results
    path = write_result(run_dir, MODULE, meta["tag"], payload, arrays)
    return {**payload, "output_path": str(path)}


def summary_lines(payload: dict) -> list[str]:
    """Compact per-formula table for the CLI."""
    res = payload["results"]
    lines = [f"{payload['run_id']} step={payload['step']} K={res['key_frequencies']} "
             f"family={res['harmonic_family']} top_m={res['control_top_m_frequencies']}"]
    z = res["z_vs_control_random_m"]

    def fmt(x, digits: int = 4) -> str:
        return "undefined" if x is None else f"{x:.{digits}f}"

    for fid, rep in res["formulas"].items():
        lines.append(f"  {fid:<20s} n_params={rep['n_params']:>4d} r2_all={fmt(rep['r2_all'])} "
                     f"r2_test={fmt(rep['r2_test'])} acc_test={fmt(rep['argmax_accuracy_test'])} "
                     f"aic={fmt(rep['aic'], 1)} z(r2_test)={fmt(z[fid]['r2_test'], 2)}")
    c = res["control_random_m"]
    lines.append(f"  control_random_m     n={c['n_control']} size={c['set_size']} "
                 f"r2_test mean={fmt(c['r2_test']['control_mean'])} "
                 f"q95={fmt(c['r2_test']['control_q95'])}")
    lines.append(f"  aic ranking: {res['aic_comparison']['ranking']}")
    return lines


def _parse_key_freqs(text: str | None) -> list[int] | None:
    if text is None:
        return None
    try:
        return [int(t) for t in text.replace(" ", "").split(",") if t]
    except ValueError as exc:
        raise ValueError(f"--key-freqs must be comma-separated integers, got {text!r}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end formula fit to the logit tensor (§7)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: legacy final)")
    ap.add_argument("--key-freqs", type=str, default=None, help="explicit key set, e.g. 18,15,1")
    ap.add_argument("--key-rule", type=str, default=None, choices=KEY_RULES,
                    help=f"INTERFACES §4 rule (primary: {PRIMARY_KEY_RULE}); rules other than "
                         "embedding_top8 need analysis.key_frequencies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-control", type=int, default=50)
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="run on every entry of checkpoints.json (v2 runs)")
    args = ap.parse_args()

    key_freqs = _parse_key_freqs(args.key_freqs)
    if args.all_checkpoints:
        from ..checkpoints import list_checkpoints
        steps = [int(e["step"]) for e in list_checkpoints(args.run_dir)]
    else:
        steps = [args.step]
    for step in steps:
        payload = analyse(args.run_dir, step, key_freqs, args.key_rule, args.seed, args.n_control)
        print("\n".join(summary_lines(payload)))
        print(json.dumps({"output": payload["output_path"], "params": payload["params"]}, indent=2))
        print(f"[saved] {payload['output_path']}", flush=True)


if __name__ == "__main__":
    main()
