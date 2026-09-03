"""Waveform model comparison on 1-D periodic curves (docs/dev/INTERFACES.md §3).

A curve ``y[n]``, ``n = 0..p-1``, is fitted at a given fundamental ``k`` (``theta_n =
2*pi*k*n/p``) by four models, all written in the §0 phase convention ``A cos(theta - phi)``
(INTERFACES §3 prints the same models with ``+ phi``; the sign is a relabelling
``phi -> -phi`` and the convention used is echoed in every result as ``phase_convention``):

    sinusoid                y = alpha cos(theta - phi) + beta                              3 params
    square                  y = alpha sign(cos(theta - phi)) + beta                        3 params
    odd_harmonics           y = beta + sum_{j in 1,3,5,7} alpha_j cos(j theta - phi_j)     9 params
    odd_harmonics_1_over_j  y = beta + alpha sum_{j in 1,3,5,7} s_j (1/j) cos(j (theta-phi)) 3 params
                            with s_j = (-1)^((j-1)/2), i.e. +, -, +, - : the truncated Fourier
                            series of the ideal square wave. Its analytic ceiling on a square
                            wave is (8/pi^2)(1 + 1/9 + 1/25 + 1/49) = 0.9496, not 1.

``sinusoid`` and ``odd_harmonics`` are linear least squares (``cos``/``sin`` pairs + const).
``square`` and ``odd_harmonics_1_over_j`` are linear in ``(alpha, beta)`` for a fixed ``phi``;
``phi`` is found by a grid of ``4p`` phases (offset by half a step so that no grid phase puts
a sample exactly on a sign change) followed by golden-section refinement of the residual sum
of squares inside the bracket of one grid step around the best grid phase. For ``square``
the RSS is piecewise constant in ``phi`` (it changes only when a sample crosses a zero of the
cosine), so the refined phase is finally centred in its flat interval — every phase in that
interval predicts the same wave; the centre is the reproducible representative.

Per fit (``N`` fitted positions, ``q`` parameters):
``rss = sum (y - pred)^2``, ``nmse = rss / sum (y - mean y)^2``, ``r2 = 1 - nmse``,
``aic = N ln(max(rss, RSS_FLOOR) / N) + 2 q``, ``aicc = aic + 2 q (q + 1) / (N - q - 1)``.

Sign of ``delta_aic_sinusoid_minus_square``: positive means the square model has the lower
AIC. Nothing here decides which waveform a neuron "is"; the numbers are model-comparison
measurements to be read with their controls (a wrong ``k``, a scrambled curve).

CLI (from ``training/``)::

    python -m grokverse.analysis.wave_fitting <run_dir> [--step N] [--seed S]
        [--curve u_a u_b out] [--cv-folds F] [--all-checkpoints]

fits every hidden neuron's effective curves at their own dominant frequency and writes
``<run_dir>/analysis/wave_fitting/<tag>.json`` (+ ``.npz`` with the per-neuron arrays).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import envelope, load_model_at, summarize as dist_summary, write_result
from .fourier import alias_frequency
from .metrics import harmonic_indices, power_spectrum

MODULE = "wave_fitting"
MODULE_VERSION = "1.0"
MODEL_NAMES: tuple[str, ...] = ("sinusoid", "square", "odd_harmonics", "odd_harmonics_1_over_j")
ODD_HARMONICS: tuple[int, ...] = (1, 3, 5, 7)
N_PARAMS: dict[str, int] = {"sinusoid": 3, "square": 3, "odd_harmonics": 9,
                            "odd_harmonics_1_over_j": 3}
#: Guard for a perfect fit: ``aic`` uses ``max(rss, RSS_FLOOR)`` so ``ln(0)`` never occurs.
RSS_FLOOR = 1e-12
PHASE_CONVENTION = "y[n] = A cos(2*pi*k*n/p - phi), phi in (-pi, pi]"
GOLDEN_TOL = 1e-8          # radians: bracket width at which golden-section stops
GOLDEN_MAX_ITER = 200
CURVE_NAMES: tuple[str, ...] = ("u_a", "u_b", "out")


# --------------------------------------------------------------------------- #
# validation / small helpers                                                   #
# --------------------------------------------------------------------------- #
def _wrap(x) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]``."""
    return np.angle(np.exp(1j * np.asarray(x, dtype=np.float64)))


def _as_columns(Y, name: str = "y") -> np.ndarray:
    """``[p]`` or ``[p, m]`` -> float64 ``[p, m]``, finite, odd ``p >= 3``."""
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"{name} must be [p] or [p, m], got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    p = arr.shape[0]
    if p < 3 or p % 2 == 0:
        raise ValueError(f"{name} must have odd length p >= 3, got {p}")
    return arr


def _validate_k(k, p: int) -> int:
    half = (p - 1) // 2
    if int(k) != k or not 1 <= int(k) <= half:
        raise ValueError(f"k={k!r} must be an integer in 1..{half} for p={p}")
    idx = harmonic_indices(int(k), p, max(ODD_HARMONICS))[[j - 1 for j in ODD_HARMONICS], 0]
    if (idx == 0).any() or len(set(idx.tolist())) != len(ODD_HARMONICS):
        raise ValueError(f"harmonics {ODD_HARMONICS} of k={k} alias onto each other or the "
                         f"constant at p={p} ({idx.tolist()}); harmonic models are ill-defined")
    return int(k)


def _theta(n: np.ndarray, k: int, p: int) -> np.ndarray:
    return 2.0 * np.pi * k * np.asarray(n, dtype=np.float64) / p


def _model_index(name: str) -> int:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choices: {MODEL_NAMES}")
    return MODEL_NAMES.index(name)


# --------------------------------------------------------------------------- #
# the four models: design / basis / predict                                     #
# --------------------------------------------------------------------------- #
def _design(n: np.ndarray, k: int, p: int, model: str) -> np.ndarray:
    """Linear design matrix ``[N, q]`` of ``sinusoid`` (cos, sin, 1) or ``odd_harmonics``."""
    th = _theta(n, k, p)
    if model == "sinusoid":
        cols = [np.cos(th), np.sin(th)]
    elif model == "odd_harmonics":
        cols = []
        for j in ODD_HARMONICS:
            cols += [np.cos(j * th), np.sin(j * th)]
    else:
        raise ValueError(f"{model} is not a linear model")
    cols.append(np.ones_like(th))
    return np.column_stack(cols)


def _basis(model: str, th: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Unit-amplitude wave of the phase models: ``[len(phi), len(th)]``."""
    arg = th[None, :] - np.asarray(phi, dtype=np.float64)[:, None]
    if model == "square":
        return np.where(np.cos(arg) >= 0, 1.0, -1.0)
    if model == "odd_harmonics_1_over_j":
        # BUG FIX 2026-09-03: the alternating sign (-1)^((j-1)/2) was missing, so this basis
        # was NOT the square-wave truncation it is documented to be. Without it the model fit
        # a square wave with r2 = 0.571 -- worse than a plain sinusoid (0.811) -- which is
        # impossible for a correct 1/j truncation. With it the fit reaches 0.9495 against the
        # analytic j<=7 ceiling (8/pi^2)(1 + 1/9 + 1/25 + 1/49) = 0.9496.
        # sign(cos psi) = (4/pi) * sum_{j odd} (-1)^((j-1)/2) cos(j psi) / j.
        return sum((-1.0) ** ((j - 1) // 2) * np.cos(j * arg) / j for j in ODD_HARMONICS)
    raise ValueError(f"{model} is not a phase model")


def _predict(params: dict, n: np.ndarray, k: int, p: int, model: str) -> np.ndarray:
    """Model prediction ``[N, m]`` at positions ``n`` from batched parameters."""
    th = _theta(n, k, p)
    if model == "odd_harmonics":
        pred = np.broadcast_to(params["beta"][None, :], (len(th), len(params["beta"]))).copy()
        for i, j in enumerate(ODD_HARMONICS):
            pred += params["alpha_j"][i][None, :] * np.cos(j * th[:, None] - params["phi_j"][i][None, :])
        return pred
    if model == "sinusoid":
        wave = np.cos(th[:, None] - params["phi"][None, :])
    else:
        wave = _basis(model, th, params["phi"]).T
    return params["alpha"][None, :] * wave + params["beta"][None, :]


# --------------------------------------------------------------------------- #
# linear fits                                                                  #
# --------------------------------------------------------------------------- #
def _fit_linear(Y: np.ndarray, n: np.ndarray, k: int, p: int, model: str) -> dict:
    X = _design(n, k, p, model)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)                # [q, m]
    if model == "sinusoid":
        return {"alpha": np.hypot(coef[0], coef[1]), "phi": np.arctan2(coef[1], coef[0]),
                "beta": coef[2]}
    c, s = coef[0:8:2], coef[1:8:2]                              # [4, m] each
    alpha_j, phi_j = np.hypot(c, s), np.arctan2(s, c)
    js = np.array(ODD_HARMONICS[1:], dtype=np.float64)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(alpha_j[0] > 0, alpha_j[1:] / alpha_j[0], np.nan)
    return {"alpha_j": alpha_j, "phi_j": phi_j, "beta": coef[8],
            "harmonics": np.array(ODD_HARMONICS),
            "amplitude_ratio_to_fundamental": ratio,             # [3, m], j = 3, 5, 7
            "phase_residual_j": _wrap(phi_j[1:] - js * phi_j[0])}   # 0 for an ideal square wave


# --------------------------------------------------------------------------- #
# phase models: grid -> golden section -> (square only) centre in flat interval  #
# --------------------------------------------------------------------------- #
def _affine_rss_grid(B: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """RSS of ``y ~ alpha b + beta`` for every basis row ``b`` of ``B [G, N]`` and every
    column of ``Y [N, m]`` -> ``[G, m]`` (closed form of the 2-parameter least squares)."""
    Bc = B - B.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    bb = (Bc ** 2).sum(axis=1)                                   # [G]
    by = Bc @ Yc                                                 # [G, m]
    yy = (Yc ** 2).sum(axis=0)                                   # [m]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(bb[:, None] > 0, by ** 2 / bb[:, None], 0.0)
    return np.maximum(yy[None, :] - gain, 0.0)


def _affine_coefficients(B: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column: ``alpha, beta`` of ``y ~ alpha b + beta`` with ``B [m, N]`` (row = column)."""
    Bc = B - B.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    bb = (Bc ** 2).sum(axis=1)
    by = (Bc * Yc.T).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(bb > 0, by / bb, 0.0)
    beta = Y.mean(axis=0) - alpha * B.mean(axis=1)
    return alpha, beta


def _affine_rss_columns(B: np.ndarray, Y: np.ndarray) -> np.ndarray:
    alpha, beta = _affine_coefficients(B, Y)
    resid = Y - (alpha[None, :] * B.T + beta[None, :])
    return (resid ** 2).sum(axis=0)


def _golden_section(f, lo: np.ndarray, hi: np.ndarray, tol: float,
                    max_iter: int) -> tuple[np.ndarray, int]:
    """Vectorized golden-section minimization of ``f: [m] -> [m]`` on ``[lo, hi]`` per column."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo.astype(np.float64).copy(), hi.astype(np.float64).copy()
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    n_iter = 0
    while n_iter < max_iter and (b - a).max() > tol:
        left = fc < fd                                   # minimum lies in [a, d]
        b = np.where(left, d, b)
        a = np.where(left, a, c)
        c_new, d_new = b - gr * (b - a), a + gr * (b - a)
        x = np.where(left, c_new, d_new)                 # the one new point per column
        fx = f(x)
        fc, fd = np.where(left, fx, fd), np.where(left, fc, fx)
        c, d = c_new, d_new
        n_iter += 1
    return (a + b) / 2.0, n_iter


def _square_breakpoints(n: np.ndarray, k: int, p: int) -> np.ndarray:
    """Phases in ``[-pi, pi)`` at which some sample of ``sign(cos(theta_n - phi))`` flips."""
    th = _theta(n, k, p)
    bp = np.concatenate([th + np.pi / 2, th - np.pi / 2])
    bp = ((bp + np.pi) % (2 * np.pi)) - np.pi
    return np.unique(np.round(bp, 12))


def _centre_in_flat_interval(phi: np.ndarray, bp: np.ndarray) -> np.ndarray:
    """Midpoint of the breakpoint interval containing each ``phi`` (circular)."""
    x = ((np.asarray(phi, dtype=np.float64) + np.pi) % (2 * np.pi)) - np.pi
    i = np.searchsorted(bp, x, side="right")
    hi = np.where(i < len(bp), bp[np.minimum(i, len(bp) - 1)], bp[0] + 2 * np.pi)
    lo = np.where(i > 0, bp[np.maximum(i - 1, 0)], bp[-1] - 2 * np.pi)
    return _wrap((lo + hi) / 2.0)


def _fit_phase_model(Y: np.ndarray, n: np.ndarray, k: int, p: int, model: str) -> tuple[dict, dict]:
    th = _theta(n, k, p)
    n_grid = 4 * p
    step = 2 * np.pi / n_grid
    grid = -np.pi + step * (np.arange(n_grid) + 0.5)
    rss_grid = _affine_rss_grid(_basis(model, th, grid), Y)      # [G, m]
    phi0 = grid[rss_grid.argmin(axis=0)]
    phi1, n_iter = _golden_section(lambda ph: _affine_rss_columns(_basis(model, th, ph), Y),
                                   phi0 - step, phi0 + step, GOLDEN_TOL, GOLDEN_MAX_ITER)
    phi = _centre_in_flat_interval(phi1, _square_breakpoints(n, k, p)) if model == "square" else _wrap(phi1)
    alpha, beta = _affine_coefficients(_basis(model, th, phi), Y)
    neg = alpha < 0                                              # -wave(phi) == wave(phi + pi)
    if neg.any():
        phi = np.where(neg, _wrap(phi + np.pi), phi)
        alpha, beta = _affine_coefficients(_basis(model, th, phi), Y)
    info = {"n_grid": n_grid, "grid_offset_half_step": True, "bracket_halfwidth": step,
            "golden_section_tol": GOLDEN_TOL, "golden_section_iters": n_iter,
            "phase_centred_in_flat_interval": model == "square"}
    return {"alpha": alpha, "phi": phi, "beta": beta}, info


# --------------------------------------------------------------------------- #
# batched fit of one model + statistics                                        #
# --------------------------------------------------------------------------- #
def _fit_statistics(Y: np.ndarray, pred: np.ndarray, q: int) -> dict:
    """rss, nmse, r2, aic, aicc per column (formulas in the module docstring)."""
    N = Y.shape[0]
    if N <= q + 1:
        raise ValueError(f"{N} positions cannot support a {q}-parameter model (AICc undefined)")
    rss = ((Y - pred) ** 2).sum(axis=0)
    ss = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    zero = np.flatnonzero(ss <= 0)
    if zero.size:
        raise ValueError(f"column(s) {zero.tolist()} have zero variance — nmse/r2 undefined")
    nmse = rss / ss
    aic = N * np.log(np.maximum(rss, RSS_FLOOR) / N) + 2 * q
    return {"rss": rss, "nmse": nmse, "r2": 1.0 - nmse, "aic": aic,
            "aicc": aic + 2.0 * q * (q + 1) / (N - q - 1), "n_positions": N}


def _fit_batch(Y: np.ndarray, n: np.ndarray, k: int, p: int, model: str) -> dict:
    """Fit ``model`` to every column of ``Y [N, m]`` sampled at positions ``n`` (shared ``k``)."""
    if model in ("sinusoid", "odd_harmonics"):
        params, info = _fit_linear(Y, n, k, p, model), None
    else:
        params, info = _fit_phase_model(Y, n, k, p, model)
    pred = _predict(params, n, k, p, model)
    out = {"model": model, "n_params": N_PARAMS[model], "params": params, "pred": pred,
           **_fit_statistics(Y, pred, N_PARAMS[model])}
    if info is not None:
        out["phase_search"] = info
    return out


def _squeeze_fit(fit: dict) -> dict:
    """Single-column batch result -> scalars / 1-D arrays."""
    def sq(v):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[-1] == 1:
            v = v[..., 0]
            return float(v) if v.ndim == 0 else v
        return v
    out = {key: sq(v) for key, v in fit.items() if key != "params"}
    out["params"] = {key: sq(v) for key, v in fit["params"].items()}
    return out


# --------------------------------------------------------------------------- #
# public single-curve API                                                      #
# --------------------------------------------------------------------------- #
def fit_model(y, k: int, model: str) -> dict:
    """Fit one model to one curve ``y [p]`` at fundamental ``k`` on all ``p`` positions."""
    Y = _as_columns(y)
    p = Y.shape[0]
    _model_index(model)
    return _squeeze_fit(_fit_batch(Y, np.arange(p), _validate_k(k, p), p, model))


def compare_models(y, k: int, folds: int = 5, seed: int = 0) -> dict:
    """All four fits of ``y`` at ``k`` plus AIC/AICc ranking, ``delta_aic`` and CV ranking."""
    Y = _as_columns(y)
    p = Y.shape[0]
    k = _validate_k(k, p)
    n = np.arange(p)
    fits = {m: _squeeze_fit(_fit_batch(Y, n, k, p, m)) for m in MODEL_NAMES}
    aic = {m: fits[m]["aic"] for m in MODEL_NAMES}
    aicc = {m: fits[m]["aicc"] for m in MODEL_NAMES}
    best_aic = min(MODEL_NAMES, key=lambda m: aic[m])
    best_aicc = min(MODEL_NAMES, key=lambda m: aicc[m])
    cv = cross_validate(Y[:, 0], k, folds, seed)
    return {
        "p": p, "k": k, "phase_convention": PHASE_CONVENTION, "rss_floor": RSS_FLOOR,
        "models": fits,
        "aic": aic, "aicc": aicc,
        "best_by_aic": best_aic, "best_by_aicc": best_aicc,
        "delta_aic": {m: aic[m] - aic[best_aic] for m in MODEL_NAMES},
        "delta_aic_sinusoid_minus_square": aic["sinusoid"] - aic["square"],
        "r2": {m: fits[m]["r2"] for m in MODEL_NAMES},
        "cv": cv, "best_by_cv": cv["best_by_cv"],
    }


def cross_validate(y, k: int, folds: int = 5, seed: int = 0) -> dict:
    """Seeded K-fold over the ``p`` token positions: fit on the rest, score the held-out ones.

    Per model and fold: ``heldout_nmse = sum_held (y - pred)^2 / sum_held (y - mean_all y)^2``;
    reported as the mean over folds (decides ``best_by_cv``) and pooled over all positions
    (each position is predicted exactly once). Folds are ``rng.permutation(p)`` split into
    ``folds`` nearly equal parts with ``np.random.default_rng(seed)``.
    """
    Y = _as_columns(y)
    p = Y.shape[0]
    k = _validate_k(k, p)
    folds = int(folds)
    if not 2 <= folds <= p:
        raise ValueError(f"folds must be in 2..{p}, got {folds}")
    rng = np.random.default_rng(int(seed))
    held_sets = np.array_split(rng.permutation(p), folds)
    y1 = Y[:, 0]
    ybar = y1.mean()
    ss_full = ((y1 - ybar) ** 2).sum()
    per_fold: dict[str, list[float]] = {m: [] for m in MODEL_NAMES}
    pooled: dict[str, float] = {}
    for m in MODEL_NAMES:
        pred_all = np.empty(p)
        for held in held_sets:
            train = np.setdiff1d(np.arange(p), held)
            fit = _fit_batch(Y[train], train, k, p, m)
            pred = _predict(fit["params"], held, k, p, m)[:, 0]
            pred_all[held] = pred
            denom = ((y1[held] - ybar) ** 2).sum()
            if denom <= 0:
                raise ValueError("held-out positions carry no variance — fold NMSE undefined")
            per_fold[m].append(float(((y1[held] - pred) ** 2).sum() / denom))
        pooled[m] = float(((y1 - pred_all) ** 2).sum() / ss_full)
    mean_nmse = {m: float(np.mean(per_fold[m])) for m in MODEL_NAMES}
    return {
        "folds": folds, "seed": int(seed), "fold_sizes": [int(len(h)) for h in held_sets],
        "heldout_nmse_per_fold": per_fold,
        "heldout_nmse_mean": mean_nmse,
        "heldout_nmse_pooled": pooled,
        "best_by_cv": min(MODEL_NAMES, key=lambda m: mean_nmse[m]),
        "best_by_cv_pooled": min(MODEL_NAMES, key=lambda m: pooled[m]),
    }


def frequency_sensitivity(y, ks=None) -> dict:
    """Refit all models at every ``k`` in ``ks`` and report the best model (by AIC) per ``k``.

    ``ks=None`` uses the rule ``top3_spectral_peaks_plus_dominant_pm1`` (the three largest
    spectral peaks of ``y`` and ``alias(k_dom ± 1)``); the rule and the ``ks`` used are echoed.
    """
    Y = _as_columns(y)
    p = Y.shape[0]
    spec = power_spectrum(Y, p)
    k_dom = int(spec["dominant_freq"][0])
    if ks is None:
        order = np.argsort(-spec["power"][:, 0], kind="stable")
        cand = [int(o) + 1 for o in order[:3]] + [alias_frequency(k_dom - 1, p),
                                                    alias_frequency(k_dom + 1, p)]
        ks = sorted({c for c in cand if c > 0})
        rule = "top3_spectral_peaks_plus_dominant_pm1"
    else:
        ks = sorted({int(c) for c in ks})
        rule = "user_supplied"
    n = np.arange(p)
    per_k = {}
    for kk in ks:
        kk = _validate_k(kk, p)
        fits = {m: _squeeze_fit(_fit_batch(Y, n, kk, p, m)) for m in MODEL_NAMES}
        aic = {m: fits[m]["aic"] for m in MODEL_NAMES}
        best = min(MODEL_NAMES, key=lambda m: aic[m])
        per_k[kk] = {"best_by_aic": best, "aic": aic,
                     "delta_aic": {m: aic[m] - aic[best] for m in MODEL_NAMES},
                     "r2": {m: fits[m]["r2"] for m in MODEL_NAMES}}
    return {"ks": ks, "ks_rule": rule, "dominant_frequency": k_dom, "per_k": per_k,
            "best_model_per_k": {kk: per_k[kk]["best_by_aic"] for kk in ks}}


# --------------------------------------------------------------------------- #
# matrix API                                                                   #
# --------------------------------------------------------------------------- #
def fit_matrix(Y, k_per_column, cv_folds: int = 0, seed: int = 0) -> dict:
    """Fit all four models to every column of ``Y [p, n]`` at that column's own ``k``.

    Columns sharing a ``k`` are fitted as one batch (shared design matrix / phase grid).
    ``cv_folds > 0`` additionally runs ``cross_validate`` per column (echoed as a parameter;
    ``0`` means not computed). Returns a dict of per-column arrays.
    """
    Y = _as_columns(Y, "Y")
    p, n_cols = Y.shape
    ks = np.broadcast_to(np.atleast_1d(np.asarray(k_per_column)), (n_cols,))
    if not np.issubdtype(ks.dtype, np.integer):
        raise ValueError("k_per_column must be integers")
    ks = ks.astype(np.int64)
    for kk in np.unique(ks):
        _validate_k(int(kk), p)
    n = np.arange(p)
    order, batches = [], {m: [] for m in MODEL_NAMES}
    for kk in np.unique(ks):
        cols = np.flatnonzero(ks == kk)
        order.append(cols)
        for m in MODEL_NAMES:
            batches[m].append(_fit_batch(Y[:, cols], n, int(kk), p, m))
    inv = np.argsort(np.concatenate(order), kind="stable")

    def gather(model: str, pick) -> np.ndarray:
        return np.concatenate([np.asarray(pick(f)) for f in batches[model]], axis=-1)[..., inv]

    def gather_scalar(model: str, pick) -> np.ndarray:
        return np.concatenate([np.full(f["pred"].shape[1], pick(f)) for f in batches[model]])[inv]

    out: dict = {"p": p, "n_columns": n_cols, "k": ks, "models": list(MODEL_NAMES),
                 "n_params": dict(N_PARAMS), "phase_convention": PHASE_CONVENTION,
                 "rss_floor": RSS_FLOOR, "cv_folds": int(cv_folds), "seed": int(seed),
                 "pred": {}, "params": {}}
    for stat in ("rss", "nmse", "r2", "aic", "aicc"):
        out[stat] = {m: gather(m, lambda f, s=stat: f[s]) for m in MODEL_NAMES}
    for m in MODEL_NAMES:
        out["pred"][m] = gather(m, lambda f: f["pred"])
        keys = [key for key, v in batches[m][0]["params"].items() if isinstance(v, np.ndarray)
                and v.shape[-1] == batches[m][0]["pred"].shape[1]]
        out["params"][m] = {key: gather(m, lambda f, kk=key: f["params"][kk]) for key in keys}
        if "phase_search" in batches[m][0]:
            out["params"][m]["golden_section_iters"] = gather_scalar(
                m, lambda f: f["phase_search"]["golden_section_iters"])
    out["params"]["odd_harmonics"]["harmonics"] = np.array(ODD_HARMONICS)
    aic = np.stack([out["aic"][m] for m in MODEL_NAMES])            # [4, n]
    aicc = np.stack([out["aicc"][m] for m in MODEL_NAMES])
    out["best_by_aic"] = aic.argmin(axis=0)
    out["best_by_aicc"] = aicc.argmin(axis=0)
    out["delta_aic"] = {m: out["aic"][m] - aic.min(axis=0) for m in MODEL_NAMES}
    out["delta_aic_sinusoid_minus_square"] = out["aic"]["sinusoid"] - out["aic"]["square"]
    if int(cv_folds) > 0:
        cv_nmse = np.stack([[cross_validate(Y[:, j], int(ks[j]), cv_folds, seed)
                             ["heldout_nmse_mean"][m] for j in range(n_cols)] for m in MODEL_NAMES])
        out["cv"] = {"heldout_nmse_mean": {m: cv_nmse[i] for i, m in enumerate(MODEL_NAMES)},
                     "best_by_cv": cv_nmse.argmin(axis=0)}
    return out


def summarize(fits: dict) -> dict:
    """Distribution over columns of a ``fit_matrix`` result (JSON-sized, no arrays)."""
    n = fits["n_columns"]
    best = fits["best_by_aic"]
    ratios = fits["params"]["odd_harmonics"]["amplitude_ratio_to_fundamental"]   # [3, n]
    resid = fits["params"]["odd_harmonics"]["phase_residual_j"]
    ks, counts = np.unique(fits["k"], return_counts=True)
    out = {
        "n_columns": int(n),
        "models": list(MODEL_NAMES),
        "n_params": dict(N_PARAMS),
        "fraction_best_by_aic": {m: float((best == i).mean()) for i, m in enumerate(MODEL_NAMES)},
        "fraction_best_by_aicc": {m: float((fits["best_by_aicc"] == i).mean())
                                  for i, m in enumerate(MODEL_NAMES)},
        "r2": {m: dist_summary(fits["r2"][m]) for m in MODEL_NAMES},
        "delta_aic_sinusoid_minus_square": dist_summary(fits["delta_aic_sinusoid_minus_square"]),
        "fraction_square_aic_below_sinusoid": float((fits["delta_aic_sinusoid_minus_square"] > 0).mean()),
        "amplitude_ratio_to_fundamental": {str(j): dist_summary(ratios[i])
                                           for i, j in enumerate(ODD_HARMONICS[1:])},
        "continuous_ideal_amplitude_ratio": {str(j): 1.0 / j for j in ODD_HARMONICS[1:]},
        "abs_phase_residual_j": {str(j): dist_summary(np.abs(resid[i]))
                                 for i, j in enumerate(ODD_HARMONICS[1:])},
        "k_histogram": {int(kk): int(c) for kk, c in zip(ks, counts)},
        "cv_folds": int(fits["cv_folds"]),
    }
    if "cv" in fits:
        out["fraction_best_by_cv"] = {m: float((fits["cv"]["best_by_cv"] == i).mean())
                                      for i, m in enumerate(MODEL_NAMES)}
        out["heldout_nmse_mean"] = {m: dist_summary(fits["cv"]["heldout_nmse_mean"][m])
                                    for m in MODEL_NAMES}
    return out


# --------------------------------------------------------------------------- #
# run-level analysis + CLI                                                     #
# --------------------------------------------------------------------------- #
def _effective_curves(cfg, state: dict) -> dict[str, np.ndarray]:
    """``u_a``, ``u_b``, ``out`` ``[p, d_mlp]`` for the MLP architectures (INTERFACES §5)."""
    if cfg.arch == "mlp":
        from .mlp_mechanism import effective_curves     # lazy: mlp_mechanism imports this module
        return effective_curves(state, cfg)
    if cfg.arch == "mlp_twohot":
        p = cfg.p
        W_in = np.asarray(state["W_in"], dtype=np.float64)
        return {"u_a": W_in[:p], "u_b": W_in[p:], "out": np.asarray(state["W_out"], dtype=np.float64).T}
    raise ValueError(f"wave_fitting CLI handles arch 'mlp'/'mlp_twohot'; got {cfg.arch!r} — "
                     "transformer effective curves come from analysis.transformer_mechanism")


def _flatten_arrays(prefix: str, fits: dict) -> dict[str, np.ndarray]:
    arrays = {f"{prefix}__k": fits["k"], f"{prefix}__best_by_aic": fits["best_by_aic"],
              f"{prefix}__best_by_aicc": fits["best_by_aicc"],
              f"{prefix}__delta_aic_sinusoid_minus_square": fits["delta_aic_sinusoid_minus_square"]}
    for m in MODEL_NAMES:
        for stat in ("rss", "nmse", "r2", "aic", "aicc"):
            arrays[f"{prefix}__{m}__{stat}"] = fits[stat][m]
        arrays[f"{prefix}__{m}__pred"] = fits["pred"][m].astype(np.float32)
        for key, v in fits["params"][m].items():
            arrays[f"{prefix}__{m}__{key}"] = v
    if "cv" in fits:
        arrays[f"{prefix}__best_by_cv"] = fits["cv"]["best_by_cv"]
        for m in MODEL_NAMES:
            arrays[f"{prefix}__{m}__heldout_nmse_mean"] = fits["cv"]["heldout_nmse_mean"][m]
    return arrays


def analyse(run_dir, step: int | None = None, curve_names=CURVE_NAMES, seed: int = 0,
            cv_folds: int = 0) -> tuple[dict, Path]:
    """Fit every neuron's curves at its own dominant frequency; write JSON + npz."""
    run_dir = Path(run_dir)
    cfg, _model, state, meta = load_model_at(run_dir, step)
    curves = _effective_curves(cfg, state)
    bad = [c for c in curve_names if c not in curves]
    if bad:
        raise ValueError(f"{run_dir}: unknown curve name(s) {bad}; choices {list(CURVE_NAMES)}")
    params = {"step": meta["step"], "seed": int(seed), "cv_folds": int(cv_folds),
              "curves": list(curve_names), "k_rule": "dominant_frequency_of_each_column",
              "models": list(MODEL_NAMES), "n_params": dict(N_PARAMS), "rss_floor": RSS_FLOOR,
              "phase_convention": PHASE_CONVENTION, "harmonics": list(ODD_HARMONICS),
              "phase_search": {"n_grid": "4p", "grid_offset_half_step": True,
                               "golden_section_tol": GOLDEN_TOL,
                               "square_phase_centred_in_flat_interval": True}}
    payload = envelope(MODULE, MODULE_VERSION, meta, params)
    arrays: dict[str, np.ndarray] = {}
    # fit_matrix is called through the per-k split, NOT directly. It decides which parameter
    # entries are per-column by comparing shapes against its first batch, so `odd_harmonics`'
    # constant (1, 3, 5, 7) is mistaken for a per-column array whenever the lowest-k group holds
    # exactly four neurons, and the gather raises IndexError. That was documented in
    # mlp_mechanism_activation.fit_curve_matrix as a defect of this file and left unfixed; it then
    # failed 4 of the 20 primary MLP runs at the crossing checkpoint on 2026-09-03. The import is
    # deferred to keep the module-level dependency direction unchanged.
    from .mlp_mechanism_activation import fit_curve_matrix

    for name in curve_names:
        Y = curves[name]
        k = power_spectrum(Y, cfg.p)["dominant_freq"]
        fits = fit_curve_matrix(Y, k, cv_folds, seed)
        payload["results"][name] = summarize(fits)
        arrays.update(_flatten_arrays(name, fits))
    path = write_result(run_dir, MODULE, meta["tag"], payload, arrays)
    return payload, path


def main() -> None:
    ap = argparse.ArgumentParser(description="Waveform model comparison per hidden neuron")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: model_final.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve", nargs="+", default=list(CURVE_NAMES), choices=list(CURVE_NAMES))
    ap.add_argument("--cv-folds", type=int, default=0, help="0 = no cross-validation")
    ap.add_argument("--all-checkpoints", action="store_true")
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
        payload, path = analyse(args.run_dir, step, tuple(args.curve), args.seed, args.cv_folds)
        print(json.dumps({key: payload[key] for key in ("module", "run_id", "step", "params")}, indent=2))
        print(json.dumps(payload["results"], indent=2))
        print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
