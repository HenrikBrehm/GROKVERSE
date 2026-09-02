"""Paired-by-seed statistics for the architecture study — numpy only (docs/dev/INTERFACES.md §11).

Every architecture comparison of the study is *paired by seed*: transformer seed s and MLP seed s
share the same train/test split, so the per-seed difference ``d_s = a_s - b_s`` removes the split's
contribution to the variance. This module reports that difference with several intervals and tests
side by side; it never decides anything. Every tunable (``n_boot``, ``seed``, ``alpha``,
``max_exact_n``, ``mad_scale``) is echoed in the returned dictionary, and a quantity that is
undefined for the given data is returned as ``None`` with an explicit reason, never as a stand-in
number.

Definitions, with ``d`` = the n aligned per-seed differences ``a - b``:

* ``mean_diff``, ``median_diff``; percentile bootstrap intervals of both: ``n_boot`` resamples of the
  n differences with replacement from ``np.random.default_rng(seed)``; the interval is the
  ``(alpha/2, 1 - alpha/2)`` quantiles of the resampled statistic.
* exact sign test: ties (``d == 0``) excluded and counted;
  ``p = min(1, 2 * P[Bin(m, 1/2) <= min(n_pos, n_neg)])`` with ``m = n_pos + n_neg``.
* exact Wilcoxon signed-rank test: zeros dropped and counted, ``|d|`` ranked with average ranks for
  tied values, ``W+`` = rank sum of the positive differences, ``W-`` of the negative ones; the null
  distribution of ``W+`` is obtained by enumerating all ``2^m`` sign assignments (``m <= max_exact_n``);
  two-sided ``p = min(1, 2 * P[W+ <= min(W+, W-)])`` (the enumeration is symmetric about m(m+1)/4).
* Cohen's ``d_z = mean(d) / sd(d, ddof=1)`` — undefined (``None``) when n < 2 or sd = 0.
* Cliff's ``delta = (#{(i, j): a_i > b_j} - #{(i, j): a_i < b_j}) / n^2`` over all cross pairs of the
  two aligned samples — a between-sample effect size, not a paired one.
* ``direction``: ``'a>b'`` if every non-zero difference is positive, ``'a<b'`` if every one is
  negative, ``'mixed'`` if both signs occur, ``'all_tied'`` if every difference is zero.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

MODULE_VERSION = "1.0"
#: MAD * 1.4826 is a consistent estimator of sigma for normal data; echoed in every ``robust`` result.
MAD_SCALE = 1.4826
DEFAULT_N_BOOT = 10_000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 0
#: Largest number of non-zero differences for which the 2^n Wilcoxon enumeration is run.
MAX_EXACT_WILCOXON_N = 20
DIRECTIONS = ("a>b", "a<b", "mixed", "all_tied")

__all__ = [
    "align_by_seed", "bootstrap_ci", "cliffs_delta", "cohens_dz", "paired", "plot_paired",
    "robust", "sign_test_exact", "wilcoxon_signed_rank_exact",
]


# --------------------------------------------------------------------------- #
# input handling                                                               #
# --------------------------------------------------------------------------- #
def _as_float_or_none(v) -> float | None:
    """``None``, NaN and inf count as missing; anything else must convert to a real number."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value {v!r} is not a number") from exc
    return f if math.isfinite(f) else None


def _finite_array(values, name: str) -> np.ndarray:
    """1-D float array; raises on None/NaN/inf so that missing data is never silently a number."""
    try:
        x = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric entry") from exc
    if x.size and not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains None/NaN/inf — drop missing values first "
                         "(align_by_seed / robust do that and count them)")
    return x


def align_by_seed(values_a: Sequence, values_b: Sequence,
                  seeds_a: Sequence[int], seeds_b: Sequence[int]) -> dict:
    """Align two per-seed samples on their seed labels.

    A pair is kept only if its seed occurs in both samples and both values are finite numbers.
    Everything else is dropped **and counted**: seeds present on one side only (``unpaired``) and
    seeds whose value is None/NaN on either side (``missing_value``). Kept pairs are sorted by seed.
    """
    va, vb = list(values_a), list(values_b)
    sa, sb = [int(s) for s in seeds_a], [int(s) for s in seeds_b]
    if len(va) != len(sa) or len(vb) != len(sb):
        raise ValueError("values and seeds must have equal length: "
                         f"a {len(va)} values / {len(sa)} seeds, b {len(vb)} values / {len(sb)} seeds")
    for label, seeds in (("a", sa), ("b", sb)):
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds in sample {label}: {sorted(seeds)} — "
                             "alignment by seed is ambiguous")
    map_a = {s: _as_float_or_none(v) for s, v in zip(sa, va)}
    map_b = {s: _as_float_or_none(v) for s, v in zip(sb, vb)}
    common = sorted(set(map_a) & set(map_b))
    unpaired_a = sorted(set(map_a) - set(map_b))
    unpaired_b = sorted(set(map_b) - set(map_a))
    missing = [s for s in common if map_a[s] is None or map_b[s] is None]
    kept = [s for s in common if map_a[s] is not None and map_b[s] is not None]
    dropped = sorted(unpaired_a + unpaired_b + missing)
    return {
        "seeds": kept,
        "values_a": [map_a[s] for s in kept],
        "values_b": [map_b[s] for s in kept],
        "n_total_a": len(va),
        "n_total_b": len(vb),
        "n_pairs": len(kept),
        "n_dropped": len(dropped),
        "dropped_seeds": dropped,
        "n_unpaired_a": len(unpaired_a),
        "n_unpaired_b": len(unpaired_b),
        "n_missing_value": len(missing),
        "unpaired_seeds_a": unpaired_a,
        "unpaired_seeds_b": unpaired_b,
        "missing_value_seeds": missing,
    }


# --------------------------------------------------------------------------- #
# exact tests                                                                  #
# --------------------------------------------------------------------------- #
def sign_test_exact(differences) -> dict:
    """Exact two-sided sign test on paired differences.

    Ties (``d == 0``) are excluded and counted. With ``m = n_pos + n_neg`` and
    ``k = min(n_pos, n_neg)``: ``p = min(1, 2 * sum_{i<=k} C(m, i) / 2^m)`` (exact integer
    arithmetic). ``m = 0`` (every pair tied) gives ``p = 1`` — no information either way.
    """
    d = _finite_array(differences, "differences")
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    n_tie = int((d == 0).sum())
    m, k = n_pos + n_neg, min(n_pos, n_neg)
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {
        "p": float(p),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_ties_excluded": n_tie,
        "n_nonzero": m,
        "method": "exact two-sided binomial: p = min(1, 2 * P[Bin(n_nonzero, 1/2) <= min(n_pos, n_neg)])",
    }


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n of ``x``; tied values receive the mean of the ranks they would occupy."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0     # positions i..j hold ranks i+1..j+1
        i = j + 1
    return ranks


def _enumerate_rank_sums(doubled_ranks: np.ndarray) -> np.ndarray:
    """All ``2^m`` values of ``2 * W+`` — one per sign assignment — as exact integers.

    Doubling makes average ranks (multiples of 1/2) integers, so the comparison with the observed
    statistic is exact. Each concatenation step adds the choice "this rank is positive" for one
    difference, so the final array enumerates every subset = every sign assignment once.
    """
    sums = np.zeros(1, dtype=np.int64)
    for r in doubled_ranks:
        sums = np.concatenate([sums, sums + int(r)])
    return sums


def wilcoxon_signed_rank_exact(differences, max_exact_n: int = MAX_EXACT_WILCOXON_N) -> dict:
    """Exact two-sided Wilcoxon signed-rank test by full enumeration of the ``2^m`` sign assignments.

    Zeros are dropped and counted; ``|d|`` is ranked with average ranks for ties;
    ``W+ = sum(rank | d > 0)``, ``W- = sum(rank | d < 0)``, ``T = min(W+, W-)``;
    ``p = min(1, 2 * P_H0[W+ <= T])`` where the null distribution of ``W+`` is the multiset of rank
    sums over all ``2^m`` sign vectors (symmetric about ``m(m+1)/4``, so this equals
    ``P[W+ <= T] + P[W+ >= m(m+1)/2 - T]``). Raises for ``m > max_exact_n``.
    """
    d = _finite_array(differences, "differences")
    nz = d[d != 0]
    m, n_zero = int(nz.size), int(d.size - nz.size)
    if m > max_exact_n:
        raise ValueError(f"exact Wilcoxon enumeration limited to n_nonzero <= {max_exact_n}; got {m}")
    ranks = _average_ranks(np.abs(nz))
    w_plus, w_minus = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
    t = min(w_plus, w_minus)
    doubled = np.rint(2 * ranks).astype(np.int64)
    if not np.allclose(doubled, 2 * ranks):
        raise ValueError("internal error: average ranks are not multiples of 1/2")
    sums = _enumerate_rank_sums(doubled)
    n_le = int((sums <= int(round(2 * t))).sum())
    p = min(1.0, 2.0 * n_le / sums.size)
    return {
        "p": float(p),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "n_nonzero": m,
        "n_zero_dropped": n_zero,
        "n_tied_abs_groups": int(len(ranks) - len(np.unique(np.abs(nz)))),
        "n_sign_assignments_enumerated": int(sums.size),
        "max_exact_n": int(max_exact_n),
        "method": "exact two-sided, all 2^n_nonzero sign assignments enumerated; "
                  "average ranks for tied |d|; p = min(1, 2 * P[W+ <= min(W+, W-)])",
    }


# --------------------------------------------------------------------------- #
# effect sizes                                                                 #
# --------------------------------------------------------------------------- #
def cohens_dz(differences) -> float | None:
    """Paired-sample effect size ``d_z = mean(d) / sd(d, ddof=1)``.

    Returns ``None`` when it is undefined: fewer than two differences, or ``sd == 0`` (identical
    samples or a perfectly constant shift, where the ratio is 0/0 or ±inf).
    """
    d = _finite_array(differences, "differences")
    if d.size < 2:
        return None
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return None
    return float(d.mean() / sd)


def cliffs_delta(values_a, values_b) -> float:
    """Cliff's delta ``(#{a_i > b_j} - #{a_i < b_j}) / (n_a * n_b)`` over all cross pairs.

    Between-sample ordinal effect size in [-1, 1]: +1 means every a exceeds every b.
    """
    a = _finite_array(values_a, "values_a")
    b = _finite_array(values_b, "values_b")
    if a.size == 0 or b.size == 0:
        raise ValueError("cliffs_delta needs at least one value in each sample")
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (a.size * b.size))


# --------------------------------------------------------------------------- #
# robust summary and bootstrap                                                 #
# --------------------------------------------------------------------------- #
def robust(values: Sequence, mad_scale: float = MAD_SCALE) -> dict:
    """Median, scaled MAD, IQR, min, max, n and the number of missing (None/NaN/inf) entries.

    ``mad = mad_scale * median(|x - median(x)|)``; ``iqr = q75 - q25`` with numpy's linear
    quantile interpolation. With ``n == 0`` every statistic is ``None`` (nothing is measured).
    """
    kept = [f for f in (_as_float_or_none(v) for v in values) if f is not None]
    n_missing = len(values) - len(kept)
    if not kept:
        return {"median": None, "mad": None, "mad_raw": None, "mad_scale": float(mad_scale),
                "iqr": None, "q25": None, "q75": None, "min": None, "max": None,
                "n": 0, "n_missing": n_missing}
    x = np.asarray(kept, dtype=float)
    med = float(np.median(x))
    mad_raw = float(np.median(np.abs(x - med)))
    q25, q75 = (float(q) for q in np.quantile(x, [0.25, 0.75]))
    return {
        "median": med,
        "mad": mad_scale * mad_raw,
        "mad_raw": mad_raw,
        "mad_scale": float(mad_scale),
        "iqr": q75 - q25,
        "q25": q25,
        "q75": q75,
        "min": float(x.min()),
        "max": float(x.max()),
        "n": int(x.size),
        "n_missing": int(n_missing),
    }


def bootstrap_ci(fn: Callable[[np.ndarray], float], x, n_boot: int, seed: int,
                 alpha: float = DEFAULT_ALPHA) -> dict:
    """Percentile bootstrap interval of the scalar statistic ``fn(x)``.

    Rows of ``x`` (its first axis) are resampled with replacement ``n_boot`` times using
    ``np.random.default_rng(seed)``; the interval is the ``(alpha/2, 1 - alpha/2)`` quantiles of the
    ``n_boot`` resampled statistics. ``fn`` must return a finite scalar.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0 or x.shape[0] == 0:
        raise ValueError("bootstrap_ci needs a non-empty array with a sample axis 0")
    if not np.all(np.isfinite(x)):
        raise ValueError("bootstrap_ci: x contains None/NaN/inf — drop missing values first")
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = x.shape[0]
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_boot), n))
    stats = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        stats[i] = _scalar(fn(x[idx[i]]), "fn")
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": _scalar(fn(x), "fn"),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n": int(n),
        "method": "percentile bootstrap, rows resampled with replacement",
    }


def _scalar(v, name: str) -> float:
    arr = np.asarray(v, dtype=float)
    if arr.size != 1 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must return one finite scalar, got {v!r}")
    return float(arr.reshape(()))


# --------------------------------------------------------------------------- #
# the paired-by-seed comparison                                                #
# --------------------------------------------------------------------------- #
def _direction(n_pos: int, n_neg: int) -> str:
    if n_pos and not n_neg:
        return "a>b"
    if n_neg and not n_pos:
        return "a<b"
    return "mixed" if (n_pos and n_neg) else "all_tied"


def _wilcoxon_or_reason(d: np.ndarray, max_exact_n: int) -> dict:
    """The exact test when it is feasible; otherwise ``p = None`` with the reason stated."""
    m = int((d != 0).sum())
    if m > max_exact_n:
        return {"p": None, "n_nonzero": m, "n_zero_dropped": int(d.size - m),
                "max_exact_n": int(max_exact_n), "method": None,
                "not_computed_reason": f"n_nonzero={m} exceeds max_exact_n={max_exact_n}; "
                                       "only the exact 2^n enumeration is implemented"}
    return wilcoxon_signed_rank_exact(d, max_exact_n)


def paired(values_a: Sequence, values_b: Sequence, seeds_a: Sequence[int], seeds_b: Sequence[int],
           n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
           max_exact_n: int = MAX_EXACT_WILCOXON_N) -> dict:
    """Paired-by-seed comparison of two per-seed samples (module docstring for every formula).

    Pairs are aligned by seed through ``align_by_seed``; unpaired seeds and missing values are
    dropped and counted. Differences are ``d_s = a_s - b_s``. Raises ``ValueError`` when no pair
    survives. The 95 % interval (``alpha = 0.05``) is fixed because the key is named ``ci95``.
    """
    al = align_by_seed(values_a, values_b, seeds_a, seeds_b)
    if al["n_pairs"] == 0:
        raise ValueError(f"no seed pairs to compare — dropped {al['n_dropped']} "
                         f"(seeds {al['dropped_seeds']})")
    a, b = np.asarray(al["values_a"], dtype=float), np.asarray(al["values_b"], dtype=float)
    d = a - b
    n = int(d.size)
    boot_mean = bootstrap_ci(np.mean, d, n_boot, seed, DEFAULT_ALPHA)
    boot_median = bootstrap_ci(np.median, d, n_boot, seed, DEFAULT_ALPHA)
    sign = sign_test_exact(d)
    wil = _wilcoxon_or_reason(d, max_exact_n)
    n_pos, n_neg, n_zero = sign["n_positive"], sign["n_negative"], sign["n_ties_excluded"]
    dz = cohens_dz(d)
    sd = float(d.std(ddof=1)) if n > 1 else None
    if dz is None:
        dz_reason = "fewer than two pairs" if n < 2 else "sd of the differences is 0"
    else:
        dz_reason = None
    return {
        "module": "statistics",
        "module_version": MODULE_VERSION,
        "difference_definition": "d_s = a_s - b_s for every seed s present in both samples",
        "params": {"n_boot": int(n_boot), "seed": int(seed), "alpha": DEFAULT_ALPHA,
                   "max_exact_n": int(max_exact_n)},
        **{k: al[k] for k in ("n_total_a", "n_total_b", "n_pairs", "n_dropped", "dropped_seeds",
                              "n_unpaired_a", "n_unpaired_b", "n_missing_value")},
        "seeds": al["seeds"],
        "values_a": a.tolist(),
        "values_b": b.tolist(),
        "differences": d.tolist(),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "sd_diff_ddof1": sd,
        "bootstrap_ci95": [boot_mean["ci_low"], boot_mean["ci_high"]],
        "bootstrap_ci95_median_diff": [boot_median["ci_low"], boot_median["ci_high"]],
        "bootstrap_ci95_excludes_zero": bool(boot_mean["ci_low"] > 0 or boot_mean["ci_high"] < 0),
        "bootstrap": {"statistic": "mean_diff", **{k: boot_mean[k] for k in
                                                    ("method", "n_boot", "seed", "alpha")}},
        "sign_test_p": sign["p"],
        "sign_test": sign,
        "wilcoxon_signed_rank_p": wil["p"],
        "wilcoxon_signed_rank": wil,
        "cohens_dz": dz,
        "cohens_dz_undefined_reason": dz_reason,
        "cliffs_delta": cliffs_delta(a, b),
        "cliffs_delta_definition": "(#{a_i > b_j} - #{a_i < b_j}) / n_pairs^2 over all cross pairs "
                                   "of the aligned samples",
        "direction": _direction(n_pos, n_neg),
        "direction_definition": "'a>b'/'a<b': every non-zero difference has that sign; "
                                "'mixed': both signs occur; 'all_tied': every difference is 0",
        "all_positive": bool(n_pos == n),
        "all_negative": bool(n_neg == n),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
    }


# --------------------------------------------------------------------------- #
# figure                                                                       #
# --------------------------------------------------------------------------- #
def plot_paired(values_a: Sequence, values_b: Sequence, seeds: Sequence[int],
                labels: Sequence[str], path, ylabel: str | None = None,
                title: str | None = None) -> Path:
    """Draw every seed as a connected pair (a at x=0, b at x=1) with the medians marked.

    Pairs with a missing value on either side are omitted from the drawing; the title states how
    many pairs were drawn out of how many seeds. Returns the written path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (len(values_a) == len(values_b) == len(seeds)):
        raise ValueError("values_a, values_b and seeds must have equal length")
    if len(labels) != 2:
        raise ValueError(f"labels must be (label_a, label_b), got {labels!r}")
    pairs = [(int(s), _as_float_or_none(va), _as_float_or_none(vb))
             for s, va, vb in zip(seeds, values_a, values_b)]
    pairs = [(s, va, vb) for s, va, vb in pairs if va is not None and vb is not None]
    if not pairs:
        raise ValueError("plot_paired: no complete (a, b) pair to draw")

    fig, ax = plt.subplots(figsize=(4.4, 4.8))
    for s, va, vb in pairs:
        ax.plot([0, 1], [va, vb], color="#4c8dff", marker="o", ms=4, lw=1.1, alpha=0.75)
        ax.annotate(str(s), (1.04, vb), fontsize=7, va="center", color="#555555")
    med_a = float(np.median([va for _, va, _ in pairs]))
    med_b = float(np.median([vb for _, _, vb in pairs]))
    ax.hlines([med_a, med_b], [-0.18, 0.82], [0.18, 1.18], color="#ff6b6b", lw=3,
              label=f"median ({med_a:.4g} vs {med_b:.4g})")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(labels))
    ax.set_xlim(-0.45, 1.55)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(f"{title or 'paired by seed'} — {len(pairs)}/{len(seeds)} pairs", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
