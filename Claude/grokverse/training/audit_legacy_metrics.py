"""Phase-1 audit: descriptive structure metrics on all 16 legacy runs' final embeddings.

AI-drafted (Claude), 2026-09-02 -- not yet human-reviewed.
Descriptive only: no hypothesis is tested here (docs/LEGACY_METRIC_AUDIT.md).

Run from training/:
    python audit_legacy_metrics.py
Writes ../docs/data/legacy_metrics_all_runs.json
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from grokverse.analysis.fourier import (IDEAL_SQUARE_ODD_SHARE, alias_frequency,
                                        dominant_frequencies,
                                        embedding_power_spectrum,
                                        harmonic_family_concentration,
                                        harmonic_shape)

RUNS = Path(__file__).resolve().parent / "runs"
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "legacy_metrics_all_runs.json"
N_F_VALUES = (2, 4, 8)
HARMONICS_FOR_COLLISIONS = (2, 3, 4, 5, 6, 7)
THRESHOLD = 0.9
LEGACY_TEST_FUND = [18, 15, 11, 1, 13, 22, 56, 36]      # test_core.check_harmonic_families
#: The 16 pre-study runs listed in docs/BASELINE.md. Pinned explicitly: new
#: study runs (run format v2, docs/dev/RUN_FORMAT_V2.md) appear in the same
#: directory and must NOT enter this audit.
LEGACY_RUN_IDS = (
    "mlp_add_p113_wd1.0_frac0.3_seed0", "mlp_add_p113_wd1.0_frac0.3_seed1",
    "mlp_add_p113_wd1.0_frac0.3_seed2",
    "mlp_add_p113_wd1.0_frac0.5_gf2.0_seed0", "mlp_add_p113_wd1.0_frac0.5_gf2.0_seed1",
    "mlp_add_p113_wd1.0_frac0.5_gf2.0_seed2",
    "txf_add_p113_wd1.0_frac0.3_gf2.0_seed0",
    "txf_add_p113_wd1.0_frac0.3_seed0", "txf_add_p113_wd1.0_frac0.3_seed1",
    "txf_add_p113_wd1.0_frac0.3_seed2",
    "txf_add_p113_wd1.0_frac0.5_gf2.0_seed0", "txf_add_p113_wd1.0_frac0.5_gf2.0_seed1",
    "txf_add_p113_wd1.0_frac0.5_gf2.0_seed2",
    "txf_mul_p113_wd1.0_frac0.5_gf2.0_seed0", "txf_mul_p113_wd1.0_frac0.5_gf2.0_seed1",
    "txf_mul_p113_wd1.0_frac0.5_gf2.0_seed2",
)


# --------------------------------------------------------------------------- #
# descriptive statistics of a power vector (indexed by frequency-1)           #
# --------------------------------------------------------------------------- #
def topk_fraction(power: np.ndarray, k: int) -> float:
    srt = np.sort(power)[::-1]
    return float(srt[:k].sum() / power.sum())


def spectral_entropy(power: np.ndarray) -> dict:
    q = power / power.sum()
    nz = q[q > 0]
    h = float(-(nz * np.log(nz)).sum())
    return {"nats": h, "normalized": h / float(np.log(len(power))),
            "log_n_freqs": float(np.log(len(power)))}


def participation_ratio(power: np.ndarray) -> float:
    return float(power.sum() ** 2 / (power ** 2).sum())


def n_freqs_for(power: np.ndarray, thr: float) -> int:
    cum = np.cumsum(np.sort(power)[::-1] / power.sum())
    return int(np.searchsorted(cum, thr) + 1)


def aliasing_collisions(fundamentals, p: int,
                        harmonics=HARMONICS_FOR_COLLISIONS) -> dict:
    """Harmonics j*k (j in `harmonics`) of a selected fundamental k that alias
    onto ANOTHER selected fundamental k'.  Pure combinatorics of the index set."""
    fset = set(int(k) for k in fundamentals)
    hits = []
    for k in fundamentals:
        for j in harmonics:
            a = alias_frequency(j * int(k), p)
            if a in fset and a != int(k):
                hits.append({"k": int(k), "j": int(j), "aliases_to": int(a),
                             "parity": "even" if j % 2 == 0 else "odd"})
    return {"count": len(hits), "n_odd": sum(h["j"] % 2 for h in hits),
            "n_even": sum(1 - h["j"] % 2 for h in hits), "list": hits}


def odd_even_index_sets(fundamentals, p: int) -> dict:
    """Reproduce harmonic_shape's index bookkeeping and expose the overlap."""
    base = {alias_frequency(k, p) for k in fundamentals} - {0}

    def idx(js):
        s: set[int] = set()
        for j in js:
            s |= {alias_frequency(j * k, p) for k in fundamentals}
        return {i for i in s if i > 0} - base

    odd, even = idx((3, 5, 7)), idx((2, 4, 6))
    return {"base": sorted(base), "odd_set": sorted(odd), "even_set": sorted(even),
            "odd_and_even_overlap": sorted(odd & even),
            "n_odd": len(odd), "n_even": len(even), "n_overlap": len(odd & even)}


def index_sources(fundamentals, p: int, max_j: int = 7) -> dict:
    """For every frequency index, the (k, j) pairs (j <= max_j) that alias onto it."""
    src: dict[int, list] = {}
    for k in fundamentals:
        for j in range(1, max_j + 1):
            a = alias_frequency(j * int(k), p)
            if a > 0:
                src.setdefault(a, []).append([int(k), int(j)])
    return {str(i): v for i, v in sorted(src.items())}


# --------------------------------------------------------------------------- #
# per-run audit                                                                #
# --------------------------------------------------------------------------- #
def audit_run(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    cfg = run["config"]
    p = int(cfg["p"])
    embeds = np.load(run_dir / "embeddings.npy")
    W = np.asarray(embeds[-1][:p], dtype=np.float64)
    spec = embedding_power_spectrum(W, p)
    power = np.asarray(spec["power"], dtype=float)
    dom = dominant_frequencies(W, p, threshold=THRESHOLD, max_k=8)

    frac = power / power.sum()
    rank = {int(f): int(i) + 1 for i, f in enumerate(np.argsort(power)[::-1] + 1)}
    fam = {}
    for n_f in N_F_VALUES:
        r = harmonic_family_concentration(W, p, n_f=n_f, n_null=2000, seed=0)
        r["collisions_j2to7"] = aliasing_collisions(r["fundamentals"], p)
        r["index_sets"] = odd_even_index_sets(r["fundamentals"], p)
        # where each selected fundamental and its capped family members sit in
        # the plain power ranking (1 = strongest of the 56 frequencies)
        r["fundamental_ranks"] = [
            {"k": int(k), "rank": rank[int(k)], "fraction": float(frac[int(k) - 1]),
             "members": [{"j": j, "index": alias_frequency(j * int(k), p),
                          "rank": rank[alias_frequency(j * int(k), p)],
                          "fraction": float(frac[alias_frequency(j * int(k), p) - 1])}
                         for j in (3, 5, 7)]}
            for k in r["fundamentals"]]
        fam[str(n_f)] = r

    return {
        "run_id": cfg["run_id"],
        "arch": cfg["arch"],
        "task": cfg["task"],
        "train_frac": cfg["train_frac"],
        "grokfast": bool(cfg["grokfast"]),
        "seed": cfg["seed"],
        "git_commit": run.get("git_commit"),
        "n_snapshots": int(embeds.shape[0]),
        "embedding_shape": list(embeds.shape),
        "snapshot_step": int(run["logged_steps"][-1]),
        "rows_used": f"embeddings.npy[-1][:{p}]",
        "final_test_acc": run["transition"].get("final_test_acc"),
        "test_generalized_step": run["transition"].get("test_generalized_step"),
        "const_power": spec["const_power"],
        "total_freq_power": spec["total_power"],
        "top1": topk_fraction(power, 1),
        "top3": topk_fraction(power, 3),
        "top8": topk_fraction(power, 8),
        "ranked_freqs_top8": dom["ranked_freqs"][:8],
        "ranked_fraction_top8": dom["ranked_fraction"][:8],
        "legacy_dominant": dom["dominant"],
        "legacy_dominant_fraction": dom["dominant_fraction"],
        "n_keep": dom["n_keep"],
        "n_freqs_for_threshold_90": dom["n_freqs_for_threshold"],
        "cap_binding": dom["cap_binding"],
        "n_freqs_for_50": n_freqs_for(power, 0.5),
        "n_freqs_for_75": n_freqs_for(power, 0.75),
        "n_freqs_for_95": n_freqs_for(power, 0.95),
        "spectral_entropy": spectral_entropy(power),
        "participation_ratio": participation_ratio(power),
        "participation_ratio_over_n_freqs": participation_ratio(power) / len(power),
        "legacy_top8_collisions_j2to7": aliasing_collisions(dom["dominant"], p),
        "harmonic_family": fam,
    }


def agg(vals):
    v = [float(x) for x in vals]
    return {"mean": float(np.mean(v)), "std_ddof1": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}


# --------------------------------------------------------------------------- #
# the failing check, reproduced verbatim                                       #
# --------------------------------------------------------------------------- #
def synthetic_check(p: int = 113) -> dict:
    """Byte-for-byte the synthetic part of test_core.check_harmonic_families
    (same generator, same draw order), plus a per-index decomposition."""
    n = np.arange(p)
    rng = np.random.default_rng(0)
    FUND = list(LEGACY_TEST_FUND)

    def synth(kind, noise, gen):
        cols = []
        for f in FUND:
            for ph in gen.uniform(0, 2 * np.pi, 4):
                wave = np.sin(2 * np.pi * f * n / p + ph)
                cols.append(np.sign(wave + 1e-12) if kind == "square" else wave)
        W = np.stack(cols, axis=1)
        return W + noise * gen.standard_normal(W.shape)

    out = {"fundamentals": FUND, "index_sets": odd_even_index_sets(FUND, p),
           "collisions_j2to7": aliasing_collisions(FUND, p),
           "index_sources_j1to7": index_sources(FUND, p), "cases": {}}
    sets = out["index_sets"]
    for kind in ("sinusoid", "square"):
        for noise in (0.0, 0.6):
            W = synth(kind, noise, rng)
            pw = np.array(embedding_power_spectrum(W, p)["power"])
            sh = harmonic_shape(pw, FUND, p)
            base = sum(pw[i - 1] for i in sets["base"])
            per_idx = {}
            for i in sorted(set(sets["odd_set"]) | set(sets["even_set"])):
                per_idx[str(i)] = {
                    "power_over_base": float(pw[i - 1] / base),
                    "in_odd_set": i in sets["odd_set"],
                    "in_even_set": i in sets["even_set"],
                    "sources": out["index_sources_j1to7"].get(str(i), []),
                }
            overlap_power = float(sum(pw[i - 1] for i in sets["odd_and_even_overlap"]) / base)
            out["cases"][f"{kind}_noise{noise}"] = {
                "kind": kind, "noise": noise, "shape": sh,
                "overlap_power_over_base": overlap_power,
                "odd_share_excluding_overlap": sh["odd_share"] - overlap_power,
                "even_share_excluding_overlap": sh["even_share"] - overlap_power,
                "per_index": per_idx,
            }
    # Exact attribution for the noise-free square case: power adds over columns,
    # and every column is one square wave at a known fundamental, so the power
    # at each index can be split by (source fundamental, harmonic order j).
    rng2 = np.random.default_rng(0)
    _ = synth("sinusoid", 0.0, rng2)         # consume the same draws as the test
    _ = synth("sinusoid", 0.6, rng2)
    Wsq = synth("square", 0.0, rng2)
    out["square_noise0.0_attribution"] = attribute_by_harmonic_order(Wsq, FUND, p, sets)
    out["test_assertions"] = {
        "square_odd_share_gt_0.15": out["cases"]["square_noise0.0"]["shape"]["odd_share"] > 0.15,
        "square_odd_share_within_0.12_of_ideal":
            abs(out["cases"]["square_noise0.0"]["shape"]["odd_share"] - IDEAL_SQUARE_ODD_SHARE) < 0.12,
        "sinusoid_abs_odd_minus_even_lt_0.02":
            abs(out["cases"]["sinusoid_noise0.0"]["shape"]["odd_minus_even"]) < 0.02,
        "odd_minus_even_separates_by_0.10_noise0":
            out["cases"]["square_noise0.0"]["shape"]["odd_minus_even"]
            > out["cases"]["sinusoid_noise0.0"]["shape"]["odd_minus_even"] + 0.10,
        "odd_minus_even_separates_by_0.10_noise0.6":
            out["cases"]["square_noise0.6"]["shape"]["odd_minus_even"]
            > out["cases"]["sinusoid_noise0.6"]["shape"]["odd_minus_even"] + 0.10,
        "sinusoid_even_share_rises_with_noise":
            out["cases"]["sinusoid_noise0.6"]["shape"]["even_share"]
            > out["cases"]["sinusoid_noise0.0"]["shape"]["even_share"],
    }
    return out


def attribute_by_harmonic_order(W: np.ndarray, fundamentals, p: int, sets: dict,
                                cols_per_fundamental: int = 4) -> dict:
    """Split the odd/even shares of `harmonic_shape` by where the power comes from.

    Column c belongs to fundamental k_c. For each index i there is exactly one
    j in 1..(p-1)/2 with alias(j*k_c) == i (bijection for prime p), so the power
    column c puts at i is 'the j-th harmonic of k_c'. Buckets:
      intrinsic_even : j even                         (discrete-grid even energy)
      odd_j_le_7     : j odd, 3 <= j <= 7             (the harmonics the shape statistic targets)
      odd_j_ge_9     : j odd, j >= 9                  (the uncapped tail)
      fundamental_j1 : j == 1                         (a fundamental's own power)
    All numbers are divided by the fundamentals' total power (same denominator as the shares).
    """
    half = (p - 1) // 2
    Wc = np.asarray(W, dtype=float)
    col_power = np.stack([np.asarray(embedding_power_spectrum(Wc[:, c:c + 1], p)["power"])
                          for c in range(Wc.shape[1])], axis=1)          # [half, n_cols]
    col_fund = [int(fundamentals[c // cols_per_fundamental]) for c in range(Wc.shape[1])]
    base = float(sum(col_power[i - 1].sum() for i in sets["base"]))
    inv_j = {}                                                          # (k, i) -> j
    for k in set(col_fund):
        for j in range(1, half + 1):
            inv_j[(k, alias_frequency(j * k, p))] = j

    def bucket(indices):
        b = {"intrinsic_even": 0.0, "odd_j_le_7": 0.0, "odd_j_ge_9": 0.0, "fundamental_j1": 0.0}
        for i in indices:
            for c in range(Wc.shape[1]):
                j = inv_j[(col_fund[c], i)]
                key = ("fundamental_j1" if j == 1 else "intrinsic_even" if j % 2 == 0
                       else "odd_j_le_7" if j <= 7 else "odd_j_ge_9")
                b[key] += float(col_power[i - 1, c]) / base
        b["total"] = sum(b.values())
        return b

    odd_only = sorted(set(sets["odd_set"]) - set(sets["odd_and_even_overlap"]))
    even_only = sorted(set(sets["even_set"]) - set(sets["odd_and_even_overlap"]))
    return {
        "denominator": "sum of power at the 8 fundamentals (same as harmonic_shape)",
        "odd_set_total": bucket(sets["odd_set"]),
        "even_set_total": bucket(sets["even_set"]),
        "overlap_indices": bucket(sets["odd_and_even_overlap"]),
        "odd_only_indices": bucket(odd_only),
        "even_only_indices": bucket(even_only),
        "excluded_odd_harmonics_on_fundamentals": bucket([13, 36, 22]) if p == 113 else None,
    }


# --------------------------------------------------------------------------- #
# discrete square wave at odd p                                                #
# --------------------------------------------------------------------------- #
def square_wave_diagnostic(p: int = 113, k: int = 1) -> dict:
    n = np.arange(p)
    sq = np.sign(np.sin(2 * np.pi * k * n / p) + 1e-12)      # exactly as in test_core
    spec = embedding_power_spectrum(sq[:, None], p)
    power = np.asarray(spec["power"], dtype=float)
    half = (p - 1) // 2
    # for k=1 the j-th harmonic sits at index alias(j) = j for j <= half
    idx = {j: alias_frequency(j * k, p) for j in range(1, p)}
    fund = power[idx[1] - 1]
    # every index 1..half is hit exactly twice among j=1..p-1 (j and p-j); use j<=half only
    odd_j = [j for j in range(3, half + 1, 2)]
    even_j = [j for j in range(2, half + 1, 2)]
    p_odd = float(sum(power[idx[j] - 1] for j in odd_j))
    p_even = float(sum(power[idx[j] - 1] for j in even_j))
    total_with_const = float(spec["total_power"] + spec["const_power"])
    n_plus, n_minus = int((sq > 0).sum()), int((sq < 0).sum())
    # closed form for a two-level +-1 sequence with a run of L ones:
    # |X[m]|^2 = 4 sin^2(pi m L / p) / sin^2(pi m / p)  (m != 0)
    L = n_plus
    m = np.arange(1, half + 1)
    closed = 4 * np.sin(np.pi * m * L / p) ** 2 / np.sin(np.pi * m / p) ** 2
    closed_fraction = closed / closed.sum()
    measured_fraction = power / power.sum()
    return {
        "p": p, "k": k, "sequence": "sign(sin(2*pi*k*n/p) + 1e-12), n = 0..p-1",
        "n_plus_one": n_plus, "n_minus_one": n_minus,
        "sample_mean": float(sq.mean()),
        "const_power_fraction_of_all": float(spec["const_power"] / total_with_const),
        "fundamental_fraction_of_freq_power": float(fund / power.sum()),
        "odd_harmonics_j3_to_55_fraction_of_freq_power": p_odd / float(power.sum()),
        "even_harmonics_j2_to_56_fraction_of_freq_power": p_even / float(power.sum()),
        "odd_share_j357_over_fundamental": float(sum(power[idx[j] - 1] for j in (3, 5, 7)) / fund),
        "even_share_j246_over_fundamental": float(sum(power[idx[j] - 1] for j in (2, 4, 6)) / fund),
        "per_harmonic_over_fundamental": {str(j): float(power[idx[j] - 1] / fund) for j in range(2, 12)},
        "ideal_continuous_1_over_j2": {str(j): (1.0 / j ** 2 if j % 2 else 0.0) for j in range(2, 12)},
        "ideal_square_odd_share_constant": IDEAL_SQUARE_ODD_SHARE,
        "harmonic_shape_single_fundamental": harmonic_shape(power, [k], p),
        "closed_form": {
            "formula": "|X[m]|^2 = 4 sin^2(pi m L/p) / sin^2(pi m/p), L = number of +1 samples",
            "max_abs_diff_fraction_vs_measured": float(np.abs(closed_fraction - measured_fraction).max()),
            "even_m_power_over_fundamental_closed_form":
                {str(mm): float(closed[mm - 1] / closed[0]) for mm in (2, 4, 6)},
        },
        "phase_invariance_check": _phase_invariance(p, k),
        "all_test_fundamentals_isolated": {
            str(f): harmonic_shape(np.asarray(embedding_power_spectrum(
                np.sign(np.sin(2 * np.pi * f * n / p) + 1e-12)[:, None], p)["power"]), [f], p)
            for f in LEGACY_TEST_FUND
        },
    }


def _phase_invariance(p: int, k: int) -> dict:
    n = np.arange(p)
    out = {}
    for ph in (0.0, 0.7, 1.9, 3.3):
        sq = np.sign(np.sin(2 * np.pi * k * n / p + ph) + 1e-12)
        pw = np.asarray(embedding_power_spectrum(sq[:, None], p)["power"])
        sh = harmonic_shape(pw, [k], p)
        out[str(ph)] = {"n_plus_one": int((sq > 0).sum()), "odd_share": sh["odd_share"],
                        "even_share": sh["even_share"]}
    return out


def even_p_contrast(p_even: int = 112) -> dict:
    """FFT-only contrast (the project basis needs odd p): balanced +-1 wave at even p."""
    n = np.arange(p_even)
    sq = np.where(n < p_even // 2, 1.0, -1.0)
    X = np.fft.rfft(sq)
    pw = np.abs(X[1:p_even // 2]) ** 2
    m = np.arange(1, p_even // 2)
    return {"p": p_even, "note": "numpy rfft, not the project basis; balanced runs of 56/56",
            "even_m_power_fraction": float(pw[m % 2 == 0].sum() / pw.sum()),
            "odd_m_power_fraction": float(pw[m % 2 == 1].sum() / pw.sum())}


def alias_bijection_check(p: int = 113, js=range(2, 8)) -> dict:
    """For prime p and any j in 1..p-1, k -> alias(j*k) permutes {1..(p-1)/2}:
    every frequency f is the aliased j-th harmonic of exactly one k."""
    half = (p - 1) // 2
    full = set(range(1, half + 1))
    return {str(j): sorted({alias_frequency(j * k, p) for k in range(1, half + 1)}) == sorted(full)
            for j in js}


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return None


def main() -> None:
    run_dirs = [RUNS / rid for rid in LEGACY_RUN_IDS]
    missing = [d.name for d in run_dirs
               if not ((d / "run.json").exists() and (d / "embeddings.npy").exists())]
    if missing:
        raise SystemExit(f"legacy run(s) missing: {missing}")
    skipped = sorted(d.name for d in RUNS.iterdir() if d.is_dir() and d.name not in LEGACY_RUN_IDS)
    runs = [audit_run(d) for d in run_dirs]
    print(f"audited {len(runs)} legacy runs; skipped non-legacy dirs: {skipped}")

    groups: dict[str, list] = {}
    for r in runs:
        key = f"{r['arch']}_{r['task']}_frac{r['train_frac']}_{'gf' if r['grokfast'] else 'plain'}"
        groups.setdefault(key, []).append(r)
    group_stats = {}
    for key, rs in groups.items():
        g = {"run_ids": [r["run_id"] for r in rs], "n": len(rs)}
        for m in ("top1", "top3", "top8", "participation_ratio"):
            g[m] = agg([r[m] for r in rs])
        g["spectral_entropy_normalized"] = agg([r["spectral_entropy"]["normalized"] for r in rs])
        g["n_freqs_for_threshold_90"] = agg([r["n_freqs_for_threshold_90"] for r in rs])
        for n_f in N_F_VALUES:
            f = str(n_f)
            g[f"family_fraction_nf{f}"] = agg([r["harmonic_family"][f]["family_fraction"] for r in rs])
            g[f"matched_top_m_fraction_nf{f}"] = agg([r["harmonic_family"][f]["matched_top_m_fraction"] for r in rs])
            g[f"odd_minus_even_nf{f}"] = agg([r["harmonic_family"][f]["shape"]["odd_minus_even"] for r in rs])
        group_stats[key] = g

    out = {
        "meta": {
            "title": "Legacy structure metrics on all existing runs (Phase 1 audit)",
            "status": "AI-drafted (Claude), 2026-09-02 -- not yet human-reviewed",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": "training/audit_legacy_metrics.py",
            "git_head": git_head(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(),
            "input": "training/runs/<run_id>/embeddings.npy[-1][:p]  (last logged snapshot, number tokens only)",
            "legacy_run_ids": list(LEGACY_RUN_IDS),
            "non_legacy_dirs_skipped": skipped,
            "definitions": {
                "power": "grokverse.analysis.fourier.embedding_power_spectrum: cos_k^2 + sin_k^2 summed over d_model, k = 1..(p-1)/2, constant mode excluded from the denominator",
                "topK": "sum of the K largest power[k] / sum of all power[k]",
                "n_freqs_for_threshold_90": "dominant_frequencies(threshold=0.9): smallest K with cumulative fraction >= 0.9 (uncapped); cap_binding = (K > max_k=8)",
                "spectral_entropy.normalized": "-sum q_k log q_k / log(56), q_k = power_k / sum power",
                "participation_ratio": "(sum power_k)^2 / sum power_k^2, in [1, 56]",
                "harmonic_family": "grokverse.analysis.fourier.harmonic_family_concentration(n_f, max_harmonic=7, n_null=2000, seed=0)",
                "collisions_j2to7": "pairs (k, j), j in 2..7, k in the selected fundamentals, with alias(j*k) equal to ANOTHER selected fundamental",
                "index_sets": "the odd (j=3,5,7) and even (j=2,4,6) aliased index sets that harmonic_shape sums over, after removing the fundamentals; odd_and_even_overlap = indices counted in BOTH shares",
            },
        },
        "runs": runs,
        "groups": group_stats,
        "synthetic_check_test_core": synthetic_check(),
        "square_wave_k1_p113": square_wave_diagnostic(),
        "even_p_contrast_fft": even_p_contrast(),
        "alias_bijection_p113_j2to7": alias_bijection_check(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
