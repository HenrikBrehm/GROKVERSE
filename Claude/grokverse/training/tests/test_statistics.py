"""Checks for analysis/statistics.py (docs/dev/INTERFACES.md §11).

Run from training/:  python tests/test_statistics.py
Same check() convention as test_core.py; exits non-zero on the first failure. Every expected number
below is derived by hand in a comment next to it; no value was copied from the implementation.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis.statistics import (MAX_EXACT_WILCOXON_N, align_by_seed,  # noqa: E402
                                           bootstrap_ci, cliffs_delta, cohens_dz,
                                           paired, plot_paired, robust,
                                           sign_test_exact,
                                           wilcoxon_signed_rank_exact)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def _close(x, y, tol=1e-12) -> bool:
    return x is not None and y is not None and abs(float(x) - float(y)) <= tol


# --------------------------------------------------------------------------- #
def check_wilcoxon_hand_enumerated():
    print("-- exact Wilcoxon signed-rank, hand-enumerated (n = 6) --")
    # d = [3, -1, 4, 2, 5, 6]: |d| = 3,1,4,2,5,6 -> ranks 3,1,4,2,5,6 (no ties).
    # W- = rank of |-1| = 1; W+ = 20; T = 1. Subsets of {1..6} with sum <= 1: {} and {1} -> 2 of 64.
    # two-sided p = 2 * 2/64 = 4/64 = 0.0625.
    w = wilcoxon_signed_rank_exact([3, -1, 4, 2, 5, 6])
    check("n=6, no ties: W+ = 20, W- = 1", w["w_plus"] == 20.0 and w["w_minus"] == 1.0)
    check("n=6, no ties: p = 4/64 = 0.0625 (hand-enumerated)", w["p"] == 0.0625)
    check("n=6: 2^6 = 64 sign assignments were enumerated",
          w["n_sign_assignments_enumerated"] == 64 and w["n_nonzero"] == 6)

    # d = [2, -2, 3, 5, 5, -1]: |d| = 2,2,3,5,5,1 -> average ranks 2.5,2.5,4,5.5,5.5,1.
    # W- = 2.5 (the -2) + 1 (the -1) = 3.5; W+ = 2.5 + 4 + 5.5 + 5.5 = 17.5; T = 3.5.
    # Subsets of {1, 2.5, 2.5, 4, 5.5, 5.5} with sum <= 3.5: {}, {1}, {2.5}x2, {1, 2.5}x2 -> 6 of 64.
    # two-sided p = 2 * 6/64 = 12/64 = 0.1875.
    w2 = wilcoxon_signed_rank_exact([2, -2, 3, 5, 5, -1])
    check("n=6 with ties: average ranks give W+ = 17.5, W- = 3.5",
          w2["w_plus"] == 17.5 and w2["w_minus"] == 3.5)
    check("n=6 with ties: p = 12/64 = 0.1875 (hand-enumerated)", w2["p"] == 0.1875)
    check("n=6 with ties: two tie groups counted", w2["n_tied_abs_groups"] == 2)

    # A zero difference is dropped and counted; the rest is case 1 again.
    w3 = wilcoxon_signed_rank_exact([0, 3, -1, 4, 2, 5, 6])
    check("zero difference dropped (counted) and p unchanged",
          w3["n_zero_dropped"] == 1 and w3["n_nonzero"] == 6 and w3["p"] == 0.0625)

    # Sign-balanced control d = [1, -2, 3, -4, 5, -6]: W+ = 1+3+5 = 9, W- = 12, T = 9.
    # Subset-sum counts of {1..6} for s = 0..9 are 1,1,1,2,2,3,4,4,4,5 -> 27 of 64 have sum <= 9.
    # two-sided p = 2 * 27/64 = 54/64 = 0.84375: NOT significant.
    wb = wilcoxon_signed_rank_exact([1, -2, 3, -4, 5, -6])
    check("balanced-sign control: W+ = 9, W- = 12", wb["w_plus"] == 9.0 and wb["w_minus"] == 12.0)
    check("balanced-sign control: p = 54/64 = 0.84375 (hand-enumerated), does NOT pass",
          wb["p"] == 0.84375 and wb["p"] > 0.05)

    # Symmetry: negating every difference swaps W+ and W- and leaves p unchanged.
    wn = wilcoxon_signed_rank_exact([-3, 1, -4, -2, -5, -6])
    check("negated differences: W+/W- swap, p identical",
          wn["w_plus"] == 1.0 and wn["w_minus"] == 20.0 and wn["p"] == w["p"])

    # n = 10 all positive: T = W- = 0; only the empty subset has sum <= 0 -> p = 2/1024.
    w10 = wilcoxon_signed_rank_exact(np.arange(1, 11))
    check("n=10 all positive: p = 2/1024", w10["p"] == 2 / 1024)
    # n = 20 (the documented limit) all positive: p = 2 / 2^20.
    w20 = wilcoxon_signed_rank_exact(np.arange(1, 21))
    check("n=20 all positive at the exact limit: p = 2/2^20",
          w20["p"] == 2 / 2 ** 20 and w20["n_sign_assignments_enumerated"] == 2 ** 20)
    check("n=21 non-zero differences exceed the exact limit -> ValueError",
          _raises(lambda: wilcoxon_signed_rank_exact(np.arange(1, 22))))
    check("n=21 is computable when the caller raises max_exact_n explicitly",
          wilcoxon_signed_rank_exact(np.arange(1, 22), max_exact_n=21)["p"] == 2 / 2 ** 21)
    check("NaN in the differences is refused, not silently dropped",
          _raises(lambda: wilcoxon_signed_rank_exact([1.0, float("nan"), 2.0])))


# --------------------------------------------------------------------------- #
def check_sign_test():
    print("\n-- exact sign test --")
    # 8 positive, 2 negative: p = 2 * (C(10,0) + C(10,1) + C(10,2)) / 2^10 = 2 * 56 / 1024.
    s = sign_test_exact([1] * 8 + [-1] * 2)
    check("8+/2-: p = 112/1024 = 0.109375", s["p"] == 112 / 1024)
    check("5+/5-: p capped at 1", sign_test_exact([1] * 5 + [-1] * 5)["p"] == 1.0)
    # ties excluded: [0, 0, 1, 1, 1] -> m = 3, k = 0 -> p = 2 * 1/8 = 0.25
    t = sign_test_exact([0, 0, 1, 1, 1])
    check("ties excluded and counted: p = 0.25, n_ties = 2",
          t["p"] == 0.25 and t["n_ties_excluded"] == 2 and t["n_nonzero"] == 3)
    check("10 positive: p = 2/1024", sign_test_exact([1] * 10)["p"] == 2 / 1024)
    z = sign_test_exact([0, 0, 0])
    check("all tied: p = 1 with n_nonzero = 0 (no information)", z["p"] == 1.0 and z["n_nonzero"] == 0)


# --------------------------------------------------------------------------- #
def check_effect_sizes():
    print("\n-- effect sizes --")
    # d = [1, 2, 3]: mean 2, sd(ddof=1) = 1 -> d_z = 2
    check("cohens_dz([1,2,3]) = 2.0", cohens_dz([1, 2, 3]) == 2.0)
    check("cohens_dz undefined (None) for a constant shift", cohens_dz([1, 1, 1]) is None)
    check("cohens_dz undefined (None) for a single pair", cohens_dz([1]) is None)
    # a = [1,2,3], b = [0,0,4]: per a_i two b's below, one above -> (6 - 3) / 9 = 1/3
    check("cliffs_delta hand case = 1/3", _close(cliffs_delta([1, 2, 3], [0, 0, 4]), 1 / 3))
    check("cliffs_delta of identical samples = 0", cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0)
    check("cliffs_delta = +1 / -1 under complete separation",
          cliffs_delta([5, 6], [1, 2]) == 1.0 and cliffs_delta([1, 2], [5, 6]) == -1.0)
    check("cliffs_delta refuses an empty sample", _raises(lambda: cliffs_delta([], [1])))


# --------------------------------------------------------------------------- #
def check_robust():
    print("\n-- robust summary --")
    r = robust([1, 2, 3, 4, 100, None, float("nan")])
    # kept: 1,2,3,4,100 -> median 3; |x-3| = 2,1,0,1,97 -> MAD_raw 1 -> MAD 1.4826;
    # q25 = 2, q75 = 4 (linear) -> IQR 2
    check("robust: n = 5, n_missing = 2", r["n"] == 5 and r["n_missing"] == 2)
    check("robust: median 3, MAD_raw 1, MAD 1.4826", r["median"] == 3.0 and r["mad_raw"] == 1.0
          and _close(r["mad"], 1.4826))
    check("robust: IQR 2, min 1, max 100", r["iqr"] == 2.0 and r["min"] == 1.0 and r["max"] == 100.0)
    check("robust echoes its MAD scale", r["mad_scale"] == 1.4826)
    e = robust([None, None])
    check("robust of nothing: n = 0, every statistic None, missing counted",
          e["n"] == 0 and e["median"] is None and e["mad"] is None and e["n_missing"] == 2)
    check("robust refuses a non-numeric entry", _raises(lambda: robust(["x", 1])))


# --------------------------------------------------------------------------- #
def check_bootstrap():
    print("\n-- percentile bootstrap --")
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 200)
    res = bootstrap_ci(np.mean, x, n_boot=2000, seed=0)
    check("bootstrap CI of the mean covers the true mean 5 (n=200, sd=1)",
          res["ci_low"] < 5.0 < res["ci_high"])
    check("bootstrap CI brackets the point estimate",
          res["ci_low"] < res["estimate"] < res["ci_high"])
    # 1.96 * 1/sqrt(200) ~ 0.139 half-width -> full width ~0.28
    check("bootstrap CI width is ~0.28 (2 * 1.96 / sqrt(200))",
          0.2 < res["ci_high"] - res["ci_low"] < 0.4)
    check("bootstrap echoes n_boot, seed, alpha, n, method",
          res["n_boot"] == 2000 and res["seed"] == 0 and res["alpha"] == 0.05 and res["n"] == 200
          and "percentile" in res["method"])
    again = bootstrap_ci(np.mean, x, n_boot=2000, seed=0)
    check("bootstrap is deterministic under its seed",
          again["ci_low"] == res["ci_low"] and again["ci_high"] == res["ci_high"])
    other = bootstrap_ci(np.mean, x, n_boot=2000, seed=1)
    check("a different seed gives a different resample",
          other["ci_low"] != res["ci_low"] or other["ci_high"] != res["ci_high"])
    wide = bootstrap_ci(np.mean, x, n_boot=2000, seed=0, alpha=0.5)
    check("alpha = 0.5 gives a narrower interval than alpha = 0.05",
          wide["ci_high"] - wide["ci_low"] < res["ci_high"] - res["ci_low"])
    check("bootstrap refuses n_boot < 1", _raises(lambda: bootstrap_ci(np.mean, x, 0, 0)))
    check("bootstrap refuses an empty sample", _raises(lambda: bootstrap_ci(np.mean, [], 10, 0)))
    check("bootstrap refuses alpha outside (0, 1)",
          _raises(lambda: bootstrap_ci(np.mean, x, 10, 0, alpha=1.0)))
    check("bootstrap refuses a non-scalar statistic",
          _raises(lambda: bootstrap_ci(lambda v: v, x, 10, 0)))
    check("bootstrap refuses NaN input", _raises(lambda: bootstrap_ci(np.mean, [1.0, np.nan], 10, 0)))


# --------------------------------------------------------------------------- #
def check_paired_known_shift():
    print("\n-- paired: a known shift recovers sign, CI excludes 0 --")
    rng = np.random.default_rng(0)
    seeds = list(range(10))
    b = rng.normal(0.0, 1.0, 10)
    a = b + 1.0 + rng.normal(0.0, 0.1, 10)            # shift +1, per-seed noise sd 0.1
    res = paired(a, b, seeds, seeds)                  # default n_boot = 10000, seed = 0
    check("all 10 seeds paired, nothing dropped", res["n_pairs"] == 10 and res["n_dropped"] == 0)
    check("mean difference ~ +1", abs(res["mean_diff"] - 1.0) < 0.15)
    check("every per-seed difference is positive -> direction 'a>b', all_positive",
          res["direction"] == "a>b" and res["all_positive"] and not res["all_negative"])
    check("bootstrap CI95 excludes 0 and lies near the shift",
          res["bootstrap_ci95_excludes_zero"] and res["bootstrap_ci95"][0] > 0.8
          and res["bootstrap_ci95"][1] < 1.2)
    check("sign test p = 2/1024 for 10 consistent signs", res["sign_test_p"] == 2 / 1024)
    check("Wilcoxon p = 2/1024 for 10 consistent signs", res["wilcoxon_signed_rank_p"] == 2 / 1024)
    check("Cohen's d_z large and positive (shift 1, noise sd 0.1)", res["cohens_dz"] > 5.0)
    check("Cliff's delta positive", res["cliffs_delta"] > 0.0)
    check("defaults echoed: n_boot = 10000, seed = 0, alpha = 0.05, max_exact_n = 20",
          res["params"] == {"n_boot": 10000, "seed": 0, "alpha": 0.05,
                            "max_exact_n": MAX_EXACT_WILCOXON_N})
    check("per-seed differences are listed with their seeds",
          res["seeds"] == seeds and len(res["differences"]) == 10
          and _close(res["differences"][3], a[3] - b[3]))
    check("result is JSON-serializable without numpy types", json.dumps(res) is not None)
    again = paired(a, b, seeds, seeds)
    check("paired() is deterministic under its seed", again["bootstrap_ci95"] == res["bootstrap_ci95"])

    rev = paired(b, a, seeds, seeds)
    check("swapping the samples negates the difference and flips the direction",
          _close(rev["mean_diff"], -res["mean_diff"]) and rev["direction"] == "a<b"
          and rev["all_negative"])
    check("swapping the samples leaves both exact p-values unchanged",
          rev["sign_test_p"] == res["sign_test_p"]
          and rev["wilcoxon_signed_rank_p"] == res["wilcoxon_signed_rank_p"])
    check("swapping negates d_z and Cliff's delta",
          _close(rev["cohens_dz"], -res["cohens_dz"]) and _close(rev["cliffs_delta"], -res["cliffs_delta"]))


# --------------------------------------------------------------------------- #
def check_paired_identical():
    print("\n-- paired: identical samples --")
    seeds = list(range(10))
    x = np.linspace(0.1, 1.0, 10)
    res = paired(x, x.copy(), seeds, seeds, n_boot=2000)
    check("identical samples: mean and median difference 0",
          res["mean_diff"] == 0.0 and res["median_diff"] == 0.0)
    check("identical samples: CI95 contains 0",
          res["bootstrap_ci95"][0] <= 0.0 <= res["bootstrap_ci95"][1]
          and not res["bootstrap_ci95_excludes_zero"])
    check("identical samples: sign-test p = 1, Wilcoxon p = 1",
          res["sign_test_p"] == 1.0 and res["wilcoxon_signed_rank_p"] == 1.0)
    check("identical samples: every pair tied -> direction 'all_tied', n_zero = 10",
          res["direction"] == "all_tied" and res["n_zero"] == 10
          and not res["all_positive"] and not res["all_negative"])
    check("identical samples: d_z is None with the reason stated",
          res["cohens_dz"] is None and "sd" in res["cohens_dz_undefined_reason"])
    check("identical samples: Cliff's delta = 0", res["cliffs_delta"] == 0.0)


# --------------------------------------------------------------------------- #
def check_unpaired_dropped():
    print("\n-- paired: unpaired seeds and missing values are dropped and counted --")
    seeds_a = list(range(10))
    seeds_b = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]
    va = [float(i) for i in range(10)]
    vb = [float(i) + 0.5 for i in range(10)]
    va[3] = None                                       # missing on side a
    vb[5] = float("nan")                               # missing on side b
    res = paired(va, vb, seeds_a, seeds_b, n_boot=500)
    # seeds in both: 0..7; minus 3 (None) and 5 (NaN) -> 0,1,2,4,6,7
    check("kept pairs = seeds 0,1,2,4,6,7", res["seeds"] == [0, 1, 2, 4, 6, 7] and res["n_pairs"] == 6)
    check("dropped seeds listed and counted: 3, 5 (missing), 8, 9 (only a), 11, 12 (only b)",
          res["dropped_seeds"] == [3, 5, 8, 9, 11, 12] and res["n_dropped"] == 6)
    check("breakdown: 2 unpaired a, 2 unpaired b, 2 missing values",
          res["n_unpaired_a"] == 2 and res["n_unpaired_b"] == 2 and res["n_missing_value"] == 2)
    check("n_total counts every entry given, dropped included",
          res["n_total_a"] == 10 and res["n_total_b"] == 10)
    check("kept values belong to their seeds",
          res["values_a"] == [0.0, 1.0, 2.0, 4.0, 6.0, 7.0] and res["values_b"][3] == 4.5)
    check("every kept difference is -0.5", all(d == -0.5 for d in res["differences"]))

    shuffled = paired(va, list(reversed(vb)), seeds_a, list(reversed(seeds_b)), n_boot=500)
    check("input order does not matter - alignment is by seed",
          shuffled["seeds"] == res["seeds"] and shuffled["differences"] == res["differences"])

    check("length mismatch between values and seeds is refused",
          _raises(lambda: align_by_seed([1, 2], [1, 2], [0], [0, 1])))
    check("duplicate seeds are refused (ambiguous alignment)",
          _raises(lambda: align_by_seed([1, 2], [1, 2], [0, 0], [0, 1])))
    check("no overlapping seed -> ValueError, not an empty result",
          _raises(lambda: paired([1, 2], [1, 2], [0, 1], [2, 3])))
    check("a non-numeric value is refused", _raises(lambda: paired(["x"], [1.0], [0], [0])))


# --------------------------------------------------------------------------- #
def check_permuted_pairing_control():
    print("\n-- negative control: permuting the pairing destroys a real paired effect --")
    rng = np.random.default_rng(0)
    seeds = list(range(10))
    b = 10.0 * np.arange(10) + rng.normal(0.0, 0.3, 10)   # strong per-seed level (spread ~90)
    a = b + 0.5                                           # tiny but perfectly consistent shift
    real = paired(a, b, seeds, seeds, n_boot=2000)
    check("real pairing: all 10 differences positive, CI95 excludes 0",
          real["direction"] == "a>b" and real["bootstrap_ci95_excludes_zero"])
    check("real pairing: sign p = Wilcoxon p = 2/1024",
          real["sign_test_p"] == 2 / 1024 and real["wilcoxon_signed_rank_p"] == 2 / 1024)

    perm = np.random.default_rng(1).permutation(10)
    check("the permutation is not the identity", not np.array_equal(perm, np.arange(10)))
    ctrl = paired(a, b[perm], seeds, seeds, n_boot=2000)
    check("permuted pairing: unpaired means unchanged",
          _close(ctrl["mean_a"], real["mean_a"], 1e-9) and _close(ctrl["mean_b"], real["mean_b"], 1e-9))
    check("permuted pairing: robust summary of b unchanged",
          robust(b)["median"] == robust(b[perm])["median"] and robust(b)["mad"] == robust(b[perm])["mad"])
    check("permuted pairing: mean difference itself is unchanged (it only depends on the means)",
          _close(ctrl["mean_diff"], real["mean_diff"], 1e-9))
    check("permuted pairing: CI95 now contains 0",
          ctrl["bootstrap_ci95"][0] < 0.0 < ctrl["bootstrap_ci95"][1]
          and not ctrl["bootstrap_ci95_excludes_zero"])
    check("permuted pairing: signs are mixed, sign test and Wilcoxon are NOT significant",
          ctrl["direction"] == "mixed" and ctrl["sign_test_p"] > 0.05
          and ctrl["wilcoxon_signed_rank_p"] > 0.05)
    check("permuted pairing: |d_z| collapses (real d_z is None: sd 0; control |d_z| < 1)",
          real["cohens_dz"] is None and abs(ctrl["cohens_dz"]) < 1.0)


# --------------------------------------------------------------------------- #
def check_exact_limit_in_paired():
    print("\n-- paired: beyond the exact Wilcoxon limit --")
    seeds = list(range(21))
    b = np.linspace(0.0, 1.0, 21)
    a = b + 1.0
    res = paired(a, b, seeds, seeds, n_boot=500)
    check("21 non-zero pairs: Wilcoxon p is None with the reason stated, nothing guessed",
          res["wilcoxon_signed_rank_p"] is None
          and "exceeds max_exact_n=20" in res["wilcoxon_signed_rank"]["not_computed_reason"])
    check("21 pairs: the sign test is still exact", res["sign_test_p"] == 2 / 2 ** 21)
    res21 = paired(a, b, seeds, seeds, n_boot=500, max_exact_n=21)
    check("raising max_exact_n explicitly computes it: p = 2/2^21",
          res21["wilcoxon_signed_rank_p"] == 2 / 2 ** 21 and res21["params"]["max_exact_n"] == 21)
    one = paired([1.0], [0.5], [0], [0], n_boot=50)
    check("a single pair: statistics defined where possible, d_z None with reason",
          one["n_pairs"] == 1 and one["mean_diff"] == 0.5 and one["cohens_dz"] is None
          and one["sign_test_p"] == 1.0 and one["wilcoxon_signed_rank_p"] == 1.0)


# --------------------------------------------------------------------------- #
def check_plot():
    print("\n-- plot_paired --")
    seeds = list(range(10))
    b = np.linspace(0.2, 0.9, 10)
    a = list(b + 0.1)
    a[4] = None                                        # one incomplete pair is skipped, not fatal
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "paired.png"
        out = plot_paired(a, b, seeds, ("transformer", "mlp"), path, ylabel="metric", title="demo")
        check("plot_paired returns the path and writes a non-empty PNG",
              out == path and path.exists() and path.stat().st_size > 1000)
        check("plot_paired refuses mismatched lengths",
              _raises(lambda: plot_paired(a[:5], b, seeds, ("x", "y"), Path(tmp) / "bad.png")))
        check("plot_paired refuses a label list that is not (label_a, label_b)",
              _raises(lambda: plot_paired(a, b, seeds, ("only",), Path(tmp) / "bad.png")))
        check("plot_paired refuses to draw when no pair is complete",
              _raises(lambda: plot_paired([None] * 10, b, seeds, ("x", "y"), Path(tmp) / "bad.png")))


def main():
    check_wilcoxon_hand_enumerated()
    check_sign_test()
    check_effect_sizes()
    check_robust()
    check_bootstrap()
    check_paired_known_shift()
    check_paired_identical()
    check_unpaired_dropped()
    check_permuted_pairing_control()
    check_exact_limit_in_paired()
    check_plot()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
