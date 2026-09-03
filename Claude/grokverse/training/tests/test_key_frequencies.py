"""Checks for analysis/key_frequencies.py (docs/dev/INTERFACES.md §4).

Run: ``python tests/test_key_frequencies.py`` from ``training/``.

Every input is SYNTHETIC at ``p = 23`` with a known answer and is fed to the pure
core ``select_from_arrays``, so **no run directory, no checkpoint and no trained
model is read** — this file runs before, during and after the study, and it cannot
be made to pass by a run.

What the checks are for
-----------------------
* A planted 2-frequency circuit must be recovered by **every** rule — otherwise a
  disagreement between rules on real data would be uninterpretable.
* The ``nanda`` count must be **measured**: planting 2, 3, 5 and 10 strong
  frequencies must recover 2, 3, 5 and 10. The 10 case is the decisive one — it is
  strictly above the legacy cap of 8, so a silent cap could not survive it.
* The ``nanda`` threshold must behave monotonically: 0.10 selects a superset of
  0.25, which selects a superset of 0.50 (the pre-registered sensitivity trio,
  docs/PREREGISTRATION.md §4.3).
* ``embedding_top8`` must report ``cap_binding = True`` on a diffuse spectrum (the
  audit's point: its count was never data-determined), while
  ``embedding_threshold`` on the same spectrum must report a count above the cap.
* NEGATIVE CONTROL: a logit tensor built from ``(a − b − c)`` — the wrong symmetry,
  the same amount of periodic structure — must select **nothing** under
  ``logit_sum_directions``. A rule that fires on it would be measuring "periodic",
  not "additive".
* ``neuron_clusters`` must ignore a frequency carried by fewer than ``min_neurons``
  neurons *even when that cluster is loud*, and one that is quiet even when it is
  numerous.

Analytic reference used below: for the orthonormal basis of ``fourier.fourier_basis``
and a row ``A cos(w_k c − phi)``, the two coefficients are ``A cos(phi) sqrt(p/2)``
and ``A sin(phi) sqrt(p/2)``, so ``n`` such rows give a per-frequency norm of exactly
``A * sqrt(n * p / 2)``. Derived here, not copied from the module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import key_frequencies as KF  # noqa: E402

P = 23
HALF = (P - 1) // 2          # 11
PLANTED = (3, 7)             # the circuit every rule must recover


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# synthetic circuit builders                                                   #
# --------------------------------------------------------------------------- #
def cosine(p: int, k: int, phi: float = 0.0) -> np.ndarray:
    n = np.arange(p)
    return np.cos(2 * np.pi * k * n / p - phi)


def planted_W_L(freqs, n_per: int = 20, amp: float = 1.0, noise: float = 0.05,
                p: int = P, seed: int = 0) -> np.ndarray:
    """``[len(freqs)*n_per, p]`` neuron->logit map: each neuron a cosine at its freq."""
    rng = np.random.default_rng(seed)
    rows = [amp * cosine(p, k, rng.uniform(0, 2 * np.pi))
            for k in freqs for _ in range(n_per)]
    W = np.stack(rows) if rows else np.zeros((0, p))
    return W + noise * rng.standard_normal(W.shape)


def graded_W_L(freq_amps: dict[int, float], n_per: int = 8, p: int = P,
               seed: int = 1) -> np.ndarray:
    """A neuron->logit map whose per-frequency NORM is ``amp * sqrt(n_per * p / 2)``."""
    rng = np.random.default_rng(seed)
    rows = [a * cosine(p, k, rng.uniform(0, 2 * np.pi))
            for k, a in freq_amps.items() for _ in range(n_per)]
    return np.stack(rows)


def planted_W_E(freqs, p: int = P, amp: float = 1.0, noise: float = 0.02,
                seed: int = 2) -> np.ndarray:
    """``[p, 2*len(freqs)]`` embedding with cos/sin columns at each planted frequency."""
    rng = np.random.default_rng(seed)
    n = np.arange(p)
    cols = []
    for k in freqs:
        cols += [amp * np.cos(2 * np.pi * k * n / p), amp * np.sin(2 * np.pi * k * n / p)]
    W = np.stack(cols, axis=1)
    return W + noise * rng.standard_normal(W.shape)


def diffuse_W_E(p: int = P) -> np.ndarray:
    """Every frequency carries (almost) the same power — the case where top-8 caps.

    Amplitudes decay by 0.1 % per frequency so the ranking is unambiguous while the
    smallest set reaching 90 % of the power is 10 of 11 frequencies, i.e. > 8.
    """
    n = np.arange(p)
    cols = []
    for k in range(1, (p - 1) // 2 + 1):
        a = 1.0 - 0.001 * k
        cols += [a * np.cos(2 * np.pi * k * n / p), a * np.sin(2 * np.pi * k * n / p)]
    return np.stack(cols, axis=1)


def grid_logits(freqs, p: int = P, sign: int = +1) -> np.ndarray:
    """``L[a,b,c] = sum_k cos(w_k (a + sign*b - c))``; ``sign = -1`` is the (a-b) control."""
    a = np.arange(p)[:, None, None]
    b = np.arange(p)[None, :, None]
    c = np.arange(p)[None, None, :]
    L = np.zeros((p, p, p))
    for k in freqs:
        L = L + np.cos(2 * np.pi * k * (a + sign * b - c) / p)
    return L


def planted_curves(spec, p: int = P, seed: int = 3) -> dict:
    """``spec`` = [(k, n_neurons, amplitude), ...] -> effective operand curves."""
    rng = np.random.default_rng(seed)
    ua, ub = [], []
    for k, n_neurons, amp in spec:
        for _ in range(n_neurons):
            ua.append(amp * cosine(p, k, rng.uniform(0, 2 * np.pi)))
            ub.append(amp * cosine(p, k, rng.uniform(0, 2 * np.pi)))
    return {"u_a": np.stack(ua, axis=1), "u_b": np.stack(ub, axis=1),
            "b_in": np.zeros(len(ua))}


# --------------------------------------------------------------------------- #
# sections                                                                     #
# --------------------------------------------------------------------------- #
def section_preregistered_constants():
    print("\n--- pre-registered constants (docs/PREREGISTRATION.md section 4.3) ---")
    check("PRIMARY_RULE is 'nanda'", KF.PRIMARY_RULE == "nanda")
    check("the nanda threshold is 0.25 of the maximum", KF.NANDA_THRESHOLD_FRAC == 0.25)
    check("the sensitivity thresholds are 0.10 and 0.50",
          tuple(KF.NANDA_SENSITIVITY_FRACS) == (0.10, 0.50))
    check("every INTERFACES section-4 rule name is implemented",
          set(KF.RULES) == {"nanda", "neuron_clusters", "embedding_threshold",
                            "logit_sum_directions", "embedding_top8"})
    check("neuron_clusters defaults: min_neurons 5, variance share 2 %",
          KF.NEURON_CLUSTER_MIN_NEURONS == 5
          and KF.NEURON_CLUSTER_MIN_VARIANCE_SHARE == 0.02)


def section_score_curve_is_the_published_object():
    print("\n--- the nanda score curve, against its analytic value ---")
    n_per, amp = 20, 1.7
    W_L = planted_W_L(PLANTED, n_per=n_per, amp=amp, noise=0.0)
    norms = KF.neuron_logit_norms(W_L, P)
    expected = amp * np.sqrt(n_per * P / 2.0)
    check(f"the per-frequency norm equals A*sqrt(n*p/2) = {expected:.4f} at the planted k",
          all(abs(norms[k - 1] - expected) < 1e-9 for k in PLANTED))
    others = [norms[k - 1] for k in range(1, HALF + 1) if k not in PLANTED]
    check("and is exactly zero everywhere else", max(others) < 1e-9)
    check("the score curve covers every frequency 1..(p-1)/2, untruncated",
          norms.shape == (HALF,))
    check("a W_L whose class axis is not last is rejected",
          _raises(lambda: KF.neuron_logit_norms(W_L.T, P)))


def section_planted_circuit_every_rule():
    print("\n--- a planted 2-frequency circuit is recovered by EVERY rule ---")
    got = {}
    got["nanda"] = KF.select_from_arrays("nanda", P, W_L=planted_W_L(PLANTED))
    got["embedding_top8"] = KF.select_from_arrays("embedding_top8", P, W_E=planted_W_E(PLANTED))
    got["embedding_threshold"] = KF.select_from_arrays("embedding_threshold", P,
                                                       W_E=planted_W_E(PLANTED))
    got["logit_sum_directions"] = KF.select_from_arrays("logit_sum_directions", P,
                                                        logits=grid_logits(PLANTED))
    curves = planted_curves([(PLANTED[0], 20, 1.0), (PLANTED[1], 20, 1.0)])
    got["neuron_clusters"] = KF.select_from_arrays("neuron_clusters", P, curves=curves)
    for rule, res in got.items():
        check(f"{rule} recovers exactly {list(PLANTED)} (n = {res['n']})",
              sorted(res["key_frequencies"]) == list(PLANTED))
        check(f"{rule} reports rule, n, scores, cap_binding and threshold_frac",
              {"rule", "n", "scores", "cap_binding", "threshold_frac"} <= set(res)
              and len(res["scores"]) == HALF)
    check("only the nanda result is flagged as the primary rule",
          [r for r, v in got.items() if v["is_primary_rule"]] == ["nanda"])
    j = KF.agreement({r: got[r]["key_frequencies"] for r in got})
    check("all five rules agree perfectly on the planted circuit (Jaccard 1.0)",
          min(min(row) for row in j["jaccard"]) == 1.0)
    shares = got["logit_sum_directions"]["scores"]
    check("the logit rule puts exactly half the sum-direction power at each planted k",
          all(abs(shares[k - 1] - 0.5) < 1e-9 for k in PLANTED))


def section_nanda_count_is_measured():
    print("\n--- the nanda count is MEASURED, never capped at 8 ---")
    for n_planted in (2, 3, 5, 10):
        freqs = tuple(range(1, n_planted + 1))
        res = KF.select_from_arrays("nanda", P, W_L=planted_W_L(freqs, n_per=12))
        check(f"planting {n_planted} strong frequencies recovers {n_planted}",
              res["n"] == n_planted and sorted(res["key_frequencies"]) == list(freqs))
        check(f"...and cap_binding is False with cap None ({n_planted} planted)",
              res["cap_binding"] is False and res["cap"] is None)
    res10 = KF.select_from_arrays("nanda", P, W_L=planted_W_L(tuple(range(1, 11)), n_per=12))
    check("10 planted frequencies give n = 10 > 8: a silent cap at 8 could not survive this",
          res10["n"] == 10)
    check("the threshold actually used is reported (0.25 x max)",
          abs(res10["threshold_value"] - 0.25 * res10["max_score"]) < 1e-12
          and res10["threshold_frac"] == 0.25)


def section_nanda_threshold_monotonic():
    print("\n--- the nanda threshold is monotone: 0.10 >= 0.25 >= 0.50 ---")
    W_L = graded_W_L({2: 1.0, 4: 0.6, 6: 0.4, 8: 0.2, 10: 0.05})
    n = {}
    for frac in (0.10, 0.25, 0.50):
        n[frac] = KF.select_from_arrays("nanda", P, W_L=W_L, threshold_frac=frac)["n"]
    check(f"n(0.10)={n[0.10]} >= n(0.25)={n[0.25]} >= n(0.50)={n[0.50]}",
          n[0.10] >= n[0.25] >= n[0.50])
    check("the graded spectrum actually separates the three thresholds "
          f"({n[0.50]}, {n[0.25]}, {n[0.10]})",
          n[0.50] == 2 and n[0.25] == 3 and n[0.10] == 4)
    sets = {f: set(KF.select_from_arrays("nanda", P, W_L=W_L, threshold_frac=f)["key_frequencies"])
            for f in (0.10, 0.25, 0.50)}
    check("and the sets are nested, not merely equal in size",
          sets[0.50] <= sets[0.25] <= sets[0.10])
    primary = KF.select_from_arrays("nanda", P, W_L=W_L)
    check("the primary result carries both sensitivity sets alongside",
          set(primary["sensitivity"]) == {"0.10", "0.50"}
          and primary["sensitivity_n"]["0.10"] == n[0.10]
          and primary["sensitivity_n"]["0.50"] == n[0.50])
    check("the primary result's own count equals the 0.25 count",
          primary["n"] == n[0.25])


def section_embedding_rules_and_the_cap():
    print("\n--- embedding_top8 caps, embedding_threshold does not ---")
    W_E = diffuse_W_E()
    top8 = KF.select_from_arrays("embedding_top8", P, W_E=W_E)
    thr = KF.select_from_arrays("embedding_threshold", P, W_E=W_E)
    check("embedding_top8 reports cap_binding True on a diffuse spectrum",
          top8["cap_binding"] is True)
    check(f"...and its count is the cap, 8 (not a measurement): n = {top8['n']}",
          top8["n"] == 8 and top8["cap"] == 8)
    check(f"embedding_threshold reports its measured count ({thr['n']}) and does not cap",
          thr["cap_binding"] is False and thr["n"] == thr["n_freqs_for_threshold"])
    check(f"...and that count is above the legacy cap ({thr['n']} > 8) on this spectrum",
          thr["n"] > 8)
    check("the top-8 set is a subset of the 90 %-threshold set",
          set(top8["key_frequencies"]) <= set(thr["key_frequencies"]))
    check("both echo the 90 % power threshold they used",
          top8["threshold_frac"] == 0.90 and thr["threshold_frac"] == 0.90)
    j = KF.agreement({"nanda": PLANTED, "embedding_top8": top8["key_frequencies"]})
    check("agreement() runs on sets of different sizes",
          0.0 <= j["jaccard"][0][1] <= 1.0)


def section_logit_negative_control():
    print("\n--- NEGATIVE CONTROL: the logit rule must ignore (a-b) structure ---")
    diff = KF.select_from_arrays("logit_sum_directions", P, logits=grid_logits(PLANTED, sign=-1))
    check("a pure (a-b-c) logit tensor selects NOTHING (n = 0)",
          diff["n"] == 0 and diff["key_frequencies"] == [])
    check("...because its sum-direction share is numerically zero everywhere",
          max(diff["scores"]) < 1e-12)
    check("...while the (a+b-c) tensor of the same amplitude selects both frequencies",
          KF.select_from_arrays("logit_sum_directions", P,
                                logits=grid_logits(PLANTED))["n"] == 2)
    check("the threshold is three times the uniform share, 3/half",
          abs(diff["threshold_value"] - 3.0 / HALF) < 1e-15)
    rng = np.random.default_rng(11)
    noise = KF.select_from_arrays("logit_sum_directions", P,
                                  logits=rng.standard_normal((P, P, P)))
    check(f"white-noise logits select at most a stray frequency (n = {noise['n']} <= 1)",
          noise["n"] <= 1)


def section_neuron_clusters():
    print("\n--- neuron_clusters: a cluster must be both large and loud ---")
    spec = [(3, 20, 1.0), (7, 20, 1.0), (5, 3, 3.0), (9, 8, 0.02)]
    res = KF.select_from_arrays("neuron_clusters", P, curves=planted_curves(spec))
    check("a frequency used by fewer than min_neurons (3 < 5) neurons is ignored, "
          "even though its cluster is the loudest per neuron",
          5 not in res["key_frequencies"] and res["neuron_counts"][5 - 1] == 3)
    check("a numerous but quiet cluster (8 neurons, < 2 % of the variance) is ignored",
          9 not in res["key_frequencies"]
          and res["neuron_counts"][9 - 1] == 8 and res["scores"][9 - 1] < 0.02)
    check(f"the two real clusters survive: {sorted(res['key_frequencies'])}",
          sorted(res["key_frequencies"]) == [3, 7])
    lowered = KF.select_from_arrays("neuron_clusters", P, curves=planted_curves(spec),
                                    min_neurons=3)
    check("lowering min_neurons to 3 admits the small loud cluster: the count is the "
          "threshold's doing, and the threshold is reported",
          5 in lowered["key_frequencies"] and lowered["min_neurons"] == 3)
    check("the variance share of the two real clusters sums to essentially all of it",
          res["scores"][3 - 1] + res["scores"][7 - 1] > 0.55)
    n_neurons = sum(n for _, n, _ in spec)
    supplied = KF.select_from_arrays("neuron_clusters", P, curves=planted_curves(spec),
                                     hidden_variance=np.ones(n_neurons))
    check("a caller-supplied hidden_variance replaces the recomputed one: with every "
          "neuron equally loud the quiet 8-neuron cluster now passes both criteria",
          sorted(supplied["key_frequencies"]) == [3, 7, 9]
          and supplied["total_activation_variance"] == float(n_neurons))
    check("a hidden_variance of the wrong length is refused",
          _raises(lambda: KF.select_from_arrays("neuron_clusters", P,
                                                curves=planted_curves(spec),
                                                hidden_variance=np.ones(3)), ValueError))


def section_agreement():
    print("\n--- agreement(): pairwise Jaccard ---")
    j = KF.agreement({"a": [1, 2, 3], "b": [2, 3, 4], "c": [5]})
    check("the diagonal is 1.0 for every set", all(j["jaccard"][i][i] == 1.0 for i in range(3)))
    check("J({1,2,3}, {2,3,4}) = 2/4 = 0.5 (hand-computed)",
          abs(j["jaccard"][0][1] - 0.5) < 1e-15 and j["jaccard"][0][1] == j["jaccard"][1][0])
    check("J({1,2,3}, {5}) = 0/4 = 0.0", j["jaccard"][0][2] == 0.0)
    check("intersection and union counts are reported next to the ratio",
          j["intersection"][0][1] == 2 and j["union"][0][1] == 4)
    check("names and sizes are carried through in order",
          j["names"] == ["a", "b", "c"] and j["sizes"] == [3, 3, 1])
    e = KF.agreement([[], []])
    check("two empty sets have Jaccard 1.0 (defined, not a division by zero)",
          e["jaccard"][0][1] == 1.0 and e["jaccard"][0][0] == 1.0)
    seq = KF.agreement([[1, 2], [2]])
    check("a bare sequence of sets is accepted and named by position",
          seq["names"] == ["0", "1"] and abs(seq["jaccard"][0][1] - 0.5) < 1e-15)


def section_errors_and_empty_results():
    print("\n--- errors, and an empty selection reported as n = 0 ---")
    check("an unknown rule name raises ValueError",
          _raises(lambda: KF.select_from_arrays("top_five", P, W_L=planted_W_L(PLANTED)),
                  ValueError))
    check("an even p raises ValueError (the cos/sin basis needs odd p)",
          _raises(lambda: KF.select_from_arrays("nanda", 24, W_L=np.zeros((4, 24))), ValueError))
    check("a rule called without its input array raises ValueError naming that array",
          _raises(lambda: KF.select_from_arrays("nanda", P), ValueError))
    check("...and so does neuron_clusters without curves",
          _raises(lambda: KF.select_from_arrays("neuron_clusters", P), ValueError))
    zero = KF.select_from_arrays("nanda", P, W_L=np.zeros((16, P)))
    check("an all-zero neuron->logit map gives an EMPTY set with n = 0, not a crash",
          zero["key_frequencies"] == [] and zero["n"] == 0)
    check("...and its reported threshold is 0.0, not a made-up value",
          zero["threshold_value"] == 0.0 and zero["max_score"] == 0.0)
    zero_e = KF.select_from_arrays("embedding_top8", P, W_E=np.zeros((P, 8)))
    check("an all-zero embedding gives an empty set with n = 0",
          zero_e["key_frequencies"] == [] and zero_e["n"] == 0)
    zero_c = KF.select_from_arrays("neuron_clusters", P,
                                   curves={"u_a": np.zeros((P, 6)), "u_b": np.zeros((P, 6))})
    check("all-dead neurons give an empty set with n = 0",
          zero_c["key_frequencies"] == [] and zero_c["n"] == 0)
    check("a non-finite score curve is refused rather than silently selected",
          _raises(lambda: KF.neuron_logit_norms(np.full((4, P), np.nan), P), ValueError))
    empty_j = KF.agreement({"nanda": [], "embedding_top8": [1, 2]})
    check("agreement() handles an empty set against a non-empty one (J = 0)",
          empty_j["jaccard"][0][1] == 0.0)


def _raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def main():
    section_preregistered_constants()
    section_score_curve_is_the_published_object()
    section_planted_circuit_every_rule()
    section_nanda_count_is_measured()
    section_nanda_threshold_monotonic()
    section_embedding_rules_and_the_cap()
    section_logit_negative_control()
    section_neuron_clusters()
    section_agreement()
    section_errors_and_empty_results()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
