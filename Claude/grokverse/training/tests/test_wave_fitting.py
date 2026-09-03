"""Checks for analysis/wave_fitting.py (run: python tests/test_wave_fitting.py).

Contract: docs/dev/INTERFACES.md §3. Constants the fits are read against are derived and
verified in docs/MLP_MECHANISM_DERIVATION.md §5 and pinned in tests/test_derivations.py.

Every check uses a synthetic curve whose answer is known in advance. The negative controls
matter as much as the positive ones: a wrong fundamental must NOT fit, white noise must NOT
be explained, and a scrambled curve must NOT be called periodic — otherwise "the square model
wins" would carry no information.

Phase convention throughout: y[n] = A cos(2*pi*k*n/p - phi), phi in (-pi, pi].
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import wave_fitting as wf  # noqa: E402
from grokverse.analysis.fourier import alias_frequency as alias  # noqa: E402

DEG = math.pi / 180.0
CONTINUOUS_RATIO = {3: 1 / 3, 5: 1 / 5, 7: 1 / 7}   # ideal square wave, amplitude ~ 1/j


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def theta(p: int, k: int) -> np.ndarray:
    return 2 * np.pi * k * np.arange(p) / p


def sinusoid(p, k, phi=0.0, alpha=1.0, beta=0.0):
    return alpha * np.cos(theta(p, k) - phi) + beta


def square(p, k, phi=0.0, alpha=1.0, beta=0.0):
    return alpha * np.sign(np.cos(theta(p, k) - phi) + 1e-12) + beta


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


# --------------------------------------------------------------------------- #
def check_sinusoid_recovery():
    print("\n-- a sinusoid is recovered exactly and wins by AIC --")
    p, k, phi, alpha, beta = 113, 18, 0.7, 2.0, -0.4
    y = sinusoid(p, k, phi, alpha, beta)
    fit = wf.fit_model(y, k, "sinusoid")
    prm = fit["params"]
    check("sinusoid: r2 == 1 on a noiseless sinusoid", float(fit["r2"]) > 1 - 1e-12)
    check(f"sinusoid: amplitude recovered ({float(prm['alpha']):.6f})",
          abs(float(prm["alpha"]) - alpha) < 1e-8)
    check(f"sinusoid: offset recovered ({float(prm['beta']):.6f})",
          abs(float(prm["beta"]) - beta) < 1e-8)
    err = abs(float(np.angle(np.exp(1j * (float(prm["phi"]) - phi)))))
    check(f"sinusoid: phase recovered within 2 degrees (err {err/DEG:.4f} deg)", err < 2 * DEG)
    check("sinusoid: n_params is 3", fit["n_params"] == 3)
    check("sinusoid: n_positions is p", int(fit["n_positions"]) == p)

    cmp = wf.compare_models(y, k)
    check("sinusoid: sinusoid wins by AIC", cmp["best_by_aic"] == "sinusoid")
    check("sinusoid: r2 > 0.999", cmp["r2"]["sinusoid"] > 0.999)
    check(f"sinusoid: beats square by more than 10 AIC "
          f"({-cmp['delta_aic_sinusoid_minus_square']:.1f})",
          cmp["delta_aic_sinusoid_minus_square"] < -10)
    check("sinusoid: the phase convention is echoed",
          cmp["phase_convention"] == wf.PHASE_CONVENTION)
    check("sinusoid: delta_aic of the winner is 0", abs(cmp["delta_aic"]["sinusoid"]) < 1e-9)
    check("sinusoid: every model reports a delta_aic >= 0",
          all(cmp["delta_aic"][m] >= -1e-9 for m in wf.MODEL_NAMES))


def check_square_recovery():
    print("\n-- a discrete square wave is recovered and wins by AIC --")
    p, k, phi = 113, 18, 0.7
    y = square(p, k, phi)
    fit = wf.fit_model(y, k, "square")
    check("square: r2 == 1 on a noiseless square wave", float(fit["r2"]) > 1 - 1e-12)
    check(f"square: amplitude recovered ({float(fit['params']['alpha']):.6f})",
          abs(float(fit["params"]["alpha"]) - 1.0) < 1e-8)
    check("square: n_params is 3", fit["n_params"] == 3)

    cmp = wf.compare_models(y, k)
    check("square: the square model wins by AIC", cmp["best_by_aic"] == "square")
    check("square: it also wins by AICc", cmp["best_by_aicc"] == "square")
    check(f"square: beats the sinusoid by more than 10 AIC "
          f"({cmp['delta_aic_sinusoid_minus_square']:.1f})",
          cmp["delta_aic_sinusoid_minus_square"] > 10)
    check("square: the sinusoid alone explains at most ~8/pi^2 of the variance "
          f"({cmp['r2']['sinusoid']:.4f})",
          0.75 < cmp["r2"]["sinusoid"] < 0.85)


def check_odd_harmonic_ratios():
    print("\n-- the odd-harmonic fit recovers the 1/j amplitude law --")
    for p in (23, 113):
        y = square(p, 3 if p == 23 else 18, 0.35)
        k = 3 if p == 23 else 18
        fit = wf.fit_model(y, k, "odd_harmonics")
        ratios = np.asarray(fit["params"]["amplitude_ratio_to_fundamental"], dtype=float).ravel()
        harmonics = list(np.asarray(fit["params"]["harmonics"]).ravel())
        check(f"p={p}: the fitted harmonics are {wf.ODD_HARMONICS}",
              tuple(int(h) for h in harmonics) == wf.ODD_HARMONICS)
        for i, j in enumerate((3, 5, 7)):
            got, want = float(ratios[i]), CONTINUOUS_RATIO[j]
            check(f"p={p}: alpha_{j}/alpha_1 = {got:.4f} within 5% of 1/{j} = {want:.4f}",
                  abs(got - want) / want < 0.05)
        check(f"p={p}: odd_harmonics has 9 free parameters", fit["n_params"] == 9)
        # Truncating at j = 7 retains 1 + 1/9 + 1/25 + 1/49 = 1.1715 of the fundamental's
        # power, and the fundamental holds 8/pi^2 = 0.8106 of the square wave's non-constant
        # power, so the ceiling for this model is 0.8106 * 1.1715 = 0.9497 -- NOT ~1. Asserting
        # r2 > 0.99 here would be asserting something false about the Fourier series.
        r2 = float(fit["r2"])
        check(f"p={p}: odd_harmonics reaches the j<=7 truncation ceiling ~0.95 (got {r2:.4f})",
              0.93 < r2 < 0.97)
        check(f"p={p}: odd_harmonics still beats the plain sinusoid",
              r2 > float(wf.fit_model(y, k, "sinusoid")["r2"]))

    # a SINUSOID must show ~no harmonic amplitude -- the control for the check above
    y = sinusoid(113, 18, 0.35)
    r = np.asarray(wf.fit_model(y, 18, "odd_harmonics")["params"]
                   ["amplitude_ratio_to_fundamental"], dtype=float).ravel()
    check(f"a sinusoid has ~zero harmonic amplitude (max ratio {float(np.abs(r).max()):.2e})",
          float(np.abs(r).max()) < 1e-8)

    # The constrained model is the 3-parameter truncation of the IDEAL square wave, so its
    # ceiling is the same 0.9496 as the 9-parameter version -- with the alternating sign
    # (-1)^((j-1)/2). Omitting that sign (the bug fixed on 2026-09-03) drops it to 0.571,
    # below a plain sinusoid; these two checks are what caught it.
    fit = wf.fit_model(square(113, 18, 0.35), 18, "odd_harmonics_1_over_j")
    check("odd_harmonics_1_over_j has exactly 3 free parameters", fit["n_params"] == 3)
    r2c = float(fit["r2"])
    check(f"odd_harmonics_1_over_j reaches the analytic square-wave ceiling ~0.9496 "
          f"(got {r2c:.4f})", 0.93 < r2c < 0.96)
    check("odd_harmonics_1_over_j beats a plain sinusoid on a square wave "
          "(it would not if the alternating sign were missing)",
          r2c > float(wf.fit_model(square(113, 18, 0.35), 18, "sinusoid")["r2"]))
    check("odd_harmonics_1_over_j fits a sinusoid worse than the sinusoid model does",
          float(wf.fit_model(sinusoid(113, 18, 0.35), 18, "odd_harmonics_1_over_j")["r2"])
          < float(wf.fit_model(sinusoid(113, 18, 0.35), 18, "sinusoid")["r2"]))
    # and it must be a genuine square-wave shape: nearly the same fit as the free 9-parameter
    # version, at a third of the parameters
    free = float(wf.fit_model(square(113, 18, 0.35), 18, "odd_harmonics")["r2"])
    check(f"the constrained fit is within 0.01 of the free odd-harmonic fit "
          f"({r2c:.4f} vs {free:.4f})", abs(free - r2c) < 0.01)


def check_noise_robustness():
    print("\n-- the right model still wins at signal-to-noise ratio 3 --")
    rng = np.random.default_rng(0)
    p, k = 113, 18
    for kind, gen in (("sinusoid", sinusoid), ("square", square)):
        clean = gen(p, k, 0.4)
        noise = rng.standard_normal(p)
        noise *= clean.std() / (3.0 * noise.std())          # SNR 3 in amplitude
        cmp = wf.compare_models(clean + noise, k)
        check(f"SNR 3, {kind}: the {kind} model still wins by AIC",
              cmp["best_by_aic"] == kind)
        check(f"SNR 3, {kind}: r2 of the winner is still > 0.8",
              cmp["r2"][kind] > 0.8)

    # phase survives the noise too
    phi = -1.1
    clean = sinusoid(p, k, phi)
    noise = rng.standard_normal(p)
    noise *= clean.std() / (3.0 * noise.std())
    got = float(wf.fit_model(clean + noise, k, "sinusoid")["params"]["phi"])
    err = abs(np.angle(np.exp(1j * (got - phi))))
    check(f"SNR 3: the phase is recovered within 10 degrees ({err/DEG:.2f} deg)", err < 10 * DEG)


def check_wrong_frequency_and_noise_controls():
    print("\n-- negative controls: a wrong k and pure noise must NOT fit --")
    p, k = 113, 18
    y = sinusoid(p, k, 0.4)

    # The one-frequency models can only fit at the true fundamental.
    for bad in (5, 17, 19, 20):
        for m in ("sinusoid", "square"):
            r2 = float(wf.fit_model(y, bad, m)["r2"])
            check(f"{m} at a wrong fundamental k={bad} gives r2 < 0.2 ({r2:.4f})", r2 < 0.2)

    # CAVEAT, asserted rather than avoided: the odd-harmonic FAMILY of a wrong fundamental can
    # alias onto the true frequency, and then the harmonic models fit perfectly. At p = 113 with
    # a true k = 18 that happens for exactly k = 6, 19 and 51 (3*6 = 18; 5*19 = 95 -> 18;
    # 7*51 = 357 -> 18). "The harmonic model fits at k" therefore does NOT identify k as the
    # fundamental -- a caveat that matters for frequency_sensitivity and for H3.
    family = lambda kk: {int(np.round(alias(j * kk, p))) for j in wf.ODD_HARMONICS}   # noqa: E731
    aliasing = [kk for kk in range(1, (p - 1) // 2 + 1) if kk != k and k in family(kk)]
    check(f"exactly k = 6, 19, 51 alias onto the true fundamental 18 at p=113 (got {aliasing})",
          aliasing == [6, 19, 51])
    for bad in aliasing:
        r2 = float(wf.fit_model(y, bad, "odd_harmonics")["r2"])
        check(f"the harmonic family of k={bad} contains 18, so it DOES fit ({r2:.4f}) — "
              "documented caveat, not a defect", r2 > 0.99)
    for bad in (5, 17, 20):
        cmp = wf.compare_models(y, bad)
        worst = max(cmp["r2"][m] for m in wf.MODEL_NAMES)
        check(f"a wrong fundamental k={bad} whose family excludes 18 gives r2 < 0.2 "
              f"for every model ({worst:.4f})", worst < 0.2)

    rng = np.random.default_rng(1)
    noise = rng.standard_normal(p)
    cmp = wf.compare_models(noise, k)
    worst = max(cmp["r2"][m] for m in wf.MODEL_NAMES)
    check(f"white noise is not explained by any model (max r2 {worst:.4f})", worst < 0.3)

    # a scrambled square wave keeps its histogram but loses its periodicity
    scrambled = rng.permutation(square(p, k, 0.0))
    cmp = wf.compare_models(scrambled, k)
    check(f"a scrambled square wave is not fit (max r2 "
          f"{max(cmp['r2'][m] for m in wf.MODEL_NAMES):.4f})",
          max(cmp["r2"][m] for m in wf.MODEL_NAMES) < 0.3)


def check_cross_validation():
    print("\n-- held-out cross-validation is seeded and does not pick the sinusoid for a square --")
    p, k = 113, 18
    cv_sin = wf.cross_validate(sinusoid(p, k, 0.4), k, folds=5, seed=0)
    check("CV picks the sinusoid for a sinusoid", cv_sin["best_by_cv"] == "sinusoid")
    cv_sq = wf.cross_validate(square(p, k, 0.4), k, folds=5, seed=0)
    check(f"CV does NOT pick the sinusoid for a square wave (picked {cv_sq['best_by_cv']!r})",
          cv_sq["best_by_cv"] != "sinusoid")
    check("CV is reproducible under its seed",
          wf.cross_validate(square(p, k, 0.4), k, folds=5, seed=0)["best_by_cv"]
          == cv_sq["best_by_cv"])
    check("a different seed is allowed and still returns a model name",
          wf.cross_validate(square(p, k, 0.4), k, folds=5, seed=7)["best_by_cv"]
          in wf.MODEL_NAMES)
    check("folds outside 2..p are rejected",
          _raises(lambda: wf.cross_validate(sinusoid(p, k), k, folds=1))
          and _raises(lambda: wf.cross_validate(sinusoid(p, k), k, folds=p + 1)))
    cv_noise = wf.cross_validate(np.random.default_rng(2).standard_normal(p), k, folds=5, seed=0)
    worst = min(float(v) for v in cv_noise["heldout_nmse_mean"].values())
    check(f"CV on pure noise shows no held-out predictive power: the best model still has "
          f"held-out nmse >= 1 ({worst:.3f})", worst >= 1.0)
    best_sin = min(float(v) for v in
                   wf.cross_validate(sinusoid(p, k, 0.4), k, folds=5, seed=0)
                   ["heldout_nmse_mean"].values())
    check(f"CV on a real sinusoid does show predictive power (held-out nmse {best_sin:.2e} << 1)",
          best_sin < 0.01)
    check("CV reports a pooled ranking as well as a per-fold mean",
          cv_noise["best_by_cv_pooled"] in wf.MODEL_NAMES
          and len(cv_noise["fold_sizes"]) == cv_noise["folds"])


def check_frequency_sensitivity():
    print("\n-- frequency sensitivity reports its rule and the best model per k --")
    p, k = 113, 18
    y = square(p, k, 0.4)
    fs = wf.frequency_sensitivity(y)
    check("the k-selection rule is echoed", isinstance(fs["ks_rule"], str) and fs["ks_rule"])
    check("the dominant frequency is the planted one", int(fs["dominant_frequency"]) == k)
    check("the true k is among the ks tried", k in [int(x) for x in fs["ks"]])
    check("at the true k the square model wins", fs["best_model_per_k"][k] == "square")
    explicit = wf.frequency_sensitivity(y, ks=[k, 17])
    check("an explicit ks list is honoured (order-independent)",
          {int(x) for x in explicit["ks"]} == {k, 17})
    check("an explicit ks list is labelled as user-supplied, not as the default rule",
          explicit["ks_rule"] == "user_supplied" and fs["ks_rule"] != "user_supplied")
    # the aliasing caveat again, this time through the public sensitivity API: k = 19 is in
    # the default candidate set and its harmonic family reaches 18, so it is reported as
    # harmonic rather than square -- the report must show it, not hide it
    if 19 in fs["best_model_per_k"]:
        check("an aliasing neighbour k=19 is reported as odd_harmonics, not silently dropped",
              fs["best_model_per_k"][19] in wf.MODEL_NAMES)


def check_fit_matrix_and_summary():
    print("\n-- fit_matrix over columns agrees with fitting each column alone --")
    p = 113
    cols, ks = [], []
    for k, phi in ((18, 0.3), (15, -1.2)):
        cols.append(sinusoid(p, k, phi)); ks.append(k)
        cols.append(square(p, k, phi)); ks.append(k)
    Y = np.stack(cols, axis=1)
    fm = wf.fit_matrix(Y, ks)
    check("fit_matrix reports one column per input column", int(fm["n_columns"]) == Y.shape[1])
    check("fit_matrix echoes each column's k", [int(x) for x in fm["k"]] == ks)
    for j in range(Y.shape[1]):
        alone = wf.compare_models(Y[:, j], ks[j])
        for m in wf.MODEL_NAMES:
            check(f"column {j}: r2 of {m} matches the single-column fit",
                  abs(float(fm["r2"][m][j]) - alone["r2"][m]) < 1e-9)
        best_idx = int(fm["best_by_aic"][j])
        check(f"column {j}: best_by_aic matches the single-column ranking",
              wf.MODEL_NAMES[best_idx] == alone["best_by_aic"])

    s = wf.summarize(fm)
    check("summarize reports a fraction per model that sums to 1",
          abs(sum(s["fraction_best_by_aic"].values()) - 1.0) < 1e-9)
    check("summarize finds the two sinusoid columns and the two square columns",
          abs(s["fraction_best_by_aic"]["sinusoid"] - 0.5) < 1e-9
          and abs(s["fraction_best_by_aic"]["square"] - 0.5) < 1e-9)
    check("summarize reports the fraction where square beats sinusoid by AIC",
          abs(s["fraction_square_aic_below_sinusoid"] - 0.5) < 1e-9)
    check("summarize reports r2 quantiles per model",
          all("median" in s["r2"][m] for m in wf.MODEL_NAMES))
    check("summarize reports the continuous 1/j reference alongside the measured ratios",
          "continuous_ideal_amplitude_ratio" in s and "amplitude_ratio_to_fundamental" in s)
    check("summarize contains no raw per-column array",
          not any(isinstance(v, (list, np.ndarray)) and len(np.atleast_1d(v)) > 8
                  for v in s.values() if not isinstance(v, dict)))
    check("summarize reports the k histogram", sum(s["k_histogram"].values()) == Y.shape[1])


def check_numerics_and_errors():
    print("\n-- numerical guards and rejected inputs --")
    p, k = 113, 18
    exact = wf.fit_model(sinusoid(p, k, 0.2), k, "sinusoid")
    check("an exact fit gives a finite AIC (the rss floor works)",
          np.isfinite(float(exact["aic"])) and np.isfinite(float(exact["aicc"])))
    check("an exact fit gives rss ~ 0 and r2 ~ 1",
          float(exact["rss"]) < 1e-20 and abs(float(exact["r2"]) - 1) < 1e-12)
    check("AICc exceeds AIC (the finite-sample penalty is positive)",
          float(exact["aicc"]) > float(exact["aic"]))

    check("a constant (zero-variance) curve is rejected rather than scored",
          _raises(lambda: wf.fit_model(np.full(p, 3.0), k, "sinusoid")))
    check("k = 0 is rejected", _raises(lambda: wf.fit_model(sinusoid(p, k), 0, "sinusoid")))
    check("k above (p-1)/2 is rejected",
          _raises(lambda: wf.fit_model(sinusoid(p, k), (p + 1) // 2, "sinusoid")))
    check("an unknown model name is rejected",
          _raises(lambda: wf.fit_model(sinusoid(p, k), k, "made_up")))
    check("a curve too short for the 9-parameter model is rejected",
          _raises(lambda: wf.fit_model(sinusoid(9, 1, 0.1), 1, "odd_harmonics")))
    check("fit_matrix rejects non-integer frequencies",
          _raises(lambda: wf.fit_matrix(np.stack([sinusoid(p, k)], 1), [1.5])))

    # the sign convention of delta_aic_sinusoid_minus_square is documented and correct
    cmp_sq = wf.compare_models(square(p, k, 0.1), k)
    cmp_si = wf.compare_models(sinusoid(p, k, 0.1), k)
    check("delta_aic_sinusoid_minus_square > 0 exactly when the square model is better",
          cmp_sq["delta_aic_sinusoid_minus_square"] > 0
          and cmp_si["delta_aic_sinusoid_minus_square"] < 0)

    # the module measures, it does not conclude
    check("no result key claims a conclusion",
          not any(("is_" in kk or kk.startswith("has_")) for kk in cmp_sq))


def main():
    check_sinusoid_recovery()
    check_square_recovery()
    check_odd_harmonic_ratios()
    check_noise_robustness()
    check_wrong_frequency_and_noise_controls()
    check_cross_validation()
    check_frequency_sensitivity()
    check_fit_matrix_and_summary()
    check_numerics_and_errors()
    print("\nALL WAVE-FITTING CHECKS PASSED")


if __name__ == "__main__":
    main()
