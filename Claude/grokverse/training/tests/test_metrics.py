"""Checks for analysis/metrics.py (docs/dev/INTERFACES.md §2).

Run: ``python tests/test_metrics.py`` from ``training/``.

Every input is SYNTHETIC with a known answer — no trained run is read, so this file can run
before, during and after the study. The reference numbers are the ones verified in
``docs/MLP_MECHANISM_DERIVATION.md`` §5 (and re-derived independently in
``tests/test_derivations.py``); the source-borrowed definitions are checked against
``docs/sources/doshi2023_grok_or_not.md`` §2.1/§6.1 (IPR) and
``docs/sources/swaroop2026_relu_mlp_square_waves.md`` §4 (periodicity score, Eq. 1).

Two places where the honest answer differs from the one-line expectation in INTERFACES §2,
both checked here in the form that is actually true and flagged in the check names:

* **white noise, participation ratio.** A *single* white-noise periodogram is exponentially
  distributed, not flat, so ``PR ~ half/2`` and ``H ~ 1 - (1-gamma)/log(half) = 0.895`` —
  not ``PR ~ half`` / ``H ~ 1``. The ENSEMBLE-MEAN spectrum of white noise is flat and does
  give ``PR -> half``, ``H -> 1``; both statements are asserted, with the analytic values.
  ``metrics.spectral_entropy``'s own docstring states this.
* **zero-power column.** The module deliberately returns ``NaN`` (documented: "never a
  substitute value") and counts the column in ``n_undefined_columns``. The check therefore
  asserts no crash, NaN confined to that column, and that a good column in the same matrix
  keeps finite metrics — not the absence of NaN.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import metrics as M  # noqa: E402
from grokverse.analysis.fourier import alias_frequency  # noqa: E402
from grokverse.analysis.mlp_mechanism import curve_spectra  # noqa: E402

P = 113
HALF = (P - 1) // 2
EULER_GAMMA = 0.5772156649015329

#: docs/MLP_MECHANISM_DERIVATION.md §5, verified table (p = 113, sign(cos(2 pi k n / p))).
DISCRETE_SQUARE_FUNDAMENTAL_SHARE = 0.8107      # of total non-constant power
DISCRETE_SQUARE_ODD_TO_FUND = 0.1717            # (j = 3,5,7) / fundamental
DISCRETE_SQUARE_EVEN_TO_FUND = 5.8e-4           # (j = 2,4,6) / fundamental
CONTINUOUS_FUNDAMENTAL_SHARE = 8.0 / math.pi ** 2       # 0.81057
CONTINUOUS_ODD_TO_FUND = 1 / 9 + 1 / 25 + 1 / 49        # 0.17151
#: docs/sources/swaroop2026_relu_mlp_square_waves.md §4, own-computation line for p = 113.
SWAROOP_SQUARE_SCORE_P113 = 17.991


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# synthetic curves                                                             #
# --------------------------------------------------------------------------- #
def sinusoid(p=P, k=5, phi=0.7, amp=1.7, offset=0.0) -> np.ndarray:
    n = np.arange(p)
    return amp * np.cos(2 * np.pi * k * n / p - phi) + offset


def discrete_square(p=P, k=5, phi=0.7, amp=1.0, offset=0.0) -> np.ndarray:
    n = np.arange(p)
    return amp * np.where(np.cos(2 * np.pi * k * n / p - phi) >= 0, 1.0, -1.0) + offset


def white_noise(p=P, n_cols=400, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((p, n_cols))


def power_of(curve, p=P) -> np.ndarray:
    """Half-spectrum power of a curve; 1-D curve -> 1-D power (so the metrics return scalars).

    ``power_spectrum`` always returns ``[half, n]`` (a 1-D curve becomes one column), while
    the metrics squeeze on a 1-D POWER array — so a single curve must be unwrapped here.
    """
    power = M.power_spectrum(curve, p)["power"]
    return power[:, 0] if np.ndim(curve) == 1 else power


# --------------------------------------------------------------------------- #
# 1. conventions: the spectrum this module measures                            #
# --------------------------------------------------------------------------- #
def section_spectrum_convention():
    print("\n-- power_spectrum: same object and phase convention as curve_spectra --")
    Y = np.column_stack([sinusoid(k=5, phi=0.7), discrete_square(k=18, phi=-1.2),
                         white_noise(n_cols=1, seed=3)[:, 0]])
    mine = M.power_spectrum(Y, P)
    theirs = curve_spectra(Y, P)
    check("power equals mlp_mechanism.curve_spectra's power (<1e-12)",
          np.abs(mine["power"] - theirs["power"]).max() < 1e-12)
    check("phase equals curve_spectra's phase (<1e-12)",
          np.abs(mine["phase"] - theirs["phase"]).max() < 1e-12)
    check("dominant_freq equals curve_spectra's dominant_freq",
          np.array_equal(mine["dominant_freq"], theirs["dominant_freq"]))

    k, phi = 7, -2.1
    spec = M.power_spectrum(sinusoid(k=k, phi=phi, amp=0.9), P)
    check(f"phase convention y = A cos(2 pi k n / p - phi): recovers k={k}",
          int(spec["dominant_freq"][0]) == k)
    err = abs(float(np.angle(np.exp(1j * (spec["dominant_phase"][0] - phi)))))
    check(f"phase convention recovers phi={phi} (err {err:.2e} rad)", err < 1e-10)

    check("power_spectrum rejects a curve whose length is not p",
          _raises(lambda: M.power_spectrum(np.zeros(P - 2), P), ValueError))


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# --------------------------------------------------------------------------- #
# 2. injected pure sinusoid                                                    #
# --------------------------------------------------------------------------- #
def section_injected_sinusoid():
    print("\n-- injected pure sinusoid: entropy ~ 0, PR ~ 1, top-1 = 1 --")
    for k, phi, amp in ((5, 0.7, 1.7), (18, -2.4, 0.3), (56, 0.0, 12.0)):
        power = power_of(sinusoid(k=k, phi=phi, amp=amp))
        H = M.spectral_entropy(power)
        pr = M.participation_ratio(power)
        top1 = M.topk_concentration(power, 1)
        check(f"k={k}: spectral entropy ~ 0 (got {H:.2e})", H < 1e-12)
        check(f"k={k}: participation ratio ~ 1 (got {pr:.12f})", abs(pr - 1.0) < 1e-9)
        check(f"k={k}: top-1 concentration = 1 (got {top1:.12f})", abs(top1 - 1.0) < 1e-9)
        check(f"k={k}: dominant_fraction = 1", abs(M.dominant_fraction(power) - 1.0) < 1e-9)
        check(f"k={k}: dominant_frequency = {k}", M.dominant_frequency(power) == k)

    power = power_of(sinusoid(k=5, offset=3.0))
    check("a constant offset does not enter the (non-constant) power spectrum",
          abs(M.participation_ratio(power) - 1.0) < 1e-9)
    check("topk_concentration is monotone non-decreasing in k",
          M.topk_concentration(power_of(white_noise(n_cols=1)[:, 0]), 1)
          <= M.topk_concentration(power_of(white_noise(n_cols=1)[:, 0]), 4))
    check("topk_concentration rejects k outside 1..half",
          _raises(lambda: M.topk_concentration(power, HALF + 1), ValueError))


# --------------------------------------------------------------------------- #
# 3. white noise                                                               #
# --------------------------------------------------------------------------- #
def section_white_noise():
    print("\n-- white noise: entropy near 1, participation ratio near the frequency count --")
    W = white_noise(n_cols=400, seed=0)
    power = M.power_spectrum(W, P)["power"]
    H = M.spectral_entropy(power)
    pr = M.participation_ratio(power)

    # Single realization: the periodogram is exponential, NOT flat (module docstring).
    analytic_H = 1.0 - (1.0 - EULER_GAMMA) / math.log(HALF)          # 0.8950
    check(f"single-realization entropy matches the analytic exponential-periodogram value "
          f"{analytic_H:.4f} (got {H.mean():.4f})", abs(H.mean() - analytic_H) < 0.01)
    check(f"single-realization entropy is near 1 and far above a sinusoid's "
          f"(got {H.mean():.3f})", 0.85 < H.mean() < 1.0)
    check("every noise column has entropy > 0.8", H.min() > 0.8)
    check(f"single-realization PR ~ half/2 = {HALF / 2:.1f} (got {pr.mean():.1f}), the "
          f"exponential-periodogram value", abs(pr.mean() - HALF / 2) < 0.1 * HALF)
    check(f"single-realization PR is an order of magnitude above a sinusoid's 1 "
          f"(got min {pr.min():.1f})", pr.min() > 10.0)

    # Ensemble-mean spectrum IS flat -> the INTERFACES §2 statement in its exact form.
    mean_spectrum = power.mean(axis=1)
    pr_mean = M.participation_ratio(mean_spectrum)
    H_mean = M.spectral_entropy(mean_spectrum)
    check(f"ensemble-mean noise spectrum: PR ~ half = {HALF} (got {pr_mean:.2f})",
          abs(pr_mean - HALF) < 0.5)
    check(f"ensemble-mean noise spectrum: entropy ~ 1 (got {H_mean:.5f})",
          abs(H_mean - 1.0) < 1e-3)

    check("noise top-1 concentration is small (< 0.25)",
          M.topk_concentration(power, 1).max() < 0.25)
    check("entropy needs at least 2 frequencies",
          _raises(lambda: M.spectral_entropy(np.array([[1.0]])), ValueError))
    check("negative power is rejected",
          _raises(lambda: M.spectral_entropy(-np.ones((4, 2))), ValueError))


# --------------------------------------------------------------------------- #
# 4. the CONTINUOUS ideal square wave                                          #
# --------------------------------------------------------------------------- #
def section_continuous_square_wave():
    print("\n-- continuous ideal square wave: odd-harmonic power -> 1/9, 1/25, 1/49 --")
    ideal = M.CONTINUOUS_SQUARE_POWER_RATIO
    check("CONTINUOUS_SQUARE_POWER_RATIO is the 1/j^2 law",
          ideal == {3: 1 / 9, 5: 1 / 25, 7: 1 / 49})
    check("CONTINUOUS_SQUARE_AMPLITUDE_RATIO is the 1/j law",
          M.CONTINUOUS_SQUARE_AMPLITUDE_RATIO == {3: 1 / 3, 5: 1 / 5, 7: 1 / 7})

    # Sampling sign(cos) ever more finely approaches the continuous Fourier series.
    errs = {}
    for p in (113, 1009, 4001, 20011):
        n = np.arange(p)
        y = np.where(np.cos(2 * np.pi * n / p) >= 0, 1.0, -1.0)
        q = M.power_spectrum(y, p)["power"][:, 0]
        errs[p] = {j: abs(q[j - 1] / q[0] - ideal[j]) for j in (3, 5, 7)}
        got = {j: q[j - 1] / q[0] for j in (3, 5, 7)}
        check(f"p={p}: power(3k,5k,7k)/power(k) = "
              f"{got[3]:.5f}, {got[5]:.5f}, {got[7]:.5f} vs 1/9, 1/25, 1/49 (within 1%)",
              all(errs[p][j] < 0.01 * ideal[j] for j in (3, 5, 7)))
        fund = q[0] / q.sum()
        check(f"p={p}: fundamental share of non-constant power ~ 8/pi^2 (got {fund:.4f})",
              abs(fund - CONTINUOUS_FUNDAMENTAL_SHARE) < 0.005)
    check("the ratios converge to the continuous law as p grows (p=20011 beats p=113)",
          all(errs[20011][j] < errs[113][j] for j in (3, 5, 7)))

    n = np.arange(20011)
    y = np.where(np.cos(2 * np.pi * n / 20011) >= 0, 1.0, -1.0)
    q = M.power_spectrum(y, 20011)["power"][:, 0]
    even = sum(q[j - 1] for j in (2, 4, 6)) / q[0]
    check(f"a fine-grained square wave has ~no even-harmonic power (got {even:.2e})",
          even < 1e-6)


# --------------------------------------------------------------------------- #
# 5. discrete_square_reference: reproduces itself and §5 of the derivation      #
# --------------------------------------------------------------------------- #
def section_discrete_square_reference():
    print("\n-- discrete_square_reference at p=113 (docs/MLP_MECHANISM_DERIVATION.md §5) --")
    for k in (1, 5, 18, 56):
        for phi in (0.0, 0.7, -2.3):
            ref = M.discrete_square_reference(P, k, phi)

            # (a) the reference reproduces ITSELF: measuring its own curve returns it.
            power = power_of(ref["curve"])
            shares = M.harmonic_shares(power, k, P, reference_phi=phi)
            same = all(abs(shares[key] - ref[key]) < 1e-12 for key in
                       ("fundamental_share", "odd_share", "even_share", "odd_minus_even"))
            check(f"k={k}, phi={phi}: discrete_square_reference reproduces itself", same)
            check(f"k={k}, phi={phi}: ideal_square_* of harmonic_shares equals the reference",
                  all(abs(shares[f"ideal_square_{key}"] - ref[key]) < 1e-12 for key in
                      ("fundamental_share", "odd_share", "even_share", "odd_minus_even")))

            # (b) the derivation's verified numbers, to 2 decimal places as constants.
            check(f"k={k}, phi={phi}: fundamental share = {ref['fundamental_share']:.4f} "
                  f"vs the module's own {DISCRETE_SQUARE_FUNDAMENTAL_SHARE} (2 dp)",
                  round(ref["fundamental_share"], 2)
                  == round(DISCRETE_SQUARE_FUNDAMENTAL_SHARE, 2))
            check(f"k={k}, phi={phi}: odd/fundamental = "
                  f"{ref['odd_to_fundamental_ratio']:.4f} vs {DISCRETE_SQUARE_ODD_TO_FUND} (2 dp)",
                  round(ref["odd_to_fundamental_ratio"], 2)
                  == round(DISCRETE_SQUARE_ODD_TO_FUND, 2))
            check(f"k={k}, phi={phi}: even/fundamental = "
                  f"{ref['even_to_fundamental_ratio']:.2e} ~ {DISCRETE_SQUARE_EVEN_TO_FUND:.1e}",
                  abs(ref["even_to_fundamental_ratio"] - DISCRETE_SQUARE_EVEN_TO_FUND) < 1e-4)
            check(f"k={k}, phi={phi}: even harmonics are present but tiny (0 < even < 1e-3)",
                  0.0 < ref["even_to_fundamental_ratio"] < 1e-3)
            check(f"k={k}, phi={phi}: no harmonic collision at prime p",
                  ref["n_collisions"] == 0)

    # Tight agreement with the derivation table to 4 decimals (one representative k).
    ref = M.discrete_square_reference(P, 5)
    check(f"fundamental share = {ref['fundamental_share']:.4f} (table: 0.8107)",
          abs(ref["fundamental_share"] - 0.8107) < 5e-5)
    check(f"odd/fundamental = {ref['odd_to_fundamental_ratio']:.4f} (table: 0.1717)",
          abs(ref["odd_to_fundamental_ratio"] - 0.1717) < 5e-5)
    check(f"even/fundamental = {ref['even_to_fundamental_ratio']:.2e} (table: 5.8e-04)",
          abs(ref["even_to_fundamental_ratio"] - 5.8e-4) < 5e-6)
    check("discrete shares bracket the continuous ideal: fundamental 0.8107 vs 8/pi^2 0.8106",
          abs(ref["fundamental_share"] - CONTINUOUS_FUNDAMENTAL_SHARE) < 5e-4)
    check("discrete odd share 0.1717 vs continuous 1/9+1/25+1/49 = 0.1715",
          abs(ref["odd_to_fundamental_ratio"] - CONTINUOUS_ODD_TO_FUND) < 5e-4)
    check("the discrete wave has a small constant component at odd p (0 < share < 1e-3)",
          0.0 < ref["constant_share"] < 1e-3)
    check("n_positive_samples = (p+1)/2 at odd p (unequal level counts)",
          ref["n_positive_samples"] == (P + 1) // 2)
    check("amplitude_ratio_to_fundamental ~ 1/3, 1/5, 1/7 (within 1%)",
          all(abs(ref["amplitude_ratio_to_fundamental"][j] - 1 / j) < 0.01 / j
              for j in (3, 5, 7)))
    check("sign convention is recorded ('cos >= 0 -> +1')",
          ref["sign_convention"] == "cos >= 0 -> +1")
    check("discrete_square_reference rejects even p",
          _raises(lambda: M.discrete_square_reference(112, 5), ValueError))


# --------------------------------------------------------------------------- #
# 6. harmonic_shares uses each column's OWN fundamental                        #
# --------------------------------------------------------------------------- #
def section_harmonic_shares():
    print("\n-- harmonic_shares: each column's OWN fundamental; no collisions at prime p --")
    # Every k at p = 113: the aliased odd harmonics are distinct from each other and from k.
    all_zero, distinct = True, True
    for k in range(1, HALF + 1):
        sh = M.harmonic_shares(power_of(discrete_square(k=k)), k, P)
        all_zero &= sh["n_collisions"] == 0
        idx = [alias_frequency(j * k, P) for j in (1, 3, 5, 7)]
        distinct &= len(set(idx)) == 4 and 0 not in idx
    check(f"n_collisions == 0 for every k in 1..{HALF} at prime p={P}", all_zero)
    check("aliased odd harmonics {k,3k,5k,7k} are 4 distinct nonzero frequencies for every k",
          distinct)

    # A matrix whose columns have DIFFERENT fundamentals: each is measured at its own k.
    ks = np.array([5, 18, 44, 56])
    Y = np.column_stack([discrete_square(k=int(k), phi=0.3 * i) for i, k in enumerate(ks)])
    power = power_of(Y)
    sh = M.harmonic_shares(power, ks, P)
    check("per-column fundamental_share ~ 0.8107 at each column's own k",
          np.allclose(sh["fundamental_share"], DISCRETE_SQUARE_FUNDAMENTAL_SHARE, atol=1e-3))
    check("harmonic_index_by_j row 1 is exactly each column's own k",
          np.array_equal(sh["harmonic_index_by_j"][0], ks))
    check("harmonic_index_by_j rows 3,5,7 are alias(j*k) per column",
          all(sh["harmonic_index_by_j"][j - 1][i] == alias_frequency(j * int(ks[i]), P)
              for j in (3, 5, 7) for i in range(len(ks))))
    check("using the WRONG fundamental for every column collapses the fundamental share",
          M.harmonic_shares(power, np.full(len(ks), 2), P)["fundamental_share"].max()
          < 0.05)
    check("odd_minus_even > 0 for square-wave columns",
          (sh["odd_minus_even"] > 0).all())
    check("family_share = fundamental + odd",
          np.allclose(sh["family_share"], sh["fundamental_share"] + sh["odd_share"]))
    check("shares are fractions of the column's total power (each <= 1)",
          (sh["family_share"] + sh["even_share"] <= 1.0 + 1e-12).all())

    # A sinusoid has (essentially) no harmonic power: odd_share ~ 0.
    power_sin = power_of(np.column_stack([sinusoid(k=k) for k in ks]))
    sh_sin = M.harmonic_shares(power_sin, ks, P)
    check("a sinusoid column has fundamental_share = 1 and odd_share ~ 0",
          np.allclose(sh_sin["fundamental_share"], 1.0, atol=1e-9)
          and sh_sin["odd_share"].max() < 1e-12)

    # collisions ARE possible when a k is chosen for which they exist (small p).
    p_small = 11
    n = np.arange(p_small)
    pw = power_of(np.where(np.cos(2 * np.pi * 3 * n / p_small) >= 0, 1.0, -1.0), p_small)
    sh_small = M.harmonic_shares(pw, 3, p_small)
    check(f"n_collisions is nonzero where harmonics really do alias (p={p_small}, k=3: "
          f"{sh_small['n_collisions']})", sh_small["n_collisions"] > 0)
    check("harmonic_shares rejects a power array whose length contradicts p",
          _raises(lambda: M.harmonic_shares(np.ones((HALF, 1)), 5, 97), ValueError))
    check("harmonic_indices rejects a fundamental outside 1..half",
          _raises(lambda: M.harmonic_indices(HALF + 1, P), ValueError))


# --------------------------------------------------------------------------- #
# 7. inverse participation ratio (Doshi et al. 2023, Eq. 3 / Eq. 4)            #
# --------------------------------------------------------------------------- #
def section_ipr():
    print("\n-- IPR: Doshi et al. 2310.13061 Eq. 3 (r=2), and the _ours variant --")
    check("IPR_DEFINITION_STATUS records where the definition came from",
          "doshi2023_grok_or_not.md" in M.IPR_DEFINITION_STATUS
          and "resolved" in M.IPR_DEFINITION_STATUS)
    check("IPR_DEFINITIONS names all three definitions (doshi_rfft, doshi_fft, ours)",
          set(M.IPR_DEFINITIONS) == {"doshi_rfft", "doshi_fft", "ours"})
    check("IPR_DEFINITIONS['doshi_rfft'] cites the source and its equation",
          "2310.13061" in M.IPR_DEFINITIONS["doshi_rfft"]
          and "Eq. 3" in M.IPR_DEFINITIONS["doshi_rfft"])
    check("IPR_DEFINITIONS['ours'] is labelled a GROKVERSE definition",
          "GROKVERSE" in M.IPR_DEFINITIONS["ours"])
    check("the _ours variant is exposed separately under its own name",
          M.inverse_participation_ratio_ours is not M.inverse_participation_ratio_doshi)
    check("both DFT conventions of the source are exposed",
          M.IPR_DFT_CONVENTIONS == ("rfft", "fft"))

    cos = sinusoid(k=5, phi=-0.3, amp=2.0)
    power = power_of(cos)
    check(f"ideal single-frequency curve: IPR_doshi(rfft) = 1 "
          f"(got {M.inverse_participation_ratio_doshi(cos):.12f})",
          abs(M.inverse_participation_ratio_doshi(cos) - 1.0) < 1e-9)
    check(f"ideal single-frequency curve: IPR_doshi(fft) = 0.5 (the paper's stated range "
          f"[1/p,1]) (got {M.inverse_participation_ratio_doshi(cos, 'fft'):.12f})",
          abs(M.inverse_participation_ratio_doshi(cos, "fft") - 0.5) < 1e-9)
    check(f"ideal single-frequency curve: IPR_ours = 1 "
          f"(got {M.inverse_participation_ratio_ours(power):.12f})",
          abs(M.inverse_participation_ratio_ours(power) - 1.0) < 1e-9)
    check("inverse_participation_ratio (plain name) = IPR_ours on the power array",
          M.inverse_participation_ratio(power) == M.inverse_participation_ratio_ours(power))
    check("IPR_ours = 1 / participation_ratio exactly",
          abs(M.inverse_participation_ratio_ours(power)
              - 1.0 / M.participation_ratio(power)) < 1e-12)

    one_hot = np.zeros(P)
    one_hot[3] = 1.0
    check(f"one-hot curve hits the rfft floor 1/(p//2+1) = {1 / (P // 2 + 1):.5f}",
          abs(M.inverse_participation_ratio_doshi(one_hot) - 1 / (P // 2 + 1)) < 1e-12)
    check(f"one-hot curve hits the fft floor 1/p = {1 / P:.5f}",
          abs(M.inverse_participation_ratio_doshi(one_hot, "fft") - 1 / P) < 1e-12)

    W = white_noise(n_cols=400, seed=1)
    ipr_ours = M.inverse_participation_ratio_ours(power_of(W))
    ipr_doshi = M.inverse_participation_ratio_doshi(W)
    n_freq_ours, n_freq_doshi = HALF, P // 2 + 1
    check(f"white noise: IPR_ours = {ipr_ours.mean():.4f} is of order 1/n_freq = "
          f"{1 / n_freq_ours:.4f} (exponential periodogram gives ~2/n_freq)",
          1 / n_freq_ours <= ipr_ours.mean() <= 4 / n_freq_ours)
    check(f"white noise: IPR_doshi(rfft) = {ipr_doshi.mean():.4f} is of order 1/n_freq = "
          f"{1 / n_freq_doshi:.4f}", 1 / n_freq_doshi <= ipr_doshi.mean() <= 4 / n_freq_doshi)
    check("white noise IPR is far below the single-frequency value 1 (max < 0.2)",
          ipr_doshi.max() < 0.2 and ipr_ours.max() < 0.2)

    # Eq. 4: the per-neuron IPR is the plain mean of the three curve IPRs.
    neu = M.ipr_doshi_neuron(cos, cos * 3.0, cos * -0.5)
    check("Eq. 4 on three single-frequency curves gives ipr_neuron = 1",
          abs(neu["ipr_neuron"] - 1.0) < 1e-9)
    mixed = M.ipr_doshi_neuron(cos, W[:, 0], W[:, 1])
    check("Eq. 4 is the plain mean of the three per-vector IPRs",
          abs(mixed["ipr_neuron"] - (mixed["ipr_u_a"] + mixed["ipr_u_b"]
                                     + mixed["ipr_out"]) / 3.0) < 1e-12)
    check("Eq. 4 result echoes the DFT convention it used",
          mixed["dft"] == "rfft" and "2310.13061" in mixed["definition"])
    check("Eq. 4 rejects curves of differing shape",
          _raises(lambda: M.ipr_doshi_neuron(cos, W[:, :2], cos), ValueError))
    check("an unknown DFT convention is rejected",
          _raises(lambda: M.inverse_participation_ratio_doshi(cos, "welch"), ValueError))
    check("no IPR threshold is applied anywhere (the source defines none: it ranks)",
          not any("threshold" in str(v).lower() for v in M.IPR_DEFINITIONS.values()))


# --------------------------------------------------------------------------- #
# 8. periodicity score (Swaroop 2026, Eq. 1)                                   #
# --------------------------------------------------------------------------- #
def section_periodicity_score():
    print("\n-- periodicity_score: Swaroop 2603.23784 Eq. 1, thresholds exposed not applied --")
    check("SWAROOP_PERIODICITY_THRESHOLDS is a labelled constant naming its source and p",
          M.SWAROOP_PERIODICITY_THRESHOLDS["structured_gt"] == 12.0
          and M.SWAROOP_PERIODICITY_THRESHOLDS["unstructured_lt"] == 5.0
          and M.SWAROOP_PERIODICITY_THRESHOLDS["p_of_source"] == 97
          and "2603.23784" in M.SWAROOP_PERIODICITY_THRESHOLDS["source"])
    check("the thresholds are NOT applied silently: periodicity_score returns a raw score, "
          "no boolean/classification key exists in the module",
          not any(name.startswith(("is_", "classify_")) for name in dir(M)))

    cos = sinusoid(k=5, phi=0.4, amp=2.0)
    score_cos = M.periodicity_score(cos)
    check(f"a clean sinusoid attains the maximum (p-1)/2 = {M.periodicity_score_max(P)} "
          f"(got {score_cos:.4f})", abs(score_cos - M.periodicity_score_max(P)) < 1e-9)
    check("Eq. 1 is max|DFT| over mean|DFT| (k = 1..p-1) — the reference implementation "
          "agrees to 1e-12",
          abs(score_cos - _swaroop_eq1(cos)) < 1e-12)
    check("periodicity_score_from_power equals periodicity_score for real curves at odd p",
          abs(M.periodicity_score_from_power(power_of(cos)) - score_cos) < 1e-9)
    check("periodicity_score_swaroop is the same function (source-named alias)",
          M.periodicity_score_swaroop is M.periodicity_score)

    sq = discrete_square(k=5, phi=0.3)
    score_sq = M.periodicity_score(sq)
    check(f"an ideal sign(cos) square wave scores {score_sq:.3f} at p=113, reproducing the "
          f"source note's own computation ({SWAROOP_SQUARE_SCORE_P113})",
          abs(score_sq - SWAROOP_SQUARE_SCORE_P113) < 0.01)
    check("the score measures single-frequency dominance, not square-ness: a cosine scores "
          "~3x an ideal square wave", score_cos / score_sq > 2.5)

    W = white_noise(n_cols=400, seed=2)
    score_noise = M.periodicity_score(W)
    check(f"white noise scores near the flat-spectrum floor 1 (mean {score_noise.mean():.2f}, "
          f"max {score_noise.max():.2f})", score_noise.mean() < 4.0 and score_noise.max() < 6.0)
    check("white noise never reaches the source's 'structured' threshold 12",
          score_noise.max() < M.SWAROOP_PERIODICITY_THRESHOLDS["structured_gt"])
    check("periodicity_score is NaN (not 0, not a substitute) for an all-zero curve",
          np.isnan(M.periodicity_score(np.zeros(P))))
    check("periodicity_score needs p >= 3",
          _raises(lambda: M.periodicity_score(np.zeros(2)), ValueError))


def _swaroop_eq1(y: np.ndarray) -> float:
    """Eq. 1 written out literally from the source quote, independent of the module."""
    mags = np.abs(np.fft.fft(np.asarray(y, dtype=float)))
    tail = mags[1:]                                     # k = 1..p-1
    return float(tail.max() / (tail.sum() / len(tail)))


# --------------------------------------------------------------------------- #
# 9. binarization score (ours)                                                 #
# --------------------------------------------------------------------------- #
def section_binarization():
    print("\n-- binarization_score_ours: 1 for a square wave, small for a sinusoid --")
    check("binarization_score_ours is 1.0 for a perfect square wave",
          M.binarization_score_ours(discrete_square(k=5, phi=0.9)) == 1.0)
    check("still 1.0 for a scaled/negated square wave",
          M.binarization_score_ours(-3.4 * discrete_square(k=18)) == 1.0)
    frac = M.binarization_score_ours(sinusoid(k=5, phi=0.4))
    analytic = 2 * math.acos(0.8) / math.pi                      # 0.4097
    check(f"a pure sinusoid scores ~2 arccos(0.8)/pi = {analytic:.4f} (got {frac:.4f})",
          abs(frac - analytic) < 0.02)
    check("a sinusoid scores far below a square wave", frac < 0.5)
    check(f"white noise scores small (max {M.binarization_score_ours(white_noise(n_cols=200)).max():.3f})",
          M.binarization_score_ours(white_noise(n_cols=200)).max() < 0.3)
    check("the threshold is a parameter, not a hidden constant (0.5 admits more samples)",
          M.binarization_score_ours(sinusoid(), 0.5) > M.binarization_score_ours(sinusoid(), 0.9))
    check("binarization_score is the labelled alias of the _ours definition",
          M.binarization_score is M.binarization_score_ours)
    check("an out-of-range threshold is rejected",
          _raises(lambda: M.binarization_score_ours(sinusoid(), 1.5), ValueError))
    check("binarization_score_ours is NaN for an all-zero curve",
          np.isnan(M.binarization_score_ours(np.zeros(P))))


# --------------------------------------------------------------------------- #
# 10. curve_metrics wrapper                                                    #
# --------------------------------------------------------------------------- #
DOCUMENTED_CURVE_METRIC_KEYS = {
    "params", "n_columns", "n_undefined_columns", "dominant_frequency", "dominant_phase",
    "dominant_fraction", "topk_concentration", "spectral_entropy", "participation_ratio",
    "inverse_participation_ratio_ours", "inverse_participation_ratio_doshi_rfft",
    "inverse_participation_ratio_doshi_fft", "periodicity_score_swaroop",
    "binarization_score_ours", "harmonic_shares", "total_power",
}
DOCUMENTED_HARMONIC_KEYS = {
    "fundamental_share", "odd_share", "even_share", "family_share", "odd_minus_even",
    "odd_to_fundamental_ratio", "even_to_fundamental_ratio", "harmonic_share_by_j",
    "harmonic_index_by_j", "n_collisions", "ideal_square_fundamental_share",
    "ideal_square_odd_share", "ideal_square_even_share", "ideal_square_family_share",
    "ideal_square_odd_minus_even", "params",
}


def section_curve_metrics():
    print("\n-- curve_metrics: every documented key for a [p, n] matrix; 1-D accepted --")
    ks = [5, 18, 44]
    Y = np.column_stack([sinusoid(k=ks[0], phi=0.7), discrete_square(k=ks[1], phi=-1.1),
                         white_noise(n_cols=1, seed=7)[:, 0]])
    out = M.curve_metrics(Y, P)
    check("curve_metrics returns exactly the documented top-level keys",
          set(out) == DOCUMENTED_CURVE_METRIC_KEYS)
    check("harmonic_shares carries every documented key",
          set(out["harmonic_shares"]) == DOCUMENTED_HARMONIC_KEYS)
    check("params echoes p, topk, max_harmonic, binarization_threshold and the IPR definition",
          set(out["params"]) == {"p", "topk", "max_harmonic", "binarization_threshold",
                                 "ipr_definition"} and out["params"]["p"] == P)
    check("n_columns = 3, n_undefined_columns = 0", out["n_columns"] == 3
          and out["n_undefined_columns"] == 0)
    check("every per-column array has length n_columns",
          all(np.shape(out[key]) == (3,) for key in
              ("dominant_frequency", "dominant_phase", "dominant_fraction", "spectral_entropy",
               "participation_ratio", "inverse_participation_ratio_ours",
               "inverse_participation_ratio_doshi_rfft", "periodicity_score_swaroop",
               "binarization_score_ours", "total_power")))
    check("topk_concentration is reported per requested k (1, 4, 8)",
          set(out["topk_concentration"]) == {1, 4, 8}
          and all(np.shape(v) == (3,) for v in out["topk_concentration"].values()))
    check("the sinusoid / square / noise columns are ordered as expected by dominant_fraction",
          out["dominant_fraction"][0] > out["dominant_fraction"][1]
          > out["dominant_fraction"][2])
    check("each column is measured at its OWN dominant frequency",
          list(out["dominant_frequency"][:2]) == ks[:2])
    check("the square column's fundamental share ~ 0.8107, the sinusoid's ~ 1",
          abs(out["harmonic_shares"]["fundamental_share"][1]
              - DISCRETE_SQUARE_FUNDAMENTAL_SHARE) < 1e-3
          and abs(out["harmonic_shares"]["fundamental_share"][0] - 1.0) < 1e-9)
    check("the wrapper's values equal the standalone functions",
          np.allclose(out["spectral_entropy"], M.spectral_entropy(power_of(Y)))
          and np.allclose(out["periodicity_score_swaroop"], M.periodicity_score(Y)))

    # 1-D input: accepted and treated as a single column.
    single = M.curve_metrics(sinusoid(k=5, phi=0.7), P)
    check("1-D input is accepted and squeezed to a single column (n_columns == 1)",
          single["n_columns"] == 1)
    check("1-D input gives length-1 per-column arrays, not a ragged/2-D result",
          all(np.shape(single[key]) == (1,) for key in
              ("dominant_frequency", "dominant_fraction", "spectral_entropy",
               "participation_ratio", "periodicity_score_swaroop")))
    check("1-D input agrees value-for-value with the same curve as a [p, 1] matrix",
          np.allclose(single["dominant_fraction"],
                      M.curve_metrics(sinusoid(k=5, phi=0.7)[:, None], P)["dominant_fraction"]))
    check("1-D input agrees with the [p, n] result for the same column",
          int(single["dominant_frequency"][0]) == int(out["dominant_frequency"][0]))


# --------------------------------------------------------------------------- #
# 11. edge cases                                                               #
# --------------------------------------------------------------------------- #
def section_edge_cases():
    print("\n-- edge cases: zero-power column, even p, malformed input --")
    Y = np.column_stack([sinusoid(k=5, phi=0.7), np.zeros(P)])
    out = M.curve_metrics(Y, P)                                   # must not raise
    check("a zero-power column does not crash curve_metrics", out["n_columns"] == 2)
    check("the zero column is COUNTED, not silently substituted (n_undefined_columns == 1)",
          out["n_undefined_columns"] == 1)
    check("the good column keeps finite metrics next to a zero column",
          np.isfinite([out["dominant_fraction"][0], out["spectral_entropy"][0],
                       out["participation_ratio"][0],
                       out["periodicity_score_swaroop"][0]]).all())
    check("the good column's values are unchanged by the zero column's presence",
          abs(out["dominant_fraction"][0]
              - M.curve_metrics(Y[:, :1], P)["dominant_fraction"][0]) < 1e-15)
    check("the zero column reports NaN (documented: never a substitute value)",
          np.isnan(out["dominant_fraction"][1])
          and np.isnan(out["spectral_entropy"][1])
          and np.isnan(out["binarization_score_ours"][1]))
    check("the zero column's dominant_frequency is the reserved 0 = undefined",
          out["dominant_frequency"][1] == 0)
    check("a zero column's harmonic shares are NaN, not 0",
          np.isnan(out["harmonic_shares"]["fundamental_share"][1]))
    check("an ALL-zero matrix raises instead of returning meaningless numbers",
          _raises(lambda: M.curve_metrics(np.zeros((P, 2)), P), ValueError))
    check("a zero column does not make the standalone metrics raise",
          np.isnan(M.spectral_entropy(power_of(np.zeros((P, 1))))).all())

    # even p: the Fourier basis is undefined (cos/sin pairing gives only p-1 rows).
    for p_even in (112, 8):
        check(f"p={p_even} (even) is rejected by power_spectrum",
              _raises(lambda pe=p_even: M.power_spectrum(np.ones(pe), pe), ValueError))
        check(f"p={p_even} (even) is rejected by curve_metrics",
              _raises(lambda pe=p_even: M.curve_metrics(np.ones((pe, 1)), pe), ValueError))
        check(f"p={p_even} (even) is rejected by discrete_square_reference",
              _raises(lambda pe=p_even: M.discrete_square_reference(pe, 1), ValueError))

    check("non-finite input is rejected, not propagated",
          _raises(lambda: M.spectral_entropy(np.array([[np.nan], [1.0]])), ValueError))
    check("a 3-D array is rejected",
          _raises(lambda: M.power_spectrum(np.ones((P, 2, 2)), P), ValueError))
    check("an empty array is rejected",
          _raises(lambda: M.spectral_entropy(np.zeros((0, 2))), ValueError))


# --------------------------------------------------------------------------- #
# 12. NEGATIVE CONTROL                                                         #
# --------------------------------------------------------------------------- #
def section_negative_control():
    print("\n-- NEGATIVE CONTROL: white noise must NOT be scored as periodic --")
    W = white_noise(n_cols=500, seed=11)
    out = M.curve_metrics(W, P)
    sin_out = M.curve_metrics(np.column_stack([sinusoid(k=k) for k in (5, 18, 44)]), P)
    sq_out = M.curve_metrics(np.column_stack([discrete_square(k=k) for k in (5, 18, 44)]), P)

    check("noise: NO column reaches Swaroop's 'structured' periodicity threshold (>12) — "
          f"max {out['periodicity_score_swaroop'].max():.2f}",
          out["periodicity_score_swaroop"].max()
          < M.SWAROOP_PERIODICITY_THRESHOLDS["structured_gt"])
    check("noise: every column is below the periodicity score of an ideal square wave",
          out["periodicity_score_swaroop"].max() < SWAROOP_SQUARE_SCORE_P113)
    check(f"noise: no column has dominant_fraction > 0.3 (max "
          f"{out['dominant_fraction'].max():.3f})", out["dominant_fraction"].max() < 0.3)
    check("noise: no column has top-1 concentration above the weakest sinusoid's",
          out["topk_concentration"][1].max() < sin_out["topk_concentration"][1].min())
    check(f"noise: top-8 concentration (max {out['topk_concentration'][8].max():.2f}) stays "
          f"below every square wave's ({sq_out['topk_concentration'][8].min():.2f}) and every "
          f"sinusoid's (1.00) — but its MEAN is already "
          f"{out['topk_concentration'][8].mean():.2f} because 8 of {HALF} frequencies carry "
          f"that much of any spectrum, so top-8 alone is a weak periodicity statistic (H3)",
          out["topk_concentration"][8].max() < sq_out["topk_concentration"][8].min()
          and out["topk_concentration"][8].mean() > 0.3)
    check("noise: every column's participation ratio is far above 1 (min "
          f"{out['participation_ratio'].min():.1f})", out["participation_ratio"].min() > 5.0)
    check("noise: every column's spectral entropy exceeds every sinusoid's and every "
          "square wave's", out["spectral_entropy"].min()
          > max(sin_out["spectral_entropy"].max(), sq_out["spectral_entropy"].max()))
    check("noise: no column reaches IPR 0.2 (single-frequency = 1)",
          out["inverse_participation_ratio_doshi_rfft"].max() < 0.2
          and out["inverse_participation_ratio_ours"].max() < 0.2)
    check("noise: no column looks near-binary (binarization < 0.3)",
          out["binarization_score_ours"].max() < 0.3)
    check("noise: no column looks like a square wave at its own dominant frequency "
          "(fundamental share < 0.3 everywhere; an ideal square wave has 0.81)",
          out["harmonic_shares"]["fundamental_share"].max() < 0.3)
    check("noise: odd_minus_even does not separate it from a square wave in the wrong "
          "direction (square wave's median is far larger)",
          float(np.median(sq_out["harmonic_shares"]["odd_minus_even"]))
          > 10 * float(np.median(out["harmonic_shares"]["odd_minus_even"])))

    # A scrambled square wave: same values, destroyed periodicity.
    rng = np.random.default_rng(5)
    scrambled = np.column_stack([rng.permutation(discrete_square(k=5, phi=0.3))
                                 for _ in range(200)])
    sc = M.curve_metrics(scrambled, P)
    check("scrambled square wave (same values, shuffled positions) loses its periodicity "
          f"score (max {sc['periodicity_score_swaroop'].max():.2f} < 12)",
          sc["periodicity_score_swaroop"].max()
          < M.SWAROOP_PERIODICITY_THRESHOLDS["structured_gt"])
    check("scrambled square wave loses its fundamental share (max < 0.3 vs 0.81)",
          sc["harmonic_shares"]["fundamental_share"].max() < 0.3)
    check("but the scrambled curve is STILL near-binary — binarization measures shape, not "
          "periodicity (and is therefore not a periodicity statistic)",
          (sc["binarization_score_ours"] == 1.0).all())


def main():
    section_spectrum_convention()
    section_injected_sinusoid()
    section_white_noise()
    section_continuous_square_wave()
    section_discrete_square_reference()
    section_harmonic_shares()
    section_ipr()
    section_periodicity_score()
    section_binarization()
    section_curve_metrics()
    section_edge_cases()
    section_negative_control()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
