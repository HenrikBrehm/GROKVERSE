"""Lightweight correctness checks for the GROKVERSE core (run: python test_core.py).

Not a full test suite — fast asserts on the science-bearing functions
(determinism, dataset labels, Fourier-basis orthonormality + frequency recovery,
PCA shapes, MLP wiring) so regressions are caught. Exits non-zero on failure.
"""
from __future__ import annotations

import numpy as np

from grokverse.analysis.compare import agg
from grokverse.analysis.fourier import (alias_frequency, dominant_frequencies,
                                        embedding_power_spectrum, fourier_basis,
                                        harmonic_family,
                                        harmonic_family_concentration,
                                        harmonic_shape, select_fundamentals)
from grokverse.analysis.mask_protocols import (IMPLEMENTED, build_protocol,
                                               operator_trace)
from grokverse.analysis.mlp_mechanism import (PROPOSED, classify_neurons,
                                              curve_spectra, effective_curves,
                                              phase_relation)
from grokverse.analysis.pca import fit_pca, project
from grokverse.analysis.progress_measures import _mode_indices, fwd2d, inv2d
from grokverse.config import get_config
from grokverse.data import make_dataset
from grokverse.models import build_model
from grokverse.seed import set_seed
from grokverse.train import detect_transition
from grokverse.utils import config_diff, log_step_schedule


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def main():
    cfg = get_config("nanda", p=23, seed=0)
    d = make_dataset(cfg)
    x, y = d["all_x"], d["all_y"]
    a, b = x[:, 0], x[:, 1]
    check("labels == (a+b) mod p", bool(((a + b) % cfg.p == y).all()))
    check("equals token in last column", bool((x[:, 2] == cfg.equals_token).all()))
    check("train+test partition complete", d["train_x"].shape[0] + d["test_x"].shape[0] == cfg.p * cfg.p)
    check("split is deterministic", bool((d["train_x"] == make_dataset(cfg)["train_x"]).all()))

    set_seed(0); o1 = build_model(cfg).logits_last(x).sum().item()
    set_seed(0); o2 = build_model(cfg).logits_last(x).sum().item()
    check("model init deterministic", o1 == o2)

    cfgm = get_config("nanda", p=23, task="mul")
    dm = make_dataset(cfgm)
    check("mul labels == (a*b) mod p", bool(((dm["all_x"][:, 0] * dm["all_x"][:, 1]) % cfgm.p == dm["all_y"]).all()))

    cfgmlp = get_config("nanda", p=23, arch="mlp")
    out = build_model(cfgmlp).logits_last(make_dataset(cfgmlp)["all_x"])
    check("mlp outputs p classes", out.shape[-1] == cfgmlp.p)

    F, _ = fourier_basis(cfg.p)
    check("fourier basis orthonormal", np.allclose(F @ F.T, np.eye(cfg.p), atol=1e-8))
    check("fourier basis has p rows", F.shape[0] == cfg.p)

    p, k = cfg.p, 3
    n = np.arange(p)
    W = np.stack([np.cos(2 * np.pi * k * n / p), np.sin(2 * np.pi * k * n / p)], axis=1)
    spec = embedding_power_spectrum(W, p)
    check("spectrum recovers injected frequency k", spec["freqs"][int(np.argmax(spec["fraction"]))] == k)

    Xr = np.random.RandomState(0).randn(p, 8)
    coords = project(Xr, fit_pca(Xr, 3))
    check("pca projects to 3D", coords.shape == (p, 3))

    # --- the functions the headline numbers come from ---
    curves = {"step": [0, 100, 200, 300], "train_acc": [0.1, 0.995, 1.0, 1.0],
              "test_acc": [0.01, 0.02, 0.05, 0.97]}
    t = detect_transition(curves)
    check("transition: train sat at 0.99 crossing", t["train_saturated_step"] == 100)
    check("transition: test gen at 0.95 crossing", t["test_generalized_step"] == 300)
    check("transition: grok gap", t["grok_gap"] == 200)
    t2 = detect_transition({"step": [0, 1], "train_acc": [1.0, 1.0], "test_acc": [0.1, 0.2]})
    check("transition: no generalization -> None gap", t2["grok_gap"] is None)

    sched = log_step_schedule(1000, 20)
    check("log schedule includes 0 and final step", sched[0] == 0 and sched[-1] == 1000)
    check("log schedule strictly increasing", all(b > a for a, b in zip(sched, sched[1:])))
    check("log schedule for smoke run", log_step_schedule(1, 2) == [0, 1])

    Fb, _ = fourier_basis(p)
    L = np.random.RandomState(1).randn(p, p, 4)
    check("2D fourier roundtrip inverts", np.abs(inv2d(fwd2d(L, Fb), Fb) - L).max() < 1e-9)
    keep, is_key = _mode_indices(p, [3, 5])
    check("mode masks: const kept, not a key freq", bool(keep[0]) and not bool(is_key[0]))
    check("mode masks: 2 rows per key freq", int(is_key.sum()) == 4 and int(keep.sum()) == 5)

    a3 = agg([675, 859, 716])
    check("agg uses sample std (ddof=1)", abs(a3["std"] - np.std([675, 859, 716], ddof=1)) < 0.01)
    check("agg drops None but counts totals", agg([100, None])["n"] == 1 and agg([100, None])["n_total"] == 2)

    check_mask_protocols()
    check_harmonic_families()
    check_mlp_mechanism()
    check_config_additions()

    print("\nALL CHECKS PASSED")


# --------------------------------------------------------------------------- #
# RESEARCH_SPEC §3.1 — the named restricted/excluded mask variants             #
# --------------------------------------------------------------------------- #
def check_mask_protocols():
    print("\n-- mask protocols (RESEARCH_SPEC §3.1) --")
    P, KEYS = 113, [18, 15, 1, 11, 13, 22, 56, 36]   # canonical run's key freqs
    protos = {n: build_protocol(n, P, KEYS) for n in IMPLEMENTED}

    # Component counts pinned to the numbers computed in the spec's §3.1 table.
    expected = {"legacy_broad_mask": (289, 3360),
                "same_frequency_block": (33, 32),
                "sum_directions_only": (17, 16)}
    for name, (n_keep, n_rm) in expected.items():
        pr = protos[name]
        check(f"{name}: keeps {n_keep} of {P*P} components", pr.n_kept == n_keep)
        check(f"{name}: excluded removes {n_rm} components", pr.n_removed == n_rm)
    check("legacy excluded deletes 26.3% of the logit tensor",
          abs(protos["legacy_broad_mask"].n_removed / (P * P) - 0.263) < 0.001)
    check("legacy keeps ~9x more than same-frequency, ~17x more than sum-directions",
          289 // 33 == 8 and 289 // 17 == 17)

    # The defect itself: only the legacy variant keeps cross-frequency blocks.
    check("legacy KEEPS cross-frequency blocks (the §3.1 defect)",
          protos["legacy_broad_mask"].keeps_cross_frequency())
    for name in ("same_frequency_block", "sum_directions_only"):
        check(f"{name}: no cross-frequency blocks", not protos[name].keeps_cross_frequency())

    # Analytic counts verified numerically: trace of a projection == its rank.
    small_keys = [3, 5]
    for name in IMPLEMENTED:
        pr = build_protocol(name, 23, small_keys)
        tr_r = operator_trace(pr.restrict, 23)
        tr_e = operator_trace(pr.exclude, 23)
        check(f"{name}: n_kept matches operator trace (p=23)", abs(tr_r - pr.n_kept) < 1e-9)
        check(f"{name}: n_removed matches 23^2 - trace(exclude)",
              abs((23 * 23 - tr_e) - pr.n_removed) < 1e-9)

    # Constant term: restricted keeps it, excluded keeps it too (both variants).
    for name in IMPLEMENTED:
        pr = build_protocol(name, 23, small_keys)
        c = np.zeros((23, 23, 1)); c[0, 0, 0] = 1.0
        check(f"{name}: restricted keeps the constant", pr.restrict(c)[0, 0, 0] == 1.0)
        check(f"{name}: excluded keeps the constant", pr.exclude(c)[0, 0, 0] == 1.0)

    # Idempotent projections, and restrict/exclude are complementary on the
    # circuit subspace for the exact variants (n_kept == n_removed + 1).
    rng = np.random.RandomState(3)
    Lh = rng.randn(23, 23, 4)
    for name in IMPLEMENTED:
        pr = build_protocol(name, 23, small_keys)
        check(f"{name}: restrict is idempotent",
              np.abs(pr.restrict(pr.restrict(Lh)) - pr.restrict(Lh)).max() < 1e-12)
        check(f"{name}: exclude is idempotent",
              np.abs(pr.exclude(pr.exclude(Lh)) - pr.exclude(Lh)).max() < 1e-12)
    for name in ("same_frequency_block", "sum_directions_only"):
        pr = build_protocol(name, 23, small_keys)
        check(f"{name}: n_kept == n_removed + 1 (const)", pr.n_kept == pr.n_removed + 1)

    # sum_directions_only must isolate cos/sin(w_k(a+b)) EXACTLY: an ideal
    # circuit survives restriction untouched and is annihilated by exclusion.
    p, k = 23, 3
    n = np.arange(p)
    A, B = np.meshgrid(n, n, indexing="ij")
    w = 2 * np.pi * k / p
    ideal = np.cos(w * (A + B))[:, :, None]                 # a pure sum wave
    diff_wave = np.cos(w * (A - B))[:, :, None]             # the control
    Fb, _ = fourier_basis(p)
    pr = build_protocol("sum_directions_only", p, [k])
    ideal_hat = fwd2d(ideal, Fb)
    check("sum_directions: ideal cos(w(a+b)) survives restriction",
          np.abs(inv2d(pr.restrict(ideal_hat), Fb) - ideal).max() < 1e-10)
    check("sum_directions: ideal cos(w(a+b)) is destroyed by exclusion",
          np.abs(inv2d(pr.exclude(ideal_hat), Fb)).max() < 1e-10)
    diff_hat = fwd2d(diff_wave, Fb)
    check("sum_directions: cos(w(a-b)) is NOT mistaken for the circuit",
          np.abs(inv2d(pr.restrict(diff_hat), Fb)).max() < 1e-10)
    check("same_frequency_block DOES keep cos(w(a-b)) (it is the looser variant)",
          np.abs(inv2d(build_protocol("same_frequency_block", p, [k]).restrict(diff_hat),
                       Fb)).max() > 0.1)

    # Cross-frequency leakage, concretely: a cos(w_3 a)cos(w_5 b) term.
    cross = (np.cos(2 * np.pi * 3 * A / p) * np.cos(2 * np.pi * 5 * B / p))[:, :, None]
    cross_hat = fwd2d(cross, Fb)
    leak = {name: float(np.abs(inv2d(build_protocol(name, p, [3, 5]).restrict(cross_hat),
                                     Fb)).max()) for name in IMPLEMENTED}
    check("legacy restricted leaks a cross-frequency term", leak["legacy_broad_mask"] > 0.1)
    check("exact variants leak nothing",
          leak["same_frequency_block"] < 1e-10 and leak["sum_directions_only"] < 1e-10)

    # The legacy protocol object must reproduce the ORIGINAL implementation
    # bit-for-bit, or "compare the variants" would compare against a straw man.
    keep, is_key = _mode_indices(p, [3, 5])
    legacy_restr = keep[:, None] & keep[None, :]
    legacy_excl = ~(is_key[:, None] | is_key[None, :])
    lp = build_protocol("legacy_broad_mask", p, [3, 5])
    check("legacy protocol == original _mode_indices masks",
          np.abs(lp.restrict(Lh) - Lh * legacy_restr[:, :, None]).max() == 0.0
          and np.abs(lp.exclude(Lh) - Lh * legacy_excl[:, :, None]).max() == 0.0)

    check("nanda_exact refuses to guess the published protocol",
          _raises(lambda: build_protocol("nanda_exact", p, [3]).restrict(Lh)))
    check("unknown protocol name rejected",
          _raises(lambda: build_protocol("made_up", p, [3])))
    check("empty key set rejected", _raises(lambda: build_protocol(IMPLEMENTED[0], p, [])))
    check("out-of-range key freq rejected",
          _raises(lambda: build_protocol(IMPLEMENTED[0], p, [99])))


# --------------------------------------------------------------------------- #
# RESEARCH_SPEC §4/H3 — harmonic families and their degrees-of-freedom control #
# --------------------------------------------------------------------------- #
def check_harmonic_families():
    print("\n-- harmonic families (RESEARCH_SPEC §4/H3) --")
    p = 113
    check("alias folds k and p-k together",
          alias_frequency(56, p) == 56 and alias_frequency(57, p) == 56)
    check("alias sends multiples of p to 0 (the constant, not a frequency)",
          alias_frequency(0, p) == 0 and alias_frequency(p, p) == 0)

    # The §4/H3 aliasing table, recomputed.
    check("aliased odd harmonics of k=18 are {54, 23, 13}",
          [alias_frequency(j * 18, p) for j in (3, 5, 7)] == [54, 23, 13])
    check("aliased odd harmonics of k=15 are {45, 38, 8}",
          [alias_frequency(j * 15, p) for j in (3, 5, 7)] == [45, 38, 8])
    check("aliased odd harmonics of k=11 are {33, 55, 36}",
          [alias_frequency(j * 11, p) for j in (3, 5, 7)] == [33, 55, 36])

    # THE reason the cap exists: uncapped, one fundamental covers everything.
    half = (p - 1) // 2
    check("uncapped odd family of a single fundamental covers ALL frequencies",
          len(harmonic_family([18], p, max_harmonic=2 * p)) == half)
    check("capped family of one fundamental has 4 members",
          len(harmonic_family([18], p, 7)) == 4)
    check("capped family of 8 fundamentals has at most 32 members",
          len(harmonic_family([18, 15, 1, 11, 13, 22, 56, 36], p, 7)) <= 32)

    # A synthetic square wave: power at odd harmonics only, decaying ~1/j.
    n = np.arange(p)
    k = 5
    square = np.sign(np.sin(2 * np.pi * k * n / p) + 1e-12)
    W = np.stack([square, np.roll(square, 3)], axis=1)
    power = np.array(embedding_power_spectrum(W, p)["power"])
    check("square wave: fundamental k is the strongest frequency",
          int(np.argmax(power)) + 1 == k)
    check("square wave: >99% of power sits in the uncapped odd family",
          power[[f - 1 for f in harmonic_family([k], p, 2 * p)]].sum() / power.sum() > 0.99)

    # Greedy selection is deterministic and picks the strongest fundamental first.
    fnd = select_fundamentals(power, p, 2, 7)
    check("greedy selection picks the square wave's fundamental first", fnd[0] == k)
    check("greedy selection is deterministic", select_fundamentals(power, p, 2, 7) == fnd)

    res = harmonic_family_concentration(W, p, n_f=2, n_null=200, seed=0)
    check("family concentration reports a cardinality-matched top-m control",
          res["m"] == len(res["family"]) and len(res["matched_top_m"]) == res["m"])
    check("family concentration reports a random-set null", res["null_n"] == 200)
    check("family concentration is reproducible under its seed",
          harmonic_family_concentration(W, p, n_f=2, n_null=200,
                                        seed=0)["null_mean"] == res["null_mean"])
    # The inequality that makes "family beats top-m" an UNUSABLE criterion:
    # top-m is the argmax over sets of size m, so the family can never exceed it.
    # Pinned so nobody reintroduces it as a support criterion for H3.
    check("family can never beat the matched top-m control (forced inequality)",
          res["family_minus_matched_top_m"] <= 1e-12)

    # What DOES discriminate: the odd-vs-even harmonic shape statistic.
    rng = np.random.default_rng(0)
    FUND = [18, 15, 11, 1, 13, 22, 56, 36]

    def synth(kind, noise, gen):
        cols = []
        for f in FUND:
            for ph in gen.uniform(0, 2 * np.pi, 4):
                wave = np.sin(2 * np.pi * f * n / p + ph)
                cols.append(np.sign(wave + 1e-12) if kind == "square" else wave)
        W = np.stack(cols, axis=1)
        return W + noise * gen.standard_normal(W.shape)

    shapes = {}
    for kind in ("sinusoid", "square"):
        for noise in (0.0, 0.6):
            pw = np.array(embedding_power_spectrum(synth(kind, noise, rng), p)["power"])
            shapes[(kind, noise)] = harmonic_shape(pw, FUND, p)
    check("square waves put energy at ODD harmonics (j=3,5,7)",
          shapes[("square", 0.0)]["odd_share"] > 0.15)
    check("square-wave odd share is near the ideal 1/j^2 law (0.1715)",
          abs(shapes[("square", 0.0)]["odd_share"]
              - shapes[("square", 0.0)]["ideal_square_odd_share"]) < 0.12)
    check("sinusoids put ~no energy at harmonics",
          abs(shapes[("sinusoid", 0.0)]["odd_minus_even"]) < 0.02)
    # --- fourier.harmonic_shape is SUPERSEDED (2026-09-03) -------------------
    # It is defined on a SET of fundamentals, and inside the canonical set six harmonics
    # alias onto other members (2*18->36, 7*18->13, 2*11->22, 7*11->36, 7*13->22, 2*56->1).
    # Those land in `base_idx` and are subtracted from both shares, so a clean square wave
    # reports even_share = 0.083 where the truth is 0.0005 -- inflated ~165x -- and the
    # odd-minus-even separation falls to 0.092. The check that used to sit here demanded
    # > 0.10 and therefore asserted a property this statistic cannot have; it was the one
    # failing check in the suite. It is replaced by checks on the DIAGNOSED behaviour plus
    # a check that the replacement, metrics.harmonic_shares (per curve, on that curve's OWN
    # fundamental, collision-free at prime p), does separate. See docs/LEGACY_METRIC_AUDIT.md
    # and docs/BASELINE.md.
    sep = (shapes[("square", 0.0)]["odd_minus_even"]
           - shapes[("sinusoid", 0.0)]["odd_minus_even"])
    check("set-based shape: separation is degraded to ~0.09 by in-set aliasing collisions",
          0.07 < sep < 0.10)
    check("set-based shape: a clean square wave reports a spurious even share ~0.08",
          0.06 < shapes[("square", 0.0)]["even_share"] < 0.10)
    collisions = [(j, k) for k in FUND for j in (2, 3, 4, 5, 6, 7)
                  if alias_frequency(j * k, p) in FUND]
    check("set-based shape: the canonical fundamental set has exactly 6 in-set collisions",
          len(collisions) == 6)

    # the replacement, on the same synthetic data, gets the even share right and separates
    from grokverse.analysis import metrics as _M
    rng2 = np.random.default_rng(0)
    per_curve = {}
    for kind in ("sinusoid", "square"):
        for noise in (0.0, 0.6):
            W = synth(kind, noise, rng2)
            sp = _M.power_spectrum(W, p)
            hs = _M.harmonic_shares(sp["power"], sp["dominant_freq"], p)
            per_curve[(kind, noise)] = {kk: float(np.median(vv)) for kk, vv in hs.items()
                                        if isinstance(vv, np.ndarray) and vv.ndim == 1}
            check(f"per-curve shape: no harmonic collisions for {kind} at noise {noise}",
                  int(np.max(hs["n_collisions"])) == 0)
    check("per-curve shape: a clean square wave has a near-zero even share (not 0.08)",
          per_curve[("square", 0.0)]["even_share"] < 0.005)
    check("per-curve shape: odd-minus-even separates square from sinusoid by > 0.10",
          per_curve[("square", 0.0)]["odd_minus_even"]
          > per_curve[("sinusoid", 0.0)]["odd_minus_even"] + 0.10)
    check("per-curve shape: the separation survives noise that swamps the raw shares",
          per_curve[("square", 0.6)]["odd_minus_even"]
          > per_curve[("sinusoid", 0.6)]["odd_minus_even"] + 0.09)
    check("degenerate case is flagged via fundamental_power_fraction",
          harmonic_shape(np.ones(half), [1], p)["fundamental_power_fraction"] < 0.05)

    # §3.2: the cap must be reported, not silent.
    rng = np.random.RandomState(0)
    dom = dominant_frequencies(rng.randn(p, 16), p)
    check("dominant_frequencies reports whether the max_k cap bound",
          dom["cap_binding"] is True and dom["n_keep"] == 8)
    pure = np.stack([np.cos(2 * np.pi * 4 * n / p), np.sin(2 * np.pi * 4 * n / p)], axis=1)
    check("cap does NOT bind on a single-frequency embedding",
          dominant_frequencies(pure, p)["cap_binding"] is False)


# --------------------------------------------------------------------------- #
# RESEARCH_SPEC §3.3 — effective MLP weights, not W_E                          #
# --------------------------------------------------------------------------- #
def check_mlp_mechanism():
    print("\n-- MLP mechanism (RESEARCH_SPEC §3.3) --")
    import torch
    cfg = get_config("nanda", p=23, arch="mlp", d_mlp=16, seed=0)
    set_seed(0)
    model = build_model(cfg)
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    cur = effective_curves(state, cfg)
    p, d = cfg.p, cfg.d_model
    check("effective curves are [p, d_mlp]",
          cur["u_a"].shape == (p, cfg.d_mlp) and cur["out"].shape == (p, cfg.d_mlp))

    # Hand-computed reference: neuron i's a-curve is W_E[a] . W_in[:d, i].
    # The reference must be taken in float64, as effective_curves does. Computing it in the
    # stored float32 leaves a ~2e-9 rounding gap on a 128-term dot product, which is larger
    # than the 1e-9 tolerance -- that was a defect of this check, not of the module, and it
    # sat masked behind the harmonic-shape failure above until 2026-09-03.
    i, a = 3, 7
    f64 = lambda x: np.asarray(x, dtype=np.float64)          # noqa: E731
    ref = float(f64(state["W_E"][a]) @ f64(state["W_in"][:d, i]))
    check("u_a matches a hand-computed reference (float64)",
          abs(cur["u_a"][a, i] - ref) < 1e-12)
    ref_b = float(f64(state["W_E"][7]) @ f64(state["W_in"][d:, i]))
    check("u_b reads the SECOND half of W_in (float64)",
          abs(cur["u_b"][7, i] - ref_b) < 1e-12)
    check("the same reference taken in float32 agrees only to ~1e-8 (why float64 is used)",
          abs(cur["u_a"][a, i] - float(state["W_E"][a] @ state["W_in"][:d, i])) < 1e-7)

    # And the curves must reconstruct the model's real forward pass.
    with torch.no_grad():
        toks = torch.tensor([[5, 9, cfg.equals_token]])
        want = model.logits_last(toks).numpy()[0]
    act = np.maximum(cur["u_a"][5] + cur["u_b"][9] + cur["b_in"], 0.0)
    got = act @ state["W_out"] + state["b_out"]
    check("effective curves reproduce the model's forward pass",
          np.abs(got - want).max() < 1e-4)

    # Spectra: inject a known frequency and phase, recover both.
    k, phi = 4, 0.9
    nn = np.arange(p)
    curve = np.cos(2 * np.pi * k * nn / p - phi)[:, None]
    sp = curve_spectra(curve, p)
    check("curve_spectra recovers the injected frequency", int(sp["dominant_freq"][0]) == k)
    check("curve_spectra recovers the injected phase",
          abs(float(np.angle(np.exp(1j * (sp["dominant_phase"][0] - phi))))) < 1e-6)
    check("a pure sinusoid concentrates ~all power in one frequency",
          sp["dominant_fraction"][0] > 0.999)

    # H1 end-to-end on a SYNTHETIC circuit: phi_out = phi_a + phi_b by
    # construction => resultant length ~1 and the permutation null is beaten.
    rng = np.random.RandomState(0)
    n_neu = 40
    pa = rng.uniform(-np.pi, np.pi, n_neu)
    pb = rng.uniform(-np.pi, np.pi, n_neu)
    ks = rng.randint(1, (p - 1) // 2 + 1, n_neu)
    def build(ph):
        return np.stack([np.cos(2 * np.pi * ks[j] * nn / p - ph[j]) for j in range(n_neu)],
                        axis=1)
    sa, sb = curve_spectra(build(pa), p), curve_spectra(build(pb), p)
    so = curve_spectra(build(pa + pb), p)
    cls = classify_neurons(sa, sb, so, PROPOSED)
    check("synthetic circuit: every neuron classified structured",
          cls["n_structured"] == n_neu)
    pr = phase_relation(sa, sb, so, np.ones(n_neu, bool), n_boot=200, n_perm=200)
    check("synthetic circuit: phase relation is tight (R > 0.99)",
          pr["resultant_length"] > 0.99)
    check("synthetic circuit: beats the permutation null", pr["exceeds_null_q95"])

    # ... and a scrambled control must NOT pass, or the test proves nothing.
    so_bad = curve_spectra(build(rng.uniform(-np.pi, np.pi, n_neu)), p)
    pr_bad = phase_relation(sa, sb, so_bad, np.ones(n_neu, bool), n_boot=200, n_perm=200)
    check("scrambled control: phase relation is NOT tight",
          pr_bad["resultant_length"] < 0.5)

    # Thresholds must remain a human decision, and stricter must never select more.
    from grokverse.analysis.mlp_mechanism import SENSITIVITY
    counts = [classify_neurons(sa, sb, so, c)["n_structured"]
              for c in sorted((PROPOSED, *SENSITIVITY),
                              key=lambda c: c.min_variance_explained)]
    check("stricter variance thresholds never select MORE neurons",
          all(x >= y for x, y in zip(counts, counts[1:])))
    check("sensitivity criteria are shipped alongside the proposal", len(SENSITIVITY) >= 2)


# --------------------------------------------------------------------------- #
# RESEARCH_SPEC §3.6 / §3.8 — parameter matching and thread pinning            #
# --------------------------------------------------------------------------- #
def check_config_additions():
    print("\n-- config: parameter count, threads, run_id (§3.6, §3.8) --")
    txf = get_config("nanda")
    mlp = get_config("nanda", arch="mlp")
    matched = get_config("mlp_param_matched")
    check("transformer has 226,176 parameters", txf.n_params == 226_176)
    check("default MLP has 204,017 parameters", mlp.n_params == 204_017)
    check("MLP has 9.8% fewer parameters than the transformer",
          abs((1 - mlp.n_params / txf.n_params) - 0.098) < 0.001)
    check("d_mlp=572 matches the transformer to +0.02%",
          matched.n_params == 226_217
          and abs(matched.n_params / txf.n_params - 1) < 0.0005)

    # n_params must track the real modules, not a stale constant.
    set_seed(0)
    for cfg in (txf, mlp, matched):
        built = sum(prm.numel() for prm in build_model(cfg).parameters())
        check(f"n_params matches the built {cfg.arch} (d_mlp={cfg.d_mlp})",
              built == cfg.n_params)

    check("threads defaults to 1 (pinned for a reproducible matrix)", txf.threads == 1)
    check("threads is recorded in the run config", "threads" in txf.to_dict())

    # The param-matched control must not collide with the baseline run dir.
    check("d_mlp tag keeps the param-matched control in its own run dir",
          matched.run_id != get_config("nanda", arch="mlp").run_id
          and "dm572" in matched.run_id)
    check("existing run_ids are unchanged (no tag at the default d_mlp)",
          get_config("nanda", arch="mlp", train_frac=0.3, seed=0).run_id
          == "mlp_add_p113_wd1.0_frac0.3_seed0")
    check("canonical transformer run_id unchanged",
          get_config("nanda", train_frac=0.3, seed=0).run_id
          == "txf_add_p113_wd1.0_frac0.3_seed0")

    # A field added after a run was recorded must not invalidate that run.
    recorded = {k: v for k, v in txf.to_dict().items() if k != "threads"}
    diff, added = config_diff(recorded, txf.to_dict())
    check("a newly added config field is 'added since', not a mismatch",
          diff == {} and added == ["threads"])
    changed = dict(txf.to_dict()); changed["weight_decay"] = 0.5
    diff2, _ = config_diff(changed, txf.to_dict())
    check("a real config difference still blocks", "weight_decay" in diff2)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    main()
