"""Checks for analysis/mlp_mechanism.py + mlp_mechanism_activation.py (INTERFACES §5).

Run: ``python tests/test_mlp_mechanism.py`` from ``training/``.

Every input is SYNTHETIC or a freshly initialized model — no trained run is read, so this
file runs before, during and after the study. Each positive check is paired with a control
that must NOT pass, because a test only a correct implementation can pass is worth nothing
if a broken one passes it too.

THE ONE PREDICTION THIS FILE DELIBERATELY DOES *NOT* MAKE
---------------------------------------------------------
``sum_direction_share >> diff_direction_share`` for a single neuron. Rectifying
``cos(w a - phi_a) + cos(w b - phi_b)`` produces the ``(a+b)`` and the ``(a-b)`` cross term
with the SAME amplitude ``8/(3 pi^2)`` (``docs/MLP_MECHANISM_DERIVATION.md`` §4.1,
``tests/test_derivations.py``). The check below therefore asserts that the two are
approximately EQUAL, and a second check asserts that the "sum dominates" reading is false —
so the corrected prediction cannot be quietly re-broken. Sum-over-difference dominance is a
statement about the logits and the neuron population and is tested in
``analysis/logit_formula_fit``.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import mlp_mechanism as MM  # noqa: E402
from grokverse.analysis import mlp_mechanism_activation as MA  # noqa: E402
from grokverse.analysis.common import center_logits  # noqa: E402
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

P = 23                       # prime, so the odd harmonics of any k stay distinct
HALF = (P - 1) // 2
CROSS_TERM_AMPLITUDE = 8.0 / (3.0 * math.pi ** 2)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# synthetic circuits                                                           #
# --------------------------------------------------------------------------- #
def cos_curve(k, phi, p=P, amp=1.0) -> np.ndarray:
    """``amp * cos(2 pi k n / p - phi)`` in the §0 phase convention, as a column."""
    n = np.arange(p)
    return amp * np.cos(2 * np.pi * np.asarray(k)[None, :] * n[:, None] / p
                        - np.asarray(phi)[None, :])


def ideal_circuit(n_neurons=24, p=P, seed=0, out_phase="sum", out_freq_shift=0,
                  noise=0.0, b_in=0.0) -> dict:
    """A hand-built population: u_a, u_b at frequency k, out at phi_a + phi_b.

    ``out_phase="scrambled"`` randomizes the output phase; ``out_freq_shift`` moves the
    output curve to a different frequency — the two negative controls of H1.
    """
    rng = np.random.default_rng(seed)
    ks = rng.integers(1, HALF + 1, n_neurons)
    pa = rng.uniform(-np.pi, np.pi, n_neurons)
    pb = rng.uniform(-np.pi, np.pi, n_neurons)
    po = (pa + pb) if out_phase == "sum" else rng.uniform(-np.pi, np.pi, n_neurons)
    ks_out = np.where(ks + out_freq_shift > HALF, ks - out_freq_shift, ks + out_freq_shift)
    curves = {"u_a": cos_curve(ks, pa, p), "u_b": cos_curve(ks, pb, p),
              "out": cos_curve(ks_out, po, p),
              "b_in": np.full(n_neurons, float(b_in))}
    if noise > 0:
        for name in ("u_a", "u_b", "out"):
            curves[name] = curves[name] + noise * rng.standard_normal(curves[name].shape)
    curves.update({"k": ks, "phi_a": pa, "phi_b": pb, "phi_out": po})
    return curves


def single_neuron(k, phi_a, phi_b, p=P, b_in=0.0) -> dict:
    return {"u_a": cos_curve([k], [phi_a], p), "u_b": cos_curve([k], [phi_b], p),
            "out": cos_curve([k], [phi_a + phi_b], p), "b_in": np.array([float(b_in)])}


# --------------------------------------------------------------------------- #
# 1. neuron_tables on a hand-built ideal circuit                               #
# --------------------------------------------------------------------------- #
def check_neuron_tables():
    print("\n-- neuron_tables (INTERFACES 5, bullet 1) --")
    cur = ideal_circuit()
    t = MM.neuron_tables(cur, P)
    per = t["per_curve"]

    check("tables cover u_a, u_b and out", set(per) == set(MM.CURVE_NAMES))
    check("ideal circuit: u_a dominant frequency == the injected k",
          bool((per["u_a"]["dominant_frequency"] == cur["k"]).all()))
    check("ideal circuit: u_b and out share that frequency",
          bool(t["same_ab"].all()) and bool(t["same_ab_out"].all()))
    check("ideal circuit: dominant fraction ~ 1 on every curve",
          all(float(per[c]["dominant_fraction"].min()) > 0.999 for c in MM.CURVE_NAMES))
    check("ideal circuit: family fraction ~ 1 on every curve",
          all(float(per[c]["family_fraction"].min()) > 0.999 for c in MM.CURVE_NAMES))
    check("ideal circuit: the recovered phase equals the injected phi_a",
          float(np.abs(np.angle(np.exp(1j * (per["u_a"]["dominant_phase"] - cur["phi_a"]))))
                .max()) < 1e-8)

    # every metrics.py statistic is present, per curve
    wanted = {"dominant_fraction", "spectral_entropy", "participation_ratio",
              "inverse_participation_ratio_ours", "inverse_participation_ratio_doshi_rfft",
              "inverse_participation_ratio_doshi_fft", "periodicity_score_swaroop",
              "binarization_score_ours", "topk_concentration", "harmonic_shares",
              "dominant_frequency", "dominant_phase", "total_power"}
    check("every metrics.py statistic is in the table",
          wanted <= set(per["u_a"]["metrics"]))
    check("harmonic shares carry the discrete square-wave reference",
          "ideal_square_family_share" in per["u_a"]["metrics"]["harmonic_shares"])
    check("no harmonic collisions at prime p for a single fundamental",
          int(np.asarray(per["u_a"]["metrics"]["harmonic_shares"]["n_collisions"]).sum()) == 0)

    # top-3 frequencies and phases: a two-frequency curve must expose BOTH
    k1, k2 = 3, 7
    y = (cos_curve([k1], [0.4]) + 0.5 * cos_curve([k2], [-1.1]))
    t2 = MM.neuron_tables({"u_a": y, "u_b": y, "out": y, "b_in": np.zeros(1)}, P)
    top = t2["per_curve"]["u_a"]
    check("top-3 frequencies are ranked by power",
          [int(top["top_frequency"][0, 0]), int(top["top_frequency"][1, 0])] == [k1, k2])
    check("the phase of the 2nd frequency is recovered too",
          abs(float(np.angle(np.exp(1j * (top["top_phase"][1, 0] + 1.1))))) < 1e-8)
    check("top-3 power shares are decreasing",
          bool((np.diff(top["top_fraction"][:, 0]) <= 1e-12).all()))

    # contingency table
    con = t["contingency"]
    check("contingency counts partition the neurons",
          con["same_ab_and_same_out"] + con["same_ab_not_same_out"]
          + con["not_same_ab_but_out_matches_a"] + con["neither"] == con["n_neurons"])
    check("ideal circuit: the contingency table is all in the agreeing cell",
          con["same_ab_and_same_out"] == con["n_neurons"])

    # CONTROL: a scrambled-frequency output must break the output-frequency condition
    bad = ideal_circuit(out_freq_shift=1)
    tb = MM.neuron_tables(bad, P)
    check("CONTROL scrambled output frequency: same_ab still holds", bool(tb["same_ab"].all()))
    check("CONTROL scrambled output frequency: same_ab_out fails for every neuron",
          int(tb["same_ab_out"].sum()) == 0)
    check("CONTROL: its contingency table moves out of the agreeing cell",
          tb["contingency"]["same_ab_and_same_out"] == 0)

    # the summary must be JSON-writable (no arrays, no NaN leaking through)
    js = json.dumps(_jsonable(MM.summarize_tables(t)))
    check("summarize_tables is JSON-serializable and finite", "NaN" not in js and len(js) > 0)


def _jsonable(obj):
    from grokverse.analysis.common import _jsonable as conv
    return conv(obj)


# --------------------------------------------------------------------------- #
# 2. activation_analysis                                                       #
# --------------------------------------------------------------------------- #
def check_activation():
    print("\n-- activation_analysis (INTERFACES 5, bullet 2) --")
    k, phi_a, phi_b = 3, 0.7, -1.3
    cur = single_neuron(k, phi_a, phi_b)
    act = MM.activation_analysis(cur, P, [k])
    pn = act["per_neuron"]
    s = float(pn["sum_direction_share"][0])
    d = float(pn["diff_direction_share"][0])

    check("ideal neuron is alive", not bool(pn["dead"][0]))
    check("sum and diff direction shares are both substantial", s > 0.05 and d > 0.05)
    # MLP_MECHANISM_DERIVATION 4.1: EQUAL amplitudes, therefore equal shares.
    check("sum_direction_share ~= diff_direction_share (derivation 4.1)",
          abs(s - d) < 0.02 * max(s, d))
    check("CONTROL: the false 'sum >> diff' per-neuron prediction is NOT satisfied",
          not (s > 1.5 * d))

    # the analytic block decomposition equals mask_protocols.SumDirectionsOnly
    chk = act["summary"]["sum_projection_check"]
    check("the (a+b) power agrees with mask_protocols.SumDirectionsOnly",
          chk["n_checked"] >= 1 and chk["max_relative_difference"] < 1e-10)
    check("the (a-b) projection source is recorded",
          "DiffDirectionsOnly" in act["params"]["diff_projection_source"]
          or "local" in act["params"]["diff_projection_source"])

    # the measured cross-term amplitude must match the derivation's 8/(3 pi^2)
    n = np.arange(P)
    a2d = np.maximum(np.cos(2 * np.pi * k * n / P - phi_a)[:, None]
                     + np.cos(2 * np.pi * k * n / P - phi_b)[None, :], 0.0)
    m = ((a2d * np.cos(2 * np.pi * k * (n[:, None] + n[None, :]) / P
                       - (phi_a + phi_b))).sum() * 2 / P ** 2)
    check("the (a+b) cross-term amplitude is 8/(3 pi^2)",
          abs(m - CROSS_TERM_AMPLITUDE) < 5e-3)

    # swap symmetry
    same = MM.activation_analysis(single_neuron(k, 0.9, 0.9), P, [k])
    diff = MM.activation_analysis(single_neuron(k, 0.0, np.pi / 2), P, [k])
    check("swap_symmetry ~ 1 when phi_a == phi_b",
          abs(float(same["per_neuron"]["swap_symmetry"][0]) - 1.0) < 1e-9)
    check("CONTROL swap_symmetry < 1 when phi_a != phi_b",
          float(diff["per_neuron"]["swap_symmetry"][0]) < 0.9)

    # ev_hypothesized with the TRUE parameters
    check("ev_hypothesized(sinusoid) ~ 1 for a sinusoidal neuron",
          float(pn["ev_hypothesized_sinusoid"][0]) > 0.9999)
    check("CONTROL ev_hypothesized(square) is clearly worse for a sinusoidal neuron",
          float(pn["ev_hypothesized_square"][0]) < float(pn["ev_hypothesized_sinusoid"][0]) - 0.01)

    # a square-wave neuron is the mirror image
    n = np.arange(P)
    sq = np.where(np.cos(2 * np.pi * k * n / P - 0.3) >= 0, 1.0, -1.0)[:, None]
    sq_cur = {"u_a": sq, "u_b": sq, "out": sq, "b_in": np.zeros(1)}
    sq_act = MM.activation_analysis(sq_cur, P, [k])
    check("ev_hypothesized(square) ~ 1 for a square-wave neuron",
          float(sq_act["per_neuron"]["ev_hypothesized_square"][0]) > 0.9999)

    # dead neuron (a strongly negative bias, as in a real model): everything undefined,
    # nothing invented
    dead = {"u_a": cos_curve([k], [0.0]), "u_b": cos_curve([k], [0.0]),
            "out": cos_curve([k], [0.0]), "b_in": np.array([-9.0])}
    da = MM.activation_analysis(dead, P, [k])
    check("a dead neuron is flagged dead", bool(da["per_neuron"]["dead"][0]))
    check("a dead neuron's swap_symmetry is NaN, not a substitute value",
          bool(np.isnan(da["per_neuron"]["swap_symmetry"][0])))
    check("the summary counts the dead neuron", da["summary"]["n_dead"] == 1)
    flat = {"u_a": np.zeros((P, 1)), "u_b": cos_curve([k], [0.0]), "out": cos_curve([k], [0.0]),
            "b_in": np.zeros(1)}
    check("a constant (zero-variance) curve raises instead of returning a made-up fit",
          _raises(lambda: MM.activation_analysis(flat, P, [k])))

    # 2-D spectrum, top modes, exemplars
    check("the top-5 modes are reported per neuron", pn["top_mode_i"].shape[0] == 5)
    # ReLU(x) = (x + |x|)/2: the LINEAR half puts the single strongest non-constant mode on
    # the single-axis modes (amplitude 1/2 each), above the cross terms (8/(3 pi^2) = 0.27).
    # The same-frequency block modes are the cross terms and sit just below them.
    kinds = [str(x) for x in MA._mode_kind(pn["top_mode_i"][:, 0], pn["top_mode_j"][:, 0])]
    check(f"the strongest modes of an ideal neuron are the linear single-axis ones ({kinds[0]})",
          kinds[0] in ("single_axis_a", "single_axis_b"))
    check("the same-frequency (cross-term) modes appear in the top-5",
          "same_frequency" in kinds)
    ci, si = MA._cos_sin_rows(P)
    same_rows = {int(ci[k - 1]), int(si[k - 1])}
    check("and those same-frequency modes sit at the neuron's own frequency k",
          all({int(i), int(j)} <= same_rows
              for i, j, kind in zip(pn["top_mode_i"][:, 0], pn["top_mode_j"][:, 0], kinds)
              if kind == "same_frequency"))
    check("the top-mode shares are ordered",
          bool((np.diff(pn["top_mode_share"][:, 0]) <= 1e-12).all()))
    pop = ideal_circuit(n_neurons=20, seed=3)
    pa = MM.activation_analysis(pop, P, sorted(set(int(x) for x in pop["k"])), n_exemplars=4)
    check("at most n_exemplars 2-D spectra are stored", pa["exemplars"]["power2d"].shape[2] <= 4)
    check("exemplar spectra are [p, p, n]", pa["exemplars"]["power2d"].shape[:2] == (P, P))
    check("chunking does not change any per-neuron number",
          _same_arrays(pa["per_neuron"],
                       MM.activation_analysis(pop, P, sorted(set(int(x) for x in pop["k"])),
                                              n_exemplars=4, chunk=3)["per_neuron"]))


def _same_arrays(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(np.allclose(np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float),
                           rtol=1e-12, atol=1e-12, equal_nan=True) for k in a)


# --------------------------------------------------------------------------- #
# 3. logit_contributions                                                       #
# --------------------------------------------------------------------------- #
def check_logit_contributions():
    print("\n-- logit_contributions (INTERFACES 5, bullet 3) --")
    rng = np.random.default_rng(7)
    n_neu = 12
    cur = ideal_circuit(n_neurons=n_neu, seed=5, b_in=-0.2)
    W_out = rng.standard_normal((n_neu, P)) * 0.5
    b_out = rng.standard_normal(P) * 0.1
    res = MM.logit_contributions(cur, W_out, b_out, P)

    # brute-force reference forward pass
    act = np.maximum(cur["u_a"][:, None, :] + cur["u_b"][None, :, :] + cur["b_in"], 0.0)
    L = np.einsum("abn,nc->abc", act, W_out) + b_out
    Lc = center_logits(L)
    check("the centered logit tensor is reproduced exactly",
          float(np.abs(res["logits_centered"] - Lc).max()) < 1e-9)

    cls = (np.arange(P)[:, None] + np.arange(P)[None, :]) % P
    correct = L[np.arange(P)[:, None], np.arange(P)[None, :], cls]
    ident = res["identities"]
    check("per-neuron mean correct-logit contributions sum to the mean correct logit",
          abs(ident["mean_correct_logit"] - float(correct.mean())) < 1e-9)
    check("the logit-variance shares plus the bias share sum to exactly 1",
          abs(ident["total_variance_share"] - 1.0) < 1e-9)
    margin = Lc[np.arange(P)[:, None], np.arange(P)[None, :], cls]
    check("the margin contributions sum to the mean correct-class margin",
          abs(ident["mean_correct_class_margin"] - float(margin.mean())) < 1e-9)
    check("the margin shares sum to 1",
          abs(float(res["per_neuron"]["margin_share"].sum()) - 1.0) < 1e-9)
    check("chunking does not change the contributions",
          _same_arrays(res["per_neuron"],
                       MM.logit_contributions(cur, W_out, b_out, P, chunk=5)["per_neuron"]))

    # CONTROL: a neuron whose readout row is zero contributes nothing
    W_zero = W_out.copy()
    W_zero[3] = 0.0
    r0 = MM.logit_contributions(cur, W_zero, b_out, P)
    check("CONTROL: a neuron with a zero readout row has zero variance share",
          abs(float(r0["per_neuron"]["logit_variance_share"][3])) < 1e-12)
    check("CONTROL: and zero mean correct-logit contribution",
          abs(float(r0["per_neuron"]["mean_correct_logit_contribution"][3])) < 1e-12)
    check("the wrong W_out shape is rejected",
          _raises(lambda: MM.logit_contributions(cur, W_out.T, b_out, P)))


def _raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# --------------------------------------------------------------------------- #
# 4. structured_neuron_definitions (docs/PREREGISTRATION.md 4.2)               #
# --------------------------------------------------------------------------- #
def check_structured_definitions():
    print("\n-- structured_neuron_definitions (PREREGISTRATION 4.2) --")
    cur = ideal_circuit(n_neurons=24, seed=11)
    ks = sorted(set(int(x) for x in cur["k"]))
    tables = MM.neuron_tables(cur, P)
    act = MM.activation_analysis(cur, P, ks)
    defs = MM.structured_neuron_definitions(tables, act)

    primary = defs["definitions"][MM.PRIMARY_DEFINITION]
    check("the primary definition is the pre-registered one",
          defs["primary"] == "primary_family_0.50"
          and primary["params"]["family_threshold"] == 0.50)
    check("ideal circuit: every neuron passes the PRIMARY definition",
          primary["n"] == tables["n_neurons"])
    check("the mandatory sensitivity variants are present",
          {"sensitivity_family_0.30", "sensitivity_family_0.70", "top1_0.50",
           "ipr_doshi_rank_matched", "periodicity_swaroop_gt12",
           "periodicity_swaroop_lt5"} <= set(defs["definitions"]))
    check("Doshi's IPR carries the 'source states no threshold' label",
          "NOT FOUND IN SOURCE" in
          defs["definitions"]["ipr_doshi_rank_matched"]["params"]["threshold_source"])
    check("Swaroop's 12/5 cuts are labelled as his p=97 histogram cuts",
          "post-hoc" in defs["definitions"]["periodicity_swaroop_gt12"]["params"]["caveat"])
    check("counts per frequency are reported per definition",
          sum(primary["counts_per_frequency"].values()) == primary["n"])
    check("pairwise Jaccard overlaps are reported",
          len(defs["jaccard"]) == len(defs["definitions"]) * (len(defs["definitions"]) - 1) // 2)

    # CONTROL 1: a scrambled OUTPUT FREQUENCY must fail condition (4) of 4.2
    bad = ideal_circuit(n_neurons=24, seed=11, out_freq_shift=1)
    tb = MM.neuron_tables(bad, P)
    ab = MM.activation_analysis(bad, P, ks)
    db = MM.structured_neuron_definitions(tb, ab)
    check("CONTROL scrambled output frequency: the primary definition selects nobody",
          db["definitions"][MM.PRIMARY_DEFINITION]["n"] == 0)

    # CONTROL 2: a stricter family threshold must never select more, and 0.99 fewer than 0.50
    noisy = ideal_circuit(n_neurons=48, seed=2, noise=0.75)
    tn = MM.neuron_tables(noisy, P)
    an = MM.activation_analysis(noisy, P, sorted(set(int(x) for x in noisy["k"])))
    counts = [MM.structured_neuron_definitions(tn, an, family_threshold=t,
                                               family_sensitivity=())
              ["definitions"][f"primary_family_{t:.2f}"]["n"]
              for t in (0.30, 0.50, 0.70, 0.99)]
    check(f"family thresholds 0.30/0.50/0.70/0.99 select {counts} (monotone non-increasing)",
          all(x >= y for x, y in zip(counts, counts[1:])))
    check("a family threshold of 0.99 selects strictly FEWER neurons than 0.50",
          counts[3] < counts[1])
    check("the noisy population is not degenerate (0.50 selects some neurons)", counts[1] > 0)

    # a dead neuron can never be structured (condition 1 of 4.2)
    dead_cur = ideal_circuit(n_neurons=6, seed=4)
    dead_cur["b_in"] = np.array([-9.0] * 6)
    td = MM.neuron_tables(dead_cur, P)
    ad = MM.activation_analysis(dead_cur, P, sorted(set(int(x) for x in dead_cur["k"])))
    dd = MM.structured_neuron_definitions(td, ad)
    check("CONTROL: dead neurons are excluded even when every curve is perfect",
          bool(ad["per_neuron"]["dead"].all())
          and dd["definitions"][MM.PRIMARY_DEFINITION]["n"] == 0)

    # phase relation on the primary set vs the scrambled-phase control
    spec = {c: MM.curve_spectra(cur[c], P) for c in MM.CURVE_NAMES}
    pr = MM.phase_relation(spec["u_a"], spec["u_b"], spec["out"], primary["mask"],
                           n_boot=200, n_perm=200)
    scr = ideal_circuit(n_neurons=24, seed=11, out_phase="scrambled")
    spec2 = {c: MM.curve_spectra(scr[c], P) for c in MM.CURVE_NAMES}
    pr2 = MM.phase_relation(spec2["u_a"], spec2["u_b"], spec2["out"],
                            np.ones(24, bool), n_boot=200, n_perm=200)
    check("ideal circuit: the phase relation on the primary set is tight (R > 0.99)",
          pr["resultant_length"] > 0.99 and pr["exceeds_null_q95"])
    check("CONTROL scrambled output phase: the phase relation is NOT tight",
          pr2["resultant_length"] < 0.5)


# --------------------------------------------------------------------------- #
# 5. both architectures + the run-level entry point                            #
# --------------------------------------------------------------------------- #
def _make_run(tmp: Path, arch: str, seed: int = 0):
    cfg = get_config("nanda", p=P, arch=arch, d_mlp=16, seed=seed)
    set_seed(seed)
    model = build_model(cfg)
    model.eval()
    run_dir = tmp / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "git_commit": "synthetic-test"}))
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    state = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    return cfg, model, state, run_dir


def check_architectures(tmp: Path):
    print("\n-- both MLP architectures (INTERFACES 5, last bullet) --")
    for arch in ("mlp", "mlp_twohot"):
        cfg, model, state, _ = _make_run(tmp, arch)
        cur = MM.effective_curves(state, cfg)
        check(f"{arch}: effective curves are [p, d_mlp]",
              cur["u_a"].shape == (P, cfg.d_mlp) and cur["u_b"].shape == (P, cfg.d_mlp))
        a, b = 5, 9
        with torch.no_grad():
            toks = torch.tensor([[a, b, cfg.equals_token]])
            want = model.logits_last(toks).numpy()[0][:P]
        act = np.maximum(cur["u_a"][a] + cur["u_b"][b] + cur["b_in"], 0.0)
        got = act @ state["W_out"] + state["b_out"]
        check(f"{arch}: the effective curves reproduce the model's forward pass",
              float(np.abs(got - want).max()) < 1e-4)
    check("the two-hot a-curve IS the first half of W_in",
          float(np.abs(MM.effective_curves(state, cfg)["u_a"] - state["W_in"][:P]).max()) == 0.0)
    check("an unsupported arch is rejected by name",
          _raises(lambda: MM.effective_curves(state, get_config("nanda", p=P, arch="transformer"))))


def check_analyse(tmp: Path):
    print("\n-- analyse(): run-level entry point + output contract --")
    for arch in ("mlp", "mlp_twohot"):
        _cfg, _model, _state, run_dir = _make_run(tmp, arch, seed=1)
        payload, path = MM.analyse(run_dir, None, MM.PRIMARY_KEY_RULE, 0,
                                   n_boot=50, n_perm=50, n_exemplars=4)
        check(f"{arch}: the result lands at analysis/mlp_mechanism/final.json",
              path == run_dir / "analysis" / "mlp_mechanism" / "final.json" and path.exists())
        check(f"{arch}: the npz of per-neuron arrays is written next to it",
              (path.with_suffix(".npz")).exists())
        on_disk = json.loads(path.read_text())
        check(f"{arch}: the provenance envelope is complete",
              all(on_disk.get(k) for k in ("module", "module_version", "run_id", "arch", "p",
                                           "checkpoint_sha256", "created_utc"))
              and on_disk["arch"] == arch)
        res = on_disk["results"]
        check(f"{arch}: every INTERFACES 5 result block is present",
              {"neuron_tables", "wave_fits", "activation", "logit_contributions",
               "structured_neurons"} <= set(res))
        check(f"{arch}: the structured-neuron fraction is reported per definition",
              MM.PRIMARY_DEFINITION in res["structured_neurons"]["definitions"])
        check(f"{arch}: the key-frequency rule (and any fallback) is recorded",
              "key_rule" in on_disk["params"]["key_frequency_selection"])
        with np.load(path.with_suffix(".npz")) as z:
            keys = set(z.files)
            n = int(on_disk["results"]["n_neurons"])
            check(f"{arch}: per-neuron fit results are stored (best-by-AIC, r2, alpha3/alpha1)",
                  {"fit__u_a__best_by_aic", "fit__u_a__r2__square",
                   "fit__u_a__alpha3_over_alpha1"} <= keys
                  and z["fit__u_a__best_by_aic"].shape == (n,))
            check(f"{arch}: the structured masks are stored",
                  f"mask__{MM.PRIMARY_DEFINITION}" in keys and z["mask__alive"].shape == (n,))
            check(f"{arch}: no per-neuron [p,p,p] tensor is stored",
                  all(z[k].size <= P * P * 16 for k in keys))
            check(f"{arch}: at most 16 exemplar 2-D spectra are stored",
                  z["act__exemplar_power2d"].shape[2] <= 16)


def check_key_frequency_fallback(tmp: Path):
    print("\n-- key-frequency selection with analysis.key_frequencies absent --")
    cfg, _model, state, run_dir = _make_run(tmp, "mlp", seed=2)
    keys, info = MM.resolve_key_frequencies(run_dir, None, MM.PRIMARY_KEY_RULE, state, cfg)
    check("the requested (pre-registered primary) rule is recorded verbatim",
          info["requested_key_rule"] == "nanda" and info["primary_key_rule"] == "nanda")
    check("a key set is returned either way", len(keys) >= 1 and all(1 <= k <= HALF for k in keys))
    try:
        import grokverse.analysis.key_frequencies  # noqa: F401
        have = True
    except ImportError:
        have = False
    if have:
        check("key_frequencies exists, so no fallback was used", not info["fallback_used"])
    else:
        check("key_frequencies is absent, so the embedding-top-8 fallback fired and is recorded",
              info["fallback_used"] and info["key_rule"] == "embedding_top8"
              and "note" in info and "cap_binding" in info)
    cfg2 = get_config("nanda", p=P, arch="mlp_twohot", d_mlp=16)
    set_seed(0)
    st2 = {k: v.detach().numpy().astype(np.float64)
           for k, v in build_model(cfg2).state_dict().items()}
    k2, i2 = MM.resolve_key_frequencies(run_dir, None, MM.PRIMARY_KEY_RULE, st2, cfg2)
    check("the two-hot model (which has no W_E) still resolves a key set",
          len(k2) >= 1 and (not i2["fallback_used"] or "W_in" in i2["object"]))


# --------------------------------------------------------------------------- #
def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_neuron_tables()
        check_activation()
        check_logit_contributions()
        check_structured_definitions()
        check_architectures(tmp)
        check_key_frequency_fallback(tmp)
        check_analyse(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
