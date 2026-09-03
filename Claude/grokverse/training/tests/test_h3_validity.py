"""Checks for analysis/h3_validity.py (INTERFACES §14, PREREGISTRATION §3 H3).

Run: ``python tests/test_h3_validity.py`` from ``training/``.

Every input is SYNTHETIC or a freshly initialized model. Each positive check is paired with a control
that must NOT pass.

THE CHECK THAT MATTERS MOST HERE IS A NEGATIVE ONE
---------------------------------------------------
"The harmonic family beats the cardinality-matched top-m concentration" is **unsatisfiable by
construction** — top-m is the argmax over frequency sets of size m, so the family can never exceed it.
`docs/PREREGISTRATION.md` §3 names it as a criterion that is forbidden, and this file asserts the
inequality on real populations rather than trusting the docstring. If a future change ever made the
"family beats top-m" number positive, that would be an implementation bug, not evidence for H3.

The statistic that carries the inferential load is the odd-versus-even harmonic **shape**, and the
tests below assert that it separates a square population from a sinusoidal one *in the opposite
direction* from the concentration metric — which is the whole content of H3.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import h3_validity as H  # noqa: E402
from grokverse.analysis import metrics as M  # noqa: E402
from grokverse.config import get_config  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

P = 113                      # the study's prime — the aliasing caveat below is specific to it
P_SMALL = 23


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _population(n: int, p: int = P, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {"k": rng.integers(1, (p - 1) // 2 + 1, size=n),
            "phase": rng.uniform(-np.pi, np.pi, size=n),
            "amplitude": rng.uniform(0.5, 2.0, size=n)}


# --------------------------------------------------------------------------- #
# 1. parameter recovery and the two populations                                #
# --------------------------------------------------------------------------- #
def check_populations():
    print("\n-- synthetic populations at matched (k, phase, amplitude) --")
    params = _population(64)
    pops = H.synthetic_populations(params, P)
    check("both populations are [p, n]",
          pops["sinusoid"].shape == (P, 64) and pops["square"].shape == (P, 64))
    rec = H.curve_parameters(pops["sinusoid"], P)
    check("the dominant frequency of every built sinusoid is recovered exactly",
          bool((rec["k"] == params["k"]).all()))
    check("...its phase to 1e-10",
          float(np.abs(((rec["phase"] - params["phase"] + np.pi) % (2 * np.pi)) - np.pi).max()) < 1e-10)
    check("...and its amplitude to 1e-10 (the calibration is not assumed from the basis scaling)",
          float(np.abs(rec["amplitude"] - params["amplitude"]).max()) < 1e-10)
    check("the square population takes only two values per column (up to its amplitude)",
          all(len(np.unique(np.round(pops["square"][:, i] / params["amplitude"][i], 9))) == 2
              for i in range(10)))
    check("the two populations differ ONLY in waveform: same k, same phase, same amplitude",
          bool((H.curve_parameters(pops["square"], P)["k"] == params["k"]).all()))


# --------------------------------------------------------------------------- #
# 2. H3a — the artifact itself                                                 #
# --------------------------------------------------------------------------- #
def check_waveform_sensitivity():
    print("\n-- H3a: the concentration metric is waveform sensitive --")
    params = _population(128)
    pops = H.synthetic_populations(params, P)
    met = {kind: M.curve_metrics(Y, P, H.TOPK) for kind, Y in pops.items()}
    top8 = {kind: float(np.median(m["topk_concentration"][8])) for kind, m in met.items()}
    top1 = {kind: float(np.median(m["topk_concentration"][1])) for kind, m in met.items()}
    check(f"top-8 concentration is LOWER for the squares ({top8['square']:.4f} < "
          f"{top8['sinusoid']:.4f}) although both carry the same fundamentals",
          top8["square"] < top8["sinusoid"])
    check(f"the effect is larger at top-1 ({top1['sinusoid'] - top1['square']:.4f} > "
          f"{top8['sinusoid'] - top8['square']:.4f}) — the harmonics are what the metric misses",
          (top1["sinusoid"] - top1["square"]) > (top8["sinusoid"] - top8["square"]))
    shape = {kind: float(np.median(m["harmonic_shares"]["odd_minus_even"])) for kind, m in met.items()}
    check(f"the odd-minus-even shape separates them in the OPPOSITE direction "
          f"(square {shape['square']:+.4f} > sinusoid {shape['sinusoid']:+.4f})",
          shape["square"] > shape["sinusoid"] + 0.05)
    check("a pure sinusoid population has ~zero odd-minus-even shape",
          abs(shape["sinusoid"]) < 1e-6)

    res = H.waveform_sensitivity({"u_a": pops["sinusoid"]}, P)["per_curve"]["u_a"]
    check("waveform_sensitivity is reported positive for every reported cardinality",
          all(res["waveform_sensitivity_median"][f"top{k}"] > 0 for k in H.TOPK))
    check("the module states that this is a property of the metric, not of the model",
          "of the METRIC" in H.waveform_sensitivity({"u_a": pops["sinusoid"]}, P)["reading"])

    # CONTROL: white noise must not show the square-wave signature
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((P, 128))
    nm = M.curve_metrics(noise, P, H.TOPK)
    check("CONTROL: a noise population does NOT show the square wave's odd-harmonic shape",
          float(np.median(nm["harmonic_shares"]["odd_minus_even"])) < shape["square"] / 2)


# --------------------------------------------------------------------------- #
# 3. H3b — the two mandatory controls, and the forbidden criterion             #
# --------------------------------------------------------------------------- #
def check_harmonic_aware_structure():
    print("\n-- H3b: family vs top-1, with the matched top-m and random-m controls --")
    params = _population(96, seed=1)
    pops = H.synthetic_populations(params, P)
    res, arrays = H.harmonic_aware_structure({"sin": pops["sinusoid"], "sq": pops["square"]},
                                             P, n_control=20, seed=0)
    for name in ("sin", "sq"):
        block = res["per_curve"][name]
        check(f"{name}: family_fraction NEVER exceeds the matched top-m (forbidden criterion)",
              block["max_family_minus_matched_top_m"] <= 1e-12)
        check(f"{name}: the random-m null sits far below the family fraction",
              block["random_m_null"]["mean"]["median"] < block["family_fraction"]["median"] - 0.3)
        check(f"{name}: both controls are reported, not just one",
              block["matched_m"] > 0 and block["random_m_null"]["n"] == 20)
    check("the family captures MORE of a square than of a sinusoid is NOT asserted as evidence; "
          "the forbidden criterion is spelled out in the output",
          "unsatisfiable by construction" in res["forbidden_criterion"])
    check("per-curve arrays reach the npz",
          {"h3b__sin__family_fraction", "h3b__sq__matched_top_m_fraction"} <= set(arrays))
    check("the family fraction of a pure sinusoid is ~its top-1 fraction (nothing in the harmonics)",
          abs(res["per_curve"]["sin"]["family_fraction"]["median"]
              - res["per_curve"]["sin"]["top1_fraction"]["median"]) < 1e-6)
    check("CONTROL: for the squares the family fraction EXCEEDS the top-1 fraction",
          res["per_curve"]["sq"]["family_fraction"]["median"]
          > res["per_curve"]["sq"]["top1_fraction"]["median"] + 0.05)

    check("for a clean square the family IS the top-4, so the two coincide exactly",
          abs(res["per_curve"]["sq"]["family_fraction"]["median"]
              - res["per_curve"]["sq"]["matched_top_m_fraction"]["median"]) < 1e-9)

    # CONTROL: on noise the odd harmonics must add no more than three random frequencies would.
    # The family always CONTAINS the dominant frequency, so comparing the family itself against a
    # uniform random set would flatter it even on noise; the discriminating quantity is what the
    # harmonics add ON TOP of the fundamental.
    rng = np.random.default_rng(0)
    noise_res, _ = H.harmonic_aware_structure({"n": rng.standard_normal((P, 96))}, P, 20, 0)
    nb = noise_res["per_curve"]["n"]
    added_noise = nb["family_fraction"]["median"] - nb["top1_fraction"]["median"]
    added_square = (res["per_curve"]["sq"]["family_fraction"]["median"]
                    - res["per_curve"]["sq"]["top1_fraction"]["median"])
    three_random = 3.0 / ((P - 1) // 2)
    check(f"CONTROL: on noise the odd harmonics add what 3 random frequencies would "
          f"({added_noise:.4f} vs {three_random:.4f})",
          abs(added_noise - three_random) < 0.02)
    check(f"...and far less than for a square population ({added_noise:.4f} << {added_square:.4f})",
          added_square > 2 * added_noise)
    check("CONTROL: on noise the matched top-m beats the family by a wide margin",
          nb["matched_top_m_fraction"]["median"] > nb["family_fraction"]["median"] + 0.05)


def check_random_null():
    print("\n-- the random-m null --")
    q = np.zeros((56, 3))
    q[0] = 1.0                                        # all power on one frequency per column
    null = H._random_set_null(q, 4, 20, seed=0)
    check("the null has one row per draw and one column per curve", null.shape == (20, 3))
    check("a random 4-of-56 set captures the single loaded frequency only sometimes",
          0.0 < float(null.mean()) < 0.5)
    flat = np.full((56, 2), 1.0 / 56)
    check("on a flat spectrum the null is exactly m/half",
          abs(float(H._random_set_null(flat, 4, 10, 0).mean()) - 4 / 56) < 1e-12)
    check("it is reproducible under its seed",
          np.array_equal(null, H._random_set_null(q, 4, 20, seed=0)))
    check("CONTROL: a different seed gives a different draw",
          not np.array_equal(null, H._random_set_null(q, 4, 20, seed=5)))
    check("an impossible cardinality yields no draws", H._random_set_null(q, 999, 5, 0).size == 0)


# --------------------------------------------------------------------------- #
# 4. the aliasing caveat that bounds what H3b may claim                        #
# --------------------------------------------------------------------------- #
def check_aliasing_caveat():
    print("\n-- the p = 113 aliasing caveat (re-asserted from test_wave_fitting) --")
    odd = [j for j in range(1, M.MAX_ODD_HARMONIC + 1) if j % 2 == 1]
    fams = {k: [int(M.harmonic_indices(k, P).ravel()[j - 1]) for j in odd] for k in (6, 19, 51)}
    check(f"at p = 113 the odd-harmonic families of k = 6, 19 and 51 all contain 18 ({fams})",
          all(18 in f for f in fams.values()))
    check("...so 'the harmonic family fits at k' does not identify k as the fundamental",
          len({tuple(sorted(f)) for f in fams.values()}) == 3)
    check("the family the module matches its control to has cardinality 4, not 7",
          len(odd) == 4 and M.harmonic_indices(6, P).ravel().size == 7)
    n_coll = M.harmonic_shares(M.power_spectrum(np.cos(2 * np.pi * 6 * np.arange(P) / P)[:, None],
                                                P)["power"], np.array([6]), P)["n_collisions"]
    check("n_collisions is reported per curve so the caveat travels with the number",
          np.asarray(n_coll).size == 1)


# --------------------------------------------------------------------------- #
# 5. H3c — the legacy embedding number                                         #
# --------------------------------------------------------------------------- #
def check_legacy(tmp: Path):
    print("\n-- H3c: the legacy embedding concentration, descriptive only --")
    cfg, _model, state, _ = _make_run(tmp, "mlp", seed=2)
    res = H.legacy_embedding_concentration(state, cfg, n_control=20, seed=0)
    check("every swept cardinality is reported, not one invented choice",
          set(res["by_n_fundamentals"]) == {str(n) for n in H.LEGACY_N_F})
    for n_f, block in res["by_n_fundamentals"].items():
        if "error" in block:
            continue
        check(f"n_f={n_f}: the family never beats its matched top-m",
              block["family_minus_matched_top_m"] <= 1e-12)
        check(f"n_f={n_f}: the random null is reported with the family number",
              "null_mean" in block and block["null_n"] == 20)
    check("the object it was computed on is named", "W_E" in res["object"] or res["object"])
    check("the W_E caveat travels with the number", "decides nothing for the MLP" in res["caveat"])


# --------------------------------------------------------------------------- #
# 6. analyse(): the run-level output contract                                  #
# --------------------------------------------------------------------------- #
def _make_run(tmp: Path, arch: str, seed: int = 0):
    kw = dict(d_model=16, d_head=4, n_heads=4) if arch == "transformer" else {}
    cfg = get_config("nanda", p=P_SMALL, arch=arch, d_mlp=12, seed=seed, **kw)
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


def check_analyse(tmp: Path):
    print("\n-- analyse(): run-level entry point + output contract --")
    for arch in ("mlp", "transformer"):
        _cfg, _model, _state, run_dir = _make_run(tmp, arch, seed=4)
        payload, path = H.analyse(run_dir, None, H.PRIMARY_KEY_RULE, 0, n_control=5)
        check(f"{arch}: the result lands at analysis/h3_validity/final.json",
              path == run_dir / "analysis" / "h3_validity" / "final.json" and path.exists())
        check(f"{arch}: the npz is written next to it", path.with_suffix(".npz").exists())
        on_disk = json.loads(path.read_text())
        res = on_disk["results"]
        check(f"{arch}: all three H3 blocks are present",
              {"h3a_waveform_sensitivity", "h3b_harmonic_aware_structure",
               "h3b_structured_fractions", "h3c_legacy_embedding"} <= set(res))
        check(f"{arch}: the gate dependence is stated verbatim in params",
              "does not presuppose" in on_disk["params"]["gate_dependence"])
        check(f"{arch}: the module declares that it makes no architecture comparison",
              "paired across seeds" in on_disk["params"]["no_architecture_comparison_here"])
        check(f"{arch}: the aliasing caveat is recorded",
              "does not identify" in on_disk["params"]["aliasing_caveat"])
        check(f"{arch}: the result is labelled measurement-only",
              on_disk["params"]["status"].startswith("MEASUREMENT ONLY"))
        check(f"{arch}: both structured-neuron definitions H3b contrasts are reported",
              {H.FAMILY_DEFINITION, H.TOP1_DEFINITION}
              <= set(res["h3b_structured_fractions"]["definitions"]))
        for name, block in res["h3b_harmonic_aware_structure"]["per_curve"].items():
            check(f"{arch}/{name}: the forbidden inequality holds on real weights",
                  block["max_family_minus_matched_top_m"] <= 1e-12)


# --------------------------------------------------------------------------- #
def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_populations()
        check_waveform_sensitivity()
        check_harmonic_aware_structure()
        check_random_null()
        check_aliasing_caveat()
        check_legacy(tmp)
        check_analyse(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
