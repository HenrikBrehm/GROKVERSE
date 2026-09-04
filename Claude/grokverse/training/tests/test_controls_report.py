"""Checks for analysis/controls_report.py (STATISTICAL_ANALYSIS_PLAN §7–§8, master prompt §14).

Run: ``python tests/test_controls_report.py`` from ``training/``.

Synthetic aggregate rows only, with effects planted so the decomposition's answer is known.

WHY THE FACTORIAL NEEDS ITS OWN TESTS
--------------------------------------
Master prompt §14 exists because the project's own history contains the mistake it guards against: a
speed ratio moved from ~2.9× to ~1.3× between two settings that differed in **both** Grokfast and the
training fraction, and that was once read as a statement about Grokfast. A 2 × 2 decomposition is only
worth anything if it actually separates the two knobs, so the checks below plant a pure Grokfast
effect, a pure training-fraction effect, and a pure interaction in turn, and assert that each shows up
in the right term and **not** in the others.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import controls_report as CR  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _row(arch, seed, gf, frac, value, d_mlp=512, **over):
    # the run_id must be unique per RUN — real ids carry the width (…_dm572_…), and without it the
    # default-width and parameter-matched MLPs collide in compute()'s merge-by-run_id
    row = {"run_id": f"{arch}_gf{int(gf)}_f{frac}_dm{d_mlp}_s{seed}", "arch": arch, "seed": seed,
           "grokfast": gf, "train_frac": frac, "d_mlp": d_mlp,
           "tag": "step025000", "step": 25000, "steps_completed": 25000,
           "generalization_step": value, "memorization_step": 100,
           "structured_fraction_of_live": 0.3, "phase_R": 0.9,
           "u_a__fraction_best_aic_square": 0.4, "u_a__fraction_best_aic_odd_harmonics": 0.1}
    row.update(over)
    return row


def _agg(tmp: Path, rows, name="a") -> Path:
    d = tmp / name / "aggregate"
    d.mkdir(parents=True, exist_ok=True)
    for module in CR.MODULES:
        (d / f"{module}.json").write_text(json.dumps({"module": module, "rows": rows}),
                                          encoding="utf-8")
    return d


def _cells(arch, base, gf_eff, fr_eff, inter, seeds=(0, 1, 2)):
    """Four cells with a planted effect decomposition and a little per-seed offset."""
    rows = []
    for s in seeds:
        off = 10 * s
        rows += [
            _row(arch, s, False, 0.3, base + off),
            _row(arch, s, True, 0.3, base + off + gf_eff),
            _row(arch, s, False, 0.5, base + off + fr_eff),
            _row(arch, s, True, 0.5, base + off + gf_eff + fr_eff + inter),
        ]
    return rows


def check_separation(tmp: Path):
    print("\n-- the 2x2 decomposition separates the two knobs --")
    pure_gf = _cells("transformer", 5000, gf_eff=-800, fr_eff=0, inter=0)
    b = CR.factorial(pure_gf, "generalization_step",
                     CR.METRICS["generalization_step"])["per_architecture"]["transformer"]
    check("a pure Grokfast effect lands in the Grokfast main effect",
          abs(b["main_effect_grokfast"]["mean"] - (-800)) < 1e-9)
    check("...and NOT in the training-fraction effect",
          abs(b["main_effect_train_frac"]["mean"]) < 1e-9)
    check("...and NOT in the interaction", abs(b["interaction"]["mean"]) < 1e-9)

    pure_fr = _cells("transformer", 5000, gf_eff=0, fr_eff=-6000, inter=0)
    b2 = CR.factorial(pure_fr, "generalization_step",
                      CR.METRICS["generalization_step"])["per_architecture"]["transformer"]
    check("a pure training-fraction effect lands in its own term",
          abs(b2["main_effect_train_frac"]["mean"] - (-6000)) < 1e-9
          and abs(b2["main_effect_grokfast"]["mean"]) < 1e-9)

    pure_int = _cells("transformer", 5000, gf_eff=0, fr_eff=0, inter=400)
    b3 = CR.factorial(pure_int, "generalization_step",
                      CR.METRICS["generalization_step"])["per_architecture"]["transformer"]
    check("a pure interaction lands in the interaction term",
          abs(b3["interaction"]["mean"] - 400) < 1e-9)
    check("...and splits evenly across the two main effects, as a 2x2 must",
          abs(b3["main_effect_grokfast"]["mean"] - 200) < 1e-9
          and abs(b3["main_effect_train_frac"]["mean"] - 200) < 1e-9)
    check("the simple effects at each level are reported, not only the mains",
          {"simple_effect_grokfast_at_frac0.3",
           "simple_effect_grokfast_at_frac0.5"} <= set(b3))
    check("the small-sample caveat travels with the decomposition",
          "intervals are wide" in b3["reading"] and "unless this decomposition separates" in b3["reading"])


def check_incomplete_cells(tmp: Path):
    print("\n-- a missing cell is reported, never silently averaged over --")
    rows = [r for r in _cells("mlp", 100, 5, 5, 0) if not (r["grokfast"] and r["train_frac"] == 0.5)]
    b = CR.factorial(rows, "generalization_step",
                     CR.METRICS["generalization_step"])["per_architecture"]["mlp"]
    check("with one cell missing the decomposition is not computed",
          "not_computed" in b and b["n_pairs"] == 0)
    check("...and the cells that ARE present are listed",
          any(v for v in b["cells_present"].values()))
    partial = _cells("mlp", 100, 5, 5, 0, seeds=(0, 1, 2))
    partial = [r for r in partial if not (r["seed"] == 2 and r["grokfast"])]
    b2 = CR.factorial(partial, "generalization_step",
                      CR.METRICS["generalization_step"])["per_architecture"]["mlp"]
    check("a seed missing from one cell is dropped from the pairing and counted",
          b2["n_pairs"] == 2 and b2["paired_seeds"] == [0, 1])


def check_controls_are_separate(tmp: Path):
    print("\n-- the parameter-matched and two-hot controls --")
    rows = ([_row("transformer", s, False, 0.3, 5000 + 10 * s) for s in range(5)]
            + [_row("mlp", s, False, 0.3, 9000 + 10 * s, d_mlp=572) for s in range(5)]
            + [_row("mlp", s, False, 0.3, 9500 + 10 * s) for s in range(5)]
            + [_row("mlp_twohot", s, False, 0.3, 12000 + 10 * s) for s in range(3)])
    rep = CR.compute(_agg(tmp, rows, "sep"))
    pm = rep["parameter_matched"]["generalization_step"]
    check("the parameter-matched control pairs the transformer with the d_mlp=572 MLP",
          pm["n_pairs"] == 5 and abs(pm["median_diff"] - (-4000)) < 1e-9)
    check("...and it does NOT pick up the default-width MLP",
          all(v >= 9000 for v in pm["values_b"]) and all(v < 9500 for v in pm["values_b"]))
    th = rep["two_hot"]["generalization_step"]
    check("the two-hot control pairs against the shared-embedding MLP, on its 3 seeds",
          th["n_pairs"] == 3 and abs(th["median_diff"] - 2500) < 1e-9)
    check("the two-hot block states that a null there is weak evidence",
          "WEAK EVIDENCE" in th["what_it_can_support"])
    check("the parameter-matched block states what it cannot equalize",
          "shape of the budget" in pm["what_it_can_support"])
    check("the separation rule is written into the artifact",
          "unless the 2x2 decomposition separates them" in rep["separation_rule"])
    check("...as is the rule that controls are never merged with the primary comparison",
          "never merged" in rep["controls_are_separate"])


def check_metrics(tmp: Path):
    print("\n-- the metric extractors --")
    r = _row("mlp", 0, False, 0.3, 9000)
    check("the grokking gap is generalization minus memorization",
          CR.METRICS["grokking_gap"](r) == 9000 - 100)
    check("the square-or-harmonic fraction sums the two shares",
          abs(CR.METRICS["square_or_harmonic_fraction"](r) - 0.5) < 1e-12)
    check("a missing input gives None, not 0",
          CR.METRICS["grokking_gap"]({"generalization_step": None}) is None)
    check("a boolean is not treated as a number", CR._num(True) is None)


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_separation(tmp)
        check_incomplete_cells(tmp)
        check_controls_are_separate(tmp)
        check_metrics(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
