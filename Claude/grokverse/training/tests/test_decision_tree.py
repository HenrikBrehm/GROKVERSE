"""Checks for analysis/decision_tree.py (INTERFACES §13, PREREGISTRATION §5–6).

Run: ``python tests/test_decision_tree.py`` from ``training/``.

Every input is a synthetic aggregate row, so the gate arithmetic is tested against answers known by
construction rather than against whatever the study happens to have measured.

THE THREE THINGS THIS FILE EXISTS TO PREVENT
---------------------------------------------
1. **"Not evaluable" silently becoming "failed".** A criterion whose input is missing has not been
   tested. Counting it as a failure would manufacture evidence against the hypothesis; counting it as
   a pass would manufacture evidence for it. Both directions are asserted here.
2. **An unfinished analysis reading as a verdict.** With fewer than the design's ten seeds analysed,
   the gate must be `None` (undetermined) and the branch must be `undetermined` — never
   `neither_passes`, which is a claim about the architectures.
3. **A threshold drifting.** The constants are compared against `docs/PREREGISTRATION.md` §5 values
   literally, so an accidental edit fails a test instead of quietly changing a verdict.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import decision_tree as DT  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _row(**over) -> dict:
    """An aggregate row on which all four criteria pass, before ``over`` is applied."""
    row = {
        "run_id": "mlp_s0_arch25k", "arch": "mlp", "seed": 0, "p": 113,
        "tag": "step025000", "step": 25000, "steps_completed": 25000,
        "generalization_step": 9000,
        # G1
        "structured_fraction_of_live": 0.40, "family_fraction_median_u_a": 0.60,
        "family_fraction_median_u_b": 0.58,
        # G2
        "phase_R": 0.95, "phase_null_q95": 0.14, "phase_insufficient_neurons": False, "phase_n": 200,
        # G3
        "sparse_sinusoid__argmax_acc_test": 0.97, "sparse_sinusoid__r2_test": 0.80,
        "odd_harmonics__argmax_acc_test": 0.96, "odd_harmonics__r2_test": 0.82,
        "control_difference__r2_test": 0.10,
        "control_random_m__r2_test_q95": 0.30, "control_random_m__r2_test_std": 0.05,
        "control_set_size": 8,
        # G4
        "g4_passed": True, "g4_remove_structured_necessary": True,
        "g4_remove_key_freqs_necessary": True, "g4_keep_structured_sufficient": True,
    }
    row.update(over)
    return row


# --------------------------------------------------------------------------- #
# 1. thresholds are the pre-registered ones                                    #
# --------------------------------------------------------------------------- #
def check_thresholds():
    print("\n-- the constants are PREREGISTRATION 5's, unchanged --")
    check("G1 thresholds", DT.G1_MIN_STRUCTURED_FRACTION == 0.25
          and DT.G1_MIN_MEDIAN_FAMILY_FRACTION == 0.50)
    check("G2 threshold", DT.G2_MIN_R == 0.50)
    check("G3 thresholds", DT.G3_MIN_ARGMAX_ACC_TEST == 0.90
          and DT.G3_DIFFERENCE_CONTROL_RATIO == 0.5)
    check("the seed rule is 8 of 10", (DT.SEEDS_REQUIRED, DT.SEEDS_TOTAL) == (8, 10))
    check("both sensitivity definitions are declared",
          set(DT.SENSITIVITY_DEFINITIONS) == {"sensitivity_family_0.30", "sensitivity_family_0.70"})


# --------------------------------------------------------------------------- #
# 2. the four criteria                                                         #
# --------------------------------------------------------------------------- #
def check_criteria():
    print("\n-- G1..G4 truth tables, including 'not evaluable' --")
    check("G1 passes when both conditions hold", DT.g1(_row())[0] is True)
    check("G1 fails on too few structured neurons",
          DT.g1(_row(structured_fraction_of_live=0.24))[0] is False)
    check("G1 fails when only ONE curve reaches the family threshold",
          DT.g1(_row(family_fraction_median_u_b=0.49))[0] is False)
    v, d = DT.g1(_row(family_fraction_median_u_a=None))
    check("G1 with a missing input is NOT evaluable, not failed",
          v is None and "missing" in d["not_evaluable"])
    check("G1 reads the requested sensitivity definition",
          DT.g1(_row(**{"structured_fraction__sensitivity_family_0.70": 0.10}),
                "sensitivity_family_0.70")[0] is False)

    check("G2 passes above the null and above 0.5", DT.g2(_row())[0] is True)
    check("G2 fails when R is below the null", DT.g2(_row(phase_R=0.10))[0] is False)
    check("G2 fails when R clears the null but is below 0.5",
          DT.g2(_row(phase_R=0.30, phase_null_q95=0.05))[0] is False)
    v, d = DT.g2(_row(phase_R=None, phase_insufficient_neurons=True))
    check("G2 on an empty structured set is NOT evaluable, and says so",
          v is None and "structured set was empty" in d["not_evaluable"])

    check("G3 passes on a good fit with a working control", DT.g3(_row(), 113)[0] is True)
    check("G3 fails when argmax accuracy is below 0.90",
          DT.g3(_row(sparse_sinusoid__argmax_acc_test=0.5,
                     odd_harmonics__argmax_acc_test=0.5), 113)[0] is False)
    check("G3 fails when the (a-b) control fits nearly as well",
          DT.g3(_row(control_difference__r2_test=0.79), 113)[0] is False)
    v, d = DT.g3(_row(control_set_size=56), 113)
    check("G3 with a control drawing EVERY frequency is not evaluable, not passed",
          v is None and d["control_degenerate"] and "same set every time" in d["not_evaluable"])
    v, d = DT.g3(_row(sparse_sinusoid__argmax_acc_test=None,
                      odd_harmonics__argmax_acc_test=None), 113)
    check("G3 with no fit numbers at all is not evaluable", v is None)

    check("G4 is read from causal_ablation's own verdict", DT.g4(_row())[0] is True)
    check("G4 fails when the module says so", DT.g4(_row(g4_passed=False))[0] is False)
    v, d = DT.g4(_row(g4_passed=None))
    check("G4 with no verdict is not evaluable, not failed",
          v is None and "did not report" in d["not_evaluable"])


# --------------------------------------------------------------------------- #
# 3. the measurement point must not double-count a seed                        #
# --------------------------------------------------------------------------- #
def check_point_resolution():
    print("\n-- _is_point: final vs crossing --")
    final = _row(step=25000)
    crossing = _row(step=9000, tag="step009000")
    check("the final checkpoint is the last completed step", DT._is_point(final, "final") is True)
    check("...and is not the crossing", DT._is_point(final, "crossing") is False)
    check("the crossing checkpoint matches the logged generalization step",
          DT._is_point(crossing, "crossing") is True)
    check("...and is not the final point", DT._is_point(crossing, "final") is False)
    check("a legacy run tagged 'final' with no step counts as the final point",
          DT._is_point({"tag": "final", "step": None}, "final") is True)
    raised = False
    try:
        DT._is_point(final, "halfway")
    except ValueError:
        raised = True
    check("an unknown measurement point raises", raised)


# --------------------------------------------------------------------------- #
# 4. the gate and the branch                                                   #
# --------------------------------------------------------------------------- #
def _write_aggregate(tmp: Path, rows_by_module: dict) -> Path:
    d = tmp / "aggregate"
    d.mkdir(parents=True, exist_ok=True)
    for module, rows in rows_by_module.items():
        (d / f"{module}.json").write_text(json.dumps(
            {"module": module, "rows": rows, "missing": [], "n_rows": len(rows),
             "n_missing": 0, "n_not_applicable": 0}), encoding="utf-8")
    return d


def _seed_rows(arch: str, n_pass: int, n_total: int = 10, **bad) -> list[dict]:
    rows = []
    for seed in range(n_total):
        over = {"arch": arch, "seed": seed, "run_id": f"{arch}_s{seed}_arch25k"}
        if seed >= n_pass:
            over.update(bad or {"g4_passed": False})
        rows.append(_row(**over))
    return rows


def check_gate(tmp: Path):
    print("\n-- evaluate(): the 8-of-10 rule and the branch --")
    agg = _write_aggregate(tmp, {"mlp_mechanism": _seed_rows("mlp", 10),
                                 "transformer_mechanism": _seed_rows("transformer", 10),
                                 "logit_formula_fit": _seed_rows("mlp", 10)
                                 + _seed_rows("transformer", 10),
                                 "causal_ablation": _seed_rows("mlp", 10)
                                 + _seed_rows("transformer", 10)})
    rep = DT.evaluate(agg)
    check("both architectures pass when every seed passes",
          all(b["passes_gate"] is True for b in rep["per_architecture"].values()))
    check("the branch is both_pass", rep["branch"]["branch"] == "both_pass")
    check("the permitted conclusion is quoted, and the banned one named",
          "different internal representations" in rep["branch"]["permitted_conclusion"]
          and "same circuit" in rep["branch"]["banned"])
    check("the thresholds that produced the verdict are echoed with their provenance",
          rep["thresholds"]["seeds_required"] == 8
          and "PREREGISTRATION" in rep["thresholds"]["provenance"])
    check("every seed is listed, not just a count",
          len(rep["per_architecture"]["mlp"]["seeds"]) == 10
          and rep["per_architecture"]["mlp"]["seeds_passed"] == list(range(10)))

    agg2 = _write_aggregate(tmp / "b", {"mlp_mechanism": _seed_rows("mlp", 7),
                                        "transformer_mechanism": _seed_rows("transformer", 10),
                                        "logit_formula_fit": _seed_rows("mlp", 7)
                                        + _seed_rows("transformer", 10),
                                        "causal_ablation": _seed_rows("mlp", 7)
                                        + _seed_rows("transformer", 10)})
    rep2 = DT.evaluate(agg2)
    check("7 of 10 does NOT pass the gate", rep2["per_architecture"]["mlp"]["passes_gate"] is False)
    check("the transformer still passes", rep2["per_architecture"]["transformer"]["passes_gate"] is True)
    check("the branch is one_passes and names which architecture gets the bounded analysis",
          rep2["branch"]["branch"] == "one_passes"
          and rep2["branch"]["architectures_failing"] == ["mlp"]
          and "bounded_alternative" in rep2["branch"]["action"])

    agg3 = _write_aggregate(tmp / "c", {"mlp_mechanism": _seed_rows("mlp", 0),
                                        "transformer_mechanism": _seed_rows("transformer", 0),
                                        "logit_formula_fit": _seed_rows("mlp", 0)
                                        + _seed_rows("transformer", 0),
                                        "causal_ablation": _seed_rows("mlp", 0)
                                        + _seed_rows("transformer", 0)})
    rep3 = DT.evaluate(agg3)
    check("when neither passes the branch says so, and prescribes the bounded analysis for both",
          rep3["branch"]["branch"] == "neither_passes"
          and "bounded alternative" in rep3["branch"]["action"])


def check_incomplete(tmp: Path):
    print("\n-- an unfinished analysis is UNDETERMINED, never a verdict --")
    agg = _write_aggregate(tmp / "d", {"mlp_mechanism": _seed_rows("mlp", 3, n_total=3),
                                       "logit_formula_fit": _seed_rows("mlp", 3, n_total=3),
                                       "causal_ablation": _seed_rows("mlp", 3, n_total=3)})
    rep = DT.evaluate(agg)
    block = rep["per_architecture"]["mlp"]
    check("with 3 of 10 seeds analysed the gate is None, not False",
          block["passes_gate"] is None)
    check("...and the reason names the incompleteness",
          "only 3 of 10 seeds" in block["verdict_reason"])
    check("the branch is undetermined, not neither_passes",
          rep["branch"]["branch"] == "undetermined")
    check("...and it forbids taking a branch meanwhile",
          "no branch may be taken" in rep["branch"]["action"])

    # all ten seeds present, but one criterion untestable on four of them
    rows = _seed_rows("mlp", 6, n_total=10, g4_passed=None)
    agg2 = _write_aggregate(tmp / "e", {"mlp_mechanism": rows, "logit_formula_fit": rows,
                                        "causal_ablation": rows})
    b2 = DT.evaluate(agg2)["per_architecture"]["mlp"]
    check("6 passed + 4 not evaluable leaves the gate open (they could still reach 8)",
          b2["passes_gate"] is None and b2["n_not_evaluable"] == 4)
    check("...and the reason says exactly that",
          "could still reach 8" in b2["verdict_reason"])
    rows2 = _seed_rows("mlp", 1, n_total=10, g4_passed=None)
    b3 = DT.evaluate(_write_aggregate(tmp / "f", {"mlp_mechanism": rows2,
                                                  "logit_formula_fit": rows2,
                                                  "causal_ablation": rows2}))["per_architecture"]["mlp"]
    check("1 passed + 9 not evaluable CAN still reach 8, so it stays open",
          b3["passes_gate"] is None)
    check("the per-criterion counts separate passed from not-evaluable",
          b3["per_criterion_not_evaluable"]["G4"] == 9
          and b3["per_criterion_passed"]["G1"] == 10)


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_thresholds()
        check_criteria()
        check_point_resolution()
        check_gate(tmp)
        check_incomplete(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
