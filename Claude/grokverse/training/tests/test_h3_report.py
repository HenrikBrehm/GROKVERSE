"""Checks for analysis/h3_report.py (PREREGISTRATION §3 H3, master prompt §12).

Run: ``python tests/test_h3_report.py`` from ``training/``.

Synthetic aggregate tables only.

THE MISTAKE THIS FILE EXISTS TO PREVENT
----------------------------------------
The pre-registration lists conditions under which **H3** is refuted. H3₀ is the *null* — "the gap is
the same under both definitions". A first version of the module named its verdict block after H3₀ and
attached a note claiming that refuting it is what H3 needs, which inverts the study's headline. The
checks below pin the direction of every criterion against inputs whose answer is known by
construction, in both directions, so the labelling cannot silently flip again.

The other thing pinned here is the gate precondition: master prompt §12 permits a harmonic-family
mechanism reading **only** for an architecture that passes the evidence gate, so the report must state
which architectures passed and must say plainly when none did.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import h3_report as H3R  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _row(arch, seed, family, top1, **over):
    row = {"run_id": f"{arch}_s{seed}", "arch": arch, "seed": seed, "p": 113,
           "tag": "step025000", "step": 25000, "steps_completed": 25000,
           "generalization_step": 9000,
           H3R.FAMILY_KEY: family, H3R.TOP1_KEY: top1,
           "h3c__object": "W_E", "h3c__top8_concentration": 0.4,
           "h3c__nf2__family_fraction": 0.5, "h3c__nf2__null_mean": 0.07}
    for curve in H3R.CURVES:
        for k in H3R.TOPK:
            row[f"h3a__{curve}__waveform_sensitivity_top{k}"] = 0.19
        row[f"h3a__{curve}__shape_separation"] = 0.14
        row[f"h3b__{curve}__family_minus_matched_max"] = 0.0
    row.update(over)
    return row


def _agg(tmp: Path, rows, name="a") -> Path:
    d = tmp / name / "aggregate"
    d.mkdir(parents=True, exist_ok=True)
    (d / "h3_validity.json").write_text(json.dumps({"module": "h3_validity", "rows": rows}),
                                        encoding="utf-8")
    return d


def _gate(tmp: Path, passing, name="a") -> Path:
    p = tmp / name / "decision_tree_final.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "measurement_point": "final",
        "branch": {"branch": "both_pass" if len(passing) == 2 else
                   ("one_passes" if passing else "neither_passes")},
        "per_architecture": {a: {"passes_gate": a in passing} for a in ("mlp", "transformer")}}),
        encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
def check_gate_precondition(tmp: Path):
    print("\n-- the gate precondition (master prompt 12) --")
    rows = [_row(a, s, 0.40 if a == "transformer" else 0.30, 0.42 if a == "transformer" else 0.30)
            for a in ("transformer", "mlp") for s in range(6)]
    none = H3R.compute(_agg(tmp, rows, "g0"), _gate(tmp, [], "g0"))
    check("with no architecture passing, the report says NONE is cleared",
          none["gate"]["architectures_passing"] == []
          and "NONE" in none["gate"]["consequence"])
    check("...and states the numbers are about the metric, not a mechanism",
          "about the METRIC" in none["gate"]["consequence"])
    both = H3R.compute(_agg(tmp, rows, "g2"), _gate(tmp, ["mlp", "transformer"], "g2"))
    check("with both passing, both are listed as cleared",
          sorted(both["gate"]["architectures_passing"]) == ["mlp", "transformer"])
    check("the §12 rule is quoted in the output",
          "only for an architecture that passes" in none["gate"]["interpretation_rule"].lower())
    missing = H3R.compute(_agg(tmp, rows, "g3"), tmp / "g3" / "nope.json")
    check("a missing gate file is reported, not assumed to be a pass",
          missing["gate"]["available"] is False
          and "may not be interpreted" in missing["gate"]["note"])


def check_criteria_directions(tmp: Path):
    print("\n-- both refutation criteria, in both directions --")
    # (a) H3's own prediction: the family definition CLOSES the gap (family gap ~ 0)
    supportive = ([_row("transformer", s, 0.30, 0.42) for s in range(8)]
                  + [_row("mlp", s, 0.30, 0.30) for s in range(8)])
    rep = H3R.compute(_agg(tmp, supportive, "c1"), _gate(tmp, [], "c1"))
    c1, c2 = rep["h3b_criterion_1_gap_difference"], rep["h3b_criterion_2_family_gap"]
    check("criterion 1: a real difference between the two gaps does NOT refute H3",
          c1["ci95_contains_zero"] is False and c1["refutes_H3"] is False)
    check("criterion 2: a family gap of zero does NOT refute H3",
          c2["refutes_H3"] is False)
    check("with neither criterion met, H3 is not refuted", rep["h3_verdict"]["h3_refuted"] is False)

    # (b) the family definition does NOT close the gap -> criterion 2 refutes
    deficit = ([_row("transformer", s, 0.40, 0.42) for s in range(8)]
               + [_row("mlp", s, 0.30, 0.30) for s in range(8)])
    rep2 = H3R.compute(_agg(tmp, deficit, "c2"), _gate(tmp, [], "c2"))
    check("criterion 2 refutes H3 when the MLP is still lower under the family definition",
          rep2["h3b_criterion_2_family_gap"]["refutes_H3"] is True)
    check("...and the verdict names that criterion",
          rep2["h3_verdict"]["h3_refuted"] is True
          and rep2["h3_verdict"]["refuting_criteria"] == ["criterion_2_family_gap"])
    check("...with the reading spelling out 'a real loss of structure'",
          "real loss of structure" in rep2["h3b_criterion_2_family_gap"]["reading"])

    # (c) the two gaps are indistinguishable -> criterion 1 refutes
    same = ([_row("transformer", s, 0.40, 0.40) for s in range(8)]
            + [_row("mlp", s, 0.30, 0.30) for s in range(8)])
    rep3 = H3R.compute(_agg(tmp, same, "c3"), _gate(tmp, [], "c3"))
    check("criterion 1 refutes H3 when the CI of the gap difference contains zero",
          rep3["h3b_criterion_1_gap_difference"]["ci95_contains_zero"] is True
          and rep3["h3b_criterion_1_gap_difference"]["refutes_H3"] is True)
    check("the verdict block is named for H3, not H3_0",
          "h3_verdict" in rep3 and "h3_0_verdict" not in rep3)
    check("...and states H3's prediction and what H3_0 is, so the direction is unambiguous",
          "closes under the family" in rep3["h3_verdict"]["h3_prediction"]
          and "null" in rep3["h3_verdict"]["h3_0_is"])


def check_gap_accounting(tmp: Path):
    print("\n-- how much of the gap the family definition closes --")
    rows = ([_row("transformer", s, 0.40, 0.45) for s in range(8)]
            + [_row("mlp", s, 0.30, 0.30) for s in range(8)])
    rep = H3R.compute(_agg(tmp, rows, "acc"), _gate(tmp, [], "acc"))
    closed = rep["h3_verdict"]["how_much_of_the_gap_the_family_closes"]
    check("the top-1 gap is reported (0.45 - 0.30)",
          abs(closed["gap_under_top1_median"] - 0.15) < 1e-9)
    check("the family gap is reported (0.40 - 0.30)",
          abs(closed["gap_under_family_median"] - 0.10) < 1e-9)
    check("the closed amount is their difference",
          abs(closed["closed_median"] - 0.05) < 1e-9)
    check("...expressed as a fraction of the top-1 gap",
          abs(closed["closed_fraction_of_the_top1_gap"] - (0.05 / 0.15)) < 1e-9)


def check_audit_and_h3a(tmp: Path):
    print("\n-- the forbidden criterion and H3a --")
    rows = ([_row("transformer", s, 0.4, 0.42) for s in range(5)]
            + [_row("mlp", s, 0.3, 0.3) for s in range(5)])
    rep = H3R.compute(_agg(tmp, rows, "au"), _gate(tmp, [], "au"))
    a = rep["forbidden_criterion_audit"]
    check("the audit holds when family never beats matched top-m", a["holds"] is True)
    check("...and says it is an audit, never support for H3",
          "never support" in a["statement"])
    bad = [dict(r, **{f"h3b__u_a__family_minus_matched_max": 0.05}) for r in rows]
    check("CONTROL: a positive value fails the audit",
          H3R.compute(_agg(tmp, bad, "au2"),
                      _gate(tmp, [], "au2"))["forbidden_criterion_audit"]["holds"] is False)
    h3a = rep["h3a_metric_waveform_sensitivity"]
    check("H3a is reported per architecture for every curve and cardinality",
          all(f"{c}__top{k}" in h3a for c in H3R.CURVES for k in H3R.TOPK))
    check("...and is labelled a property of the metric",
          "of the METRIC" in h3a["definition"])
    check("H3c is reported with its W_E caveat",
          "decides nothing for the MLP" in rep["h3c_legacy_embedding"]["caveat"])


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_gate_precondition(tmp)
        check_criteria_directions(tmp)
        check_gap_accounting(tmp)
        check_audit_and_h3a(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
