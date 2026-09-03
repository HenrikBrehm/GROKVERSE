"""Checks for analysis/figures_study.py (INTERFACES §13, STATISTICAL_ANALYSIS_PLAN §3).

Run: ``python tests/test_figures_study.py`` from ``training/``.

Synthetic aggregate tables only — no checkpoint and no driver output is read.

THE TWO PROPERTIES WORTH PINNING
---------------------------------
1. **The figure count is fixed in advance.** `docs/STATISTICAL_ANALYSIS_PLAN.md` §3 fixes exactly
   eight paired comparisons "so the count is fixed in advance and cannot grow after the fact". A test
   that asserts the number is the cheapest guard against a ninth headline figure quietly widening the
   study's claims.
2. **A figure that cannot be drawn says why.** Silence would look like an absent effect; the module
   must record which seeds it had for each architecture and why no pair was formed.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import figures_study as FS  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _row(arch, seed, **over):
    row = {"run_id": f"{arch}_s{seed}_arch25k", "arch": arch, "seed": seed, "p": 113,
           "tag": "step025000", "step": 25000, "steps_completed": 25000,
           "generalization_step": 9000 + 100 * seed, "memorization_step": 150,
           "structured_fraction_of_live": 0.3 + 0.01 * seed, "phase_R": 0.9,
           "sparse_sinusoid__r2_test": 0.8, "odd_harmonics__r2_test": 0.7,
           "u_a__fraction_best_aic_square": 0.4, "u_a__fraction_best_aic_odd_harmonics": 0.2,
           "h3b__u_a__family_median": 0.6, "h3b__u_a__top1_median": 0.4,
           "remove_structured__drop": 0.9, "remove_structured__control_mean_drop": 0.1}
    row.update(over)
    return row


def _agg(tmp: Path, rows_by_module: dict) -> Path:
    d = tmp / "aggregate"
    d.mkdir(parents=True, exist_ok=True)
    for module, rows in rows_by_module.items():
        (d / f"{module}.json").write_text(json.dumps({"module": module, "rows": rows}),
                                          encoding="utf-8")
    return d


def check_specification():
    print("\n-- the pre-specified comparison set --")
    check("exactly eight comparisons are declared (the count is fixed in advance)",
          len(FS.COMPARISONS) == 8)
    ids = [c["id"] for c in FS.COMPARISONS]
    check("they are numbered p1..p8 in the plan's order",
          ids == sorted(ids, key=lambda s: int(s[1])) and ids[0].startswith("p1")
          and ids[-1].startswith("p8"))
    check("every comparison declares its modules, points and an extractor",
          all({"id", "label", "modules", "points", "get"} <= set(c) for c in FS.COMPARISONS))
    check("the two structure comparisons are drawn at BOTH measurement points (plan §3)",
          set(FS.COMPARISONS[2]["points"]) == {"crossing", "final"}
          and set(FS.COMPARISONS[3]["points"]) == {"crossing", "final"})


def check_extractors():
    print("\n-- the per-comparison extractors --")
    r = _row("mlp", 0)
    by_id = {c["id"]: c for c in FS.COMPARISONS}
    check("p2 is the gap between the two logged transitions",
          by_id["p2_grokking_gap"]["get"](r) == 9000 - 150)
    check("p5 sums the square and odd-harmonic shares",
          abs(by_id["p5_square_or_harmonic_fraction"]["get"](r) - 0.6) < 1e-12)
    check("p6 is family minus top-1",
          abs(by_id["p6_family_minus_top1"]["get"](r) - 0.2) < 1e-12)
    check("p7 is damage minus its size-matched control",
          abs(by_id["p7_ablation_damage_minus_control"]["get"](r) - 0.8) < 1e-12)
    check("p8 takes the better of the two Fourier formulas",
          abs(by_id["p8_best_fourier_r2"]["get"](r) - 0.8) < 1e-12)
    check("a missing input yields None, never 0",
          by_id["p6_family_minus_top1"]["get"](_row("mlp", 0, h3b__u_a__family_median=None)) is None)
    check("a boolean is not mistaken for a number",
          by_id["p8_best_fourier_r2"]["get"](
              _row("mlp", 0, sparse_sinusoid__r2_test=True, odd_harmonics__r2_test=None)) is None)


def check_point_filtering(tmp: Path):
    print("\n-- values are taken at the requested measurement point only --")
    rows = [_row("mlp", 0), _row("mlp", 0, tag="step009000", step=9000,
                                 structured_fraction_of_live=0.11)]
    got_final = FS.values_by_arch(rows, "final", lambda r: r.get("structured_fraction_of_live"))
    got_cross = FS.values_by_arch(rows, "crossing", lambda r: r.get("structured_fraction_of_live"))
    check("the final point picks the last completed step", got_final["mlp"][0] == 0.3)
    check("the crossing point picks the logged generalization step", got_cross["mlp"][0] == 0.11)
    check("one seed is never counted twice at one point", len(got_final["mlp"]) == 1)


def check_figures(tmp: Path):
    print("\n-- figures(): drawing, captions and the reasons for not drawing --")
    rows = [_row(a, s) for a in ("mlp", "transformer") for s in range(5)]
    agg = _agg(tmp, {"mlp_mechanism": [r for r in rows if r["arch"] == "mlp"],
                     "transformer_mechanism": [r for r in rows if r["arch"] == "transformer"],
                     "wave_fitting": rows, "h3_validity": rows,
                     "causal_ablation": rows, "logit_formula_fit": rows})
    out = tmp / "figs"
    index = FS.figures(agg, out, tmp / "no_decision.json")

    drawn = [f for f in index["figures"] if f.get("drawn")]
    check("the eight comparisons produce at least eight drawn figures", len(drawn) >= 8)
    check("each drawn figure exists on disk",
          all((out / f["path"]).exists() for f in drawn if "path" in f))
    check("every figure names the aggregate files it came from",
          all(f.get("source_files") for f in index["figures"] if f["id"] != "gate_summary"))
    one = next(f for f in drawn if f["id"] == "p3_structured_fraction")
    check("the caption names the sources and the pair count",
          "aggregate/" in one["caption"] and "5 paired seeds" in one["caption"])
    check("the paired statistics travel with the figure",
          {"n_pairs", "median_diff", "bootstrap_ci95"} <= set(one["statistics"]))
    check("every seed is in the figure's record, not just a count",
          one["seeds"] == [0, 1, 2, 3, 4])
    check("the index states that nothing is recomputed",
          "no checkpoint is read" in index["source_rule"])
    check("the index states that the count is fixed in advance",
          "does not grow" in index["rule"])
    check("a missing decision_tree.json makes the gate figure NOT drawn, with its reason",
          any(f["id"] == "gate_summary" and not f["drawn"] and "no decision_tree" in f["reason"]
              for f in index["figures"]))

    # a comparison with no overlapping seeds must say so rather than fall silent
    lop = _agg(tmp / "lop", {"mlp_mechanism": [_row("mlp", 0)],
                             "transformer_mechanism": [_row("transformer", 7)]})
    idx2 = FS.figures(lop, tmp / "figs2", tmp / "none.json")
    p3 = next(f for f in idx2["figures"] if f["id"] == "p3_structured_fraction"
              and f["point"] == "final")
    check("with no shared seed the figure is not drawn and the reason lists what it had",
          not p3["drawn"] and "[0]" in p3["reason"] and "[7]" in p3["reason"])
    check("...and the unpaired seeds are recorded per architecture",
          p3["n_only_transformer"] == [7] and p3["n_only_mlp"] == [0])


def check_gate_figure(tmp: Path):
    print("\n-- the gate summary figure --")
    dt = tmp / "dt.json"
    dt.write_text(json.dumps({
        "measurement_point": "final",
        "thresholds": {"seeds_required": 8, "seeds_total": 10},
        "branch": {"branch": "one_passes"},
        "per_architecture": {
            "mlp": {"per_criterion_passed": {"G1": 9, "G2": 10, "G3": 4, "G4": 2},
                    "per_criterion_not_evaluable": {"G1": 0, "G2": 0, "G3": 6, "G4": 0},
                    "passes_gate": False},
            "transformer": {"per_criterion_passed": {"G1": 10, "G2": 10, "G3": 9, "G4": 8},
                            "per_criterion_not_evaluable": {"G1": 0, "G2": 0, "G3": 1, "G4": 0},
                            "passes_gate": True}}}), encoding="utf-8")
    out = tmp / "figs3"
    out.mkdir(parents=True, exist_ok=True)
    entry = FS.gate_figure(dt, out)
    check("the gate figure is drawn", entry["drawn"] and (out / entry["path"]).exists())
    check("it records the branch and each architecture's verdict",
          entry["branch"] == "one_passes"
          and entry["passes_gate"] == {"mlp": False, "transformer": True})
    check("its caption names the source and the shading convention",
          "dt.json" in entry["caption"] and "not evaluable" in entry["caption"])


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_specification()
        check_extractors()
        check_point_filtering(tmp)
        check_figures(tmp)
        check_gate_figure(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
