"""Checks for analysis/aggregate.py (INTERFACES §13).

Run: ``python tests/test_aggregate.py`` from ``training/``.

A synthetic runs tree is built here, so this file needs no trained run and no driver output.

WHAT THIS FILE GUARDS
----------------------
Aggregation is where a study quietly loses seeds. The checks below pin the three ways that happens:

1. a **missing** analysis must appear as a gap with its reason, never as a smaller denominator;
2. a module that is **not defined** for an architecture must not be counted as missing — otherwise the
   missing count is inflated and completeness looks worse than it is;
3. the markdown table must print **every** row, because master prompt §16 requires all individual
   seeds to be shown and a mean alone is not evidence.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import aggregate as AG  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _make_run(base: Path, run_id: str, arch: str, seed: int, modules: dict) -> Path:
    d = base / run_id
    (d / "analysis").mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({
        "config": {"arch": arch, "seed": seed, "p": 113, "train_frac": 0.3, "weight_decay": 1.0,
                   "grokfast": False, "d_mlp": 512, "steps": 25000},
        "transitions": {"primary": {"memorization": {"first_crossing_step": 150},
                                    "generalization": {"first_crossing_step": 9000}}}}))
    (d / "manifest.json").write_text(json.dumps({
        "status": "completed", "split_hash": f"hash{seed}", "steps_completed": 25000,
        "git_commit": "deadbeef"}))
    for module, payload in modules.items():
        md = d / "analysis" / module
        md.mkdir(parents=True, exist_ok=True)
        (md / "step025000.json").write_text(json.dumps(payload))
    return d


def _mech_payload(frac: float, R: float, fam: float) -> dict:
    return {"module": "mlp_mechanism", "module_version": "2.0", "step": 25000,
            "analysis_git_commit": "cafe", "checkpoint_sha256": "abc",
            "params": {"key_frequencies": [3, 5]},
            "results": {
                "n_neurons": 512,
                "structured_neurons": {
                    "primary": "primary_family_0.50", "n_live": 500,
                    "definitions": {
                        "primary_family_0.50": {
                            "n": 150, "fraction_of_live_neurons": frac,
                            "phase_relation": {"resultant_length": R, "null_resultant_q95": 0.14,
                                               "exceeds_null_q95": R > 0.14, "n": 150,
                                               "insufficient_neurons": False}},
                        "sensitivity_family_0.30": {"fraction_of_live_neurons": frac + 0.1,
                                                    "phase_relation": {}}}},
                "neuron_tables": {"per_curve": {
                    "u_a": {"family_fraction": {"median": fam},
                            "dominant_fraction": {"median": 0.4},
                            "topk_concentration": {"8": {"median": 0.7}}},
                    "u_b": {"family_fraction": {"median": fam},
                            "dominant_fraction": {"median": 0.4},
                            "topk_concentration": {"8": {"median": 0.7}}},
                    "out": {"family_fraction": {"median": 0.3},
                            "dominant_fraction": {"median": 0.2},
                            "topk_concentration": {"8": {"median": 0.5}}}}},
                "sum_dependence": {"median_sum": 0.9, "median_diff": 0.2}}}


def check_collect(tmp: Path):
    print("\n-- collect_module: rows, gaps, and not-applicable --")
    base = tmp / "runs"
    _make_run(base, "mlp_a_arch25k", "mlp", 0, {"mlp_mechanism": _mech_payload(0.3, 0.99, 0.6)})
    _make_run(base, "mlp_b_arch25k", "mlp", 1, {})                       # analysis missing
    _make_run(base, "txf_a_arch25k", "transformer", 0,
              {"transformer_mechanism": _mech_payload(0.4, 0.95, 0.55)})
    runs = sorted(base.iterdir())

    block = AG.collect_module("mlp_mechanism", runs)
    check("one row per run that has the analysis", block["n_rows"] == 1)
    check("a run whose analysis is absent becomes a MISSING row, with its reason",
          block["n_missing"] == 1 and "no analysis/" in block["missing"][0]["reason"])
    check("a transformer run is NOT counted as missing for an MLP-only module",
          block["n_not_applicable"] == 1
          and "not defined for arch" in block["not_applicable"][0]["reason"])
    check("the applicable denominator excludes the not-applicable run",
          block["n_runs_applicable"] == 2)

    row = block["rows"][0]
    check("run metadata travels with every row",
          row["arch"] == "mlp" and row["seed"] == 0 and row["split_hash"] == "hash0")
    check("the transition steps are carried through",
          row["memorization_step"] == 150 and row["generalization_step"] == 9000)
    check("the source file is named so any number can be traced back",
          "mlp_mechanism" in row["source_file"] and row["source_file"].endswith(".json"))
    check("provenance from the envelope is kept",
          row["analysis_git_commit"] == "cafe" and row["checkpoint_sha256"] == "abc")
    check("the G1 inputs are extracted",
          row["structured_fraction_of_live"] == 0.3 and row["family_fraction_median_u_a"] == 0.6)
    check("the G2 inputs are extracted",
          row["phase_R"] == 0.99 and row["phase_null_q95"] == 0.14)
    check("every structured definition present is extracted, not only the primary",
          "structured_fraction__sensitivity_family_0.30" in row)

    tblock = AG.collect_module("transformer_mechanism", runs)
    check("the transformer module sees its own run and calls the MLPs not-applicable",
          tblock["n_rows"] == 1 and tblock["n_not_applicable"] == 2)


def check_bad_file(tmp: Path):
    print("\n-- an unreadable analysis file --")
    base = tmp / "runs2"
    d = _make_run(base, "mlp_c_arch25k", "mlp", 2, {"mlp_mechanism": _mech_payload(0.3, 0.9, 0.6)})
    (d / "analysis" / "mlp_mechanism" / "step025000.json").write_text("{ this is not json")
    block = AG.collect_module("mlp_mechanism", [d])
    check("a corrupt file is recorded as missing with the parse error, not crashed on",
          block["n_rows"] == 0 and block["n_missing"] == 1
          and "unreadable" in block["missing"][0]["reason"])


def check_extractor_error(tmp: Path):
    print("\n-- an extractor that cannot read a payload --")
    base = tmp / "runs3"
    d = _make_run(base, "mlp_d_arch25k", "mlp", 3, {"mlp_mechanism": {
        "step": 25000, "results": {"structured_neurons": "not a dict"}}})
    block = AG.collect_module("mlp_mechanism", [d])
    check("a payload the extractor cannot read still produces a row, flagged with the error",
          block["n_rows"] == 1
          and ("extractor_error" in block["rows"][0]
               or block["rows"][0].get("structured_fraction_of_live") is None))


def check_markdown(tmp: Path):
    print("\n-- markdown_table: every seed shown --")
    base = tmp / "runs4"
    for seed in range(4):
        _make_run(base, f"mlp_s{seed}_arch25k", "mlp", seed,
                  {"mlp_mechanism": _mech_payload(0.2 + 0.05 * seed, 0.9, 0.5)})
    block = AG.collect_module("mlp_mechanism", sorted(base.iterdir()))
    md = AG.markdown_table(block)
    check("one table line per seed — nothing is averaged away",
          all(f"mlp_s{seed}_arch25k" in md for seed in range(4)))
    check("the header names the row and gap counts", "4 rows, 0 missing" in md)
    check("a missing run is listed under the table",
          "Missing" in AG.markdown_table(AG.collect_module("logit_formula_fit",
                                                           sorted(base.iterdir()))))
    check("None renders as an em dash, never as a silent blank or a 0",
          "—" in AG.markdown_table(AG.collect_module("h3_validity", sorted(base.iterdir())))
          or AG.collect_module("h3_validity", sorted(base.iterdir()))["n_rows"] == 0)


def check_full_aggregate(tmp: Path):
    print("\n-- aggregate(): the written artifacts --")
    base = tmp / "runs5"
    for seed in range(3):
        _make_run(base, f"mlp_f{seed}_arch25k", "mlp", seed,
                  {"mlp_mechanism": _mech_payload(0.3, 0.9, 0.6)})
    out = tmp / "agg_out"
    index = AG.aggregate("*_arch25k", ("mlp_mechanism", "causal_ablation"), base=base, out_dir=out)
    check("an index and a markdown file are written",
          (out / "index.json").exists() and (out / "TABLES.md").exists())
    check("one json per module", (out / "mlp_mechanism.json").exists()
          and (out / "causal_ablation.json").exists())
    check("the index records rows, gaps and the applicable denominator per module",
          {"n_rows", "n_missing", "n_not_applicable", "n_runs_applicable"}
          <= set(index["modules"]["mlp_mechanism"]))
    check("the commit and the run pattern are recorded for reproducibility",
          index["analysis_git_commit"] and index["runs_pattern"] == ["*_arch25k"])

    # Several globs must be accepted. A single pattern cannot isolate the primary block:
    # `*_frac0.3_seed*_arch25k` also matches the two-hot runs (`m2h_..._frac0.3_seed0_arch25k`),
    # which put a third architecture into the primary tables and, through decision_tree, turned the
    # gate's branch from `neither_passes` into `undetermined` (2026-09-04).
    base2 = tmp / "runs6"
    _make_run(base2, "txf_x_arch25k", "transformer", 0,
              {"mlp_mechanism": _mech_payload(0.3, 0.9, 0.6)})
    _make_run(base2, "mlp_x_arch25k", "mlp", 0, {"mlp_mechanism": _mech_payload(0.3, 0.9, 0.6)})
    _make_run(base2, "m2h_x_arch25k", "mlp_twohot", 0,
              {"mlp_mechanism": _mech_payload(0.3, 0.9, 0.6)})
    multi = AG.aggregate(["txf_*_arch25k", "mlp_*_arch25k"], ("mlp_mechanism",), base=base2,
                         out_dir=tmp / "agg_multi")
    check("several globs are accepted and both are recorded",
          multi["runs_pattern"] == ["txf_*_arch25k", "mlp_*_arch25k"] and multi["n_runs"] == 2)
    check("...and the pattern that was NOT given is excluded",
          "m2h_x_arch25k" not in multi["runs"])
    one = AG.aggregate("*_arch25k", ("mlp_mechanism",), base=base2, out_dir=tmp / "agg_one")
    check("a single glob given as a bare string still works", one["n_runs"] == 3)
    check("the 'all seeds shown' rule is stated in the document itself",
          "All seeds are shown" in (out / "TABLES.md").read_text(encoding="utf-8"))
    check("a module with no outputs at all is still listed, with zero rows",
          index["modules"]["causal_ablation"]["n_rows"] == 0
          and index["modules"]["causal_ablation"]["n_missing"] == 3)


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_collect(tmp)
        check_bad_file(tmp)
        check_extractor_error(tmp)
        check_markdown(tmp)
        check_full_aggregate(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
