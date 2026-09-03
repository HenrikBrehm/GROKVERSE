"""Collect every per-run analysis JSON into one table per module (INTERFACES §13).

WHY THIS MODULE EXISTS
----------------------
Every analysis module writes one JSON per run and measurement point. Nothing in the study may be read
from a single one of those files: the design is **paired by seed**, and master prompt §16 requires that
*all individual seeds are shown*, never only a mean. This module is the only place that walks those
files, and it is deliberately dumb — it extracts, it does not compute. Every number it emits already
exists in a per-run file, and the row records the file it came from.

WHAT IT DOES NOT DO
--------------------
* It never recomputes a metric. If a number is not in a per-run JSON, it is absent here too.
* It never averages away a seed. `results/aggregate/<module>.json` holds one row per (run, point), and
  the markdown tables print every row.
* It never applies a threshold or decides anything. `analysis/decision_tree.py` reads these tables and
  applies the **pre-registered** gate; this module has no opinion.

A missing file is recorded as a missing row with its reason, never skipped silently — an analysis that
failed on three seeds must be visible as three gaps, not as a smaller denominator.

STATUS: measurement plumbing only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils import git_commit, runs_dir, utcnow

MODULE = "aggregate"
MODULE_VERSION = "1.0"
#: Modules collected, in the order the report reads them.
MODULES: tuple[str, ...] = (
    "key_frequencies", "mlp_mechanism", "transformer_mechanism", "wave_fitting",
    "logit_formula_fit", "h3_validity", "causal_ablation", "structure_over_time",
    "progress_measures",
)
#: The two pre-registered measurement points, resolved from the file tag.
POINT_OF_TAG = {"final": "final"}


# --------------------------------------------------------------------------- #
# per-module extractors — each returns a FLAT dict of scalars                  #
# --------------------------------------------------------------------------- #
def _q(d: dict, *path, default=None):
    """Nested lookup that returns ``default`` instead of raising — a missing key is data."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _mechanism(res: dict) -> dict:
    """`mlp_mechanism` / `transformer_mechanism`: the G1 and G2 inputs, plus headline numbers."""
    sn = res.get("structured_neurons", {})
    primary_name = sn.get("primary")
    prim = _q(sn, "definitions", primary_name, default={}) or {}
    pr = prim.get("phase_relation", {}) or {}
    pc = _q(res, "neuron_tables", "per_curve", default={}) or {}
    out = {
        "n_neurons": res.get("n_neurons"),
        "n_live": sn.get("n_live"),
        "structured_definition": primary_name,
        "structured_n": prim.get("n"),
        "structured_fraction_of_live": prim.get("fraction_of_live_neurons"),
        "phase_R": pr.get("resultant_length"),
        "phase_null_q95": pr.get("null_resultant_q95"),
        "phase_exceeds_null": pr.get("exceeds_null_q95"),
        "phase_n": pr.get("n"),
        "phase_insufficient_neurons": pr.get("insufficient_neurons"),
        "sum_dependence_median": _q(res, "sum_dependence", "median_sum"),
        "diff_dependence_median": _q(res, "sum_dependence", "median_diff"),
        "additivity_r2_median": _q(res, "additivity", "summary", "median"),
        "direct_path_share": res.get("direct_path_share_of_logit_variance"),
    }
    for curve in ("u_a", "u_b", "out"):
        out[f"family_fraction_median_{curve}"] = _q(pc, curve, "family_fraction", "median")
        out[f"dominant_fraction_median_{curve}"] = _q(pc, curve, "dominant_fraction", "median")
        out[f"top8_median_{curve}"] = _q(pc, curve, "topk_concentration", "8", "median")
    for name, entry in (sn.get("definitions") or {}).items():
        out[f"structured_fraction__{name}"] = (entry or {}).get("fraction_of_live_neurons")
    return out


def _logit_formula_fit(res: dict) -> dict:
    """G3's inputs: each candidate formula's test fit, and the random-frequency control."""
    out = {"key_frequencies": res.get("key_frequencies"),
           "family_size": res.get("family_size"),
           "control_set_size": _q(res, "control_random_m", "set_size"),
           "control_n": _q(res, "control_random_m", "n_control")}
    for name, block in (res.get("formulas") or {}).items():
        out[f"{name}__r2_test"] = block.get("r2_test")
        out[f"{name}__r2_all"] = block.get("r2_all")
        out[f"{name}__argmax_acc_test"] = block.get("argmax_accuracy_test")
        out[f"{name}__aic"] = block.get("aic")
    for field in ("r2_test", "r2_all"):
        out[f"control_random_m__{field}_q95"] = _q(res, "control_random_m", field, "control_q95")
        out[f"control_random_m__{field}_std"] = _q(res, "control_random_m", field, "control_std")
    return out


def _causal_ablation(res: dict) -> dict:
    """G4's inputs: the gate verdict and the three ablations it reads."""
    g4 = res.get("gate_criterion_g4", {}) or {}
    out = {"baseline_test_acc": _q(res, "baseline", "test_acc"),
           "n_structured": None, "g4_passed": g4.get("g4_passed"),
           "g4_remove_structured_necessary": g4.get("remove_structured_necessary"),
           "g4_remove_key_freqs_necessary": g4.get("remove_key_freqs_necessary"),
           "g4_keep_structured_sufficient": g4.get("keep_structured_sufficient"),
           "g4_ids_used": g4.get("ids_used")}
    ns = res.get("n_structured") or {}
    if isinstance(ns, dict) and ns:
        out["n_structured"] = list(ns.values())[0]
    for name, entry in (res.get("ablations") or {}).items():
        if not isinstance(entry, dict) or "observed" not in entry:
            continue
        out[f"{name}__test_acc"] = _q(entry, "observed", "test_acc")
        out[f"{name}__drop"] = _q(entry, "observed", "test_accuracy_drop")
        out[f"{name}__control_mean_drop"] = _q(entry, "control", "test_accuracy_drop", "mean")
        out[f"{name}__z"] = entry.get("z")
        out[f"{name}__necessary"] = _q(entry, "reading", "necessary_by_plan_rule")
        out[f"{name}__sufficient"] = _q(entry, "reading", "sufficient_by_plan_rule")
        out[f"{name}__no_damage"] = _q(entry, "reading", "no_damage")
    return out


def _h3_validity(res: dict) -> dict:
    """H3's per-run quantities; the architecture comparison itself is not made here."""
    out = {}
    for curve, block in (_q(res, "h3a_waveform_sensitivity", "per_curve", default={}) or {}).items():
        for k in (1, 4, 8):
            out[f"h3a__{curve}__waveform_sensitivity_top{k}"] = _q(
                block, "waveform_sensitivity_median", f"top{k}")
        out[f"h3a__{curve}__shape_separation"] = block.get("shape_separation_median")
    for curve, block in (_q(res, "h3b_harmonic_aware_structure", "per_curve",
                            default={}) or {}).items():
        out[f"h3b__{curve}__family_median"] = _q(block, "family_fraction", "median")
        out[f"h3b__{curve}__top1_median"] = _q(block, "top1_fraction", "median")
        out[f"h3b__{curve}__matched_top_m_median"] = _q(block, "matched_top_m_fraction", "median")
        out[f"h3b__{curve}__random_null_median"] = _q(block, "random_m_null", "mean", "median")
        out[f"h3b__{curve}__family_minus_matched_max"] = block.get("max_family_minus_matched_top_m")
    for name, entry in (_q(res, "h3b_structured_fractions", "definitions", default={}) or {}).items():
        out[f"h3b__structured_fraction__{name}"] = (entry or {}).get("fraction_of_live_neurons")
    return out


def _structure_over_time(res: dict) -> dict:
    """H4: the init → pre_generalization change and the onset step of each metric."""
    out = {f"role__{r}": s for r, s in (res.get("roles") or {}).items()}
    for name, entry in (_q(res, "h4", "per_metric", default={}) or {}).items():
        out[f"h4__{name}__init"] = entry.get("value_at_init")
        out[f"h4__{name}__pre_gen"] = entry.get("value_at_pre_generalization")
        out[f"h4__{name}__change"] = entry.get("change")
        out[f"h4__{name}__onset_step"] = _q(entry, "onset", "step")
    for name, entry in (_q(res, "h4", "bootstrap_over_neurons", default={}) or {}).items():
        out[f"h4__boot__{name}__mean_diff"] = entry.get("mean_difference")
        out[f"h4__boot__{name}__ci95_low"] = entry.get("ci95_low")
        out[f"h4__boot__{name}__ci95_high"] = entry.get("ci95_high")
    return out


def _wave_fitting(res: dict) -> dict:
    """Per curve: the share of neurons each waveform model wins by AIC.

    The result is keyed `results[curve]["fraction_best_by_aic"][model]` — an earlier version of this
    extractor guessed flat `fraction_best_aic_<model>` keys, found nothing, and silently produced
    rows with no waveform columns at all. The H2 figure would have stayed blank with no error.
    """
    out = {}
    for curve, block in (res or {}).items():
        if not isinstance(block, dict):
            continue
        for model, value in (block.get("fraction_best_by_aic") or {}).items():
            out[f"{curve}__fraction_best_aic_{model}"] = value
        for model, value in (block.get("fraction_best_by_aicc") or {}).items():
            out[f"{curve}__fraction_best_aicc_{model}"] = value
        if "fraction_square_aic_below_sinusoid" in block:
            out[f"{curve}__fraction_square_aic_below_sinusoid"] = \
                block["fraction_square_aic_below_sinusoid"]
        delta = block.get("delta_aic_sinusoid_minus_square")
        if isinstance(delta, dict) and "median" in delta:
            out[f"{curve}__delta_aic_sinusoid_minus_square_median"] = delta["median"]
    return out


def _key_frequencies(res: dict) -> dict:
    out = {}
    for rule, block in (res.get("rules") or res or {}).items():
        if isinstance(block, dict) and "key_frequencies" in block:
            out[f"{rule}__keys"] = block["key_frequencies"]
            out[f"{rule}__n_keys"] = len(block["key_frequencies"])
    return out


def _progress_measures(res: dict) -> dict:
    """Restricted / excluded loss per protocol and split, at every checkpoint that carries one."""
    out = {}
    ck = res.get("checkpoints") or res.get("per_checkpoint") or []
    if isinstance(ck, list) and ck:
        last = ck[-1]
        for proto, block in (last.get("protocols") or {}).items():
            if isinstance(block, dict):
                out[f"{proto}__restricted"] = _q(block, "restricted", "loss", default=
                                                 block.get("restricted_loss"))
                out[f"{proto}__excluded"] = _q(block, "excluded", "loss", default=
                                               block.get("excluded_loss"))
    return out


EXTRACTORS = {
    "mlp_mechanism": _mechanism, "transformer_mechanism": _mechanism,
    "logit_formula_fit": _logit_formula_fit, "causal_ablation": _causal_ablation,
    "h3_validity": _h3_validity, "structure_over_time": _structure_over_time,
    "wave_fitting": _wave_fitting, "key_frequencies": _key_frequencies,
    "progress_measures": _progress_measures,
}


# --------------------------------------------------------------------------- #
# collection                                                                   #
# --------------------------------------------------------------------------- #
def run_metadata(run_dir: Path) -> dict:
    """Architecture, seed, split hash and the logged transition steps of one run."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    cfg = run.get("config", {})
    meta = {"run_id": run_dir.name, "arch": cfg.get("arch"), "seed": cfg.get("seed"),
            "p": cfg.get("p"), "train_frac": cfg.get("train_frac"),
            "weight_decay": cfg.get("weight_decay"), "grokfast": cfg.get("grokfast"),
            "d_mlp": cfg.get("d_mlp"), "steps": cfg.get("steps")}
    man = run_dir / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        meta.update({"split_hash": m.get("split_hash"), "status": m.get("status"),
                     "steps_completed": m.get("steps_completed"),
                     "git_commit": m.get("git_commit")})
    tr = _q(run, "transitions", "primary", default={}) or {}
    meta["memorization_step"] = _q(tr, "memorization", "first_crossing_step")
    meta["generalization_step"] = _q(tr, "generalization", "first_crossing_step")
    return meta


def _module_arch(module: str):
    """Which architectures a module is defined for, read from the driver's own registry."""
    from .driver import PER_CHECKPOINT, PER_RUN
    for spec in PER_RUN + PER_CHECKPOINT:
        if spec["module"] == module:
            return spec["arch"]
    return None


def collect_module(module: str, run_dirs: list[Path]) -> dict:
    """One row per (run, measurement point) for ``module``.

    A run for which the module is *not defined* (``mlp_mechanism`` on a transformer) is listed under
    ``not_applicable``, NOT under ``missing``. Mixing the two would inflate the missing count and
    quietly change the denominator a reader judges completeness by.
    """
    rows, missing, not_applicable = [], [], []
    extract = EXTRACTORS.get(module)
    arch_ok = _module_arch(module)
    for run_dir in run_dirs:
        meta = run_metadata(run_dir)
        if arch_ok is not None and meta.get("arch") not in arch_ok:
            not_applicable.append({**meta, "reason": f"{module} is not defined for arch "
                                                     f"{meta.get('arch')!r}"})
            continue
        files = sorted((run_dir / "analysis" / module).glob("*.json")) \
            if (run_dir / "analysis" / module).is_dir() else []
        if not files:
            missing.append({**meta, "reason": f"no analysis/{module}/*.json"})
            continue
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:                             # recorded, never skipped
                missing.append({**meta, "file": str(path), "reason": f"unreadable: {exc}"})
                continue
            row = {**meta,
                   "tag": path.stem,
                   "step": payload.get("step"),
                   "module_version": payload.get("module_version"),
                   "analysis_git_commit": payload.get("analysis_git_commit"),
                   "checkpoint_sha256": payload.get("checkpoint_sha256"),
                   "source_file": str(path.relative_to(run_dir.parent.parent)),
                   "key_frequencies_used": _q(payload, "params", "key_frequencies")}
            if extract is not None:
                try:
                    row.update(extract(payload.get("results", {}) or {}))
                except Exception as exc:                          # a bad row is data, not a crash
                    row["extractor_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    # n_rows counts (run, measurement point) pairs — two per run for a per-checkpoint module — while
    # n_missing and n_not_applicable count RUNS. They are reported under names that say which, so the
    # two are never added into a denominator that means neither.
    return {"module": module, "n_rows": len(rows), "n_missing_runs": len(missing),
            "n_not_applicable_runs": len(not_applicable),
            "n_runs_applicable": len(rows and {r["run_id"] for r in rows} or set()) + len(missing),
            "n_missing": len(missing), "n_not_applicable": len(not_applicable),
            "rows": rows, "missing": missing, "not_applicable": not_applicable}


def markdown_table(block: dict, columns: list[str] | None = None, max_columns: int = 12) -> str:
    """A markdown table with **every** seed shown (master prompt §16), never a mean alone."""
    rows = block["rows"]
    if not rows:
        # A module with no rows still has GAPS, and they must be visible. Returning a bare
        # "no rows" line here would hide, say, twenty runs whose analysis failed.
        na = block.get("n_not_applicable", 0)
        head = (f"### {block['module']}  (0 rows, {block.get('n_missing', 0)} missing"
                + (f", {na} not applicable to this architecture)" if na else ")"))
        body = ["", "_No rows._"]
        if block.get("missing"):
            body += ["", f"**Missing ({block['n_missing']}):** "
                     + ", ".join(f"{m.get('run_id')} ({m.get('reason')})"
                                 for m in block["missing"])]
        return head + "\n".join(body) + "\n"
    lead = [c for c in ("run_id", "arch", "seed", "tag", "step") if c in rows[0]]
    rest = [c for c in (columns or []) if c in rows[0]]
    if not rest:
        skip = set(lead) | {"source_file", "checkpoint_sha256", "analysis_git_commit",
                            "module_version", "p", "steps", "d_mlp", "git_commit", "status"}
        rest = [c for c in rows[0] if c not in skip
                and isinstance(rows[0][c], (int, float, bool, type(None)))][:max_columns]
    cols = lead + rest

    def cell(v):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    na = block.get("n_not_applicable", 0)
    out = [f"### {block['module']}  ({block['n_rows']} rows, {block['n_missing']} missing"
           + (f", {na} not applicable to this architecture)" if na else ")"), "",
           "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in sorted(rows, key=lambda r: (str(r.get("arch")), r.get("seed") or 0, str(r.get("tag")))):
        out.append("| " + " | ".join(cell(r.get(c)) for c in cols) + " |")
    if block["missing"]:
        out += ["", f"**Missing ({block['n_missing']}):** "
                + ", ".join(f"{m.get('run_id')} ({m.get('reason')})" for m in block["missing"])]
    return "\n".join(out) + "\n"


def aggregate(runs: str = "*_arch25k", modules=MODULES, base: Path | None = None,
              out_dir: Path | None = None) -> dict:
    """Collect every module over every matching completed run and write the tables."""
    from .driver import resolve_runs
    run_dirs = resolve_runs([runs], base)
    out_dir = Path(out_dir) if out_dir else (runs_dir().parent / "results" / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"module": MODULE, "module_version": MODULE_VERSION, "created_utc": utcnow(),
             "analysis_git_commit": git_commit(), "runs_pattern": runs,
             "n_runs": len(run_dirs), "runs": [d.name for d in run_dirs], "modules": {}}
    md = [f"# Aggregated analysis tables\n",
          f"Generated {index['created_utc']} at commit `{index['analysis_git_commit']}` over "
          f"{len(run_dirs)} runs matching `{runs}`.\n",
          "Every row is one run at one measurement point. **All seeds are shown** — no row is "
          "averaged away (master prompt §16). Numbers are copied from the per-run files named in "
          "`source_file`; nothing here is recomputed.\n"]
    for module in modules:
        block = collect_module(module, run_dirs)
        (out_dir / f"{module}.json").write_text(json.dumps(block, indent=2, default=str), encoding="utf-8")
        index["modules"][module] = {"n_rows": block["n_rows"], "n_missing": block["n_missing"],
                                    "n_not_applicable": block["n_not_applicable"],
                                    "n_runs_applicable": block["n_runs_applicable"],
                                    "file": f"aggregate/{module}.json"}
        md.append(markdown_table(block))
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    # utf-8 is pinned: the tables carry em dashes and section signs, and Path.write_text defaults to
    # the locale encoding (cp1252 on this machine), which silently mangles them.
    (out_dir / "TABLES.md").write_text("\n".join(md), encoding="utf-8")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-run analysis JSONs (INTERFACES §13)")
    ap.add_argument("--runs", default="*_arch25k")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    index = aggregate(args.runs, tuple(args.only) if args.only else MODULES, out_dir=args.out)
    print(f"{index['n_runs']} runs")
    for module, info in index["modules"].items():
        print(f"  {module:24s} rows={info['n_rows']:3d}  missing={info['n_missing']}")


if __name__ == "__main__":
    main()
