"""Every figure of the study, drawn from the aggregate tables only (INTERFACES §13).

WHY THIS MODULE EXISTS
----------------------
Master prompt §23 item 15 requires that **every figure is reproducible from stored raw data**. The way
to guarantee that is to forbid this module from touching a checkpoint: it reads
`results/aggregate/*.json` and nothing else, and every figure records the files it was drawn from in
its caption and in `results/figures/index.json`. If a number is not already in an aggregate table, no
figure here can show it.

Taking a difference of two stored numbers (an ablation's damage minus its control's) is arithmetic on
stored data, not a recomputed metric — the distinction that matters is that nothing is re-derived from
a model.

WHAT IS DRAWN, AND WHY EXACTLY THIS
------------------------------------
`docs/STATISTICAL_ANALYSIS_PLAN.md` §3 fixes **eight** paired comparisons that carry the study's
claims, "so the count is fixed in advance and cannot grow after the fact". This module draws those
eight and a gate summary — not a gallery. Everything else in the plan is explicitly secondary and
descriptive, and adding a ninth headline figure here would quietly widen the study's claims.

Every figure shows **every seed** as a connected pair (`statistics.plot_paired`), because master
prompt §16 forbids a mean standing in for the seeds behind it.

STATUS: drawing only. It applies no threshold and decides nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils import git_commit, runs_dir, utcnow
from . import statistics as ST
from .decision_tree import _is_point

MODULE = "figures_study"
MODULE_VERSION = "1.0"
#: The architectures compared, in the order they appear on every figure.
ARCH_A, ARCH_B = "transformer", "mlp"


def _f(x):
    """A usable float, or None (booleans are not numbers)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _gap(row, a, b):
    x, y = _f(row.get(a)), _f(row.get(b))
    return None if x is None or y is None else x - y


def _best_fourier_r2(row):
    vals = [_f(row.get(f"{n}__r2_test")) for n in ("sparse_sinusoid", "odd_harmonics")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _square_or_harmonic_fraction(row):
    vals = [_f(row.get(f"u_a__fraction_best_aic_{n}")) for n in ("square", "odd_harmonics")]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


#: The eight pre-specified comparisons of STATISTICAL_ANALYSIS_PLAN §3, in that order.
COMPARISONS: tuple[dict, ...] = (
    {"id": "p1_generalization_step", "label": "generalization step",
     "modules": ("mlp_mechanism", "transformer_mechanism"), "points": ("final",),
     "get": lambda r: _f(r.get("generalization_step")),
     "note": "reported with its evaluation interval in the statistics, not on the figure"},
    {"id": "p2_grokking_gap", "label": "grokking gap (generalization − memorization)",
     "modules": ("mlp_mechanism", "transformer_mechanism"), "points": ("final",),
     "get": lambda r: _gap(r, "generalization_step", "memorization_step")},
    {"id": "p3_structured_fraction", "label": "structured-neuron fraction (primary definition)",
     "modules": ("mlp_mechanism", "transformer_mechanism"), "points": ("crossing", "final"),
     "get": lambda r: _f(r.get("structured_fraction_of_live"))},
    {"id": "p4_phase_R", "label": "phase-relation resultant length R",
     "modules": ("mlp_mechanism", "transformer_mechanism"), "points": ("crossing", "final"),
     "get": lambda r: _f(r.get("phase_R"))},
    {"id": "p5_square_or_harmonic_fraction",
     "label": "fraction of neurons best fit by square or odd-harmonic (H2)",
     "modules": ("wave_fitting",), "points": ("final",), "get": _square_or_harmonic_fraction},
    {"id": "p6_family_minus_top1", "label": "family-fraction minus top-1 fraction (H3b)",
     "modules": ("h3_validity",), "points": ("final",),
     "get": lambda r: _gap(r, "h3b__u_a__family_median", "h3b__u_a__top1_median")},
    {"id": "p7_ablation_damage_minus_control",
     "label": "ablation damage: structured minus size-matched control (H5)",
     "modules": ("causal_ablation",), "points": ("final",),
     "get": lambda r: _gap(r, "remove_structured__drop", "remove_structured__control_mean_drop")},
    {"id": "p8_best_fourier_r2", "label": "end-to-end logit-fit R² of the best Fourier formula (G3)",
     "modules": ("logit_formula_fit",), "points": ("final",), "get": _best_fourier_r2},
)


# --------------------------------------------------------------------------- #
# reading the aggregate tables                                                 #
# --------------------------------------------------------------------------- #
def load_rows(agg_dir: Path, modules) -> list[dict]:
    rows = []
    for module in modules:
        path = agg_dir / f"{module}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")).get("rows", []):
            rows.append({**row, "_module": module, "_source": f"aggregate/{module}.json"})
    return rows


def values_by_arch(rows: list[dict], point: str, get) -> dict:
    """``{arch: {seed: value}}`` at one measurement point, keeping only usable numbers."""
    out: dict[str, dict[int, float]] = {}
    for row in rows:
        if not _is_point(row, point):
            continue
        seed, arch = row.get("seed"), row.get("arch")
        if seed is None or arch is None:
            continue
        value = get(row)
        if value is None:
            continue
        out.setdefault(arch, {})[int(seed)] = value
    return out


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #
def paired_figure(spec: dict, rows: list[dict], point: str, out_dir: Path) -> dict:
    """One pre-specified comparison, every seed shown, with its paired statistics beside it."""
    per_arch = values_by_arch(rows, point, spec["get"])
    a, b = per_arch.get(ARCH_A, {}), per_arch.get(ARCH_B, {})
    seeds = sorted(set(a) & set(b))
    sources = sorted({r["_source"] for r in rows})
    entry = {"id": spec["id"], "label": spec["label"], "point": point,
             "source_files": sources, "n_pairs": len(seeds),
             "seeds": seeds,
             "n_only_transformer": sorted(set(a) - set(b)),
             "n_only_mlp": sorted(set(b) - set(a))}
    if spec.get("note"):
        entry["note"] = spec["note"]
    if not seeds:
        entry["drawn"] = False
        entry["reason"] = (f"no seed has {ARCH_A} and {ARCH_B} values at point={point} yet "
                           f"(have {sorted(a)} vs {sorted(b)})")
        return entry
    va = [a[s] for s in seeds]
    vb = [b[s] for s in seeds]
    path = out_dir / f"{spec['id']}__{point}.png"
    caption = (f"{spec['label']} — {point} checkpoint, {len(seeds)} paired seeds. "
               f"Drawn from {', '.join(sources)}; nothing recomputed.")
    try:
        ST.plot_paired(va, vb, seeds, (ARCH_A, ARCH_B), path,
                       ylabel=spec["label"], title=caption)
        entry["drawn"] = True
        entry["path"] = str(path.name)
    except Exception as exc:                                     # recorded, never swallowed
        entry["drawn"] = False
        entry["reason"] = f"{type(exc).__name__}: {exc}"
    entry["caption"] = caption
    try:
        stats = ST.paired(va, vb, seeds, seeds)
        entry["statistics"] = {k: stats[k] for k in
                               ("n_pairs", "median_a", "median_b", "median_diff", "mean_diff",
                                "bootstrap_ci95", "direction", "sign_test_p",
                                "wilcoxon_signed_rank_p", "cohens_dz", "cliffs_delta")
                               if k in stats}
        entry["statistics"]["difference_definition"] = stats.get("difference_definition")
    except Exception as exc:
        entry["statistics_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def gate_figure(decision_path: Path, out_dir: Path) -> dict:
    """Per-criterion pass / fail / not-evaluable counts per architecture, from decision_tree.json."""
    entry = {"id": "gate_summary", "label": "evidence gate: criteria passed per architecture",
             "source_files": [str(decision_path.name)]}
    if not decision_path.exists():
        entry.update({"drawn": False, "reason": "no decision_tree.json yet"})
        return entry
    report = json.loads(decision_path.read_text(encoding="utf-8"))
    per_arch = report.get("per_architecture", {})
    if not per_arch:
        entry.update({"drawn": False, "reason": "decision_tree.json has no architectures"})
        return entry
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                     # pragma: no cover
        entry.update({"drawn": False, "reason": f"matplotlib unavailable: {exc}"})
        return entry
    criteria = ["G1", "G2", "G3", "G4"]
    archs = sorted(per_arch)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(archs), 1)
    x = np.arange(len(criteria))
    for i, arch in enumerate(archs):
        block = per_arch[arch]
        passed = [block["per_criterion_passed"].get(c, 0) for c in criteria]
        ne = [block["per_criterion_not_evaluable"].get(c, 0) for c in criteria]
        pos = x + i * width - 0.4 + width / 2
        ax.bar(pos, passed, width * 0.9, label=f"{arch}: passed")
        ax.bar(pos, ne, width * 0.9, bottom=passed, alpha=0.35,
               label=f"{arch}: not evaluable")
    ax.axhline(report["thresholds"]["seeds_required"], color="k", ls="--", lw=1,
               label=f"rule: ≥ {report['thresholds']['seeds_required']} of "
                     f"{report['thresholds']['seeds_total']} seeds")
    ax.set_xticks(x, criteria)
    ax.set_ylabel("seeds")
    ax.set_title(f"Evidence gate at point={report.get('measurement_point')} — "
                 f"branch: {report.get('branch', {}).get('branch')}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "gate_summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    entry.update({"drawn": True, "path": path.name,
                  "caption": (f"Evidence gate per architecture, from {decision_path.name}. "
                              "Solid = seeds passing the criterion, shaded = seeds on which it is "
                              "not evaluable. Nothing recomputed."),
                  "branch": report.get("branch", {}).get("branch"),
                  "passes_gate": {a: b.get("passes_gate") for a, b in per_arch.items()}})
    return entry


def figures(agg_dir: Path | str | None = None, out_dir: Path | str | None = None,
            decision_path: Path | str | None = None) -> dict:
    """Draw the eight pre-specified comparisons plus the gate summary."""
    results = runs_dir().parent / "results"
    agg_dir = Path(agg_dir) if agg_dir else results / "aggregate"
    out_dir = Path(out_dir) if out_dir else results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_path = Path(decision_path) if decision_path else results / "decision_tree.json"

    index = {"module": MODULE, "module_version": MODULE_VERSION, "created_utc": utcnow(),
             "analysis_git_commit": git_commit(), "aggregate_dir": str(agg_dir),
             "rule": ("exactly the eight paired comparisons of STATISTICAL_ANALYSIS_PLAN §3, plus a "
                      "gate summary; the count is fixed in advance and does not grow"),
             "source_rule": "drawn from results/aggregate/*.json only — no checkpoint is read",
             "figures": []}
    for spec in COMPARISONS:
        rows = load_rows(agg_dir, spec["modules"])
        for point in spec["points"]:
            index["figures"].append(paired_figure(spec, rows, point, out_dir))
    index["figures"].append(gate_figure(decision_path, out_dir))
    index["n_drawn"] = sum(1 for f in index["figures"] if f.get("drawn"))
    index["n_not_drawn"] = sum(1 for f in index["figures"] if not f.get("drawn"))
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Study figures from the aggregate tables (§13)")
    ap.add_argument("--aggregate-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--decision", type=Path, default=None)
    args = ap.parse_args()
    index = figures(args.aggregate_dir, args.out, args.decision)
    print(f"{index['n_drawn']} drawn, {index['n_not_drawn']} not drawn")
    for f in index["figures"]:
        state = f.get("path") if f.get("drawn") else f"NOT DRAWN — {f.get('reason', '')[:70]}"
        print(f"  {f['id']:38s} {f.get('point', ''):9s} {state}")


if __name__ == "__main__":
    main()
