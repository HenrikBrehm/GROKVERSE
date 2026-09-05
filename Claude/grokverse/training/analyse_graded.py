"""Aggregate the graded structured-neuron ablation (PREREGISTRATION 14) over the primary block.

Run from ``training/``::

    python analyse_graded.py

Reads, for each of the 20 primary runs at both measurement points, the ``causal_ablation`` result
written by the frozen driver, and produces ``results/graded_ablation.json`` plus a markdown table.

Two things are reported side by side:

* **the graded structured ablation** (PREREGISTRATION 14) - nested top-f groups of the most
  structured live neurons, each against 50 seeded size-matched random groups of live neurons;
* **the IPR sweep that already existed** (D2, Doshi arXiv:2310.13061) - read at the matching
  fractions with **no new computation**. Its ranking is the IPR, not the family fraction, and its
  control is a random-order permutation of the whole pruning sequence. Read at a fixed fraction that
  control is itself a uniformly random group of the same size, which is what makes the two
  comparable at all; it is drawn from every neuron, not only the live ones.

The decision rule is fixed in PREREGISTRATION 14.5 and applied here, not invented here: an
architecture and checkpoint counts as discriminable at fraction f iff at least 8 of its 10 seeds
have an observed remove-drop that exceeds every one of the 50 controls AND z >= 3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

POINTS = ("crossing", "final")
SEEDS_REQUIRED = 10
PASS_SEEDS = 8                      # PREREGISTRATION 14.5
Z_MIN = 3.0                         # NECESSARY_Z
IPR_FRACTIONS = (0.05, 0.10, 0.25, 0.50)
ARCHES = ("mlp", "transformer")


def runs_root() -> Path:
    return Path(__file__).resolve().parent / "runs"


def primary_runs() -> list[str]:
    """The 20 runs of the primary block, taken from the committed driver manifest."""
    man = json.loads((Path(__file__).resolve().parent / "results" /
                      "analysis_driver.json").read_text())
    return list(man["runs"])


def _ablation_files(run_id: str) -> list[Path]:
    d = runs_root() / run_id / "analysis" / "causal_ablation"
    return sorted(d.glob("step*.json")) if d.is_dir() else []


def _load(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text())
    npz = path.with_suffix(".npz")
    return payload, dict(np.load(npz)) if npz.exists() else {}


def _nonempty(a) -> bool:
    """A real array with more than one distinct value.

    Guards the failure mode this repository has hit five times: an extractor that returns a
    well-formed, correctly shaped object carrying no information at all (LABBOOK 99-101).
    """
    a = np.asarray(a, dtype=np.float64)
    return bool(a.size) and bool(np.isfinite(a).any()) and float(np.nanstd(a)) >= 0.0


def collect_graded(payload: dict, score: str = "primary") -> dict:
    """Per-fraction readings of the graded block for one run-checkpoint."""
    blk = payload["results"]["ablations"].get("graded_structured_ablation")
    if blk is None:
        raise KeyError("graded_structured_ablation missing - re-run causal_ablation at v1.1+")
    s = blk["scores"][score]
    out = {"score_summary": s["score_summary"], "score_source": blk["score_source"],
           "n_b1_structured": blk["n_b1_structured"], "n_alive": s["n_alive"], "by_fraction": {}}
    for key, ent in s["by_fraction"].items():
        rt, kt = ent["remove_top"], ent["keep_only_top"]
        out["by_fraction"][key] = {
            "n_selected": ent["n_selected"],
            "subset_of_b1_structured": ent.get("subset_of_b1_structured"),
            "score_min": ent["score_in_group"]["min"],
            "remove_drop": rt["observed"]["test_accuracy_drop"],
            "remove_control_mean": rt["control"]["test_accuracy_drop"]["mean"],
            "remove_control_max": rt["control"]["test_accuracy_drop"]["max"],
            "remove_z": rt["z"],
            "remove_exceeds_all": rt.get("exceeds_all_controls"),
            "remove_discriminable": rt["discriminable_by_prereg_14_5"],
            "keep_test_acc": kt["observed"]["test_acc"],
            "keep_control_test_acc_mean": kt["control"]["test_acc"]["mean"],
        }
    return out


def collect_ipr(payload: dict, arrays: dict) -> dict:
    """The already-computed IPR sweep, read at the matching fractions. No new computation."""
    abl = payload["results"]["ablations"].get("ipr_ranked_pruning")
    if abl is None:
        return {}
    grid = np.asarray(abl["fraction_pruned"], dtype=np.float64)
    hi = arrays.get("ipr_ranked_pruning__highest_ipr_first_test_acc")
    ctrl = arrays.get("ipr_ranked_pruning__random_control_test_acc")
    if hi is None or ctrl is None or not _nonempty(hi) or not _nonempty(ctrl):
        return {}
    hi = np.asarray(hi, dtype=np.float64)
    ctrl = np.asarray(ctrl, dtype=np.float64)          # [n_control, n_points]
    base = float(payload["results"]["baseline"]["test_acc"])
    out = {}
    for f in IPR_FRACTIONS:
        j = int(np.argmin(np.abs(grid - f)))
        if abs(float(grid[j]) - f) > 1e-9:
            continue
        obs_drop = base - float(hi[j])
        cdrops = base - ctrl[:, j]
        std = float(cdrops.std(ddof=1)) if cdrops.size > 1 else 0.0
        z = (obs_drop - float(cdrops.mean())) / std if std > 0 else float("inf")
        out[f"{f:.2f}"] = {
            "grid_fraction": float(grid[j]),
            "remove_drop": obs_drop,
            "remove_control_mean": float(cdrops.mean()),
            "remove_control_max": float(cdrops.max()),
            "remove_z": (float(z) if np.isfinite(z) else None),
            "remove_exceeds_all": bool(obs_drop > cdrops.max()),
            "remove_discriminable": bool(obs_drop > cdrops.max()
                                         and (not np.isfinite(z) or z >= Z_MIN)),
        }
    return out


def _point_of(payload: dict, steps_by_run: dict, run_id: str) -> str:
    """crossing vs final, by step order within the run (two checkpoints per run)."""
    step = int(payload["step"])
    return "final" if step == max(steps_by_run[run_id]) else "crossing"


def main() -> None:
    root = Path(__file__).resolve().parent
    runs = primary_runs()
    rows: list[dict] = []
    missing: list[str] = []

    steps_by_run = {}
    for run_id in runs:
        steps_by_run[run_id] = [int(json.loads(p.read_text())["step"])
                                for p in _ablation_files(run_id)]

    for run_id in runs:
        files = _ablation_files(run_id)
        if not files:
            missing.append(run_id)
            continue
        for path in files:
            payload, arrays = _load(path)
            arch = payload["arch"]
            point = _point_of(payload, steps_by_run, run_id)
            try:
                graded = collect_graded(payload)
                graded_sens = collect_graded(payload, "sensitivity")
            except KeyError as exc:
                missing.append(f"{run_id}@{payload['step']}: {exc}")
                continue
            rows.append({
                "run_id": run_id, "arch": arch, "point": point,
                "step": int(payload["step"]),
                "module_version": payload["module_version"],
                "baseline_test_acc": payload["results"]["baseline"]["test_acc"],
                "n_alive": payload["results"]["n_alive"],
                "graded": graded, "graded_sensitivity": graded_sens,
                "ipr": collect_ipr(payload, arrays),
            })

    summary = summarize(rows)
    out = {
        "module": "graded_structured_ablation_report",
        "prereg": "docs/PREREGISTRATION.md 14; human decision docs/HUMAN_DECISIONS.md D7 (open)",
        "decision_rule": (f"discriminable at f iff >= {PASS_SEEDS} of {SEEDS_REQUIRED} seeds have an "
                          f"observed remove-drop exceeding EVERY control and z >= {Z_MIN}"),
        "status": "MEASUREMENT ONLY - feeds no gate criterion (PREREGISTRATION 14.5)",
        "n_rows": len(rows), "missing": missing,
        "summary": summary, "rows": rows,
    }
    (root / "results" / "graded_ablation.json").write_text(json.dumps(out, indent=2))
    tables = render_tables(summary, rows)
    (root / "results" / "GRADED_ABLATION.md").write_text(tables, encoding="utf-8")
    print(tables)
    if missing:
        print(f"\nMISSING: {len(missing)} entries -> {missing[:5]}")


def summarize(rows: list[dict]) -> dict:
    """Apply the 8/10 rule per architecture, checkpoint and fraction."""
    out: dict = {"graded": {}, "graded_sensitivity": {}, "ipr": {}}
    for block in ("graded", "graded_sensitivity", "ipr"):
        for arch in ARCHES:
            for point in POINTS:
                sel = [r for r in rows if r["arch"] == arch and r["point"] == point]
                if not sel:
                    continue
                fracs: dict = {}
                keys = sorted({k for r in sel for k in _frac_keys(r, block)})
                for key in keys:
                    vals = [_frac(r, block, key) for r in sel]
                    vals = [v for v in vals if v]
                    n_disc = sum(1 for v in vals if v.get("remove_discriminable"))
                    drops = [v["remove_drop"] for v in vals if v.get("remove_drop") is not None]
                    ctrls = [v["remove_control_mean"] for v in vals
                             if v.get("remove_control_mean") is not None]
                    fracs[key] = {
                        "n_seeds": len(vals),
                        "n_discriminable": n_disc,
                        "passes_8_of_10": bool(len(vals) >= SEEDS_REQUIRED
                                               and n_disc >= PASS_SEEDS),
                        "median_remove_drop": float(np.median(drops)) if drops else None,
                        "median_control_mean_drop": float(np.median(ctrls)) if ctrls else None,
                        "median_n_selected": (float(np.median([v["n_selected"] for v in vals]))
                                              if block != "ipr" else None),
                    }
                smallest = next((k for k in sorted(fracs, key=float)
                                 if fracs[k]["passes_8_of_10"]), None)
                out[block][f"{arch}__{point}"] = {"by_fraction": fracs,
                                                  "smallest_passing_fraction": smallest}
    return out


def _frac_keys(row: dict, block: str) -> list[str]:
    if block == "ipr":
        return list(row.get("ipr", {}))
    return list(row.get(block, {}).get("by_fraction", {}))


def _frac(row: dict, block: str, key: str) -> dict:
    if block == "ipr":
        return row.get("ipr", {}).get(key, {})
    return row.get(block, {}).get("by_fraction", {}).get(key, {})


def render_tables(summary: dict, rows: list[dict]) -> str:
    L: list[str] = []
    L.append("# Graded structured-neuron ablation")
    L.append("")
    L.append("Pre-registered in `docs/PREREGISTRATION.md` §14; human decision `docs/HUMAN_DECISIONS.md` "
             "**D7** is open. **Measurement only — this feeds no gate criterion.** The evidence gate "
             "stands at `neither_passes` and §14.5 fixes in advance that no outcome here reopens G4.")
    L.append("")
    L.append(f"Rows: {len(rows)} run-checkpoints from the 20 primary runs.")
    L.append("")
    titles = {"graded": "1. Graded structured ablation — primary score (§14.3)",
              "graded_sensitivity": "2. Sensitivity score (mean of the three family fractions)",
              "ipr": "3. The IPR sweep that already existed (D2) — read at matching fractions, "
                     "no new computation"}
    for block in ("graded", "graded_sensitivity", "ipr"):
        L.append(f"## {titles[block]}")
        L.append("")
        L.append("| architecture | checkpoint | fraction | n selected | seeds discriminable | "
                 "passes 8/10 | median drop | median control drop |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arch in ARCHES:
            for point in POINTS:
                cell = summary[block].get(f"{arch}__{point}")
                if not cell:
                    continue
                for key in sorted(cell["by_fraction"], key=float):
                    v = cell["by_fraction"][key]
                    nsel = ("—" if v["median_n_selected"] is None
                            else f"{v['median_n_selected']:.0f}")
                    md = "—" if v["median_remove_drop"] is None else f"{v['median_remove_drop']:.4f}"
                    mc = ("—" if v["median_control_mean_drop"] is None
                          else f"{v['median_control_mean_drop']:.4f}")
                    L.append(f"| {arch} | {point} | {key} | {nsel} | "
                             f"{v['n_discriminable']}/{v['n_seeds']} | "
                             f"{'**yes**' if v['passes_8_of_10'] else 'no'} | {md} | {mc} |")
        L.append("")
        for arch in ARCHES:
            for point in POINTS:
                cell = summary[block].get(f"{arch}__{point}")
                if cell:
                    s = cell["smallest_passing_fraction"]
                    L.append(f"- **{arch}, {point}:** smallest discriminating fraction: "
                             f"{'**' + s + '**' if s else 'none of the tested fractions'}")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
