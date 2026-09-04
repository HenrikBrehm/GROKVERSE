"""H4 across seeds: does structure form during the memorization plateau? (PREREGISTRATION §3 H4)

WHY THIS MODULE EXISTS
----------------------
`analysis/structure_over_time.py` computes the trajectory and the onset indicator **per run**. H4 is a
statement about the ordering of three step counts — memorization, structure onset, generalization —
and that ordering only becomes a result when it holds across seeds. This module reads the aggregate
table and reports, per architecture and per metric, how many seeds place the structure onset before
the generalization crossing.

WHAT IT CANNOT SHOW, STATED IN THE OUTPUT
------------------------------------------
This analysis is **correlational**. A metric rising before the generalization jump is consistent with
structure forming early; it cannot show that the structure caused the jump, and it cannot show that it
had to form. Causal claims come from `analysis/causal_ablation.py` alone
(`docs/CAUSAL_ABLATION_PLAN.md` §9). The onset indicator inherits the fragility
`structure_over_time` documents: its baseline standard deviation is taken over **two** checkpoints, so
the step is sensitive to the checkpoint grid. It is an indicator, not a measured transition, and both
facts travel with every number below.

A metric whose onset is undefined on some seeds is reported with that count, never dropped — the
denominator is what a reader judges the claim by.

STATUS: applies a pre-registered ordering question to already-measured numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import git_commit, runs_dir, utcnow
from . import statistics as ST

MODULE = "h4_report"
MODULE_VERSION = "1.0"
#: Metrics H4 is stated over. Each must exist as `h4__<metric>__onset_step` in the aggregate.
METRICS: tuple[str, ...] = (
    "structured_fraction_of_live", "phase_relation_R", "embedding_top8_concentration",
    "logit_key_subspace_share", "median_odd_minus_even_u_a", "median_family_fraction",
    "fraction_best_aic_square",
)


def compute(agg_dir: Path | str | None = None) -> dict:
    results = runs_dir().parent / "results"
    agg_dir = Path(agg_dir) if agg_dir else results / "aggregate"
    path = agg_dir / "structure_over_time.json"
    if not path.exists():
        raise ValueError(f"{path} not found — run analysis.aggregate first")
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])

    report: dict = {
        "module": MODULE, "module_version": MODULE_VERSION, "created_utc": utcnow(),
        "analysis_git_commit": git_commit(), "n_rows": len(rows),
        "hypothesis": ("H4: mechanistic structure metrics rise already during the memorization "
                       "plateau and precede the generalization jump"),
        "reading": ("CORRELATIONAL. Structure preceding generalization is consistent with H4; it "
                    "cannot show that the structure caused the jump, nor that it had to form. "
                    "Causal claims come from analysis/causal_ablation.py alone."),
        "onset_caveat": ("the onset indicator's baseline standard deviation is taken over two "
                         "checkpoints, so the step is sensitive to the checkpoint grid; it is an "
                         "indicator, not a measured transition"),
        "per_architecture": {},
    }
    for arch in sorted({r.get("arch") for r in rows if r.get("arch")}):
        rs = [r for r in rows if r.get("arch") == arch]
        mem = [r["memorization_step"] for r in rs if r.get("memorization_step") is not None]
        gen = [r["generalization_step"] for r in rs if r.get("generalization_step") is not None]
        block: dict = {
            "n_seeds": len(rs),
            "memorization_step": ST.robust(mem),
            "generalization_step": ST.robust(gen),
            "metrics": {},
        }
        for metric in METRICS:
            onset_key = f"h4__{metric}__onset_step"
            if onset_key not in (rs[0] if rs else {}):
                continue
            per_seed = []
            for r in rs:
                onset, g = r.get(onset_key), r.get("generalization_step")
                per_seed.append({
                    "seed": r.get("seed"), "onset_step": onset,
                    "memorization_step": r.get("memorization_step"),
                    "generalization_step": g,
                    "onset_before_generalization": (None if onset is None or g is None
                                                    else bool(onset < g)),
                    "onset_after_memorization": (
                        None if onset is None or r.get("memorization_step") is None
                        else bool(onset > r["memorization_step"])),
                    "value_at_init": r.get(f"h4__{metric}__init"),
                    "value_at_pre_generalization": r.get(f"h4__{metric}__pre_gen"),
                    "change": r.get(f"h4__{metric}__change"),
                })
            defined = [d for d in per_seed if d["onset_step"] is not None]
            before = [d for d in defined if d["onset_before_generalization"]]
            after_mem = [d for d in defined if d["onset_after_memorization"]]
            changes = [d["change"] for d in per_seed if d["change"] is not None]
            block["metrics"][metric] = {
                "n_seeds": len(rs), "n_onset_defined": len(defined),
                "n_onset_before_generalization": len(before),
                "n_onset_after_memorization": len(after_mem),
                "onset_step": ST.robust([d["onset_step"] for d in defined]) if defined else {"n": 0},
                "change_init_to_pre_generalization": ST.robust(changes) if changes else {"n": 0},
                "n_change_undefined": len(rs) - len(changes),
                "per_seed": per_seed,
                "reading": (
                    f"{len(before)} of {len(defined)} seeds with a defined onset place it before the "
                    f"generalization crossing"
                    + (f"; {len(rs) - len(defined)} seed(s) have no defined onset for this metric"
                       if len(defined) < len(rs) else "")),
            }
        report["per_architecture"][arch] = block

    # the ordering H4 asserts, summarized without asserting causation
    summary = {}
    for arch, block in report["per_architecture"].items():
        holds = {m: (b["n_onset_before_generalization"], b["n_onset_defined"])
                 for m, b in block["metrics"].items()}
        summary[arch] = {
            "memorization_median": block["memorization_step"].get("median"),
            "generalization_median": block["generalization_step"].get("median"),
            "onset_before_generalization_by_metric": {m: f"{a}/{b}" for m, (a, b) in holds.items()},
        }
    report["summary"] = summary
    return report


def summary_lines(rep: dict) -> list[str]:
    lines = [rep["hypothesis"]]
    for arch, block in sorted(rep["per_architecture"].items()):
        lines.append(f"  {arch}: memorization median "
                     f"{block['memorization_step'].get('median')}, generalization median "
                     f"{block['generalization_step'].get('median')} (n={block['n_seeds']})")
        for metric, m in block["metrics"].items():
            onset = m["onset_step"].get("median")
            lines.append(f"      {metric:32s} onset median {onset}  "
                         f"before generalization {m['n_onset_before_generalization']}/"
                         f"{m['n_onset_defined']}  after memorization "
                         f"{m['n_onset_after_memorization']}/{m['n_onset_defined']}  "
                         f"change median {m['change_init_to_pre_generalization'].get('median')}")
    lines.append(f"  {rep['reading']}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="H4 across seeds (PREREGISTRATION §3 H4)")
    ap.add_argument("--aggregate-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = compute(args.aggregate_dir)
    out = args.out or (runs_dir().parent / "results" / "h4_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("\n".join(summary_lines(rep)))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
