"""Run every analysis module over a set of runs and checkpoints (INTERFACES §0 entry points).

    python -m grokverse.analysis.driver --runs "*_arch25k" --points crossing final
    python -m grokverse.analysis.driver --runs mlp_add_p113_wd1.0_frac0.3_seed0_arch25k --dry-run
    python -m grokverse.analysis.driver --runs "*_arch25k" --only mlp_mechanism wave_fitting --workers 4

Why this exists
---------------
Each analysis module owns one measurement and writes ``run_dir/analysis/<module>/<tag>.json``. The study
needs all of them, over 40+ runs, at two declared measurement points (the generalization crossing and the
final budget step — ``docs/dev/PREREG_BRIEF.md``). This driver is the only place that loop lives, so the
set of analyses that ran is recorded once, in one manifest, instead of being reconstructed from shell
history.

It deliberately does **no** science: it resolves runs, resolves checkpoint steps from the pre-registered
role rule, calls each module's documented ``analyse(...)`` entry point, and records what succeeded, what
failed and what was skipped. A module that is not implemented yet is reported as ``missing``, never
silently skipped.
"""
from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..checkpoints import assign_roles
from ..utils import git_commit, runs_dir, utcnow

#: Modules invoked per (run, checkpoint), in dependency order. ``needs_key_rule`` modules receive the
#: pre-registered key-frequency rule; ``arch`` restricts a module to the architectures it is defined for.
PER_CHECKPOINT: tuple[dict, ...] = (
    {"module": "key_frequencies", "needs_key_rule": False, "arch": None},
    {"module": "mlp_mechanism", "needs_key_rule": True, "arch": ("mlp", "mlp_twohot")},
    {"module": "transformer_mechanism", "needs_key_rule": True, "arch": ("transformer",)},
    # MLP-only: wave_fitting reads mlp_mechanism.effective_curves and refuses a transformer. The
    # transformer's waveform comparison is not missing — transformer_mechanism computes it from its
    # own effective curves and reports it under `wave_fits`, in the identical shape. Declaring the
    # module arch-agnostic made the driver attempt 20 calls that could never succeed.
    {"module": "wave_fitting", "needs_key_rule": False, "arch": ("mlp", "mlp_twohot")},
    {"module": "logit_formula_fit", "needs_key_rule": True, "arch": None},
    {"module": "h3_validity", "needs_key_rule": True, "arch": None},
    {"module": "causal_ablation", "needs_key_rule": True, "arch": None},
)

#: Modules invoked once per run (they iterate checkpoints themselves).
PER_RUN: tuple[dict, ...] = (
    {"module": "structure_over_time", "needs_key_rule": True, "arch": None},
    {"module": "progress_measures", "needs_key_rule": True, "arch": None,
     "entry": "compute_from_checkpoints"},
)

#: Measurement points of the pre-registration, mapped to checkpoint roles.
POINTS: dict[str, str] = {"crossing": "generalization", "final": "final"}


@dataclass
class Outcome:
    run: str
    module: str
    point: str = ""
    step: int | None = None
    status: str = "ok"           # ok | missing | skipped | failed
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Plan:
    calls: list[tuple] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)


def resolve_runs(patterns: list[str], base: Path | None = None) -> list[Path]:
    """Run directories matching any glob pattern, completed runs only, sorted by name."""
    base = base or runs_dir()
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d / "run.json").exists():
            continue
        if not any(fnmatch.fnmatch(d.name, pat) for pat in patterns):
            continue
        manifest = d / "manifest.json"
        if manifest.exists():
            status = json.loads(manifest.read_text()).get("status")
            if status != "completed":
                continue
        out.append(d)
    if not out:
        raise ValueError(f"no completed run directories match {patterns} under {base}")
    return out


def resolve_steps(run_dir: Path, points: list[str]) -> dict[str, int | None]:
    """Map each requested measurement point to a checkpoint step via the role rule."""
    unknown = [p for p in points if p not in POINTS]
    if unknown:
        raise ValueError(f"unknown measurement point(s) {unknown}; choices: {list(POINTS)}")
    if not (run_dir / "checkpoints.json").exists():
        return {p: None for p in points}          # legacy run: model_final.pt only
    roles = assign_roles(run_dir)
    resolved: dict[str, int | None] = {}
    for p in points:
        role = roles.get(POINTS[p])
        resolved[p] = int(role["step"]) if role else None
    return resolved


def _load_entry(spec: dict):
    """Import a module's entry point, or return None if the module does not exist yet."""
    name = spec["module"]
    entry = spec.get("entry", "analyse")
    try:
        mod = importlib.import_module(f".{name}", package="grokverse.analysis")
    except ModuleNotFoundError as exc:
        if name in str(exc):
            return None
        raise
    return getattr(mod, entry, None)


def _arch_of(run_dir: Path) -> str:
    return json.loads((run_dir / "run.json").read_text())["config"]["arch"]


def build_plan(run_dirs: list[Path], points: list[str], only: list[str] | None,
               key_rule: str, seed: int) -> Plan:
    """Resolve every (run, module, checkpoint) call without executing anything."""
    plan = Plan()
    wanted = set(only) if only else None
    for run_dir in run_dirs:
        arch = _arch_of(run_dir)
        steps = resolve_steps(run_dir, points)
        for spec in PER_RUN + PER_CHECKPOINT:
            name = spec["module"]
            if wanted is not None and name not in wanted:
                continue
            if spec["arch"] is not None and arch not in spec["arch"]:
                plan.outcomes.append(Outcome(run_dir.name, name, status="skipped",
                                             detail=f"not defined for arch {arch!r}"))
                continue
            entry = _load_entry(spec)
            if entry is None:
                plan.outcomes.append(Outcome(run_dir.name, name, status="missing",
                                             detail="module or entry point not implemented yet"))
                continue
            kwargs = {"key_rule": key_rule, "seed": seed} if spec["needs_key_rule"] else {"seed": seed}
            if spec in PER_RUN:
                plan.calls.append((run_dir, name, entry, "", None, kwargs))
                continue
            for point in points:
                step = steps[point]
                if step is None and (run_dir / "checkpoints.json").exists():
                    plan.outcomes.append(Outcome(run_dir.name, name, point, status="skipped",
                                                 detail=f"no checkpoint for role {POINTS[point]!r}"))
                    continue
                plan.calls.append((run_dir, name, entry, point, step, kwargs))
    return plan


def _execute(call) -> Outcome:
    import time
    run_dir, name, entry, point, step, kwargs = call
    t0 = time.perf_counter()
    try:
        # a per-run module (point == "") iterates checkpoints itself and takes no step;
        # a per-checkpoint module always takes one, where step=None means model_final.pt
        if point:
            entry(run_dir, step=step, **kwargs)
        else:
            entry(run_dir, **kwargs)
        return Outcome(run_dir.name, name, point, step, "ok",
                       seconds=round(time.perf_counter() - t0, 2))
    except TypeError as exc:
        # a module whose signature does not match the contract must be loud, not silently retried
        return Outcome(run_dir.name, name, point, step, "failed",
                       f"signature mismatch: {exc}", round(time.perf_counter() - t0, 2))
    except Exception as exc:                                   # noqa: BLE001 - recorded, not swallowed
        return Outcome(run_dir.name, name, point, step, "failed",
                       f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                       round(time.perf_counter() - t0, 2))


def run(plan: Plan, workers: int) -> list[Outcome]:
    outcomes = list(plan.outcomes)
    if workers <= 1:
        outcomes += [_execute(c) for c in plan.calls]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outcomes += list(ex.map(_execute, plan.calls))
    return outcomes


def summarize(outcomes: list[Outcome]) -> dict:
    by_status: dict[str, int] = {}
    for o in outcomes:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    return {
        "counts": by_status,
        "failed": [{"run": o.run, "module": o.module, "point": o.point, "detail": o.detail}
                   for o in outcomes if o.status == "failed"],
        "missing_modules": sorted({o.module for o in outcomes if o.status == "missing"}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run every analysis module over a set of runs")
    ap.add_argument("--runs", nargs="+", default=["*_arch25k"], help="glob pattern(s) of run directories")
    ap.add_argument("--points", nargs="+", default=list(POINTS), choices=list(POINTS),
                    help="measurement points (pre-registered: crossing and final)")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these modules")
    ap.add_argument("--key-rule", default="nanda", help="pre-registered primary key-frequency rule")
    ap.add_argument("--seed", type=int, default=0, help="seed for every random control")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the driver manifest (default: results/analysis_driver.json)")
    args = ap.parse_args()

    run_dirs = resolve_runs(args.runs)
    plan = build_plan(run_dirs, args.points, args.only, args.key_rule, args.seed)
    header = {"runs": [d.name for d in run_dirs], "points": args.points,
              "key_rule": args.key_rule, "seed": args.seed,
              "n_calls": len(plan.calls), "analysis_git_commit": git_commit()}
    if args.dry_run:
        print(json.dumps({**header, "planned": [
            {"run": c[0].name, "module": c[1], "point": c[3], "step": c[4]} for c in plan.calls],
            "pre_resolved": summarize(plan.outcomes)}, indent=2))
        return

    outcomes = run(plan, args.workers)
    manifest = {**header, "started_utc": utcnow(),
                "outcomes": [vars(o) for o in outcomes], "summary": summarize(outcomes)}
    out = args.out or (runs_dir().parent / "results" / "analysis_driver.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**header, "summary": manifest["summary"], "manifest": str(out)}, indent=2))
    if manifest["summary"]["counts"].get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
