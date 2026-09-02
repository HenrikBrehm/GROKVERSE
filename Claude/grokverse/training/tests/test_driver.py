"""Checks for the analysis driver (run: python tests/test_driver.py).

The driver contains no science; what must not break is its bookkeeping: which runs it selects,
which checkpoint step each measurement point resolves to, that an unimplemented module is
reported as ``missing`` rather than skipped silently, that an architecture-specific module is
skipped only on the other architecture, and that a failing module is recorded as ``failed``
with its exception instead of aborting the sweep.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.analysis import driver  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _fake_run(base: Path, name: str, arch: str, *, completed: bool = True,
              checkpoints: bool = True) -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "config": {"arch": arch, "p": 113, "run_id": name, "steps": 5000},
        "transitions": {"primary": {
            "train_thr": 0.99, "test_thr": 0.95,
            "memorization": {"first_crossing_step": 140, "previous_evaluated_step": 130,
                             "eval_every": 10},
            "generalization": {"first_crossing_step": 3300, "previous_evaluated_step": 3275,
                               "eval_every": 25},
            "grok_gap": 3160}}}))
    (d / "manifest.json").write_text(json.dumps(
        {"status": "completed" if completed else "running"}))
    if checkpoints:
        (d / "checkpoints.json").write_text(json.dumps([
            {"step": 0, "path": "ckpt_step000000.pt", "kind": "grid"},
            {"step": 2000, "path": "ckpt_step002000.pt", "kind": "grid"},
            {"step": 140, "path": "ckpt_step000140.pt", "kind": "memorization"},
            {"step": 3300, "path": "ckpt_step003300.pt", "kind": "generalization"},
            {"step": 5000, "path": "ckpt_step005000.pt", "kind": "grid,final"},
        ]))
    return d


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="driver_test_"))
    try:
        base = tmp / "runs"
        base.mkdir()
        txf = _fake_run(base, "txf_x_arch25k", "transformer")
        mlp = _fake_run(base, "mlp_x_arch25k", "mlp")
        _fake_run(base, "txf_running_arch25k", "transformer", completed=False)
        _fake_run(base, "mlp_legacy", "mlp", checkpoints=False)

        # --- run selection -------------------------------------------------
        found = [d.name for d in driver.resolve_runs(["*_arch25k"], base=base)]
        check("only completed runs are selected", found == ["mlp_x_arch25k", "txf_x_arch25k"])
        check("a still-running run is excluded", "txf_running_arch25k" not in found)
        try:
            driver.resolve_runs(["nothing_matches_*"], base=base)
            raised = False
        except ValueError:
            raised = True
        check("an empty selection raises instead of doing nothing", raised)

        # --- measurement points --------------------------------------------
        steps = driver.resolve_steps(txf, ["crossing", "final"])
        check("crossing resolves to the generalization checkpoint", steps["crossing"] == 3300)
        check("final resolves to the last step", steps["final"] == 5000)
        check("a legacy run without checkpoints resolves to None (model_final.pt)",
              driver.resolve_steps(base / "mlp_legacy", ["final"])["final"] is None)
        try:
            driver.resolve_steps(txf, ["halfway"])
            raised = False
        except ValueError:
            raised = True
        check("an unknown measurement point raises", raised)

        # --- planning ------------------------------------------------------
        plan = driver.build_plan([txf, mlp], ["crossing", "final"], None, "nanda", 0)
        missing = {o.module for o in plan.outcomes if o.status == "missing"}
        check("unimplemented modules are reported as missing, never skipped silently",
              "causal_ablation" in missing and "structure_over_time" in missing)
        skipped = {(o.run, o.module) for o in plan.outcomes if o.status == "skipped"}
        check("an MLP-only module is skipped on the transformer run",
              ("txf_x_arch25k", "mlp_mechanism") in skipped)
        check("the same module is NOT skipped on the MLP run",
              ("mlp_x_arch25k", "mlp_mechanism") not in skipped)
        planned = {(c[0].name, c[1], c[3], c[4]) for c in plan.calls}
        check("every planned per-checkpoint call carries a resolved step",
              all(step is not None for _, _, point, step in planned if point))
        check("both measurement points are planned for an implemented module",
              {p for r, m, p, s in planned if m == "mlp_mechanism"} == {"crossing", "final"})

        only = driver.build_plan([mlp], ["final"], ["wave_fitting"], "nanda", 0)
        check("--only restricts the plan to the named module",
              {c[1] for c in only.calls} == {"wave_fitting"} and len(only.calls) == 1)

        # --- execution bookkeeping -----------------------------------------
        def boom(run_dir, **kw):
            raise RuntimeError("planted failure")

        def fine(run_dir, **kw):
            return {"ok": True}

        outcomes = driver.run(driver.Plan(calls=[
            (mlp, "boom", boom, "final", 5000, {"seed": 0}),
            (mlp, "fine", fine, "final", 5000, {"seed": 0}),
        ]), workers=1)
        by = {o.module: o for o in outcomes}
        check("a failing module is recorded as failed, not raised", by["boom"].status == "failed")
        check("the failure detail names the exception", "planted failure" in by["boom"].detail)
        check("a sibling module still runs after a failure", by["fine"].status == "ok")
        summary = driver.summarize(outcomes)
        check("the summary counts one failure", summary["counts"].get("failed") == 1)
        check("the summary lists the failing run and module",
              summary["failed"][0]["module"] == "boom")

        # a per-run module (point == "") must be called WITHOUT a step argument
        seen = {}

        def per_run(run_dir, **kw):
            seen["kwargs"] = kw

        driver.run(driver.Plan(calls=[(mlp, "per_run", per_run, "", None, {"seed": 0})]), workers=1)
        check("a per-run module is called without a step argument", "step" not in seen["kwargs"])

        print("\nALL DRIVER CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
