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


def check_entry_points_accept_driver_kwargs() -> None:
    """Every module the driver lists must actually RUN with the kwargs the driver passes.

    This is an integration check on purpose. `inspect.signature` cannot catch the failure it exists
    for: on 2026-09-03 both `key_frequencies.analyse` and `progress_measures.compute_from_checkpoints`
    accepted ``**kw`` and forwarded it into a helper that rejects ``seed``, so the signature bound
    fine and every one of their 60 driver calls raised TypeError at runtime. The driver recorded them
    as failed and carried on, which is correct behaviour — but nothing failed *loudly* until the
    outputs were counted. Only a real invocation catches that class of bug.

    A module may legitimately raise something else here (a synthetic run is small and odd); the
    assertion is specifically that it is never a TypeError about the driver's own arguments.
    """
    import dataclasses

    import torch

    from grokverse import checkpoints as CK
    from grokverse.config import get_config
    from grokverse.models import build_model
    from grokverse.seed import set_seed

    print("\n-- every driver module runs with the driver's kwargs --")
    steps = (0, 10, 100)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for arch in ("mlp", "transformer"):
            extra = dict(d_model=16, d_head=4, n_heads=4) if arch == "transformer" else {}
            cfg = get_config("nanda", p=23, arch=arch, d_mlp=8, seed=0, **extra)
            set_seed(0)
            model = build_model(cfg)
            model.eval()
            run_dir = tmp / f"{arch}_{cfg.run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            entries: list[dict] = []
            for step in steps:
                CK.save_checkpoint(run_dir, model, step, "grid", entries)
            CK.write_checkpoint_index(run_dir, entries)
            (run_dir / "run.json").write_text(json.dumps({
                "config": dataclasses.asdict(cfg), "git_commit": "synthetic-test",
                "steps_completed": steps[-1],
                "transitions": {"primary": {"memorization": {"first_crossing_step": 10},
                                            "generalization": {"first_crossing_step": 100}}}}))
            for spec in driver.PER_RUN + driver.PER_CHECKPOINT:
                if spec["arch"] is not None and cfg.arch not in spec["arch"]:
                    continue
                entry = driver._load_entry(spec)
                if entry is None:
                    continue
                kwargs = ({"key_rule": "nanda", "seed": 0} if spec["needs_key_rule"]
                          else {"seed": 0})
                if spec in driver.PER_CHECKPOINT:
                    kwargs["step"] = steps[-1]
                try:
                    entry(run_dir, **kwargs)
                    ok, detail = True, "ran"
                except TypeError as exc:
                    ok, detail = False, f"TypeError: {exc}"
                except Exception as exc:                       # not a contract failure
                    ok, detail = True, f"ran ({type(exc).__name__}, not a signature problem)"
                check(f"{arch}: {spec['module']} accepts the driver's kwargs — {detail}", ok)


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
        # This used to name `causal_ablation` and `structure_over_time` literally, and so failed the
        # day one of them was implemented (2026-09-03). The property worth pinning is the *mechanism*
        # — an unimplemented module is surfaced, never dropped — so it is now asserted against a name
        # that can never exist, plus the driver's own list resolved dynamically. It needs no edit as
        # the remaining modules land.
        check("a module that does not exist resolves to no entry point",
              driver._load_entry({"module": "no_such_module_xyz", "needs_key_rule": False,
                                  "arch": None}) is None)
        unimportable = {s["module"] for s in driver.PER_RUN + driver.PER_CHECKPOINT
                        if driver._load_entry(s) is None}
        check("every module the driver lists but cannot import is reported missing, never skipped "
              "silently", unimportable <= missing)
        check("...and an implemented, architecture-agnostic module is planned instead",
              "causal_ablation" not in missing
              and any(c[1] == "causal_ablation" for c in plan.calls))
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

        check_entry_points_accept_driver_kwargs()

        print("\nALL DRIVER CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
