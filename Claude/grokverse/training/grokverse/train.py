"""Training loop: logarithmic legacy logging + run-format-v2 dense evaluation,
pre-specified checkpoints and a machine-readable manifest.

Run examples (from training/):
    python -m grokverse.train --smoke                  # 1-step sanity check
    python -m grokverse.train --determinism-test       # same seed => identical tensor
    python -m grokverse.train --config fast             # CPU de-risk grok run
    python -m grokverse.train --config nanda --seed 0   # canonical reproduction
    python -m grokverse.train --config arch25k --arch mlp --seed 3 --threads 1   # study run

What v2 adds (docs/dev/RUN_FORMAT_V2.md): train-split accuracy/loss every
``eval_every_train`` steps and test-split every ``eval_every_test`` steps
("dense" curves), transition detection on those curves for every threshold set
in ``config.THRESHOLD_SETS`` reported as an INTERVAL, full ``state_dict``
checkpoints at a fixed step grid plus at the first dense crossings of the
primary thresholds, and ``manifest.json``. All of it runs under ``no_grad`` and
consumes no RNG, so the parameter trajectory is bit-identical to a run without
it (asserted in tests/test_run_format_v2.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoints import save_checkpoint, write_checkpoint_index
from .config import PRESETS, THRESHOLD_SETS, Config, get_config
from .data import make_dataset, split_hash
from .manifest import build_manifest, weight_norms, write_manifest
from .models import build_model, embedding_object, embedding_snapshot
from .seed import set_seed
from .utils import (config_diff, git_commit, lib_versions, log_step_schedule,
                    runs_dir, utcnow)

#: dense-progress lines are printed at most this often (keeps train.log small)
DENSE_PRINT_EVERY = 500
RESULT_FILES = ("run.json", "embeddings.npy", "model_final.pt", "checkpoints.json",
                "manifest.json", "train.log")


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
             p: int) -> tuple[float, float]:
    logits = model.logits_last(x)[:, :p]
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def apply_grokfast(model: torch.nn.Module, gf_ema: dict, cfg: Config) -> None:
    """Grokfast (arXiv:2405.20233): amplify the slow (EMA) gradient component.

    Mutates ``prm.grad`` in place between ``loss.backward()`` and ``opt.step()``.
    Shared by train() and analysis.progress_measures so the two training loops
    cannot drift apart numerically.
    """
    for name, prm in model.named_parameters():
        if prm.grad is None:
            continue
        prev = gf_ema.get(name)
        ema = (prm.grad.detach().clone() if prev is None
               else cfg.grokfast_alpha * prev
               + (1 - cfg.grokfast_alpha) * prm.grad.detach())
        gf_ema[name] = ema
        prm.grad.add_(ema, alpha=cfg.grokfast_lambda)


# --------------------------------------------------------------------------- #
# transition detection                                                         #
# --------------------------------------------------------------------------- #
def detect_transition(curves: dict) -> dict:
    """Legacy detector on the logarithmic log-schedule curves (0.99 / 0.95).

    Returns the first LOGGED step above each threshold; on the log grid that
    carries an unstated uncertainty of one grid spacing — use the dense
    ``transitions`` block for any quantitative timing claim.
    """
    step, tr, te = curves["step"], curves["train_acc"], curves["test_acc"]
    train_sat = next((step[i] for i, a in enumerate(tr) if a >= 0.99), None)
    test_gen = next((step[i] for i, a in enumerate(te) if a >= 0.95), None)
    gap = (test_gen - train_sat) if (train_sat is not None and test_gen is not None) else None
    return {
        "train_saturated_step": train_sat,
        "test_generalized_step": test_gen,
        "grok_gap": gap,
        "final_train_acc": tr[-1] if tr else None,
        "final_test_acc": te[-1] if te else None,
        "legacy_log_grid": True,
    }


def _first_crossing(steps: list[int], values: list[float], thr: float,
                    eval_every: int) -> dict:
    """First evaluated step with value >= thr, plus the previously evaluated
    step. The true crossing lies in ``(previous_evaluated_step, first_crossing_step]``."""
    for i, v in enumerate(values):
        if v >= thr:
            return {"first_crossing_step": int(steps[i]),
                    "previous_evaluated_step": int(steps[i - 1]) if i > 0 else None,
                    "eval_every": int(eval_every)}
    return {"first_crossing_step": None,
            "last_evaluated_step": int(steps[-1]) if steps else None,
            "eval_every": int(eval_every)}


def detect_transition_dense(dense: dict, train_thr: float, test_thr: float) -> dict:
    """Memorization / generalization crossings on the dense curves as intervals.

    ``grok_gap`` is the difference of the two first-crossing steps; its interval
    ``[gen.prev - mem.first, gen.first - mem.prev]`` bounds the true gap given
    the two evaluation grids. Neither number is a point measurement.
    """
    mem = _first_crossing(dense["train_steps"], dense["train_acc"], train_thr,
                          dense["eval_every_train"])
    gen = _first_crossing(dense["test_steps"], dense["test_acc"], test_thr,
                          dense["eval_every_test"])
    gap, interval = None, None
    if mem["first_crossing_step"] is not None and gen["first_crossing_step"] is not None:
        m_first, g_first = mem["first_crossing_step"], gen["first_crossing_step"]
        m_prev = mem["previous_evaluated_step"] if mem["previous_evaluated_step"] is not None else m_first
        g_prev = gen["previous_evaluated_step"] if gen["previous_evaluated_step"] is not None else g_first
        gap = g_first - m_first
        interval = [g_prev - m_first, g_first - m_prev]
    return {"train_thr": float(train_thr), "test_thr": float(test_thr),
            "memorization": mem, "generalization": gen,
            "grok_gap": gap, "grok_gap_interval": interval}


def all_transitions(dense: dict) -> dict:
    return {name: detect_transition_dense(dense, tr, te)
            for name, (tr, te) in THRESHOLD_SETS.items()}


def _empty_dense(cfg: Config) -> dict:
    return {"train_steps": [], "train_acc": [], "train_loss": [],
            "test_steps": [], "test_acc": [], "test_loss": [],
            "eval_every_train": cfg.eval_every_train,
            "eval_every_test": cfg.eval_every_test}


def _result_dict(cfg: Config, curves: dict, dense: dict | None = None,
                 emb_obj: str = "W_E", steps_completed: int | None = None) -> dict:
    out = {
        "config": cfg.to_dict(),
        "protocol_version": 2 if dense is not None else 1,
        "lib_versions": lib_versions(),
        # bit-exactness holds per environment; the thread count is part of it
        # (CPU reductions differ across thread counts), so it is recorded
        "torch_num_threads": torch.get_num_threads(),
        "git_commit": git_commit(),
        "created_utc": utcnow(),
        "embedding_object": emb_obj,
        "logged_steps": curves["step"],
        "curves": {k: curves[k] for k in ("train_loss", "test_loss", "train_acc", "test_acc")},
        "transition": detect_transition(curves),
    }
    if dense is not None:
        out["dense"] = dense
        out["transitions"] = all_transitions(dense)
        out["steps_completed"] = int(steps_completed if steps_completed is not None
                                     else (curves["step"][-1] if curves["step"] else 0))
    return out


def _write_run_json(out_dir: Path, res: dict) -> None:
    (out_dir / "run.json").write_text(json.dumps(res, indent=2))


def _guard_existing_run(out_dir: Path, cfg: Config) -> None:
    """Refuse to write into a run dir recorded under a different config.

    run_id does not encode every config field (e.g. steps), so a differing
    config writing into an existing run dir would silently mix protocols.
    """
    try:
        recorded = json.loads((out_dir / "run.json").read_text())["config"]
    except (json.JSONDecodeError, KeyError):
        print(f"[note] {out_dir / 'run.json'} unreadable (interrupted write?) "
              "— treating as no recorded run", flush=True)
        return
    diff, added = config_diff(recorded, cfg.to_dict())
    if added:
        print(f"[note] {out_dir / 'run.json'} predates config field(s) "
              f"{added} — not treated as a mismatch", flush=True)
    if diff:
        raise RuntimeError(
            f"{out_dir} already holds a run with a different config ({diff}); "
            "delete the directory or pass matching flags (--force to override)")


class _Run:
    """State of one training run: model, data, curves, dense curves, checkpoints."""

    def __init__(self, cfg: Config, out_dir: Path | None, verbose: bool):
        self.cfg, self.out_dir, self.verbose = cfg, out_dir, verbose
        self.p = cfg.p
        self.device = torch.device(cfg.device)
        data = make_dataset(cfg)
        self.train_x, self.train_y = data["train_x"].to(self.device), data["train_y"].to(self.device)
        self.test_x, self.test_y = data["test_x"].to(self.device), data["test_y"].to(self.device)
        self.split = split_hash(data["train_idx"])
        self.model = build_model(cfg).to(self.device)
        self.emb_obj = embedding_object(self.model)
        self.curves = {k: [] for k in ("step", "train_loss", "test_loss", "train_acc", "test_acc")}
        self.embeds: list[np.ndarray] = []
        self.dense = _empty_dense(cfg)
        self.ckpts: list[dict] = []
        self.events = {"memorization": False, "generalization": False}
        self.grid = {s for s in cfg.checkpoint_grid if s <= cfg.steps}
        self.manifest: dict | None = None
        self.steps_completed = 0

    # --- legacy logarithmic snapshot -------------------------------------- #
    def snapshot(self, step: int) -> None:
        tr_l, tr_a = evaluate(self.model, self.train_x, self.train_y, self.p)
        te_l, te_a = evaluate(self.model, self.test_x, self.test_y, self.p)
        c = self.curves
        c["step"].append(int(step))
        c["train_loss"].append(tr_l); c["test_loss"].append(te_l)
        c["train_acc"].append(tr_a); c["test_acc"].append(te_a)
        self.embeds.append(embedding_snapshot(self.model))
        if self.verbose:
            print(f"step {step:6d}  train[loss {tr_l:.4f} acc {tr_a:.3f}]  "
                  f"test[loss {te_l:.4f} acc {te_a:.3f}]", flush=True)
        if self.out_dir is not None:
            _write_run_json(self.out_dir, self.result())

    # --- v2: dense evaluation + event checkpoints -------------------------- #
    def dense_eval(self, step: int) -> None:
        cfg, d = self.cfg, self.dense
        if step % cfg.eval_every_train == 0:
            loss, acc = evaluate(self.model, self.train_x, self.train_y, self.p)
            d["train_steps"].append(int(step)); d["train_acc"].append(acc); d["train_loss"].append(loss)
            if not self.events["memorization"] and acc >= THRESHOLD_SETS["primary"][0]:
                self.events["memorization"] = True
                self.checkpoint(step, "memorization")
        if step % cfg.eval_every_test == 0:
            loss, acc = evaluate(self.model, self.test_x, self.test_y, self.p)
            d["test_steps"].append(int(step)); d["test_acc"].append(acc); d["test_loss"].append(loss)
            if not self.events["generalization"] and acc >= THRESHOLD_SETS["primary"][1]:
                self.events["generalization"] = True
                self.checkpoint(step, "generalization")
        if self.verbose and step % DENSE_PRINT_EVERY == 0 and d["train_acc"] and d["test_acc"]:
            print(f"[dense] step {step:6d}  train acc {d['train_acc'][-1]:.3f}  "
                  f"test acc {d['test_acc'][-1]:.3f}", flush=True)

    def checkpoint(self, step: int, kind: str) -> None:
        if self.out_dir is None:
            return
        save_checkpoint(self.out_dir, self.model, step, kind, self.ckpts)
        write_checkpoint_index(self.out_dir, self.ckpts)

    def result(self) -> dict:
        return _result_dict(self.cfg, self.curves, self.dense, self.emb_obj,
                            self.steps_completed)

    # --- manifest ------------------------------------------------------------ #
    def start_manifest(self) -> None:
        if self.out_dir is None:
            return
        self.manifest = build_manifest(self.cfg, self.split, self.model, utcnow())
        write_manifest(self.out_dir, self.manifest)

    def finish_manifest(self, status: str, t0: float, reason: str | None = None) -> None:
        if self.manifest is None:
            return
        m = self.manifest
        m.update({"status": status, "abort_reason": reason, "end_utc": utcnow(),
                  "elapsed_seconds": float(time.perf_counter() - t0),
                  "steps_completed": int(self.steps_completed),
                  "weight_norms_final": weight_norms(self.model),
                  "checkpoints": list(self.ckpts),
                  "result_files": [f for f in RESULT_FILES if (self.out_dir / f).exists()]})
        write_manifest(self.out_dir, m)


def train(cfg: Config, out_dir: Path | None = None, verbose: bool = True,
          early_stop_acc: float | None = None,
          allow_overwrite: bool = False) -> tuple[dict, torch.nn.Module, np.ndarray]:
    if out_dir is not None and (out_dir / "run.json").exists() and not allow_overwrite:
        _guard_existing_run(out_dir, cfg)
    # Pin the CPU thread count BEFORE any tensor work: torch reductions differ
    # across thread counts, so an unpinned matrix is not reproducible (§3.8).
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    run = _Run(cfg, out_dir, verbose)
    opt = torch.optim.AdamW(
        run.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    logged = set(log_step_schedule(cfg.steps, cfg.n_logged_steps))
    t0 = time.perf_counter()
    run.start_manifest()
    try:
        _train_loop(run, opt, logged, early_stop_acc)
    except KeyboardInterrupt:
        run.finish_manifest("aborted", t0, "KeyboardInterrupt")
        raise
    except Exception as e:  # noqa: BLE001 — recorded, then re-raised
        run.finish_manifest("failed", t0, f"{type(e).__name__}: {e}")
        raise
    result = run.result()
    stacked = np.stack(run.embeds)
    if out_dir is not None:
        run.checkpoint(run.steps_completed, "final")
        _write_run_json(out_dir, result)
        np.save(out_dir / "embeddings.npy", stacked)
        torch.save(run.model.state_dict(), out_dir / "model_final.pt")
        run.finish_manifest("completed", t0)
        if verbose:
            print(f"[saved] {out_dir}  (embeddings {stacked.shape}, "
                  f"{len(run.ckpts)} checkpoints)", flush=True)
    return result, run.model, stacked


def _train_loop(run: _Run, opt, logged: set, early_stop_acc: float | None) -> None:
    cfg, p = run.cfg, run.p
    if 0 in logged:
        run.snapshot(0)
    if 0 in run.grid:
        run.checkpoint(0, "grid")
    run.dense_eval(0)
    gf_ema: dict = {}  # grokfast: per-parameter EMA of gradients
    for step in range(1, cfg.steps + 1):
        run.model.train()
        logits = run.model.logits_last(run.train_x)[:, :p]
        loss = F.cross_entropy(logits, run.train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grokfast:
            apply_grokfast(run.model, gf_ema, cfg)
        opt.step()
        run.steps_completed = step
        if step in run.grid:
            run.checkpoint(step, "grid")
        run.dense_eval(step)
        if step in logged:
            run.snapshot(step)
            if early_stop_acc is not None and run.curves["test_acc"][-1] >= early_stop_acc:
                if run.verbose:
                    print(f"[early-stop] test acc {run.curves['test_acc'][-1]:.3f} "
                          f">= {early_stop_acc} at step {step}", flush=True)
                break


def determinism_test(seed: int = 0) -> bool:
    """Same seed => identical first-batch logits (PLAN Phase 0 acceptance)."""
    cfg = get_config("nanda", seed=seed)

    def first_batch_sum() -> float:
        set_seed(seed)
        data = make_dataset(cfg)
        model = build_model(cfg)
        with torch.no_grad():
            return float(model.logits_last(data["train_x"]).double().sum().item())

    a, b = first_batch_sum(), first_batch_sum()
    ok = a == b
    print(f"determinism: run1={a!r}  run2={b!r}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def _overrides_from_args(args) -> dict:
    overrides: dict = {}
    for attr, key in (("seed", "seed"), ("steps", "steps"), ("arch", "arch"), ("task", "task"),
                      ("train_frac", "train_frac"), ("weight_decay", "weight_decay"),
                      ("grokfast_lambda", "grokfast_lambda"), ("grokfast_alpha", "grokfast_alpha"),
                      ("study", "study"), ("eval_every_train", "eval_every_train"),
                      ("eval_every_test", "eval_every_test"), ("d_mlp", "d_mlp"),
                      ("threads", "threads")):
        v = getattr(args, attr)
        if v is not None:
            overrides[key] = v
    if args.grokfast:
        overrides["grokfast"] = True
    return overrides


def main() -> None:
    ap = argparse.ArgumentParser(description="GROKVERSE trainer")
    ap.add_argument("--config", "--preset", dest="config", default="nanda", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--arch", default=None, choices=["transformer", "mlp", "mlp_twohot"])
    ap.add_argument("--task", default=None, choices=["add", "mul"])
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--grokfast", action="store_true", help="enable Grokfast EMA")
    ap.add_argument("--grokfast-lambda", type=float, default=None)
    ap.add_argument("--grokfast-alpha", type=float, default=None)
    ap.add_argument("--study", default=None, help="run_id suffix (run format v2)")
    ap.add_argument("--eval-every-train", type=int, default=None)
    ap.add_argument("--eval-every-test", type=int, default=None)
    ap.add_argument("--d-mlp", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="1-step sanity check (no save)")
    ap.add_argument("--determinism-test", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--early-stop-acc", type=float, default=None,
                    help="stop once test accuracy >= this (saves time once grokked)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing run dir even if its config differs")
    args = ap.parse_args()

    if args.determinism_test:
        sys.exit(0 if determinism_test() else 1)

    preset = "smoke" if args.smoke else args.config
    cfg = get_config(preset, **_overrides_from_args(args))

    save = not (args.no_save or args.smoke)
    out_dir = runs_dir() / cfg.run_id if save else None
    print(f"[run] {cfg.run_id}  steps={cfg.steps}  device={cfg.device}  "
          f"threads={cfg.threads}", flush=True)
    try:
        result, _, _ = train(cfg, out_dir=out_dir, early_stop_acc=args.early_stop_acc,
                             allow_overwrite=args.force)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    print(json.dumps({"transition": result["transition"],
                      "transitions_primary": result["transitions"]["primary"]}, indent=2))


if __name__ == "__main__":
    main()
