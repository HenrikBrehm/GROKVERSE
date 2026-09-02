"""Training loop with logarithmic checkpoint logging + smoke/determinism modes.

Run examples (from training/):
    python -m grokverse.train --smoke                  # 1-step sanity check
    python -m grokverse.train --determinism-test       # same seed => identical tensor
    python -m grokverse.train --config fast             # CPU de-risk grok run
    python -m grokverse.train --config nanda --seed 0   # canonical reproduction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import PRESETS, Config, get_config
from .data import make_dataset
from .models import build_model
from .seed import set_seed
from .utils import (config_diff, git_commit, lib_versions, log_step_schedule,
                    runs_dir, utcnow)


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


def detect_transition(curves: dict) -> dict:
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
    }


def _result_dict(cfg: Config, curves: dict) -> dict:
    return {
        "config": cfg.to_dict(),
        "lib_versions": lib_versions(),
        # bit-exactness holds per environment; the thread count is part of it
        # (CPU reductions differ across thread counts), so it is recorded
        "torch_num_threads": torch.get_num_threads(),
        "git_commit": git_commit(),
        "created_utc": utcnow(),
        "logged_steps": curves["step"],
        "curves": {k: curves[k] for k in ("train_loss", "test_loss", "train_acc", "test_acc")},
        "transition": detect_transition(curves),
    }


def _write_run_json(out_dir: Path, cfg: Config, curves: dict) -> None:
    (out_dir / "run.json").write_text(json.dumps(_result_dict(cfg, curves), indent=2))


def train(cfg: Config, out_dir: Path | None = None, verbose: bool = True,
          early_stop_acc: float | None = None,
          allow_overwrite: bool = False) -> tuple[dict, torch.nn.Module, np.ndarray]:
    # run_id does not encode every config field (e.g. steps), so a differing
    # config writing into an existing run dir would silently mix protocols.
    if out_dir is not None and (out_dir / "run.json").exists() and not allow_overwrite:
        try:
            recorded = json.loads((out_dir / "run.json").read_text())["config"]
        except (json.JSONDecodeError, KeyError):
            print(f"[note] {out_dir / 'run.json'} unreadable (interrupted write?) "
                  "— treating as no recorded run", flush=True)
            recorded = None
        if recorded is not None:
            diff, added = config_diff(recorded, cfg.to_dict())
            if added:
                print(f"[note] {out_dir / 'run.json'} predates config field(s) "
                      f"{added} — not treated as a mismatch", flush=True)
            if diff:
                raise RuntimeError(
                    f"{out_dir} already holds a run with a different config ({diff}); "
                    "delete the directory or pass matching flags (--force to override)")
    # Pin the CPU thread count BEFORE any tensor work: torch reductions differ
    # across thread counts, so an unpinned matrix is not reproducible (§3.8).
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    data = make_dataset(cfg)
    train_x, train_y = data["train_x"].to(device), data["train_y"].to(device)
    test_x, test_y = data["test_x"].to(device), data["test_y"].to(device)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    logged = set(log_step_schedule(cfg.steps, cfg.n_logged_steps))
    curves = {k: [] for k in ("step", "train_loss", "test_loss", "train_acc", "test_acc")}
    embeds: list[np.ndarray] = []
    p = cfg.p

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(step: int) -> None:
        tr_l, tr_a = evaluate(model, train_x, train_y, p)
        te_l, te_a = evaluate(model, test_x, test_y, p)
        curves["step"].append(int(step))
        curves["train_loss"].append(tr_l)
        curves["test_loss"].append(te_l)
        curves["train_acc"].append(tr_a)
        curves["test_acc"].append(te_a)
        embeds.append(model.W_E.detach().cpu().numpy().copy())
        if verbose:
            print(f"step {step:6d}  train[loss {tr_l:.4f} acc {tr_a:.3f}]  "
                  f"test[loss {te_l:.4f} acc {te_a:.3f}]", flush=True)
        if out_dir is not None:
            _write_run_json(out_dir, cfg, curves)

    if 0 in logged:
        snapshot(0)
    gf_ema: dict = {}  # grokfast: per-parameter EMA of gradients
    for step in range(1, cfg.steps + 1):
        model.train()
        logits = model.logits_last(train_x)[:, :p]
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grokfast:
            apply_grokfast(model, gf_ema, cfg)
        opt.step()
        if step in logged:
            snapshot(step)
            if early_stop_acc is not None and curves["test_acc"][-1] >= early_stop_acc:
                if verbose:
                    print(f"[early-stop] test acc {curves['test_acc'][-1]:.3f} "
                          f">= {early_stop_acc} at step {step}", flush=True)
                break

    result = _result_dict(cfg, curves)
    stacked = np.stack(embeds)
    if out_dir is not None:
        _write_run_json(out_dir, cfg, curves)
        np.save(out_dir / "embeddings.npy", stacked)
        torch.save(model.state_dict(), out_dir / "model_final.pt")
        if verbose:
            print(f"[saved] {out_dir}  (embeddings {stacked.shape})", flush=True)
    return result, model, stacked


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


def main() -> None:
    ap = argparse.ArgumentParser(description="GROKVERSE trainer")
    ap.add_argument("--config", default="nanda", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--arch", default=None, choices=["transformer", "mlp"])
    ap.add_argument("--task", default=None, choices=["add", "mul"])
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--grokfast", action="store_true", help="enable Grokfast EMA")
    ap.add_argument("--grokfast-lambda", type=float, default=None)
    ap.add_argument("--grokfast-alpha", type=float, default=None)
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
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.arch is not None:
        overrides["arch"] = args.arch
    if args.task is not None:
        overrides["task"] = args.task
    if args.train_frac is not None:
        overrides["train_frac"] = args.train_frac
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.grokfast:
        overrides["grokfast"] = True
    if args.grokfast_lambda is not None:
        overrides["grokfast_lambda"] = args.grokfast_lambda
    if args.grokfast_alpha is not None:
        overrides["grokfast_alpha"] = args.grokfast_alpha
    cfg = get_config(preset, **overrides)

    save = not (args.no_save or args.smoke)
    out_dir = runs_dir() / cfg.run_id if save else None
    print(f"[run] {cfg.run_id}  steps={cfg.steps}  device={cfg.device}", flush=True)
    try:
        result, _, _ = train(cfg, out_dir=out_dir, early_stop_acc=args.early_stop_acc,
                             allow_overwrite=args.force)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    print(json.dumps(result["transition"], indent=2))


if __name__ == "__main__":
    main()
