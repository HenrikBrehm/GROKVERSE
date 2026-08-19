"""Restricted & excluded loss — the Nanda et al. 2023 progress measures (Phase 3).

These are the *named contribution* of "Progress Measures for Grokking via
Mechanistic Interpretability" (arXiv:2301.05217) and the held-out-independent way
to show the three phases of grokking (memorization -> circuit formation ->
cleanup) without peeking at test accuracy.

Idea. A grokked modular-addition network computes ``(a + b) mod p`` with a sparse
set of *key frequencies* k: it writes the answer as a sum of waves
``cos/sin(2*pi*k*(a+b)/p)``. Take the model's logits over the full ``(a, b)``
grid, ``L[a, b, c]``, and 2D-Fourier-transform them over the two *input* axes
``a`` and ``b`` in the orthonormal real Fourier basis ``F`` over Z_p:

    Lhat[i, j, c] = sum_{a,b} F[i,a] F[j,b] L[a,b,c]            (forward)
    L[a, b, c]    = sum_{i,j} F[i,a] F[j,b] Lhat[i,j,c]         (inverse; F is orthonormal)

* **restricted loss** — keep *only* the 2D components built from key-frequency
  modes (plus the constant), rebuild the logits, and measure the loss. If the
  network's computation really *is* the key-frequency trig circuit, the
  restricted logits already solve the task -> restricted loss collapses to ~full
  loss as the circuit forms.
* **excluded loss** — do the opposite: *delete* every component that touches a
  key frequency (keep the constant and all non-key frequencies). This destroys
  the circuit. Early (memorization) the network is not using the key frequencies,
  so excluded loss stays low; as the circuit forms, excluded loss rises sharply.

The crossing/divergence of the two curves is the mechanistic signature of
grokking. This module re-trains deterministically, captures the model state at
every logged step, fixes the key frequencies from the *final* embedding, and
applies them retroactively to every step (Nanda's protocol). Honesty checks: it
refuses to run with a config that differs from an existing ``run.json`` in the
same run directory, and after re-training it compares the recovered transition
against the recorded one (stored as ``sanity["matches_recorded_run"]``; a
mismatch is loudly warned about, e.g. a thread-count-induced grid wobble).

Protocol note: the losses here are measured over the FULL (a, b) grid (train
and held-out points together). Nanda et al. evaluate restricted loss on test
data and excluded loss on train data; the full-grid variant carries the same
divergence signature and is what RESULTS.md §2 describes.

Usage (from training/) — pass the SAME --steps as the recorded run:
    python -m grokverse.analysis.progress_measures --config grokfast --train-frac 0.5 --steps 8000 --seed 0 --early-stop-acc 0.98
    python -m grokverse.analysis.progress_measures --config nanda --steps 40000 --seed 0 --early-stop-acc 0.95   # canonical (slow)
"""
from __future__ import annotations

import argparse
import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ..config import PRESETS, Config, get_config  # noqa: E402
from ..data import make_dataset  # noqa: E402
from ..models import build_model  # noqa: E402
from ..seed import set_seed  # noqa: E402
from ..train import apply_grokfast, detect_transition  # noqa: E402
from ..utils import log_step_schedule, runs_dir  # noqa: E402
from .fourier import dominant_frequencies, fourier_basis  # noqa: E402


# --------------------------------------------------------------------------- #
# deterministic re-train that keeps a CPU copy of the model state every logged  #
# step (train.py only persists per-step embeddings, not full weights)          #
# --------------------------------------------------------------------------- #
def train_capturing_states(cfg: Config, early_stop_acc: float | None = None):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    data = make_dataset(cfg)
    train_x, train_y = data["train_x"].to(device), data["train_y"].to(device)
    test_x, test_y = data["test_x"].to(device), data["test_y"].to(device)
    p = cfg.p

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    logged = set(log_step_schedule(cfg.steps, cfg.n_logged_steps))
    curves: dict[str, list] = {k: [] for k in ("step", "train_acc", "test_acc")}
    states: list[dict] = []

    @torch.no_grad()
    def acc(x: torch.Tensor, y: torch.Tensor) -> float:
        lg = model.logits_last(x)[:, :p]
        return (lg.argmax(-1) == y).float().mean().item()

    def snapshot(step: int) -> None:
        curves["step"].append(int(step))
        curves["train_acc"].append(acc(train_x, train_y))
        curves["test_acc"].append(acc(test_x, test_y))
        states.append(copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))

    if 0 in logged:
        snapshot(0)
    gf_ema: dict = {}
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
                break
    return model, data, curves, states


# --------------------------------------------------------------------------- #
# 2D Fourier transform over the (a, b) input grid                              #
# --------------------------------------------------------------------------- #
def fwd2d(L: np.ndarray, Fb: np.ndarray) -> np.ndarray:
    """L[a,b,c] -> Lhat[i,j,c] = sum_{a,b} F[i,a] F[j,b] L[a,b,c]."""
    t = np.einsum("ia,abc->ibc", Fb, L, optimize=True)
    return np.einsum("jb,ibc->ijc", Fb, t, optimize=True)


def inv2d(Lhat: np.ndarray, Fb: np.ndarray) -> np.ndarray:
    """Lhat[i,j,c] -> L[a,b,c] = sum_{i,j} F[i,a] F[j,b] Lhat[i,j,c] (F orthonormal)."""
    t = np.einsum("ia,ijc->ajc", Fb, Lhat, optimize=True)
    return np.einsum("jb,ajc->abc", Fb, t, optimize=True)


def _mode_indices(p: int, key_freqs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over the p Fourier-basis rows.

    Row layout (see fourier.fourier_basis): row 0 = const; then for k=1.. the
    pair (cos_k, sin_k) at rows 1+2(k-1), 2+2(k-1). Returns:
      keep_restricted : const + the key cos/sin rows
      is_key_freq     : the key cos/sin rows only (const excluded)
    """
    keep = np.zeros(p, dtype=bool)
    is_key = np.zeros(p, dtype=bool)
    keep[0] = True  # constant (DC / bias) always kept in the restricted circuit
    for k in key_freqs:
        cos_row, sin_row = 1 + 2 * (k - 1), 2 + 2 * (k - 1)
        for r in (cos_row, sin_row):
            keep[r] = True
            is_key[r] = True
    return keep, is_key


def _grid_logits(model, all_x: torch.Tensor, p: int) -> np.ndarray:
    with torch.no_grad():
        lg = model.logits_last(all_x)[:, :p].double().cpu().numpy()
    return lg.reshape(p, p, p)  # [a, b, c]  (a outer, b inner — matches data.py)


def _ce(logits_grid: np.ndarray, y: torch.Tensor) -> float:
    flat = torch.from_numpy(logits_grid.reshape(-1, logits_grid.shape[-1]))
    return float(F.cross_entropy(flat, y).item())


def compute(cfg: Config, early_stop_acc: float | None = None,
            max_steps_measured: int = 120) -> dict:
    model, data, curves, states = train_capturing_states(cfg, early_stop_acc)
    p = cfg.p
    Fb, _ = fourier_basis(p)
    all_x, all_y = data["all_x"], data["all_y"]

    # Key frequencies fixed from the FINAL embedding (Nanda: define once, apply to all steps)
    final_W_E = model.W_E.detach().cpu().numpy()
    dom = dominant_frequencies(final_W_E, p)
    key_freqs = dom["dominant"]
    keep_restr, is_key = _mode_indices(p, key_freqs)
    mask_restr = keep_restr[:, None] & keep_restr[None, :]      # key 2D block (+const)
    mask_excl = ~(is_key[:, None] | is_key[None, :])            # everything NOT touching a key freq

    steps = curves["step"]
    # subsample if a long un-accelerated run produced many checkpoints
    idx = (np.linspace(0, len(steps) - 1, max_steps_measured).round().astype(int)
           if len(steps) > max_steps_measured else np.arange(len(steps)))
    idx = np.unique(idx)

    base = build_model(cfg)
    full_loss, restr_loss, excl_loss, m_steps = [], [], [], []
    for t in idx:
        base.load_state_dict(states[t])
        base.eval()
        L = _grid_logits(base, all_x, p)
        Lhat = fwd2d(L, Fb)
        L_restr = inv2d(Lhat * mask_restr[:, :, None], Fb)
        L_excl = inv2d(Lhat * mask_excl[:, :, None], Fb)
        full_loss.append(_ce(L, all_y))
        restr_loss.append(_ce(L_restr, all_y))
        excl_loss.append(_ce(L_excl, all_y))
        m_steps.append(steps[t])

    # ---- sanity checks (honesty: a wrong measure is worse than none) ----
    base.load_state_dict(states[-1])
    base.eval()
    L = _grid_logits(base, all_x, p)
    Lhat = fwd2d(L, Fb)
    recon_err = float(np.abs(inv2d(Lhat, Fb) - L).max())          # (1) F is a valid orthonormal transform
    all_modes = inv2d(Lhat * np.ones_like(mask_restr)[:, :, None], Fb)
    all_modes_err = float(np.abs(all_modes - L).max())            # (2) keeping all modes == identity
    full_final = _ce(L, all_y)
    restr_final = _ce(inv2d(Lhat * mask_restr[:, :, None], Fb), all_y)
    excl_final = _ce(inv2d(Lhat * mask_excl[:, :, None], Fb), all_y)
    sanity = {
        "reconstruction_max_abs_err": recon_err,
        "all_modes_max_abs_err": all_modes_err,
        "n_key_freqs": len(key_freqs),
        "final_full_loss": full_final,
        "final_restricted_loss": restr_final,   # expect ~ full_loss (circuit IS the key freqs)
        "final_excluded_loss": excl_final,      # expect >> full_loss (circuit destroyed)
        "restricted_recovers_full": bool(restr_final < full_final + 0.10),
        "excluded_destroys_solution": bool(excl_final > full_final + 1.0),
    }

    transition = detect_transition(curves)

    # Honesty check: the deterministic re-train must recover the transition the
    # recorded run.json logged (same config => same grid => same crossings).
    run_json = runs_dir() / cfg.run_id / "run.json"
    if run_json.exists():
        try:
            recorded = json.loads(run_json.read_text())["transition"]
        except (json.JSONDecodeError, KeyError):
            print(f"[note] {run_json} unreadable — skipping recorded-run comparison",
                  flush=True)
            recorded = None
        if recorded is not None:
            keys = ("train_saturated_step", "test_generalized_step")
            matches = all(transition[k] == recorded[k] for k in keys)
            sanity["matches_recorded_run"] = matches
            sanity["recorded_transition"] = recorded
            if not matches:
                print(f"[WARNING] re-trained transition {transition} != recorded "
                      f"{recorded} — check --steps/config match the recorded run",
                      flush=True)

    return {
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "torch_num_threads": torch.get_num_threads(),
        "key_frequencies": key_freqs,
        "transition": transition,
        "measured_steps": [int(s) for s in m_steps],
        "full_loss": full_loss,
        "restricted_loss": restr_loss,
        "excluded_loss": excl_loss,
        "test_acc_at_measured": [curves["test_acc"][t] for t in idx],
        "train_acc_at_measured": [curves["train_acc"][t] for t in idx],
        "sanity": sanity,
    }


def plot(res: dict, out_path) -> None:
    s = np.clip(np.array(res["measured_steps"]), 1, None)
    tr = res["transition"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(s, res["full_loss"], color="#9aa4b8", lw=1.6, label="full loss (all freqs)")
    ax.plot(s, res["restricted_loss"], color="#4c8dff", lw=2.0,
            label="restricted loss (key freqs only)")
    ax.plot(s, res["excluded_loss"], color="#ff6b6b", lw=2.0,
            label="excluded loss (key freqs removed)")
    if tr.get("test_generalized_step"):
        ax.axvline(tr["test_generalized_step"], ls="--", color="#39d98a", alpha=0.8,
                   label=f"generalize @ {tr['test_generalized_step']}")
    if tr.get("train_saturated_step"):
        ax.axvline(tr["train_saturated_step"], ls=":", color="#c0c0c0", alpha=0.7,
                   label=f"memorize @ {tr['train_saturated_step']}")
    ax.set(xscale="log", yscale="log", xlabel="step", ylabel="cross-entropy loss",
           title=f"Progress measures (Nanda 2023) — {res['run_id']}\n"
                 f"key freqs k = {res['key_frequencies']}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Restricted/excluded loss progress measures")
    ap.add_argument("--config", default="grokfast", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--early-stop-acc", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    overrides: dict = {"seed": args.seed}
    if args.train_frac is not None:
        overrides["train_frac"] = args.train_frac
    if args.steps is not None:
        overrides["steps"] = args.steps
    cfg = get_config(args.config, **overrides)

    # Refuse a config that differs from the recorded run in the same directory:
    # run_id does not encode every field (e.g. steps), and a mismatched grid
    # silently produces measure curves on a different snapshot schedule.
    run_json = runs_dir() / cfg.run_id / "run.json"
    if run_json.exists():
        try:
            recorded_cfg = json.loads(run_json.read_text())["config"]
        except (json.JSONDecodeError, KeyError):
            print(f"[note] {run_json} unreadable — skipping config guard", flush=True)
            recorded_cfg = None
        if recorded_cfg is not None:
            ours = cfg.to_dict()
            diff = {k: (recorded_cfg.get(k), ours[k]) for k in ours
                    if recorded_cfg.get(k) != ours[k]}
            if diff:
                raise SystemExit(
                    f"config differs from recorded {run_json} in {diff}; "
                    f"pass matching flags (e.g. --steps {recorded_cfg.get('steps')})")

    print(f"[progress-measures] {cfg.run_id}  (re-training to capture states)", flush=True)
    res = compute(cfg, early_stop_acc=args.early_stop_acc)

    out_dir = runs_dir() / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "progress_measures.json").write_text(json.dumps(res, indent=2))
    fig_path = runs_dir().parents[0] / "figures" / f"{cfg.run_id}_progress.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot(res, fig_path)

    print(json.dumps({"transition": res["transition"],
                      "key_frequencies": res["key_frequencies"],
                      "sanity": res["sanity"]}, indent=2))
    print(f"[saved] {out_dir / 'progress_measures.json'}\n[saved] {fig_path}", flush=True)


if __name__ == "__main__":
    main()
