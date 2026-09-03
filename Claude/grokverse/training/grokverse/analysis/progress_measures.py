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

TWO PATHS, both in this module:

1. ``compute(cfg)`` — the ORIGINAL re-train path, unchanged. It re-trains
   deterministically, measures the ``IMPLEMENTED`` variants on the FULL grid, and
   fills the frozen explorer's top-level ``restricted_loss``/``excluded_loss``
   keys from ``TOP_LEVEL_PROTOCOL``. Legacy runs use this.
2. ``compute_from_checkpoints(run_dir)`` — the run-format-v2 path (INTERFACES
   §10). NO re-training: it reads ``checkpoints.json``, fixes the key set once
   from the FINAL checkpoint, and measures every implemented protocol *and* the
   five named functions of ``docs/MASK_PROTOCOL_AUDIT.md`` §4 on every
   checkpoint, reporting each loss on all three splits (``test``, ``train``,
   ``all``) with the split labelled. Writes
   ``analysis/progress_measures/all_checkpoints.json``.

Usage (from training/) — pass the SAME --steps as the recorded run:
    python -m grokverse.analysis.progress_measures --config grokfast --train-frac 0.5 --steps 8000 --seed 0 --early-stop-acc 0.98
    python -m grokverse.analysis.progress_measures --config nanda --steps 40000 --seed 0 --early-stop-acc 0.95   # canonical (slow)
    python -m grokverse.analysis.progress_measures --from-checkpoints runs/<run_id>   # v2 runs, no re-training
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ..checkpoints import list_checkpoints  # noqa: E402
from ..config import PRESETS, Config, get_config  # noqa: E402
from ..data import make_dataset  # noqa: E402
from ..models import build_model  # noqa: E402
from ..seed import set_seed  # noqa: E402
from ..train import apply_grokfast, detect_transition  # noqa: E402
from ..utils import config_diff, log_step_schedule, runs_dir  # noqa: E402
from .common import (envelope, grid_logits, load_model_at,  # noqa: E402
                     masked_ce_and_acc, split_masks, write_result)
from .fourier import dominant_frequencies, fourier_basis  # noqa: E402
from .mask_protocols import (ALL_IMPLEMENTED, IMPLEMENTED, SPLITS,  # noqa: E402
                             build_protocol, full_grid_extension_excluded_loss,
                             full_grid_extension_restricted_loss,
                             legacy_broad_mask_variant, nanda_exact_excluded_loss,
                             nanda_exact_restricted_loss, per_frequency_shares)


# --------------------------------------------------------------------------- #
# deterministic re-train that keeps a CPU copy of the model state every logged  #
# step (train.py only persists per-step embeddings, not full weights)          #
# --------------------------------------------------------------------------- #
def train_capturing_states(cfg: Config, early_stop_acc: float | None = None):
    torch.set_num_threads(cfg.threads)   # must match train.py exactly (§3.8)
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


#: Which variant fills the top-level ``restricted_loss``/``excluded_loss`` keys.
#: Pinned to the legacy variant because ``export.py`` ships those keys to the
#: frontend and the explorer is frozen (RESEARCH_SPEC §11) — changing the
#: meaning of an existing key silently would rewrite published numbers. Every
#: variant is stored alongside under ``variants``, and the file records which
#: protocol the top-level keys came from, so nothing is ambiguous.
TOP_LEVEL_PROTOCOL = "legacy_broad_mask"


def _mode_indices(p: int, key_freqs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over the p Fourier-basis rows.

    SUPERSEDED by ``analysis.mask_protocols`` (RESEARCH_SPEC §3.1): the outer
    product built from these 1D masks keeps cross-frequency blocks the
    trig-identity circuit never uses. Kept because ``mask_protocols`` reproduces
    it as the named ``legacy_broad_mask`` variant and the tests pin both.

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
            max_steps_measured: int = 120,
            protocols: tuple[str, ...] = IMPLEMENTED) -> dict:
    model, data, curves, states = train_capturing_states(cfg, early_stop_acc)
    p = cfg.p
    Fb, _ = fourier_basis(p)
    all_x, all_y = data["all_x"], data["all_y"]

    # Key frequencies fixed from the FINAL embedding (Nanda: define once, apply to all steps)
    final_W_E = model.W_E.detach().cpu().numpy()
    dom = dominant_frequencies(final_W_E, p)
    key_freqs = dom["dominant"]
    # All mask variants side by side (§3.1): the width of the mask is a stated
    # choice, not an accident of an outer product.
    protos = {name: build_protocol(name, p, key_freqs) for name in protocols}
    if TOP_LEVEL_PROTOCOL not in protos:
        raise ValueError(f"protocols must include {TOP_LEVEL_PROTOCOL!r} "
                         "(it fills the top-level keys the explorer reads)")

    steps = curves["step"]
    # subsample if a long un-accelerated run produced many checkpoints
    idx = (np.linspace(0, len(steps) - 1, max_steps_measured).round().astype(int)
           if len(steps) > max_steps_measured else np.arange(len(steps)))
    idx = np.unique(idx)

    base = build_model(cfg)
    full_loss, m_steps = [], []
    variants: dict[str, dict[str, list]] = {
        name: {"restricted_loss": [], "excluded_loss": []} for name in protos}
    for t in idx:
        base.load_state_dict(states[t])
        base.eval()
        L = _grid_logits(base, all_x, p)
        Lhat = fwd2d(L, Fb)
        full_loss.append(_ce(L, all_y))
        for name, pr in protos.items():
            variants[name]["restricted_loss"].append(_ce(inv2d(pr.restrict(Lhat), Fb), all_y))
            variants[name]["excluded_loss"].append(_ce(inv2d(pr.exclude(Lhat), Fb), all_y))
        m_steps.append(steps[t])
    restr_loss = variants[TOP_LEVEL_PROTOCOL]["restricted_loss"]
    excl_loss = variants[TOP_LEVEL_PROTOCOL]["excluded_loss"]

    # ---- sanity checks (honesty: a wrong measure is worse than none) ----
    base.load_state_dict(states[-1])
    base.eval()
    L = _grid_logits(base, all_x, p)
    Lhat = fwd2d(L, Fb)
    recon_err = float(np.abs(inv2d(Lhat, Fb) - L).max())          # (1) F is a valid orthonormal transform
    all_modes = inv2d(Lhat * np.ones((p, p, 1)), Fb)
    all_modes_err = float(np.abs(all_modes - L).max())            # (2) keeping all modes == identity
    full_final = _ce(L, all_y)
    restr_final = _ce(inv2d(protos[TOP_LEVEL_PROTOCOL].restrict(Lhat), Fb), all_y)
    excl_final = _ce(inv2d(protos[TOP_LEVEL_PROTOCOL].exclude(Lhat), Fb), all_y)
    sanity = {
        "reconstruction_max_abs_err": recon_err,
        "all_modes_max_abs_err": all_modes_err,
        "n_key_freqs": len(key_freqs),
        "final_full_loss": full_final,
        "final_restricted_loss": restr_final,   # expect ~ full_loss (circuit IS the key freqs)
        "final_excluded_loss": excl_final,      # expect >> full_loss (circuit destroyed)
        "restricted_recovers_full": bool(restr_final < full_final + 0.10),
        "excluded_destroys_solution": bool(excl_final > full_final + 1.0),
        # the key-frequency COUNT is capped, not measured (§3.2)
        "key_freq_cap_binding": dom["cap_binding"],
        "n_freqs_for_90pct_threshold": dom["n_freqs_for_threshold"],
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
        # Top-level keys are the LEGACY variant (see TOP_LEVEL_PROTOCOL): kept
        # for the frozen explorer's data contract. They are NOT a reproduction
        # of Nanda et al. — see variants[] and docs/MASK_PROTOCOL_AUDIT.md.
        "top_level_protocol": TOP_LEVEL_PROTOCOL,
        "restricted_loss": restr_loss,
        "excluded_loss": excl_loss,
        "variants": {name: {**variants[name], **pr.describe()}
                     for name, pr in protos.items()},
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



# --------------------------------------------------------------------------- #
# Checkpoint path (INTERFACES §10): every protocol on every saved checkpoint,   #
# NO re-training. The re-train path above is untouched — legacy runs use it.    #
# --------------------------------------------------------------------------- #
MODULE = "progress_measures"
MODULE_VERSION = "2.0"

#: Output tag: ``<run_dir>/analysis/progress_measures/all_checkpoints.json``.
FROM_CHECKPOINTS_TAG = "all_checkpoints"

#: Pre-registered primary key-frequency rule (INTERFACES §4, PREREGISTRATION §4.3).
#: NOT the legacy top-8 — the audit's §1 row 7 records that the top-8 cap binds on
#: every legacy run, so that count was never data-determined.
DEFAULT_KEY_RULE = "nanda"

#: Used only when ``analysis.key_frequencies`` cannot be imported. Recorded in the
#: output as ``key_selection.fallback_used`` with the reason, never silently.
FALLBACK_KEY_RULE = "embedding_top8"


def _select_key_frequencies(run_dir, step, key_rule: str, state: dict, p: int) -> tuple[list[int], dict]:
    """Key set fixed ONCE from the final checkpoint (Nanda's protocol, audit §1 row 7).

    Delegates to ``analysis.key_frequencies.select`` (INTERFACES §4) when that module
    exists. If it cannot be imported the legacy ``embedding_top8`` rule is used and the
    result records ``fallback_used=True`` together with the import error and the fact
    that the top-8 **cap** is a fixed number, not a measurement.
    """
    try:
        from . import key_frequencies  # noqa: WPS433 — optional module (INTERFACES §4)
    except Exception as exc:          # noqa: BLE001 — an unimportable module is "absent"
        reason = f"{type(exc).__name__}: {exc}"
        key_frequencies = None
    else:
        reason = None

    if key_frequencies is not None:
        sel = key_frequencies.select(run_dir, step, key_rule)
        ks = [int(k) for k in sel["key_frequencies"]]
        info = {k: v for k, v in sel.items() if not isinstance(v, np.ndarray)}
        return ks, {**info, "key_rule": key_rule, "key_frequencies": ks,
                    "source": "analysis.key_frequencies.select", "fallback_used": False,
                    "fixed_from_step": step}

    if "W_E" not in state:
        raise ValueError(
            f"{run_dir}: analysis.key_frequencies is unavailable ({reason}) and this model "
            "has no W_E, so the embedding_top8 fallback is undefined. Install/await "
            "analysis.key_frequencies (INTERFACES §4) or pass an explicit key set.")
    dom = dominant_frequencies(np.asarray(state["W_E"], dtype=np.float64)[:p], p)
    ks = [int(k) for k in dom["dominant"]]
    return ks, {
        "key_rule": FALLBACK_KEY_RULE,
        "key_frequencies": ks,
        "source": "analysis.fourier.dominant_frequencies on W_E[:p]",
        "fallback_used": True,
        "fallback_reason": f"analysis.key_frequencies not importable ({reason})",
        "requested_key_rule": key_rule,
        "fixed_from_step": step,
        "threshold": dom["threshold"], "max_k": dom["max_k"], "n_keep": dom["n_keep"],
        "n_freqs_for_threshold": dom["n_freqs_for_threshold"],
        "cap_binding": dom["cap_binding"],
        "dominant_fraction": dom["dominant_fraction"],
        "note": ("embedding_top8 caps the count at 8; audit §1 row 7 records that the cap "
                 "binds on every legacy run, so this count is FIXED, not measured."),
    }


def _split_masks_dict(cfg: Config) -> dict[str, np.ndarray]:
    train_mask, test_mask = split_masks(cfg)
    return {"test": test_mask, "train": train_mask,
            "all": np.ones((cfg.p, cfg.p), dtype=bool)}


def _eval_all_splits(L: np.ndarray, masks: dict[str, np.ndarray], p: int) -> dict:
    out = {}
    for name, m in masks.items():
        ce, acc = masked_ce_and_acc(L, m, p)
        out[name] = {"loss": ce, "accuracy": acc, "n_cells": int(m.sum())}
    return out


def compute_from_checkpoints(run_dir, key_rule: str = DEFAULT_KEY_RULE,
                             protocols: tuple[str, ...] = ALL_IMPLEMENTED,
                             per_frequency: bool = True,
                             write: bool = True) -> dict:
    """Every implemented protocol on EVERY checkpoint of a v2 run — no re-training.

    Differences from ``compute`` (which stays exactly as it was, for legacy runs):

    * the states come from ``checkpoints.json``, not from a deterministic re-train,
      so nothing depends on reproducing a training grid;
    * the key set is fixed ONCE from the FINAL checkpoint by ``key_rule``
      (``analysis.key_frequencies.select``, else the ``embedding_top8`` fallback with
      a recorded note) and applied unchanged to every earlier checkpoint — Nanda's
      protocol (audit §1 row 7; Colab ``get_metrics`` reuses the global ``key_freqs``);
    * every loss is reported on **all three splits** (``test``, ``train``, ``all``)
      with the split labelled, so no number travels without its convention;
    * the five named functions of MASK_PROTOCOL_AUDIT §4 are called directly, so the
      published operator, our full-grid extension and the legacy mask are measured by
      the same code path a caller would use.

    Writes ``<run_dir>/analysis/progress_measures/all_checkpoints.json`` through
    ``common.write_result``. Returns the payload.
    """
    run_dir = Path(run_dir)
    unknown = [n for n in protocols if n not in ALL_IMPLEMENTED]
    if unknown:
        raise ValueError(f"unknown protocol(s) {unknown}; choices: {list(ALL_IMPLEMENTED)}")

    entries = list_checkpoints(run_dir)
    if not entries:
        raise ValueError(f"{run_dir}/checkpoints.json is empty — nothing to measure")
    final_step = int(entries[-1]["step"])

    cfg, _model, final_state, final_meta = load_model_at(run_dir, final_step)
    p = cfg.p
    if cfg.task != "add":
        raise ValueError(
            f"{final_meta['run_id']}: task {cfg.task!r} — the split evaluation labels cells "
            "(a+b) mod p (common.masked_ce_and_acc); only modular addition is defined here")
    key_freqs, key_info = _select_key_frequencies(run_dir, final_step, key_rule, final_state, p)

    Fb, _ = fourier_basis(p)
    masks = _split_masks_dict(cfg)
    # Protocols whose numbers do NOT come from a named §4 function.
    table_only = [n for n in protocols if n not in ("legacy_broad_mask", "nanda_exact")]

    checkpoints: list[dict] = []
    for entry in entries:
        step = int(entry["step"])
        cfg_i, model_i, _state_i, meta_i = load_model_at(run_dir, step)
        L = grid_logits(model_i, cfg_i)
        Lhat = fwd2d(L, Fb)

        # --- the five named functions of MASK_PROTOCOL_AUDIT §4 ---
        functions = {
            "nanda_exact_restricted_loss":
                nanda_exact_restricted_loss(L, key_freqs, cfg_i),
            "nanda_exact_excluded_loss":
                nanda_exact_excluded_loss(L, key_freqs, cfg_i, per_frequency=per_frequency),
            "full_grid_extension_restricted_loss":
                full_grid_extension_restricted_loss(L, key_freqs, cfg_i),
            "full_grid_extension_excluded_loss":
                full_grid_extension_excluded_loss(L, key_freqs, cfg_i),
            "legacy_broad_mask_variant_restricted":
                legacy_broad_mask_variant(L, key_freqs, cfg_i, which="restricted"),
            "legacy_broad_mask_variant_excluded":
                legacy_broad_mask_variant(L, key_freqs, cfg_i, which="excluded"),
        }

        # --- the protocol table (same splits, one row per operator) ---
        table: dict[str, dict] = {}
        if "legacy_broad_mask" in protocols:
            table["legacy_broad_mask"] = {
                "restricted": functions["legacy_broad_mask_variant_restricted"]["splits"],
                "excluded": functions["legacy_broad_mask_variant_excluded"]["splits"],
                **build_protocol("legacy_broad_mask", p, key_freqs).describe(),
            }
        if "nanda_exact" in protocols:
            table["nanda_exact"] = {
                "restricted": functions["nanda_exact_restricted_loss"]["splits"],
                "excluded": functions["nanda_exact_excluded_loss"]["splits"],
                **build_protocol("nanda_exact", p, key_freqs, split="all").describe(),
            }
        for name in table_only:
            pr = build_protocol(name, p, key_freqs)
            table[name] = {
                "restricted": _eval_all_splits(inv2d(pr.restrict(Lhat), Fb), masks, p),
                "excluded": _eval_all_splits(inv2d(pr.exclude(Lhat), Fb), masks, p),
                **pr.describe(),
            }

        checkpoints.append({
            "step": step,
            "kind": entry.get("kind"),
            "checkpoint_file": entry.get("path"),
            "checkpoint_sha256": entry.get("sha256"),
            "full_loss": _eval_all_splits(L, masks, p),
            "protocols": table,
            "functions": functions,
            "per_frequency_shares": per_frequency_shares(Lhat, key_freqs),
        })
        print(f"[from-checkpoints] step {step:>6d}  "
              f"full(all)={checkpoints[-1]['full_loss']['all']['loss']:.5f}", flush=True)

    params = {
        "key_rule": key_rule,
        "key_frequencies": key_freqs,
        "key_selection": key_info,
        "key_frequencies_fixed_from": {"step": final_step, "role": "final checkpoint"},
        "protocols": list(protocols),
        "per_frequency": bool(per_frequency),
        "splits_reported": list(SPLITS),
        "n_checkpoints": len(entries),
        "retrained": False,
        "loss": "cross-entropy, float64, on the model's own train/test split "
                "(common.masked_ce_and_acc)",
        "audit_reference": "docs/MASK_PROTOCOL_AUDIT.md §4; docs/dev/INTERFACES.md §10",
    }
    meta_all = {**final_meta, "step": FROM_CHECKPOINTS_TAG,
                "checkpoint_file": None, "checkpoint_sha256": None}
    payload = envelope(MODULE, MODULE_VERSION, meta_all, params)
    payload["results"] = {
        "steps": [c["step"] for c in checkpoints],
        "checkpoints": checkpoints,
        "final_step": final_step,
        "top_level_protocol_note": (
            "This file has NO top-level restricted_loss/excluded_loss keys. Every number "
            "is under checkpoints[].protocols[<name>][restricted|excluded][<split>] or "
            "checkpoints[].functions[<function name>], always with its protocol and split. "
            f"The frozen explorer's keys come from {TOP_LEVEL_PROTOCOL!r} via compute()."),
    }
    if write:
        path = write_result(run_dir, MODULE, FROM_CHECKPOINTS_TAG, payload)
        payload["output_path"] = str(path)
        print(f"[saved] {path}", flush=True)
    return payload


def from_checkpoints_summary(payload: dict, step: int | None = None) -> dict:
    """Compact legacy-vs-exact contrast at one checkpoint (default: the last one)."""
    cps = payload["results"]["checkpoints"]
    cp = cps[-1] if step is None else next(c for c in cps if c["step"] == int(step))
    fn = cp["functions"]
    return {
        "run_id": payload["run_id"],
        "step": cp["step"],
        "key_frequencies": payload["params"]["key_frequencies"],
        "key_rule": payload["params"]["key_selection"]["key_rule"],
        "key_rule_fallback_used": payload["params"]["key_selection"]["fallback_used"],
        "full_loss": {s: cp["full_loss"][s]["loss"] for s in SPLITS},
        "legacy_broad_mask": {
            "restricted": {s: fn["legacy_broad_mask_variant_restricted"]["splits"][s]["loss"]
                           for s in SPLITS},
            "excluded": {s: fn["legacy_broad_mask_variant_excluded"]["splits"][s]["loss"]
                         for s in SPLITS},
            "n_kept": fn["legacy_broad_mask_variant_restricted"]["n_components_kept_by_restricted"],
            "n_removed": fn["legacy_broad_mask_variant_excluded"]["n_components_removed_by_excluded"],
            "is_reproduction": False,
        },
        "nanda_exact": {
            "restricted": {s: fn["nanda_exact_restricted_loss"]["splits"][s]["loss"]
                           for s in SPLITS},
            "restricted_quoted_split": fn["nanda_exact_restricted_loss"]["quoted_split"],
            "restricted_paper_split": fn["nanda_exact_restricted_loss"]["paper_split"],
            "excluded": {s: fn["nanda_exact_excluded_loss"]["splits"][s]["loss"]
                         for s in SPLITS},
            "excluded_split": fn["nanda_exact_excluded_loss"]["split"],
            "n_kept": fn["nanda_exact_restricted_loss"]["n_components_kept_by_restricted"],
            "n_removed": fn["nanda_exact_excluded_loss"]["n_components_removed_by_excluded"],
        },
        "sum_direction_share_of_nonconstant_power": cp["per_frequency_shares"]["sum_share_total"],
        "diff_direction_share_of_nonconstant_power": cp["per_frequency_shares"]["diff_share_total"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Restricted/excluded loss progress measures")
    ap.add_argument("--config", default="grokfast", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--early-stop-acc", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--protocols", nargs="+", default=list(IMPLEMENTED),
                    choices=list(IMPLEMENTED),
                    help="mask variants to compute side by side (§3.1); re-train path only")
    ap.add_argument("--from-checkpoints", type=Path, default=None, metavar="RUN_DIR",
                    help="measure every protocol on every checkpoint of a run-format-v2 "
                         "directory instead of re-training (INTERFACES §10)")
    ap.add_argument("--key-rule", default=DEFAULT_KEY_RULE,
                    help="key-frequency rule for --from-checkpoints (INTERFACES §4; "
                         f"primary: {DEFAULT_KEY_RULE})")
    ap.add_argument("--no-per-frequency", action="store_true",
                    help="skip the per-key excluded-loss variant (Colab excl_loss / Fig. 15)")
    ap.add_argument("--from-checkpoints-protocols", nargs="+", default=list(ALL_IMPLEMENTED),
                    choices=list(ALL_IMPLEMENTED),
                    help="protocols for --from-checkpoints (default: all implemented)")
    args = ap.parse_args()

    # --- checkpoint path: no re-training, nothing below this block runs ---
    if args.from_checkpoints is not None:
        payload = compute_from_checkpoints(
            args.from_checkpoints, key_rule=args.key_rule,
            protocols=tuple(args.from_checkpoints_protocols),
            per_frequency=not args.no_per_frequency)
        print(json.dumps(from_checkpoints_summary(payload), indent=2))
        print("[note] legacy_broad_mask is NOT a reproduction of Nanda et al. "
              "(docs/MASK_PROTOCOL_AUDIT.md §3); the restricted split of the paper's own "
              "figure is [NOT FOUND IN SOURCE], so all three splits are reported.",
              flush=True)
        return

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
            diff, added = config_diff(recorded_cfg, cfg.to_dict())
            if added:
                print(f"[note] {run_json} predates config field(s) {added} "
                      "— not treated as a mismatch", flush=True)
            if diff:
                raise SystemExit(
                    f"config differs from recorded {run_json} in {diff}; "
                    f"pass matching flags (e.g. --steps {recorded_cfg.get('steps')})")

    print(f"[progress-measures] {cfg.run_id}  (re-training to capture states)", flush=True)
    res = compute(cfg, early_stop_acc=args.early_stop_acc,
                  protocols=tuple(args.protocols))

    out_dir = runs_dir() / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "progress_measures.json").write_text(json.dumps(res, indent=2))
    fig_path = runs_dir().parents[0] / "figures" / f"{cfg.run_id}_progress.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot(res, fig_path)

    variant_summary = {
        name: {"n_kept": v["n_components_kept_by_restricted"],
               "n_removed": v["n_components_removed_by_excluded"],
               "cross_frequency": v["keeps_cross_frequency_blocks"],
               "final_restricted": round(v["restricted_loss"][-1], 5),
               "final_excluded": round(v["excluded_loss"][-1], 5)}
        for name, v in res["variants"].items()}
    print(json.dumps({"transition": res["transition"],
                      "key_frequencies": res["key_frequencies"],
                      "final_full_loss": round(res["full_loss"][-1], 5),
                      "variants": variant_summary,
                      "sanity": res["sanity"]}, indent=2))
    print("[note] top-level restricted/excluded keys use "
          f"{res['top_level_protocol']!r}, which is NOT a reproduction of "
          "Nanda et al. — compare the variants above (RESEARCH_SPEC §3.1).",
          flush=True)
    print(f"[saved] {out_dir / 'progress_measures.json'}\n[saved] {fig_path}", flush=True)


if __name__ == "__main__":
    main()
