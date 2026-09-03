"""Bounded alternative-mechanism analysis (INTERFACES §13, PREREGISTRATION §6.3).

WHEN THIS RUNS
--------------
Only for an architecture that **fails** the evidence gate. Master prompt §12 is explicit that ending
with "may use something else" is not acceptable: if the Fourier tests do not identify the mechanism,
the study owes one bounded level of further investigation. This module is that level, and it is
pre-committed — it is not a search that continues until something is found.

WHAT IS BOUNDED ABOUT IT
-------------------------
Four questions, fixed in advance by PREREGISTRATION §6.3, and no fifth:

1. **Is `(a+b) mod p` decodable from the hidden layer at all?** A linear and a one-hidden-layer probe,
   trained on the model's *own* train split and scored on its *own* test split, each against a
   **shuffled-label control** trained identically. Without that control a probe's accuracy says
   nothing: a rich enough representation plus enough parameters fits noise.
2. **Is there compact structure?** Singular spectrum and effective rank of `hidden` and of the logits.
3. **Does it recur across seeds?** Linear CKA between the hidden representations of paired seeds.
4. **Is it causal?** Removing the top-`r` singular directions versus removing `r` random directions —
   the same size-matched logic every other ablation in this study uses.

**The project does not promise to find an alternative algorithm.** Master prompt §12: "A well-founded
negative result remains valid if the bounded analysis identifies none." Nothing here is tuned until
something appears.

THE KHANH CAVEAT (arXiv:2607.06639)
------------------------------------
Representation metrics read *at* the grokking transition can differ from their converged values, so
every quantity here is computed at whichever checkpoint the caller names and the checkpoint is recorded
in the envelope. A rank or CKA number without its checkpoint is not interpretable, and this module
never reports one.

STATUS: measurement code only. It decides nothing; `decision_tree` decides whether it should run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from .common import envelope, load_model_at, split_masks, summarize as dist_summary, write_result
from . import transformer_mechanism as TM
from .mlp_mechanism import effective_curves

MODULE = "bounded_alternative"
MODULE_VERSION = "1.0"
#: Probe training budget — fixed in advance, identical for the real and the shuffled-label run.
PROBE_STEPS = 400
PROBE_LR = 0.01
PROBE_HIDDEN = 128
PROBE_WEIGHT_DECAY = 0.0
#: Size-matched random draws for the singular-direction ablation.
N_CONTROL = 20
#: Ranks removed in the causal test.
REMOVE_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# the model's hidden representation                                            #
# --------------------------------------------------------------------------- #
def hidden_and_logits(state: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """``hidden [p*p, n]`` and ``logits [p, p, p]`` for either architecture."""
    p = cfg.p
    if cfg.arch == "transformer":
        d = TM.forward_decomposition(state, cfg)
        return d["hidden"].reshape(p * p, -1), d["logits"]
    curves = effective_curves(state, cfg)
    act = np.maximum(curves["u_a"][:, None, :] + curves["u_b"][None, :, :] + curves["b_in"], 0.0)
    flat = act.reshape(p * p, -1)
    logits = (flat @ np.asarray(state["W_out"], dtype=np.float64)
              + np.asarray(state["b_out"], dtype=np.float64)).reshape(p, p, p)
    return flat, logits


# --------------------------------------------------------------------------- #
# 1. probes                                                                    #
# --------------------------------------------------------------------------- #
def _train_probe(X: np.ndarray, y: np.ndarray, train_idx, test_idx, hidden: int | None,
                 seed: int, steps: int = PROBE_STEPS, lr: float = PROBE_LR) -> dict:
    """Train one probe with a fixed budget and report its train/test accuracy.

    ``hidden=None`` is the linear probe; an integer gives one hidden ReLU layer of that width. The
    budget, learning rate and seed are identical for the real and the shuffled-label run, so the two
    differ in nothing but the labels.
    """
    torch.manual_seed(int(seed))
    Xt = torch.tensor(X, dtype=torch.float32)
    Xt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-6)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    n_classes = int(yt.max().item()) + 1
    layers = ([torch.nn.Linear(Xt.shape[1], n_classes)] if hidden is None
              else [torch.nn.Linear(Xt.shape[1], hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, n_classes)])
    model = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=PROBE_WEIGHT_DECAY)
    tr = torch.tensor(np.asarray(train_idx), dtype=torch.long)
    te = torch.tensor(np.asarray(test_idx), dtype=torch.long)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = lossf(model(Xt[tr]), yt[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(Xt).argmax(1)
        return {"train_acc": float((pred[tr] == yt[tr]).float().mean()),
                "test_acc": float((pred[te] == yt[te]).float().mean()),
                "final_train_loss": float(loss.item()),
                "n_params": int(sum(prm.numel() for prm in model.parameters())),
                "hidden_width": hidden, "steps": int(steps), "lr": float(lr), "seed": int(seed)}


def probe_battery(hidden: np.ndarray, cfg: Config, seed: int = 0,
                  steps: int = PROBE_STEPS) -> dict:
    """Linear and one-hidden-layer probes for `(a+b) mod p`, each with a shuffled-label control."""
    p = cfg.p
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    y = (a + b) % p
    train_mask, test_mask = split_masks(cfg)
    train_idx = np.flatnonzero(train_mask.ravel())
    test_idx = np.flatnonzero(test_mask.ravel())
    rng = np.random.default_rng(int(seed))
    y_shuffled = rng.permutation(y)

    out: dict = {"chance_level": 1.0 / p, "n_train": int(train_idx.size),
                 "n_test": int(test_idx.size),
                 "protocol": ("trained on the model's OWN train split and scored on its OWN test "
                              "split; the shuffled-label control uses the same architecture, budget, "
                              "learning rate and seed and differs only in the labels")}
    for name, width in (("linear", None), ("one_hidden_layer", PROBE_HIDDEN)):
        real = _train_probe(hidden, y, train_idx, test_idx, width, seed, steps)
        ctrl = _train_probe(hidden, y_shuffled, train_idx, test_idx, width, seed, steps)
        out[name] = {"real": real, "shuffled_label_control": ctrl,
                     "test_acc_above_control": real["test_acc"] - ctrl["test_acc"],
                     "test_acc_above_chance": real["test_acc"] - 1.0 / p}
    out["reading"] = ("a probe accuracy is only interpretable against its shuffled-label control; a "
                      "control that also scores well means the probe is fitting the split, not the "
                      "representation")
    return out


# --------------------------------------------------------------------------- #
# 2. spectra                                                                   #
# --------------------------------------------------------------------------- #
def spectrum(X: np.ndarray, name: str) -> dict:
    """Singular spectrum, participation ratio and two effective-rank definitions, both named."""
    A = np.asarray(X, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    total = float((s ** 2).sum())
    if total <= 0:
        return {"object": name, "n_singular_values": int(s.size), "degenerate": True}
    q = (s ** 2) / total
    nz = q[q > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    cum = np.cumsum(q)
    return {
        "object": name,
        "n_singular_values": int(s.size),
        "effective_rank_entropy": float(np.exp(entropy)),
        "participation_ratio": float(1.0 / (q ** 2).sum()),
        "n_components_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "n_components_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "top1_share": float(q[0]), "top8_share": float(q[:8].sum()),
        "singular_values": dist_summary(s),
        "definitions": {"effective_rank_entropy": "exp(-sum q log q) on squared singular values",
                        "participation_ratio": "1 / sum q^2"},
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representations of the same inputs (Kornblith et al. 2019)."""
    A = np.asarray(X, dtype=np.float64)
    B = np.asarray(Y, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    hsic = float(np.linalg.norm(B.T @ A, "fro") ** 2)
    na = float(np.linalg.norm(A.T @ A, "fro"))
    nb = float(np.linalg.norm(B.T @ B, "fro"))
    return float(hsic / (na * nb)) if na > 0 and nb > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 3. the causal test on singular directions                                    #
# --------------------------------------------------------------------------- #
def singular_direction_ablation(hidden: np.ndarray, readout: np.ndarray, bias, cfg: Config,
                                ranks=REMOVE_RANKS, n_control: int = N_CONTROL,
                                seed: int = 0) -> dict:
    """Remove the top-`r` singular directions of `hidden` versus `r` random directions.

    The readout is applied to the projected activation, so this asks whether the model's *output*
    depends on those directions — not merely whether the representation changes.
    """
    from .causal_ablation import evaluate, report

    p = cfg.p
    A = np.asarray(hidden, dtype=np.float64)
    mean = A.mean(axis=0, keepdims=True)
    C = A - mean
    _u, _s, Vt = np.linalg.svd(C, full_matrices=False)
    n_dim = Vt.shape[0]

    def logits_from(H: np.ndarray) -> np.ndarray:
        out = H @ np.asarray(readout, dtype=np.float64)
        if bias is not None:
            out = out + np.asarray(bias, dtype=np.float64).reshape(1, -1)[:, :p]
        return out[:, :p].reshape(p, p, p)

    base = evaluate(logits_from(A), cfg)
    rng = np.random.default_rng(int(seed))
    out: dict = {"n_dimensions": int(n_dim), "ranks": [int(r) for r in ranks],
                 "baseline": {k: v for k, v in base.items() if k != "logits"}, "per_rank": {}}
    for r in ranks:
        if r > n_dim:
            continue
        V = Vt[:r]
        obs = report(base, evaluate(logits_from(mean + C - (C @ V.T) @ V), cfg), cfg, int(r))
        ctrl = []
        for _ in range(int(n_control)):
            Q, _ = np.linalg.qr(rng.standard_normal((n_dim, r)))
            ctrl.append(report(base, evaluate(logits_from(mean + C - (C @ Q) @ Q.T), cfg), cfg,
                               int(r)))
        drops = np.array([c["test_accuracy_drop"] for c in ctrl])
        std = float(drops.std(ddof=1)) if drops.size > 1 else 0.0
        out["per_rank"][str(int(r))] = {
            "observed": {k: v for k, v in obs.items() if not k.startswith("_")},
            "control": {"n": int(drops.size), "mean_drop": float(drops.mean()), "std_drop": std,
                        "max_drop": float(drops.max()) if drops.size else None},
            "z": (float((obs["test_accuracy_drop"] - drops.mean()) / std) if std > 0 else None),
            "exceeds_all_controls": (bool(obs["test_accuracy_drop"] > drops.max())
                                     if drops.size else None),
        }
    out["control_definition"] = (f"{n_control} random rank-r orthonormal subspaces of the same "
                                 "dimension, projected out of the centred hidden layer")
    return out


# --------------------------------------------------------------------------- #
# run-level analysis                                                           #
# --------------------------------------------------------------------------- #
def analyse(run_dir, step: int | None = None, seed: int = 0, n_control: int = N_CONTROL,
            probe_steps: int = PROBE_STEPS, key_rule: str | None = None) -> tuple[dict, Path]:
    """The bounded follow-up on one checkpoint (PREREGISTRATION §6.3). ``key_rule`` is accepted and
    unused, so the driver may call this module with the same signature as the others."""
    run_dir = Path(run_dir)
    cfg, _model, state, meta = load_model_at(run_dir, step)
    st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    hidden, logits = hidden_and_logits(st, cfg)

    if cfg.arch == "transformer":
        readout = st["W_out"] @ st["W_U"][:, :cfg.p]
        direct = TM.forward_decomposition(st, cfg)["direct_path_logits"].reshape(cfg.p * cfg.p, -1)
        bias = None
        abl = singular_direction_ablation(hidden, readout, None, cfg, REMOVE_RANKS,
                                          int(n_control), int(seed))
        abl["note"] = ("the transformer's direct (attention-only) path is not included in this "
                       "reconstruction, so the baseline here is the MLP-path logits alone")
    else:
        readout = st["W_out"]
        bias = st["b_out"]
        abl = singular_direction_ablation(hidden, readout, bias, cfg, REMOVE_RANKS,
                                          int(n_control), int(seed))

    params = {"step": meta["step"], "seed": int(seed), "n_control": int(n_control),
              "probe_steps": int(probe_steps), "probe_hidden_width": PROBE_HIDDEN,
              "probe_lr": PROBE_LR, "ranks_removed": [int(r) for r in REMOVE_RANKS],
              "khanh_caveat": ("representation metrics read at the grokking transition can differ "
                               "from their converged values (arXiv:2607.06639); every number here "
                               "belongs to the checkpoint named in this envelope and to no other"),
              "scope": ("bounded by PREREGISTRATION §6.3: four pre-committed questions and no fifth; "
                        "the project does not promise to find an alternative algorithm, and a "
                        "well-founded negative result is a valid outcome"),
              "when_this_runs": ("only for an architecture that FAILED the evidence gate; "
                                 "analysis/decision_tree.py decides that, not this module"),
              "status": "MEASUREMENT ONLY"}
    payload = envelope(MODULE, MODULE_VERSION, meta, params)
    payload["results"] = {
        "probes": probe_battery(hidden, cfg, int(seed), int(probe_steps)),
        "spectrum_hidden": spectrum(hidden, "hidden"),
        "spectrum_logits": spectrum(logits.reshape(cfg.p * cfg.p, cfg.p), "logits"),
        "singular_direction_ablation": abl,
        "n_hidden_units": int(hidden.shape[1]),
    }
    path = write_result(run_dir, MODULE, meta["tag"], payload, None)
    return payload, path


def compare_seeds(run_dirs, step: int | None = None, max_units: int = 512) -> dict:
    """Linear CKA between the hidden representations of several runs at the same checkpoint."""
    reps, ids = [], []
    for run_dir in run_dirs:
        cfg, _m, state, meta = load_model_at(Path(run_dir), step)
        st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
        hidden, _ = hidden_and_logits(st, cfg)
        reps.append(hidden[:, :max_units])
        ids.append({"run_id": meta["run_id"], "arch": meta["arch"], "step": meta["step"]})
    n = len(reps)
    matrix = [[(1.0 if i == j else linear_cka(reps[i], reps[j])) for j in range(n)]
              for i in range(n)]
    off = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
    return {"runs": ids, "cka_matrix": matrix,
            "off_diagonal": dist_summary(np.array(off)) if off else {"n": 0},
            "definition": "linear CKA (Kornblith et al. 2019) on the centred hidden representations",
            "note": ("computed at one checkpoint for all runs; a CKA number without its checkpoint "
                     "is not interpretable (Khanh caveat)")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded alternative-mechanism analysis (§13, §6.3)")
    ap.add_argument("run_dir", type=Path, nargs="+")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--cka", action="store_true", help="only the cross-seed CKA over the given runs")
    args = ap.parse_args()
    if args.cka:
        print(json.dumps(compare_seeds(args.run_dir, args.step)["off_diagonal"], indent=2))
        return
    for run_dir in args.run_dir:
        payload, path = analyse(run_dir, args.step, args.seed, args.n_control)
        r = payload["results"]
        lin = r["probes"]["linear"]
        print(f"{payload['run_id']} step={payload['step']}  "
              f"linear probe test={lin['real']['test_acc']:.3f} "
              f"(control {lin['shuffled_label_control']['test_acc']:.3f}, "
              f"chance {r['probes']['chance_level']:.3f})  "
              f"eff.rank(hidden)={r['spectrum_hidden']['effective_rank_entropy']:.1f}  -> {path}")


if __name__ == "__main__":
    main()
