"""Parameter-count and weight-norm audit for docs/CAPACITY_AND_CONFOUNDS.md.

Master prompt §14 ("Parameter and capacity confounds") / docs/RESEARCH_SPEC.md §3.6.

Computes, from the actual model classes and the stored run artifacts:

* total / trainable parameter count and per-module counts for the transformer,
  the shared-embedding MLP (d_mlp=512), the parameter-matched MLP (d_mlp=572)
  and the two-hot MLP;
* the initial L2 norm per module and in total after ``set_seed(seed)`` +
  ``build_model(cfg)`` for seeds 0, 1, 2, next to the analytic expectation
  ``sqrt(numel) * init_std`` transcribed from ``models/*.py``;
* for every run directory under ``training/runs/`` the final per-module and
  total L2 norms from ``model_final.pt``, the ratio final/initial, and a check
  that the stored step-0 ``embeddings.npy`` slice equals the current-code
  initial ``W_E`` (i.e. that the init code has not drifted between commits);
* the AdamW decoupled per-step decay factor ``1 - lr * weight_decay``.

Usage (cwd = training/)::

    python -m grokverse.analysis.capacity_report            # writes JSON, prints tables
    python -m grokverse.analysis.capacity_report --tables   # prints the Markdown tables only

Output: ``docs/data/capacity_report.json``. AI-drafted (Claude), 2026-09-02 —
not yet human-reviewed.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..config import Config, get_config
from ..models import build_model
from ..seed import set_seed

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "training" / "runs"
OUT = REPO / "docs" / "data" / "capacity_report.json"
LR, WD = 1e-3, 1.0  # Config defaults; asserted against get_config() below
INIT_SEEDS = (0, 1, 2)

#: Verbatim quotes fetched 2026-09-02 (see docs/CAPACITY_AND_CONFOUNDS.md §9).
SOURCES = {
    "omnigrok": {
        "id": "arXiv:2210.01117v2 (Liu, Michaud, Tegmark, 'Omnigrok: Grokking Beyond Algorithmic Data')",
        "fetched_from": ["https://arxiv.org/abs/2210.01117", "https://ar5iv.labs.arxiv.org/html/2210.01117"],
        "fetched_on": "2026-09-02",
        "quotes": {
            "sec2_heading": "2 The LU mechanism for grokking",
            "sec2_reduced_landscape": (
                "Letting w denote the weights of a model, any function f(w) (e.g, train/test loss/accuracy) "
                "depends on both the weight norm w≡‖w‖₂ and the angular direction ŵ≡w/w. Similar to "
                "[Fort and Scherlis 2019], we define a reduced function f̃(w) by minimizing training loss "
                "l_train(w) over angular directions"),
            "sec2_goldilocks": (
                "There is a spherical shell in the weight space (the \"Goldilocks\" zone), where generalization "
                "is better than outside this zone. We illustrate the Goldilocks zone as the green area with "
                "average radius w_c in Figure 1(a)"),
            "sec2_weight_decay_time": (
                "Fortunately, there are usually explicit and/or implicit regularizations that can drive the "
                "weight vector towards the Goldilocks zone w≈w_c. When the regularization magnitude is non-zero "
                "but small, the radial motion can be (arbitrarily) slow. If weight decay is the only source of "
                "regularization, and training loss is negligible after overfitting, then weight decay γ causes "
                "w(t)≈exp(−γt)w₀, when w₀>w_c, so it takes time t≈ln(w₀/w_c)/γ∝γ⁻¹ to generalize."),
            "sec2_alpha": "our model is initialized by multiplying a factor α≡w/w₀ to the standard initialization",
            "sec2_constrained": (
                "In practice, we perform the constrained minimization by rescaling the model weights back to "
                "their original norm after each unconstrained optimization step."),
            "sec3_alpha_reg": (
                "large initialization (α=2.0) can demonstrate no generalization (no reg), grokking (small reg) "
                "and fast generalization (large reg)."),
            "sec5_1_constant_norm": (
                "constraining optimization to hold model weight norm constant over training brings train "
                "accuracy and test accuracy learning curves together, almost eliminating grokking"),
        },
        "not_found": "No sentence relating parameter count / width / model size to grokking was found in the "
                     "fetched text [NOT FOUND IN SOURCE].",
    },
    "pytorch_adamw": {
        "id": "torch.optim.AdamW documentation (docs.pytorch.org, stable → 2.13)",
        "fetched_from": ["https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html"],
        "fetched_on": "2026-09-02",
        "quotes": {
            "decay_step": "θt←θt−1−γλθt−1",
            "description": "Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance.",
            "weight_decay_param": "weight_decay (float, optional) – weight decay coefficient (default: 1e-2)",
            "reference": "For further details regarding the algorithm we refer to Decoupled Weight Decay Regularization",
        },
    },
}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception as exc:  # noqa: BLE001 — provenance only, never fatal
        return f"unknown ({exc})"


def _norms(state: dict) -> tuple[dict, float]:
    per, sumsq = {}, 0.0
    for k, v in state.items():
        v = v.detach().double()
        n2 = float((v * v).sum())
        per[k] = {"numel": int(v.numel()), "shape": list(v.shape), "l2": math.sqrt(n2)}
        sumsq += n2
    return per, math.sqrt(sumsq)


def init_std_table(cfg: Config) -> dict[str, float]:
    """Analytic init std per module, transcribed from ``models/*.py::_init_weights``."""
    s, d, dm, dh, h, p = cfg.init_scale, cfg.d_model, cfg.d_mlp, cfg.d_head, cfg.n_heads, cfg.p
    if cfg.arch == "transformer":
        return {"W_E": s / math.sqrt(d), "W_pos": s / math.sqrt(d), "W_Q": s / math.sqrt(d),
                "W_K": s / math.sqrt(d), "W_V": s / math.sqrt(d), "W_O": s / math.sqrt(dh * h),
                "W_in": s / math.sqrt(d), "W_out": s / math.sqrt(dm), "W_U": s / math.sqrt(d)}
    if cfg.arch == "mlp":
        return {"W_E": s / math.sqrt(d), "W_in": s / math.sqrt(2 * d), "W_out": s / math.sqrt(dm),
                "b_in": 0.0, "b_out": 0.0}
    if cfg.arch == "mlp_twohot":
        return {"W_in": s / math.sqrt(2 * p), "W_out": s / math.sqrt(dm), "b_in": 0.0, "b_out": 0.0}
    raise ValueError(cfg.arch)


def describe_architecture(cfg: Config, seeds: tuple[int, ...] = INIT_SEEDS) -> dict:
    params = list(build_model(cfg).named_parameters())
    stds = init_std_table(cfg)
    modules = {n: {"shape": list(p.shape), "numel": int(p.numel()), "requires_grad": bool(p.requires_grad),
                   "init_std": stds[n], "expected_l2_at_init": math.sqrt(p.numel()) * stds[n]}
               for n, p in params}
    seed_norms = {}
    for sd in seeds:
        set_seed(sd)
        per, tot = _norms(dict(build_model(cfg).named_parameters()))
        seed_norms[str(sd)] = {"per_module": {k: v["l2"] for k, v in per.items()}, "total": tot}
    return {"config": cfg.to_dict(),
            "n_params_formula": cfg.n_params,
            "n_params_built": sum(p.numel() for _, p in params),
            "n_trainable_built": sum(p.numel() for _, p in params if p.requires_grad),
            "modules": modules,
            "expected_total_l2_at_init": math.sqrt(sum(m["expected_l2_at_init"] ** 2 for m in modules.values())),
            "init_norms_by_seed": seed_norms}


def describe_run(run_dir: Path) -> dict:
    rj = json.loads((run_dir / "run.json").read_text())
    c = {k: v for k, v in rj["config"].items() if k not in ("vocab_size", "run_id")}
    cfg = Config(**c)  # fields added after the run (threads, study, ...) take their defaults
    state = torch.load(run_dir / "model_final.pt", map_location="cpu")
    fin_per, fin_tot = _norms(state)
    set_seed(cfg.seed)
    m0 = build_model(cfg)
    ini_per, ini_tot = _norms(dict(m0.named_parameters()))
    emb0 = np.asarray(np.load(run_dir / "embeddings.npy", mmap_mode="r")[0], dtype=np.float64)
    we_now = m0.W_E.detach().double().numpy()
    last_step = int(rj["logged_steps"][-1])
    per_module = {k: {"numel": fin_per[k]["numel"], "init_l2": ini_per[k]["l2"], "final_l2": fin_per[k]["l2"],
                      "ratio_final_over_init": (fin_per[k]["l2"] / ini_per[k]["l2"]) if ini_per[k]["l2"] > 0 else None}
                  for k in fin_per}
    return {
        "run_id": rj["config"]["run_id"], "arch": cfg.arch, "d_mlp": cfg.d_mlp, "seed": cfg.seed,
        "task": cfg.task, "train_frac": cfg.train_frac, "grokfast": cfg.grokfast,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay,
        "steps_budget": cfg.steps, "last_logged_step": last_step,
        "run_git_commit": rj.get("git_commit"), "torch_num_threads_recorded": rj.get("torch_num_threads"),
        "created_utc": rj.get("created_utc"), "transition": rj.get("transition"),
        "n_params_state_dict": sum(v["numel"] for v in fin_per.values()),
        "state_dict_keys": list(state.keys()),
        "per_module": per_module,
        "total_init_l2": ini_tot, "total_final_l2": fin_tot, "total_ratio_final_over_init": fin_tot / ini_tot,
        "pure_decay_factor_over_last_logged_step": (1 - cfg.lr * cfg.weight_decay) ** last_step,
        "init_check": {"embeddings_npy_step0_W_E_l2": float(np.sqrt((emb0 ** 2).sum())),
                       "current_code_init_W_E_l2": ini_per["W_E"]["l2"],
                       "max_abs_elementwise_diff": float(np.abs(emb0 - we_now).max()) if emb0.shape == we_now.shape else None,
                       "emb0_shape": list(emb0.shape), "W_E_shape": list(we_now.shape)},
    }


def build_report() -> dict:
    torch.set_num_threads(1)
    base = get_config("nanda")
    assert (base.lr, base.weight_decay) == (LR, WD), "Config defaults changed; update LR/WD"
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "training/grokverse/analysis/capacity_report.py (python -m grokverse.analysis.capacity_report); "
                     "AI-drafted (Claude), 2026-09-02 — not yet human-reviewed",
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "torch_num_threads": torch.get_num_threads(),
                        "git_head": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                        "git_dirty_files": _git("status", "--short").splitlines()},
        "optimizer": {"class": "torch.optim.AdamW", "lr": LR, "weight_decay": WD, "betas": [base.beta1, base.beta2],
                      "lr_times_wd": LR * WD, "per_step_decay_factor": 1.0 - LR * WD,
                      "applies_to": "every tensor in model.parameters() — a single param group, biases included",
                      "source": "training/grokverse/train.py: torch.optim.AdamW(model.parameters(), lr=cfg.lr, "
                                "weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))"},
        "architectures": {},
        "runs": [],
        "sources": SOURCES,
    }
    cfgs = {
        "transformer_d512": get_config("nanda", arch="transformer"),
        "mlp_d512": get_config("nanda", arch="mlp"),
        "mlp_param_matched_d572": get_config("mlp_param_matched"),
        "mlp_twohot_d512": get_config("arch25k_twohot"),
    }
    for name, cfg in cfgs.items():
        try:
            report["architectures"][name] = describe_architecture(cfg)
        except Exception as exc:  # noqa: BLE001 — two-hot factory entry is uncommitted
            report["architectures"][name] = {"config": cfg.to_dict(), "n_params_formula": cfg.n_params,
                                             "build_error": repr(exc)}
    # Only the legacy (pre-study) runs are in scope: every run.json whose config
    # has no ``study`` tag (the field did not exist when they were recorded) or
    # an empty one. Study/smoke-test runs are listed under ``excluded_runs`` so
    # the exclusion is visible, never silent.
    report["excluded_runs"] = []
    for rd in sorted(RUNS.iterdir()):
        if not ((rd / "run.json").exists() and (rd / "model_final.pt").exists()):
            continue
        cfg_rec = json.loads((rd / "run.json").read_text())["config"]
        if cfg_rec.get("study", ""):
            report["excluded_runs"].append({"run_id": cfg_rec.get("run_id", rd.name),
                                            "study": cfg_rec["study"], "steps": cfg_rec.get("steps"),
                                            "reason": "study/smoke-test run, not one of the 16 legacy runs"})
            continue
        report["runs"].append(describe_run(rd))
    return report


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #

def _f(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def render_tables(report: dict) -> str:
    out = []
    archs = report["architectures"]
    out.append("### T1 — parameter counts\n")
    out.append("| architecture | preset / override | total (built) | trainable | `Config.n_params` formula | vs transformer |")
    out.append("|---|---|---:|---:|---:|---:|")
    t = archs["transformer_d512"]["n_params_built"]
    labels = {"transformer_d512": "`nanda`, `arch=transformer`", "mlp_d512": "`nanda`, `arch=mlp`",
              "mlp_param_matched_d572": "`mlp_param_matched` (`arch=mlp`, `d_mlp=572`)",
              "mlp_twohot_d512": "`arch25k_twohot` (`arch=mlp_twohot`)"}
    for name, a in archs.items():
        if "n_params_built" not in a:
            out.append(f"| {name} | {labels[name]} | build failed | – | {a['n_params_formula']:,} | – |")
            continue
        n = a["n_params_built"]
        out.append(f"| {name} | {labels[name]} | {n:,} | {a['n_trainable_built']:,} | {a['n_params_formula']:,} "
                   f"| {n - t:+,} ({100 * (n - t) / t:+.3f} %) |")
    for name, a in archs.items():
        if "modules" not in a:
            continue
        out.append(f"\n### T2-{name} — per-module counts and analytic initial norm\n")
        out.append("| parameter | shape | numel | init std (`_init_weights`) | expected ‖·‖₂ at init = √numel·std |")
        out.append("|---|---|---:|---:|---:|")
        for k, m in a["modules"].items():
            out.append(f"| `{k}` | {m['shape']} | {m['numel']:,} | {m['init_std']:.6f} | {m['expected_l2_at_init']:.4f} |")
        out.append(f"| **total** | | **{a['n_params_built']:,}** | | **{a['expected_total_l2_at_init']:.4f}** |")
    out.append("\n### T3 — measured initial L2 norms after `set_seed(seed)` + `build_model(cfg)` (threads=1)\n")
    for name, a in archs.items():
        if "init_norms_by_seed" not in a:
            continue
        mods = list(a["modules"].keys())
        out.append(f"\n**{name}** (expected total {a['expected_total_l2_at_init']:.4f})\n")
        out.append("| seed | " + " | ".join(f"`{m}`" for m in mods) + " | **total** |")
        out.append("|---|" + "---:|" * (len(mods) + 1))
        for sd, v in a["init_norms_by_seed"].items():
            out.append(f"| {sd} | " + " | ".join(_f(v["per_module"][m]) for m in mods) + f" | **{v['total']:.4f}** |")
    out.append("\n### T4 — the 16 existing runs: total L2 norm, initial vs final\n")
    out.append("| # | run_id | run commit | threads | steps budget | last logged step | n_params (state_dict) | ‖θ₀‖₂ | ‖θ_final‖₂ | final/init | (1−lr·wd)^last | init check max|Δ| |")
    out.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(report["runs"], 1):
        thr = r["torch_num_threads_recorded"]
        out.append(f"| {i} | `{r['run_id']}` | `{(r['run_git_commit'] or '')[:7]}` | {thr if thr is not None else '–'} "
                   f"| {r['steps_budget']:,} | {r['last_logged_step']:,} | {r['n_params_state_dict']:,} "
                   f"| {r['total_init_l2']:.4f} | {r['total_final_l2']:.4f} | {r['total_ratio_final_over_init']:.4f} "
                   f"| {r['pure_decay_factor_over_last_logged_step']:.2e} | {r['init_check']['max_abs_elementwise_diff']} |")
    for arch in ("transformer", "mlp"):
        rs = [r for r in report["runs"] if r["arch"] == arch]
        if not rs:
            continue
        mods = list(rs[0]["per_module"].keys())
        out.append(f"\n### T5-{arch} — per-module final L2 norm (final/init ratio in parentheses; '–' = init norm 0)\n")
        out.append("| # | run_id | " + " | ".join(f"`{m}`" for m in mods) + " |")
        out.append("|---:|---|" + "---:|" * len(mods))
        for r in rs:
            i = report["runs"].index(r) + 1
            cells = []
            for m in mods:
                pm = r["per_module"][m]
                ratio = pm["ratio_final_over_init"]
                cells.append(f"{pm['final_l2']:.2f} ({'–' if ratio is None else f'{ratio:.2f}'})")
            out.append(f"| {i} | `{r['run_id']}` | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    report = build_report()
    if "--tables" not in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {OUT}", file=sys.stderr)
    sys.stdout.reconfigure(encoding="utf-8")
    print(render_tables(report))


if __name__ == "__main__":
    main(sys.argv[1:])
