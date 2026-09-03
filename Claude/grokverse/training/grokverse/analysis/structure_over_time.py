"""H4 — does structure form before generalization? (INTERFACES §12, PREREGISTRATION §3 H4)

WHY THIS MODULE EXISTS
----------------------
Every other analysis module measures **one** checkpoint. H4 is the only hypothesis about *timing*:

> Mechanistic structure metrics rise already during the memorization plateau and predict the later
> generalization jump.

So this module walks **every** checkpoint of a v2 run, computes a compact metric set at each, and
reports two H4 statistics per metric: the change from `init` to `pre_generalization`, and the first
checkpoint at which the metric leaves its own initial noise band.

WHAT IT CAN AND CANNOT SHOW
----------------------------
This analysis is **correlational**. A metric rising before the generalization jump is consistent with
structure forming early; it cannot show that the structure *caused* the jump, and it cannot show that
it had to form. `analysis/causal_ablation.py` is where causal claims are tested, and it operates on
fixed checkpoints. `docs/CAUSAL_ABLATION_PLAN.md` §9 states this separation explicitly, and it is
repeated in the output so a reader of the JSON alone cannot miss it.

Khanh (arXiv:2607.06639) is the reason this module reports the whole trajectory rather than a single
"at the transition" number: representation metrics read at the grokking transition can differ from
their converged values, so both the crossing and the final checkpoint are reported, never one alone.

THE FRAGILITY OF THE "FIRST EXCEEDING" STATISTIC, STATED RATHER THAN HIDDEN
---------------------------------------------------------------------------
INTERFACES §12 defines the onset as the first checkpoint exceeding ``init + 3 * std`` where the std is
taken **over the first two checkpoints**. A standard deviation of two numbers is a very weak estimate,
so the resulting step is sensitive to the checkpoint grid and to noise at initialization. It is
implemented exactly as specified — no threshold is invented here — and the estimate's own inputs
(``init_value``, ``baseline_std``, ``n_baseline = 2``) are reported next to it so the fragility is
visible in the output. It is an onset *indicator*, not a measured transition.

STATUS: measurement code only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..checkpoints import assign_roles
from ..config import Config
from . import metrics as M
from .common import (envelope, load_model_at, masked_ce_and_acc, split_masks,
                     summarize as dist_summary, write_result)
from .fourier import dominant_frequencies
from .mlp_mechanism import (CURVE_NAMES, PRIMARY_DEFINITION, PRIMARY_KEY_RULE, curve_spectra,
                            effective_curves, fit_curve_matrix, neuron_tables, phase_relation,
                            resolve_key_frequencies, structured_neuron_definitions)
from . import key_frequencies as KF
from . import transformer_mechanism as TM

MODULE = "structure_over_time"
MODULE_VERSION = "1.0"
#: Onset rule of INTERFACES §12: init + Z_ONSET * std(first two checkpoints).
Z_ONSET = 3.0
#: Roles the H4 contrast is taken between (docs/dev/RUN_FORMAT_V2.md).
H4_FROM, H4_TO = "init", "pre_generalization"
#: Random draws for the logit key-subspace null in the time series (the measurement-point modules
#: use the full INTERFACES §0 default of 50; here the series is the object, not the null).
N_CONTROL_SERIES = 10


# --------------------------------------------------------------------------- #
# per-checkpoint metric set                                                    #
# --------------------------------------------------------------------------- #
def _opt_float(x) -> float | None:
    """``float(x)`` or ``None`` — ``common._jsonable`` rejects nan/inf, and a missing statistic
    (an empty structured set at `init`, say) must travel as JSON null, never as a made-up 0."""
    if x is None:
        return None
    f = float(x)
    return f if np.isfinite(f) else None


def _curves_and_logits(state: dict, cfg: Config) -> tuple[dict, np.ndarray, np.ndarray]:
    """Effective curves, the model's logits, and the per-neuron ``dead`` mask, per architecture."""
    p = cfg.p
    if cfg.arch == "transformer":
        abar = TM.attention_weights(state, cfg).reshape(p * p, cfg.n_heads, 3).mean(axis=0)
        curves = TM.effective_curves(state, cfg, abar)
        decomp = TM.forward_decomposition(state, cfg)
        hidden = decomp["hidden"]
        dead = hidden.reshape(p * p, -1).max(axis=0) <= 0        # the TRUE hidden layer
        return curves, decomp["logits"], dead
    curves = effective_curves(state, cfg)
    # max over the grid of (u_a[a] + u_b[b] + b_in) is max(u_a) + max(u_b) + b_in — exact, O(p n)
    dead = (curves["u_a"].max(axis=0) + curves["u_b"].max(axis=0) + curves["b_in"]) <= 0
    act = np.maximum(curves["u_a"][:, None, :] + curves["u_b"][None, :, :] + curves["b_in"], 0.0)
    logits = (act.reshape(p * p, -1) @ np.asarray(state["W_out"], dtype=np.float64)
              + np.asarray(state["b_out"], dtype=np.float64)).reshape(p, p, p)
    return curves, logits, dead


def checkpoint_metrics(state: dict, cfg: Config, key_freqs, seed: int, n_boot: int, n_perm: int,
                       n_control: int) -> tuple[dict, dict]:
    """The compact metric set of INTERFACES §12 at one checkpoint. Returns (scalars, per-neuron)."""
    p = cfg.p
    curves, logits, dead = _curves_and_logits(state, cfg)
    alive = ~dead
    tables = neuron_tables(curves, p)
    spec = {name: curve_spectra(curves[name], p) for name in CURVE_NAMES}
    defs = structured_neuron_definitions(tables, {"per_neuron": {"dead": dead}})
    primary = defs["definitions"][PRIMARY_DEFINITION]
    pr = phase_relation(spec["u_a"], spec["u_b"], spec["out"], primary["mask"],
                        int(n_boot), int(n_perm), int(seed))

    fits = {name: fit_curve_matrix(curves[name], tables["per_curve"][name]["dominant_frequency"],
                                   0, int(seed)) for name in ("u_a", "u_b")}
    best = np.concatenate([np.asarray(f["best_by_aic"]) for f in fits.values()])
    emb, emb_source = KF.embedding_matrix(state, cfg)
    dom = dominant_frequencies(np.asarray(emb, dtype=np.float64), p)
    emb_spec = M.power_spectrum(np.asarray(emb, dtype=np.float64), p)
    emb_power = emb_spec["power"].sum(axis=1, keepdims=True)

    train_mask, test_mask = split_masks(cfg)
    train_loss, train_acc = masked_ce_and_acc(logits, train_mask, p)
    test_loss, test_acc = masked_ce_and_acc(logits, test_mask, p)
    sub = TM.key_subspace_variance(logits, p, key_freqs, int(n_control), int(seed))
    sub.pop("_control_values", None)

    fam_a = np.asarray(tables["per_curve"]["u_a"]["family_fraction"], dtype=np.float64)
    fam_b = np.asarray(tables["per_curve"]["u_b"]["family_fraction"], dtype=np.float64)
    odd_minus_even = np.asarray(
        tables["per_curve"]["u_a"]["metrics"]["harmonic_shares"]["odd_minus_even"], dtype=np.float64)
    norms = {name: float(np.linalg.norm(np.asarray(v, dtype=np.float64)))
             for name, v in state.items()}

    scalars = {
        "embedding_top8_concentration": float(
            np.asarray(M.topk_concentration(emb_power, 8)).ravel()[0]),
        "embedding_threshold_count": int(dom["n_keep"]),
        "embedding_threshold_cap_binding": bool(dom["cap_binding"]),
        "embedding_object": emb_source,
        "structured_fraction_of_live": float(primary["fraction_of_live_neurons"]),
        "n_structured": int(primary["n"]),
        "n_live": int(alive.sum()),
        # At `init` the structured set is usually empty, and `phase_relation` then short-circuits
        # without a resultant length. That is a real state of the trajectory, not an error: it is
        # reported as null with `insufficient_neurons` set, never as a fabricated 0.
        "phase_relation_R": _opt_float(pr.get("resultant_length")),
        "phase_relation_null_q95": _opt_float(pr.get("null_resultant_q95")),
        "phase_relation_exceeds_null": bool(pr.get("exceeds_null_q95", False)),
        "phase_relation_insufficient_neurons": bool(pr.get("insufficient_neurons", False)),
        "phase_relation_n": int(pr.get("n", 0)),
        "fraction_best_aic_square": float((best == "square").mean()),
        "fraction_best_aic_sinusoid": float((best == "sinusoid").mean()),
        "median_odd_minus_even_u_a": float(np.median(odd_minus_even)),
        "median_family_fraction": float(np.median(np.concatenate([fam_a, fam_b]))),
        "logit_key_subspace_share": float(sub["share_total"]),
        "train_loss": train_loss, "test_loss": test_loss,
        "train_acc": train_acc, "test_acc": test_acc,
        "weight_norm_total": float(np.sqrt(sum(v ** 2 for v in norms.values()))),
    }
    scalars.update({f"weight_norm_{k}": v for k, v in norms.items()})
    per_neuron = {"family_fraction_u_a": fam_a, "family_fraction_u_b": fam_b,
                  "odd_minus_even_u_a": odd_minus_even,
                  "structured": np.asarray(primary["mask"], dtype=bool),
                  "alive": alive}
    return scalars, per_neuron


# --------------------------------------------------------------------------- #
# H4 statistics over the trajectory                                            #
# --------------------------------------------------------------------------- #
def onset_step(steps, values, z: float = Z_ONSET) -> dict:
    """First checkpoint exceeding ``init + z * std(first two checkpoints)`` (INTERFACES §12).

    Implemented exactly as specified. A std of two numbers is a very weak estimate, so the inputs
    are returned with the answer: the step is an onset *indicator*, not a measured transition.
    """
    v = np.asarray(values, dtype=np.float64)
    s = list(steps)
    if v.size < 3 or not np.isfinite(v[:2]).all():
        return {"step": None, "reason": "fewer than three finite checkpoints"}
    baseline_std = float(np.std(v[:2], ddof=1))
    threshold = float(v[0] + z * baseline_std)
    above = np.flatnonzero(np.isfinite(v) & (v > threshold))
    above = above[above >= 2]
    return {"step": (int(s[above[0]]) if above.size else None),
            "init_value": _opt_float(v[0]), "baseline_std": _opt_float(baseline_std),
            "n_baseline": 2, "threshold": _opt_float(threshold), "z": float(z),
            "caveat": ("std over two checkpoints is a very weak estimate; this is an onset "
                       "indicator, sensitive to the checkpoint grid, not a measured transition")}


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap CI of ``mean(b) - mean(a)`` over neurons (paired by neuron index)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return {"n": 0, "note": "neuron counts differ between the two checkpoints"}
    rng = np.random.default_rng(int(seed))
    diff = b - a
    draws = np.array([diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(int(n_boot))])
    return {"n": int(diff.size), "mean_difference": _opt_float(diff.mean()),
            "ci95_low": _opt_float(np.quantile(draws, 0.025)),
            "ci95_high": _opt_float(np.quantile(draws, 0.975)),
            "n_boot": int(n_boot), "paired_by": "neuron index"}


def h4_statistics(series: dict, roles: dict, per_neuron_at: dict, n_boot: int, seed: int) -> dict:
    """The two H4 statistics per metric: the init → pre_generalization change, and the onset step."""
    steps = series["step"]
    out: dict = {"from_role": H4_FROM, "to_role": H4_TO, "per_metric": {},
                 "roles_available": sorted(roles)}
    idx = {role: (steps.index(int(roles[role]["step"])) if role in roles
                  and int(roles[role]["step"]) in steps else None)
           for role in (H4_FROM, H4_TO)}
    out["role_steps"] = {r: (None if idx[r] is None else steps[idx[r]]) for r in idx}
    for name, values in series.items():
        if name == "step" or not all(isinstance(v, (int, float)) or v is None for v in values):
            continue
        v = np.array([np.nan if x is None else float(x) for x in values], dtype=np.float64)
        entry = {"onset": onset_step(steps, v)}
        if idx[H4_FROM] is not None and idx[H4_TO] is not None:
            entry["value_at_init"] = _opt_float(v[idx[H4_FROM]])
            entry["value_at_pre_generalization"] = _opt_float(v[idx[H4_TO]])
            entry["change"] = _opt_float(v[idx[H4_TO]] - v[idx[H4_FROM]])
        out["per_metric"][name] = entry
    if idx[H4_FROM] is not None and idx[H4_TO] is not None:
        a_step, b_step = steps[idx[H4_FROM]], steps[idx[H4_TO]]
        out["bootstrap_over_neurons"] = {
            key: _bootstrap_diff(per_neuron_at[a_step][key], per_neuron_at[b_step][key],
                                 n_boot, seed)
            for key in ("family_fraction_u_a", "family_fraction_u_b", "odd_minus_even_u_a")
            if a_step in per_neuron_at and b_step in per_neuron_at}
    out["reading"] = ("CORRELATIONAL. A metric rising before the generalization jump is consistent "
                      "with structure forming early; it cannot show that the structure caused the "
                      "jump. Causal claims come from analysis/causal_ablation.py only "
                      "(CAUSAL_ABLATION_PLAN 9).")
    return out


# --------------------------------------------------------------------------- #
# figure                                                                       #
# --------------------------------------------------------------------------- #
def _figure(series: dict, roles: dict, out_path: Path, run_id: str) -> Path | None:
    """Trajectory figure, drawn from the stored series only (master prompt §23 item 15)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                            # pragma: no cover - optional
        return None
    steps = np.array(series["step"], dtype=float)
    panels = [("test_acc", "train_acc"), ("structured_fraction_of_live", "phase_relation_R"),
              ("embedding_top8_concentration", "logit_key_subspace_share"),
              ("median_odd_minus_even_u_a", "fraction_best_aic_square")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 10), sharex=True)
    for ax, keys in zip(axes, panels):
        for key in keys:
            if key in series:
                v = np.array([np.nan if x is None else float(x) for x in series[key]])
                ax.plot(np.maximum(steps, 1), v, marker="o", ms=3, label=key)
        ax.set_xscale("log")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    for role, colour in (("memorization", "tab:orange"), ("generalization", "tab:green")):
        if role in roles:
            for ax in axes:
                ax.axvline(max(int(roles[role]["step"]), 1), color=colour, ls="--", lw=1, alpha=0.7)
    axes[-1].set_xlabel("step (log scale); dashed: memorization (orange), generalization (green)")
    axes[0].set_title(f"{run_id} — structure over time (H4)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# run-level analysis                                                           #
# --------------------------------------------------------------------------- #
def analyse(run_dir, key_rule: str = PRIMARY_KEY_RULE, seed: int = 0, n_boot: int = 2000,
            n_perm: int = 2000, n_control: int = N_CONTROL_SERIES,
            with_progress_measures: bool = True, figure: bool = True) -> tuple[dict, Path]:
    """Walk every checkpoint of a v2 run and report the H4 trajectory (INTERFACES §12).

    A per-run module: the driver calls it once per run and it iterates the checkpoints itself.
    """
    run_dir = Path(run_dir)
    ck = run_dir / "checkpoints.json"
    if not ck.exists():
        raise ValueError(f"{run_dir}: no checkpoints.json — structure_over_time needs a v2 run "
                         "(a legacy run has only model_final.pt)")
    entries = json.loads(ck.read_text())
    steps = [int(e["step"]) for e in entries]
    roles = assign_roles(run_dir)

    # The key set is fixed ONCE from the FINAL checkpoint and applied to every earlier one — the
    # same rule progress_measures.compute_from_checkpoints uses, and Nanda's own protocol. Selecting
    # it per checkpoint would let the metric chase a different subspace at every step.
    _cfg0, _m0, state_final, meta_final = load_model_at(run_dir, steps[-1])
    cfg = _cfg0
    key_freqs, key_info = resolve_key_frequencies(run_dir, steps[-1], key_rule, state_final, cfg)

    series: dict[str, list] = {"step": []}
    per_neuron_at: dict[int, dict] = {}
    for step in steps:
        _c, _m, state, _meta = load_model_at(run_dir, step)
        st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
        scalars, per_neuron = checkpoint_metrics(st, cfg, key_freqs, int(seed), int(n_boot),
                                                 int(n_perm), int(n_control))
        series["step"].append(int(step))
        for key, value in scalars.items():
            series.setdefault(key, []).append(value)
        per_neuron_at[int(step)] = per_neuron

    h4 = h4_statistics(series, roles, per_neuron_at, int(n_boot), int(seed))

    progress: dict = {"computed": False}
    if with_progress_measures:
        try:
            from .progress_measures import compute_from_checkpoints
            pm = compute_from_checkpoints(run_dir, key_rule=key_rule, write=True)
            progress = {"computed": True,
                        "written_to": "analysis/progress_measures/all_checkpoints.json",
                        "protocols": sorted(pm.get("protocols", []) if isinstance(pm, dict) else [])}
        except Exception as exc:                                 # recorded, never swallowed
            progress = {"computed": False, "error": f"{type(exc).__name__}: {exc}"}

    params = {
        "n_checkpoints": len(steps), "steps": steps, "seed": int(seed),
        "n_boot": int(n_boot), "n_perm": int(n_perm), "n_control": int(n_control),
        "key_frequency_selection": key_info,
        "key_frequencies": [int(k) for k in key_freqs],
        "key_frequency_rule_note": ("fixed once from the FINAL checkpoint and applied unchanged to "
                                    "every earlier one (Nanda's protocol; see "
                                    "progress_measures.compute_from_checkpoints)"),
        "structured_definition": PRIMARY_DEFINITION,
        "onset_rule": f"first checkpoint above init + {Z_ONSET} * std(first two checkpoints)",
        "output_path_deviation": ("INTERFACES §12 spells the output analysis/structure_over_time.json; "
                                  "it is written at analysis/structure_over_time/all_checkpoints.json "
                                  "to follow the §0 convention every other module uses"),
        "khanh_caveat": ("representation metrics read at the grokking transition can differ from "
                         "their converged values (arXiv:2607.06639), so the whole trajectory is "
                         "reported and no single point is called 'the' value"),
        "status": "MEASUREMENT ONLY — correlational; causal claims come from causal_ablation",
    }
    payload = envelope(MODULE, MODULE_VERSION, meta_final, params)
    payload["results"] = {
        "roles": {r: int(v["step"]) for r, v in roles.items()},
        "series": series,
        "h4": h4,
        "progress_measures": progress,
        "summary": {k: dist_summary(np.array([np.nan if x is None else x for x in v],
                                             dtype=np.float64))
                    for k, v in series.items()
                    if k != "step" and all(isinstance(x, (int, float)) or x is None for x in v)},
    }
    arrays: dict[str, np.ndarray] = {}
    for step, pn in per_neuron_at.items():
        for key, value in pn.items():
            arrays[f"step{step:06d}__{key}"] = value
    path = write_result(run_dir, MODULE, "all_checkpoints", payload, arrays)
    if figure:
        fig_path = _figure(series, roles, path.parent / "structure_over_time.png",
                           payload["run_id"])
        if fig_path is not None:
            payload["results"]["figure"] = str(fig_path.relative_to(run_dir))
            path = write_result(run_dir, MODULE, "all_checkpoints", payload, arrays)
    return payload, path


def main() -> None:
    ap = argparse.ArgumentParser(description="H4 structure-over-time trajectory (INTERFACES §12)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--key-rule", default=PRIMARY_KEY_RULE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress-measures", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    payload, path = analyse(args.run_dir, args.key_rule, args.seed,
                            with_progress_measures=not args.no_progress_measures,
                            figure=not args.no_figure)
    h4 = payload["results"]["h4"]
    sf = h4["per_metric"].get("structured_fraction_of_live", {})
    print(f"{payload['run_id']}  checkpoints={payload['params']['n_checkpoints']}  "
          f"structured_fraction: init={sf.get('value_at_init')} "
          f"pre_gen={sf.get('value_at_pre_generalization')} "
          f"onset_step={sf.get('onset', {}).get('step')}  -> {path}")


if __name__ == "__main__":
    main()
