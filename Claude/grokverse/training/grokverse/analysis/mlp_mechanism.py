"""Per-neuron mechanism of the 2-layer MLP (docs/RESEARCH_SPEC.md §3.3, WP-2).

WHY THIS MODULE EXISTS
----------------------
Every structural claim in the repo so far rests on the shared embedding matrix
``W_E`` alone. For the MLP that is the wrong object. ``models/mlp.py``
concatenates the two operand embeddings *before* ``W_in``, so what hidden neuron
``i`` actually sees is::

    u_a[a, i] = W_E[a] . W_in[:d_model, i]        # effective a-curve of neuron i
    u_b[b, i] = W_E[b] . W_in[d_model:, i]        # effective b-curve of neuron i
    pre[a, b, i] = u_a[a, i] + u_b[b, i] + b_in[i]
    act[a, b, i] = ReLU(pre[a, b, i])
    out[c, i]    = W_out[i, c]                    # effective output curve of neuron i

``W_E`` is a *shared* table that both operand paths read through different halves
of ``W_in``. A per-neuron circuit can be perfectly periodic while the raw ``W_E``
spectrum looks diffuse — ``W_in`` is free to select and rotate. Measuring only
``W_E`` and concluding "distributed solution" is a non-sequitur.

Note the structural fact this makes visible: **the MLP is exactly additive in
(a, b) up to the ReLU.** No path lets a and b interact before the nonlinearity,
so in this architecture the ReLU is provably the only possible source of the
multiplication in the trig identity — which makes the MLP the cleaner test case
of the mechanism, not the messier one.

WHAT H1 PREDICTS
----------------
Writing each curve as ``A cos(w_k n - phi)``, rectifying a sum of two same-
frequency sinusoids produces a cross term in (a+b) with phase ``phi_a + phi_b``.
For the readout to peak at ``c = a + b`` the output curve must carry that same
phase, giving the per-neuron, falsifiable prediction::

    phi_out ~= phi_a + phi_b   (mod 2*pi)

``phase_relation`` measures it against a permutation null (neuron identities
shuffled), which is H1's stated null hypothesis.

STATUS: measurement code only. It reports distributions; it does not decide
whether H1 holds — and the "structured neuron" thresholds remain a [HUMAN]
decision (§9 item 6), so nothing here silently blesses a threshold.

Usage (from training/):
    python -m grokverse.analysis.mlp_mechanism runs/mlp_add_p113_wd1.0_frac0.3_seed0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..models import build_model
from .fourier import fourier_basis


# --------------------------------------------------------------------------- #
# "structured neuron" criteria — thresholds are a [HUMAN] decision (§5, §9.6)  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StructuredNeuronCriteria:
    """A neuron counts as *structured* iff

    1. its a-curve and b-curve share a dominant frequency k,
    2. that frequency explains >= ``min_variance_explained`` of each curve's power,
    3. its output curve's dominant frequency is also k.

    The thresholds are NOT settled science. §9 item 6 reserves them for the
    human authors, and §5 makes a sensitivity analysis over >= 2 alternative
    thresholds mandatory. ``PROPOSED`` below is the spec's proposal awaiting
    approval — it is deliberately not a module-level default that could quietly
    become the answer.
    """

    min_variance_explained: float
    require_same_output_frequency: bool = True
    label: str = "unnamed"


#: The §5 proposal, pending approval in docs/HUMAN_DECISIONS.md.
PROPOSED = StructuredNeuronCriteria(0.50, True, "proposed_0.50")
#: Mandatory sensitivity companions (§5). Never report PROPOSED alone.
SENSITIVITY = (
    StructuredNeuronCriteria(0.30, True, "sensitivity_0.30"),
    StructuredNeuronCriteria(0.70, True, "sensitivity_0.70"),
)


# --------------------------------------------------------------------------- #
# effective per-neuron curves                                                  #
# --------------------------------------------------------------------------- #
def effective_curves(state: dict, cfg: Config) -> dict[str, np.ndarray]:
    """The three curves that define a neuron's circuit, all shaped [p, d_mlp].

    Derived from the actual forward pass of ``models/mlp.py`` (verified against
    a hand-computed reference in test_core.py), not from a redrawn diagram.
    """
    if cfg.arch != "mlp":
        raise ValueError(f"effective_curves is MLP-specific; got arch={cfg.arch!r}")
    p, d = cfg.p, cfg.d_model
    W_E = np.asarray(state["W_E"], dtype=np.float64)[:p]      # [p, d]
    W_in = np.asarray(state["W_in"], dtype=np.float64)        # [2d, d_mlp]
    W_out = np.asarray(state["W_out"], dtype=np.float64)      # [d_mlp, p]
    b_in = np.asarray(state["b_in"], dtype=np.float64)        # [d_mlp]
    return {
        "u_a": W_E @ W_in[:d],        # [p, d_mlp]
        "u_b": W_E @ W_in[d:],        # [p, d_mlp]
        "out": W_out.T,               # [p, d_mlp] — class axis first, to match
        "b_in": b_in,                 # [d_mlp]
    }


def curve_spectra(curves: np.ndarray, p: int) -> dict:
    """Per-column Fourier power, phase, and dominant frequency.

    ``curves`` is [p, n]. Phase convention: the column is written as
    ``A cos(w_k n - phi)``, so ``phi = atan2(coeff_sin, coeff_cos)`` — valid
    because the cos_k and sin_k basis rows carry identical normalization.
    """
    F, _ = fourier_basis(p)
    coeff = F @ np.asarray(curves, dtype=np.float64)          # [p, n]
    half = (p - 1) // 2
    cos_rows = 1 + 2 * (np.arange(1, half + 1) - 1)
    sin_rows = 2 + 2 * (np.arange(1, half + 1) - 1)
    c, s = coeff[cos_rows], coeff[sin_rows]                   # [half, n]
    power = c ** 2 + s ** 2
    phase = np.arctan2(s, c)
    total = power.sum(axis=0)
    dom_idx = power.argmax(axis=0)
    n = power.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        dom_frac = np.where(total > 0, power[dom_idx, np.arange(n)] / total, 0.0)
    return {
        "power": power,                                       # [half, n]
        "phase": phase,                                       # [half, n]
        "freqs": np.arange(1, half + 1),
        "dominant_freq": dom_idx + 1,                         # [n]
        "dominant_fraction": dom_frac,                        # [n]
        "dominant_phase": phase[dom_idx, np.arange(n)],       # [n]
        "total_power": total,                                 # [n]
    }


def classify_neurons(spec_a: dict, spec_b: dict, spec_out: dict,
                     crit: StructuredNeuronCriteria) -> dict:
    """Apply one criteria set. Returns boolean masks and the counts."""
    same_ab = spec_a["dominant_freq"] == spec_b["dominant_freq"]
    strong = ((spec_a["dominant_fraction"] >= crit.min_variance_explained)
              & (spec_b["dominant_fraction"] >= crit.min_variance_explained))
    structured = same_ab & strong
    if crit.require_same_output_frequency:
        structured = structured & (spec_out["dominant_freq"] == spec_a["dominant_freq"])
    return {
        "criteria": dataclasses.asdict(crit),
        "n_neurons": int(structured.size),
        "n_same_ab_frequency": int(same_ab.sum()),
        "n_above_variance_threshold": int(strong.sum()),
        "n_structured": int(structured.sum()),
        "fraction_structured": float(structured.mean()),
        "mask": structured,
    }


# --------------------------------------------------------------------------- #
# H1: phi_out ~= phi_a + phi_b, against a permutation null                     #
# --------------------------------------------------------------------------- #
def _wrap(x: np.ndarray) -> np.ndarray:
    """Wrap angles into (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(x)))


def _resultant_length(err: np.ndarray) -> float:
    """Mean resultant length R in [0, 1] — the circular concentration statistic.

    R ~ 1 means the phase errors pile up at one angle (H1's prediction);
    R ~ 0 means they are spread like the uniform null.
    """
    return float(np.abs(np.exp(1j * np.asarray(err)).mean())) if err.size else 0.0


def phase_relation(spec_a: dict, spec_b: dict, spec_out: dict,
                   mask: np.ndarray, n_boot: int = 2000, n_perm: int = 2000,
                   seed: int = 0) -> dict:
    """Circular test of ``phi_out - (phi_a + phi_b)`` on the selected neurons.

    The null shuffles neuron identity between the operand side and the output
    side, destroying the per-neuron correspondence while leaving both marginal
    phase distributions intact — H1_0 as stated in §4.
    """
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return {"n": int(idx.size), "insufficient_neurons": True}
    pa = spec_a["dominant_phase"][idx]
    pb = spec_b["dominant_phase"][idx]
    po = spec_out["dominant_phase"][idx]
    err = _wrap(po - (pa + pb))
    R = _resultant_length(err)

    rng = np.random.default_rng(seed)
    boot = np.array([_resultant_length(err[rng.integers(0, err.size, err.size)])
                     for _ in range(int(n_boot))])
    null = np.array([_resultant_length(_wrap(po[rng.permutation(idx.size)] - (pa + pb)))
                     for _ in range(int(n_perm))])
    return {
        "n": int(idx.size),
        "insufficient_neurons": False,
        "circular_mean_error": float(np.angle(np.exp(1j * err).mean())),
        "median_abs_error": float(np.median(np.abs(err))),
        "resultant_length": R,
        "resultant_ci95": [float(np.quantile(boot, 0.025)),
                           float(np.quantile(boot, 0.975))],
        "null_resultant_mean": float(null.mean()),
        "null_resultant_q95": float(np.quantile(null, 0.95)),
        # H1 is falsified if the observed R sits inside the null's bulk
        "exceeds_null_q95": bool(R > np.quantile(null, 0.95)),
        "n_boot": int(n_boot), "n_perm": int(n_perm), "seed": int(seed),
    }


# --------------------------------------------------------------------------- #
# is the activation a function of (a + b)?                                     #
# --------------------------------------------------------------------------- #
def sum_dependence(curves: dict, p: int) -> dict:
    """Fraction of each neuron's activation variance explained by (a+b) mod p.

    A neuron implementing the circuit is (to first order) a function of a+b
    alone, so ``Var(E[act | a+b]) / Var(act)`` should be near 1. Computed
    exactly — every residue class of a+b holds exactly p of the p^2 pairs — and
    reported next to the same quantity for (a-b), which the addition circuit has
    no use for and which therefore acts as a built-in control.
    """
    u_a, u_b, b_in = curves["u_a"], curves["u_b"], curves["b_in"]
    act = np.maximum(u_a[:, None, :] + u_b[None, :, :] + b_in, 0.0)   # [p, p, n]
    a_idx = np.arange(p)[:, None]
    b_idx = np.arange(p)[None, :]

    def explained(group: np.ndarray) -> np.ndarray:
        n = act.shape[-1]
        sums = np.zeros((p, n))
        np.add.at(sums, group.ravel(), act.reshape(-1, n))
        means = sums / p                                    # each class has p pairs
        grand = act.reshape(-1, n).mean(axis=0)
        between = ((means - grand) ** 2).sum(axis=0) * p
        total = ((act.reshape(-1, n) - grand) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total > 0, between / total, 0.0)

    ev_sum = explained((a_idx + b_idx) % p)
    ev_diff = explained((a_idx - b_idx) % p)
    return {
        "variance_explained_by_sum": ev_sum,
        "variance_explained_by_diff": ev_diff,
        "median_sum": float(np.median(ev_sum)),
        "median_diff": float(np.median(ev_diff)),
        "n_sum_above_0.9": int((ev_sum > 0.9).sum()),
        "dead_neurons": int((act.max(axis=(0, 1)) <= 0).sum()),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _config_from_run(run: dict) -> Config:
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in run["config"].items() if k in fields})


def analyse(run_dir: Path, n_boot: int = 2000, n_perm: int = 2000,
            seed: int = 0) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    cfg = _config_from_run(run)
    state = torch.load(run_dir / "model_final.pt", map_location="cpu")
    state = {k: v.numpy() for k, v in state.items()}
    model = build_model(cfg)                     # shape check: the state must fit
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})

    curves = effective_curves(state, cfg)
    p = cfg.p
    spec_a = curve_spectra(curves["u_a"], p)
    spec_b = curve_spectra(curves["u_b"], p)
    spec_out = curve_spectra(curves["out"], p)
    sums = sum_dependence(curves, p)

    per_criteria = []
    for crit in (PROPOSED, *SENSITIVITY):
        cls = classify_neurons(spec_a, spec_b, spec_out, crit)
        mask = cls.pop("mask")
        cls["phase_relation"] = phase_relation(spec_a, spec_b, spec_out, mask,
                                               n_boot, n_perm, seed)
        per_criteria.append(cls)

    def curve_summary(spec: dict) -> dict:
        return {
            "median_dominant_fraction": float(np.median(spec["dominant_fraction"])),
            "dominant_freq_histogram": {int(k): int(v) for k, v in
                                        zip(*np.unique(spec["dominant_freq"],
                                                       return_counts=True))},
        }

    return {
        "run_id": cfg.run_id,
        "arch": cfg.arch,
        "p": p,
        "d_mlp": cfg.d_mlp,
        "n_params": cfg.n_params,
        "status": "MEASUREMENT ONLY — thresholds pending docs/HUMAN_DECISIONS.md §9.6",
        "u_a": curve_summary(spec_a),
        "u_b": curve_summary(spec_b),
        "out": curve_summary(spec_out),
        "sum_dependence": {k: v for k, v in sums.items()
                           if not isinstance(v, np.ndarray)},
        "criteria_sweep": per_criteria,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MLP per-neuron mechanism analysis")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None,
                    help="write JSON here (default: <run_dir>/mlp_mechanism.json)")
    args = ap.parse_args()

    res = analyse(args.run_dir, args.n_boot, args.n_perm, args.seed)
    out = args.out or (args.run_dir / "mlp_mechanism.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
