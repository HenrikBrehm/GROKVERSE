"""Attention-pattern analysis: where does the '=' read-out position look?

The grokked modular-addition circuit must combine BOTH operands a and b before
predicting (a + b) mod p. In a 1-layer transformer the only place tokens can
exchange information is the attention layer, and the prediction is read off the
final '=' position (index 2). So a faithful circuit should show the '=' position
attending to the two operand positions a (index 0) and b (index 1) — not only to
itself. This module measures, per head, the average attention FROM '=' TO each of
the three positions over all p*p inputs, plots a grouped bar chart, and prints a
JSON summary so the claim ("=" attends to both operands) is checkable, not asserted.

Usage (from training/):
    python -m grokverse.analysis.attention runs/txf_add_p113_wd1.0_frac0.5_gf2.0_seed0
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..config import Config  # noqa: E402
from ..data import make_dataset  # noqa: E402
from ..models import build_model  # noqa: E402

# Position indices in the [a, b, '='] token sequence.
POS_A, POS_B, POS_EQ = 0, 1, 2
POS_LABELS = ["a (pos 0)", "b (pos 1)", "= (pos 2)"]


def _config_from_run(run: dict) -> Config:
    """Rebuild the dataclass Config from a logged run config.

    The logged config block includes the computed properties ``vocab_size`` and
    ``run_id`` (and possibly other future extras) which are NOT constructor
    arguments. Keep only the real dataclass fields.
    """
    fields = {f.name for f in dataclasses.fields(Config)}
    params = {k: v for k, v in run["config"].items() if k in fields}
    return Config(**params)


def main() -> None:
    run_dir = Path(sys.argv[1])
    fig_dir = run_dir.parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    run = json.loads((run_dir / "run.json").read_text())
    cfg = _config_from_run(run)
    rid = cfg.run_id

    model = build_model(cfg)
    state = torch.load(run_dir / "model_final.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x = make_dataset(cfg)["all_x"]                 # [p*p, 3]
    attn = model.attention_pattern(x)              # [p*p, n_heads, 3, 3]

    # Attention FROM the '=' position (row index 2) TO each of [a, b, =],
    # averaged over all p*p inputs. Shape after mean: [n_heads, 3].
    from_eq = attn[:, :, POS_EQ, :]                # [p*p, n_heads, 3]
    mean_from_eq = from_eq.mean(dim=0).cpu().numpy()   # [n_heads, 3]

    n_heads = mean_from_eq.shape[0]

    # --- grouped bar chart: per head, attention from '=' to each position ---
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.26
    xpos = np.arange(n_heads)
    colors = ["#4c8dff", "#ff6b6b", "#39d98a"]
    for j, (lbl, col) in enumerate(zip(POS_LABELS, colors)):
        ax.bar(xpos + (j - 1) * width, mean_from_eq[:, j], width,
               label=f"to {lbl}", color=col)
    ax.set(xlabel="head", ylabel="mean attention from '=' position",
           ylim=(0, 1),
           title=f"Where the '=' read-out looks — {rid}")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"head {h}" for h in range(n_heads)])
    ax.legend()
    fig.tight_layout()
    out_png = fig_dir / f"{rid}_attention.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    # Per-head summary + a checkable read-out: does '=' attend to BOTH operands?
    # We call an operand "attended" if it receives a non-trivial share (>0.1) of
    # the '=' position's attention mass in that head.
    per_head = []
    for h in range(n_heads):
        a_w, b_w, eq_w = (float(mean_from_eq[h, POS_A]),
                          float(mean_from_eq[h, POS_B]),
                          float(mean_from_eq[h, POS_EQ]))
        per_head.append({
            "head": h,
            "to_a": round(a_w, 4),
            "to_b": round(b_w, 4),
            "to_eq": round(eq_w, 4),
            "attends_both_operands": bool(a_w > 0.1 and b_w > 0.1),
        })

    # Aggregate across heads: total attention each operand gets, summed over heads
    # (an operand can be read by different heads). '=' attends to both operands if
    # at least one head reads a and at least one head reads b.
    any_head_reads_a = bool((mean_from_eq[:, POS_A] > 0.1).any())
    any_head_reads_b = bool((mean_from_eq[:, POS_B] > 0.1).any())

    summary = {
        "run_id": rid,
        "p": cfg.p,
        "n_inputs": int(x.shape[0]),
        "n_heads": n_heads,
        "attention_from_eq_position": {
            "per_head": per_head,
            "mean_over_heads": {
                "to_a": round(float(mean_from_eq[:, POS_A].mean()), 4),
                "to_b": round(float(mean_from_eq[:, POS_B].mean()), 4),
                "to_eq": round(float(mean_from_eq[:, POS_EQ].mean()), 4),
            },
        },
        "eq_attends_to_both_operands": bool(any_head_reads_a and any_head_reads_b),
        "figure": str(out_png),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
