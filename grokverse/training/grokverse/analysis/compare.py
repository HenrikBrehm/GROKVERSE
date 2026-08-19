"""Cross-architecture comparison (PLAN Phase 4): transformer vs MLP grokking.

Aggregates, across seeds for each architecture, the grokking timing
(generalization step, grok gap), final accuracy, and embedding Fourier sparsity,
and produces a comparative figure. Answers the original-experiment question: do
different architectures grok modular addition, and do they converge to the same
sparse periodic (trig-identity-style) structure, or a different one?

Usage (from training/):
    python -m grokverse.analysis.compare                        # Grokfast frac=0.5 setting
    python -m grokverse.analysis.compare --frac 0.3 --unaccelerated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils import runs_dir  # noqa: E402
from .fourier import dominant_frequencies  # noqa: E402

ARCH_COLOR = {"transformer": "#4c8dff", "mlp": "#ff6b6b"}
ARCH_PREFIX = {"transformer": "txf", "mlp": "mlp"}


def discover(arch: str, frac: float = 0.5, grokfast: bool = True) -> list[Path]:
    """Run dirs for one architecture at one setting. run_id encodes the
    train fraction and (only when enabled) a ``_gf<lambda>`` tag, so
    ``frac<frac>_gf`` selects Grokfast runs and ``frac<frac>_seed`` the
    un-accelerated ones."""
    pref = ARCH_PREFIX[arch]
    tag = f"frac{frac}_gf" if grokfast else f"frac{frac}_seed"
    out = []
    for d in sorted(runs_dir().iterdir()):
        if not (d / "run.json").exists() or not (d / "embeddings.npy").exists():
            continue
        if d.name.startswith(pref) and tag in d.name:
            out.append(d)
    return out


def summarize_run(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    t = run["transition"]
    embeds = np.load(run_dir / "embeddings.npy")
    dom = dominant_frequencies(embeds[-1], run["config"]["p"])
    return {
        "run_id": run["config"]["run_id"],
        "seed": run["config"]["seed"],
        "test_generalized_step": t["test_generalized_step"],
        "grok_gap": t["grok_gap"],
        "final_test_acc": t["final_test_acc"],
        "dominant_fraction": dom["dominant_fraction"],
        "dominant": dom["dominant"],
        "curves": run["curves"],
        "logged_steps": run["logged_steps"],
    }


def agg(values: list) -> dict:
    """Mean ± SAMPLE std (ddof=1) over seeds; None values (never grokked) are
    excluded but counted via n vs n_total."""
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0, "n_total": len(values)}
    std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"mean": round(float(np.mean(v)), 4), "std": round(std, 4),
            "n": len(v), "n_total": len(values)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-architecture grokking comparison")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--unaccelerated", action="store_true",
                    help="compare the no-Grokfast runs instead of the Grokfast ones")
    args = ap.parse_args()
    grokfast = not args.unaccelerated
    setting = f"frac={args.frac}, {'Grokfast' if grokfast else 'un-accelerated'}"

    archs = ["transformer", "mlp"]
    data = {a: [summarize_run(d) for d in discover(a, args.frac, grokfast)] for a in archs}

    summary = {}
    for a in archs:
        runs = data[a]
        summary[a] = {
            "n_runs": len(runs),
            "grokked": sum(1 for r in runs if r["test_generalized_step"] is not None),
            "seeds": [r["seed"] for r in runs],
            "test_generalized_step": agg([r["test_generalized_step"] for r in runs]),
            "grok_gap": agg([r["grok_gap"] for r in runs]),
            "final_test_acc": agg([r["final_test_acc"] for r in runs]),
            "dominant_fraction": agg([r["dominant_fraction"] for r in runs]),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for a in archs:
        for j, r in enumerate(data[a]):
            steps = np.clip(np.array(r["logged_steps"]), 1, None)
            ax1.plot(steps, r["curves"]["test_acc"], color=ARCH_COLOR[a], alpha=0.55,
                     label=a if j == 0 else None)
    ax1.set(xscale="log", xlabel="step", ylabel="test accuracy",
            title=f"Grokking: transformer vs MLP (test acc, all seeds; {setting})")
    ax1.axhline(0.95, ls=":", color="#888", alpha=0.6)
    ax1.legend()

    xs = np.arange(len(archs))
    for x, a in zip(xs, archs):
        s = summary[a]["test_generalized_step"]
        if s["mean"] is None:
            # never plot a 0-height bar for "never grokked" — annotate honestly
            ax2.text(x, 0.02, "did not grok", ha="center", va="bottom",
                     transform=ax2.get_xaxis_transform(), color=ARCH_COLOR[a])
            continue
        ax2.bar([x], [s["mean"]], yerr=[s["std"] or 0], color=ARCH_COLOR[a],
                capsize=6, alpha=0.85)
        ax2.text(x, 0.9, f"n={s['n']}/{s['n_total']}", ha="center",
                 transform=ax2.get_xaxis_transform(), color="#555", fontsize=9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(archs)
    ax2.set(ylabel="generalization step (mean ± sample std)",
            title=f"Grokking delay by architecture ({setting})")

    fig_dir = runs_dir().parents[0] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    suffix = "" if grokfast else "_unaccelerated"
    fig.savefig(fig_dir / f"crossarch_comparison{suffix}.png", dpi=130)
    plt.close(fig)

    print(json.dumps({"setting": setting, **summary}, indent=2))


if __name__ == "__main__":
    main()
