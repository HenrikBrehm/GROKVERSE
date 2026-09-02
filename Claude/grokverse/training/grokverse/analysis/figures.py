"""Generate publication figures from a saved run (PLAN Phase 3).

Usage (from training/):
    python -m grokverse.analysis.figures runs/txf_add_p113_wd1.0_frac0.3_seed0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fourier import (  # noqa: E402
    dominant_frequencies,
    frequency_concentration_over_time,
)
from .pca import fit_pca, project  # noqa: E402


def _load(run_dir: Path):
    run = json.loads((run_dir / "run.json").read_text())
    embeds = np.load(run_dir / "embeddings.npy")
    return run, embeds


def main() -> None:
    run_dir = Path(sys.argv[1])
    fig_dir = run_dir.parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run, embeds = _load(run_dir)
    p = run["config"]["p"]
    rid = run["config"]["run_id"]
    steps = np.clip(np.array(run["logged_steps"]), 1, None)
    c = run["curves"]
    t = run["transition"]

    dom = dominant_frequencies(embeds[-1], p)
    spec = dom["spectrum"]
    conc = frequency_concentration_over_time(embeds, p, dom["dominant"])

    # --- Figure 1: the grokking proof (loss, accuracy, progress measure) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    a_loss, a_acc, a_prog = axes
    a_loss.plot(steps, c["train_loss"], label="train", color="#4c8dff")
    a_loss.plot(steps, c["test_loss"], label="test", color="#ff6b6b")
    a_loss.set(xscale="log", yscale="log", xlabel="step", ylabel="loss", title="Loss")
    a_loss.legend()
    a_acc.plot(steps, c["train_acc"], label="train", color="#4c8dff")
    a_acc.plot(steps, c["test_acc"], label="test", color="#ff6b6b")
    a_acc.set(xscale="log", xlabel="step", ylabel="accuracy",
              title="Accuracy — grokking transition")
    a_acc.legend()
    a_prog.plot(steps, conc, color="#39d98a")
    a_prog.set(xscale="log", xlabel="step",
               ylabel=f"power in k={dom['dominant']}",
               title="Progress measure: key-frequency concentration")
    for ax in axes:
        if t.get("train_saturated_step"):
            ax.axvline(max(t["train_saturated_step"], 1), ls=":", color="#888", alpha=0.7)
        if t.get("test_generalized_step"):
            ax.axvline(t["test_generalized_step"], ls="--", color="#c9a227", alpha=0.8)
    fig.suptitle(f"GROKVERSE — {rid}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{rid}_curves.png", dpi=130)
    plt.close(fig)

    # --- Figure 2: embedding Fourier spectrum (final) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(spec["freqs"], spec["fraction"], color="#4c8dff")
    ax.set(xlabel="frequency k", ylabel="fraction of embedding power",
           title=f"Embedding Fourier spectrum — dominant k = {dom['dominant']} "
                 f"({dom['dominant_fraction']*100:.1f}% of power)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{rid}_fourier.png", dpi=130)
    plt.close(fig)

    # --- Figure 3: PCA ring (final embedding, number tokens) ---
    num = embeds[-1][:p]
    pca = fit_pca(num, 3)
    coords = project(num, pca)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(p), cmap="twilight", s=24)
    ax.set(aspect="equal", xlabel="PC1", ylabel="PC2",
           title="Embedding PCA (final) — periodic ring")
    fig.colorbar(sc, label="token n")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{rid}_pca_ring.png", dpi=130)
    plt.close(fig)

    print(json.dumps({
        "saved_to": str(fig_dir),
        "dominant_frequencies": dom["dominant"],
        "dominant_fraction": round(dom["dominant_fraction"], 4),
        "final_key_freq_concentration": round(conc[-1], 4),
        "transition": t,
    }, indent=2))


if __name__ == "__main__":
    main()
