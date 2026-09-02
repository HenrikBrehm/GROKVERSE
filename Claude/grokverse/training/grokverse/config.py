"""Typed run configuration + named presets.

Every field that affects a run lives here so it can be logged verbatim into the
run metadata (PROMPT.md §2). The canonical ``nanda`` preset reproduces the
Nanda et al. 2023 modular-addition setup; ``fast`` is a CPU-friendly de-risking
variant; ``smoke`` is a 1-step sanity check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Task = Literal["add", "mul"]
Arch = Literal["transformer", "mlp"]


@dataclass(frozen=True)
class Config:
    # --- task / data ---
    p: int = 113
    task: Task = "add"
    train_frac: float = 0.3
    seed: int = 0

    # --- architecture (Nanda recipe defaults) ---
    arch: Arch = "transformer"
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    n_ctx: int = 3
    use_ln: bool = False
    init_scale: float = 1.0

    # --- optimization ---
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    steps: int = 30000

    # --- grokfast (arXiv:2405.20233): amplify slow-varying gradient components
    #     to accelerate the (faithful) grokking transition; used only to keep
    #     CPU run times manageable. The un-accelerated run is the reference. ---
    grokfast: bool = False
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0

    # --- logging / runtime ---
    n_logged_steps: int = 150
    device: str = "cpu"
    label: str = ""
    # CPU thread count, pinned so a parallel run matrix stays reproducible:
    # torch reductions differ across thread counts (RESULTS.md §5 reports ~0.5%
    # final-accuracy wobble from it), so the matrix is parallelized across
    # PROCESSES with threads=1 each, never across threads.
    # (docs/RESEARCH_SPEC.md §3.8) Runs recorded before this field existed have
    # no "threads" key; utils.config_diff treats that as "added later", not as a
    # config mismatch.
    threads: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError(
                f"train_frac must be strictly between 0 and 1, got {self.train_frac} "
                "(0 or 1 leaves an empty split and produces NaN curves)")
        if self.p < 2:
            raise ValueError(f"p must be >= 2, got {self.p}")

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def equals_token(self) -> int:
        return self.p

    @property
    def n_params(self) -> int:
        """Trainable parameter count, derived from the shapes the model classes
        allocate. Kept here (not on the model) so a config can be compared
        *before* anything is built — the parameter-matched control of
        docs/RESEARCH_SPEC.md §3.6 is chosen with it.
        """
        d, v, p, dm = self.d_model, self.vocab_size, self.p, self.d_mlp
        if self.arch == "transformer":
            h, dh = self.n_heads, self.d_head
            return (v * d            # W_E
                    + self.n_ctx * d  # W_pos
                    + 3 * h * d * dh  # W_Q, W_K, W_V
                    + h * dh * d      # W_O
                    + d * dm + dm * d  # W_in, W_out
                    + d * v)          # W_U
        if self.arch == "mlp":
            return (p * d             # W_E (numbers only — no '=' token)
                    + 2 * d * dm      # W_in
                    + dm * p          # W_out
                    + dm + p)         # b_in, b_out
        raise ValueError(f"n_params not defined for arch {self.arch!r}")

    @property
    def run_id(self) -> str:
        a = {"transformer": "txf", "mlp": "mlp"}.get(self.arch, self.arch[:3])
        gf = f"_gf{self.grokfast_lambda}" if self.grokfast else ""
        # d_mlp tag only when it differs from the default, so every run_id
        # recorded before the parameter-matched control existed is unchanged.
        # Without it the d_mlp=572 control (§3.6) would collide with the
        # d_mlp=512 baseline run directory.
        dm = f"_dm{self.d_mlp}" if self.d_mlp != 512 else ""
        return (f"{a}_{self.task}_p{self.p}_wd{self.weight_decay}"
                f"_frac{self.train_frac}{gf}{dm}_seed{self.seed}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vocab_size"] = self.vocab_size
        d["run_id"] = self.run_id
        return d


PRESETS: dict[str, dict] = {
    "nanda": {},
    "fast": {"train_frac": 0.5, "steps": 15000},
    # Grokfast-accelerated canonical dims: same p=113 / d_model=128 / wd=1.0,
    # but grokking compressed into a CPU-feasible number of steps.
    "grokfast": {"grokfast": True, "steps": 4000},
    "smoke": {"steps": 1, "n_logged_steps": 2},
    # Parameter-matched MLP control (docs/RESEARCH_SPEC.md §3.6): the default
    # d_mlp=512 MLP has 9.8% FEWER parameters than the transformer, and
    # weight_decay=1.0 acts on all of them, so the two architectures sit under
    # different effective regularization pressure. d_mlp=572 matches the
    # transformer's 226,176 parameters to +0.02% (asserted in test_core.py).
    "mlp_param_matched": {"arch": "mlp", "d_mlp": 572},
}


def get_config(preset: str = "nanda", **overrides) -> Config:
    """Build a Config from a named preset with optional field overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choices: {list(PRESETS)}")
    params = dict(PRESETS[preset])
    params.update(overrides)
    return Config(**params)
