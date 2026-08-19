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
    def run_id(self) -> str:
        a = {"transformer": "txf", "mlp": "mlp"}.get(self.arch, self.arch[:3])
        gf = f"_gf{self.grokfast_lambda}" if self.grokfast else ""
        return (f"{a}_{self.task}_p{self.p}_wd{self.weight_decay}"
                f"_frac{self.train_frac}{gf}_seed{self.seed}")

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
}


def get_config(preset: str = "nanda", **overrides) -> Config:
    """Build a Config from a named preset with optional field overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choices: {list(PRESETS)}")
    params = dict(PRESETS[preset])
    params.update(overrides)
    return Config(**params)
