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
Arch = Literal["transformer", "mlp", "mlp_twohot"]

#: Steps at which a full ``state_dict`` is saved (docs/dev/RUN_FORMAT_V2.md §1).
#: Fixed BEFORE the architecture-study runs; only entries <= ``steps`` are used.
CHECKPOINT_GRID: tuple[int, ...] = (
    0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    12000, 14000, 16000, 18000, 20000, 22500, 25000,
)

#: (train-accuracy threshold, test-accuracy threshold) pairs used for transition
#: detection on the dense evaluation curves. "primary" decides; the two
#: sensitivity sets are always reported next to it (master prompt §15).
THRESHOLD_SETS: dict[str, tuple[float, float]] = {
    "primary": (0.99, 0.95),
    "sens_loose": (0.98, 0.90),
    "sens_strict": (1.0, 0.99),
}

#: ``run_id`` prefix per architecture.
ARCH_PREFIX: dict[str, str] = {"transformer": "txf", "mlp": "mlp", "mlp_twohot": "m2h"}


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

    # --- run format v2 (docs/dev/RUN_FORMAT_V2.md) ---
    # ``study`` is appended to run_id as "_<study>" so the architecture-study
    # runs never collide with (or overwrite) the archived legacy runs; ""
    # keeps every legacy run_id byte-identical.
    study: str = ""
    # dense accuracy/loss evaluation periods (steps); evaluation runs under
    # no_grad and consumes no RNG, so it never changes the trajectory
    eval_every_train: int = 10
    eval_every_test: int = 25
    # full state_dict checkpoints at these steps (only entries <= steps)
    checkpoint_grid: tuple[int, ...] = CHECKPOINT_GRID

    def __post_init__(self) -> None:
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError(
                f"train_frac must be strictly between 0 and 1, got {self.train_frac} "
                "(0 or 1 leaves an empty split and produces NaN curves)")
        if self.p < 2:
            raise ValueError(f"p must be >= 2, got {self.p}")
        if self.eval_every_train < 1 or self.eval_every_test < 1:
            raise ValueError("eval_every_train / eval_every_test must be >= 1")
        if self.arch not in ARCH_PREFIX:
            raise ValueError(f"unknown arch {self.arch!r}; choices: {list(ARCH_PREFIX)}")
        # a run.json round-trip turns the tuple into a list; normalize so that
        # configs compare equal regardless of where they were loaded from
        grid = tuple(sorted({int(s) for s in self.checkpoint_grid}))
        if any(s < 0 for s in grid):
            raise ValueError(f"checkpoint_grid must be non-negative, got {grid}")
        object.__setattr__(self, "checkpoint_grid", grid)

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
        if self.arch == "mlp_twohot":
            return (2 * p * dm        # W_in on the concatenated two-hot input
                    + dm * p          # W_out
                    + dm + p)         # b_in, b_out
        raise ValueError(f"n_params not defined for arch {self.arch!r}")

    @property
    def run_id(self) -> str:
        a = ARCH_PREFIX[self.arch]
        gf = f"_gf{self.grokfast_lambda}" if self.grokfast else ""
        # d_mlp tag only when it differs from the default, so every run_id
        # recorded before the parameter-matched control existed is unchanged.
        # Without it the d_mlp=572 control (§3.6) would collide with the
        # d_mlp=512 baseline run directory.
        dm = f"_dm{self.d_mlp}" if self.d_mlp != 512 else ""
        study = f"_{self.study}" if self.study else ""
        return (f"{a}_{self.task}_p{self.p}_wd{self.weight_decay}"
                f"_frac{self.train_frac}{gf}{dm}_seed{self.seed}{study}")

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON has no tuples: store the grid as a list so a config loaded back
        # from run.json compares equal to a freshly built one (config_diff)
        d["checkpoint_grid"] = list(self.checkpoint_grid)
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
    # --- architecture study, run format v2 (docs/dev/RUN_FORMAT_V2.md) ---
    # Fixed 25k-step budget for EVERY run (same training steps for both
    # architectures, master prompt §14; a fixed post-generalization budget
    # rather than an early stop at the crossing, cf. Khanh arXiv:2607.06639).
    "arch25k": {"steps": 25000, "study": "arch25k", "train_frac": 0.3,
                "weight_decay": 1.0},
    "arch25k_param_matched": {"steps": 25000, "study": "arch25k", "train_frac": 0.3,
                              "weight_decay": 1.0, "arch": "mlp", "d_mlp": 572},
    "arch25k_twohot": {"steps": 25000, "study": "arch25k", "train_frac": 0.3,
                       "weight_decay": 1.0, "arch": "mlp_twohot"},
}


def get_config(preset: str = "nanda", **overrides) -> Config:
    """Build a Config from a named preset with optional field overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choices: {list(PRESETS)}")
    params = dict(PRESETS[preset])
    params.update(overrides)
    return Config(**params)
