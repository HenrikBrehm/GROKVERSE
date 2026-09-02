# Run format v2 — dense evaluation, pre-specified checkpoints, manifest

Status: **specification for implementation** (2026-09-02). Governs every run launched for the
architecture study. Legacy runs (`training/runs/*` without `manifest.json`) keep their format and are
never rewritten; they are archived under `archive/pre_arch_study_2026-09-02/`.

Design decisions taken (with the human author's answers of 2026-09-02):

| Decision | Value | Source |
|---|---|---|
| Stopping rule | **fixed budget, `steps = 25000` for every primary run**, no early stop | human choice (master prompt §14 "same training steps"; Khanh 2607.06639: at-grok ≠ converged) |
| Primary setting | `p=113`, task `add`, `train_frac=0.3`, `weight_decay=1.0`, no Grokfast, `lr=1e-3`, `betas=(0.9,0.98)`, full batch | master prompt §14 recommended standard |
| Seeds | paired seeds `0..9` for transformer and MLP (same seed → same split) | master prompt §14 (target 10) |
| Threads | `threads=1`, pinned; matrix parallelized across processes | RESEARCH_SPEC §3.8 |
| Primary thresholds | memorization: train acc ≥ 0.99; generalization: test acc ≥ 0.95 | master prompt §15 |
| Sensitivity thresholds | (train ≥ 0.98, test ≥ 0.90) and (train = 1.00, test ≥ 0.99) | master prompt §15 ("at least one further threshold") |
| Dense evaluation | train acc+loss every 10 steps, test acc+loss every 25 steps | master prompt §15 |
| Run directory | `training/runs/<run_id>/` with `run_id = <legacy id>_<study>`; primary study tag `arch25k` | keeps every legacy `run_id` unchanged |

## 1. Config additions (`grokverse/config.py`)

```python
study: str = ""                 # run_id suffix "_<study>" when non-empty; "" keeps legacy ids
eval_every_train: int = 10      # dense train-set evaluation period (steps)
eval_every_test: int = 25       # dense test-set evaluation period (steps)
checkpoint_grid: tuple[int, ...] = CHECKPOINT_GRID   # full state_dict saved at these steps (<= steps)
arch: Literal["transformer", "mlp", "mlp_twohot"]     # new two-hot literature-control MLP
```

```python
CHECKPOINT_GRID = (0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                   12000, 14000, 16000, 18000, 20000, 22500, 25000)
```

`run_id` prefix per arch: `txf`, `mlp`, `m2h`. New presets:

```python
"arch25k":            {"steps": 25000, "study": "arch25k", "train_frac": 0.3, "weight_decay": 1.0}
"arch25k_param_matched": arch25k + {"arch": "mlp", "d_mlp": 572}
"arch25k_twohot":        arch25k + {"arch": "mlp_twohot"}
```

`Config.n_params` must cover `mlp_twohot`: `2*p*d_mlp + d_mlp + d_mlp*p + p`.

## 2. Two-hot MLP (`grokverse/models/mlp_twohot.py`)

Literature control. Swaroop (arXiv:2603.23784, §2, verified 2026-09-02) uses two-hot `2p` → 256 ReLU → `p`
at `p = 97` with a split stratified by `c` and early stopping; GROKVERSE deliberately keeps `d_mlp = 512`,
`p = 113`, the seeded random split and the fixed 25k-step budget so that the control differs from the
shared-embedding MLP in the input parametrization only. It is *our* two-hot variant of that setup and is
labelled as such (PREREG_BRIEF addendum 2026-09-03):

```
x = concat(onehot_p(a), onehot_p(b))            # [B, 2p]
h = ReLU(x @ W_in + b_in)                        # W_in [2p, d_mlp]
logits = h @ W_out + b_out                       # W_out [d_mlp, p]
```

Init: `W_in ~ N(0, init_scale/sqrt(2p))`, `W_out ~ N(0, init_scale/sqrt(d_mlp))`, biases 0.
`logits_last(tokens)` reads columns 0 and 1 of the `[a, b, =]` token tensor like `TwoLayerMLP`.
Effective per-neuron curves are simply `u_a = W_in[:p]`, `u_b = W_in[p:]` (no embedding).

Every model exposes `embedding_snapshot() -> np.ndarray [p_or_vocab, d]` used for the legacy
`embeddings.npy` log: transformer/MLP return `W_E`, two-hot returns `W_in[:p]` (the a-half). `run.json`
records `"embedding_object": "W_E" | "W_in_a_half"`.

## 3. Training loop (`grokverse/train.py`)

Unchanged: seeding, AdamW, full-batch CE on `logits[:, :p]`, Grokfast hook, legacy log schedule
(`n_logged_steps`) with `curves` + `embeddings.npy` + `model_final.pt`, config-collision guard,
`torch.set_num_threads(cfg.threads)` before any tensor op.

Added (all under `torch.no_grad()`, so the parameter trajectory is bit-identical to a run without it):

* **Dense evaluation.** At every step `s` with `s % eval_every_train == 0` (and `s == 0`) evaluate the
  train split; at every `s % eval_every_test == 0` (and `s == 0`) the test split. Stored in `run.json`:
  ```json
  "dense": {"train_steps": [...], "train_acc": [...], "train_loss": [...],
            "test_steps":  [...], "test_acc":  [...], "test_loss":  [...],
            "eval_every_train": 10, "eval_every_test": 25}
  ```
* **Transition detection** `detect_transition_dense(dense, train_thr, test_thr)` for every threshold
  pair in `THRESHOLD_SETS = {"primary": (0.99, 0.95), "sens_loose": (0.98, 0.90), "sens_strict": (1.0, 0.99)}`:
  ```json
  "transitions": {"primary": {"train_thr": 0.99, "test_thr": 0.95,
                              "memorization": {"first_crossing_step": 150, "previous_evaluated_step": 140, "eval_every": 10},
                              "generalization": {"first_crossing_step": 8375, "previous_evaluated_step": 8350, "eval_every": 25},
                              "grok_gap": 8225, "grok_gap_interval": [8225-10-25, 8225+10+25] (see below)},
                  "sens_loose": {...}, "sens_strict": {...}}
  ```
  Interval semantics: the true crossing lies in `(previous_evaluated_step, first_crossing_step]`. The gap
  interval is `[gen.prev - mem.first, gen.first - mem.prev]`. A crossing that never happens is `null`
  with the last evaluated step recorded. **The word "exact" is never used for these.**
  The legacy `transition` block (from the log-schedule `curves`, thresholds 0.99/0.95) is kept for
  backward compatibility and labelled `"legacy_log_grid": true`.
* **Checkpoints.** Full `state_dict` saved to `ckpt_step{step:06d}.pt` at every grid step ≤ `steps`, plus
  event checkpoints at the *first dense evaluation* that crosses the primary thresholds (memorization:
  train acc ≥ 0.99; generalization: test acc ≥ 0.95). `checkpoints.json`:
  ```json
  [{"step": 0, "path": "ckpt_step000000.pt", "kind": "grid", "sha256": "..."},
   {"step": 150, "path": "ckpt_step000150.pt", "kind": "memorization", "sha256": "..."},
   {"step": 8375, "path": "ckpt_step008375.pt", "kind": "generalization", "sha256": "..."},
   {"step": 25000, "path": "ckpt_step025000.pt", "kind": "grid,final", "sha256": "..."}]
  ```
  `model_final.pt` remains a copy of the final state for legacy tooling.
* **Pre-specified checkpoint roles** (`grokverse/checkpoints.py: assign_roles(run_dir)`), a *rule* fixed
  before the runs, applied after:
  | role | rule |
  |---|---|
  | `init` | step 0 |
  | `pre_memorization` | last grid checkpoint with step < memorization first-crossing (or `null`) |
  | `memorization` | event checkpoint |
  | `mid_plateau` | grid checkpoint nearest to `(memorization + generalization) / 2` |
  | `pre_generalization` | last grid checkpoint with step < generalization first-crossing |
  | `generalization` | event checkpoint |
  | `final` | step `steps` (fixed budget; called "final", never "converged", unless a metric plateau is shown) |

## 4. Manifest (`grokverse/manifest.py`)

Written to `runs/<run_id>/manifest.json` at start (`status: "running"`) and rewritten at the end
(`status: "completed" | "failed" | "aborted"`, with `abort_reason`). Fields:

```
run_id, study, arch, seed, task, p, train_frac, weight_decay, grokfast, steps,
split_hash            sha256 of the sorted train index array (int64, little-endian bytes)
git_commit, python_version, torch_version, numpy_version,
platform, processor, cpu_count, torch_num_threads,
config                full Config.to_dict()
start_utc, end_utc, elapsed_seconds, steps_completed, status, abort_reason,
n_params              {"total", "trainable", "per_module": {name: count}}
weight_norms_init     {"total": L2, "per_module": {...}}   (after seeding, before step 1)
weight_norms_final    same, after the last step
effective_weight_decay_per_step   lr * weight_decay   (AdamW decoupled shrink factor)
checkpoints           list from checkpoints.json
result_files          ["run.json", "embeddings.npy", "model_final.pt", "checkpoints.json", "manifest.json", "train.log"]
```

`python -m grokverse.manifest` aggregates every `runs/*/manifest.json` into
`results/run_manifest.csv` and `results/run_manifest.json` (one row per run; nested dicts flattened
with dotted keys in the CSV). `validate_manifest(d)` raises `ValueError` naming the first missing or
mistyped field; the aggregator calls it on every file and refuses to write a partial aggregate.

## 5. Matrix launcher (`grokverse/matrix.py`)

```
python -m grokverse.matrix --block primary --workers 8 [--dry-run] [--seeds 0-9]
```

Blocks (each cell = list of `Config` objects; both architectures always get **identical seeds**):

| block | contents |
|---|---|
| `pilot` | `arch25k`, seed 0, transformer + mlp |
| `primary` | `arch25k`, seeds 0–9, transformer + mlp |
| `confound` | `arch25k` variants: {grokfast ∈ {False, True}} × {train_frac ∈ {0.3, 0.5}}, seeds 0–2, both archs; the (False, 0.3) cell is the primary block and is not re-run |
| `param_matched` | `arch25k_param_matched` (mlp, d_mlp=572), seeds 0–9 |
| `twohot` | `arch25k_twohot`, seeds 0–2 |

Each run is a separate `python -m grokverse.train ...` subprocess with `threads=1`, stdout/stderr to
`runs/<run_id>/train.log`; runs whose manifest says `completed` are skipped; longest runs
(transformer, then frac 0.5) are scheduled first. Exit status per run is collected and printed as JSON.

## 6. Tests (`training/tests/test_run_format_v2.py`, `check()` convention of `test_core.py`)

* dense transition detection on synthetic curves: first crossing, previous step, interval, `null` case
* every threshold set reported; sensitivity never yields an *earlier* memorization than a looser threshold
* checkpoint grid filtered to `<= steps`; event checkpoints appear once; roles rule on a synthetic run
* manifest validation: complete manifest passes; each required key removed in turn raises
* split hash: deterministic across calls; differs between seeds; independent of `arch`
* two-hot MLP: output shape `[B, p]`, `n_params` equals the built module, forward matches a hand computation
* paired seeds: every matrix block yields identical seed lists for both architectures
* run_id: legacy ids unchanged when `study == ""`; suffix applied otherwise; `m2h` prefix
* end-to-end smoke: `steps=30`, `eval_every_train=10`, `eval_every_test=15`, grid `(0, 20, 30)` into a temp
  dir → `run.json` has `dense` + `transitions`, `checkpoints.json` lists 3 grid entries, manifest validates,
  `model_final.pt` equals `ckpt_step000030.pt`
