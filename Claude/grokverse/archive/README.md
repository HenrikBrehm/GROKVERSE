# archive/ — kept, not load-bearing

Everything in here was moved out of the working tree on **2026-09-04** to make the repository easier
to navigate. **Nothing was deleted.** Nothing in this folder is read by the code, the tests, the
analysis chain, `README.md` or `RESULTS.md`; every current number traces to `training/results/`, and
every current figure to `training/results/figures/`.

If a document elsewhere still cites one of the old paths below, this table is the forwarding address.
Labbook entries are not rewritten (`CLAUDE.md`: old entries stay as written), so pointers in
`docs/LABBOOK.md` and `PROGRESS.md` to the old locations are historical and resolve here.

## Tracked (in git)

| now | was | what it is |
|---|---|---|
| `logs/analysis_chain.log` | `training/results/analysis_chain.log` | stdout of `run_analysis_chain.ps1` (driver pass 1, aggregate). A re-run recreates the original path. |
| `logs/matrix_launch.log` | `training/results/matrix_launch.log` | stdout of `run_matrix_chain.ps1` (the 51-run training chain, incl. the restart after power loss). Recreated on re-run. |
| `logs/confound_analysis_{out,err}.log` | `training/results/…` | driver stdout/stderr over the confound block |
| `logs/controls_analysis_{out,err}.log` | `training/results/…` | driver stdout/stderr over the parameter-matched and two-hot blocks |
| `logs/bounded_alternative_{out,err}.log` | `training/results/…` | `run_bounded_alternative.py` stdout/stderr |
| `logs/rerun_e6{,_out,_err}.log` | `training/results/…` | the post-freeze re-run of the primary block (labbook E6) |
| `logs/sot_rerun_{out,err}.log` | `training/results/…` | the `structure_over_time` re-run after the index-vs-name fix (labbook 101/105) |
| `logs/analysis_driver_sot_probe.json` | `training/results/analysis_driver_sot_probe.json` | a one-run timing probe before that re-run; not a study artifact |
| `legacy_figures/*.png` (21) | `training/figures/*.png` | the **pre-study** figures (Phases 3–4, June 2026). Their claims were withdrawn in `RESULTS.md` §11; the current figures are `training/results/figures/`. Cited historically by `docs/BASELINE.md`, `docs/CURRENT_EVIDENCE_AUDIT.md` and `pitch/shot_list.md`. |
| `session_handoff/HANDOFF_2026-09-04.md` | `docs/dev/HANDOFF_2026-09-04.md` | the kickoff/loop prompts and stage plan for the unattended session. Its job is done; the live status is `docs/dev/HANDOFF_PROGRESS.md`. |

The `*_err.log` files are 0 bytes: no analysis run wrote to stderr. They are kept because "the error
log was empty" is itself a record.

## Not tracked (gitignored, this machine only)

| path | what it is |
|---|---|
| `untracked/docs.zip` | a stale zip containing only `docs/RESEARCH_SPEC.md`, which exists in `docs/` |
| `untracked/sources_raw/` | raw HTML/text dumps of the Khanh and Swaroop papers plus scratch figures, used to write `docs/sources/*.md` (was `docs/sources/_raw/`) |
| `untracked/sources_tmp/` | further raw paper dumps (was `docs/sources/_tmp/`) |
| `interrupted_20260903T1708Z/` | 8 transformer runs killed by the power loss of 2026-09-03 and restarted from scratch; kept for the labbook's account of that day |
| `pre_arch_study_2026-09-02/` | a full snapshot (docs, figures, runs, `MANIFEST.sha256`) of the repository state **before** the architecture study — the baseline `docs/BASELINE.md` describes |

## What was deliberately left in place

- `docs/BASELINE.md`, `docs/CAPACITY_AND_CONFOUNDS.md`, `docs/LEGACY_METRIC_AUDIT.md`, `docs/data/` —
  they answer master-prompt requirements (§3 rule 6, §14) and `analysis/capacity_report.py` and
  `training/audit_legacy_metrics.py` write into `docs/data/`.
- `docs/dev/{INTERFACES,RUN_FORMAT_V2,EXPLORER_UPDATE_PLAN,PREREG_BRIEF,HANDOFF_PROGRESS}.md` — live
  specifications, the pending explorer plan (E3), and the §23 status checklist.
- `training/runs/` — gitignored; the 16 legacy runs are still read by `python -m grokverse.manifest`
  and by `audit_legacy_metrics.py`, and the `txf_mul_*` runs are reserved for the human author (E6).
- `training/{audit_legacy_metrics.py, run_*_chain.ps1, run_bounded_alternative.py, test_core.py}` —
  each is the reproducible command for a committed artifact, or part of the test suite.
