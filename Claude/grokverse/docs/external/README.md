# docs/external — documents handed back with the externally executed follow-up blocks

**Status: received 2026-09-08, verified for integrity and (where the frozen code allows) re-derived; not yet reproduced on the authors' hardware; interpretation open (`docs/HUMAN_DECISIONS.md` D9).**

The four Markdown files in this folder were written by the study's external reviewer (the mentor credited in `RESULTS.md` §6.1 and `PREREGISTRATION.md` §14) and are stored **verbatim** as they arrived in the data bundle. Nothing in them was edited here. They are the reviewer's measurements and the reviewer's assessment; `RESULTS.md` §16 is this repository's own, separately labelled report of the same artifacts.

| file | author | what it is |
|---|---|---|
| `CONVERGENCE_RUNBOOK_S1.md` | reviewer | how the 100,000-step block (S1) was run on AWS EC2, and the binding honesty conditions (§0): external compute is disclosed, interpretation stays with the students, **at least one seed pair is reproduced locally before any S1 number enters a submission**, competition eligibility of external compute is the students' question, no threshold or pass rule changes |
| `CONVERGENCE_FINDINGS_S1.md` | reviewer | measurements of the S1 block against `LIMITATIONS.md` B3 |
| `S2_S3_FINDINGS.md` | reviewer | the reviewer's own follow-up analyses on the S1 checkpoints (S2 ranked ablation, S2b key-frequency-ranked percentile ablation, S3 frequency-family comparison) and the two-hot control extended to 10 seeds |
| `SO_WHAT_MENTOR_NOTE.md` | reviewer | the reviewer's assessment of what the record earns — explicitly *not* the students' interpretation |

## Where the data is

| what | where | in git? |
|---|---|---|
| the bundle as received, `grokverse_run_data.zip` (1,329,140,232 bytes, SHA-256 `c6e532ef0783a4a1833d5e8ffebacd51618aef21bc7641592b24369cc37add44`) | `training/external/grokverse_run_data.zip` | no (`training/external/` is gitignored) |
| the bundle unpacked: 20 S1 runs with full checkpoint history, 10 two-hot runs, logs, status files, the executed source tree with its `.git` | `training/external/{conv100k,twohot10,docs,README.md}` | no |
| the small analysis outputs (aggregates, reports, S2/S2b/S3 outputs, run info) — copies, unmodified | `training/results/external/` | **yes** |
| these four documents — copies, unmodified | `docs/external/` | **yes** |
| the reviewer's branch `arch-study-convergence` at commit `42dd79a` | fetched from the bundled source tree into this clone as a local branch; **not pushed** | local only |

## Provenance, as recorded in the bundle and checked here

| item | value | checked |
|---|---|---|
| code | commit `42dd79a49225abe33f1b4f144acb98ec26d47f69` on branch `arch-study-convergence` | `git diff --stat fda066e 42dd79a` on the bundled tree: two files — `config.py` (`CHECKPOINT_GRID` gains ten entries above 25,000; entries ≤ 25,000 unchanged) and the runbook. `fda066e` is on this branch. No other code change. |
| S1 | 20 runs, seeds 0–9 × {transformer, mlp}, `--steps 100000 --study conv100k`, otherwise the frozen `arch25k` flags | all 20 `manifest.json`: `git_commit=42dd79a`, `steps_completed=100000`, `status=completed`, Linux x86_64, torch 2.12.1+cpu, numpy 2.4.6, Python 3.12.14 |
| two-hot ×10 | 10 runs, seeds 0–9, frozen 25,000-step flags | all 10 manifests: `42dd79a`, 25,000 steps, completed |
| split hashes | per seed identical to this repository's runs of the same seed | 23 comparisons (20 S1 vs. the primary block, 3 two-hot vs. ours), 0 mismatches |
| checkpoint integrity | every file listed in every `checkpoints.json` | 829 SHA-256 recomputed, 829 match, 0 missing |
| S1 reports | `training/results/external/conv100k/conv100k_reports/` | this repository's frozen `decision_tree`, `statistics_report`, `h4_report` and `h3_report` re-run on the bundle's `aggregate_conv100k` reproduce the bundle's four reports **byte-for-byte** apart from timestamps, commit stamps and paths |
| two-hot reproduction (seeds 0–2 vs. this repository's Windows runs) | memorization step 230/230/230 vs. 230/240/230; generalization step 12,575/14,100/12,125 vs. 12,775/13,300/12,500 | read from both sets of `run.json`; cross-platform drift, same behaviour |
| S2 / S2b / S3 | outputs present in `training/results/external/conv100k/s2_ranked_ablation/`, `s2b_freq_ranked_ablation/`, `s3_frequency_families/` and `twohot10/s2_ranked_ablation/` | **the scripts that produced them (`training/run_s2_ranked_ablation.py`, `run_s2b_freq_ranked_ablation.py`, `run_s3_frequency_families.py`) are not in the bundle and not in the bundled source tree**; these numbers are reported, not re-derived. Requested from the reviewer 2026-09-08. |

## What is *not* done

- No S1 run has been reproduced on the authors' hardware (runbook §0.3). Estimated cost per the runbook: ~3.6 h for one MLP seed and ~16 h for one transformer seed at 100,000 steps, single-threaded.
- The `CHECKPOINT_GRID` extension is **not merged** into this branch; `training/grokverse/config.py` is unchanged. Reproducing an S1 run requires the reviewer's branch (locally available) or a decision to adopt the extension (D9).
- Whether externally executed compute is admissible under the competition rules has not been checked (runbook §0.4).

## Naming

The public documents of this repository refer to the reviewer as "an external reviewer". `CONVERGENCE_RUNBOOK_S1.md` §0 names the reviewer, as does `docs/RESEARCH_SPEC.md`; the bundled source tree's git history carries the reviewer's work e-mail as commit author and is therefore kept out of git. Whether the reviewer is named consistently is the human authors' decision (D9).
