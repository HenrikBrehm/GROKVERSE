# training/results/external — analysis outputs of the externally executed blocks

Unmodified copies of the small analysis artifacts from the reviewer's data bundle of 2026-09-08
(`README_BUNDLE.md` is the bundle's own README). The runs themselves — checkpoints, `run.json`,
manifests, embeddings, logs — are in the gitignored `training/external/` (1.5 GB). Provenance and
verification: `docs/external/README.md`; report: `RESULTS.md` §16; decision: `docs/HUMAN_DECISIONS.md` D9.

| path | produced by | re-derived here? |
|---|---|---|
| `conv100k/aggregate_conv100k/` | frozen `aggregate` over the 20 S1 runs (`--study conv100k`, 100,000 steps) | inputs to the checks below |
| `conv100k/conv100k_reports/{decision_tree,statistics,h3_report,h4_report,analysis_driver}.json` | frozen `decision_tree`, `statistics_report`, `h3_report`, `h4_report` at commit `42dd79a` | **yes** — byte-identical apart from timestamps/paths when re-run with this branch's code on `aggregate_conv100k` |
| `conv100k/s2_ranked_ablation/` | reviewer's own script (not in the bundle) | no — reported only |
| `conv100k/s2b_freq_ranked_ablation/` | reviewer's own script (not in the bundle) | no — reported only |
| `conv100k/s3_frequency_families/` | reviewer's own script (not in the bundle) | no — reported only; the per-rule key-frequency **counts** it quotes were re-derived from the runs' frozen `key_frequencies` artifacts and agree |
| `twohot10/aggregate_twohot10/` | frozen `aggregate` over the 10 two-hot runs (25,000 steps) | inputs; the `nanda` count of 56/56 in all 10 seeds re-derived from the runs' artifacts |
| `twohot10/s2_ranked_ablation/` | reviewer's own script | no — reported only |
| `*/RUN_INFO.txt` | the runner's status file (commit, instance, versions, finish time) | — |

The five top-level 25k-block files that the reviewer's analysis chain overwrote on the instance
(`analysis_driver.json`, `decision_tree.json`, `h3_report.json`, `h4_report.json`, `statistics.json`)
were restored by the reviewer before upload; the bundle's copies of this repository's 25k artifacts are
**not** copied here (they are ours already; a spot check of `function_agreement/…seed0…` found 0 of 242
values differing).
