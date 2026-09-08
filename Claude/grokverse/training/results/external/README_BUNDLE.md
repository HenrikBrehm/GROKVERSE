# Grokverse follow-up run data (S1 convergence block + two-hot control)

External compute executed by the mentor; artifacts handed back unmodified.
**Measurements only** — interpretation remains with the student authors
(docs/HUMAN_DECISIONS.md §F). See docs/CONVERGENCE_RUNBOOK_S1.md §0 for the
binding honesty conditions, including that at least one seed pair should be
reproduced on your own hardware before any S1 number enters a submission.

## Contents

    conv100k/     S1 convergence block: 20 runs (seeds 0-9 x {transformer, mlp})
                  at 100,000 steps. runs/ has the full checkpoint history
                  (31 per run), embeddings, per-run analysis, results/, logs/.
    twohot10/     Two-hot control: 10 runs at the frozen 25,000-step budget.
    docs/         The four findings / assessment documents.

`_src/grokverse-src.tar.gz` in conv100k is the exact source tree that was
executed, including .git, so `git rev-parse HEAD` inside it returns the commit
recorded in every manifest.

## Provenance

    code commit      42dd79a49225abe33f1b4f144acb98ec26d47f69
    environment      Linux x86_64 (Amazon Linux 2023), AWS EC2 c7i
                     Python 3.12.14, torch 2.12.1+cpu, numpy 2.4.6
    your 25k block   Windows 11 AMD64, Python 3.12.10, torch 2.12.1+cpu, numpy 2.4.6

Same torch build and instruction set as your runs; OS and a Python patch level
differ. Every run records its own git_commit, split_hash, platform and library
versions in manifest.json, and every checkpoint has a SHA-256 in checkpoints.json.

## Declared deviations from the frozen configuration

1. `CHECKPOINT_GRID` extended above 25000 (entries 30000..100000) so the block
   could measure anything past 25k. Additive: entries <= 25000 are unchanged, so
   a 25,000-step run still selects exactly the same 19 grid steps. This is a
   deviation from a pre-registered constant and needs an AI_DISCLOSURE.md row.
2. `--steps` 100000 and `--study conv100k` for the S1 block. Nothing else.
3. The S2/S2b/S3 follow-up analyses ran on macOS arm64 with the plain
   torch 2.12.1 build (the pinned +cpu wheel has no macOS build). Analysis only,
   no training; baselines reproduced exactly.

The two-hot runs use flags byte-identical to your
results/matrix_twohot_20260904T022540Z.json invocation; seeds 0-2 therefore
double as a cross-platform reproduction check of your own three runs.

## Note

`run_manifest.csv` does not index the S1 runs (it was not regenerated). The
20-run index for S1 is conv100k/results/aggregate_conv100k/index.json.
