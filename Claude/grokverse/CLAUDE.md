# GROKVERSE — Project Constitution

BWKI 2026 Hauptpreis entry. Reproduce grokking, explain it mechanistically, run one original experiment, and ship an interactive 3D explorer.

Full operating manual and plan are imported and binding:

@PROMPT.md

@PLAN.md

## Non-negotiables (full rules in PROMPT.md)

- Never fabricate, hard-code, mock or visually fake any result; every UI/writeup number traces to a real seeded run.
- Never claim grokking without the logged loss/accuracy curve proving it.
- Verify every task empirically (RUN it) before marking it done.
- Determinism: every run seeded; same config+seed => same curve.
- Maintain AI_DISCLOSURE.md from day one (BWKI + EU AI Act).
- Reproduce reality first, make it beautiful second.

---

## Obsidian Knowledge System

My Obsidian vault is called `bwki`.

The GROKVERSE project and its knowledge base are located inside:

`GROKVERSE/`

Do not modify the separate `PREP/` area unless explicitly instructed.

You have direct access to the vault and may create and edit Markdown files inside it.

Use Obsidian as the long-term knowledge base for GROKVERSE.

### Important rule

Save important knowledge that could be useful later.

Do NOT save every conversation, trivial debugging output, greetings, temporary thoughts, or obvious information.

Prefer quality over quantity.

### Vault structure

Use:

GROKVERSE/Project Log/
GROKVERSE/Sessions/
GROKVERSE/Learnings/
GROKVERSE/Experiments/
GROKVERSE/Decisions/
GROKVERSE/Bugs/
GROKVERSE/Ideas/

Do not create GROKVERSE knowledge notes outside these folders unless there is a clear reason.

Do not modify `PREP/` unless explicitly requested.

---

### Session notes

At the end of every meaningful work session, create or update a session note in:

GROKVERSE/Sessions/

Filename:

YYYY-MM-DD - GROKVERSE.md

Use this structure:

# GROKVERSE Session

## What we worked on

Short summary of the work completed.

## Changes

Important changes made to the codebase, experiment setup, UI, documentation, or methodology.

## Experiments

Document experiments performed during the session.

For every experiment include when available:

- purpose
- configuration
- seed
- dataset/task
- model
- optimizer
- important hyperparameters
- result
- relevant output files or paths

Never invent experiment results.

Only record results that were actually produced by real runs.

## Learnings

Reusable technical or scientific knowledge learned during the session.

## Problems

Important problems encountered.

## Solutions

How the problems were solved.

## Decisions

Important technical, scientific, UX, architecture, or methodology decisions.

Include the reasoning behind important decisions.

## Open Tasks

- [ ] unfinished tasks

## Related Notes

Add relevant Obsidian wikilinks.

---

### Learnings

Reusable knowledge should be stored in:

GROKVERSE/Learnings/

Examples:

- grokking concepts
- mechanistic interpretability
- neural network training behavior
- optimization
- loss and accuracy analysis
- visualization methods
- PyTorch
- experiment methodology

Before creating a new learning note, search for an existing related note.

If a related note exists, update it instead of creating a duplicate.

---

### Experiments

Important experiments should have their own notes in:

GROKVERSE/Experiments/

Each experiment note should include:

# Experiment Name

## Question

What are we trying to find out?

## Hypothesis

What do we expect and why?

## Configuration

- seed:
- task:
- dataset:
- model:
- optimizer:
- learning rate:
- weight decay:
- batch size:
- training steps:
- other relevant parameters:

## Procedure

What was actually run?

## Results

Only include real measured results.

Include references to the source data, logs, checkpoints, plots, or result files.

## Interpretation

What do the results suggest?

Clearly separate measured facts from interpretation.

## Limitations

What can we not conclude from this experiment?

## Related

Add relevant [[wikilinks]].

---

### Decisions

Important project decisions should be stored in:

GROKVERSE/Decisions/

Include:

- decision
- reason
- alternatives considered
- consequences

---

### Bugs

Important solved bugs should be stored in:

GROKVERSE/Bugs/

Include:

- problem
- symptoms
- cause
- solution
- how to prevent it in the future

---

### Ideas

Interesting ideas that are not implemented yet should be stored in:

GROKVERSE/Ideas/

Do not describe an idea as an implemented feature or confirmed result.

---

### GROKVERSE evidence rules

The existing GROKVERSE project constitution always has priority.

The Obsidian knowledge base must follow the same evidence standards.

Therefore:

- Never save fabricated experiment results.
- Never convert speculation into fact.
- Never claim grokking unless the real logged curves support it.
- Never write numbers unless they trace back to real project output.
- Preserve seeds and experiment configurations whenever relevant.
- Link experiment notes to the real logs, configs, plots, checkpoints, or source files when possible.
- Clearly distinguish observation, interpretation, hypothesis, and idea.
- If an experiment has not been run yet, label it as planned.
- Do not mark a task or experiment as completed unless it was actually verified.

---

### Wikilinks

Use Obsidian wikilinks to connect related knowledge.

Examples:

[[Grokking]]

[[Mechanistic Interpretability]]

[[Weight Decay]]

[[Modular Arithmetic]]

[[Training Dynamics]]

Do not create unnecessary duplicate notes simply to create more links.

---

### Automatic knowledge maintenance

During work:

1. Notice important reusable knowledge.
2. Notice important experiments.
3. Notice important decisions.
4. Notice solved bugs.
5. Notice promising ideas.
6. Update Obsidian when the information is valuable enough to preserve.

At the end of a meaningful session:

1. Create or update the session note.
2. Extract reusable learnings.
3. Update experiment notes.
4. Record important decisions.
5. Record important solved bugs.
6. Record unfinished tasks.
7. Link related notes with [[wikilinks]].
8. Avoid duplicate notes.
9. Update the GROKVERSE Project Log.

---

## GROKVERSE Project Log

Maintain a continuous project log for the GROKVERSE BWKI project.

The log is part of the project documentation and should reflect the real development history.

Use:

GROKVERSE/Project Log/

Maintain one main file:

GROKVERSE/Project Log/GROKVERSE Project Log.md

At the end of every meaningful work session, append a new dated entry.

Use this format:

## YYYY-MM-DD

### What was done

- important work completed
- features implemented
- experiments run
- documentation updated

### Results

- real measured results only
- include seeds, configs, paths, metrics or plots when relevant

### Decisions

- important decisions made
- why they were made

### Problems

- important problems encountered

### Solutions

- how problems were solved

### Open tasks

- [ ] unfinished work

### Next step

- most important next action

### Evidence

Link relevant:

- experiment notes
- logs
- configs
- plots
- checkpoints
- source files

Important rules:

- Never fabricate progress.
- Never write that something is finished unless it was actually verified.
- Never invent experiment results.
- Preserve dates and chronological order.
- Prefer concise factual entries.
- Do not rewrite old entries unless correcting a factual error.
- If correcting an old entry, clearly mark the correction.
- Do not modify `PREP/` unless explicitly instructed.