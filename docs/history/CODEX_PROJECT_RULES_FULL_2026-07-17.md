# Codex Project Rules for AAAI Paper Development（2026-07-17 完整归档）

> 当前精简规则位于仓库根目录 `CODEX_PROJECT_RULES.md`。本文件保存原始完整模板与说明，默认不在新会话启动时全文读取。

This document defines the working rules for using Codex to build and maintain an AAAI-oriented research code project. The goal is to keep the project easy to continue across multiple Codex sessions, especially when the real experiments will run on a cloud server rather than locally.

## Core Documents

Maintain these four documents in the project root:

1. `PROJECT_PLAN.md`
2. `IMPLEMENTATION_PLAN.md`
3. `ARCHITECTURE_RISKS.md`
4. `DEV_LOG.md`

These documents serve different purposes and should not be merged unless the project is still in a very early sketch stage.

Document hierarchy:

- `PROJECT_PLAN.md` is the strategic research source of truth. It defines the paper story, claim, scope, experiment goals, and reviewer-facing risks.
- `IMPLEMENTATION_PLAN.md` is the primary coding reference for Codex. It translates the research plan into a realistic, file-level engineering plan that can actually be implemented.
- `ARCHITECTURE_RISKS.md` is the active risk register. It defines implementation invariants, failure modes, mitigations, and evidence gates that must be checked before long training or strong paper claims.
- `DEV_LOG.md` is the continuity record. It preserves what changed, what was verified, what is still uncertain, and what the next Codex session should do.

When these documents conflict, prefer this order:

1. Use `PROJECT_PLAN.md` to decide whether a coding task serves the paper.
2. Use `IMPLEMENTATION_PLAN.md` to decide how Codex should implement it.
3. Use `ARCHITECTURE_RISKS.md` to decide whether an implementation or claim crosses an unresolved safety, validity, or reviewer-facing boundary.
4. Use `DEV_LOG.md` to understand the latest actual project state.

## `PROJECT_PLAN.md`

`PROJECT_PLAN.md` is responsible for the paper-level research plan and the overall project direction.

It is a big-picture document, not the main coding guide. It should describe what the project is trying to prove, why the method is publishable, what experiments are needed, and what constraints the implementation must respect.

It should answer:

- Why is this project worth doing?
- What is the core AAAI-style claim?
- What weakness in current work does this address?
- What method is proposed?
- What experiments would support the claim?
- What risks would reviewers likely raise?

Recommended sections:

- Paper title candidates
- One-line claim
- Research problem
- Motivation
- Existing method weaknesses
- Target contributions
- Method overview
- Difference from related work
- Expected datasets and benchmarks
- Main metrics
- Figure and table plan
- Reviewer-risk analysis
- Fallback plan

Do not write fake results or imply that experiments have been completed.

Avoid putting detailed file-by-file coding tasks here unless they are necessary to explain the research design. Detailed implementation work belongs in `IMPLEMENTATION_PLAN.md`.

## `IMPLEMENTATION_PLAN.md`

`IMPLEMENTATION_PLAN.md` is responsible for the engineering plan and is the main reference document Codex should use when programming.

It should turn the big project idea in `PROJECT_PLAN.md` into a reasonable, feasible, staged implementation. The plan should be specific enough that another Codex session can start coding from it without reinventing the architecture.

It should answer:

- What code needs to be written?
- Which files should be added or modified?
- What APIs, classes, functions, and configs are expected?
- What is the minimum viable implementation?
- What can be implemented now, and what should remain `TBD`?
- What is the smallest smoke test?
- What commands should be run on the cloud server?
- What counts as done for each task?

Before writing or updating this document, Codex should inspect the real repository structure if a repo exists. Do not invent file paths, training entrypoints, or benchmark scripts that do not match the repo. If the repo is not available yet, write a proposed structure and clearly mark it as `proposed` or `TBD`.

The implementation plan should prefer:

- Small modules over large rewrites
- Minimal viable paths before full training
- Config-driven changes when possible
- Clear adapters or wrappers around upstream code
- Smoke tests before long cloud runs
- Explicit done criteria for every task

Each implementation task should be written in an executable format:

````markdown
## Task N: <Task Name>

Goal:
<What this task should accomplish.>

Files:
- `path/to/file.py`
- `path/to/config.yaml`

Expected API:
- `<function_or_class_signature>`
- input: `<expected input>`
- output: `<expected output>`

Commands:
```bash
<minimal command or cloud command template>
```

Done Criteria:
- <Import or syntax check passes>
- <One-sample smoke path works>
- <No long training required for this step>
````

Use `TBD` for unknown paths, datasets, checkpoints, environment variables, or cloud machine details.

## `DEV_LOG.md`

`DEV_LOG.md` records every meaningful project change.

Every Codex session that edits code or project documents should append one entry.

Use this format:

````markdown
## YYYY-MM-DD - <Short Change Title>

### Goal
<What this change tried to accomplish.>

### Changed
- <File or module changed>
- <File or module changed>

### Commands Run
```bash
<commands actually run>
```

If no commands were run, write:

No training or cloud experiments were run in this step.

### Result
<What was verified or produced.>

### Issues
- <Known limitation, blocker, or uncertainty>

### Next
- <Recommended next task>
````

Never invent experiment results. Only record commands and results that were actually run or observed.

## Operating Rules for Codex

Before editing:

- Read `PROJECT_PLAN.md`.
- Read `IMPLEMENTATION_PLAN.md`.
- Read `ARCHITECTURE_RISKS.md`.
- Read the latest entries in `DEV_LOG.md`.
- If code already exists, inspect the real repo structure before proposing file paths.
- If a path, dataset, checkpoint, or environment detail is uncertain, mark it as `TBD`.

While editing:

- Keep changes aligned with the paper claim in `PROJECT_PLAN.md`.
- Treat `IMPLEMENTATION_PLAN.md` as the main programming reference.
- Keep tasks aligned with the file-level plan in `IMPLEMENTATION_PLAN.md`.
- If `IMPLEMENTATION_PLAN.md` is too vague to code from, improve it before making broad code changes.
- Prefer small, verifiable implementation steps.
- Do not start long training by default.
- Do not modify upstream or official files unless the plan explicitly allows it.
- Prefer local wrappers, configs, adapters, and scripts when preserving upstream code matters.

After editing:

- Run the smallest relevant smoke check if available.
- Append a new entry to `DEV_LOG.md`.
- Update task status or done criteria in `IMPLEMENTATION_PLAN.md` if needed.
- Do not claim cloud experiments passed unless they were actually run on the target machine.

## Cloud-First Experiment Policy

This project expects most heavy experiments to run on a cloud server.

Local work should focus on:

- Planning
- Code structure
- Config design
- Syntax checks
- Import checks
- Small mock-data smoke tests
- Cloud command templates
- Debug checklists

Cloud work should record:

- Machine type
- GPU type
- Environment name
- Commit or code snapshot
- Dataset path
- Checkpoint path
- Exact command
- Output directory
- Observed result
- Failure logs if any

If cloud access is not available, write the command template and mark the result as `not yet run`.

## Minimum First Step for a New Codex Session

A new Codex session should begin with:

```text
Read CODEX_PROJECT_RULES.md, PROJECT_PLAN.md, ARCHITECTURE_RISKS.md, IMPLEMENTATION_PLAN.md, and the latest DEV_LOG.md entry. Then continue from the next unfinished task. Do not invent experiment results. Use TBD for unknown paths or cloud details.
```

## Recommended Initial Prompt for Another Codex

```text
Please first read CODEX_PROJECT_RULES.md. Then create or update PROJECT_PLAN.md, ARCHITECTURE_RISKS.md, IMPLEMENTATION_PLAN.md, and DEV_LOG.md for this AAAI paper project.

The goal is to make this project continuously developable by Codex. Do not write code yet unless IMPLEMENTATION_PLAN.md is already specific enough to be used as a coding blueprint. Do not run or invent experiments. The code will mainly run on a cloud server, so unknown paths, checkpoints, datasets, and environment variables should be marked as TBD.

PROJECT_PLAN.md is the big-picture research and paper plan. It should define the paper story, central claim, motivation, related-work gap, method overview, target contributions, experiment goals, expected benchmarks, metrics, figure/table plan, reviewer risks, and fallback plan. It should answer why this project is worth doing and what would make it a plausible AAAI paper.

IMPLEMENTATION_PLAN.md is the main programming reference for Codex. It should translate PROJECT_PLAN.md into a realistic, implementable engineering plan. If a repo exists, inspect the actual repo structure before proposing files. Specify the target structure, files to add or modify, module APIs, configs, training/eval script plans, cloud command templates, smoke tests, and done criteria. Prefer a minimum viable implementation path before full training. Mark uncertain items as TBD instead of inventing them.

ARCHITECTURE_RISKS.md should record active model-validity and reviewer-facing risks, implementation invariants, mitigations, and evidence gates. DEV_LOG.md should record this initialization and every future change using Goal, Changed, Commands Run, Result, Issues, and Next. Every time code or project documents are changed, append a new DEV_LOG.md entry.

Important behavior:
- PROJECT_PLAN.md controls the research direction.
- IMPLEMENTATION_PLAN.md controls how Codex should code.
- DEV_LOG.md records what actually changed.
- Do not claim that experiments, cloud runs, or benchmarks succeeded unless they were actually run.
- If the implementation plan is not yet actionable, improve the plan first rather than writing speculative code.
```
