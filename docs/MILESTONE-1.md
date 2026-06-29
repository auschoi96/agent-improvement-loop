# Milestone 1 — Foundations + Eval Harness + Token Slice

**Scope:** Phase 0 + Phase 1 + the Phase 2 token-reduction slice. This produces
one irrefutable win (deterministic token/cost baseline + reproduction of
Example 1) and exercises the entire spine once, end-to-end, on the most
provable goal.

## Locked decisions

- **Repo:** `auschoi96/agent-improvement-loop` (private).
- **Reusability investment accepted:** `AgentAdapter`/`TraceSource` interface +
  Codex→MLflow trace-capture path are core scope.
- **GRP execution sandbox:** new schema
  `austin_choi_omni_agent_catalog.agent_improvement_loop` (dais-demo /
  `e2-demo-field-eng`).
- **Ground-truth approach:** HARVEST ai-dev-kit's GRP spine (capture → human
  approve → promote, no LLM synthesis of expected outputs) + REFERENCE
  SkillForge's `/forge` Designer⇄Critic case-design on top. Both human-anchored.

## Working model

- Every task runs in **its own git worktree** with **its own implementation
  sub-agent** and opens **its own PR**.
- Every PR gets **cross-vendor review** (Claude implements → Codex reviews, and
  vice versa) before it reaches the human. **The human merges; the loop never
  does.**
- The reviewer sees only the diff + the task's acceptance contract — never the
  implementer's worktree.

## Waves

### Wave 0 — Scaffold + ingestion seam (solo, unblocks everything)

- **Implementer:** claude_code · **Reviewer:** codex
- **Deliverables:**
  - `pyproject.toml`, package skeleton under `src/ail/`, test harness (pytest),
    lint/typecheck config (ruff + mypy or equivalent), CI stub.
  - `src/ail/ingest/base.py` — `TraceSource` + `AgentAdapter` interfaces (the
    reusability seam). Clean, documented, with type signatures.
  - `src/ail/ingest/mlflow_source.py` — HARVEST ai-dev-kit
    `trace/mlflow_integration.py` (`search_traces` ingestion), refactored to
    implement `TraceSource` and be producer-agnostic (drop the Claude-Code
    hardwiring).
  - `src/ail/ingest/adapters/claude_code.py` — HARVEST ai-dev-kit
    `agent/executor.py` Claude Code adapter behind the `AgentAdapter` interface.
  - **License-provenance check** on all harvested code → populate
    `PROVENANCE.md`, flag any incompatibility before more harvesting.
- **Acceptance:**
  - Connects to experiment `660599403165942` and pulls the 77 traces via
    `TraceSource` into the normalized record.
  - `pytest`, lint, and typecheck all green.
  - No secrets committed; harvested code attributed in `PROVENANCE.md`.

### Wave 1 — three parallel PRs (after Wave 0 merges)

**0b — L0 deterministic metrics + reproduce Example 1**
- Implementer: claude_code · Reviewer: codex
- L0 metric module: tokens, $, latency, tool-call count, redundancy rate, model
  used — computed from the normalized trace record.
- Reproduce Example 1 on the real 77 traces: surface the 549K / 943K sessions,
  the 34× re-Read, the 13–21× boilerplate re-runs. Output a baseline report.
- Acceptance: numbers reconcile against raw
  `austin_choi_omni_agent_catalog.mlflow_traces` spot-checks; report is
  reproducible from a single entrypoint.

**1a — GRP ground-truth spine + schema**
- Implementer: codex · Reviewer: claude_code
- HARVEST ai-dev-kit `grp/` (capture → human-approve → **explicit** promote via
  `promote_approved()`); HARVEST SkillForge `GroundTruthV5` schema
  (`sources` + `regression_intent` required).
- No LLM synthesis of expected outputs. Promotion writes to the Human Anchor /
  Alignment pools, never the Task Suite.
- Acceptance: round-trip test — capture a candidate, simulate human approval,
  promote, reload as `GroundTruthV5`; porting test from
  `tests/test_review_workflow.py`.

**1c — L2 judges + MemAlign + judge-vs-human agreement**
- Implementer: claude_code · Reviewer: codex
- `make_judge` scorers (correctness, modularity, groundedness); MemAlign
  `judge.align()` wired against the Alignment Set; a judge-vs-human agreement
  metric with a configurable floor.
- Acceptance: alignment runs against a labeled fixture; agreement metric
  computes and is logged; alignment cadence is decoupled from optimization
  (documented + enforced by interface).

### Wave 2 — three parallel PRs (after their Wave 1 deps)

**1b — Freeze the Task Suite** (needs 0b)
- Implementer: claude_code · Reviewer: codex
- Curate ~15–30 representative real tasks from 0b's analysis; freeze + version
  them. Optimizer access is structurally blocked.

**1d — Codex→MLflow trace capture + A/B replay** (needs Wave 0 + 1b)
- Implementer: codex · Reviewer: claude_code
- `src/ail/ingest/adapters/codex.py` — capture Codex runs into MLflow
  (experiment `660599403165942`). Replay the frozen suite through **both**
  Claude Code and Codex → controlled A/B (same tasks, two agents).
- Acceptance: Codex traces appear in MLflow and normalize through `TraceSource`
  identically to Claude Code traces.

**2 — Token-reduction lever, end-to-end** (needs 0b, 1b, 1c)
- Implementer: claude_code · Reviewer: codex
- Intervention: build a metric view (in the sandbox schema) + a tool hitting it
  + a skill update pointing the agent at the tool. Evaluate candidate vs
  baseline on the **frozen** Task Suite. Show token drop with a correctness
  guardrail (L2). Human gate.
- Acceptance: reproduces the shape of
  `test_optimize_improves_quality_and_reduces_tokens` on our data — measurable
  token reduction, correctness not regressed below baseline, attributed to the
  intervention.

### Wave 1.5 (parallel, low-risk) — thin read-only L0 leaderboard

- Implementer: claude_code · Reviewer: codex
- A minimal Databricks App surface that reads the L0 baseline so the token
  numbers are *visible* immediately. Grows into the Phase 4 product. A UI on no
  data is theater, so this lands only after 0b produces real L0 output.

## Dependency graph

```
Wave 0 ──┬─> 0b ─┬─> 1b ─┬─> 2
         │       │       │
         ├─> 1a  │       │
         │       └─> 1d  │
         └─> 1c ─────────┘
                 (2 also needs 1c)
1.5 depends on 0b
```

## Out of scope for Milestone 1

- GEPA prompt/skill optimization as a full lever (Phase 3).
- Multi-lever asset generation beyond the single token-reduction asset.
- RLM deep review (Phase 5).
- NL goal compiler beyond a minimal stub (full version in Phase 4).
