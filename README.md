# agent-improvement-loop

A reusable, agent-agnostic **self-improvement loop** for LLM agents (coding agents and other deployments). State a goal in natural language — token efficiency, coding accuracy, cost reduction — and the system measures the agent against a **frozen, human-anchored evaluation harness**, diagnoses the dominant waste or failure mode, proposes an intervention (prompt/skill optimization via GEPA, or building a helper asset such as a metric view + tool, a pipeline, or a semantic layer), evaluates the candidate against the original on a held-out task suite, and ships only what beats the goal metric without regressing guardrails.

The design's load-bearing principle: **the optimizer is never allowed to train against the evaluation set, and the judge is aligned on a separate cadence from agent optimization.** This is what separates real improvement from a dashboard that says "improved" while quality stalls (the co-adaptation trap that every reference loop we surveyed omits).

## Status

Greenfield. Built by harvesting proven pieces from:

- **`databricks-solutions/ai-dev-kit`** (`.test/src/skill_test/`) — the optimization spine: GEPA loop, MLflow `make_judge`, MemAlign `judge.align()`, `search_traces` ingestion, GRP (Generate-Review-Promote) ground-truth pipeline, Claude Code agent adapter. *Cross-vendor verified from source (Claude + GPT-5).*
- **`databricks-field-eng/skillforge`** — ground-truth methodology (`/forge` Designer⇄Critic case design) and the `GroundTruthV5` schema contract.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`docs/MILESTONE-1.md`](docs/MILESTONE-1.md) for the current build plan. Harvest provenance and license reconciliation are tracked in [`PROVENANCE.md`](PROVENANCE.md).

## Reference deployment

- Workspace: `e2-demo-field-eng` (dais-demo profile) / `fevm-austin-choi-omni-agent`
- MLflow experiment: `660599403165942`
- Trace tables: `austin_choi_omni_agent_catalog.mlflow_traces.*`
- Sandbox schema (GRP test-code execution): `austin_choi_omni_agent_catalog.agent_improvement_loop`
