# agent-improvement-loop

A reusable, agent-agnostic **self-improvement loop** for LLM agents (coding agents and other deployments). State a goal in natural language — token efficiency, coding accuracy, cost reduction — and the system measures the agent against a **frozen, human-anchored evaluation harness**, diagnoses the dominant waste or failure mode, proposes an intervention (prompt/skill optimization via GEPA, or building a helper asset such as a metric view + tool, a pipeline, or a semantic layer), evaluates the candidate against the original on a held-out task suite, and ships only what beats the goal metric without regressing guardrails.

The design's load-bearing principle: **the optimizer is never allowed to train against the evaluation set, and the judge is aligned on a separate cadence from agent optimization.** This is what separates real improvement from a dashboard that says "improved" while quality stalls (the co-adaptation trap that every reference loop we surveyed omits).

## Status

Greenfield. Built by harvesting proven pieces from:

- **`databricks-solutions/ai-dev-kit`** (`.test/src/skill_test/`) — the optimization spine: GEPA loop, MLflow `make_judge`, MemAlign `judge.align()`, `search_traces` ingestion, GRP (Generate-Review-Promote) ground-truth pipeline, Claude Code agent adapter. *Cross-vendor verified from source (Claude + GPT-5).*
- **`databricks-field-eng/skillforge`** — ground-truth methodology (`/forge` Designer⇄Critic case design) and the `GroundTruthV5` schema contract.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`docs/MILESTONE-1.md`](docs/MILESTONE-1.md) for the current build plan. Harvest provenance and license reconciliation are tracked in [`PROVENANCE.md`](PROVENANCE.md).

## L0 deterministic metrics

`ail.metrics` computes the **L0** tier — deterministic, un-gameable metrics
(tokens, USD cost, latency, tool-call redundancy) straight from normalized
traces, plus breakdowns by model/producer/status. It is original work (no
harvested code) and emits a stable, typed, JSON-serializable contract that a
downstream UI reads — documented in
[`docs/L0_METRICS_CONTRACT.md`](docs/L0_METRICS_CONTRACT.md).

Reproduce the token-waste baseline (Example 1) on the live corpus from one
entrypoint:

```bash
python -m ail.metrics.report --experiment 660599403165942 --out-dir artifacts
```

It writes `artifacts/l0_baseline_<exp>.json` (the full contract) and
`artifacts/example1_diagnosis.{md,json}` (the diagnosis). Committed copies under
[`artifacts/`](artifacts/) capture the current corpus. See the contract doc for
the Databricks auth note for the reference workspace.

## Reference deployment

- Workspace: `e2-demo-field-eng` (dais-demo profile) / `fevm-austin-choi-omni-agent`
- MLflow experiment: `660599403165942`
- Trace tables: `austin_choi_omni_agent_catalog.mlflow_traces.*`
- Sandbox schema (GRP test-code execution): `austin_choi_omni_agent_catalog.agent_improvement_loop`
