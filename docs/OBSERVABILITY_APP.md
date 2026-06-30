# Multi-agent observability app — design of record

The app is the **observability / visibility layer** for the whole
self-optimization workflow: a single place to review every agent, see what is
being optimized, inspect feedback and evals, and *confirm the agents are
actually self-improving* — with a fully auditable, revertable trail.

## Decision: one MLflow experiment per agent

Each agent (including a supervisor / multi-agent system, which is itself "an
agent" here) gets its **own MLflow experiment**. Rationale:

1. **Confident separation.** An experiment boundary is unambiguous; tags can be
   mis-applied or mixed. For an observability layer whose job is to be *trusted*,
   hard separation beats a tag convention.
2. **Per-agent judges.** Scorers/judges are registered **at the experiment
   level**. Experiment-per-agent is the only clean way to give each agent its own
   judge set and scorer schedule (a Codex agent and a Claude Code agent can be
   judged differently; a MAS gets judges tuned to orchestration quality).
3. **MAS / supervisor systems** get their own experiment, optimization surface,
   and lineage.

Tag-based cohorts (`src/ail/cohorts.py`) are **not discarded** — they drop to
*within-experiment* sub-segmentation (notably agent versions, and slices like a
nightly-regression set). The agent boundary moves up to the experiment.

## Data model

### The agent registry
An **agent** is a registered entry:

```
Agent = {
  agent_name:     str            # friendly, unique (the app's primary key)
  experiment_id:  str            # this agent's dedicated MLflow experiment
  judge_config:   {...}          # which L2 judges/scorers this agent uses
  tag_filter:     TagFilter|None # optional within-experiment sub-selection
}
```

A config-driven registry (a UC table + a typed loader) maps `agent_name →
experiment_id (+ judge_config)`. **"Specify the agent you're tracking"** in the
app = register an agent (name + experiment). **"Distinguish between agents"** =
the registry lists them; each carries its own experiment, judges, baselines, and
goals.

### Agent versions
Within an agent's experiment, every trace carries **`ail.agent_version`** — the
iteration of that agent's prompt/skill/config it ran under. Versions are the unit
the live comparison and the lineage hang on, and they map 1:1 to **prompt
registry versions** (see `PROMPT_REGISTRY.md`): registering an evolved
prompt mints a new version, and traces produced under it tag that version.

## Federate at publish, single pane at query

The app must stay one pane of glass even with N experiments. So:

- The **L0 publish job runs per registered agent-experiment** and writes into
  **unified UC tables keyed by `agent_name` (+ `agent_version`)** — not one table
  per experiment.
- The app reads that **single** set of unified tables and segments by agent /
  version in SQL. Federation cost lives at publish time; the app stays simple and
  fast. Cross-agent comparison is a `GROUP BY agent_name`, not a cross-experiment
  join at render time.

## The screens (phased)

Keep the current landing page; add an **agent switcher** and these views.

- **Phase A — agent registry + multi-agent landing.** Register/specify an agent
  (name + experiment + judges); list and distinguish agents; the existing L0
  leaderboard becomes per-agent.
- **Phase B — live baseline-vs-new version comparison** *(the priority visual)*.
  Within a selected agent's experiment, compare a **baseline `agent_version`**
  against the **current version** on L0 (tokens/cost/redundancy), L2 (judge
  scores), and readiness — refreshed as new traces land. **First real data:** the
  Phase-2 controlled result — baseline (no-skill) vs candidate
  (token-efficiency skill), the proven **35.4% token reduction with correctness
  held** — so the screen renders real numbers from day one, then extends to
  organic version-over-version trends as tagged traces accrue.
- **Phase C — feedback, evals, jobs + lineage/audit/revert.**
  - *Feedback & evals:* per agent+version — human labels + judge assessments,
    **scored-coverage %** and **judge-vs-human agreement** (from the
    readiness/eval-health module).
  - *Optimization jobs:* a panel of running/recent GEPA / RLM / alignment / asset
    jobs and their status.
  - *Lineage / audit / revert:* a per-agent **version timeline** sourced from the
    **prompt registry** — for each version: *what changed → from which
    optimization job (GEPA run, suite hash, artifact) → with what proven held-out
    delta → did it actually improve live (organic trace metrics under that
    version)*. **Revert** = re-point the champion alias to a prior version; the
    timeline is the audit trail that lets us undo a change that did not actually
    improve things.

## Liveness

"Live / updating as traces come in" = the **publish cadence**. True continuous
refresh requires the scheduled-publish job + experiment monitoring wiring (the
continuous-ops item; see `DEPLOY.md`). The comparison view and that wiring ship
as a pair.

## Trust guarantees (carried from `READINESS_AND_TRUST.md`)

- The app **never claims improvement the readiness wall has not cleared** — a
  version with insufficient traces/labels/coverage shows "collecting / not ready,"
  not a green delta.
- "Improved" is judged on **held-out** results + organic L0, never on the data
  the optimizer trained on.
- The **lineage timeline makes every change auditable and revertable** — the
  point of the whole layer is to catch a change that *looked* like an improvement
  but wasn't, and roll it back.

## Build sequencing

Depends on the **prompt registry** (the lineage source of truth, lane in flight)
landing first. Then: Phase A (minimal — enough to define/select an agent) →
**Phase B shipped and reviewed before moving on** → Phase C. Each phase is its
own PR with cross-vendor review; the human merges.
