# Multi-agent observability app — design of record

The app is the **observability / visibility layer** for the whole
self-optimization workflow: a single place to review every agent, see what is
being optimized, inspect feedback and evals, and *confirm the agents are
actually self-improving* — with a fully auditable, revertable trail.

> The app is also the **human-in-the-loop approval control plane** for the
> autonomous loop controller: the framework detects→decides→proves→proposes a
> change on its own, and a human **approves the live apply in the app** after
> reviewing *why* it's necessary and the proof it works (Option A). See
> [`LOOP_CONTROLLER.md`](LOOP_CONTROLLER.md) for that design of record.

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

The Overview separates source freshness from computed-metric freshness:

- **Trace count is live** from the selected experiment's managed
  `*_otel_spans` Delta table (distinct completed root traces).
- **Tokens, cost, percentiles, and redundancy remain atomic L0 snapshots** from
  the Python publisher. When live count is ahead, the UI reports exactly how
  many traces are awaiting L0 refresh.

Every raw OTEL table read by the app must be declared as a DAB app
`uc_securable` with `SELECT`; an undeclared source fails closed to the snapshot.
Future Lakebase serving should continuously sync a curated scalar trace-serving
Delta table, not the raw OTEL table (raw spans contain unsupported `VARIANT`
columns and are not CDF-enabled).

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

## What shipped (Phases A + B)

This section covers Phase A (minimal) and Phase B (the priority visual). The Phase-C
lineage / audit / revert surface is documented under *What shipped (Phase C — lineage
+ revert)* below.

**Phase A — agent registry + multi-agent landing.**
- `src/ail/registry.py` — the typed, config-driven registry (`Agent` /
  `AgentRegistry`, pydantic v2, `extra="forbid"`) mapping `agent_name →
  experiment_id (+ optional judge_config / tag_filter)`. Seeded with
  `claude_code → 660599403165942`; `config/agents.yaml` is the operator-facing
  mirror (a test keeps the two in sync).
- The app keeps the current landing page and adds an **agent switcher**
  (`client/src/components/AgentSwitcher.tsx`); the L0 leaderboard is now
  parameterized per selected agent's `experiment_id`.

**Phase B — live baseline-vs-new version comparison.**
- `src/ail/publish_versions.py` extends Tier A to aggregate L0 **per (agent,
  agent_version)** — reusing `ail.metrics` outputs carried in the Phase-2
  artifact (no metric recomputed) — into unified UC tables keyed by
  `(agent_name, agent_version)`: `agent_registry`, `agent_version_l0`,
  `agent_version_comparison`, `agent_version_readiness`. Writes reuse
  `ail.publish`'s atomic, idempotent staging→`REPLACE WHERE` swap, scoped by a
  composite per-version predicate.
- **Seed (committed-artifact fallback):** `artifacts/phase2_token_lever.json` is
  published as `agent='claude_code'`, `v0-baseline-no-skill` vs
  `v1-token-efficiency-skill`, so the view renders the real **−35.4% token
  reduction with correctness held** (2 PROMOTE / 3 honest BLOCK) day one without
  live trace auth.
- **Honest readiness gating.** `ail.readiness.compute_readiness` is wired into the
  publish; the comparison carries a Python-decided display status. The 35.4% is a
  real, measured controlled-comparison result, but because organic readiness has
  not cleared (5 traces < the baseline floor) the view shows
  **`controlled_proof_collecting`** (amber) — never a green "proven improvement"
  the readiness wall has not cleared, and never a fabricated delta.

**Liveness.** "Live" = the publish cadence; the view notes the controlled-proof
provenance and the organic-readiness state honestly. The publish is
per-version-capable and idempotent; the scheduled-publish job is not built here.

## What shipped (Phase C — lineage + revert)

This delivers the Phase-C *lineage / audit / revert* surface: *see what changed,
traceably and auditably, and revert anything that did not actually improve*. It is the
**prompt registry made visible** — the lineage is sourced straight from the
human-gated promote step (`src/ail/optimize/prompt_registry.py`), not recomputed —
plus a guarded CLI to roll the champion back. The two-tier discipline is unchanged:
Python computes and writes; the app SQL is `SELECT`-only.

**Lineage source of truth — reuse, not reimplementation.** Each registered prompt
*version* already carries its provenance as `ail.prompt.*` version tags, stamped at
promote time by `register_gepa_candidate` / `register_seed_prompt`
(`PromptProvenance`): `source` (seed vs gepa-evolved), `changed`, `gepa_best_val_score`,
`gepa_num_candidates`, the held-out `holdout_evolved_savings_pct` /
`holdout_seed_savings_pct` / `holdout_savings_delta_pct`, the `candidate_artifact`
pointer, the `suite_version`, and — for a force-registered non-improving candidate —
`forced` + the recorded `registration_reason`. Phase C **reads those versions back**;
it does not re-derive provenance or alias logic.

**Publish (`src/ail/publish_lineage.py`).** For each registered agent
(`ail.registry`), reads its prompt versions through a version-level registry seam
(`LineageRegistryClient`: `search_prompt_versions` + `get_prompt_version_by_alias` —
`mlflow.genai` has no version listing, so this complements
`prompt_registry`'s name-level `search_registered_prompts`), parses the `ail.prompt.*`
tags, and writes one **unified UC table `agent_prompt_lineage`** keyed by
`(agent_name, version)`:

- `version`, `source`, `changed`, `gepa_best_val_score`, `gepa_num_candidates`,
  `holdout_evolved_savings_pct`, `holdout_seed_savings_pct`,
  `holdout_savings_delta_pct`, `candidate_artifact`, `suite_version`, `uri`,
  `registered_at`.
- `is_champion` — true iff the `champion`/`production` alias points at this version,
  resolved **authoritatively** from the registry (`get_prompt_version_by_alias`),
  never inferred from a version number, so a revert is reflected the next publish.
- `is_forced_non_improving` (+ `registration_reason`) — true iff the version was
  force-registered despite `changed=False` / no held-out improvement. Read from the
  `ail.prompt.forced` tag the promote step sets **only** on a forced non-improving
  candidate, so a legitimate seed (`changed=False` but never forced) is correctly
  *not* flagged.

Writes reuse `ail.publish`'s atomic staging→`REPLACE WHERE` swap, scoped by an
**`agent_name` predicate**: re-publishing one agent replaces that agent's whole slice
(so a version removed upstream is dropped) and never disturbs another agent's rows.

```
python -m ail.publish_lineage --registry config/agents.yaml \
    --warehouse-id <SQL_WAREHOUSE_ID> --profile dais-demo
```

**Query + view.** `config/queries/prompt_lineage.sql` is `SELECT`-only from
`agent_prompt_lineage`, newest version first (a row type is added to the appkit
analytics types). The app's `client/src/components/LineageTimeline.tsx` (sibling of
`VersionComparison.tsx`, wired into the selected agent) renders the version history
newest-first: each version's source, *what changed* (the proven held-out delta with
evolved-vs-seed savings), GEPA scores, the candidate-artifact label, and a clear
**CHAMPION** marker.

- **Audit honesty (the whole point).** A `is_forced_non_improving` version is flagged
  with a warning badge ("forced / not a proven improvement") + the recorded reason and
  is **never** styled like a genuine improvement — the honesty rule lives in
  `client/src/lib/lineage.ts` (`deltaTone` returns `warning` for any forced version
  regardless of the recorded delta; the green `positive` tone is reserved for a real
  gepa-evolved version whose held-out delta beat its seed) and is unit-tested.
- **Honest empty state.** With no registered version: *"No registered prompt versions
  yet — nothing has been promoted for this agent."*

**Revert CLI (`ail-revert` → `src/ail/jobs/revert_champion.py`).** Revert = re-point a
prompt's champion alias to a prior version via `set_prompt_alias`. It is a **guarded
CLI, not an in-app write button**:

```
ail-revert <agent_name> --to-version <n> [--profile ...] [--yes]
```

- **Fail-closed.** Refuses an unknown agent and an unknown target version (exit 2) —
  it never points the champion at a version that does not exist.
- **Explicit audit.** Prints what the champion **WAS** (version + uri) and what it
  **BECOMES** (version + uri) before acting.
- **Dry-run by default.** Prints the planned change and writes nothing unless `--yes`
  is passed; the alias write is the only side effect.
- **No auto-publish.** Re-pointing the alias does not refresh `agent_prompt_lineage` —
  the CLI reminds the operator to re-run `python -m ail.publish_lineage` so the app
  reflects the reverted champion. (Revert stays a guarded CLI in this lane — not an
  in-app write button.)

**Discipline.** Two-tier `SELECT`-only SQL; `prompt_registry` provenance/alias logic
reused (not reimplemented); fail-closed everywhere; typed, ruff/mypy clean,
`pytest -m 'not live'` green; app gates green. No live MLflow call on import or in
tests — the registry seam is injected/faked throughout.
