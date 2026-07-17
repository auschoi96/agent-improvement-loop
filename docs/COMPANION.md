# Local Companion

`python -m ail.companion` is the consolidated deployer-run entrypoint for the local
companion. It runs outside Databricks serverless, on the deployer's machine, where the
Claude Agent SDK or Codex CLI can inspect and edit local workspaces.

Product flow:

1. The app and framework write proposals, approvals, and decision audit rows in Unity
   Catalog, especially `agent_proposed_actions`.
2. The companion polls Unity Catalog for work.
3. The companion plans, previews, commits, or proves locally by delegating to the
   existing AIL modules.
4. The companion writes preview diffs, produced change-set refs, commit records, and
   proposal status updates back to Unity Catalog for the app to display.

## Authentication

The companion requires static Databricks auth pinned to the target workspace host. Do
not use a Databricks CLI `--profile` OAuth login; the long-lived local process refuses
it because token refresh can fail from a background runner.

```bash
export DATABRICKS_HOST=https://<workspace-host>
export DATABRICKS_TOKEN=<pat-or-static-token>
export AIL_WAREHOUSE_ID=<sql-warehouse-id>
export AIL_SNAPSHOT_VOLUME=/Volumes/<catalog>/<schema>/<volume>/ail_snapshots
```

Secret-scope token resolution is also accepted when the existing companion modules can
resolve it:

```bash
python -m ail.companion poll \
  --token-secret-scope <scope> \
  --token-secret-key <key> \
  --host https://<workspace-host> \
  --warehouse-id <sql-warehouse-id> \
  --volume-root /Volumes/<catalog>/<schema>/<volume>/ail_snapshots
```

## Poll Loop

`poll` is the normal local runner. It is bounded by `--max-iterations`; use a process
manager if you want it restarted forever.

```bash
python -m ail.companion poll \
  --agent claude_code \
  --registry config/agents.yaml \
  --warehouse-id "$AIL_WAREHOUSE_ID" \
  --volume-root "$AIL_SNAPSHOT_VOLUME" \
  --max-iterations 20 \
  --interval-seconds 30
```

Each tick delegates to `ail.jobs.agent_executor.run`, which:

- previews `PENDING` `AGENT_TASK` proposals without a stored preview by running the
  agent in a sandbox copy of the target workspace;
- writes the real preview diff and produced change-set ref back to
  `agent_proposed_actions`;
- commits `APPROVED` `AGENT_TASK` proposals by applying the stored change-set; it does
  not re-run the agent at commit time;
- applies `APPROVED` `GEPA_PROMPT` proposals only when they carry the immutable local
  apply spec created by the GEPA job. It downloads the reviewed MLflow artifact,
  verifies the project-relative target plus seed/candidate hashes and exact diff,
  snapshots the original, rewrites atomically, and runs the registered validation
  command. A conflict applies nothing; failed validation restores the original;
- scopes GEPA polling and terminal updates by both `agent_name` and
  `experiment_id`, so a selected agent cannot consume a similarly named proposal
  from another experiment;
- records the commit in `agent_executor_commits` and advances the proposal status.

Optional planning cadence:

```bash
python -m ail.companion poll \
  --warehouse-id "$AIL_WAREHOUSE_ID" \
  --volume-root "$AIL_SNAPSHOT_VOLUME" \
  --max-iterations 20 \
  --plan-every 4 \
  --experiment <mlflow-experiment-id> \
  --goal-confirmed true
```

`--plan-every 4` runs the planner on iterations 1, 5, 9, and so on, then runs the
executor on every iteration.

## Subcommands

### `plan`

Runs one evidence-first planning pass by delegating to
`ail.jobs.companion_planner.run`.

```bash
python -m ail.companion plan \
  --agent claude_code \
  --experiment <mlflow-experiment-id> \
  --warehouse-id "$AIL_WAREHOUSE_ID" \
  --goal-confirmed true
```

This reads the agent's judge, RLM, and L0 evidence, runs Lane A deterministic rules and
the Lane B LLM planner, gates only on readiness and judge trust, and publishes
`PENDING` evidence-backed proposals to `agent_proposed_actions`. It does not prove or
apply changes.

### `execute`

Runs one executor pass by delegating to `ail.jobs.agent_executor.run`.

```bash
python -m ail.companion execute \
  --agent claude_code \
  --registry config/agents.yaml \
  --warehouse-id "$AIL_WAREHOUSE_ID" \
  --volume-root "$AIL_SNAPSHOT_VOLUME"
```

This is the same one-pass behavior used by `poll`: preview pending `AGENT_TASK`
proposals, commit approved agent tasks, and finish approved GEPA local rewrites.

### `prove`

Runs opt-in Tier-2 frozen-suite verification by delegating to
`ail.optimize.run_phase2_comparison`.

```bash
python -m ail.companion prove \
  --suite-version phase2-mini \
  --run-plan run_plan.yaml \
  --experiment /Shared/my-agent-experiment \
  --output artifacts/phase2_companion.json
```

The prover runs the baseline and candidate arms over the user's frozen suite and writes
a Phase-2 artifact with PROMOTE/BLOCK/ERRORED outcomes. It is on-demand evidence for a
candidate or proposal; it is not part of the automatic planning gate.

`prove` does not inherit `AIL_WAREHOUSE_ID`. To attach monitoring warehouse
provenance, pass both flags explicitly:

```bash
python -m ail.companion prove \
  --suite-version phase2-mini \
  --run-plan run_plan.yaml \
  --experiment /Shared/my-agent-experiment \
  --experiment-id <mlflow-experiment-id> \
  --warehouse-id "$AIL_WAREHOUSE_ID" \
  --output artifacts/phase2_companion.json
```

Exit code `0` means every task completed without BLOCK or ERRORED outcomes. Exit code
`2` means at least one task BLOCKED or ERRORED; inspect the written artifact for the
per-task result.

### `run`

`run` is an alias for `poll`.

```bash
python -m ail.companion run --max-iterations 1
```

## Existing Entry Points

The legacy console scripts remain available and delegate to the same underlying code:

- `ail-companion-planner` -> `ail.jobs.companion_planner.main`
- `ail-agent-executor` -> `ail.jobs.agent_executor.main`

The consolidated entrypoint does not replace those scripts; it gives deployers one
documented local command surface for planning, execution, proving, and bounded polling.
