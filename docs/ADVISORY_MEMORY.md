# Advisory memory (distiller)

The **advisory-memory distiller** turns recent agent-evaluation feedback into short
*memory guideline* rows in a governed Unity Catalog Delta table. It is the **write
/ system-of-record** half only — the Lakebase sync and the read / injection side
that feeds guidelines back to the agent are a separate, out-of-scope system.

It mirrors the pattern in
[`auschoi96/dc-assistant-data-review-agent`](https://github.com/auschoi96/dc-assistant-data-review-agent):
a serverless Databricks Job drives the **Claude Agent SDK** `query()` loop
in-process, authenticates to Databricks FMAPI with **zero secrets**, and writes via
a custom `@tool` that INSERTs through the SQL Statement API.

## What one run does

`resources/memory_distiller.job.yml` → `src/ail/jobs/memory_distiller.py` →
`ail.memory.distiller.run_memory_distiller`:

1. **Auth** — resolve the run-as identity into `DATABRICKS_HOST`/`TOKEN` (reusing
   `ail.jobs.publish_job.resolve_job_auth`: env → secret scope → mint a short-lived
   OAuth bearer). The same bearer becomes the FMAPI key.
2. **Watermark** — read the last-processed timestamp for this `experiment:cohort`
   scope from `agent_memory_watermark`.
3. **Read feedback** — one SQL `SELECT` on the trace store's OTEL annotation table
   returns both signal families created since the watermark:
   - RLM/HALO `rlm_*` assessments, and
   - the L2 MLflow-judge assessments `correctness`, `modularity`, `groundedness`,
     `token_efficiency`.
   Each row carries `name`, `target_id` (the trace id), `value` (score), `comment`
   (the rationale — the substance distilled), and `created_at` (the watermark).
   **Fail-closed:** an empty window writes nothing.
4. **Resolve the provenance wall** — load the frozen Task-Suite (and, if configured,
   Human-Anchor) trace-id sets. **Fail-closed:** if the Task Suite can't load, the
   run writes nothing.
5. **Distill** — the Claude Agent SDK `query()` loop reads the feedback and calls
   the `submit_memory` tool with candidate guideline rows.
6. **Validate → wall → write** — inside `submit_memory`, each candidate is validated
   (non-empty guideline, 0–1 score, known signal, ≥1 `source_trace_id`), then the
   **provenance wall** drops any row whose `source_trace_ids` touch a frozen pool,
   and the survivors are INSERTed (escaped, never raw-interpolated).
7. **Advance the watermark** to the newest assessment processed — so a re-run over
   the same window is a no-op.

## The read mechanism (verified live)

Assessments are queryable two ways: `mlflow.search_traces(...)`
(`trace.info.assessments`) or SQL on the UC annotation table. This job uses **SQL on
`<catalog>.mlflow_traces.<prefix>_otel_annotations`** (on dais-demo:
`austin_choi_omni_agent_catalog.mlflow_traces.cc_otel_annotations`). Verified against
the live workspace: a single `SELECT` there returns **both** RLM and judge FEEDBACK
with `value`, the rationale `comment`, and a `created_at` that is a clean idempotency
watermark — which `search_traces` (whole-trace pulls, no per-assessment watermark
column) does not give as cheaply. The trace store is implicitly experiment-scoped by
its `<prefix>_` convention, so the annotations-table var names exactly the
experiment's harness store. If neither path yields assessments the run writes nothing
— it never fabricates memory.

## The provenance wall (load-bearing)

`ail.memory.provenance` — the eval set must never seed memory. L2 judges score
Task-Suite traces too (e.g. suite trace `bdb3b11e597555cda869ed7ab5b123dd` carries a
`token_efficiency` score), so without the wall those would leak straight into the
agent's memory. `partition_rows` drops any candidate whose `source_trace_ids` touch
the frozen **Task-Suite** or **Human-Anchor** pools, and re-proves the kept set
disjoint with `ail.pools.assert_pools_disjoint` (the same guard the loop controller
uses) so a regression fails closed instead of leaking. Matching is exact-id **or** a
shared ≥12-char prefix, because the frozen suite stores some ids truncated. Dropped
rows are recorded with a reason, never written.

The frozen Task Suite is bundled into the wheel (`pyproject.toml`
`force-include: eval/task_suite`) so the serverless Job — where `eval/` is not on disk
— can still resolve the reserved set. The Human-Anchor pool is included when
`memory_groundtruth_root` points at a ground-truth store; when unset it is empty
(honest for a workspace with no promoted anchor cases) and the Task-Suite wall is
still fully enforced.

## The table

`ail.memory.schema._ddl` defines `agent_memory` (and its `agent_memory_watermark`
sibling) and is registered with `ail.jobs.bootstrap_tables._DDL_PRODUCERS`, so both
are created and additively column-migrated by the existing bootstrap — no hand-DDL.
They are **framework** tables (`FRAMEWORK_TABLES`), not app-read tables, so they are
deliberately absent from `APP_QUERY_TABLES`.

`austin_choi_omni_agent_catalog.agent_improvement_loop.agent_memory`:

| column | type | notes |
|---|---|---|
| `memory_id` | STRING | uuid per row |
| `cohort` | STRING | e.g. `claude_code` |
| `category` | STRING | e.g. `token_efficiency` |
| `guideline_text` | STRING | one actionable guideline |
| `score` | DOUBLE | 0–1 confidence |
| `source_trace_ids` | ARRAY<STRING> | provenance the wall checks (≥1) |
| `source_signal` | STRING | `rlm` or `judge:<name>` |
| `created_at` | STRING | ISO-8601 UTC |
| `embedding` | ARRAY<FLOAT> | nullable; populated by the separate retrieval side |

## Configuration (bundle vars, fail-closed empty defaults)

Reuses `catalog`, `schema`, `experiment_id`, `warehouse_id`, `agent_name` (cohort),
`schedule_timezone`, `token_secret_scope`/`token_secret_key`. Adds:

| var | default | meaning |
|---|---|---|
| `memory_annotations_table` | `''` | FQ OTEL annotations table (required; no hardcoded id) |
| `memory_distiller_model` | `databricks-claude-opus-4-6` | FMAPI model (verified served on dais-demo) |
| `memory_distiller_cron` | `0 30 8 * * ?` | DAILY 08:30 UTC |
| `memory_distiller_pause_status` | `PAUSED` | shipped dormant — unpause explicitly at deploy |
| `memory_max_turns` | `30` | bounds one distill run |
| `memory_max_assessments` | `200` | bounds the per-run token budget |
| `memory_task_suite_version` | `v1` | frozen suite that seeds the wall |
| `memory_groundtruth_root` | `''` | optional Human-Anchor store path |

## Deploy

The job is **PAUSED** by default (the schedule-footgun lesson) — deploy, then unpause
explicitly. It is **not** deployed live by this change.

```bash
databricks bundle deploy -t dais_demo \
  --var experiment_id=660599403165942 \
  --var warehouse_id=<WAREHOUSE_ID> \
  --var catalog=austin_choi_omni_agent_catalog \
  --var schema=agent_improvement_loop \
  --var memory_annotations_table=austin_choi_omni_agent_catalog.mlflow_traces.cc_otel_annotations \
  -p dais-demo
# then, when ready:
#   --var memory_distiller_pause_status=UNPAUSED
```

## Honest caveats

- **Auth lifetime.** The OAuth bearer is minted once at startup with no refresh
  (as in the reference). Each firing is bounded (`memory_max_turns` +
  `memory_max_assessments`) to stay well under the token lifetime; a window big
  enough to risk a >~50 min run should be split across firings (or wire a
  token-refresh before lifting the caps).
- **Human-Anchor coverage.** Without `memory_groundtruth_root` the Human-Anchor pool
  is treated as empty. That is correct today (no promoted anchor cases) but must be
  wired if/when the anchor is populated, or the wall won't protect it.
