# agent-improvement-loop

A reusable, agent-agnostic **self-improvement loop** for LLM agents (coding agents and other deployments). State a goal in natural language — token efficiency, coding accuracy, cost reduction — and the system measures the agent against a **frozen, human-anchored evaluation harness**, diagnoses the dominant waste or failure mode, proposes an intervention (prompt/skill optimization via GEPA, or building a helper asset such as a metric view + tool, a pipeline, or a semantic layer), evaluates the candidate against the original on a held-out task suite, and ships only what beats the goal metric without regressing guardrails.

The design's load-bearing principle: **the optimizer is never allowed to train against the evaluation set, and the judge is aligned on a separate cadence from agent optimization.** This is what separates real improvement from a dashboard that says "improved" while quality stalls (the co-adaptation trap that every reference loop we surveyed omits).

## Status

Greenfield. The trace-ingestion seam — `src/ail/ingest/` (the `TraceSource` /
`AgentAdapter` interfaces, the Databricks-managed MLflow source, and the Claude
Code adapter) — is **original clean-room work**, written only against this
repo's own interfaces/tests and the public docs/source of `mlflow`,
`databricks-sdk`, and `claude-agent-sdk`. See [`PROVENANCE.md`](PROVENANCE.md).

Other phases plan to draw proven pieces from:

- **`databricks-solutions/ai-dev-kit`** (`.test/src/skill_test/`) — the optimization spine: GEPA loop, MLflow `make_judge`, MemAlign `judge.align()`, GRP (Generate-Review-Promote) ground-truth pipeline. *Cross-vendor verified from source (Claude + GPT-5).*
- **`databricks-field-eng/skillforge`** — ground-truth methodology (`/forge` Designer⇄Critic case design) and the `GroundTruthV5` schema contract.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`docs/MILESTONE-1.md`](docs/MILESTONE-1.md) for the current build plan. Provenance and license reconciliation are tracked in [`PROVENANCE.md`](PROVENANCE.md).

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

## Cohorts (segment one experiment by tag)

`ail.cohorts` adds first-class **cohorts** — named, tag-defined slices of one
experiment's traces — so a single experiment can hold several agents or
deployments and still be measured apart. Cohorts respect the **user's own MLflow
UI tags** (any key); an `ail.agent` / `ail.cohort` convention is offered but not
required. Tag-aware ingestion (`MLflowTraceSource.fetch_cohort_traces`) and
per-cohort L0 metrics (`ail.metrics.compute_cohort_l0`) are strictly **additive**
over the existing read/metrics surfaces. Documented in
[`docs/COHORTS.md`](docs/COHORTS.md).

## Codex tracing (onboard a second agent)

`ail.ingest.adapters.codex` onboards the codex-native CLI harness alongside
Claude Code: `enable_codex_tracing` configures MLflow's native `@mlflow/codex`
notify hook, and `normalize_codex_rollout` / `CodexAdapter` capture Codex
rollout transcripts as `NormalizedTrace`s tagged `ail.agent=codex`. Both are
additive (import-only over the shared ingest seam). Setup, the Omnigent/GPT‑5
worker specifics, live-verify steps, and a Pi-feasibility note are documented in
[`docs/CONNECT_CODEX.md`](docs/CONNECT_CODEX.md).

## Scheduled refresh (living dashboard)

`ail.publish` (Tier A) computes the L0 contract via `ail.metrics` and writes the
three UC Delta tables the leaderboard app reads (`l0_session_metrics`,
`l0_corpus_summary`, `l0_diagnosis`). A Databricks Asset Bundle at the repo root
([`databricks.yml`](databricks.yml) + [`resources/l0_publish.job.yml`](resources/l0_publish.job.yml))
runs that step on a **schedule** so the tables track new traces as they arrive —
turning the dashboard from a one-time snapshot into a living view. No metric
logic lives in the bundle: the job's wheel-task entrypoint
([`ail.jobs.publish_job`](src/ail/jobs/publish_job.py)) only resolves auth and
delegates to `ail.publish.publish`, which keeps the write atomic and idempotent
(`INSERT … REPLACE WHERE experiment_id`).

- **Compute:** serverless. The `ail` wheel (built locally by `uv` at deploy) is
  installed into the run environment.
- **Schedule:** daily at 07:00 UTC by default; the cron is the `publish_cron`
  bundle variable (`schedule_pause_status` controls live-vs-dormant).

### Deploy & run

```bash
databricks bundle validate --strict --profile dais-demo
databricks bundle deploy           --profile dais-demo   # builds the wheel, creates the Job
databricks bundle run   l0_publish --profile dais-demo   # trigger one run now
```

Override any default without editing files, e.g. a 6-hour cadence and a
different warehouse:

```bash
databricks bundle deploy --profile dais-demo \
  --var publish_cron='0 0 */6 * * ?' --var warehouse_id=<id>
```

### Auth (and why it is resolved at runtime)

The reference experiment is UC-table-backed, read through MLflow's v4 trace REST
store, which **rejects profile-managed OAuth** for span reads — it needs an
explicit bearer (`DATABRICKS_HOST` + `DATABRICKS_TOKEN`). A serverless Job cannot
have those injected from the bundle (the serverless environment spec has no
env-var field), so the bearer is resolved at runtime by `ail.jobs.publish_job`:

1. **Pre-set env** — if `DATABRICKS_HOST`/`DATABRICKS_TOKEN` are already set
   (local, CI), use them.
2. **Secret scope** — if `--token-secret-scope`/`--token-secret-key` are given
   (the `token_secret_scope`/`token_secret_key` bundle variables), read the
   token from a Databricks **secret scope**. This is the production-hardened
   service-principal path: store the SP's token in a scope, grant the SP `READ`
   on it, and point the Job there — nothing sensitive is committed or passed as a
   Job parameter.
3. **Mint (default)** — otherwise mint a short-lived OAuth bearer from the Job's
   own run-as identity (`Config.authenticate()`) and pass it explicitly. This
   sidesteps the v4 OAuth bug with no stored credential and works the same
   whether the Job runs as a user or a service principal.

`publish` also exports the SQL warehouse id as `MLFLOW_TRACING_SQL_WAREHOUSE_ID`
(the v4 store reads UC-table traces through a warehouse).

**Run-as:** the Job runs as the deploying identity by default (so the
verification run has the catalog/warehouse grants). For production, run it as a
service principal that has been granted the data privileges — uncomment the
`run_as.service_principal_name` block in `resources/l0_publish.job.yml`.

## Reference deployment

- Workspace: `e2-demo-field-eng` (dais-demo profile) / `fevm-austin-choi-omni-agent`
- MLflow experiment: `660599403165942`
- Trace tables: `austin_choi_omni_agent_catalog.mlflow_traces.*`
- Sandbox schema (GRP test-code execution): `austin_choi_omni_agent_catalog.agent_improvement_loop`
