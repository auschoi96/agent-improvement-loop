# Deploy

**Status:** stable · **Bundles:** `ail-scheduled-publish` (root), `ail-self-optimizer` (`ail-self-optimizer/`)

This is the operations guide for deploying the framework so the SQL-warehouse
access is **turnkey** — deployers do not grant access by hand. It covers the two
deploy decisions baked into the bundles:

1. **Provide-or-create the warehouse** — accept an existing `warehouse_id`, or
   explicitly provision a small serverless SQL warehouse with
   `ail-bootstrap-grants --create-warehouse`.
2. **One framework service principal** — the app, the publish job, and every
   future scheduled job (scorers / L3 / MemAlign) run as a **single** SP, so a
   single `CAN_USE` grant covers everything.

Then the deploy: grants `CAN_USE` on the warehouse to that SP, and tags the
target MLflow experiment with the monitoring warehouse so the scheduled scorers
can actually read traces.

> [!IMPORTANT]
> **Granting `CAN_USE` and creating a warehouse require workspace authority.**
> See [§5 Admin prerequisite](#5-admin-prerequisite-read-this-first). Deploy as
> a workspace admin, or have an admin run the one-time grant/provision; thereafter
> the framework is turnkey for everyone else. This is the Databricks permission
> model — there is no bypass, and none should be added.

> [!IMPORTANT]
> `experiment_id`, `warehouse_id` / `sql_warehouse_id`, `catalog`, and `schema`
> have **no reusable default**. Pass them explicitly for your workspace; the
> bootstrap refuses empty, placeholder, or reference-workspace values before any
> deploy-time workspace changes.

---

## Prerequisites (tooling)

Install on the machine you deploy from (and, for the companion, on the host that
runs it):

- **Databricks CLI**, authenticated to the target workspace (`databricks auth login`).
- **`uv`** — the bundles build the Python wheel with `uv build --wheel`, so `uv` is
  a hard prerequisite for `databricks bundle deploy` (install:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`, or `pipx install uv`).
- **Node.js + npm** — the `ail-self-optimizer` app is an AppKit (TypeScript/React)
  app; its build shells out to `npm`. Required to deploy the app.
- **Python 3.11+ with the framework installed** for the `ail-*` CLIs and the local
  companion: `pip install -e .` for the core CLIs; add extras as needed —
  `.[claude]` (companion / executor Claude Agent SDK), `.[align]` (MemAlign
  auto-align), `.[l3]` (HALO / RLM review).
- **Workspace authority** for the one-time admin steps (create/manage warehouse,
  grant `CAN_USE`, create experiments, UC grants) — see
  [§5](#5-admin-prerequisite-read-this-first).

---

## 0. What deploys

Two independent Declarative Automation Bundles (DABs), both resolving host + auth
from the Databricks CLI profile at deploy time (nothing hardcoded):

| Bundle | Path | Contains | run-as |
|--------|------|----------|--------|
| `ail-scheduled-publish` | repo root `databricks.yml` + `resources/*.yml` | the scheduled L0 publish job, the scheduled `ail-continuous-rlm` review job, the scheduled auto-align job, **and** the on-demand `ail-apply-service` job (§7) | bundle-level `run_as` (§2) |
| `ail-self-optimizer` | `ail-self-optimizer/databricks.yml` | the L0 leaderboard App (incl. the approve/reject write-path) | the App's auto-provisioned SP (fixed by the platform) |

> **Retired:** the serverless `ail-optimization-cycle` job
> (`resources/optimization_cycle.job.yml`, entry point `ail-optimization-cycle`) was
> **removed**. The plan → propose → execute path now runs through the local companion
> (`python -m ail.companion` / `ail-companion-start`), which can use the Claude Agent
> SDK and does not clobber app-visible proposals from a competing scheduled job. The
> shared `ail.jobs.optimization_cycle` Python module remains as an importable library
> for companion/RLM seams such as feedback assembly, goal building, and proving helpers.

The **bootstrap** (`ail-bootstrap-grants`, §1/§3/§4) is a CLI an admin runs as part
of deploy — it uses an explicit warehouse id or an explicit `--create-warehouse`
opt-in, **ensures the app's tables exist
(empty)**, grants `CAN_USE`, and tags the experiment. It is the conditional glue the
bundles cannot express declaratively (see
[§6 Capability gaps](#6-dab--apps-capability-gaps-found)), and its table-ensure step
must run **before** the app bundle is deployed (§3 callout, §4).

---

## 1. Warehouse: provide, or auto-provision

The framework needs **one** SQL warehouse that the app, the publish job, the
scorers, and MLflow's monitoring job all use.

### Provide an existing warehouse (recommended)

Set the same warehouse id in both bundles:

```bash
# publish-job bundle (root)
databricks bundle deploy -t dais_demo \
  --var warehouse_id=<EXISTING_WAREHOUSE_ID> --profile dais-demo

# app bundle
cd ail-self-optimizer
databricks bundle deploy \
  --var sql_warehouse_id=<EXISTING_WAREHOUSE_ID> --profile dais-demo
```

`warehouse_id` (root bundle) and `sql_warehouse_id` (app bundle) **must be the
same warehouse** — that is what lets one grant cover the whole framework. Both
must be provided explicitly for each workspace.

### No warehouse yet: create one explicitly

If you do not have a warehouse, choose one of these paths:

- Run the bootstrap once with `--create-warehouse`. It finds or creates a small
  (`2X-Small`), serverless (`PRO`) warehouse with a 10-minute auto-stop by exact
  name (`ail-framework-serverless`) and prints the resolved `warehouse_id`.
- Create a SQL warehouse out-of-band, then use its id.

After you have the id, pass it explicitly to subsequent bootstrap runs and both
bundles:

```bash
# provision/resolve + print the id to reuse
ail-bootstrap-grants --experiment <EXPERIMENT_ID> --create-warehouse \
  --catalog <CATALOG> --schema <SCHEMA> --framework-sp-id <FRAMEWORK_SP_ID>

# then reuse the printed id
ail-bootstrap-grants --experiment <EXPERIMENT_ID> --warehouse-id <NEW_ID> \
  --catalog <CATALOG> --schema <SCHEMA> --framework-sp-id <FRAMEWORK_SP_ID>
```

Then deploy both bundles with `warehouse_id=<NEW_ID>` / `sql_warehouse_id=<NEW_ID>`
as above.

> Why a CLI and not a bundle-declared warehouse: DABs **does** support an
> `sql_warehouses` resource, but it cannot express *"create only if the deployer
> did not supply one"* — see [§6](#6-dab--apps-capability-gaps-found). The
> provide-or-explicitly-create branch therefore lives in the idempotent bootstrap.

---

## 2. One framework service principal

| Component | Identity | How it is set |
|-----------|----------|---------------|
| App | the App's **auto-provisioned** SP | fixed by the Apps platform — an app always runs as its own SP |
| Publish job + future jobs | `framework_sp_id` | bundle-level `run_as` (one knob for all jobs) |

The bundle exposes one knob, `run_as: ${var.job_run_as}`, that applies to
**every** job in the bundle — the publish job today, and any job added under
`resources/` tomorrow. It is driven by two variables:

- `framework_sp_id` — the application (client) id of the single framework SP;
  the production target defaults to the `ail-self-optimizer` App SP.
- `job_run_as` (complex) — defaults to `{user_name: ${workspace.current_user.userName}}`
  (the deploying identity), overridden to the SP by the `prod` target.

**Default target `dais_demo`** → jobs run as the deploying identity. Use this for
the admin verification deploy; no SP needed.

**Target `prod`** → jobs run as the App SP:

```bash
databricks bundle deploy -t prod --profile dais-demo
```

`dais_demo_sp` remains as a backward-compatible production-mode alias.

### Make it literally one SP: reuse the App's SP

The cleanest single-SP setup reuses the **App's** auto-provisioned SP as
`framework_sp_id`, because the App's SP cannot be reassigned and a job cannot
reference it until it exists. So the turnkey order is:

1. Run the **pre-deploy bootstrap** (§4 step 2) to provision/resolve the warehouse
   and ensure the query, request, and event tables. Then deploy the job bundle once
   as the admin/deployer so the App's declared job resources have concrete IDs.
2. Deploy the **app** with those job IDs. This creates the app, its SP, and (because
   the warehouse is declared with `permission: CAN_USE`) auto-grants warehouse use.
3. Read the App SP's application id:
   ```bash
   databricks apps get ail-self-optimizer -o json --profile dais-demo \
     | python3 -c "import sys,json; print(json.load(sys.stdin)['service_principal_client_id'])"
   ```
4. Run the bootstrap with that ID to merge warehouse and dedicated-schema grants,
   then deploy the job bundle with target `prod`. The app and every framework job
   now run as the same SP.

If you prefer a standalone SP (created once by an admin), pass its application id
as `framework_sp_id` instead — the bootstrap then grants it (and the App's own SP
keeps its auto-grant; both have `CAN_USE`).

---

## 3. Warehouse tables, the grant, and the monitoring tag (bootstrap)

`ail-bootstrap-grants` (module `ail.jobs.bootstrap_grants`) is **idempotent** and
does the following in one run:

1. **Resolve the warehouse** — use `--warehouse-id`, or, only when explicitly
   requested, find-or-create by name with `--create-warehouse` (§1).
2. **Ensure the app's tables exist (empty)** — create every UC table the deployed
   app's SQL queries read, using each writer module's **own** authoritative
   `_ddl()` `CREATE SCHEMA/TABLE IF NOT EXISTS` (module `ail.jobs.bootstrap_tables`;
   no schema is authored in the bootstrap). This is **load-bearing for the app
   build** — see the ordering callout below. On an **existing** table it then
   **additively reconciles columns**: any column in the writer's `_ddl()` that is
   missing on the live table is added via `ALTER TABLE ... ADD COLUMNS`. This is
   **additive-only and idempotent** — it never drops, renames, changes a column's
   type, or repopulates a table (no missing columns ⇒ no `ALTER`), and it **fails
   loud** (raising, changing nothing) if a live column's type genuinely conflicts
   with the DDL. So both a fresh deploy and an **upgrade** deploy that adds columns
   converge the table to the current schema before the app build.
3. **Merge the framework SP grants** — `CAN_USE` on the warehouse plus
   `USE_CATALOG`, `USE_SCHEMA`, `SELECT`, `MODIFY`, and `CREATE_TABLE` on the
   dedicated AIL catalog/schema. Existing privileges are preserved. Skipped if no
   SP is given.
4. **Tag the experiment** with `mlflow.monitoring.sqlWarehouseId = <warehouse>`
   (reusing `ail.compare.monitoring.configure_monitoring_warehouse`) so MLflow's
   monitoring job fetches the v4 Unity Catalog traces the scheduled scorers score.

```bash
# Run once, as a workspace admin, BEFORE deploying the app (see ordering below):
ail-bootstrap-grants \
  --experiment <EXPERIMENT_ID> \
  --warehouse-id <WAREHOUSE_ID> \
  --catalog <CATALOG> --schema <SCHEMA> \
  --framework-sp-id <FRAMEWORK_SP_ID> --profile dais-demo
# Use --create-warehouse instead of --warehouse-id only for the one-time
# provision/resolve path; it prints the warehouse id to reuse. Omit
# --framework-sp-id to skip the grant (e.g. on the pre-app run, before the App SP
# exists — see §4). --catalog/--schema are required workspace-specific values.
```

> [!WARNING]
> `--allow-reference-workspace` (or `AIL_ALLOW_REFERENCE=1`) is an owner-only
> escape hatch for re-deploying the live reference demo. You almost never want
> this: it bypasses only the known reference-workspace value check. Empty values,
> placeholders such as `REPLACE_ME`, and unresolved bundle references such as
> `${var.catalog}` remain fatal.

The App's own `CAN_USE` is **not** done here — it is granted natively by the Apps
platform from the `permission: CAN_USE` declaration in
`ail-self-optimizer/databricks.yml`.

> [!IMPORTANT]
> **The table-ensure must run before the app is deployed/started.** The app's
> build runs AppKit typegen (`appkit generate-types`), which performs a **live**
> `DESCRIBE QUERY` against the warehouse for **every** query in
> `ail-self-optimizer/config/queries/*.sql`. Several of those tables are created
> lazily on first write (e.g. `agent_proposed_actions`, `agent_prompt_lineage`,
> the `agent_version_*` tables), so on a **clean** workspace they do not exist yet.
> If typegen `DESCRIBE`s a missing table it fails with `TABLE_OR_VIEW_NOT_FOUND`,
> the app **build** fails, `bundle run app` fails, and any previously-running app
> goes **UNAVAILABLE**. Ensuring the empty tables first (step 2) makes typegen —
> and every runtime `SELECT` — resolve. This ordering is why the bootstrap is the
> **first** post-warehouse step in the sequence (§4), not a follow-up. The set of
> tables the bootstrap covers is drift-guarded in `tests/test_bootstrap_tables.py`
> against the actual `.sql` query files, so it cannot silently fall behind.

> **Upgrade deploys migrate columns automatically — no manual `ALTER`.** When a
> version adds columns to a table that already exists on the workspace — e.g. the
> `AGENT_TASK` fields (`change_plan`, `change_preview_diff`,
> `change_produced_change_ref`) or the verify-on-suite fields (`verify_requested`,
> `verify_status`, `verify_requested_by`, `verify_requested_at`, `verify_completed_at`,
> `verify_error`) on `agent_proposed_actions` — step 2's additive column
> reconciliation adds them to the already-created table (via `ALTER TABLE ... ADD
> COLUMNS`) **before** the app build's typegen runs. This closes the upgrade-path
> incident where a schema-additive redeploy over an existing workspace left the new
> columns missing → typegen's live `DESCRIBE QUERY` failed → the app build failed →
> the running app went **UNAVAILABLE**. The reconciliation is **additive-only** (never
> drops, renames, or changes a column's type) and **idempotent** (a re-run on an
> already-migrated table adds nothing); a genuine type conflict **fails loud**
> (raising, changing nothing) rather than mutating data — surface and resolve it
> manually if it ever occurs.

---

## 4. End-to-end turnkey sequence

**Ordering is load-bearing:** the bootstrap's warehouse + **table-ensure** + tag
must run before either bundle is deployed. The app build typegen `DESCRIBE`s every
query table, and the job bundle's table-update triggers reference the governed
`alignment_events` and `memory_events` Delta tables; both deployments can fail on
a clean workspace if those tables do not exist. The bootstrap is idempotent, so it
is run again after the App creates its SP to merge that SP's warehouse and schema
grants.

```bash
# 1. Validate the job bundle without changing remote resources.
databricks bundle validate -t dais_demo --profile dais-demo

# 2. BOOTSTRAP FIRST [ADMIN], from this checked-out revision: create/reconcile all
#    query, request, and event tables before a trigger or typegen references them.
uv run ail-bootstrap-grants --experiment <EXPERIMENT_ID> \
  --warehouse-id <WAREHOUSE_ID> \
  --catalog <CATALOG> --schema <SCHEMA> --profile dais-demo

# 3. Deploy the job bundle once as the admin/deployer. This creates the jobs whose
#    ids are declared as App resources. Record those ids from `bundle summary`.
databricks bundle deploy -t dais_demo \
  --var catalog=<CATALOG> --var schema=<SCHEMA> --profile dais-demo
databricks bundle summary -t dais_demo --profile dais-demo

# 4. Materialize the workspace-specific catalog/schema/path literals in app.yaml,
#    then deploy and start the App with every declared resource id. `app.yaml`
#    intentionally does not accept `${var.*}` interpolation.
cd ail-self-optimizer
databricks apps deploy -t default --profile dais-demo \
  --var sql_warehouse_id=<WAREHOUSE_ID> \
  --var apply_job_id=<APPLY_JOB_ID> \
  --var onboarding_job_id=<ONBOARDING_JOB_ID> \
  --var l0_job_id=<L0_JOB_ID> --var rlm_job_id=<RLM_JOB_ID> \
  --var judge_backfill_job_id=<JUDGE_BACKFILL_JOB_ID> \
  --var auto_align_job_id=<AUTO_ALIGN_JOB_ID> --var memory_job_id=<MEMORY_JOB_ID>
cd ..

# 5. Capture the App SP -> the single framework SP.
SP=$(databricks apps get ail-self-optimizer -o json --profile dais-demo \
       | python3 -c "import sys,json;print(json.load(sys.stdin)['service_principal_client_id'])")

# 6. Bootstrap again to merge warehouse + dedicated-schema grants for that SP.
uv run ail-bootstrap-grants --experiment <EXPERIMENT_ID> \
  --warehouse-id <WAREHOUSE_ID> --framework-sp-id "$SP" \
  --catalog <CATALOG> --schema <SCHEMA> --profile dais-demo

# 7. Redeploy the jobs in production mode under the App SP, then verify identity.
databricks bundle deploy -t prod --var framework_sp_id="$SP" \
  --var catalog=<CATALOG> --var schema=<SCHEMA> --profile dais-demo
uv run ail-validate-run-as --expected-sp "$SP" --profile dais-demo
```

> If you use a **standalone** framework SP (created once by an admin) instead of
> reusing the App SP, pass `--framework-sp-id <SP>` in step 2 and drop step 5's
> bootstrap re-run — the single pre-app bootstrap then does the warehouse, tables,
> grant, and tag in one shot.

> **`--var catalog=... --var schema=...` are REQUIRED on every `bundle deploy` above.**
> They wire `AIL_CATALOG` / `AIL_SCHEMA` into the jobs' and app's env; the
> approval->apply **write path** (prompt-registry, lineage, metric-view / asset
> creation) resolves the deployer's catalog from them and **fails closed with a loud
> error** if they are empty/placeholder -- so a fresh deploy can never silently write
> into the reference workspace. The write-path escape hatch (owner-only, for
> re-deploying the reference demo) is `AIL_ALLOW_REFERENCE_WORKSPACE=1` -- a
> **distinct** env var from the bootstrap's `AIL_ALLOW_REFERENCE` /
> `--allow-reference-workspace` (step 2 / §3).

---

## 5. Admin prerequisite (read this first)

Steps that **require workspace authority** (workspace admin, or `CAN_MANAGE` on
the warehouse, or the can-create-warehouse entitlement):

- creating the serverless warehouse (auto-provision path), and
- granting `CAN_USE` on the warehouse to the framework SP (the bootstrap grant,
  and the Apps-platform auto-grant during app deploy).

**A non-admin deployer's grant/provision step will fail** with a Databricks
permissions error. That is by design. Two ways to stay turnkey:

- **Deploy as an admin** — the whole sequence in §4 just works; or
- **Have an admin run the one-time grant/provision once.** After that, the
  warehouse exists and the SP is granted, so non-admins can deploy and run the
  jobs and app freely (they never need to grant again).

### Fallback: the exact one-time admin commands

If you would rather not run the bootstrap CLI, an admin can do the warehouse,
grant, and tag with the Databricks CLI below. **There is deliberately no hand-CLI
fallback for the table-ensure** (§3 step 2): hand-authoring `CREATE TABLE`
statements would duplicate — and inevitably drift from — the writer modules'
authoritative `_ddl()`. Run `ail-bootstrap-grants` for that step so the empty
tables always match the schema the writers populate (that is the whole point of
"no guessed schema"). The three hand-CLI steps:

```bash
# (a) create a small serverless warehouse (only if you are auto-provisioning)
databricks warehouses create --profile dais-demo --json '{
  "name": "ail-framework-serverless",
  "cluster_size": "2X-Small",
  "enable_serverless_compute": true,
  "warehouse_type": "PRO",
  "auto_stop_mins": 10,
  "max_num_clusters": 1
}'

# (b) grant CAN_USE on the warehouse to the framework SP (merge — keeps others)
databricks warehouses update-permissions <WAREHOUSE_ID> --profile dais-demo --json '{
  "access_control_list": [
    {"service_principal_name": "<FRAMEWORK_SP_ID>", "permission_level": "CAN_USE"}
  ]
}'

# (c) tag the experiment with the monitoring warehouse
databricks experiments set-experiment-tag <EXPERIMENT_ID> \
  mlflow.monitoring.sqlWarehouseId <WAREHOUSE_ID> --profile dais-demo
```

Setting the monitoring tag is **necessary but not sufficient**: without the
`CAN_USE` grant (b), the trace read fails with a permissions error — which is
exactly the v4-store access gap the live scoring lane is blocked on (see
`src/ail/compare/monitoring.py`).

---

## 6. DAB / Apps capability gaps found

Grounded in `databricks bundle schema` and the platform guide — stated plainly so
the design is honest about what is declarative and what is not:

- **`sql_warehouses` resource: supported.** DABs can declare a serverless
  warehouse (`enable_serverless_compute`, `cluster_size`, `auto_stop_mins`,
  `warehouse_type`) and even `permissions` on it.
- **Conditional creation: not supported.** There is no `count`/`if` — a declared
  resource is *always* created. So *"provide an existing warehouse OR create one"*
  cannot be a single declarative resource; it lives in the idempotent bootstrap
  (§1). The bootstrap's create spec mirrors what the DAB resource would declare.
- **Bundle-level `run_as`: supported.** One `run_as` covers all jobs — the
  single-SP lever for jobs (§2).
- **App SP is fixed.** An app always runs as its own auto-provisioned SP; it
  cannot be reassigned, and a job cannot reference that SP at author time (the id
  is unknown until the app exists). So "app + jobs literally share one SP in a
  single declarative pass" is **not** achievable — the turnkey path is "deploy
  app → reuse its SP as `framework_sp_id` for the jobs + bootstrap grant" (§2).
- **App warehouse grant: native.** Declaring the warehouse on the app with
  `permission: CAN_USE` auto-grants `CAN_USE` to the app SP on deploy — no
  bootstrap needed for the app itself.

---

## 7. The apply job (`ail-apply-service`) — deployed approve/reject transport

The app's authenticated Approve/Reject write-path (lane 3b,
`docs/LOOP_CONTROLLER.md`) runs the framework's **Python** apply engine. In local
dev / a self-hosted image the Node app spawns `python -m ail.loop.apply_service`
directly. The **deployed Databricks App image is Node-only** (the `ail` wheel is not
importable there), so the app instead **triggers a Databricks Job** that runs the
same engine. That job is `ail-apply-service` (`resources/apply_service.job.yml`),
deployed by the **root** `ail-scheduled-publish` bundle alongside the publish job.

It is a serverless `python_wheel_task` (entry point `ail-apply-job`), **on-demand
only** (no schedule — one run per human decision), `max_concurrent_runs: 1` with
`queue.enabled: true`, and it inherits the bundle-level `run_as` (§2) — so it runs
as the **same** single framework SP as the publish job and the app. The decision is
passed as **job parameters** at trigger time (never hardcoded in the bundle); the
job runs `run_decision` (re-checking proof + gate, fail-closed) and writes the real
`ApplyServiceResult` to the `agent_apply_results` UC Delta table
(`${var.catalog}.${var.schema}`), keyed by `(proposal_id, decided_at)`, which the
app reads back to render the outcome. A non-pending proposal (already applied /
superseded) is **refused**, so a duplicated/retried trigger never double-applies.

### What the operator must set

1. **Deploy the root bundle** (§4) — this creates `ail-apply-service`. Capture its
   numeric job id:
   ```bash
   databricks jobs list --profile dais-demo -o json \
     | python3 -c "import sys,json;print(next(j['job_id'] for j in json.load(sys.stdin) if j['settings']['name']=='ail-apply-service'))"
   ```
2. **Grant the app SP `CAN_MANAGE_RUN` on the job** so the app can trigger it and
   read run status. This is **handled automatically by step 3**: the app bundle
   declares the job as an `apply-job` app resource with `permission: CAN_MANAGE_RUN`
   (`ail-self-optimizer/databricks.yml`), so the Apps platform AUTO-GRANTS
   `CAN_MANAGE_RUN` to the app's service principal at deploy — the same mechanism
   that auto-grants `CAN_USE` on the `sql-warehouse` resource. No manual grant is
   needed. (When the app and jobs share one SP per §2, that SP already owns the job,
   so the grant is a no-op either way.) As a fallback — e.g. the deploying identity
   lacks manage rights on the job — set it explicitly:
   ```bash
   databricks jobs update-permissions <APPLY_JOB_ID> --profile dais-demo --json '{
     "access_control_list": [
       {"service_principal_name": "<APP_SP_ID>", "permission_level": "CAN_MANAGE_RUN"}
     ]
   }'
   ```
   The app SP also needs `SELECT` on `agent_apply_results` — covered by the same
   framework schema access it already uses for its two-tier reads.
3. **Deploy the app with the job id as a bundle variable.** The transport is already
   wired in `ail-self-optimizer/app.yaml`: `AIL_APPLY_TRANSPORT: job` is a committed
   literal, and `AIL_APPLY_JOB_ID` is injected from the `apply-job` app resource via
   `valueFrom: apply-job` (mirroring `DATABRICKS_WAREHOUSE_ID` <- `sql-warehouse`).
   That resource's `id` is the deploy-time bundle variable `apply_job_id`, empty in
   `main` so the bundle stays workspace-agnostic. Supply the workspace's numeric job
   id (from step 1) at deploy:
   ```bash
   # from ail-self-optimizer/
   databricks bundle deploy --profile dais-demo --var apply_job_id=<APPLY_JOB_ID>
   databricks bundle run app --profile dais-demo   # start/refresh the app
   ```
   Equivalently set `BUNDLE_VAR_apply_job_id=<APPLY_JOB_ID>` in the environment, or
   pin it in a target's `variables:` block. Do **not** hardcode the id in `app.yaml`
   or `databricks.yml` — that would tie `main` to one workspace.

   `DATABRICKS_WAREHOUSE_ID` (already injected from the app's `sql-warehouse`
   resource) is reused to read the result row back. Set `AIL_APPLY_CATALOG` /
   `AIL_APPLY_SCHEMA` to the same workspace-specific `<CATALOG>` / `<SCHEMA>`
   used for the bundles. Optional timing overrides: `AIL_APPLY_JOB_TIMEOUT_MS`
   (default 300000), `AIL_APPLY_JOB_POLL_MS` (default 3000).

Without `AIL_APPLY_TRANSPORT=job` / `AIL_APPLY_JOB_ID`, the app falls back to the
subprocess bridge — correct for local dev, but on the Node-only deployed image that
bridge cannot run. `AIL_APPLY_TRANSPORT: job` is committed in `app.yaml`, so the
only per-workspace step is supplying `apply_job_id` at deploy (a missing/empty id
leaves the deployed app selecting the Job transport but with no job to trigger — it
fails closed rather than silently applying via a bridge it can't run).

### The in-app labeling transport (deployed, no job)

The app's authenticated labeling panel (L4, `docs/LABELING_UI.md`) has the **same**
Node-only problem as approve/reject — but labeling is **rapid-fire** (one write per
grade), so a per-grade Job trigger's startup latency would be unusable. It therefore
uses a **Node-native MLflow REST** transport instead: the labeling routes read the
registered judges + progress and write each HUMAN label as an MLflow assessment
(name-matched to the judge) by calling the Databricks-managed MLflow assessments REST API
directly, under the app's service principal. **No extra job or resource is required.**

What the operator sets: **nothing per-workspace.** `AIL_LABELING_TRANSPORT: rest` is a
committed literal in `app.yaml` (the deployed image can't spawn the Python engine), and
the transport reuses the app's existing `DATABRICKS_WAREHOUSE_ID` (from the `sql-warehouse`
resource) as the SQL warehouse for the v4 UC tracing endpoints. The app SP needs read
access to the experiment's traces and write access to its assessments — the same
experiment access the framework already relies on. Without `AIL_LABELING_TRANSPORT=rest`
the app falls back to the subprocess bridge (correct for local dev only).

`AIL_LABEL_FLOOR` is materialized in `app.yaml` from the engine value
(`python -c "import ail.readiness as r; print(r.ReadinessThresholds().quality_min_labels)"`)
so the deployed panel can show the `N / floor` target without importing Python.
When that engine threshold changes, regenerate the deployment artifact in the same
change; runtime configuration validation rejects a missing, unresolved, or invalid
value rather than silently displaying a fabricated target.

Fail-closed: a grade is only reported "labeled" when the REST write returns an
`assessment_id`. If the transport can't confirm a write (missing warehouse, unresolvable
trace, scorer-list failure, unknown judge, write error), the panel shows an honest error
directing the user to the MLflow Traces UI — never a fabricated success.

---

## 8. Local companion plan → propose → execute path

The serverless `ail-optimization-cycle` job is retired. Do **not** deploy or run a
scheduled optimization-cycle job; fresh root-bundle deploys no longer create one. The
proposal path is now the local companion (`python -m ail.companion`), started by the
deployer on compute that has the Claude Agent SDK and Databricks credentials.

The companion path:

1. **plans** from fresh evidence (L0 metrics, trusted judges, and RLM feedback) using
   deterministic decision rules plus the LLM planner;
2. **publishes** PENDING proposals to `agent_proposed_actions` for app review;
3. **executes** only after approval, through the companion/executor path rather than a
   serverless scheduled job; and
4. **proves** on demand when the user requests suite-backed verification.

Use `ail-companion-start` for the durable one-command flow, or run the manual command
shown in §9:

```bash
python -m ail.companion poll --experiment <EXPERIMENT_ID> --catalog <CATALOG> --schema <SCHEMA>
```

The importable `ail.jobs.optimization_cycle` module intentionally remains because the
companion and scheduled RLM job reuse its shared seams (`build_feedback_bundle`,
`_default_prover`, `_build_goal`, and related helpers). Only the serverless job
resource, bundle variables, and console-script entry point were retired.

---

## 9. Turn on the loop (evaluate + optimize)

§0–§8 stand up the **infrastructure**. This section turns on the **loop** so the
framework actually evaluates traces and proposes optimizations. Run these against
the same `<EXPERIMENT_ID>` you deployed with.

> Nothing here surfaces an improvement until the corpus reaches the readiness
> gates (below). The app's readiness panel and `ail-readiness <EXPERIMENT_ID>`
> report exactly how far you are — this is by design, not a failure.

1. **Register the L2 judges** so they auto-score new traces:
   ```bash
   ail-register-scorers --experiment-id <EXPERIMENT_ID>
   ```
   Registers the built-in scheduled scorers (correctness, modularity, groundedness,
   token_efficiency) at 0.1 sampling. They score on the monitoring cadence — the
   `mlflow.monitoring.sqlWarehouseId` tag set by bootstrap (§3) is what lets them
   read traces. Judges start **distrusted** until aligned (step 4).

2. **Connect your agent's tracing** to the experiment so traces flow in — native
   autolog (Claude Code, OpenAI, LangChain, …) or OTEL import. See
   [`CONNECT_YOUR_AGENT.md`](CONNECT_YOUR_AGENT.md). No traces → nothing to evaluate.

3. **(quality goals only) Author a judge** for a dimension you care about — describe
   it in natural language and the tool creates an alignable `{{trace}}` judge with a
   name-matched label schema:
   ```bash
   ail-author-judge --experiment-id <EXPERIMENT_ID> --description "<what good looks like>"
   ```
   Token/cost goals need no judge (they are deterministic L0 metrics).

4. **(quality goals only) Label, then align.** Label ~20 traces along the judge's
   dimension — in the app's labeling panel or the MLflow UI; the **label name must
   match the judge name**. Then auto-align so the judge becomes *trusted* as labels
   accrue (re-runs safely; rolls a regression back):
   ```bash
   ail-auto-align --experiment <EXPERIMENT_ID> --judges <judge_name>
   ```
   (Also deployable as a scheduled job via the bundle.)

5. **Run the local companion** on a host that has the Claude Agent SDK (`.[claude]`).
   It plans (evidence → proposals) and, on your approval, executes and optionally
   proves. See [`COMPANION.md`](COMPANION.md). One command does everything —
   it mints a fresh static token from your CLI profile, exports the auth +
   catalog/schema, and runs the poll loop durably (re-minting each cycle so a long
   run survives OAuth token expiry):
   ```bash
   ail-companion-start --profile <PROFILE> --experiment <EXPERIMENT_ID> --catalog <CATALOG> --schema <SCHEMA>
   ```
   `--profile` is used only to mint the token; it is never passed to the companion
   (which requires a static token and refuses OAuth). Add `--warehouse-id <ID>` to
   forward the monitoring warehouse.

   Manual/advanced path (export `DATABRICKS_HOST`/`DATABRICKS_TOKEN` + `AIL_CATALOG`/
   `AIL_SCHEMA` yourself first): `python -m ail.companion poll --experiment <EXPERIMENT_ID> --catalog <CATALOG> --schema <SCHEMA>`.

6. **(optional) Build a frozen suite for on-demand verification (Tier 2).** Evidence
   (judge + RLM + L0) is enough to approve on — but for a high-blast-radius change you
   can *verify* it first by running the candidate vs. baseline on a suite of **your
   own** representative tasks. Scaffold candidate tasks from real traces, author a
   pass/fail check per task (a human step — no fabrication), then freeze (the freeze
   refuses to seal unless every task has a real check):
   ```bash
   ail-suite-scaffold --experiment <EXPERIMENT_ID> --out eval/suites/my_suite   # draft tasks from traces
   # …author eval/suites/my_suite/checks.yaml — one real pass/fail check per task_id…
   ail-suite-freeze eval/suites/my_suite                                        # fail-closed seal
   ```
   Then "Verify on my suite" on a proposal in the app (or `python -m ail.companion
   prove …`) runs the frozen suite and attaches the measured delta as **added
   evidence** — it never auto-approves. Skip this entirely for token/cost goals where
   the deterministic L0 delta already measures the win.

7. **Review + approve** in the app. Each proposal carries its evidence (judge + RLM
   + L0, plus any Tier-2 verify delta); you approve; the companion applies it —
   recorded in the lineage timeline and revertible (`ail-revert`).

### Readiness gates — what unlocks when

| Corpus | Unlocks |
|---|---|
| ~10 traces | L0 baseline + RLM / HALO diagnosis |
| ~20 labels (per judged dimension) | a trusted, MemAlign-aligned judge |
| ~50 traces | statistical power to *prove* a token/cost win (leaderboard amber → green) |

Until a gate is met, the readiness wall reports "collecting / not ready" and the
loop proposes nothing for that goal — honest, not broken.
