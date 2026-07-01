# Deploy

**Status:** stable · **Bundles:** `ail-scheduled-publish` (root), `ail-self-optimizer` (`ail-self-optimizer/`)

This is the operations guide for deploying the framework so the SQL-warehouse
access is **turnkey** — deployers do not grant access by hand. It covers the two
deploy decisions baked into the bundles:

1. **Provide-or-create the warehouse** — accept an existing `warehouse_id`, or
   provision a small serverless SQL warehouse if none is given.
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

---

## 0. What deploys

Two independent Declarative Automation Bundles (DABs), both resolving host + auth
from the Databricks CLI profile at deploy time (nothing hardcoded):

| Bundle | Path | Contains | run-as |
|--------|------|----------|--------|
| `ail-scheduled-publish` | repo root `databricks.yml` + `resources/*.yml` | the scheduled L0 publish job **and** the on-demand `ail-apply-service` job (§7) | bundle-level `run_as` (§2) |
| `ail-self-optimizer` | `ail-self-optimizer/databricks.yml` | the L0 leaderboard App (incl. the approve/reject write-path) | the App's auto-provisioned SP (fixed by the platform) |

The **bootstrap** (`ail-bootstrap-grants`, §1/§3/§4) is a CLI an admin runs as part
of deploy — it provisions/resolves the warehouse, **ensures the app's tables exist
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
default to the reference workspace's warehouse so an unflagged deploy still works.

### Auto-provision when none is given

If you do not have a warehouse, the bootstrap step provisions one — a small
(`2X-Small`), serverless (`PRO`) warehouse with a 10-minute auto-stop, found-or-
created **by name** (`ail-framework-serverless`) so re-runs never make a second
one. Run the bootstrap with **no** `--warehouse-id`; it prints the id it created:

```bash
ail-bootstrap-grants --experiment <EXPERIMENT_ID> --framework-sp-id <FRAMEWORK_SP_ID>
# -> [ail.jobs.bootstrap_grants] warehouse=<NEW_ID> (created) grant_can_use=<sp> ...
```

Then deploy both bundles with `warehouse_id=<NEW_ID>` / `sql_warehouse_id=<NEW_ID>`
as above.

> Why a CLI and not a bundle-declared warehouse: DABs **does** support an
> `sql_warehouses` resource, but it cannot express *"create only if the deployer
> did not supply one"* — see [§6](#6-dab--apps-capability-gaps-found). The
> provide-or-create branch therefore lives in the idempotent bootstrap.

---

## 2. One framework service principal

| Component | Identity | How it is set |
|-----------|----------|---------------|
| App | the App's **auto-provisioned** SP | fixed by the Apps platform — an app always runs as its own SP |
| Publish job + future jobs | `framework_sp_id` | bundle-level `run_as` (one knob for all jobs) |

The bundle exposes one knob, `run_as: ${var.job_run_as}`, that applies to
**every** job in the bundle — the publish job today, and any job added under
`resources/` tomorrow. It is driven by two variables:

- `framework_sp_id` — the application (client) id of the single framework SP.
- `job_run_as` (complex) — defaults to `{user_name: ${workspace.current_user.userName}}`
  (the deploying identity), overridden to the SP by the `dais_demo_sp` target.

**Default target `dais_demo`** → jobs run as the deploying identity. Use this for
the admin verification deploy; no SP needed.

**Target `dais_demo_sp`** → jobs run as the SP. Turnkey via a plain string var:

```bash
databricks bundle deploy -t dais_demo_sp \
  --var framework_sp_id=<FRAMEWORK_SP_ID> --profile dais-demo
```

> Deploying `dais_demo_sp` **without** `framework_sp_id` fails fast with
> `run_as section must specify exactly one identity` — an intentional guard, not
> a bug.

### Make it literally one SP: reuse the App's SP

The cleanest single-SP setup reuses the **App's** auto-provisioned SP as
`framework_sp_id`, because the App's SP cannot be reassigned and a job cannot
reference it until it exists. So the turnkey order is:

1. Run the **pre-app bootstrap** (§4 step 2) to provision/resolve the warehouse and
   ensure the app's tables, then deploy the **app** bundle. This creates the app,
   its SP, and (because the warehouse is declared with `permission: CAN_USE`)
   **auto-grants** `CAN_USE` on the warehouse to that SP. (The table-ensure must
   precede the app deploy so the build's typegen resolves — §3 callout.)
2. Read the App SP's application id:
   ```bash
   databricks apps get ail-self-optimizer -o json --profile dais-demo \
     | python3 -c "import sys,json; print(json.load(sys.stdin)['service_principal_client_id'])"
   ```
3. Deploy the **job** bundle with that id as `framework_sp_id` (target
   `dais_demo_sp`), and run the bootstrap (§3) with the same id. Now the app and
   every job run as the **same** SP, covered by the **same** grant.

If you prefer a standalone SP (created once by an admin), pass its application id
as `framework_sp_id` instead — the bootstrap then grants it (and the App's own SP
keeps its auto-grant; both have `CAN_USE`).

---

## 3. Warehouse tables, the grant, and the monitoring tag (bootstrap)

`ail-bootstrap-grants` (module `ail.jobs.bootstrap_grants`) is **idempotent** and
does four things in one run:

1. **Resolve the warehouse** — use `--warehouse-id`, else find-or-create (§1).
2. **Ensure the app's tables exist (empty)** — create every UC table the deployed
   app's SQL queries read, using each writer module's **own** authoritative
   `_ddl()` `CREATE SCHEMA/TABLE IF NOT EXISTS` (module `ail.jobs.bootstrap_tables`;
   no schema is authored in the bootstrap). This is **load-bearing for the app
   build** — see the ordering callout below. `CREATE ... IF NOT EXISTS` only, so a
   re-run never drops, alters, or repopulates an existing table.
3. **Grant `CAN_USE`** on the warehouse to `--framework-sp-id` via the warehouse
   permissions API (`update_permissions`, a merge — it does not clobber the App
   SP's auto-grant). Skipped if no SP is given.
4. **Tag the experiment** with `mlflow.monitoring.sqlWarehouseId = <warehouse>`
   (reusing `ail.compare.monitoring.configure_monitoring_warehouse`) so MLflow's
   monitoring job fetches the v4 Unity Catalog traces the scheduled scorers score.

```bash
# Run once, as a workspace admin, BEFORE deploying the app (see ordering below):
ail-bootstrap-grants \
  --experiment <EXPERIMENT_ID> \
  --warehouse-id <WAREHOUSE_ID> \
  --framework-sp-id <FRAMEWORK_SP_ID> --profile dais-demo
# Omit --warehouse-id to auto-provision; omit --framework-sp-id to skip the grant
# (e.g. on the pre-app run, before the App SP exists — see §4). --catalog/--schema
# default to the framework catalog.schema the app reads.
```

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

---

## 4. End-to-end turnkey sequence

**Ordering is load-bearing:** the bootstrap's warehouse + **table-ensure** + tag
(step 2) must run **before** the app is deployed/started (step 3), because the app
build's typegen `DESCRIBE`s every query's table live and fails hard on a missing
one (§3 callout). The bootstrap is idempotent, so it is run twice — once **before**
the app for the tables (no SP grant yet, since the App SP does not exist), and once
**after** to add the grant for the single framework SP.

```bash
# 1. (verification) deploy the job bundle as yourself — proves config is sound,
#    and creates the apply job whose id the app needs (§7 step 1).
databricks bundle validate -t dais_demo --profile dais-demo
databricks bundle deploy   -t dais_demo --profile dais-demo

# 2. BOOTSTRAP FIRST [ADMIN]: (provide-or-create wh) + ensure the app's tables
#    (empty) + tag experiment. No --framework-sp-id yet (the App SP is created in
#    step 3). This is what lets step 3's app build typegen resolve on a clean
#    workspace — with ZERO manual DDL.
ail-bootstrap-grants --experiment <EXPERIMENT_ID> \
  --warehouse-id <WAREHOUSE_ID> --profile dais-demo

# 3. deploy the app (creates its SP + auto-grants CAN_USE on the warehouse). Its
#    build typegen now DESCRIBEs tables that EXIST (step 2). Point it at the apply
#    job with --var apply_job_id=<id> (from step 1 / §7) so it also auto-grants
#    CAN_MANAGE_RUN on that job + injects AIL_APPLY_JOB_ID; see §7.
cd ail-self-optimizer && databricks bundle deploy --profile dais-demo \
  --var apply_job_id=<APPLY_JOB_ID> && cd ..
databricks bundle run app -t default --profile dais-demo   # start the app

# 4. capture the App SP -> the single framework SP
SP=$(databricks apps get ail-self-optimizer -o json --profile dais-demo \
       | python3 -c "import sys,json;print(json.load(sys.stdin)['service_principal_client_id'])")

# 5. bootstrap AGAIN (idempotent) to grant CAN_USE to that SP (wh reused, tables
#    no-op, tag no-op), then re-deploy the jobs to run as that SP.
ail-bootstrap-grants --experiment <EXPERIMENT_ID> \
  --warehouse-id <WAREHOUSE_ID> --framework-sp-id "$SP" --profile dais-demo
databricks bundle deploy -t dais_demo_sp --var framework_sp_id="$SP" --profile dais-demo
```

> If you use a **standalone** framework SP (created once by an admin) instead of
> reusing the App SP, pass `--framework-sp-id <SP>` in step 2 and drop step 5's
> bootstrap re-run — the single pre-app bootstrap then does the warehouse, tables,
> grant, and tag in one shot.

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
   resource) is reused to read the result row back. Optional overrides:
   `AIL_APPLY_CATALOG` / `AIL_APPLY_SCHEMA` (default the framework
   `austin_choi_omni_agent_catalog.agent_improvement_loop`),
   `AIL_APPLY_JOB_TIMEOUT_MS` (default 300000), `AIL_APPLY_JOB_POLL_MS` (default
   3000).

Without `AIL_APPLY_TRANSPORT=job` / `AIL_APPLY_JOB_ID`, the app falls back to the
subprocess bridge — correct for local dev, but on the Node-only deployed image that
bridge cannot run. `AIL_APPLY_TRANSPORT: job` is committed in `app.yaml`, so the
only per-workspace step is supplying `apply_job_id` at deploy (a missing/empty id
leaves the deployed app selecting the Job transport but with no job to trigger — it
fails closed rather than silently applying via a bridge it can't run).
