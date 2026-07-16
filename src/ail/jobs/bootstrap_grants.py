"""Post-deploy bootstrap that makes the framework's SQL-warehouse access turnkey.

``databricks bundle deploy`` uploads the publish job, the app, and (later) the
scheduled scorers, but four operational facts still have to be true before any
of them can read traces *through* a SQL warehouse:

1. a warehouse exists — provided by the deployer, or provisioned here only when
   the deployer passes the explicit create flag;
2. the single framework service principal has ``CAN_USE`` on it;
3. every table the deployed app's SQL queries read exists (empty is fine), so the
   AppKit build's typegen ``DESCRIBE QUERY`` — and every runtime ``SELECT`` — has
   a table to resolve (see :mod:`ail.jobs.bootstrap_tables`); and
4. the target MLflow experiment carries the monitoring tag
   :data:`ail.compare.monitoring.MONITORING_WAREHOUSE_TAG`
   (``mlflow.monitoring.sqlWarehouseId``) so MLflow's monitoring job fetches the
   v4 Unity Catalog traces the scorers score.

This script performs all four **idempotently**, so re-running it is a no-op
beyond confirming state. It is the *conditional* half of the warehouse story a
Declarative Automation Bundle cannot express on its own. Because the app build's
typegen fails hard on a not-yet-created table, this bootstrap must run **before**
the app bundle is deployed/started — see ``docs/DEPLOY.md`` for the enforced
sequence.

Why this is not pure bundle YAML
--------------------------------
DABs **does** support an ``sql_warehouses`` resource and even ``permissions`` on
it (verified via ``databricks bundle schema``). What it cannot express is the
operator-controlled conditional — *"use this existing warehouse, or explicitly
create the framework warehouse first"*. A declared resource is always created;
there is no ``count``/``if`` in the bundle schema. So the
**provide-or-explicitly-create** branch (and the grant against a warehouse whose
id is only known at runtime when we create it) lives here.

The app's own grant is still handled natively: the ``ail-self-optimizer`` app
declares the warehouse with ``permission: CAN_USE``, so the Apps platform
auto-grants ``CAN_USE`` to the app's service principal at deploy. When that app
SP is reused as :data:`framework_sp_id` (the recommended single-SP pattern), this
script's grant is the same principal and merely confirms it.

.. important::

   Creating a warehouse and granting ``CAN_USE`` require the *running* identity
   to hold workspace authority — workspace admin, or ``CAN_MANAGE`` on the
   warehouse, or the can-create-warehouse entitlement. That is the Databricks
   permission model; this script neither bypasses nor weakens it. A non-admin run
   fails at the warehouse-create / grant call. Run it once as an admin after
   deploy (or have an admin run the one-time grant); thereafter the framework is
   turnkey. See ``docs/DEPLOY.md``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ail.compare.monitoring import MonitoringWarehouseConfig, configure_monitoring_warehouse
from ail.jobs.bootstrap_tables import ensure_app_tables, reconcile_app_table_columns

# Re-exported for backward compatibility: these symbols moved (the workspace guard
# constants/helper now live in ``ail.workspace_guards`` to break an import cycle),
# but ``ail.jobs.bootstrap_grants`` remains their stable public import location.
from ail.publish import (  # noqa: F401
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    REFERENCE_EXPERIMENT,
)
from ail.workspace_guards import (  # noqa: F401
    REFERENCE_WORKSPACE_DEFAULTS,
    _workspace_value_error,
)

if TYPE_CHECKING:
    from mlflow import MlflowClient

#: Name used to find-or-create the framework-managed serverless warehouse when
#: the deployer does not supply an existing ``warehouse_id``. Lookup is by exact
#: name, so re-running never creates a second warehouse.
DEFAULT_WAREHOUSE_NAME = "ail-framework-serverless"

#: Smallest (cheapest) warehouse size; the framework warehouse only has to read
#: traces and refresh small L0 tables.
DEFAULT_CLUSTER_SIZE = "2X-Small"

#: Shortest auto-stop the platform allows for a non-zero value (must be 0 or
#: >= 10). Keeps an idle framework warehouse from billing.
DEFAULT_AUTO_STOP_MINS = 10


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """What :func:`bootstrap` resolved/did, for the caller's logs and tests."""

    warehouse_id: str
    warehouse_created: bool
    granted_sp_id: str | None
    tables_ensured: list[str]
    monitoring: MonitoringWarehouseConfig | None
    #: ``ALTER TABLE ... ADD COLUMNS`` statements the additive-reconcile step ran
    #: to migrate pre-existing tables to their writer-declared schema (empty when
    #: nothing needed migrating — a fresh workspace or an already-migrated one).
    columns_reconciled: list[str] = field(default_factory=list)
    #: Spans tables newly added to the arrival-triggered RLM job's ``table_update``
    #: trigger this run (empty when no ``rlm_job_id`` was supplied, the registry is
    #: empty, or every agent's table was already watched). The deploy heal that keeps
    #: the DAB-managed trigger from reverting to its single-table YAML value on redeploy.
    rlm_trigger_tables_added: list[str] = field(default_factory=list)


def validate_workspace_values(
    *,
    experiment_id: str,
    warehouse_id: str | None,
    catalog: str,
    schema: str,
    warehouse_required: bool = True,
    allow_reference_workspace: bool = False,
) -> None:
    """Fail closed on workspace-specific deploy values before any workspace calls."""
    errors = [
        error
        for error in (
            _workspace_value_error(
                "experiment_id",
                experiment_id,
                allow_reference_workspace=allow_reference_workspace,
            ),
            _workspace_value_error(
                "warehouse_id",
                warehouse_id,
                required=warehouse_required,
                allow_reference_workspace=allow_reference_workspace,
            ),
            _workspace_value_error(
                "catalog",
                catalog,
                allow_reference_workspace=allow_reference_workspace,
            ),
            _workspace_value_error(
                "schema",
                schema,
                allow_reference_workspace=allow_reference_workspace,
            ),
        )
        if error
    ]
    if errors:
        details = "; ".join(errors)
        raise SystemExit(
            "Refusing to run against empty, placeholder, or reference workspace defaults "
            f"({details}); set --experiment/--warehouse-id/--catalog/--schema to your own "
            "workspace values."
        )


def _default_workspace_client() -> Any:
    """Build a ``WorkspaceClient`` from the running (admin) credentials."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def ensure_warehouse(
    client: Any,
    *,
    warehouse_id: str | None,
    warehouse_name: str = DEFAULT_WAREHOUSE_NAME,
    cluster_size: str = DEFAULT_CLUSTER_SIZE,
    auto_stop_mins: int = DEFAULT_AUTO_STOP_MINS,
) -> tuple[str, bool]:
    """Resolve the framework warehouse, provisioning a serverless one if absent.

    Provide-or-create, idempotently:

    * ``warehouse_id`` supplied (non-blank) -> use it as-is, ``created=False``.
    * else a warehouse named ``warehouse_name`` already exists -> reuse it,
      ``created=False`` (this is what makes re-runs a no-op).
    * else create a small serverless (``PRO``) warehouse and return its id with
      ``created=True``.

    Returns ``(warehouse_id, created)``. Creating a warehouse requires the running
    identity to have the can-create-warehouse authority (see module docstring).
    """
    if warehouse_id and warehouse_id.strip():
        return warehouse_id.strip(), False

    for wh in client.warehouses.list():
        if getattr(wh, "name", None) == warehouse_name and getattr(wh, "id", None):
            return wh.id, False

    from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType

    created = client.warehouses.create(
        name=warehouse_name,
        cluster_size=cluster_size,
        enable_serverless_compute=True,
        warehouse_type=CreateWarehouseRequestWarehouseType.PRO,
        auto_stop_mins=auto_stop_mins,
        max_num_clusters=1,
    )
    # ``create`` returns a waiter; ``.response`` is the immediate create response
    # (which carries the new id). We do not block on the warehouse reaching
    # RUNNING — it starts on first query.
    new_id = created.response.id
    if not new_id:
        raise RuntimeError("warehouse create returned no id")
    return new_id, True


def grant_warehouse_can_use(client: Any, warehouse_id: str, sp_id: str) -> None:
    """Grant ``CAN_USE`` on ``warehouse_id`` to service principal ``sp_id``.

    Uses ``update_permissions`` (PATCH/merge), so existing grants — e.g. the app
    SP's auto-granted ``CAN_USE`` or an owner's ``IS_OWNER`` — are preserved, and
    re-running is a no-op. Requires the running identity to have ``CAN_MANAGE`` on
    the warehouse (see module docstring).
    """
    from databricks.sdk.service.sql import (
        WarehouseAccessControlRequest,
        WarehousePermissionLevel,
    )

    client.warehouses.update_permissions(
        warehouse_id,
        access_control_list=[
            WarehouseAccessControlRequest(
                service_principal_name=sp_id,
                permission_level=WarehousePermissionLevel.CAN_USE,
            )
        ],
    )


def grant_framework_schema_access(
    client: Any, *, catalog: str, schema: str, sp_id: str
) -> None:
    """Merge the minimum UC privileges required by the App and framework jobs.

    The dedicated framework schema contains only AIL-owned tables. ``SELECT`` is
    required by the app/readers, ``MODIFY`` by governed request/event writers, and
    ``CREATE_TABLE`` by idempotent writer/bootstrap DDL. The catalog/schema use
    grants are explicit so this works even when account-wide inheritance is absent.
    """
    from databricks.sdk.service.catalog import PermissionsChange, Privilege, SecurableType

    client.grants.update(
        SecurableType.CATALOG,
        catalog,
        changes=[PermissionsChange(principal=sp_id, add=[Privilege.USE_CATALOG])],
    )
    client.grants.update(
        SecurableType.SCHEMA,
        f"{catalog}.{schema}",
        changes=[
            PermissionsChange(
                principal=sp_id,
                add=[
                    Privilege.USE_SCHEMA,
                    Privilege.SELECT,
                    Privilege.MODIFY,
                    Privilege.CREATE_TABLE,
                ],
            )
        ],
    )


def bootstrap(
    *,
    experiment_id: str,
    warehouse_id: str | None,
    framework_sp_id: str | None,
    warehouse_name: str = DEFAULT_WAREHOUSE_NAME,
    cluster_size: str = DEFAULT_CLUSTER_SIZE,
    auto_stop_mins: int = DEFAULT_AUTO_STOP_MINS,
    catalog: str = "",
    schema: str = "",
    create_warehouse: bool = False,
    allow_reference_workspace: bool = False,
    rlm_job_id: int | None = None,
    client: Any | None = None,
    mlflow_client: MlflowClient | None = None,
) -> BootstrapResult:
    """Idempotently provision/resolve the warehouse, ensure the app tables, grant it, and tag.

    Args:
        experiment_id: MLflow experiment to tag with the monitoring warehouse.
        warehouse_id: Existing warehouse to use. Required unless
            ``create_warehouse`` is true, so reusable deploys cannot accidentally
            create or fall back to a reference workspace.
        framework_sp_id: Single framework SP to grant warehouse ``CAN_USE`` and
            least-privilege access to the dedicated UC schema; blank/``None``
            skips both grants for a human/dev deployment.
        warehouse_name, cluster_size, auto_stop_mins: serverless-warehouse spec
            used only on the create path.
        catalog, schema: Unity Catalog location of the app's tables; the
            table-ensure step creates every app-read table here.
        create_warehouse: Explicit opt-in to find-or-create the framework-managed
            serverless warehouse by ``warehouse_name`` when ``warehouse_id`` is
            omitted.
        allow_reference_workspace: Explicit opt-in for the owner-only reference
            workspace redeploy path. Bypasses only the reference-default check;
            empty, placeholder, and unresolved bundle values remain fatal.
        rlm_job_id: The arrival-triggered continuous-RLM job id. When supplied, the
            bootstrap re-reconciles that job's ``table_update`` trigger to watch every
            registered agent's ``*_otel_spans`` table (add-only) — the deploy heal that
            counters the DAB reverting the trigger to its single-table YAML value on each
            redeploy. ``None``/omitted => skip the heal (a deploy that did not wire the
            id keeps today's behavior). Fail-soft: a heal failure is logged, never fatal.
        client: Databricks ``WorkspaceClient`` (injectable for tests).
        mlflow_client: ``MlflowClient`` passed through to
            :func:`configure_monitoring_warehouse` (injectable for tests).

    Returns:
        A :class:`BootstrapResult` describing what was resolved/done.
    """
    validate_workspace_values(
        experiment_id=experiment_id,
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
        warehouse_required=not create_warehouse,
        allow_reference_workspace=allow_reference_workspace,
    )

    if client is None:
        client = _default_workspace_client()

    warehouse_to_resolve = (
        None if create_warehouse and not (warehouse_id or "").strip() else warehouse_id
    )
    resolved_id, created = ensure_warehouse(
        client,
        warehouse_id=warehouse_to_resolve,
        warehouse_name=warehouse_name,
        cluster_size=cluster_size,
        auto_stop_mins=auto_stop_mins,
    )

    # Ensure every table the app's SQL queries read exists (empty), using each
    # writer module's own authoritative _ddl(). This is what lets the AppKit
    # build's typegen DESCRIBE QUERY (and every runtime SELECT) resolve on a
    # clean workspace; idempotent CREATE ... IF NOT EXISTS, never DROP/ALTER.
    tables_ensured = ensure_app_tables(client, resolved_id, catalog=catalog, schema=schema)

    # Then additively reconcile columns: CREATE TABLE IF NOT EXISTS never adds a
    # column to a pre-existing table, so a schema-additive upgrade deploy would
    # otherwise leave new writer columns missing and fail the app build's typegen
    # DESCRIBE QUERY. Runs AFTER the CREATEs (a just-created table is a no-op) and
    # BEFORE the app build; ADD COLUMNS only, fail-loud on a type conflict.
    columns_reconciled = reconcile_app_table_columns(
        client, resolved_id, catalog=catalog, schema=schema
    )

    granted: str | None = None
    if framework_sp_id and framework_sp_id.strip():
        granted = framework_sp_id.strip()
        grant_warehouse_can_use(client, resolved_id, granted)
        grant_framework_schema_access(
            client, catalog=catalog.strip(), schema=schema.strip(), sp_id=granted
        )

    # set_env=False: this is a deploy-time bootstrap, not the in-process trace
    # read, so the persistent experiment tag is what matters — do not mutate the
    # bootstrap process's environment as a side effect.
    monitoring = configure_monitoring_warehouse(
        experiment_id.strip(),
        resolved_id,
        client=mlflow_client,
        set_env=False,
    )

    # Deploy heal for the arrival-triggered RLM job: the DAB reverts its table_update
    # trigger to the single YAML table on every deploy, so re-add every registered
    # agent's *_otel_spans table here (add-only). Runs AFTER ensure_app_tables so the
    # agent_registry table exists to read. Fail-soft: the trigger heal is an
    # optimization on top of an already-deployed, registry-driven job — a Jobs API
    # failure must not fail the bootstrap (warehouse/tables/grants already succeeded).
    rlm_trigger_tables_added = _heal_rlm_trigger(
        client, resolved_id, rlm_job_id=rlm_job_id, catalog=catalog, schema=schema
    )

    return BootstrapResult(
        warehouse_id=resolved_id,
        warehouse_created=created,
        granted_sp_id=granted,
        tables_ensured=tables_ensured,
        monitoring=monitoring,
        columns_reconciled=columns_reconciled,
        rlm_trigger_tables_added=rlm_trigger_tables_added,
    )


def _heal_rlm_trigger(
    client: Any,
    warehouse_id: str,
    *,
    rlm_job_id: int | None,
    catalog: str,
    schema: str,
) -> list[str]:
    """Re-add every registered agent's spans table to the RLM trigger (deploy heal).

    Reads the full registry through the shared fail-closed loader and unions each
    agent's ``*_otel_spans`` table into the RLM job's ``table_update`` trigger. Returns
    the tables newly added (empty on skip / nothing-to-do). Fail-soft: a missing job id
    is a quiet skip, and any read/reconcile failure is logged and swallowed so a healthy
    warehouse/tables/grants bootstrap is never reported as failed for a trigger nicety.
    """
    if rlm_job_id is None:
        return []
    try:
        from ail.jobs.rlm_trigger import reconcile_rlm_trigger_tables
        from ail.publish_versions import load_registered_agents_full

        agents = load_registered_agents_full(
            client=client, warehouse_id=warehouse_id, catalog=catalog, schema=schema
        )
        if not agents:
            return []
        result = reconcile_rlm_trigger_tables(client, rlm_job_id=rlm_job_id, agents=agents)
    except Exception as exc:  # noqa: BLE001 - the heal is best-effort; never fail the bootstrap
        print(
            f"[ail.jobs.bootstrap_grants] RLM trigger heal skipped "
            f"(bootstrap otherwise succeeded): {type(exc).__name__}: {exc}"
        )
        return []
    return list(result.added)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-deploy bootstrap: use an existing framework SQL warehouse or "
            "explicitly create one, "
            "grant CAN_USE to the single framework service principal, and tag the "
            "MLflow experiment so scheduled scorers can read traces. Idempotent; "
            "run once as a workspace admin."
        )
    )
    parser.add_argument("--experiment", default="")
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID", ""),
        help="Existing SQL warehouse id to use. Required unless --create-warehouse "
        "is set; reference workspace defaults and placeholders are refused before "
        "any workspace calls.",
    )
    parser.add_argument(
        "--create-warehouse",
        action="store_true",
        help="Explicitly find-or-create the framework-managed serverless SQL "
        "warehouse by --warehouse-name when --warehouse-id is omitted, and print "
        "the resolved id for reuse in later deploy commands.",
    )
    parser.add_argument(
        "--framework-sp-id",
        default=os.environ.get("AIL_FRAMEWORK_SP_ID", ""),
        help="Application (client) id of the single framework service principal to "
        "grant CAN_USE. Empty => skip the grant.",
    )
    parser.add_argument("--warehouse-name", default=DEFAULT_WAREHOUSE_NAME)
    parser.add_argument("--cluster-size", default=DEFAULT_CLUSTER_SIZE)
    parser.add_argument("--auto-stop-mins", default=DEFAULT_AUTO_STOP_MINS, type=int)
    parser.add_argument(
        "--catalog",
        default="",
        help="Unity Catalog catalog holding the app's tables (table-ensure step). "
        "Required; the reference catalog is refused.",
    )
    parser.add_argument(
        "--schema",
        default="",
        help="Schema (within --catalog) holding the app's tables (table-ensure step). Required.",
    )
    parser.add_argument(
        "--allow-reference-workspace",
        action="store_true",
        default=os.environ.get("AIL_ALLOW_REFERENCE") == "1",
        help="Owner-only escape hatch: permit the known reference workspace values. "
        "Empty, placeholder, and unresolved bundle values are still refused.",
    )
    parser.add_argument(
        "--rlm-job-id",
        default=os.environ.get("AIL_RLM_JOB_ID", ""),
        help="Arrival-triggered continuous-RLM job id. When set, re-reconcile its "
        "table_update trigger to watch every registered agent's *_otel_spans table "
        "(the deploy heal against DAB reverting the trigger). Empty => skip the heal.",
    )
    args = parser.parse_args(argv)
    if args.create_warehouse and (args.warehouse_id or "").strip():
        parser.error("--create-warehouse cannot be combined with --warehouse-id")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rlm_job_id: int | None = None
    raw_rlm = (args.rlm_job_id or "").strip()
    if raw_rlm:
        try:
            rlm_job_id = int(raw_rlm)
        except ValueError:
            # A malformed id is a wiring mistake, not a reason to abort a deploy
            # bootstrap: warn and skip the trigger heal (leaving today's behavior).
            print(
                f"[ail.jobs.bootstrap_grants] --rlm-job-id={raw_rlm!r} is not an int; "
                "skipping RLM trigger heal."
            )
    result = bootstrap(
        experiment_id=args.experiment,
        # Empty strings are still collapsed before bootstrap so the guard reports
        # one clear missing-value error for omitted CLI options.
        warehouse_id=args.warehouse_id or None,
        framework_sp_id=args.framework_sp_id or None,
        warehouse_name=args.warehouse_name,
        cluster_size=args.cluster_size,
        auto_stop_mins=args.auto_stop_mins,
        catalog=args.catalog,
        schema=args.schema,
        create_warehouse=args.create_warehouse,
        allow_reference_workspace=args.allow_reference_workspace,
        rlm_job_id=rlm_job_id,
    )
    action = "created" if result.warehouse_created else "reused"
    grant = result.granted_sp_id or "(skipped — no framework_sp_id)"
    print(
        f"[ail.jobs.bootstrap_grants] warehouse={result.warehouse_id} ({action}) "
        f"grant_can_use={grant} "
        f"tables_ensured={len(result.tables_ensured)} in {args.catalog}.{args.schema} "
        f"columns_reconciled={len(result.columns_reconciled)} "
        f"experiment={args.experiment} "
        f"monitoring_tag_set={result.monitoring is not None}"
    )
    for alter in result.columns_reconciled:
        print(f"[ail.jobs.bootstrap_grants] migrated: {alter}")
    if result.rlm_trigger_tables_added:
        print(
            "[ail.jobs.bootstrap_grants] RLM trigger now also watches: "
            + ", ".join(result.rlm_trigger_tables_added)
        )
    if args.create_warehouse:
        print(f"[ail.jobs.bootstrap_grants] warehouse_id={result.warehouse_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
