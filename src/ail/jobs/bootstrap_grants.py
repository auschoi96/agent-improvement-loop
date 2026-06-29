"""Post-deploy bootstrap that makes the framework's SQL-warehouse access turnkey.

``databricks bundle deploy`` uploads the publish job, the app, and (later) the
scheduled scorers, but three operational facts still have to be true before any
of them can read traces *through* a SQL warehouse:

1. a warehouse exists — provided by the deployer, or provisioned here;
2. the single framework service principal has ``CAN_USE`` on it; and
3. the target MLflow experiment carries the monitoring tag
   :data:`ail.compare.monitoring.MONITORING_WAREHOUSE_TAG`
   (``mlflow.monitoring.sqlWarehouseId``) so MLflow's monitoring job fetches the
   v4 Unity Catalog traces the scorers score.

This script performs all three **idempotently**, so re-running it is a no-op
beyond confirming state. It is the *conditional* half of the warehouse story a
Declarative Automation Bundle cannot express on its own.

Why this is not pure bundle YAML
--------------------------------
DABs **does** support an ``sql_warehouses`` resource and even ``permissions`` on
it (verified via ``databricks bundle schema``). What it cannot express is the
conditional the deployer actually wants — *"create a warehouse only if I did not
supply one"*. A declared resource is always created; there is no ``count``/``if``
in the bundle schema. So the **provide-or-create** branch (and the grant against
a warehouse whose id is only known at runtime when we create it) lives here.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ail.compare.monitoring import MonitoringWarehouseConfig, configure_monitoring_warehouse
from ail.publish import REFERENCE_EXPERIMENT

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
    monitoring: MonitoringWarehouseConfig | None


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


def bootstrap(
    *,
    experiment_id: str,
    warehouse_id: str | None,
    framework_sp_id: str | None,
    warehouse_name: str = DEFAULT_WAREHOUSE_NAME,
    cluster_size: str = DEFAULT_CLUSTER_SIZE,
    auto_stop_mins: int = DEFAULT_AUTO_STOP_MINS,
    client: Any | None = None,
    mlflow_client: MlflowClient | None = None,
) -> BootstrapResult:
    """Idempotently provision/resolve the warehouse, grant it, and tag the experiment.

    Args:
        experiment_id: MLflow experiment to tag with the monitoring warehouse.
        warehouse_id: Existing warehouse to use; blank/``None`` => find-or-create.
        framework_sp_id: Single framework SP to grant ``CAN_USE``; blank/``None``
            => skip the grant (the app SP is already granted via the app's
            resource declaration, and a job running as the deploying identity
            needs no extra grant).
        warehouse_name, cluster_size, auto_stop_mins: serverless-warehouse spec
            used only on the create path.
        client: Databricks ``WorkspaceClient`` (injectable for tests).
        mlflow_client: ``MlflowClient`` passed through to
            :func:`configure_monitoring_warehouse` (injectable for tests).

    Returns:
        A :class:`BootstrapResult` describing what was resolved/done.
    """
    if not experiment_id or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty experiment id")

    if client is None:
        client = _default_workspace_client()

    resolved_id, created = ensure_warehouse(
        client,
        warehouse_id=warehouse_id,
        warehouse_name=warehouse_name,
        cluster_size=cluster_size,
        auto_stop_mins=auto_stop_mins,
    )

    granted: str | None = None
    if framework_sp_id and framework_sp_id.strip():
        grant_warehouse_can_use(client, resolved_id, framework_sp_id.strip())
        granted = framework_sp_id.strip()

    # set_env=False: this is a deploy-time bootstrap, not the in-process trace
    # read, so the persistent experiment tag is what matters — do not mutate the
    # bootstrap process's environment as a side effect.
    monitoring = configure_monitoring_warehouse(
        experiment_id.strip(),
        resolved_id,
        client=mlflow_client,
        set_env=False,
    )

    return BootstrapResult(
        warehouse_id=resolved_id,
        warehouse_created=created,
        granted_sp_id=granted,
        monitoring=monitoring,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-deploy bootstrap: provide-or-create the framework SQL warehouse, "
            "grant CAN_USE to the single framework service principal, and tag the "
            "MLflow experiment so scheduled scorers can read traces. Idempotent; "
            "run once as a workspace admin."
        )
    )
    parser.add_argument("--experiment", default=REFERENCE_EXPERIMENT)
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID", ""),
        help="Existing SQL warehouse id to use. Empty => find-or-create a small "
        "serverless warehouse named --warehouse-name.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = bootstrap(
        experiment_id=args.experiment,
        # Empty strings (the default when the operator omits them) mean
        # "not provided" -> find-or-create / skip-grant.
        warehouse_id=args.warehouse_id or None,
        framework_sp_id=args.framework_sp_id or None,
        warehouse_name=args.warehouse_name,
        cluster_size=args.cluster_size,
        auto_stop_mins=args.auto_stop_mins,
    )
    action = "created" if result.warehouse_created else "reused"
    grant = result.granted_sp_id or "(skipped — no framework_sp_id)"
    print(
        f"[ail.jobs.bootstrap_grants] warehouse={result.warehouse_id} ({action}) "
        f"grant_can_use={grant} experiment={args.experiment} "
        f"monitoring_tag_set={result.monitoring is not None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
