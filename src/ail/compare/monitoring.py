"""Wire an experiment's **monitoring SQL warehouse** so live scoring can read traces.

This is the operational prerequisite called out in
:mod:`ail.judges.registration`: the scheduled L2 scorers (PR #11) are *registered*
and scheduled, but MLflow's v4 Unity Catalog trace store can only fetch the
traces to score **through a SQL warehouse**. Until one is wired to the
experiment, the scorers run but score nothing. This helper sets that warehouse on
the experiment so the scheduled scorers — and any future live (read-then-score)
path the comparison harness drives — can fetch traces.

It sets the configuration two ways, matching the conventions already in the
codebase:

* the experiment tag :data:`MONITORING_WAREHOUSE_TAG`
  (``mlflow.monitoring.sqlWarehouseId``) — the persistent, per-experiment setting
  the background monitoring job reads (mirrors how
  :func:`ail.judges.registration._tag_alignment` writes experiment tags); and
* the process environment variable :data:`TRACING_WAREHOUSE_ENV`
  (``MLFLOW_TRACING_SQL_WAREHOUSE_ID``) — the same variable
  :mod:`ail.publish` surfaces so an in-process trace read picks up the warehouse
  immediately.

.. important::

   **This sets configuration; it does not grant access.** The calling identity
   still needs ``CAN_USE`` on the warehouse for the read to succeed. That is a
   Databricks permission grant performed by a workspace admin (e.g. via UC /
   warehouse permissions), **not** something this code can or should do. With the
   tag set but the grant missing, the read fails with a permissions error — which
   is exactly the v4-store access gap the live scoring lane is blocked on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow import MlflowClient

__all__ = [
    "MONITORING_WAREHOUSE_TAG",
    "TRACING_WAREHOUSE_ENV",
    "MonitoringWarehouseConfig",
    "configure_monitoring_warehouse",
]

#: Experiment tag MLflow reads to find the monitoring SQL warehouse for a v4
#: (Unity Catalog) trace store. Set on the experiment so scheduled scorers can
#: fetch the traces they score.
MONITORING_WAREHOUSE_TAG = "mlflow.monitoring.sqlWarehouseId"

#: Process env var the MLflow trace-read path honours (same one
#: :mod:`ail.publish` surfaces). Set so an in-process read sees the warehouse
#: without a round-trip through the experiment tag.
TRACING_WAREHOUSE_ENV = "MLFLOW_TRACING_SQL_WAREHOUSE_ID"


@dataclass(frozen=True, slots=True)
class MonitoringWarehouseConfig:
    """What :func:`configure_monitoring_warehouse` set, for the caller's records."""

    experiment_id: str
    warehouse_id: str
    tag_key: str = MONITORING_WAREHOUSE_TAG
    env_var: str = TRACING_WAREHOUSE_ENV


def configure_monitoring_warehouse(
    experiment_id: str,
    warehouse_id: str,
    *,
    client: MlflowClient | None = None,
    set_env: bool = True,
) -> MonitoringWarehouseConfig:
    """Set the monitoring SQL warehouse on ``experiment_id`` for live trace reads.

    Writes the experiment tag :data:`MONITORING_WAREHOUSE_TAG` (so MLflow's
    scheduled-scorer / monitoring job can fetch v4 UC traces) and, when
    ``set_env`` is true, the process env var :data:`TRACING_WAREHOUSE_ENV` (so an
    immediately-following in-process read uses the same warehouse).

    Setting the warehouse is necessary but **not sufficient**: the calling
    identity must also have ``CAN_USE`` granted on the warehouse, which is an
    admin permission grant, not something this function does (see the module
    docstring).

    Args:
        experiment_id: Target MLflow experiment id.
        warehouse_id: SQL warehouse id to use for trace reads.
        client: Optional :class:`mlflow.MlflowClient`. Injectable for tests; when
            ``None`` a default client is constructed (using the ambient MLflow
            tracking configuration).
        set_env: Also set :data:`TRACING_WAREHOUSE_ENV` in the current process
            (default ``True``). The function always overwrites it: the caller is
            configuring this warehouse deliberately.

    Returns:
        A :class:`MonitoringWarehouseConfig` recording what was set.

    Raises:
        ValueError: if ``experiment_id`` or ``warehouse_id`` is blank.
    """
    if not experiment_id or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty experiment id")
    if not warehouse_id or not warehouse_id.strip():
        raise ValueError("warehouse_id must be a non-empty SQL warehouse id")

    if client is None:
        from mlflow import MlflowClient

        client = MlflowClient()
    client.set_experiment_tag(experiment_id, MONITORING_WAREHOUSE_TAG, warehouse_id)

    if set_env:
        os.environ[TRACING_WAREHOUSE_ENV] = warehouse_id

    return MonitoringWarehouseConfig(experiment_id=experiment_id, warehouse_id=warehouse_id)
