"""Append-only UC event tables that wake the asynchronous optimization lanes."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ail.publish import _build_workspace_client, _execute, _lit

ALIGNMENT_EVENTS_TABLE = "alignment_events"
MEMORY_EVENTS_TABLE = "memory_events"


def _ddl(catalog: str, schema: str) -> list[str]:
    schema_ddl = (
        f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}` "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'"
    )
    columns = """(
        event_id STRING,
        experiment_id STRING,
        source STRING,
        source_id STRING,
        actor STRING,
        created_at STRING
    ) USING DELTA"""
    return [
        schema_ddl,
        f"CREATE TABLE IF NOT EXISTS `{catalog}`.`{schema}`.{ALIGNMENT_EVENTS_TABLE} {columns}",
        f"CREATE TABLE IF NOT EXISTS `{catalog}`.`{schema}`.{MEMORY_EVENTS_TABLE} {columns}",
    ]


def append_event(
    *,
    table: str,
    experiment_id: str,
    source: str,
    source_id: str,
    warehouse_id: str,
    catalog: str,
    schema: str,
    actor: str = "",
    client: Any | None = None,
) -> str:
    """Append one immutable wake-up event and return its generated id."""
    if table not in {ALIGNMENT_EVENTS_TABLE, MEMORY_EVENTS_TABLE}:
        raise ValueError(f"unsupported event table {table!r}")
    required = (experiment_id, source, source_id, warehouse_id, catalog, schema)
    if not all(value.strip() for value in required):
        raise ValueError("event experiment/source/source_id/warehouse/catalog/schema are required")
    event_id = str(uuid4())
    workspace = client or _build_workspace_client(None)
    fqn = f"`{catalog}`.`{schema}`.`{table}`"
    _execute(
        workspace,
        warehouse_id,
        f"INSERT INTO {fqn} (event_id, experiment_id, source, source_id, actor, created_at) "
        f"VALUES ({_lit(event_id)}, {_lit(experiment_id)}, {_lit(source)}, "
        f"{_lit(source_id)}, {_lit(actor)}, CAST(current_timestamp() AS STRING))",
    )
    return event_id


def append_memory_event(**kwargs: Any) -> str:
    return append_event(table=MEMORY_EVENTS_TABLE, **kwargs)
