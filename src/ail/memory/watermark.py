"""Idempotency watermark for the distiller: the last-processed assessment
timestamp per ``experiment:cohort`` scope.

Each run reads the watermark, asks :func:`ail.memory.assessments.read_assessments`
for only the feedback created strictly after it, and — on success — advances the
watermark to the newest ``created_at`` it saw. A re-run over the same window then
finds nothing new and writes zero rows, so memory never accrues duplicates.

The watermark lives in the small governed table
:data:`ail.memory.schema.WATERMARK_TABLE` (created by the same bootstrap machinery
as ``agent_memory``). Reads/writes go through the shared statement seams
(:func:`ail.publish._execute` / :func:`ail.jobs.bootstrap_tables._read_rows`), and
the per-scope row is upserted atomically with a single ``MERGE`` keyed on ``scope``
(the same pattern :func:`ail.memory.writeback.build_memory_merge` uses for
``agent_memory``) so a run never leaves a torn watermark.
"""

from __future__ import annotations

from typing import Any

from ail.memory.schema import WATERMARK_COLUMNS, WATERMARK_TABLE
from ail.publish import _execute, _lit


def watermark_scope(experiment_id: str, cohort: str) -> str:
    """The watermark key: one row per ``experiment_id:cohort``."""
    return f"{experiment_id}:{cohort}"


def _fqn(catalog: str, schema: str) -> str:
    return f"`{catalog}`.`{schema}`.{WATERMARK_TABLE}"


def read_watermark(
    client: Any,
    warehouse_id: str,
    *,
    catalog: str,
    schema: str,
    scope: str,
) -> str | None:
    """The last-processed ``created_at`` for ``scope``, or ``None`` on first run.

    ``None`` means "no lower bound yet" — the first run distils the most recent
    window and seeds the watermark.
    """
    from ail.jobs.bootstrap_tables import _read_rows

    query = (
        f"SELECT last_created_at FROM {_fqn(catalog, schema)} WHERE scope = {_lit(scope)} LIMIT 1"
    )
    rows = _read_rows(client, warehouse_id, query)
    if not rows:
        return None
    value = rows[0].get("last_created_at")
    return str(value) if value else None


def build_watermark_merge(
    catalog: str,
    schema: str,
    *,
    scope: str,
    last_created_at: str,
    run_at: str,
    n_assessments_seen: int,
    n_memories_written: int,
    n_dropped_provenance: int,
) -> str:
    """An idempotent, escaped single-row ``MERGE`` that upserts the ``scope`` row.

    Databricks SQL rejects ``INSERT ... REPLACE WHERE`` combined with an explicit
    column list, so the per-scope upsert is a ``MERGE`` keyed on ``scope`` —
    ``WHEN MATCHED THEN UPDATE`` advances an existing watermark, ``WHEN NOT MATCHED
    THEN INSERT`` seeds it on the first run — one atomic Delta transaction, exactly
    one row per scope. Mirrors :func:`ail.memory.writeback.build_memory_merge`:
    column list/order come from :data:`WATERMARK_COLUMNS` and every value is escaped
    via :func:`ail.publish._lit`, never interpolated raw.
    """
    fqn = _fqn(catalog, schema)
    cols = ", ".join(WATERMARK_COLUMNS)
    source_cols = ", ".join(f"s.{c}" for c in WATERMARK_COLUMNS)
    update_set = ", ".join(f"{c} = s.{c}" for c in WATERMARK_COLUMNS if c != "scope")
    values = ", ".join(
        _lit(v)
        for v in (
            scope,
            last_created_at,
            run_at,
            n_assessments_seen,
            n_memories_written,
            n_dropped_provenance,
        )
    )
    # Column aliases are illegal directly on a MERGE ``USING`` source, so name the
    # columns on the inner derived table (VALUES ... AS v(cols)) and give the USING
    # source a bare alias — the same shape as ``build_memory_merge``.
    return (
        f"MERGE INTO {fqn} AS t\n"
        f"USING (SELECT * FROM (VALUES ({values})) AS v ({cols})) AS s\n"
        "ON t.scope = s.scope\n"
        f"WHEN MATCHED THEN UPDATE SET {update_set}\n"
        f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({source_cols})"
    )


def write_watermark(
    client: Any,
    warehouse_id: str,
    *,
    catalog: str,
    schema: str,
    scope: str,
    last_created_at: str,
    run_at: str,
    n_assessments_seen: int,
    n_memories_written: int,
    n_dropped_provenance: int,
) -> None:
    """Atomically upsert the watermark row for ``scope`` via :func:`build_watermark_merge`.

    The ``MERGE`` upserts on ``scope`` so exactly the one scope's row is created on
    the first run and overwritten thereafter, in a single Delta transaction — never a
    delete-then-insert gap, and never more than one row per scope.
    """
    _execute(
        client,
        warehouse_id,
        build_watermark_merge(
            catalog,
            schema,
            scope=scope,
            last_created_at=last_created_at,
            run_at=run_at,
            n_assessments_seen=n_assessments_seen,
            n_memories_written=n_memories_written,
            n_dropped_provenance=n_dropped_provenance,
        ),
    )
