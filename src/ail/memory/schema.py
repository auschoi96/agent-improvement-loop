"""The ``agent_memory`` system-of-record table (plus its idempotency watermark),
authored as a single :func:`_ddl` producer registered with the bootstrap machinery.

Single source of truth, no hand-DDL
------------------------------------
Like every other framework table, ``agent_memory``'s schema is defined ONCE here
and registered with :data:`ail.jobs.bootstrap_tables._DDL_PRODUCERS`, so the
bootstrap creates it (empty) and additively column-migrates it on upgrade deploys
using this module's own ``_ddl()`` — there is no second schema definition to drift
and no separate hand-run ``CREATE`` anywhere.

The bootstrap runtime allowlist only ever executes ``CREATE SCHEMA/TABLE IF NOT
EXISTS`` (never ``DROP``/``ALTER``/``CREATE OR REPLACE``), so :func:`_ddl` emits
exactly those shapes. The ``CREATE SCHEMA`` statement is byte-identical to the one
the L0/version/lineage/proposal producers emit, so ``table_ensure_statements``
dedupes it to a single leading statement.

``agent_memory`` is NOT read by the deployed app (the read / injection side is a
separate system), so it is intentionally absent from
:data:`ail.jobs.bootstrap_tables.APP_QUERY_TABLES` and lives in
:data:`ail.jobs.bootstrap_tables.FRAMEWORK_TABLES` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The advisory-memory system-of-record table (one row per distilled guideline).
MEMORY_TABLE = "agent_memory"

#: The small idempotency-watermark table: the last-processed assessment timestamp
#: per (experiment, cohort) scope so a re-run distils only NEW feedback.
WATERMARK_TABLE = "agent_memory_watermark"

#: The ``agent_memory`` column order, declared once and reused by both the DDL and
#: the MERGE builder (:func:`ail.memory.writeback.build_memory_merge`) so the two
#: can never drift.
MEMORY_COLUMNS: list[str] = [
    "memory_id",
    "cohort",
    "category",
    "guideline_text",
    "score",
    "source_trace_ids",
    "source_signal",
    "created_at",
    "embedding",
]

#: The ``agent_memory_watermark`` column order (see :mod:`ail.memory.watermark`).
WATERMARK_COLUMNS: list[str] = [
    "scope",
    "last_created_at",
    "last_run_at",
    "n_assessments_seen",
    "n_memories_written",
    "n_dropped_provenance",
]


@dataclass(frozen=True, slots=True)
class MemoryRow:
    """One distilled advisory-memory guideline, ready to write to ``agent_memory``.

    ``source_trace_ids`` is the provenance the wall checks against the frozen
    pools; a row must cite at least one. ``score`` is a 0–1 confidence. ``embedding``
    is populated later by the (out-of-scope) semantic-retrieval side, so it is
    ``None`` at write time here.
    """

    memory_id: str
    cohort: str
    category: str
    guideline_text: str
    score: float
    source_trace_ids: tuple[str, ...]
    source_signal: str
    created_at: str
    embedding: tuple[float, ...] | None = field(default=None)


def _ddl(catalog: str, schema: str) -> list[str]:
    """``CREATE SCHEMA/TABLE IF NOT EXISTS`` for ``agent_memory`` + its watermark.

    The leading ``CREATE SCHEMA`` is byte-identical to the other producers' so the
    bootstrap dedupes it. Registered in
    :data:`ail.jobs.bootstrap_tables._DDL_PRODUCERS`.
    """
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{MEMORY_TABLE} (
            memory_id STRING,
            cohort STRING,
            category STRING,
            guideline_text STRING,
            score DOUBLE,
            source_trace_ids ARRAY<STRING>,
            source_signal STRING,
            created_at STRING,
            embedding ARRAY<FLOAT>
        ) USING DELTA
        COMMENT 'Advisory memory: one distilled guideline per feedback signal.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{WATERMARK_TABLE} (
            scope STRING,
            last_created_at STRING,
            last_run_at STRING,
            n_assessments_seen BIGINT,
            n_memories_written BIGINT,
            n_dropped_provenance BIGINT
        ) USING DELTA
        COMMENT 'Advisory-memory watermark: last assessment created_at per scope.'""",
    ]
