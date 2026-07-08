"""``agent_memory`` is registered with the bootstrap machinery so it is created +
column-migrated like every other framework table — and is correctly kept OUT of the
app-query set (it is not read by the deployed app).
"""

from __future__ import annotations

from ail.jobs import bootstrap_tables
from ail.jobs.bootstrap_tables import (
    APP_QUERY_TABLES,
    FRAMEWORK_TABLES,
    _is_idempotent_create,
    _parse_create_table,
    table_ensure_statements,
)
from ail.memory.schema import MEMORY_COLUMNS, MEMORY_TABLE, WATERMARK_TABLE
from ail.memory.schema import _ddl as memory_ddl


def test_memory_ddl_is_registered_producer() -> None:
    assert memory_ddl in bootstrap_tables._DDL_PRODUCERS


def test_memory_tables_are_framework_not_app() -> None:
    assert MEMORY_TABLE in FRAMEWORK_TABLES
    assert WATERMARK_TABLE in FRAMEWORK_TABLES
    # Not read by the deployed app -> deliberately absent from APP_QUERY_TABLES so the
    # app-query drift guard stays exactly the app's SELECTs.
    assert MEMORY_TABLE not in APP_QUERY_TABLES
    assert WATERMARK_TABLE not in APP_QUERY_TABLES


def test_bootstrap_creates_memory_tables_idempotently() -> None:
    stmts = table_ensure_statements("cat", "sch")
    created = {p[1] for p in (_parse_create_table(s) for s in stmts) if p is not None}
    assert MEMORY_TABLE in created
    assert WATERMARK_TABLE in created
    # Every memory statement is an allowlisted idempotent CREATE (no DROP/ALTER).
    for stmt in stmts:
        assert _is_idempotent_create(stmt)


def test_agent_memory_columns_match_source_of_truth() -> None:
    stmts = table_ensure_statements("cat", "sch")
    parsed = next(
        p for p in (_parse_create_table(s) for s in stmts) if p is not None and p[1] == MEMORY_TABLE
    )
    declared = [name for name, _full, _type in parsed[2]]
    assert declared == MEMORY_COLUMNS


def test_memory_ddl_shares_one_create_schema() -> None:
    # The memory _ddl's CREATE SCHEMA must be byte-identical to the others so the
    # bootstrap still emits exactly one CREATE SCHEMA.
    stmts = table_ensure_statements("cat", "sch")
    schema_creates = [s for s in stmts if s.upper().startswith("CREATE SCHEMA")]
    assert len(schema_creates) == 1
