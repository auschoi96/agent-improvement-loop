"""Unit tests for the app-table bootstrap (:mod:`ail.jobs.bootstrap_tables`).

Two guarantees, both offline (the SQL client is faked; the ``.sql`` files are read
straight from the repo):

* the bootstrap issues the writer-owned ``_ddl()`` ``CREATE ... IF NOT EXISTS``
  statements for the **full** app-referenced table set — never a destructive
  ``DROP``/``ALTER`` and never a hand-authored schema; and
* a **drift guard** that the covered table set (:data:`APP_QUERY_TABLES`) is
  exactly the set of tables the deployed app's ``config/queries/*.sql`` read (the
  set typegen runs ``DESCRIBE QUERY`` against), so the two can never silently
  diverge as queries or tables change.
"""

from __future__ import annotations

import re
from pathlib import Path

from databricks.sdk.service.sql import StatementState

from ail.jobs.bootstrap_tables import (
    APP_QUERY_TABLES,
    ensure_app_tables,
    table_ensure_statements,
)

# The deployed app's SQL query registry — the source of truth for the drift
# guard. typegen runs a live DESCRIBE QUERY against each of these at build time.
_QUERY_DIR = Path(__file__).resolve().parents[1] / "ail-self-optimizer" / "config" / "queries"


# -- fakes -----------------------------------------------------------------


class _FakeStatementStatus:
    def __init__(self, state: StatementState) -> None:
        self.state = state
        self.error = None


class _FakeStatementResponse:
    def __init__(self, statement_id: str, state: StatementState) -> None:
        self.statement_id = statement_id
        self.status = _FakeStatementStatus(state)


class _RecordingStatementExecutionAPI:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_statement(
        self, *, warehouse_id: str, statement: str, wait_timeout: str = "50s"
    ) -> _FakeStatementResponse:
        self.statements.append(statement)
        return _FakeStatementResponse("stmt-1", StatementState.SUCCEEDED)

    def get_statement(self, statement_id: str) -> _FakeStatementResponse:
        return _FakeStatementResponse(statement_id, StatementState.SUCCEEDED)


class _RecordingClient:
    def __init__(self) -> None:
        self.statement_execution = _RecordingStatementExecutionAPI()


# -- helpers ---------------------------------------------------------------


def _created_tables(statements: list[str]) -> set[str]:
    """Bare table names from every ``CREATE TABLE IF NOT EXISTS`` statement."""
    tables: set[str] = set()
    for stmt in statements:
        m = re.search(
            r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`[^`]+`\.`[^`]+`\.(\w+)",
            stmt,
            re.IGNORECASE,
        )
        if m:
            tables.add(m.group(1))
    return tables


def _tables_referenced(sql_text: str) -> set[str]:
    """Schema-qualified tables read by a query (comments stripped, aliases/CTEs ignored)."""
    no_comments = re.sub(r"--[^\n]*", "", sql_text)
    tables: set[str] = set()
    for ref in re.findall(r'(?:\bFROM\b|\bJOIN\b)\s+([`"\w.]+)', no_comments, re.IGNORECASE):
        # Only qualified (catalog.schema.table) refs are real tables; an
        # unqualified name is a CTE/alias, not a table the app must ensure.
        if "." in ref:
            tables.add(ref.split(".")[-1].strip('`"'))
    return tables


# -- DDL: full set, idempotent, no guessed/destructive schema --------------


def test_ensure_app_tables_issues_writer_ddl_for_full_set() -> None:
    client = _RecordingClient()
    covered = ensure_app_tables(client, "wh-1", catalog="cat", schema="sch")
    stmts = client.statement_execution.statements

    # Something ran, and every statement is an idempotent CREATE SCHEMA/TABLE.
    assert stmts
    for stmt in stmts:
        head = stmt.strip().split()
        assert head[0].upper() == "CREATE"
        assert head[1].upper() in {"SCHEMA", "TABLE"}
        assert "IF NOT EXISTS" in stmt.upper()

    # Fail-closed: no destructive / mutating verb anywhere in the DDL body
    # (COMMENT prose stripped so it can't trip a substring match).
    forbidden_verbs = (
        " DROP ",
        " ALTER ",
        " INSERT ",
        " DELETE ",
        " TRUNCATE ",
        " MERGE ",
        " UPDATE ",
        " REPLACE ",
    )
    for stmt in stmts:
        without_comment = re.sub(r"COMMENT '[^']*'", "", stmt)
        body = f" {without_comment.upper()} "
        for forbidden in forbidden_verbs:
            assert forbidden not in body, f"unexpected {forbidden!r} in: {stmt[:80]}"

    # A CREATE TABLE for exactly the full app-referenced set, and the returned
    # coverage matches — proving every covered table is actually created.
    assert _created_tables(stmts) == set(APP_QUERY_TABLES)
    assert set(covered) == set(APP_QUERY_TABLES)

    # The DDL targets the caller's catalog.schema — not a hardcoded default.
    assert all("`cat`.`sch`" in stmt for stmt in stmts)


def test_schema_is_created_once_before_tables() -> None:
    stmts = table_ensure_statements("cat", "sch")
    schema_creates = [s for s in stmts if s.upper().startswith("CREATE SCHEMA")]
    # The four writer _ddl() producers each emit an identical CREATE SCHEMA; it is
    # deduped to a single statement, and it comes first.
    assert len(schema_creates) == 1
    assert stmts[0].upper().startswith("CREATE SCHEMA IF NOT EXISTS")


# -- drift guard: coverage == what the app queries actually read -----------


def test_covered_set_matches_app_query_files() -> None:
    sql_files = sorted(_QUERY_DIR.glob("*.sql"))
    assert sql_files, f"no app SQL query files found under {_QUERY_DIR}"

    queried: set[str] = set()
    for sql_file in sql_files:
        queried |= _tables_referenced(sql_file.read_text())

    covered = set(APP_QUERY_TABLES)
    assert queried == covered, (
        "bootstrap table coverage drifted from the app's SQL queries.\n"
        f"  only in queries (add its writer _ddl to bootstrap_tables): {queried - covered}\n"
        f"  only in bootstrap (no query reads it): {covered - queried}"
    )
