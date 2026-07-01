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

import pytest
from databricks.sdk.service.sql import StatementState

from ail.jobs import bootstrap_tables
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


# Tokens that may follow a table ref but are NOT an alias (clause keywords).
_NON_ALIAS_TOKENS = frozenset(
    {
        "ON", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "QUALIFY", "WINDOW",
        "UNION", "EXCEPT", "INTERSECT", "JOIN", "INNER", "LEFT", "RIGHT", "FULL",
        "CROSS", "USING", "AS", "WITH", "SELECT", "FROM", "CLUSTER", "DISTRIBUTE",
        "SORT", "LATERAL", "PIVOT", "UNPIVOT", "TABLESAMPLE",
    }
)  # fmt: skip


def _cte_names(sql: str) -> set[str]:
    """CTE names from ``WITH name AS (...)`` (incl. chained ``, name2 AS (...)``)."""
    return {name.lower() for name in re.findall(r"(\w+)\s+AS\s*\(", sql, re.IGNORECASE)}


def _alias_names(sql: str) -> set[str]:
    """Table aliases: the identifier after a QUALIFIED FROM/JOIN ref (optional ``AS``)."""
    aliases: set[str] = set()
    for m in re.finditer(
        r'(?:\bFROM\b|\bJOIN\b)\s+[`"\w]+\.[`"\w.]+\s+(?:AS\s+)?(\w+)',
        sql,
        re.IGNORECASE,
    ):
        token = m.group(1)
        if token.upper() not in _NON_ALIAS_TOKENS:
            aliases.add(token.lower())
    return aliases


def _referenced_tables(sql_text: str) -> tuple[set[str], set[str]]:
    """``(qualified tables read, unqualified FROM/JOIN refs that are NOT a CTE/alias)``.

    The second set is the drift guard's blind-spot detector: an unqualified real
    table (e.g. ``FROM agent_action_decisions``) is **not** silently dropped — it
    is surfaced so the drift test fails loudly until it is qualified or classified
    (as a CTE/alias). Qualified ``catalog.schema.table`` refs work as before.
    """
    no_comments = re.sub(r"--[^\n]*", "", sql_text)
    ctes = _cte_names(no_comments)
    aliases = _alias_names(no_comments)
    tables: set[str] = set()
    unclassified: set[str] = set()
    for ref in re.findall(r'(?:\bFROM\b|\bJOIN\b)\s+([`"\w.]+)', no_comments, re.IGNORECASE):
        if "." in ref:
            tables.add(ref.split(".")[-1].strip('`"'))
            continue
        name = ref.strip('`"')
        if name.lower() in ctes or name.lower() in aliases:
            continue
        unclassified.add(name)
    return tables, unclassified


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
    unclassified: set[str] = set()
    for sql_file in sql_files:
        tables, unknown = _referenced_tables(sql_file.read_text())
        queried |= tables
        unclassified |= unknown

    # Blind-spot guard: no unqualified FROM/JOIN ref may be silently ignored.
    assert not unclassified, (
        "unqualified FROM/JOIN reference(s) the drift guard can't classify as a "
        f"CTE/alias: {sorted(unclassified)}. Qualify them as catalog.schema.table "
        "(or add the CTE/alias) so bootstrap coverage stays enforceable."
    )

    covered = set(APP_QUERY_TABLES)
    assert queried == covered, (
        "bootstrap table coverage drifted from the app's SQL queries.\n"
        f"  only in queries (add its writer _ddl to bootstrap_tables): {queried - covered}\n"
        f"  only in bootstrap (no query reads it): {covered - queried}"
    )


def test_drift_guard_flags_unqualified_non_cte_table() -> None:
    # A future query that reads an UNQUALIFIED real table must not slip through as
    # if it were a CTE/alias — the guard surfaces it so the drift test fails loudly.
    tables, unclassified = _referenced_tables("SELECT * FROM agent_action_decisions WHERE x = 1")
    assert tables == set()
    assert unclassified == {"agent_action_decisions"}


def test_drift_guard_classifies_cte_and_captures_its_real_table() -> None:
    sql = (
        "WITH recent AS (SELECT * FROM cat.sch.real_table), "
        "extra AS (SELECT 1 AS id) "
        "SELECT * FROM recent JOIN extra ON recent.id = extra.id"
    )
    tables, unclassified = _referenced_tables(sql)
    # `recent`/`extra` are CTEs (not missing tables); the CTE body's real table is captured.
    assert tables == {"real_table"}
    assert unclassified == set()


def test_drift_guard_ignores_qualified_ref_with_alias() -> None:
    tables, unclassified = _referenced_tables(
        "SELECT t.a FROM cat.sch.real_table AS t WHERE t.a > 0"
    )
    assert tables == {"real_table"}
    assert unclassified == set()


# -- runtime allowlist: fail-closed at execution, not only in tests --------


@pytest.mark.parametrize(
    "bad_statement",
    [
        "DROP TABLE `cat`.`sch`.agent_registry",
        "ALTER TABLE `cat`.`sch`.agent_registry ADD COLUMN x STRING",
        "TRUNCATE TABLE `cat`.`sch`.agent_registry",
        "CREATE OR REPLACE TABLE `cat`.`sch`.agent_registry (a STRING) USING DELTA",
        "CREATE TABLE `cat`.`sch`.agent_registry (a STRING) USING DELTA",  # no IF NOT EXISTS
    ],
)
def test_ensure_app_tables_rejects_non_idempotent_and_executes_nothing(
    monkeypatch: pytest.MonkeyPatch, bad_statement: str
) -> None:
    def _drifted_ddl(catalog: str, schema: str) -> list[str]:
        return [
            f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`",
            bad_statement,
        ]

    monkeypatch.setattr(bootstrap_tables, "_DDL_PRODUCERS", (_drifted_ddl,))
    client = _RecordingClient()

    with pytest.raises(ValueError, match="non-idempotent-CREATE"):
        ensure_app_tables(client, "wh-1", catalog="cat", schema="sch")

    # Fail-closed: the FULL list is validated before any execution, so NOTHING ran
    # (not even the leading, valid CREATE SCHEMA) — no partial/destructive apply.
    assert client.statement_execution.statements == []


def test_table_ensure_statements_names_offending_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _drifted_ddl(catalog: str, schema: str) -> list[str]:
        return [f"DROP TABLE `{catalog}`.`{schema}`.agent_registry"]

    monkeypatch.setattr(bootstrap_tables, "_DDL_PRODUCERS", (_drifted_ddl,))

    with pytest.raises(ValueError) as excinfo:
        table_ensure_statements("cat", "sch")
    # The error names the producer so a drift is traceable to its writer module.
    assert "_drifted_ddl" in str(excinfo.value)
