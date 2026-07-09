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
    FRAMEWORK_TABLES,
    ensure_app_tables,
    reconcile_app_table_columns,
    table_ensure_statements,
)
from ail.loop.publish_proposals import PROPOSALS_TABLE
from ail.loop.publish_proposals import _ddl as proposals_ddl
from ail.publish_versions import REGISTRY_TABLE
from ail.publish_versions import _ddl as versions_ddl

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

    # A CREATE TABLE for exactly the app-referenced set PLUS the framework tables
    # the bootstrap also owns (advisory memory + its watermark — created/migrated
    # but not app-read); the returned coverage is the app-read set only, proving
    # every covered app table is actually created.
    assert _created_tables(stmts) == set(APP_QUERY_TABLES) | set(FRAMEWORK_TABLES)
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
        # compound: leading valid create, then an appended destructive statement.
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.agent_registry (a STRING) USING DELTA; "
        "DROP TABLE `cat`.`sch`.agent_registry",
        # single-string variant: valid prefix + trailing ';' + second statement.
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) USING DELTA;"
        "TRUNCATE TABLE `cat`.`sch`.t",
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


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE SCHEMA IF NOT EXISTS `cat`.`sch` COMMENT 'x'",
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) USING DELTA",
        "create table if not exists `cat`.`sch`.t (a string) using delta",  # case
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) USING DELTA;",  # lone trailing ;
        # forbidden words appear ONLY inside a COMMENT literal -> must be accepted.
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) "
        "COMMENT 'we never drop or replace this'",
    ],
)
def test_is_idempotent_create_accepts(statement: str) -> None:
    assert bootstrap_tables._is_idempotent_create(statement) is True


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE `cat`.`sch`.t",
        "ALTER TABLE `cat`.`sch`.t ADD COLUMN x STRING",
        "TRUNCATE TABLE `cat`.`sch`.t",
        "CREATE OR REPLACE TABLE `cat`.`sch`.t (a STRING) USING DELTA",
        "CREATE TABLE `cat`.`sch`.t (a STRING) USING DELTA",  # missing IF NOT EXISTS
        # compound / appended second statement — prefix matches but body is not clean.
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) USING DELTA; DROP TABLE `cat`.`sch`.t",
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) USING DELTA;"
        "INSERT INTO `cat`.`sch`.t VALUES (1)",
    ],
)
def test_is_idempotent_create_rejects(statement: str) -> None:
    assert bootstrap_tables._is_idempotent_create(statement) is False


def _main_is_idempotent_create(statement: str) -> bool:
    """Byte-for-byte reproduction of main's PRE-REFACTOR ``_is_idempotent_create``.

    Main stripped comments with a CASE-SENSITIVE ``re.sub(r"COMMENT '[^']*'", ...)``
    applied to the already-``.upper()``-ed statement. The refactor to a shared
    ``_COMMENT_LITERAL_RE`` added ``IGNORECASE`` and (on this branch) doubled-quote
    -escape handling; the CREATE allowlist's verdict must not have shifted. This
    oracle uses the module's OWN ``_ALLOWED_PREFIXES``/``_FORBIDDEN_VERBS`` (the
    refactor left those untouched) so only the comment-strip differs.
    """
    normalized = " ".join(statement.split()).upper()
    if not normalized.startswith(bootstrap_tables._ALLOWED_PREFIXES):
        return False
    stripped = re.sub(r"COMMENT '[^']*'", "", normalized)
    single = stripped[:-1].rstrip() if stripped.endswith(";") else stripped
    if ";" in single:
        return False
    body = f" {single} "
    return not any(verb in body for verb in bootstrap_tables._FORBIDDEN_VERBS)


@pytest.mark.parametrize(
    "statement",
    [
        # Mixed-case `comment` carrying a forbidden verb — the literal Finding 2
        # calls out. Main upper-cases before the (case-sensitive) match, so it
        # strips the comment and ACCEPTS; the guard must produce the same verdict.
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) comment 'we drop nothing'",
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING) CoMmEnT 'please replace me'",
        # Doubled-quote escape + an interior ';': main's `[^']*` stops at the first
        # quote and (falsely) REJECTS on the leftover ';'. The CREATE guard must
        # reproduce that verdict — i.e. it must NOT inherit the shared regex's
        # doubled-quote fix (which would strip the whole comment and ACCEPT).
        "CREATE TABLE IF NOT EXISTS `cat`.`sch`.t (a STRING COMMENT 'owner''s; note')",
    ],
)
def test_is_idempotent_create_matches_pre_refactor_verdict(statement: str) -> None:
    # Provably identical to main: the guard agrees with the pre-refactor oracle on
    # every statement, so the shared-regex refactor did not shift the allowlist.
    assert bootstrap_tables._is_idempotent_create(statement) == _main_is_idempotent_create(
        statement
    )


# ==========================================================================
# Additive column reconciliation
# ==========================================================================
#
# The whole SQL client is faked (no live workspace): information_schema probes
# return configured LIVE columns per table, and CREATE/ALTER statements are
# recorded so the assertions can inspect exactly what would run.

#: The nine columns L7b-1 (change_*) and L9 (verify_*) added to the writer DDL —
#: the real regression this reconcile step exists to migrate onto a pre-existing
#: ``agent_proposed_actions`` table.
_NINE_ADDED = [
    "change_plan",
    "change_preview_diff",
    "change_produced_change_ref",
    "verify_requested",
    "verify_status",
    "verify_requested_by",
    "verify_requested_at",
    "verify_completed_at",
    "verify_error",
]

# Independent column scan (does NOT use the module's parser) so the reconcile
# assertions can't pass merely because the parser and the test agree on a bug.
_DDL_COLUMN_RE = re.compile(
    r"^\s*(\w+)\s+(STRING|INT|BIGINT|DOUBLE|BOOLEAN|FLOAT|LONG|TIMESTAMP|DATE)\b",
    re.IGNORECASE,
)


def _declared_columns(create_stmt: str) -> list[tuple[str, str]]:
    """`(name, ddl_type)` per column, scanned line-by-line from a CREATE TABLE."""
    cols: list[tuple[str, str]] = []
    for line in create_stmt.splitlines():
        m = _DDL_COLUMN_RE.match(line)
        if m:
            cols.append((m.group(1), m.group(2)))
    return cols


def _proposals_create() -> str:
    stmts = proposals_ddl("cat", "sch")
    return next(s for s in stmts if s.strip().upper().startswith("CREATE TABLE"))


#: The three source-of-truth columns Slice 1 added to agent_registry — the real
#: regression the reconcile step must migrate onto a pre-existing table.
_REGISTRY_ADDED = ["goal_config_json", "annotations_table", "target_workspace"]


def _registry_create() -> str:
    stmts = versions_ddl("cat", "sch")
    return next(
        s
        for s in stmts
        if s.strip().upper().startswith("CREATE TABLE") and f".{REGISTRY_TABLE} (" in s
    )


# -- fakes: an information_schema-aware SQL client -------------------------

_INFO_SCHEMA_TABLE_RE = re.compile(r"LOWER\(table_name\)\s*=\s*LOWER\('([^']+)'\)", re.IGNORECASE)


class _FakeCol:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeManifestSchema:
    def __init__(self, names: list[str]) -> None:
        self.columns = [_FakeCol(n) for n in names]


class _FakeManifest:
    def __init__(self, names: list[str]) -> None:
        self.schema = _FakeManifestSchema(names)


class _FakeResultSet:
    def __init__(self, data: list[list[str]]) -> None:
        self.data_array = data


class _FakeQueryResponse:
    def __init__(
        self,
        statement_id: str,
        state: StatementState,
        manifest: _FakeManifest | None = None,
        result: _FakeResultSet | None = None,
    ) -> None:
        self.statement_id = statement_id
        self.status = _FakeStatementStatus(state)
        self.manifest = manifest
        self.result = result


class _ReconcileStatementExecutionAPI:
    """Serves information_schema.columns probes from a configured per-table map;
    records every other (CREATE/ALTER) statement."""

    def __init__(self, live_columns: dict[str, list[tuple[str, str]]]) -> None:
        self._live = live_columns
        self.statements: list[str] = []

    def execute_statement(
        self, *, warehouse_id: str, statement: str, wait_timeout: str = "50s"
    ) -> _FakeQueryResponse:
        self.statements.append(statement)
        if "information_schema.columns" in statement.lower():
            m = _INFO_SCHEMA_TABLE_RE.search(statement)
            table = m.group(1) if m else ""
            rows = self._live.get(table, [])
            return _FakeQueryResponse(
                "stmt-q",
                StatementState.SUCCEEDED,
                _FakeManifest(["column_name", "full_data_type"]),
                _FakeResultSet([[name, full_type] for name, full_type in rows]),
            )
        return _FakeQueryResponse("stmt-1", StatementState.SUCCEEDED)

    def get_statement(self, statement_id: str) -> _FakeQueryResponse:
        return _FakeQueryResponse(statement_id, StatementState.SUCCEEDED)


class _ReconcileClient:
    def __init__(self, live_columns: dict[str, list[tuple[str, str]]]) -> None:
        self.statement_execution = _ReconcileStatementExecutionAPI(live_columns)


# -- the real regression + idempotence + skip -----------------------------


def test_reconcile_migrates_missing_columns_on_existing_table() -> None:
    declared = _declared_columns(_proposals_create())
    declared_names = [n for n, _ in declared]
    # sanity: the writer DDL really declares all nine added columns
    assert set(_NINE_ADDED) <= set(declared_names)
    # LIVE = a pre-existing table missing exactly the nine L7b-1/L9 columns; the
    # rest present with their declared types (lower-cased, as information_schema
    # reports full_data_type).
    live_rows = [(n, t.lower()) for n, t in declared if n not in _NINE_ADDED]
    client = _ReconcileClient({PROPOSALS_TABLE: live_rows})

    alters = reconcile_app_table_columns(client, "wh-1", catalog="cat", schema="sch")

    # Exactly ONE ALTER (only the proposals table needed migrating; the eight
    # other app tables are absent live -> skipped).
    assert len(alters) == 1
    alter = alters[0]
    assert bootstrap_tables._is_add_columns_alter(alter)
    assert alter.startswith(f"ALTER TABLE `cat`.`sch`.{PROPOSALS_TABLE} ADD COLUMNS (")
    block = alter[alter.index("(") + 1 : alter.rindex(")")]
    added = {seg.split()[0] for seg in block.split(", ")}
    assert added == set(_NINE_ADDED)
    # name+type preserved — the one BOOLEAN among the nine and a representative STRING.
    assert "verify_requested BOOLEAN" in block
    assert "change_plan STRING" in block
    # and it was actually executed on the warehouse.
    assert alter in client.statement_execution.statements


def test_reconcile_migrates_registry_source_of_truth_columns() -> None:
    # Slice 1's regression: an EXISTING agent_registry predates goal_config_json /
    # annotations_table / target_workspace. The reconcile must ADD exactly those
    # three before anything reads them — proving agent_registry is covered by the
    # bootstrap auto-migration (via _versions_ddl in _DDL_PRODUCERS).
    declared = _declared_columns(_registry_create())
    declared_names = [n for n, _ in declared]
    assert set(_REGISTRY_ADDED) <= set(declared_names)
    # LIVE = pre-existing table missing exactly the three new columns.
    live_rows = [(n, t.lower()) for n, t in declared if n not in _REGISTRY_ADDED]
    client = _ReconcileClient({REGISTRY_TABLE: live_rows})

    alters = reconcile_app_table_columns(client, "wh-1", catalog="cat", schema="sch")

    # Exactly one ALTER (only agent_registry present live; the other tables absent).
    assert len(alters) == 1
    alter = alters[0]
    assert bootstrap_tables._is_add_columns_alter(alter)
    assert alter.startswith(f"ALTER TABLE `cat`.`sch`.{REGISTRY_TABLE} ADD COLUMNS (")
    block = alter[alter.index("(") + 1 : alter.rindex(")")]
    added = {seg.split()[0] for seg in block.split(", ")}
    assert added == set(_REGISTRY_ADDED)
    # name+type preserved for a migrated column, and actually executed.
    assert "goal_config_json STRING" in block
    assert "target_workspace STRING" in block
    assert alter in client.statement_execution.statements


def test_reconcile_is_noop_when_all_columns_present() -> None:
    declared = _declared_columns(_proposals_create())
    live_rows = [(n, t.lower()) for n, t in declared]  # nothing missing
    client = _ReconcileClient({PROPOSALS_TABLE: live_rows})

    alters = reconcile_app_table_columns(client, "wh-1", catalog="cat", schema="sch")

    assert alters == []
    assert not any(
        s.strip().upper().startswith("ALTER") for s in client.statement_execution.statements
    )


def test_reconcile_skips_absent_tables() -> None:
    client = _ReconcileClient({})  # no table present live

    alters = reconcile_app_table_columns(client, "wh-1", catalog="cat", schema="sch")

    assert alters == []
    stmts = client.statement_execution.statements
    # It probed information_schema but issued no DDL (CREATE path is unaffected).
    assert stmts
    assert all("information_schema.columns" in s.lower() for s in stmts)


# -- fail loud on a real type conflict; tolerate benign spelling ----------


def test_reconcile_raises_on_real_type_conflict() -> None:
    declared = _declared_columns(_proposals_create())
    # All present, but agent_name (declared STRING) is BIGINT live -> unreconcilable.
    live_rows = [
        ("agent_name", "bigint") if n == "agent_name" else (n, t.lower()) for n, t in declared
    ]
    client = _ReconcileClient({PROPOSALS_TABLE: live_rows})

    with pytest.raises(ValueError) as excinfo:
        reconcile_app_table_columns(client, "wh-1", catalog="cat", schema="sch")

    message = str(excinfo.value)
    assert PROPOSALS_TABLE in message
    assert "agent_name" in message
    # Fail-closed: nothing was ALTERed.
    assert not any(
        s.strip().upper().startswith("ALTER") for s in client.statement_execution.statements
    )


def test_reconcile_tolerates_case_only_type_diff() -> None:
    declared = _declared_columns(_proposals_create())
    # agent_name live type differs only by case (declared STRING vs live STRING/uppercase);
    # everything present -> no conflict, no migration.
    live_rows = [(n, "STRING") if n == "agent_name" else (n, t.lower()) for n, t in declared]
    client = _ReconcileClient({PROPOSALS_TABLE: live_rows})

    assert reconcile_app_table_columns(client, "wh-1", catalog="cat", schema="sch") == []


# -- parser: comments and complex types don't break column splitting ------


def test_parse_create_table_handles_comments_and_complex_types() -> None:
    create = (
        "CREATE TABLE IF NOT EXISTS `c`.`s`.tricky (\n"
        "  a STRING COMMENT 'has, a comma and (parens) and the word DROP',\n"
        "  b DECIMAL(10, 2),\n"
        "  c MAP<STRING,STRING>,\n"
        "  d ARRAY<STRUCT<x:INT,y:STRING>>,\n"
        "  e STRING\n"
        ") USING DELTA COMMENT 'table, comment with ) a paren'"
    )
    parsed = bootstrap_tables._parse_create_table(create)
    assert parsed is not None
    fqn, table, declared = parsed
    assert fqn == "`c`.`s`.tricky"
    assert table == "tricky"

    names = [n for n, _def, _t in declared]
    assert names == ["a", "b", "c", "d", "e"]  # comment/type commas did NOT split

    types = {n: t for n, _def, t in declared}
    assert types["a"] == "STRING"  # trailing COMMENT stripped from the type
    assert types["b"] == "DECIMAL(10, 2)"
    assert types["c"] == "MAP<STRING,STRING>"
    assert types["d"] == "ARRAY<STRUCT<x:INT,y:STRING>>"

    # the full def keeps the COMMENT verbatim, so a migrated column is identical.
    a_def = next(full for n, full, _t in declared if n == "a")
    assert "COMMENT 'has, a comma and (parens) and the word DROP'" in a_def


def test_parse_create_table_handles_doubled_quote_escape_in_comment() -> None:
    # SQL escapes a literal single-quote inside a string by DOUBLING it (`''`).
    # The comment-literal regex must span the whole `COMMENT 'owner''s, note'`
    # (through the `''` escape) so its interior comma can't be read as a top-level
    # column separator and mis-split `note` into bogus columns.
    create = (
        "CREATE TABLE IF NOT EXISTS `c`.`s`.t (\n"
        "  note STRING COMMENT 'owner''s, note',\n"
        "  b INT\n"
        ") USING DELTA"
    )
    parsed = bootstrap_tables._parse_create_table(create)
    assert parsed is not None
    _fqn, _table, declared = parsed

    names = [n for n, _def, _t in declared]
    assert names == ["note", "b"]  # the escaped-quote comment's comma did NOT split

    types = {n: t for n, _def, t in declared}
    assert types["note"] == "STRING"  # whole COMMENT (incl. '' escape) stripped from the type

    # the full def keeps the escaped comment verbatim, so a migrated column is identical.
    note_def = next(full for n, full, _t in declared if n == "note")
    assert "COMMENT 'owner''s, note'" in note_def


def test_parse_create_table_returns_none_for_create_schema() -> None:
    assert bootstrap_tables._parse_create_table("CREATE SCHEMA IF NOT EXISTS `c`.`s`") is None


# -- the ALTER allowlist: only ADD COLUMNS may ever run -------------------


@pytest.mark.parametrize(
    "statement",
    [
        "ALTER TABLE `c`.`s`.t ADD COLUMNS (a STRING, b INT)",
        "ALTER TABLE `c`.`s`.t ADD COLUMNS (a STRING);",  # lone trailing ;
        # forbidden words appear ONLY inside a COMMENT literal -> accepted.
        "ALTER TABLE `c`.`s`.t ADD COLUMNS (a STRING COMMENT 'we never drop or rename this')",
        # complex type with an inner comma inside the block.
        "ALTER TABLE `c`.`s`.t ADD COLUMNS (m MAP<STRING,STRING>)",
    ],
)
def test_is_add_columns_alter_accepts(statement: str) -> None:
    assert bootstrap_tables._is_add_columns_alter(statement) is True


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE `c`.`s`.t",
        "ALTER TABLE `c`.`s`.t DROP COLUMN a",
        "ALTER TABLE `c`.`s`.t RENAME COLUMN a TO b",
        "ALTER TABLE `c`.`s`.t ALTER COLUMN a TYPE STRING",
        "ALTER TABLE `c`.`s`.t ADD COLUMN a STRING",  # singular COLUMN, not COLUMNS
        "CREATE TABLE IF NOT EXISTS `c`.`s`.t (a STRING) USING DELTA",
        # appended second statement after a valid ADD COLUMNS.
        "ALTER TABLE `c`.`s`.t ADD COLUMNS (a STRING); DROP TABLE `c`.`s`.t",
        # trailing clause after the ADD COLUMNS block.
        "ALTER TABLE `c`.`s`.t ADD COLUMNS (a STRING) DROP COLUMN b",
    ],
)
def test_is_add_columns_alter_rejects(statement: str) -> None:
    assert bootstrap_tables._is_add_columns_alter(statement) is False
