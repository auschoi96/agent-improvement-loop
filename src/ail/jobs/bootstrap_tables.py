"""Ensure every UC table the deployed app reads exists (empty) — the table half
of the post-deploy bootstrap.

Why this exists (a real incident this permanently prevents)
-----------------------------------------------------------
The deployed ``ail-self-optimizer`` AppKit app runs typegen
(``appkit generate-types``) during its build, and typegen performs a **live**
``DESCRIBE QUERY`` against the SQL warehouse for **every** app SQL query
(``ail-self-optimizer/config/queries/*.sql``). Several of those queries read
tables that are created **lazily on first write** — the loop controller creates
``agent_proposed_actions`` on its first proposal, the lineage publisher creates
``agent_prompt_lineage`` on first prompt promotion, and the per-(agent,version)
tables appear only once a version is published. On a fresh / clean workspace
(zero prior writes) those tables do not exist yet, so ``DESCRIBE QUERY`` fails
with ``TABLE_OR_VIEW_NOT_FOUND``, the app **build** fails, ``bundle run app``
fails, and the previously-running app goes **UNAVAILABLE**.

This module makes that impossible to recur without any manual DDL: it ensures
each table exists (empty) **before** the app build's typegen runs, using each
writer module's **own** authoritative ``_ddl()`` — never a schema authored here.
So the empty table has exactly the schema the writer will later populate, and
``DESCRIBE QUERY`` (and every runtime ``SELECT``) succeeds against it.

Single source of truth, no guessed schema
------------------------------------------
The ``CREATE SCHEMA/TABLE IF NOT EXISTS`` statements come **only** from the
writer modules that own each table (:data:`_DDL_PRODUCERS`). Reusing each
writer's ``_ddl()`` means a column added/renamed there flows here automatically;
there is no second schema definition to drift.

Fail-closed and idempotent
---------------------------
Every statement is ``CREATE ... IF NOT EXISTS`` — there is **no** ``DROP`` or
``ALTER`` anywhere. Re-running on a populated workspace is a no-op: an existing
table is never touched, so no data is lost and no schema is mutated. Creating a
schema/table requires the running identity to hold ``CREATE`` on the catalog /
schema — the same workspace-authority requirement the warehouse-create and grant
steps already carry (see :mod:`ail.jobs.bootstrap_grants`).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from ail.events import ALIGNMENT_EVENTS_TABLE, MEMORY_EVENTS_TABLE
from ail.events import _ddl as _events_ddl
from ail.jobs.onboarding_job import (
    ONBOARDING_REQUESTS_TABLE,
    ONBOARDING_RESULTS_TABLE,
)
from ail.jobs.onboarding_job import (
    _ddl as _onboarding_ddl,
)
from ail.loop.publish_proposals import PROPOSALS_TABLE
from ail.loop.publish_proposals import _ddl as _proposals_ddl
from ail.memory.schema import MEMORY_TABLE, WATERMARK_TABLE
from ail.memory.schema import _ddl as _memory_ddl
from ail.publish import (
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    DIAGNOSIS_TABLE,
    SESSION_TABLE,
    SUMMARY_TABLE,
    _execute,
)
from ail.publish import _ddl as _l0_ddl
from ail.publish_lineage import LINEAGE_TABLE
from ail.publish_lineage import _ddl as _lineage_ddl
from ail.publish_versions import (
    REGISTRY_TABLE,
    VERSION_COMPARISON_TABLE,
    VERSION_L0_TABLE,
    VERSION_READINESS_TABLE,
)
from ail.publish_versions import _ddl as _versions_ddl
from ail.requirements.persistence import COMPILED_GOAL_TABLE
from ail.requirements.persistence import _ddl as _compiled_goal_ddl

#: The writer-module ``_ddl()`` producers whose ``CREATE ... IF NOT EXISTS``
#: statements the deployed app depends on. Each entry is that writer's OWN
#: authoritative DDL — the single source of truth for its table schemas. Ordered
#: base-first (the ``l0_*`` tables) then the ``agent_*`` tables. NEVER inline a
#: hand-authored ``CREATE`` here; add the table to its writer's ``_ddl()`` and
#: reference the writer from this tuple instead.
_DDL_PRODUCERS: tuple[Callable[[str, str], list[str]], ...] = (
    _l0_ddl,  # l0_session_metrics, l0_corpus_summary, l0_diagnosis
    _versions_ddl,  # agent_registry, agent_version_{l0,comparison,readiness}
    _lineage_ddl,  # agent_prompt_lineage
    _proposals_ddl,  # agent_proposed_actions
    _memory_ddl,  # agent_memory, agent_memory_watermark (framework, not app-read)
    _compiled_goal_ddl,  # agent_compiled_goals (framework: intake->loop goal bridge)
    _onboarding_ddl,  # governed async onboarding request/result transport
    _events_ddl,  # append-only wake-up events for align + memory jobs
)

#: The exact set of tables the deployed app's ``config/queries/*.sql`` SELECT
#: from — i.e. the set AppKit typegen runs ``DESCRIBE QUERY`` against at build
#: time. Every entry is a writer-module table-name constant (no string literals),
#: so renaming a table in its writer updates this in lockstep. A drift-guard test
#: asserts this equals the tables parsed out of those ``.sql`` files, so the
#: bootstrap's coverage can never silently diverge from what the app reads.
APP_QUERY_TABLES: frozenset[str] = frozenset(
    {
        SESSION_TABLE,
        SUMMARY_TABLE,
        DIAGNOSIS_TABLE,
        REGISTRY_TABLE,
        VERSION_L0_TABLE,
        VERSION_COMPARISON_TABLE,
        VERSION_READINESS_TABLE,
        LINEAGE_TABLE,
        PROPOSALS_TABLE,
    }
)

#: Framework tables the bootstrap creates + column-migrates but the deployed app
#: does NOT ``SELECT`` from — so they are deliberately NOT in
#: :data:`APP_QUERY_TABLES` (which the drift guard pins to the app's
#: ``config/queries/*.sql``). The advisory-memory system of record
#: (:data:`ail.memory.schema.MEMORY_TABLE`) and its idempotency watermark
#: (:data:`~ail.memory.schema.WATERMARK_TABLE`) are written by the scheduled
#: distiller Job and read by the separate (out-of-scope) Lakebase/retrieval side,
#: never by AppKit typegen. They still need bootstrap create + additive migration,
#: so their ``_ddl`` producer is in :data:`_DDL_PRODUCERS`; this constant is how
#: the bootstrap coverage test accounts for the tables produced beyond the app set.
#: :data:`~ail.requirements.persistence.COMPILED_GOAL_TABLE` is the intake→loop goal
#: bridge — written by the confirmed-intake step, read by the optimization loop's
#: goal-load, never ``SELECT``ed by AppKit typegen, so it too is a framework table.
FRAMEWORK_TABLES: frozenset[str] = frozenset(
    {
        MEMORY_TABLE,
        WATERMARK_TABLE,
        COMPILED_GOAL_TABLE,
        ONBOARDING_REQUESTS_TABLE,
        ONBOARDING_RESULTS_TABLE,
        ALIGNMENT_EVENTS_TABLE,
        MEMORY_EVENTS_TABLE,
    }
)


#: The ONLY statement shapes the bootstrap is ever allowed to execute against a
#: live (admin-authority) workspace. The guarantee is enforced at runtime, not
#: just in tests: a writer ``_ddl()`` that ever drifts to anything else — a
#: ``DROP``/``ALTER``/``TRUNCATE``, a ``CREATE OR REPLACE``, or a bare ``CREATE
#: TABLE`` without ``IF NOT EXISTS`` — is rejected before any statement runs.
_ALLOWED_PREFIXES: tuple[str, ...] = (
    "CREATE SCHEMA IF NOT EXISTS ",
    "CREATE TABLE IF NOT EXISTS ",
)

#: A ``COMMENT '...'`` string literal, matched INCLUDING SQL's doubled-single-
#: quote escape (``''``) — so the whole of e.g. ``COMMENT 'owner''s, note'`` is
#: one match and its interior commas/quotes/prose can never trip a structural
#: check. Column/table comments may legitimately contain commas, parens, quotes,
#: and words like "drop"/"replace", so this literal is stripped/masked before the
#: top-level-comma column split (:func:`_split_top_level_columns`), the per-column
#: type extraction (:func:`_parse_column_def`), and the reconcile ALTER guard
#: (:func:`_is_add_columns_alter`). The CREATE guard (:func:`_is_idempotent_create`)
#: deliberately does NOT use this — it keeps its own :data:`_CREATE_COMMENT_LITERAL_RE`
#: so its accept/reject verdict stays byte-for-byte identical to the pre-refactor code.
_COMMENT_LITERAL_RE = re.compile(r"COMMENT '(?:[^']|'')*'", re.IGNORECASE)

#: The CREATE guard's OWN comment strip, kept SEPARATE from :data:`_COMMENT_LITERAL_RE`.
#: :func:`_is_idempotent_create` applies this to the already-``.upper()``-ed statement,
#: exactly as the pre-refactor code did, so its allowlist verdict is provably unchanged:
#: it neither inherits ``IGNORECASE`` (a no-op on upper-cased text) nor the shared
#: regex's doubled-quote-escape handling. Case-sensitive and single-quote-terminated,
#: matching main's original ``re.sub(r"COMMENT '[^']*'", ...)`` verbatim.
_CREATE_COMMENT_LITERAL_RE = re.compile(r"COMMENT '[^']*'")

#: Destructive/mutating verbs that must never appear in a bootstrap statement's
#: body. ``CREATE OR REPLACE`` is already caught by the prefix check; `` REPLACE ``
#: is kept here for defense-in-depth.
_FORBIDDEN_VERBS: tuple[str, ...] = (
    " DROP ",
    " ALTER ",
    " TRUNCATE ",
    " DELETE ",
    " INSERT ",
    " MERGE ",
    " UPDATE ",
    " REPLACE ",
)


def _is_idempotent_create(statement: str) -> bool:
    """True iff ``statement`` is a SINGLE idempotent create with no destructive body.

    A real allowlist, not a prefix sniff. After case/whitespace normalization the
    statement must satisfy ALL of:

    * **prefix** — begin with ``CREATE SCHEMA IF NOT EXISTS`` or ``CREATE TABLE IF
      NOT EXISTS`` (so ``CREATE OR REPLACE ...`` and a bare ``CREATE TABLE``
      without ``IF NOT EXISTS`` are rejected);
    * **single statement** — one optional trailing ``;`` is allowed, but any other
      ``;`` (a second, possibly destructive, appended statement) is rejected; and
    * **clean body** — no verb in :data:`_FORBIDDEN_VERBS`, after ``COMMENT '...'``
      string literals are stripped so a legitimate column/table comment mentioning
      e.g. "drop" or "replace" in prose is not false-rejected.
    """
    normalized = " ".join(statement.split()).upper()

    if not normalized.startswith(_ALLOWED_PREFIXES):
        return False

    # Strip COMMENT '...' string literals FIRST, so prose (which legitimately may
    # contain ';', 'drop', 'replace', ... — e.g. "...PROMOTE traces; cost
    # ESTIMATE.") can't trip the single-statement or forbidden-verb checks below.
    # Uses the CREATE guard's OWN case-sensitive strip (not the shared
    # _COMMENT_LITERAL_RE) so this allowlist's verdict is identical to pre-refactor.
    stripped = _CREATE_COMMENT_LITERAL_RE.sub("", normalized)

    # Single statement only: drop one optional trailing ';'; any other ';'
    # introduces a second (possibly destructive) statement -> reject.
    single = stripped[:-1].rstrip() if stripped.endswith(";") else stripped
    if ";" in single:
        return False

    # No destructive/mutating verb in the (comment-stripped) body.
    body = f" {single} "
    return not any(verb in body for verb in _FORBIDDEN_VERBS)


def _producer_statements(catalog: str, schema: str) -> list[tuple[str, str]]:
    """``(producer_dotted_name, statement)`` pairs from every ``_ddl()`` producer."""
    pairs: list[tuple[str, str]] = []
    for produce in _DDL_PRODUCERS:
        producer = f"{produce.__module__}.{produce.__qualname__}"
        for stmt in produce(catalog, schema):
            pairs.append((producer, stmt))
    return pairs


def table_ensure_statements(
    catalog: str = DEFAULT_CATALOG, schema: str = DEFAULT_SCHEMA
) -> list[str]:
    """The ``CREATE SCHEMA/TABLE IF NOT EXISTS`` statements for every app table.

    Concatenates each writer's own ``_ddl(catalog, schema)`` in
    :data:`_DDL_PRODUCERS` order, dropping the identical ``CREATE SCHEMA``
    statement the producers share (each ``_ddl()`` emits it first). No statement
    is authored here — every one comes verbatim from a writer module.

    Fail-closed: the FULL producer output is validated against the runtime
    allowlist (:func:`_is_idempotent_create`) **before** anything is returned. If
    any producer emitted a non-idempotent-``CREATE`` statement, this raises
    :class:`ValueError` naming every offending statement and its producer — so a
    caller (:func:`ensure_app_tables`) executes **nothing**, never a partial or
    destructive apply.
    """
    pairs = _producer_statements(catalog, schema)

    violations = [(producer, stmt) for producer, stmt in pairs if not _is_idempotent_create(stmt)]
    if violations:
        detail = "; ".join(f"{producer}: {stmt.strip()[:100]!r}" for producer, stmt in violations)
        raise ValueError(
            "refusing to run non-idempotent-CREATE bootstrap statement(s) — only "
            f"'CREATE SCHEMA/TABLE IF NOT EXISTS' is allowed: {detail}"
        )

    statements: list[str] = []
    seen: set[str] = set()
    for _producer, stmt in pairs:
        if stmt not in seen:
            seen.add(stmt)
            statements.append(stmt)
    return statements


def ensure_app_tables(
    client: Any,
    warehouse_id: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> list[str]:
    """Idempotently ensure every table the app reads exists (empty) on ``warehouse_id``.

    Runs the writer-owned ``CREATE ... IF NOT EXISTS`` statements
    (:func:`table_ensure_statements`) on the warehouse via the shared
    :func:`ail.publish._execute` seam. Fail-closed and idempotent: only
    ``CREATE ... IF NOT EXISTS`` (never ``DROP``/``ALTER``), so a re-run on a
    populated workspace is a no-op and never disturbs existing data.

    ``table_ensure_statements`` validates the **full** statement list against the
    runtime allowlist before returning, so if any writer ``_ddl()`` drifted to a
    destructive/replacing statement this raises **before** the execution loop —
    nothing is applied. Returns the sorted app-read table names covered
    (:data:`APP_QUERY_TABLES`).
    """
    statements = table_ensure_statements(catalog, schema)
    for statement in statements:
        _execute(client, warehouse_id, statement)
    return sorted(APP_QUERY_TABLES)


# ---------------------------------------------------------------------------
# Additive column reconciliation
# ---------------------------------------------------------------------------
#
# Why this exists (a real incident this permanently prevents)
# -----------------------------------------------------------
# ``ensure_app_tables`` runs ``CREATE TABLE IF NOT EXISTS`` — which never adds a
# column to a table that already exists. When a writer adds columns to its
# ``_ddl()`` (L7b-1 added ``change_plan``/``change_preview_diff``/
# ``change_produced_change_ref``; L9 added the six ``verify_*`` columns to
# ``agent_proposed_actions``), deploying that schema-additive version OVER an
# existing workspace leaves the new columns missing on the pre-existing table.
# AppKit typegen's live ``DESCRIBE QUERY`` then fails on the missing column, the
# app build fails, and the running app goes UNAVAILABLE — the same failure mode
# ``ensure_app_tables`` prevents for a *fresh* table, but for an *upgrade*.
#
# This step closes that gap without any manual DDL and without a general
# migration framework: for every ``CREATE TABLE`` a writer ``_ddl()`` emits, it
# diffs the DECLARED columns against the LIVE columns and, for a pre-existing
# table, emits exactly one ``ALTER TABLE ... ADD COLUMNS (...)`` for the missing
# ones — nothing else.
#
# Fail-closed discipline (separate from the CREATE allowlist)
# -----------------------------------------------------------
# * ADD COLUMNS ONLY. This path never DROPs, RENAMEs, or ALTERs a column type; a
#   genuine type conflict fails LOUD (raises), it is never auto-"fixed".
# * Its own allowlist. The only statement shape this path may execute is
#   ``ALTER TABLE <fqn> ADD COLUMNS (...)``, enforced by :func:`_is_add_columns_alter`
#   at runtime before anything executes. It does NOT route through
#   :func:`_is_idempotent_create` (which still bans ``ALTER`` for the CREATE path).
# * Idempotent + fail-closed ordering. Diff-driven, so a re-run on an
#   already-migrated table emits zero ALTERs; it runs AFTER the CREATEs (a just
#   -created fresh table trivially has every column -> no-op) and BEFORE the app
#   build (see :mod:`ail.jobs.bootstrap_grants`). All ALTERs are validated as a
#   set before ANY executes — never a partial/malformed apply.

#: The only statement shape the reconcile path may execute. ``\S+`` for the fully
#: -qualified table name means no whitespace-delimited clause (e.g. a smuggled
#: `` DROP ``) can hide in it, and the anchored trailing ``\)`` means nothing may
#: follow the ADD COLUMNS block. ``DOTALL`` lets a column def span the (collapsed
#: -to-single-space) block. Validated on the COMMENT-stripped statement.
_ADD_COLUMNS_ALTER_RE = re.compile(r"ALTER TABLE \S+ ADD COLUMNS \(.+\)", re.IGNORECASE | re.DOTALL)

#: Verbs that must never appear in a reconcile ALTER's (comment-stripped) body —
#: defense-in-depth beyond the shape regex. `` ALTER `` itself is intentionally
#: absent (``ALTER TABLE`` is the required prefix); a second ``ALTER COLUMN`` is
#: caught by `` ALTER COLUMN `` / the anchored shape instead.
_FORBIDDEN_ALTER_VERBS: tuple[str, ...] = (
    " DROP ",
    " RENAME ",
    " ALTER COLUMN ",
    " CHANGE ",
    " REPLACE ",
    " SET ",
    " TRUNCATE ",
    " DELETE ",
    " INSERT ",
    " UPDATE ",
    " MERGE ",
)


def _is_add_columns_alter(statement: str) -> bool:
    """True iff ``statement`` is exactly one ``ALTER TABLE <fqn> ADD COLUMNS (...)``.

    The reconcile path's OWN allowlist — deliberately separate from
    :func:`_is_idempotent_create`, which governs the CREATE path and still bans
    ``ALTER``. After whitespace-collapse and stripping ``COMMENT '...'`` literals
    (so comment prose can't trip the checks) the statement must satisfy ALL of:

    * **single statement** — one optional trailing ``;`` allowed; any other ``;``
      (a second, possibly destructive, appended statement) is rejected;
    * **exact shape** — full-match :data:`_ADD_COLUMNS_ALTER_RE`, so a bare
      ``ADD COLUMN`` (singular), ``ALTER COLUMN``, ``DROP``/``RENAME``, or any
      trailing clause after the ``(...)`` block is rejected; and
    * **clean body** — no verb in :data:`_FORBIDDEN_ALTER_VERBS`.
    """
    normalized = " ".join(statement.split())
    body = normalized[:-1].rstrip() if normalized.endswith(";") else normalized
    stripped = _COMMENT_LITERAL_RE.sub("", body)
    if ";" in stripped:
        return False
    if not _ADD_COLUMNS_ALTER_RE.fullmatch(stripped):
        return False
    padded = f" {stripped.upper()} "
    return not any(verb in padded for verb in _FORBIDDEN_ALTER_VERBS)


def _split_top_level_columns(column_block: str) -> list[str]:
    """Split a ``CREATE TABLE`` column block on TOP-LEVEL commas only.

    ``column_block`` is the text between the outer ``(`` and its matching ``)``.
    Commas inside a ``COMMENT '...'`` literal or inside a nested type
    (``DECIMAL(10,2)``, ``MAP<STRING,STRING>``, ``ARRAY<...>``, ``STRUCT<...>``)
    must NOT split. Comment literals are masked to equal-length spaces (indices
    stay aligned) so the returned substrings still carry their original
    ``COMMENT`` text verbatim; the scan then splits only on commas seen at paren
    /angle-bracket depth zero in the masked view.
    """
    masked = _COMMENT_LITERAL_RE.sub(lambda m: " " * len(m.group()), column_block)
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(masked):
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(column_block[start:i])
            start = i + 1
    parts.append(column_block[start:])
    return [p.strip() for p in parts if p.strip()]


#: First tokens that mark a TABLE-level constraint (not a column) in a column
#: block. None of the writer ``_ddl()`` producers emit these today, but skipping
#: them defensively keeps a future constraint from being mis-read as a column and
#: turned into a bogus ``ADD COLUMNS`` entry.
_CONSTRAINT_LEADERS: frozenset[str] = frozenset(
    {"PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK"}
)


def _parse_column_def(column_def: str) -> tuple[str, str, str] | None:
    """``(name, full_def, type)`` for one column def, or ``None`` for a constraint.

    ``full_def`` is the whole (whitespace-collapsed) definition — name + type +
    any ``COMMENT '...'`` — so a migrated column is emitted byte-for-byte as the
    writer declared it. ``type`` is the definition minus the name and minus the
    trailing ``COMMENT`` literal, used only for the conservative conflict check.
    """
    if column_def.startswith("`"):
        end = column_def.find("`", 1)
        if end == -1:
            return None
        name = column_def[1:end]
        remainder = column_def[end + 1 :].strip()
    else:
        head, _, tail = column_def.partition(" ")
        name = head
        remainder = tail.strip()
    if not name or name.upper() in _CONSTRAINT_LEADERS:
        return None
    type_str = _COMMENT_LITERAL_RE.sub("", remainder).strip()
    return name, column_def, type_str


def _parse_create_table(statement: str) -> tuple[str, str, list[tuple[str, str, str]]] | None:
    """Parse ``(fqn, table_name, [(name, full_def, type), ...])`` from a CREATE TABLE.

    Returns ``None`` for anything that is not a ``CREATE TABLE IF NOT EXISTS`` (a
    ``CREATE SCHEMA``, say) so callers can iterate the full producer output and
    reconcile only the tables. Whitespace is collapsed first; the outer column
    ``(...)`` is located by quote-aware paren matching (so a ``)`` inside a
    ``COMMENT`` literal or a ``DECIMAL(10,2)`` type does not close it early).
    """
    collapsed = " ".join(statement.split())
    m = re.match(r"CREATE TABLE IF NOT EXISTS\s+(\S+)\s*\(", collapsed, re.IGNORECASE)
    if not m:
        return None
    fqn = m.group(1)
    table_name = fqn.rsplit(".", 1)[-1].strip("`")

    open_idx = m.end() - 1
    depth = 0
    in_str = False
    close_idx = -1
    for i in range(open_idx, len(collapsed)):
        ch = collapsed[i]
        if in_str:
            if ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx == -1:
        raise ValueError(f"unbalanced parentheses in CREATE TABLE for {fqn}")

    column_block = collapsed[open_idx + 1 : close_idx]
    declared: list[tuple[str, str, str]] = []
    for column_def in _split_top_level_columns(column_block):
        parsed = _parse_column_def(column_def)
        if parsed is not None:
            declared.append(parsed)
    return fqn, table_name, declared


def _normalize_type(type_str: str) -> str:
    """Conservative type normalization for the conflict check: casefold + drop all
    whitespace.

    Deliberately minimal so a benign spelling/spacing diff (``STRING`` vs
    ``string``, ``BIGINT`` vs ``bigint``, ``DECIMAL(10, 2)`` vs ``decimal(10,2)``)
    is NOT flagged and does not needlessly break an upgrade deploy. It does NOT
    canonicalize type aliases — only a genuinely different type text raises.
    """
    return "".join(type_str.split()).casefold()


def _read_rows(client: Any, warehouse_id: str, statement: str) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as ``{column: value}`` dicts.

    Mirrors :func:`ail.publish._execute`'s wait loop but reads the result set. A
    response with no manifest/result (e.g. a table with no matching rows) yields
    ``[]`` — treated by the caller as "table not present, nothing to reconcile".
    """
    from databricks.sdk.service.sql import StatementState

    resp = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
    )
    statement_id = resp.statement_id
    state = resp.status.state if resp.status else None
    while state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1.0)
        resp = client.statement_execution.get_statement(statement_id)
        state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        detail = ""
        if resp.status and resp.status.error:
            detail = f": {resp.status.error.message}"
        raise RuntimeError(f"statement {state}{detail}\nSQL head: {statement[:300]}")

    manifest = getattr(resp, "manifest", None)
    result = getattr(resp, "result", None)
    columns = [c.name for c in manifest.schema.columns] if manifest and manifest.schema else []
    data = result.data_array if result and result.data_array else []
    return [dict(zip(columns, row, strict=False)) for row in data]


def _live_column_types(
    client: Any, warehouse_id: str, *, catalog: str, schema: str, table: str
) -> dict[str, str]:
    """LIVE ``{column_name: full_data_type}`` for one table, ``{}`` if it is absent.

    Reads ``information_schema.columns`` (which returns zero rows for a
    nonexistent table — no error), so an empty mapping unambiguously means "not
    present". Names are lower-cased for case-insensitive diffing.
    """
    query = (
        "SELECT column_name, full_data_type FROM "
        f"`{catalog}`.information_schema.columns "
        f"WHERE LOWER(table_schema) = LOWER('{_escape_sql_literal(schema)}') "
        f"AND LOWER(table_name) = LOWER('{_escape_sql_literal(table)}')"
    )
    live: dict[str, str] = {}
    for row in _read_rows(client, warehouse_id, query):
        name = row.get("column_name")
        full_type = row.get("full_data_type")
        if name is not None and full_type is not None:
            live[str(name).lower()] = str(full_type)
    return live


def _escape_sql_literal(value: str) -> str:
    """Escape a value for use inside a single-quoted SQL string literal."""
    return value.replace("'", "''")


def _missing_column_defs(
    fqn: str, declared: list[tuple[str, str, str]], live: dict[str, str]
) -> list[str]:
    """Full defs of DECLARED columns absent LIVE; raises on an unreconcilable type
    conflict.

    Additive-only: a declared column missing live is returned for ``ADD COLUMNS``;
    a declared column present live with a genuinely different (normalized) type is
    a conflict this refuses to auto-fix — it raises :class:`ValueError` naming the
    table, column, and declared-vs-live types. A live-only column (not declared)
    is left untouched.
    """
    missing: list[str] = []
    for name, full_def, type_str in declared:
        key = name.lower()
        if key not in live:
            missing.append(full_def)
            continue
        if _normalize_type(type_str) != _normalize_type(live[key]):
            raise ValueError(
                f"schema drift on {fqn} column {name!r}: declared type {type_str!r} "
                f"conflicts with live type {live[key]!r}. Additive reconciliation refuses "
                "to ALTER/DROP a column type — resolve this drift manually."
            )
    return missing


def reconcile_app_table_columns(
    client: Any,
    warehouse_id: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> list[str]:
    """Additively migrate each app table's columns to its writer-declared schema.

    For every ``CREATE TABLE`` a writer ``_ddl()`` emits, diff the DECLARED
    columns against the table's LIVE columns and, for a PRE-EXISTING table missing
    some, emit exactly one ``ALTER TABLE <fqn> ADD COLUMNS (<missing full defs>)``.
    This closes the ``CREATE TABLE IF NOT EXISTS``-never-adds-columns gap that
    takes a running app UNAVAILABLE on a schema-additive upgrade deploy.

    Fail-closed and additive-only (see the section comment above): tables absent
    live are skipped (the CREATE already handles a fresh table); a type conflict
    raises; every emitted statement is validated by :func:`_is_add_columns_alter`
    as a set BEFORE any executes; and it is idempotent — an already-migrated table
    produces no ALTER. Returns the ALTER statements executed (empty on a no-op).
    """
    alters: list[str] = []
    for statement in table_ensure_statements(catalog, schema):
        parsed = _parse_create_table(statement)
        if parsed is None:
            continue  # CREATE SCHEMA (or non-CREATE-TABLE) — nothing to reconcile
        fqn, table_name, declared = parsed
        live = _live_column_types(
            client, warehouse_id, catalog=catalog, schema=schema, table=table_name
        )
        if not live:
            continue  # table not present yet — the CREATE in ensure_app_tables handled it
        missing = _missing_column_defs(fqn, declared, live)
        if missing:
            alters.append(f"ALTER TABLE {fqn} ADD COLUMNS ({', '.join(missing)})")

    # Fail-closed: validate the FULL set against the ADD COLUMNS allowlist before
    # executing ANY — a single malformed statement means nothing runs.
    violations = [alter for alter in alters if not _is_add_columns_alter(alter)]
    if violations:
        detail = "; ".join(repr(alter[:120]) for alter in violations)
        raise ValueError(
            f"refusing to run non-'ALTER TABLE ... ADD COLUMNS' reconcile statement(s): {detail}"
        )

    for alter in alters:
        _execute(client, warehouse_id, alter)
    return alters
