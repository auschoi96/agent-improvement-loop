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

from collections.abc import Callable
from typing import Any

from ail.loop.publish_proposals import PROPOSALS_TABLE
from ail.loop.publish_proposals import _ddl as _proposals_ddl
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


def table_ensure_statements(
    catalog: str = DEFAULT_CATALOG, schema: str = DEFAULT_SCHEMA
) -> list[str]:
    """The ``CREATE SCHEMA/TABLE IF NOT EXISTS`` statements for every app table.

    Concatenates each writer's own ``_ddl(catalog, schema)`` in
    :data:`_DDL_PRODUCERS` order, dropping the identical ``CREATE SCHEMA``
    statement the producers share (each ``_ddl()`` emits it first). No statement
    is authored here — every one comes verbatim from a writer module.
    """
    statements: list[str] = []
    seen: set[str] = set()
    for produce in _DDL_PRODUCERS:
        for stmt in produce(catalog, schema):
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

    Returns the sorted app-read table names covered (:data:`APP_QUERY_TABLES`).
    """
    for statement in table_ensure_statements(catalog, schema):
        _execute(client, warehouse_id, statement)
    return sorted(APP_QUERY_TABLES)
