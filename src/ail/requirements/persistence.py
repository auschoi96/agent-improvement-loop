"""Persist the confirmed intake goal to UC — the intake→loop bridge (GAP A).

The compiled goal was **inert**: intake computed a :class:`~ail.goals.compiler.CompiledGoal`
but the optimization loop built its *own* goal from DAB/CLI args
(:func:`ail.jobs.optimization_cycle._build_goal`), so what the user actually stated
never reached the loop. This module closes that gap by writing the **confirmed**
goal to a per-agent UC Delta table the loop reads.

It mirrors the writer-module pattern used across Tier A
(:mod:`ail.publish`, :mod:`ail.publish_lineage`): a single authoritative
``_ddl(catalog, schema)`` (additive, ``CREATE ... IF NOT EXISTS`` only — no
destructive DDL), an agent-keyed table, and the shared atomic
``staging → REPLACE WHERE`` swap (:func:`ail.publish._atomic_replace_table`). The
table's ``_ddl`` is registered with :mod:`ail.jobs.bootstrap_tables` so a fresh
workspace gets it created before first use, exactly like every other framework
table.

**Fail-closed writes.** :func:`persist_compiled_goal` refuses to write a goal whose
``human_confirmed`` is ``False`` — an unconfirmed goal is a proposal and must never
reach the loop. **Fail-soft reads.** :func:`load_persisted_goal` returns ``None``
when the table or the agent's row is absent (a first run before any intake), so the
loop cleanly falls back to its arg-based goal.

**Round-trip validity (GAP B).** A persisted goal may reference an authored judge
that is not in the static built-in allowlist. On load the goal's own judge-guardrail
names are re-admitted via :func:`ail.goals.allowlist.judge_allowlist` so
reconstruction validates — the persisted goal is self-describing about which judges
it needs, and those were authored+validated when it was confirmed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ail.goals.allowlist import judge_allowlist
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.loop.apply_service import _query_rows
from ail.publish import (
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    _atomic_replace_table,
    _execute,
    _lit,
)

__all__ = [
    "COMPILED_GOAL_TABLE",
    "COMPILED_GOAL_COLUMNS",
    "persist_compiled_goal",
    "load_persisted_goal",
    "compiled_goal_persister",
]

#: The per-agent confirmed-goal table the loop reads (one table for all agents,
#: keyed by ``agent_name`` — the same one-table-segmented-in-SQL shape as the other
#: unified framework tables).
COMPILED_GOAL_TABLE = "agent_compiled_goals"

#: Column order, declared once and reused by the DDL + the INSERT so the two can
#: never drift (mirrors :data:`ail.publish.SESSION_COLUMNS`).
COMPILED_GOAL_COLUMNS: list[str] = [
    "agent_name",
    "objective_metric",
    "direction",
    "target_value",
    "target_kind",
    "guardrails_json",
    "requires_quality",
    "cohort_name",
    "human_confirmed",
    "requirements_text",
    "generated_at",
]


def _ddl(catalog: str, schema: str) -> list[str]:
    """The additive ``CREATE ... IF NOT EXISTS`` statements for the goal table.

    Only ``CREATE SCHEMA/TABLE IF NOT EXISTS`` — no ``DROP``/``ALTER``/``REPLACE`` —
    so it satisfies the bootstrap runtime allowlist
    (:func:`ail.jobs.bootstrap_tables._is_idempotent_create`) and is a safe no-op on
    a populated workspace.
    """
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{COMPILED_GOAL_TABLE} (
            agent_name STRING,
            objective_metric STRING,
            direction STRING,
            target_value DOUBLE,
            target_kind STRING,
            guardrails_json STRING,
            requires_quality BOOLEAN,
            cohort_name STRING,
            human_confirmed BOOLEAN,
            requirements_text STRING,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Per-agent confirmed intake goal the optimization loop loads.'""",
    ]


def _guardrails_json(goal: CompiledGoal) -> str:
    """Serialize the goal's guardrails to a JSON string (stable key order)."""
    return json.dumps(
        [
            {
                "name": g.name,
                "kind": g.kind,
                "must_not_regress": g.must_not_regress,
                "threshold": g.threshold,
            }
            for g in goal.guardrails
        ]
    )


def _goal_row(
    goal: CompiledGoal, *, agent_name: str, requirements_text: str | None, stamp: str
) -> list[Any]:
    return [
        agent_name,
        goal.objective_metric,
        goal.direction,
        goal.target.value,
        goal.target.kind,
        _guardrails_json(goal),
        goal.requires_quality,
        goal.cohort_name,
        goal.human_confirmed,
        requirements_text,
        stamp,
    ]


def persist_compiled_goal(
    goal: CompiledGoal,
    *,
    agent_name: str,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    requirements_text: str | None = None,
    generated_at: str | None = None,
) -> int:
    """Write ``goal`` as ``agent_name``'s row in ``agent_compiled_goals``.

    Ensures the table exists (additive ``_ddl``) then atomically replaces the
    agent's single-row slice (``REPLACE WHERE agent_name = …``) so a re-confirm
    overwrites cleanly and never disturbs another agent.

    Fail-closed: refuses (raises :class:`ValueError`) to persist a goal whose
    ``human_confirmed`` is ``False`` — only a confirmed goal may reach the loop.

    Returns the number of rows written (always ``1`` on success).
    """
    if not goal.human_confirmed:
        raise ValueError(
            "refusing to persist an unconfirmed goal: the intake→loop table only carries "
            "human-confirmed goals. Call CompiledGoal.confirm() (via the plan's confirm()) first."
        )

    stamp = generated_at or datetime.now(UTC).isoformat()
    fqn = f"`{catalog}`.`{schema}`"
    for ddl in _ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)
    return _atomic_replace_table(
        client,
        warehouse_id,
        fqn,
        COMPILED_GOAL_TABLE,
        COMPILED_GOAL_COLUMNS,
        [_goal_row(goal, agent_name=agent_name, requirements_text=requirements_text, stamp=stamp)],
        f"agent_name = {_lit(agent_name)}",
    )


def _as_bool(value: Any) -> bool:
    """Parse a warehouse cell (string/bool/None) as a bool, defaulting to ``False``."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "t", "yes"}


def _as_float(value: Any) -> float:
    """Parse a warehouse cell as a float (values come back as strings)."""
    return float(value)


def load_persisted_goal(
    *,
    agent_name: str,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> CompiledGoal | None:
    """Load ``agent_name``'s confirmed goal from ``agent_compiled_goals``, or ``None``.

    Fail-soft: returns ``None`` when the table or the agent's row is absent (a first
    run before any intake), so the loop falls back to its arg-based goal.

    Confirmed-only (defense-in-depth on the READ side): a row whose
    ``human_confirmed`` is false is treated as **no usable persisted goal** and
    ``None`` is returned — the loader never hands an unconfirmed goal to *any* caller,
    complementing :func:`persist_compiled_goal`'s refusal to *write* one. (The live
    loop path also re-checks, but this makes the named loader safe on its own.)

    A present, confirmed row is reconstructed into a
    :class:`~ail.goals.compiler.CompiledGoal`; its own judge-guardrail names are
    re-admitted via :func:`ail.goals.allowlist.judge_allowlist` so a goal referencing
    an authored judge (not in the static built-in set) validates on reconstruction
    (GAP B).

    A backend/parse error other than "table/row absent" reads as "no persisted goal"
    (``None``) — the loop then falls back to its arg-based goal, fail-soft.
    """
    fqn = f"`{catalog}`.`{schema}`.{COMPILED_GOAL_TABLE}"
    columns = ", ".join(COMPILED_GOAL_COLUMNS)
    sql = f"SELECT {columns} FROM {fqn} WHERE agent_name = {_lit(agent_name)} LIMIT 1"
    try:
        rows = _query_rows(client, warehouse_id, sql)
    except Exception:  # noqa: BLE001 - a missing table on first run reads as "no persisted goal"
        return None
    if not rows:
        return None

    row = rows[0]
    # Confirmed-only: never return an unconfirmed goal, regardless of caller.
    if not _as_bool(row["human_confirmed"]):
        return None

    raw_guardrails = json.loads(row["guardrails_json"] or "[]")
    # The judge names to re-admit come from the RAW dicts (before constructing any
    # Guardrail), because Guardrail's own validator calls is_judge at construction —
    # so both the guardrails and the goal must be built INSIDE the allowlist context.
    judge_names = frozenset(g["name"] for g in raw_guardrails if g.get("kind") == "judge")
    with judge_allowlist(judge_names):
        guardrails = tuple(
            Guardrail(
                name=g["name"],
                kind=g["kind"],
                must_not_regress=bool(g.get("must_not_regress", False)),
                threshold=None if g.get("threshold") is None else float(g["threshold"]),
            )
            for g in raw_guardrails
        )
        return CompiledGoal(
            objective_metric=str(row["objective_metric"]),
            direction=str(row["direction"]),  # type: ignore[arg-type]
            target=GoalTarget(value=_as_float(row["target_value"]), kind=str(row["target_kind"])),  # type: ignore[arg-type]
            guardrails=guardrails,
            cohort=str(row["cohort_name"]),
            human_confirmed=_as_bool(row["human_confirmed"]),
        )


def compiled_goal_persister(
    *,
    agent_name: str,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    requirements_text: str | None = None,
) -> Any:
    """A ``(goal) -> None`` persister to hand to :func:`ail.requirements.execute_plan`.

    Adapts :func:`persist_compiled_goal` to the single-arg ``persist`` seam the
    composer's ``execute_plan`` expects, binding the workspace client, warehouse, and
    target table up front.
    """

    def _persist(goal: CompiledGoal) -> None:
        persist_compiled_goal(
            goal,
            agent_name=agent_name,
            client=client,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
            requirements_text=requirements_text,
        )

    return _persist
