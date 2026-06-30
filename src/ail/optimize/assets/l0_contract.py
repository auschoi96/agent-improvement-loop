"""The **real** L0 Delta-table column contract the metric-view generator builds on.

The metric-view generator must reference only columns that actually exist in the
published L0 tables (``docs/L0_METRICS_CONTRACT.md``); inventing a column would
produce a spec that fails the moment it is deployed. This module is that
allow-list: one :class:`L0Column` per real column of each ``l0_*`` table, with the
SQL type and a coarse :class:`ColumnKind` the generator uses to decide what a
column may be (a numeric column can back a measure; a categorical one a dimension).

The **source of truth** for the column *names* is :mod:`ail.publish` — the module
that creates and populates these tables. Those name lists carry no type/role
information, which the generator needs, so the typed registry is declared here and
kept honest by :func:`verify_against_publish`: it asserts this registry's column
names are *exactly* ``publish``'s ``*_COLUMNS`` (and the table / catalog / schema
constants match). Any drift — a column added, renamed, or dropped in ``publish`` —
fails that check loudly (a test calls it), so the registry can never silently
fabricate or omit a column. ``publish`` is imported lazily, inside that function,
so importing the generator stays free of the ingest/MLflow stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ColumnKind",
    "L0Column",
    "L0Table",
    "L0ColumnContract",
    "DEFAULT_CATALOG",
    "DEFAULT_SCHEMA",
    "SESSION_TABLE",
    "SUMMARY_TABLE",
    "DIAGNOSIS_TABLE",
    "L0_CONTRACT",
    "verify_against_publish",
]

#: Catalog / schema the L0 tables are published to. Mirrors
#: :data:`ail.publish.DEFAULT_CATALOG` / ``DEFAULT_SCHEMA`` (verified equal in
#: :func:`verify_against_publish`); duplicated as plain constants so importing the
#: generator does not drag in the publish module's MLflow/ingest imports.
DEFAULT_CATALOG = "austin_choi_omni_agent_catalog"
DEFAULT_SCHEMA = "agent_improvement_loop"

SESSION_TABLE = "l0_session_metrics"
SUMMARY_TABLE = "l0_corpus_summary"
DIAGNOSIS_TABLE = "l0_diagnosis"


class ColumnKind(StrEnum):
    """What a column may be used for in a metric view.

    * ``IDENTIFIER`` — a key (``trace_id``, ``experiment_id``): groupable but rarely
      a useful dimension on its own.
    * ``CATEGORICAL`` — a low-cardinality string to slice by (``model``, ``status``).
    * ``TEMPORAL`` — a time field (stored as ISO-8601 ``STRING`` here).
    * ``NUMERIC`` — an additive/averageable number that can back a measure.
    * ``BOOLEAN`` — a flag, usable inside a measure ``FILTER (WHERE ...)``.
    """

    IDENTIFIER = "identifier"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class L0Column:
    """One real column of an L0 table: its name, SQL type, and usage kind."""

    name: str
    sql_type: str
    kind: ColumnKind

    @property
    def is_numeric(self) -> bool:
        return self.kind is ColumnKind.NUMERIC

    @property
    def is_boolean(self) -> bool:
        return self.kind is ColumnKind.BOOLEAN


@dataclass(frozen=True, slots=True)
class L0Table:
    """A published L0 Delta table and its columns, scoped to a catalog/schema."""

    name: str
    catalog: str
    schema: str
    columns: tuple[L0Column, ...]

    @property
    def fqn(self) -> str:
        """Fully-qualified ``catalog.schema.table`` name (the metric-view source)."""
        return f"{self.catalog}.{self.schema}.{self.name}"

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def get(self, column: str) -> L0Column | None:
        return next((c for c in self.columns if c.name == column), None)

    def has(self, column: str) -> bool:
        return any(c.name == column for c in self.columns)


def _session_columns() -> tuple[L0Column, ...]:
    k = ColumnKind
    return (
        L0Column("experiment_id", "STRING", k.IDENTIFIER),
        L0Column("trace_id", "STRING", k.IDENTIFIER),
        L0Column("session_id", "STRING", k.IDENTIFIER),
        L0Column("producer", "STRING", k.CATEGORICAL),
        L0Column("model", "STRING", k.CATEGORICAL),
        L0Column("status", "STRING", k.CATEGORICAL),
        L0Column("request_time", "STRING", k.TEMPORAL),
        L0Column("duration_seconds", "DOUBLE", k.NUMERIC),
        L0Column("input_tokens", "BIGINT", k.NUMERIC),
        L0Column("output_tokens", "BIGINT", k.NUMERIC),
        L0Column("total_tokens", "BIGINT", k.NUMERIC),
        L0Column("cache_creation_input_tokens", "BIGINT", k.NUMERIC),
        L0Column("cache_read_input_tokens", "BIGINT", k.NUMERIC),
        L0Column("cache_total_tokens", "BIGINT", k.NUMERIC),
        L0Column("est_cost_usd", "DOUBLE", k.NUMERIC),
        L0Column("cost_priced", "BOOLEAN", k.BOOLEAN),
        L0Column("total_tool_calls", "INT", k.NUMERIC),
        L0Column("distinct_tool_calls", "INT", k.NUMERIC),
        L0Column("redundant_tool_calls", "INT", k.NUMERIC),
        L0Column("redundancy_rate", "DOUBLE", k.NUMERIC),
        L0Column("generated_at", "STRING", k.TEMPORAL),
    )


def _summary_columns() -> tuple[L0Column, ...]:
    k = ColumnKind
    return (
        L0Column("experiment_id", "STRING", k.IDENTIFIER),
        L0Column("schema_version", "STRING", k.CATEGORICAL),
        L0Column("generated_at", "STRING", k.TEMPORAL),
        L0Column("trace_count", "INT", k.NUMERIC),
        L0Column("total_input_tokens", "BIGINT", k.NUMERIC),
        L0Column("total_output_tokens", "BIGINT", k.NUMERIC),
        L0Column("total_tokens", "BIGINT", k.NUMERIC),
        L0Column("cache_total_tokens", "BIGINT", k.NUMERIC),
        L0Column("median_tokens", "DOUBLE", k.NUMERIC),
        L0Column("mean_tokens", "DOUBLE", k.NUMERIC),
        L0Column("p90_tokens", "DOUBLE", k.NUMERIC),
        L0Column("max_tokens", "BIGINT", k.NUMERIC),
        L0Column("min_tokens", "BIGINT", k.NUMERIC),
        L0Column("total_tool_calls", "BIGINT", k.NUMERIC),
        L0Column("redundancy_rate", "DOUBLE", k.NUMERIC),
        L0Column("total_cost_usd", "DOUBLE", k.NUMERIC),
        L0Column("priced_traces", "INT", k.NUMERIC),
        L0Column("unpriced_traces", "INT", k.NUMERIC),
    )


def _diagnosis_columns() -> tuple[L0Column, ...]:
    k = ColumnKind
    return (
        L0Column("experiment_id", "STRING", k.IDENTIFIER),
        L0Column("trace_id", "STRING", k.IDENTIFIER),
        L0Column("session_id", "STRING", k.IDENTIFIER),
        L0Column("model", "STRING", k.CATEGORICAL),
        L0Column("signature_kind", "STRING", k.CATEGORICAL),
        L0Column("tool", "STRING", k.CATEGORICAL),
        L0Column("identity", "STRING", k.CATEGORICAL),
        L0Column("repeat_count", "INT", k.NUMERIC),
        L0Column("trace_total_tool_calls", "INT", k.NUMERIC),
        L0Column("generated_at", "STRING", k.TEMPORAL),
    )


class L0ColumnContract:
    """Lookup over the real L0 tables — the generator's column allow-list.

    Tables are keyed by their short name (``l0_session_metrics``); :meth:`table_for`
    also resolves a fully-qualified ``catalog.schema.table`` source string by its
    trailing segment, so a generated spec's ``source`` can be validated against the
    contract regardless of catalog/schema.
    """

    def __init__(self, tables: tuple[L0Table, ...]) -> None:
        self._tables: dict[str, L0Table] = {t.name: t for t in tables}

    @property
    def tables(self) -> tuple[L0Table, ...]:
        return tuple(self._tables.values())

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(self._tables)

    def get_table(self, table: str) -> L0Table | None:
        """Resolve a table by short name or by the trailing segment of an fqn."""
        return self._tables.get(table.split(".")[-1])

    def table_for(self, source: str) -> L0Table | None:
        """Alias of :meth:`get_table` named for the metric-view ``source`` use."""
        return self.get_table(source)

    def has(self, table: str, column: str) -> bool:
        resolved = self.get_table(table)
        return resolved is not None and resolved.has(column)

    def column(self, table: str, column: str) -> L0Column | None:
        resolved = self.get_table(table)
        return resolved.get(column) if resolved is not None else None

    def column_names(self, table: str) -> tuple[str, ...]:
        resolved = self.get_table(table)
        return resolved.column_names if resolved is not None else ()

    def restricted(self, table: str, *, drop: set[str]) -> L0ColumnContract:
        """A copy with ``drop`` columns removed from ``table``.

        Used to model an L0 contract that does **not** back a given measure (e.g.
        an older publish that lacks the redundancy columns) so the generator's
        fabrication guard can be exercised offline.
        """
        rebuilt: list[L0Table] = []
        for t in self._tables.values():
            if t.name == table.split(".")[-1]:
                kept = tuple(c for c in t.columns if c.name not in drop)
                rebuilt.append(L0Table(t.name, t.catalog, t.schema, kept))
            else:
                rebuilt.append(t)
        return L0ColumnContract(tuple(rebuilt))


def _build_contract(
    catalog: str = DEFAULT_CATALOG, schema: str = DEFAULT_SCHEMA
) -> L0ColumnContract:
    return L0ColumnContract(
        (
            L0Table(SESSION_TABLE, catalog, schema, _session_columns()),
            L0Table(SUMMARY_TABLE, catalog, schema, _summary_columns()),
            L0Table(DIAGNOSIS_TABLE, catalog, schema, _diagnosis_columns()),
        )
    )


#: The published L0 contract over the default catalog/schema.
L0_CONTRACT = _build_contract()


def verify_against_publish() -> None:
    """Assert this registry matches :mod:`ail.publish` exactly, or raise.

    ``publish`` is the source of truth for the column *names* and the table /
    catalog / schema constants. This re-derives them and checks every table's
    column set (and order) is identical and the constants agree. Drift raises
    :class:`ValueError`; a test calls this so a change to ``publish`` that this
    registry has not tracked fails loudly rather than letting the generator emit a
    spec referencing a stale column.
    """
    from ail import publish

    expected = {
        SESSION_TABLE: list(publish.SESSION_COLUMNS),
        SUMMARY_TABLE: list(publish.SUMMARY_COLUMNS),
        DIAGNOSIS_TABLE: list(publish.DIAGNOSIS_COLUMNS),
    }
    table_consts = {
        SESSION_TABLE: publish.SESSION_TABLE,
        SUMMARY_TABLE: publish.SUMMARY_TABLE,
        DIAGNOSIS_TABLE: publish.DIAGNOSIS_TABLE,
    }
    problems: list[str] = []
    if DEFAULT_CATALOG != publish.DEFAULT_CATALOG:
        problems.append(f"catalog {DEFAULT_CATALOG!r} != publish {publish.DEFAULT_CATALOG!r}")
    if DEFAULT_SCHEMA != publish.DEFAULT_SCHEMA:
        problems.append(f"schema {DEFAULT_SCHEMA!r} != publish {publish.DEFAULT_SCHEMA!r}")
    for short_name, want in expected.items():
        if table_consts[short_name] != short_name:
            problems.append(
                f"table const for {short_name!r} != publish {table_consts[short_name]!r}"
            )
        table = L0_CONTRACT.get_table(short_name)
        got = list(table.column_names) if table is not None else []
        if got != want:
            problems.append(f"{short_name}: registry columns {got} != publish columns {want}")
    if problems:
        raise ValueError(
            "L0 column registry drifted from ail.publish (source of truth): " + "; ".join(problems)
        )
