from types import SimpleNamespace

import pytest
from databricks.sdk.service.sql import StatementState

from ail.events import MEMORY_EVENTS_TABLE, _ddl, append_event


class _Statements:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_statement(self, **kwargs):  # type: ignore[no-untyped-def]
        self.statements.append(kwargs["statement"])
        return SimpleNamespace(
            statement_id="stmt",
            status=SimpleNamespace(state=StatementState.SUCCEEDED, error=None),
        )


def test_event_ddl_is_append_only_delta_storage() -> None:
    statements = _ddl("cat", "sch")
    assert any("alignment_events" in statement for statement in statements)
    assert any("memory_events" in statement for statement in statements)
    assert all("DROP" not in statement.upper() for statement in statements)


def test_append_event_uses_one_insert_with_escaped_values() -> None:
    statements = _Statements()
    client = SimpleNamespace(statement_execution=statements)
    event_id = append_event(
        table=MEMORY_EVENTS_TABLE,
        experiment_id="exp-1",
        source="judge_backfill",
        source_id="agent'o",
        warehouse_id="wh",
        catalog="cat",
        schema="sch",
        client=client,
    )
    assert event_id
    assert len(statements.statements) == 1
    assert "INSERT INTO `cat`.`sch`.`memory_events`" in statements.statements[0]
    assert "agent''o" in statements.statements[0]


def test_append_event_refuses_unknown_tables() -> None:
    with pytest.raises(ValueError, match="unsupported event table"):
        append_event(
            table="arbitrary",
            experiment_id="exp",
            source="source",
            source_id="id",
            warehouse_id="wh",
            catalog="cat",
            schema="sch",
        )
