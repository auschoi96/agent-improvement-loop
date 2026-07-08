"""The idempotency watermark (:mod:`ail.memory.watermark`)."""

from __future__ import annotations

from ail.memory.schema import WATERMARK_COLUMNS, WATERMARK_TABLE
from ail.memory.watermark import (
    build_watermark_merge,
    read_watermark,
    watermark_scope,
    write_watermark,
)


def test_watermark_scope() -> None:
    assert watermark_scope("660599403165942", "claude_code") == "660599403165942:claude_code"


def test_read_watermark_none_on_first_run(fake_sql_client) -> None:
    client = fake_sql_client({"SELECT last_created_at": (["last_created_at"], [])})
    assert read_watermark(client, "wh", catalog="c", schema="s", scope="exp:coh") is None


def test_read_watermark_returns_value(fake_sql_client) -> None:
    client = fake_sql_client(
        {"SELECT last_created_at": (["last_created_at"], [["2026-07-03 07:57:07.085"]])}
    )
    assert (
        read_watermark(client, "wh", catalog="c", schema="s", scope="exp:coh")
        == "2026-07-03 07:57:07.085"
    )


def test_build_watermark_merge_is_upsert_keyed_on_scope() -> None:
    """Regression: the watermark upsert must be a ``MERGE`` on ``scope``, never the
    invalid ``INSERT ... REPLACE WHERE`` with an explicit column list (which crashed
    every live run with ``PARSE_SYNTAX_ERROR at or near 'REPLACE'``)."""
    sql = build_watermark_merge(
        "c",
        "s",
        scope="exp:coh",
        last_created_at="2026-07-05 01:00:00.000",
        run_at="2026-07-07T00:00:00.000Z",
        n_assessments_seen=3,
        n_memories_written=2,
        n_dropped_provenance=1,
    )
    # A MERGE keyed on scope — created on first run, updated thereafter, one row/scope.
    assert sql.startswith(f"MERGE INTO `c`.`s`.{WATERMARK_TABLE} AS t")
    assert "ON t.scope = s.scope" in sql
    assert "WHEN MATCHED THEN UPDATE SET" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    # The exact malformed shape must never reappear.
    assert "REPLACE WHERE" not in sql
    # Column names come from the single source of truth; scope is the key, not updated.
    assert f"AS v ({', '.join(WATERMARK_COLUMNS)})) AS s" in sql
    assert "scope = s.scope" not in sql.split("WHEN MATCHED THEN UPDATE SET")[1]
    # Escaped values are present.
    assert "'2026-07-05 01:00:00.000'" in sql
    assert "3, 2, 1" in sql  # the three counters


def test_write_watermark_executes_the_scope_merge(fake_sql_client) -> None:
    client = fake_sql_client({})  # MERGE -> SUCCEEDED, recorded
    write_watermark(
        client,
        "wh",
        catalog="c",
        schema="s",
        scope="exp:coh",
        last_created_at="2026-07-05 01:00:00.000",
        run_at="2026-07-07T00:00:00.000Z",
        n_assessments_seen=3,
        n_memories_written=2,
        n_dropped_provenance=1,
    )
    stmt = client.statement_execution.executed[-1]
    assert stmt.startswith(f"MERGE INTO `c`.`s`.{WATERMARK_TABLE} AS t")
    assert "ON t.scope = s.scope" in stmt
    assert "REPLACE WHERE" not in stmt
