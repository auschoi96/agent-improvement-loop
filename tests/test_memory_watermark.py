"""The idempotency watermark (:mod:`ail.memory.watermark`)."""

from __future__ import annotations

from ail.memory.schema import WATERMARK_TABLE
from ail.memory.watermark import read_watermark, watermark_scope, write_watermark


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


def test_write_watermark_is_atomic_replace_by_scope(fake_sql_client) -> None:
    client = fake_sql_client({})  # INSERT -> SUCCEEDED, recorded
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
    assert stmt.startswith(f"INSERT INTO `c`.`s`.{WATERMARK_TABLE}")
    assert "REPLACE WHERE scope = 'exp:coh'" in stmt
    assert "'2026-07-05 01:00:00.000'" in stmt
    assert "3, 2, 1" in stmt  # the three counters
