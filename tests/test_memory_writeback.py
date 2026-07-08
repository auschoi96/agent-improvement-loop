"""The write path (:mod:`ail.memory.writeback`): grounding (anti-fabrication),
strict-signal validation, deterministic ids, the idempotent MERGE builder, and the
fail-loud (never-swallowed) failure handling — all without the Claude Agent SDK
model or a live workspace.
"""

from __future__ import annotations

import asyncio

import pytest

import ail.memory.provenance as provenance
from ail.memory.provenance import ReservedPools
from ail.memory.schema import MEMORY_COLUMNS, MEMORY_TABLE, MemoryRow
from ail.memory.writeback import (
    WriteTally,
    apply_and_write,
    build_memory_merge,
    create_submit_memory_tool,
    memory_id_for,
    prepare_memory_rows,
)
from ail.pools import PoolOverlapError

_READ = frozenset({"a" * 32, "b" * 32, "c" * 32})


def _candidate(**over) -> dict:
    base = {
        "category": "token_efficiency",
        "guideline_text": "prefer grep over reading whole files",
        "score": 0.8,
        "source_trace_ids": ["a" * 32],
        "source_signal": "judge:token_efficiency",
    }
    base.update(over)
    return base


# -- deterministic id ------------------------------------------------------


def test_memory_id_is_deterministic_and_content_addressed() -> None:
    a = memory_id_for("claude_code", "cat", "be efficient", ["t2", "t1"])
    b = memory_id_for("claude_code", "cat", "be efficient", ["t1", "t2"])  # order-insensitive
    assert a == b
    assert len(a) == 64  # sha256 hex
    # Any content change -> a different id.
    assert memory_id_for("claude_code", "cat", "be efficient", ["t1"]) != a
    assert memory_id_for("claude_code", "other", "be efficient", ["t1", "t2"]) != a
    assert memory_id_for("other", "cat", "be efficient", ["t1", "t2"]) != a


# -- idempotent MERGE builder ----------------------------------------------


def test_build_memory_merge_is_idempotent_escaped_upsert() -> None:
    row = MemoryRow(
        memory_id="m" * 64,
        cohort="claude_code",
        category="tool_use",
        guideline_text="don't re-read; use grep",  # embedded single quote
        score=0.9,
        source_trace_ids=("t1", "t2"),
        source_signal="rlm",
        created_at="2026-07-07T00:00:00.000Z",
    )
    sql = build_memory_merge("cat", "sch", [row])

    assert sql.startswith(f"MERGE INTO `cat`.`sch`.{MEMORY_TABLE} AS t")
    # Idempotent upsert keyed on the deterministic id: reprocessing inserts nothing.
    assert "ON t.memory_id = s.memory_id" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    # Source column names come from the single source of truth (on the inner derived
    # table, since MERGE forbids column aliases directly on the USING source).
    assert f"AS v ({', '.join(MEMORY_COLUMNS)})) AS s" in sql
    # Escaped, arrays rendered, nullable embedding cast so the VALUES column stays typed.
    assert "don''t re-read" in sql
    assert "ARRAY('t1', 't2')" in sql
    assert "CAST(NULL AS ARRAY<FLOAT>)" in sql


# -- validation: strict signal + grounding + score -------------------------


def test_prepare_rejects_invalid_candidates() -> None:
    candidates = [
        _candidate(),  # valid
        _candidate(guideline_text=""),  # missing guideline
        _candidate(score=2.0),  # out of range
        _candidate(score="not-a-number"),
        _candidate(source_trace_ids=[]),  # no provenance
    ]
    prepared = prepare_memory_rows(
        candidates, cohort="claude_code", created_at="ts", read_trace_ids=_READ
    )
    assert len(prepared.valid) == 1
    assert len(prepared.invalid) == 4
    reasons = " ".join(r for _c, r in prepared.invalid)
    assert "guideline_text" in reasons
    assert "range" in reasons
    assert "source_trace_ids" in reasons


def test_source_signal_must_be_exact() -> None:
    good = [_candidate(source_signal="rlm"), _candidate(source_signal="judge:correctness")]
    bad = [
        _candidate(source_signal="judge:bogus"),  # not a real judge
        _candidate(source_signal="rlm_review_failed"),  # the excluded marker
        _candidate(source_signal="rlm_token_efficiency"),  # RLM sub-name, not the coarse 'rlm'
        _candidate(source_signal="judge:"),  # empty name
    ]
    prepared = prepare_memory_rows(
        good + bad, cohort="claude_code", created_at="ts", read_trace_ids=_READ
    )
    assert len(prepared.valid) == 2
    assert len(prepared.invalid) == 4
    assert all("source_signal" in r for _c, r in prepared.invalid)


def test_grounding_rejects_unread_trace_id() -> None:
    # BLOCKING 1 (unit): a format-valid, non-reserved, but UNREAD trace id is rejected.
    prepared = prepare_memory_rows(
        [
            _candidate(source_trace_ids=["a" * 32]),  # read -> valid
            _candidate(source_trace_ids=["z" * 32]),  # unread -> rejected
            _candidate(source_trace_ids=["a" * 32, "z" * 32]),  # one unread -> rejected
        ],
        cohort="claude_code",
        created_at="ts",
        read_trace_ids=_READ,
    )
    assert len(prepared.valid) == 1
    assert len(prepared.invalid) == 2
    assert all("unread trace id" in r for _c, r in prepared.invalid)


# -- apply_and_write: ground + wall + merge --------------------------------


def test_apply_and_write_merges_only_clean_grounded_rows() -> None:
    executed: list[str] = []
    reserved = ReservedPools(task_suite_ids=frozenset({"b" * 32}))
    candidates = [
        _candidate(source_trace_ids=["a" * 32]),  # clean + grounded -> merged
        _candidate(source_trace_ids=["b" * 32]),  # reserved -> dropped by wall
        _candidate(source_trace_ids=["z" * 32]),  # unread -> rejected (grounding)
        _candidate(guideline_text=""),  # invalid
    ]
    tally = WriteTally()

    summary = apply_and_write(
        candidates,
        execute=executed.append,
        catalog="cat",
        schema="sch",
        cohort="claude_code",
        created_at="ts",
        reserved=reserved,
        read_trace_ids=_READ,
        tally=tally,
    )

    assert tally.written == 1
    assert len(tally.dropped_provenance) == 1
    assert len(tally.invalid) == 2  # unread + empty guideline
    assert len(executed) == 1  # one MERGE, carrying only the clean id
    assert ("a" * 32) in executed[0]
    assert ("b" * 32) not in executed[0]
    assert executed[0].startswith("MERGE INTO")
    assert "merged 1" in summary
    assert tally.errors == []


def test_apply_and_write_no_merge_when_all_dropped() -> None:
    executed: list[str] = []
    reserved = ReservedPools(task_suite_ids=frozenset({"a" * 32}))
    tally = WriteTally()

    apply_and_write(
        [_candidate(source_trace_ids=["a" * 32])],
        execute=executed.append,
        catalog="cat",
        schema="sch",
        cohort="claude_code",
        created_at="ts",
        reserved=reserved,
        read_trace_ids=_READ,
        tally=tally,
    )

    assert tally.written == 0
    assert executed == []
    assert len(tally.dropped_provenance) == 1


# -- fail-loud: never swallow a SQL error or a provenance regression -------


def test_apply_and_write_propagates_sql_error() -> None:
    def boom(_sql: str) -> None:
        raise RuntimeError("warehouse exploded")

    with pytest.raises(RuntimeError, match="warehouse exploded"):
        apply_and_write(
            [_candidate(source_trace_ids=["a" * 32])],
            execute=boom,
            catalog="cat",
            schema="sch",
            cohort="claude_code",
            created_at="ts",
            reserved=ReservedPools(),
            read_trace_ids=_READ,
            tally=WriteTally(),
        )


def test_apply_and_write_propagates_provenance_regression(monkeypatch) -> None:
    # Simulate a drop-logic regression: _overlap misses, so a reserved id survives the
    # drop pass -> the re-verification MUST raise, and apply_and_write must NOT swallow.
    monkeypatch.setattr(provenance, "_overlap", lambda tid, reserved_ids: None)
    reserved = ReservedPools(task_suite_ids=frozenset({"a" * 32}))
    with pytest.raises(PoolOverlapError):
        apply_and_write(
            [_candidate(source_trace_ids=["a" * 32])],
            execute=lambda _sql: None,
            catalog="cat",
            schema="sch",
            cohort="claude_code",
            created_at="ts",
            reserved=reserved,
            read_trace_ids=_READ,
            tally=WriteTally(),
        )


def test_submit_memory_records_error_not_swallowed() -> None:
    # BLOCKING 2 (tool level): a failed MERGE must land on tally.errors AND surface as
    # is_error to the model — never a silent success.
    class _RaisingExec:
        def execute_statement(self, **_kwargs):
            raise RuntimeError("MERGE rejected")

    class _RaisingClient:
        statement_execution = _RaisingExec()

    tally = WriteTally()
    submit_tool = create_submit_memory_tool(
        client=_RaisingClient(),
        warehouse_id="wh",
        catalog="cat",
        schema="sch",
        cohort="claude_code",
        reserved=ReservedPools(),
        read_trace_ids=_READ,
        tally=tally,
        now=lambda: "ts",
    )
    import json

    result = asyncio.run(
        submit_tool.handler(
            {"memories_json": json.dumps([_candidate(source_trace_ids=["a" * 32])])}
        )
    )
    assert result.get("is_error") is True
    assert len(tally.errors) == 1
    assert "MERGE rejected" in tally.errors[0]
