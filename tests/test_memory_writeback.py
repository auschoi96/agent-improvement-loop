"""The write path (:mod:`ail.memory.writeback`): validation, the escaped INSERT
builder, and the validation+wall+INSERT the ``submit_memory`` tool runs — all
tested WITHOUT the Claude Agent SDK or a live model.
"""

from __future__ import annotations

from ail.memory.provenance import ReservedPools
from ail.memory.schema import MEMORY_COLUMNS, MEMORY_TABLE, MemoryRow
from ail.memory.writeback import (
    WriteTally,
    apply_and_write,
    build_memory_insert,
    prepare_memory_rows,
)


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


# -- INSERT builder --------------------------------------------------------


def test_build_memory_insert_is_valid_escaped_insert() -> None:
    row = MemoryRow(
        memory_id="m1",
        cohort="claude_code",
        category="tool_use",
        guideline_text="don't re-read; use grep",  # embedded single quote
        score=0.9,
        source_trace_ids=("t1", "t2"),
        source_signal="rlm",
        created_at="2026-07-07T00:00:00.000Z",
    )
    sql = build_memory_insert("cat", "sch", [row])

    assert sql.startswith(f"INSERT INTO `cat`.`sch`.{MEMORY_TABLE} (")
    # Column list matches the single source of truth, in order.
    assert ", ".join(MEMORY_COLUMNS) in sql
    # Single quote is doubled (escaped), not raw.
    assert "don''t re-read" in sql
    # Arrays rendered as ARRAY(...); nullable embedding is NULL.
    assert "ARRAY('t1', 't2')" in sql
    assert sql.rstrip().endswith("NULL)")


def test_build_memory_insert_multiple_rows() -> None:
    rows = [
        MemoryRow("m1", "c", "cat", "g1", 0.5, ("t1",), "rlm", "ts"),
        MemoryRow("m2", "c", "cat", "g2", 0.6, ("t2",), "judge:correctness", "ts"),
    ]
    sql = build_memory_insert("cat", "sch", rows)
    assert sql.count("ARRAY('t") == 2
    assert "'m1'" in sql and "'m2'" in sql


# -- candidate validation --------------------------------------------------


def test_prepare_rejects_invalid_candidates() -> None:
    candidates = [
        _candidate(),  # valid
        _candidate(guideline_text=""),  # missing guideline
        _candidate(score=2.0),  # out of range
        _candidate(score="not-a-number"),
        _candidate(source_trace_ids=[]),  # no provenance
        _candidate(source_signal="mystery"),  # bad signal
    ]
    prepared = prepare_memory_rows(
        candidates, cohort="claude_code", created_at="ts", id_factory=lambda: "idX"
    )
    assert len(prepared.valid) == 1
    assert len(prepared.invalid) == 5
    reasons = " ".join(r for _c, r in prepared.invalid)
    assert "guideline_text" in reasons
    assert "range" in reasons
    assert "source_trace_ids" in reasons
    assert "source_signal" in reasons


def test_prepare_sets_cohort_and_created_at() -> None:
    prepared = prepare_memory_rows(
        [_candidate()],
        cohort="my_cohort",
        created_at="2026-07-07T00:00:00.000Z",
        id_factory=lambda: "fixed",
    )
    row = prepared.valid[0]
    assert row.cohort == "my_cohort"
    assert row.created_at == "2026-07-07T00:00:00.000Z"
    assert row.memory_id == "fixed"
    assert row.embedding is None


# -- apply_and_write: validation + wall + INSERT ---------------------------


def test_apply_and_write_walls_and_inserts_only_clean_rows() -> None:
    executed: list[str] = []
    reserved = ReservedPools(task_suite_ids=frozenset({"b" * 32}))
    candidates = [
        _candidate(source_trace_ids=["a" * 32]),  # clean -> written
        _candidate(source_trace_ids=["b" * 32]),  # reserved -> dropped by wall
        _candidate(guideline_text=""),  # invalid -> rejected
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
        tally=tally,
        id_factory=lambda: "idX",
    )

    assert tally.written == 1
    assert len(tally.dropped_provenance) == 1
    assert len(tally.invalid) == 1
    # Exactly one INSERT ran, and it carries only the clean trace id.
    assert len(executed) == 1
    assert ("a" * 32) in executed[0]
    assert ("b" * 32) not in executed[0]
    assert "wrote 1" in summary


def test_apply_and_write_runs_no_insert_when_all_dropped() -> None:
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
        tally=tally,
        id_factory=lambda: "idX",
    )

    assert tally.written == 0
    assert executed == []  # nothing written when the wall drops everything
    assert len(tally.dropped_provenance) == 1
