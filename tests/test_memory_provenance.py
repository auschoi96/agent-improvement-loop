"""The provenance wall (:mod:`ail.memory.provenance`).

Load-bearing guarantee: a memory row whose ``source_trace_ids`` touch the frozen
Task-Suite or Human-Anchor pools is DROPPED (never written), reusing
:func:`ail.pools.assert_pools_disjoint` as the final verification. Overlap is
matched on exact id OR a shared ≥12-char prefix (the frozen suite stores some ids
truncated).
"""

from __future__ import annotations

import pytest

from ail.memory.provenance import (
    ReservedPools,
    partition_rows,
    resolve_reserved_pools,
)
from ail.memory.schema import MemoryRow
from ail.pools import Pool, PoolOverlapError


def _row(memory_id: str, *trace_ids: str) -> MemoryRow:
    return MemoryRow(
        memory_id=memory_id,
        cohort="claude_code",
        category="token_efficiency",
        guideline_text="be efficient",
        score=0.8,
        source_trace_ids=tuple(trace_ids),
        source_signal="judge:token_efficiency",
        created_at="2026-07-07T00:00:00.000Z",
    )


def test_drops_row_overlapping_task_suite_exact() -> None:
    reserved = ReservedPools(task_suite_ids=frozenset({"a" * 32}))
    clean = _row("keep", "b" * 32)
    dirty = _row("drop", "a" * 32)

    part = partition_rows([clean, dirty], reserved)

    assert [r.memory_id for r in part.kept] == ["keep"]
    assert len(part.dropped) == 1
    assert part.dropped[0].row.memory_id == "drop"
    assert "provenance wall" in part.dropped[0].reason
    assert Pool.TASK_SUITE.value in part.dropped[0].reason


def test_drops_row_overlapping_human_anchor() -> None:
    reserved = ReservedPools(human_anchor_ids=frozenset({"c" * 32}))
    part = partition_rows([_row("drop", "c" * 32)], reserved)
    assert part.kept == ()
    assert Pool.HUMAN_ANCHOR.value in part.dropped[0].reason


def test_drops_on_truncated_reserved_prefix() -> None:
    # The frozen suite stores some source_trace_id values as 12-char prefixes; a full
    # candidate id sharing that prefix must still be dropped (fail-closed).
    reserved = ReservedPools(task_suite_ids=frozenset({"37ed22abfdb1"}))
    dirty = _row("drop", "37ed22abfdb17d2ab106f3002478ecdf")

    part = partition_rows([dirty], reserved)

    assert part.kept == ()
    assert len(part.dropped) == 1


def test_short_prefix_below_threshold_does_not_falsely_drop() -> None:
    # A sub-12-char coincidental overlap must NOT drop a clean row.
    reserved = ReservedPools(task_suite_ids=frozenset({"abc"}))
    clean = _row("keep", "abc" + "d" * 29)
    part = partition_rows([clean], reserved)
    assert [r.memory_id for r in part.kept] == ["keep"]
    assert part.dropped == ()


def test_row_with_any_reserved_id_is_dropped_wholesale() -> None:
    # A row citing several traces is dropped if ANY of them is reserved.
    reserved = ReservedPools(task_suite_ids=frozenset({"a" * 32}))
    part = partition_rows([_row("drop", "b" * 32, "a" * 32, "e" * 32)], reserved)
    assert part.kept == ()
    assert part.dropped[0].row.memory_id == "drop"


def test_empty_reserved_keeps_everything() -> None:
    part = partition_rows([_row("k1", "a" * 32), _row("k2", "b" * 32)], ReservedPools())
    assert {r.memory_id for r in part.kept} == {"k1", "k2"}
    assert part.dropped == ()


def test_verification_gate_catches_exact_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the drop logic ever regressed and let an EXACT reserved id through, the
    # canonical assert_pools_disjoint verification must fail closed (raise), not leak.
    import ail.memory.provenance as prov

    reserved = ReservedPools(task_suite_ids=frozenset({"a" * 32}))
    # Force _overlap to miss (simulate a regression), so the reserved row survives
    # into `kept` and the re-verification has to catch it.
    monkeypatch.setattr(prov, "_overlap", lambda tid, reserved_ids: None)
    with pytest.raises(PoolOverlapError):
        partition_rows([_row("leak", "a" * 32)], reserved)


def test_verification_gate_catches_truncated_prefix_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # BLOCKING 3: a full 32-char candidate sharing a 12-char prefix with a TRUNCATED
    # reserved id survives a regressed drop pass. Exact set-intersection can't see it,
    # so the INDEPENDENT prefix guard in the re-verification must raise.
    import ail.memory.provenance as prov

    reserved = ReservedPools(task_suite_ids=frozenset({"37ed22abfdb1"}))  # 12-char truncated
    monkeypatch.setattr(prov, "_overlap", lambda tid, reserved_ids: None)  # regression
    with pytest.raises(PoolOverlapError):
        partition_rows([_row("leak", "37ed22abfdb17d2ab106f3002478ecdf")], reserved)


def test_resolve_reserved_pools_loads_frozen_task_suite() -> None:
    # The frozen v1 Task Suite (bundled/on-disk) seeds the Task-Suite reserved set.
    reserved = resolve_reserved_pools(task_suite_version="v1")
    assert len(reserved.task_suite_ids) >= 1
    # A known full source_trace_id from eval/task_suite/v1/tasks.yaml.
    assert "bdb3b11e597555cda869ed7ab5b123dd" in reserved.task_suite_ids
    assert reserved.human_anchor_ids == frozenset()


def test_resolve_reserved_pools_fails_closed_on_missing_suite(tmp_path) -> None:
    # A bad root => no wall => the resolver raises so the distiller writes nothing.
    with pytest.raises(FileNotFoundError):
        resolve_reserved_pools(task_suite_version="does-not-exist", task_suite_root=str(tmp_path))
