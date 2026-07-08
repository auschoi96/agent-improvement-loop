"""The distiller driver (:mod:`ail.memory.distiller`): fail-closed on an empty
window, watermark idempotency, and no watermark advance on a failed distill — all
without a live model (the ``distill`` step and the SQL client are injected).
"""

from __future__ import annotations

import re

import pytest

from ail.memory.distiller import DistillerConfig, DistillerDeps, run_memory_distiller
from ail.memory.provenance import ReservedPools
from ail.memory.writeback import apply_and_write

_ASSESS_COLS = ["name", "target_id", "value_str", "comment", "created_at"]
_ASSESS_DATA = [
    ["token_efficiency", "a" * 32, "4.0", "read too much", "2026-07-03 07:57:07.085"],
    ["rlm_review", "b" * 32, "80", "efficient", "2026-06-30 02:07:18.127"],
]


def _candidate(trace_id: str, **over) -> dict:
    base = {
        "category": "token_efficiency",
        "guideline_text": "prefer grep over reading whole files",
        "score": 0.8,
        "source_trace_ids": [trace_id],
        "source_signal": "judge:token_efficiency",
    }
    base.update(over)
    return base


def _config() -> DistillerConfig:
    return DistillerConfig(
        experiment_id="E",
        warehouse_id="wh",
        catalog="c",
        schema="s",
        annotations_table="cat.mlflow_traces.cc_otel_annotations",
        cohort="claude_code",
    )


def _inserts(client) -> list[str]:
    return [
        s for s in client.statement_execution.executed if s.lstrip().upper().startswith("INSERT")
    ]


def _watermark_writes(client) -> list[str]:
    # INSERTs against the watermark table only — the read SELECT also names the table.
    return [s for s in _inserts(client) if "agent_memory_watermark" in s]


def test_fail_closed_when_no_assessments(fake_sql_client) -> None:
    def responder(stmt: str):
        if "SELECT last_created_at" in stmt:
            return (["last_created_at"], [])  # first run, no watermark
        if "SELECT name, target_id" in stmt:
            return (_ASSESS_COLS, [])  # empty window
        return None

    client = fake_sql_client(responder)
    calls: list[int] = []
    deps = DistillerDeps(
        client=client,
        reserved=ReservedPools(),
        distill=lambda a, t: calls.append(len(a)),
        now=lambda: "NOW",
    )

    report = run_memory_distiller(_config(), deps=deps)

    assert report.wrote is False
    assert report.n_assessments == 0
    assert calls == []  # the model/distill step was never invoked
    assert _inserts(client) == []  # nothing written, watermark NOT advanced


def test_watermark_idempotency_rerun_writes_nothing(fake_sql_client) -> None:
    state = {"wm": None}

    def responder(stmt: str):
        if "SELECT last_created_at" in stmt:
            return (["last_created_at"], [[state["wm"]]] if state["wm"] else [])
        if "SELECT name, target_id" in stmt:
            # A windowed re-read (watermark predicate present) returns nothing new.
            return (_ASSESS_COLS, [] if "created_at >" in stmt else _ASSESS_DATA)
        return None

    client = fake_sql_client(responder)
    calls: list[int] = []

    def distill(assessments, tally):
        calls.append(len(assessments))
        tally.written += len(assessments)  # simulate rows written

    deps = DistillerDeps(
        client=client, reserved=ReservedPools(), distill=distill, now=lambda: "NOW"
    )

    # Run 1: processes both assessments, advances watermark to the newest created_at.
    r1 = run_memory_distiller(_config(), deps=deps)
    assert r1.n_assessments == 2
    assert calls == [2]
    assert r1.watermark_after == "2026-07-03 07:57:07.085"
    wm_writes = _watermark_writes(client)
    assert len(wm_writes) == 1
    assert "2026-07-03 07:57:07.085" in wm_writes[0]

    # Persist that watermark, then re-run over the SAME window.
    state["wm"] = r1.watermark_after

    r2 = run_memory_distiller(_config(), deps=deps)
    assert r2.n_assessments == 0
    assert r2.wrote is False
    assert calls == [2]  # distill NOT called again -> no duplicate rows
    # No second watermark write (empty window returns before advancing).
    assert len(_watermark_writes(client)) == 1


def test_no_watermark_advance_when_distill_raises(fake_sql_client) -> None:
    def responder(stmt: str):
        if "SELECT last_created_at" in stmt:
            return (["last_created_at"], [])
        if "SELECT name, target_id" in stmt:
            return (_ASSESS_COLS, _ASSESS_DATA)
        return None

    client = fake_sql_client(responder)

    def boom(assessments, tally):
        raise RuntimeError("model call failed")

    deps = DistillerDeps(client=client, reserved=ReservedPools(), distill=boom, now=lambda: "NOW")

    with pytest.raises(RuntimeError, match="model call failed"):
        run_memory_distiller(_config(), deps=deps)

    # Fail-closed: a failed distill must not advance the watermark, so the window is
    # retried next run.
    assert _watermark_writes(client) == []


def test_reserved_pools_resolved_when_not_injected(fake_sql_client, monkeypatch) -> None:
    # When deps.reserved is None the driver resolves the frozen pools before writing;
    # a resolution failure must propagate (fail-closed) and write nothing.
    import ail.memory.distiller as distiller_mod

    def responder(stmt: str):
        if "SELECT last_created_at" in stmt:
            return (["last_created_at"], [])
        if "SELECT name, target_id" in stmt:
            return (_ASSESS_COLS, _ASSESS_DATA)
        return None

    client = fake_sql_client(responder)

    def _boom(**_kwargs):
        raise FileNotFoundError("no frozen suite")

    monkeypatch.setattr(distiller_mod, "resolve_reserved_pools", _boom)
    deps = DistillerDeps(client=client, distill=lambda a, t: None, now=lambda: "NOW")

    with pytest.raises(FileNotFoundError):
        run_memory_distiller(_config(), deps=deps)
    assert _watermark_writes(client) == []


def _standard_responder(stmt: str):
    if "SELECT last_created_at" in stmt:
        return (["last_created_at"], [])  # never advanced (simulate first/failed-watermark run)
    if "SELECT name, target_id" in stmt:
        return (_ASSESS_COLS, _ASSESS_DATA)
    return None


def test_grounding_drops_unread_trace_id_in_a_run(fake_sql_client) -> None:
    # BLOCKING 1 (end-to-end): the model submits one row grounded in a READ trace and
    # one citing an UNREAD (non-reserved, format-valid) trace -> only the grounded row
    # is merged; the fabricated one is dropped and never written.
    client = fake_sql_client(_standard_responder)
    merges: list[str] = []

    def distill(assessments, tally):
        read_ids = frozenset(a.trace_id for a in assessments)  # exactly what the driver threads
        apply_and_write(
            [_candidate("a" * 32), _candidate("z" * 32)],  # read, then UNREAD
            execute=merges.append,
            catalog="c",
            schema="s",
            cohort="claude_code",
            created_at="ts",
            reserved=ReservedPools(),
            read_trace_ids=read_ids,
            tally=tally,
        )

    deps = DistillerDeps(
        client=client, reserved=ReservedPools(), distill=distill, now=lambda: "NOW"
    )
    report = run_memory_distiller(_config(), deps=deps)

    assert report.n_written == 1
    assert report.n_invalid == 1  # the unread citation
    assert len(merges) == 1
    assert ("a" * 32) in merges[0]
    assert ("z" * 32) not in merges[0]  # the fabricated id never reached SQL


def test_write_failure_blocks_watermark_and_raises_loudly(fake_sql_client) -> None:
    # BLOCKING 2: a recorded submit_memory failure -> driver raises + watermark NOT advanced.
    client = fake_sql_client(_standard_responder)

    def distill(assessments, tally):
        tally.errors.append("submit_memory failed: MERGE rejected")  # as the tool records it

    deps = DistillerDeps(
        client=client, reserved=ReservedPools(), distill=distill, now=lambda: "NOW"
    )

    with pytest.raises(RuntimeError, match="watermark NOT advanced"):
        run_memory_distiller(_config(), deps=deps)
    assert _watermark_writes(client) == []


def test_reprocess_same_window_inserts_zero_duplicates(fake_sql_client) -> None:
    # BLOCKING 4: insert succeeded but the watermark never advanced (so the window is
    # reprocessed). Deterministic ids + the MERGE upsert mean the second run inserts
    # zero new rows. A stateful fake models the table by tracking merged memory_ids.
    merged_ids: set[str] = set()
    net_inserted: list[int] = []

    def merge_execute(sql: str) -> None:
        ids = set(re.findall(r"[0-9a-f]{64}", sql))  # memory_ids are the only 64-hex tokens
        net_inserted.append(len(ids - merged_ids))
        merged_ids.update(ids)

    def distill(assessments, tally):
        read_ids = frozenset(a.trace_id for a in assessments)
        apply_and_write(
            [_candidate("a" * 32)],  # identical candidate each run -> identical deterministic id
            execute=merge_execute,
            catalog="c",
            schema="s",
            cohort="claude_code",
            created_at="ts",
            reserved=ReservedPools(),
            read_trace_ids=read_ids,
            tally=tally,
        )

    # Watermark never advances (responder always returns none), so run 2 reprocesses run 1's window.
    client = fake_sql_client(_standard_responder)
    deps = DistillerDeps(
        client=client, reserved=ReservedPools(), distill=distill, now=lambda: "NOW"
    )

    run_memory_distiller(_config(), deps=deps)  # run 1
    run_memory_distiller(_config(), deps=deps)  # run 2, same feedback window

    assert net_inserted == [1, 0]  # second reprocess adds zero duplicate rows
