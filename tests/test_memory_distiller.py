"""The distiller driver (:mod:`ail.memory.distiller`): fail-closed on an empty
window, watermark idempotency, and no watermark advance on a failed distill — all
without a live model (the ``distill`` step and the SQL client are injected).
"""

from __future__ import annotations

import pytest

from ail.memory.distiller import DistillerConfig, DistillerDeps, run_memory_distiller
from ail.memory.provenance import ReservedPools

_ASSESS_COLS = ["name", "target_id", "value_str", "comment", "created_at"]
_ASSESS_DATA = [
    ["token_efficiency", "a" * 32, "4.0", "read too much", "2026-07-03 07:57:07.085"],
    ["rlm_review", "b" * 32, "80", "efficient", "2026-06-30 02:07:18.127"],
]


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
