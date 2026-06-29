"""Unit tests for the Tier A publish module's pure mapping/rendering.

These cover the contract -> flat-row mapping and SQL-literal escaping in
:mod:`ail.publish` without any network/warehouse access (mirrors how
``build_example1_diagnosis`` is tested as a pure function).
"""

from __future__ import annotations

import os

import pytest
from databricks.sdk.service.sql import StatementState

import ail.publish as pub
from ail.ingest.base import NormalizedTrace, TokenUsage, ToolCall, TraceStatus
from ail.metrics.l0_deterministic import compute_l0
from ail.publish import (
    DIAGNOSIS_COLUMNS,
    SESSION_COLUMNS,
    SESSION_TABLE,
    SUMMARY_COLUMNS,
    _atomic_replace_table,
    _diagnosis_rows,
    _lit,
    _row,
    _session_rows,
    _summary_row,
)

EXPERIMENT = "660599403165942"


# -- fake warehouse client (records statements; can fail on demand) --------


class _FakeStatus:
    def __init__(self, state: StatementState) -> None:
        self.state = state
        self.error = None


class _FakeResp:
    def __init__(self, state: StatementState) -> None:
        self.statement_id = "stmt-1"
        self.status = _FakeStatus(state)


class _FakeStatementExecution:
    def __init__(self, fail_substr: str | None) -> None:
        self.statements: list[str] = []
        self._fail_substr = fail_substr

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        self.statements.append(statement)
        if self._fail_substr and self._fail_substr in statement:
            raise RuntimeError("simulated warehouse failure")
        return _FakeResp(StatementState.SUCCEEDED)

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return _FakeResp(StatementState.SUCCEEDED)


class _FakeClient:
    def __init__(self, fail_substr: str | None = None) -> None:
        self.statement_execution = _FakeStatementExecution(fail_substr)


def _priced_session() -> NormalizedTrace:
    # Re-runs the same shell prologue 3x (boilerplate) and re-reads one path 2x.
    tools: list[ToolCall] = [
        ToolCall(id=f"b{i}", name="Bash", arguments={"command": f"cd /repo\nstep {i}"})
        for i in range(3)
    ] + [
        ToolCall(id=f"r{i}", name="Read", arguments={"file_path": "/repo/a.ts", "offset": i})
        for i in range(2)
    ]
    return NormalizedTrace(
        trace_id="trace:/loc/priced",
        status=TraceStatus.OK,
        producer="claude_code",
        model="claude-opus-4-8",
        session_id="sess-1",
        token_usage=TokenUsage(input_tokens=900_000, output_tokens=43_000, _total_tokens=943_000),
        tool_calls=tools,
        execution_duration_ms=9_000_000,
    )


def _unpriced_session() -> NormalizedTrace:
    # Model not in the price book -> cost not estimated (priced=False).
    return NormalizedTrace(
        trace_id="trace:/loc/unpriced",
        status=TraceStatus.OK,
        producer=None,
        model="some-unknown-model",
        token_usage=TokenUsage(input_tokens=10_000, output_tokens=1_000, _total_tokens=11_000),
        tool_calls=[ToolCall(id="s", name="Grep", arguments={"pattern": "foo"})],
    )


def _report():  # type: ignore[no-untyped-def]
    return compute_l0(
        [_priced_session(), _unpriced_session()],
        experiment_id=EXPERIMENT,
        generated_at="2026-06-29T00:00:00Z",
    )


# -- _lit / _row -----------------------------------------------------------


def test_lit_handles_python_types() -> None:
    assert _lit(None) == "NULL"
    assert _lit(True) == "TRUE"
    assert _lit(False) == "FALSE"
    assert _lit(42) == "42"
    assert _lit("plain") == "'plain'"


def test_lit_escapes_quotes_and_backslashes() -> None:
    # A path with an apostrophe must not break out of the string literal.
    assert _lit("o'brien") == "'o''brien'"
    assert _lit("a\\b") == "'a\\\\b'"


def test_row_renders_all_columns_safely() -> None:
    rendered = _row(["x", None, 3, "it's"])
    assert rendered == "('x', NULL, 3, 'it''s')"


# -- row builders match declared column order ------------------------------


def test_session_rows_shape_and_values() -> None:
    report = _report()
    rows = _session_rows(report)
    assert len(rows) == report.n_traces == 2
    for r in rows:
        assert len(r) == len(SESSION_COLUMNS)

    by_trace = {r[SESSION_COLUMNS.index("trace_id")]: r for r in rows}
    priced = by_trace["trace:/loc/priced"]
    assert priced[SESSION_COLUMNS.index("experiment_id")] == EXPERIMENT
    assert priced[SESSION_COLUMNS.index("total_tokens")] == 943_000
    assert priced[SESSION_COLUMNS.index("cost_priced")] is True
    assert priced[SESSION_COLUMNS.index("est_cost_usd")] > 0

    unpriced = by_trace["trace:/loc/unpriced"]
    assert unpriced[SESSION_COLUMNS.index("cost_priced")] is False
    assert unpriced[SESSION_COLUMNS.index("est_cost_usd")] == 0.0


def test_summary_row_shape_and_values() -> None:
    report = _report()
    row = _summary_row(report)
    assert len(row) == len(SUMMARY_COLUMNS)
    assert row[SUMMARY_COLUMNS.index("experiment_id")] == EXPERIMENT
    assert row[SUMMARY_COLUMNS.index("trace_count")] == 2
    assert row[SUMMARY_COLUMNS.index("max_tokens")] == 943_000
    assert row[SUMMARY_COLUMNS.index("priced_traces")] == 1
    assert row[SUMMARY_COLUMNS.index("unpriced_traces")] == 1


def test_diagnosis_rows_capture_repeats() -> None:
    report = _report()
    rows = _diagnosis_rows(report)
    assert rows, "expected repeated-call diagnosis rows"
    for r in rows:
        assert len(r) == len(DIAGNOSIS_COLUMNS)

    kind_i = DIAGNOSIS_COLUMNS.index("signature_kind")
    count_i = DIAGNOSIS_COLUMNS.index("repeat_count")
    tool_i = DIAGNOSIS_COLUMNS.index("tool")

    shell = [r for r in rows if r[kind_i] == "shell"]
    paths = [r for r in rows if r[kind_i] == "path"]
    assert any(r[tool_i] == "Bash" and r[count_i] == 3 for r in shell)
    assert any(r[tool_i] == "Read" and r[count_i] == 2 for r in paths)


# -- atomic replace: happy path + failure path -----------------------------

SCHEMA_FQN = "`cat`.`sch`"


def test_atomic_replace_swaps_via_staging_then_replace_where() -> None:
    report = _report()
    client = _FakeClient()
    n = _atomic_replace_table(
        client, "wh", SCHEMA_FQN, SESSION_TABLE, SESSION_COLUMNS, _session_rows(report), EXPERIMENT
    )
    stmts = client.statement_execution.statements
    assert n == report.n_traces
    # Ordered contract: clone staging -> load staging -> atomic swap -> drop.
    assert stmts[0].startswith(f"CREATE OR REPLACE TABLE {SCHEMA_FQN}._stg_")
    assert any(s.startswith(f"INSERT INTO {SCHEMA_FQN}._stg_") and "VALUES" in s for s in stmts)
    swap = [s for s in stmts if "REPLACE WHERE" in s]
    assert len(swap) == 1
    assert f"REPLACE WHERE experiment_id = '{EXPERIMENT}'" in swap[0]
    assert "SELECT * FROM" in swap[0]
    assert any(s.startswith("DROP TABLE IF EXISTS") for s in stmts)


class _FakeSource:
    def fetch_traces(self, **kwargs):  # type: ignore[no-untyped-def]
        return []


def test_publish_exposes_warehouse_for_uc_trace_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    # The v4 trace store needs the SQL warehouse id in the env to read UC-table
    # traces; publish() has the warehouse (it also backs the writes) and must
    # surface it for the read. Run publish() fully offline via fakes.
    monkeypatch.delenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", raising=False)
    report = _report()
    monkeypatch.setattr(pub, "_build_source", lambda profile: _FakeSource())
    monkeypatch.setattr(pub, "compute_l0", lambda traces, **kwargs: report)
    monkeypatch.setattr(pub, "_build_workspace_client", lambda profile: _FakeClient())

    out = pub.publish(experiment_id=EXPERIMENT, warehouse_id="wh-xyz")

    assert out is report
    assert os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "wh-xyz"


def test_publish_respects_preset_warehouse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # setdefault: an explicit MLFLOW_TRACING_SQL_WAREHOUSE_ID wins.
    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "preset-wh")
    report = _report()
    monkeypatch.setattr(pub, "_build_source", lambda profile: _FakeSource())
    monkeypatch.setattr(pub, "compute_l0", lambda traces, **kwargs: report)
    monkeypatch.setattr(pub, "_build_workspace_client", lambda profile: _FakeClient())

    pub.publish(experiment_id=EXPERIMENT, warehouse_id="wh-xyz")

    assert os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "preset-wh"


def test_atomic_replace_failure_leaves_live_table_untouched() -> None:
    report = _report()
    # Fail on the staging load (the only statement that uses VALUES).
    client = _FakeClient(fail_substr=" VALUES")
    with pytest.raises(RuntimeError, match="simulated warehouse failure"):
        _atomic_replace_table(
            client,
            "wh",
            SCHEMA_FQN,
            SESSION_TABLE,
            SESSION_COLUMNS,
            _session_rows(report),
            EXPERIMENT,
        )
    stmts = client.statement_execution.statements
    # Invariant: the live table is mutated only by REPLACE WHERE, which must NOT
    # have run — a failed staging load leaves the prior snapshot intact.
    assert not any("REPLACE WHERE" in s for s in stmts)
    # Staging was still cleaned up.
    assert any(s.startswith("DROP TABLE IF EXISTS") for s in stmts)
