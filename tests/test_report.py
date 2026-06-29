"""Tests for the Example 1 diagnosis builder and the guarded live pull.

``build_example1_diagnosis`` is a pure function over an L0 report, so it is
tested offline against a hand-built corpus shaped like Example 1 (a couple of
huge sessions, repeated re-reads, re-run shell boilerplate). The live pull is
marked ``live`` and skips cleanly without Databricks creds.
"""

from __future__ import annotations

import os

import pytest

from ail.ingest.base import NormalizedTrace, TokenUsage, ToolCall, TraceStatus
from ail.metrics.l0_deterministic import compute_l0
from ail.metrics.report import (
    REFERENCE_EXPERIMENT,
    build_example1_diagnosis,
)

REFERENCE_PROFILE = "dais-demo"


def _big_session() -> NormalizedTrace:
    # 600K tokens, re-runs the same shell prologue 14x and re-reads one file 5x
    tools: list[ToolCall] = []
    for i in range(14):
        tools.append(
            ToolCall(id=f"b{i}", name="Bash", arguments={"command": f"cd /repo\nstep {i}"})
        )
    for i in range(5):
        tools.append(
            ToolCall(
                id=f"r{i}", name="Read", arguments={"file_path": "/repo/schema.ts", "offset": i}
            )
        )
    return NormalizedTrace(
        trace_id="trace:/loc/huge",
        status=TraceStatus.OK,
        producer="claude_code",
        model="claude-opus-4-8",
        session_id="sess-huge",
        token_usage=TokenUsage(input_tokens=560_000, output_tokens=40_000, _total_tokens=600_000),
        tool_calls=tools,
        execution_duration_ms=9_000_000,
    )


def _second_big() -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="trace:/loc/big2",
        status=TraceStatus.OK,
        producer="claude_code",
        model="claude-opus-4-8",
        token_usage=TokenUsage(input_tokens=500_000, output_tokens=20_000, _total_tokens=520_000),
        tool_calls=[ToolCall(id="x", name="Bash", arguments={"command": "cd /repo\nmake"})],
        execution_duration_ms=1_000_000,
    )


def _small() -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="trace:/loc/small",
        status=TraceStatus.OK,
        producer="claude_code",
        model="claude-sonnet-4-6",
        token_usage=TokenUsage(input_tokens=10_000, output_tokens=1_000, _total_tokens=11_000),
        tool_calls=[ToolCall(id="s", name="Grep", arguments={"pattern": "foo"})],
    )


def _report():  # type: ignore[no-untyped-def]
    return compute_l0(
        [_big_session(), _second_big(), _small()],
        experiment_id="660599403165942",
        generated_at="2026-06-29T00:00:00Z",
    )


class TestDiagnosis:
    def test_high_token_sessions_detected(self) -> None:
        _, payload = build_example1_diagnosis(_report())
        ids = [h["trace_id"] for h in payload["high_token_sessions"]]
        assert "trace:/loc/huge" in ids
        assert "trace:/loc/big2" in ids
        assert "trace:/loc/small" not in ids  # below threshold

    def test_shell_boilerplate_ranking(self) -> None:
        _, payload = build_example1_diagnosis(_report())
        shell = payload["tool_redundancy"]["shell_boilerplate_top"]
        assert shell[0]["trace_id"] == "trace:/loc/huge"
        assert shell[0]["count"] == 14
        assert shell[0]["identity"] == "cd /repo"

    def test_repeated_reads_ranking(self) -> None:
        _, payload = build_example1_diagnosis(_report())
        reads = payload["tool_redundancy"]["repeated_file_reads_top"]
        assert reads[0]["tool"] == "Read"
        assert reads[0]["count"] == 5
        assert reads[0]["identity"] == "/repo/schema.ts"

    def test_reconciliation_block(self) -> None:
        _, payload = build_example1_diagnosis(_report())
        rec = payload["reconciliation_with_doc"]
        assert rec["high_token_sessions"]["status"] == "match"
        assert "reproduced" in rec["shell_boilerplate"]["status"]
        # the synthetic corpus has max read-same-path 5x, so the 34x doc figure is flagged
        assert "NOT reproduced" in rec["read_same_path"]["status"]
        assert rec["read_same_path"]["live_max_read_same_path"] == 5

    def test_markdown_renders_key_facts(self) -> None:
        md, _ = build_example1_diagnosis(_report())
        assert "# Example 1 — Token-Waste Diagnosis" in md
        assert "660599403165942" in md
        assert "600,000" in md  # huge session total tokens
        assert "Reconciliation" in md
        assert "14×" in md  # shell boilerplate repeats


@pytest.mark.live
def test_live_pull_and_l0(tmp_path: object) -> None:
    """Acceptance: pull the reference experiment, compute L0, sanity-check it.

    Guarded by ``AIL_LIVE_MLFLOW=1`` so the default suite is green offline.
    Auth: set DATABRICKS_HOST + DATABRICKS_TOKEN, or rely on the
    ``AIL_DATABRICKS_PROFILE`` CLI profile (default ``dais-demo``).
    """
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        pytest.skip("set AIL_LIVE_MLFLOW=1 to run the live MLflow pull")

    from ail.metrics.report import _build_source

    profile = os.environ.get("AIL_DATABRICKS_PROFILE", REFERENCE_PROFILE)
    source = _build_source(profile)
    traces = source.fetch_traces(experiment_id=REFERENCE_EXPERIMENT, max_results=200)
    assert traces, "expected traces in the reference experiment"

    report = compute_l0(traces, experiment_id=REFERENCE_EXPERIMENT, generated_at="live")
    assert report.n_traces == len(traces)
    assert report.aggregate.tokens.total_tokens > 0
    assert report.aggregate.cost.priced_traces >= 1  # at least one Claude model priced
    # the contract must serialize cleanly (a UI reads exactly this)
    assert report.model_dump_json()

    md, payload = build_example1_diagnosis(report)
    assert payload["corpus"]["n_traces"] == report.n_traces
    assert "Example 1" in md
