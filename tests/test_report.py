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
    # ~943K tokens (matches the documented 943K session), re-runs the same shell
    # prologue 14x and re-reads one file 5x.
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
        token_usage=TokenUsage(input_tokens=900_000, output_tokens=43_000, _total_tokens=943_000),
        tool_calls=tools,
        execution_duration_ms=9_000_000,
    )


def _second_big() -> NormalizedTrace:
    # ~549K tokens (matches the documented 549K session)
    return NormalizedTrace(
        trace_id="trace:/loc/big2",
        status=TraceStatus.OK,
        producer="claude_code",
        model="claude-opus-4-8",
        token_usage=TokenUsage(input_tokens=520_000, output_tokens=29_000, _total_tokens=549_000),
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

    def test_reconciliation_is_derived(self) -> None:
        _, payload = build_example1_diagnosis(_report())
        rec = payload["reconciliation_with_doc"]
        # high sessions (~943K, ~549K) are within tolerance of the documented pair
        assert rec["high_token_sessions"]["status"] == "match"
        # 14x shell prologue >= documented floor 13x -> reproduced
        assert rec["shell_boilerplate"]["reproduced"] is True
        assert rec["shell_boilerplate"]["status"].startswith("reproduced")
        # max read-same-path is 5x, far below the documented 34x -> NOT reproduced
        assert rec["read_same_path"]["reproduced"] is False
        assert "NOT reproduced" in rec["read_same_path"]["status"]
        assert rec["read_same_path"]["live_max_read_same_path"] == 5

    def test_reconciliation_flags_drift_when_not_reproduced(self) -> None:
        # a corpus with no big sessions / no boilerplate must NOT report a match
        small = compute_l0([_small()], experiment_id="x", generated_at="x")
        _, payload = build_example1_diagnosis(small)
        rec = payload["reconciliation_with_doc"]
        assert rec["high_token_sessions"]["status"].startswith("drift")
        assert rec["shell_boilerplate"]["reproduced"] is False
        assert rec["read_same_path"]["reproduced"] is False

    def test_markdown_renders_key_facts(self) -> None:
        md, _ = build_example1_diagnosis(_report())
        assert "# Example 1 — Token-Waste Diagnosis" in md
        assert "660599403165942" in md
        assert "943,000" in md  # huge session total tokens
        assert "Reconciliation" in md
        assert "14×" in md  # shell boilerplate repeats


@pytest.mark.live
def test_live_pull_and_l0(tmp_path: object) -> None:
    """Acceptance: pull the reference experiment, compute L0, sanity-check it.

    Guarded by ``AIL_LIVE_MLFLOW=1`` so the default suite is green offline.

    Auth: on the reference workspace the experiment is UC-table-backed and read
    through MLflow 3's v4 trace REST store, which rejects OAuth-profile
    credentials for the span ``batchGet`` — so a CLI profile alone does **not**
    work here. Set ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` (e.g.
    ``export DATABRICKS_TOKEN=$(databricks auth token -p <profile> | jq -r .access_token)``);
    ``_build_source`` uses those. ``AIL_DATABRICKS_PROFILE`` is only a fallback
    for workspaces where profile auth does reach the span store.
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
