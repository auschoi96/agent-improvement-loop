"""Tests for the reusability seam: normalized record + interface contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ail.ingest.base import (
    AgentAdapter,
    NormalizedSpan,
    NormalizedTrace,
    SpanKind,
    TokenUsage,
    ToolCall,
    TraceSource,
    TraceStatus,
)


def test_abstract_interfaces_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        TraceSource()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        AgentAdapter()  # type: ignore[abstract]


class TestTraceStatus:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("OK", TraceStatus.OK),
            ("success", TraceStatus.OK),
            ("ERROR", TraceStatus.ERROR),
            ("failed", TraceStatus.ERROR),
            ("running", TraceStatus.IN_PROGRESS),
            ("nonsense", TraceStatus.UNKNOWN),
            (None, TraceStatus.UNKNOWN),
            (TraceStatus.OK, TraceStatus.OK),
        ],
    )
    def test_coerce(self, value: object, expected: TraceStatus) -> None:
        assert TraceStatus.coerce(value) is expected

    def test_coerce_reads_enum_value_attribute(self) -> None:
        class FakeState:
            value = "OK"

        assert TraceStatus.coerce(FakeState()) is TraceStatus.OK


class TestSpanKind:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("LLM", SpanKind.LLM),
            ("tool", SpanKind.TOOL),
            ("AGENT", SpanKind.AGENT),
            ("weird", SpanKind.UNKNOWN),
            (None, SpanKind.UNKNOWN),
        ],
    )
    def test_coerce(self, value: object, expected: SpanKind) -> None:
        assert SpanKind.coerce(value) is expected


class TestTokenUsage:
    def test_total_defaults_to_input_plus_output(self) -> None:
        assert TokenUsage(input_tokens=10, output_tokens=5).total_tokens == 15

    def test_explicit_total_is_preferred(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=5, _total_tokens=99)
        assert usage.total_tokens == 99

    def test_cache_tokens(self) -> None:
        usage = TokenUsage(cache_creation_input_tokens=3, cache_read_input_tokens=4)
        assert usage.cache_tokens == 7

    def test_add(self) -> None:
        total = TokenUsage(input_tokens=1, output_tokens=2) + TokenUsage(
            input_tokens=3, output_tokens=4
        )
        assert total.input_tokens == 4
        assert total.output_tokens == 6
        assert total.total_tokens == 10


class TestToolCall:
    def test_plain_tool_is_not_mcp(self) -> None:
        tc = ToolCall(id="1", name="Bash")
        assert tc.is_mcp is False
        assert tc.mcp_server is None

    def test_mcp_tool(self) -> None:
        tc = ToolCall(id="1", name="mcp__databricks__execute_sql")
        assert tc.is_mcp is True
        assert tc.mcp_server == "databricks"


class TestNormalizedSpan:
    def test_duration_ms(self) -> None:
        span = NormalizedSpan(
            span_id="s",
            name="n",
            start_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
            end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
        )
        assert span.duration_ms == 2000.0

    def test_duration_ms_none_without_both_ends(self) -> None:
        assert NormalizedSpan(span_id="s", name="n").duration_ms is None


class TestNormalizedTrace:
    def test_token_and_tool_aggregates(self) -> None:
        trace = NormalizedTrace(
            trace_id="t",
            token_usage=TokenUsage(input_tokens=100, output_tokens=20),
            tool_calls=[
                ToolCall(id="1", name="Read"),
                ToolCall(id="2", name="Read"),
                ToolCall(id="3", name="Bash"),
            ],
        )
        assert trace.total_tokens == 120
        assert trace.total_tool_calls == 3
        assert trace.tool_counts == {"Read": 2, "Bash": 1}

    def test_duration_prefers_execution_duration_ms(self) -> None:
        trace = NormalizedTrace(trace_id="t", execution_duration_ms=4200)
        assert trace.duration_seconds == 4.2

    def test_duration_falls_back_to_timestamps(self) -> None:
        trace = NormalizedTrace(
            trace_id="t",
            request_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
            end_time=datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
        )
        assert trace.duration_seconds == 3.0

    def test_duration_none_when_unknown(self) -> None:
        assert NormalizedTrace(trace_id="t").duration_seconds is None
