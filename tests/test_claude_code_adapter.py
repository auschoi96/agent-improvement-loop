"""Tests for the Claude Code agent adapter.

The event-stream -> ``NormalizedTrace`` mapping is pure and tested directly with
synthetic events (no Claude Agent SDK needed). The ``run`` path is exercised for
its missing-SDK behavior, since the SDK is an optional dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ail.ingest.adapters.claude_code import AgentEvent, ClaudeCodeAdapter
from ail.ingest.base import AgentTask, SpanKind, TraceStatus

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _sample_events() -> list[AgentEvent]:
    return [
        AgentEvent("assistant_turn", _at(0), {"usage": {"input_tokens": 100, "output_tokens": 30}}),
        AgentEvent(
            "tool_use", _at(1), {"id": "tu_1", "name": "Read", "input": {"file_path": "/x"}}
        ),
        AgentEvent(
            "tool_result", _at(2), {"tool_use_id": "tu_1", "content": "data", "is_error": False}
        ),
        AgentEvent("tool_use", _at(3), {"id": "tu_2", "name": "Bash", "input": {"command": "ls"}}),
        AgentEvent(
            "tool_result", _at(4), {"tool_use_id": "tu_2", "content": "boom", "is_error": True}
        ),
        AgentEvent("assistant_turn", _at(5), {"usage": {"input_tokens": 50, "output_tokens": 10}}),
        AgentEvent("result", _at(6), {"session_id": "sess-1", "duration_ms": 6000}),
    ]


class TestBuildNormalizedTrace:
    def test_tokens_summed_across_turns(self) -> None:
        trace = ClaudeCodeAdapter()._build_normalized_trace(
            _sample_events(), "sess-1", "claude-opus-4-8"
        )
        assert trace.token_usage.input_tokens == 150
        assert trace.token_usage.output_tokens == 40
        assert trace.total_tokens == 190

    def test_tool_calls_paired_with_results(self) -> None:
        trace = ClaudeCodeAdapter()._build_normalized_trace(_sample_events(), "sess-1", None)
        assert trace.tool_counts == {"Read": 1, "Bash": 1}
        read = next(tc for tc in trace.tool_calls if tc.name == "Read")
        assert read.status is TraceStatus.OK
        assert read.result == "data"
        assert read.arguments == {"file_path": "/x"}
        bash = next(tc for tc in trace.tool_calls if tc.name == "Bash")
        assert bash.status is TraceStatus.ERROR

    def test_spans_mirror_tool_calls(self) -> None:
        trace = ClaudeCodeAdapter()._build_normalized_trace(_sample_events(), "sess-1", None)
        assert all(s.kind is SpanKind.TOOL for s in trace.spans)
        assert [s.name for s in trace.spans] == ["tool_Read", "tool_Bash"]

    def test_trace_metadata(self) -> None:
        trace = ClaudeCodeAdapter()._build_normalized_trace(
            _sample_events(), "sess-1", "claude-opus-4-8"
        )
        assert trace.producer == "claude_code"
        assert trace.model == "claude-opus-4-8"
        assert trace.session_id == "sess-1"
        assert trace.status is TraceStatus.OK
        assert trace.execution_duration_ms == 6000

    def test_error_event_marks_trace_error(self) -> None:
        events = [*_sample_events(), AgentEvent("error", _at(7), {"message": "timeout"})]
        trace = ClaudeCodeAdapter()._build_normalized_trace(events, "sess-1", None)
        assert trace.status is TraceStatus.ERROR


class TestRunWithoutSdk:
    def test_run_returns_failed_result_when_sdk_missing(self) -> None:
        # claude-agent-sdk is an optional dependency and not installed in CI.
        result = ClaudeCodeAdapter().run(AgentTask(prompt="hello"))
        assert result.success is False
        assert result.error is not None
        assert "claude-agent-sdk" in result.error
        assert result.trace.status is TraceStatus.ERROR
        assert result.trace.producer == "claude_code"


def test_adapter_is_agent_adapter() -> None:
    from ail.ingest.base import AgentAdapter

    assert isinstance(ClaudeCodeAdapter(), AgentAdapter)
    assert ClaudeCodeAdapter().name == "claude_code"
