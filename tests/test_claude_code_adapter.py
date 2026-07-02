"""Tests for the Claude Code agent adapter.

The event-stream -> ``NormalizedTrace`` mapping is pure and tested directly with
synthetic events (no Claude Agent SDK needed). The ``run`` path is exercised for
its missing-SDK behavior, since the SDK is an optional dependency.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import ail.ingest.adapters.claude_code as cc
from ail.ingest.adapters.claude_code import AgentEvent, ClaudeCodeAdapter, _run_async
from ail.ingest.base import AgentRunResult, AgentTask, NormalizedTrace, SpanKind, TraceStatus

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
    def test_run_returns_failed_result_when_sdk_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # claude-agent-sdk is optional; simulate absence even if it is installed locally.
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        result = ClaudeCodeAdapter().run(AgentTask(prompt="hello"))
        assert result.success is False
        assert result.error is not None
        assert "claude-agent-sdk" in result.error
        assert result.trace.status is TraceStatus.ERROR
        assert result.trace.producer == "claude_code"


class TestRunSandboxWiring:
    def test_run_passes_required_filesystem_sandbox_to_sdk_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        sandbox_real = str(sandbox.resolve())
        captured_options: dict[str, Any] = {}

        @dataclass(init=False)
        class FakeClaudeAgentOptions:
            cwd: str
            allowed_tools: list[str] | None
            permission_mode: str
            mcp_servers: dict[str, object]
            system_prompt: str
            setting_sources: list[str]
            env: dict[str, str]
            hooks: object | None
            stderr: object
            sandbox: dict[str, object] | None = None

            def __init__(self, **kwargs: Any) -> None:
                captured_options.update(kwargs)
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeHookMatcher:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class FakeResultMessage:
            session_id = "fake-session"
            duration_ms = 1

        class FakeClaudeSDKClient:
            def __init__(self, *, options: FakeClaudeAgentOptions) -> None:
                self.options = options

            async def __aenter__(self) -> FakeClaudeSDKClient:
                return self

            async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

            async def query(self, prompt: str) -> None:
                assert prompt == "make a sandboxed edit"

            async def receive_response(self) -> AsyncIterator[object]:
                yield FakeResultMessage()

        sdk_module = types.ModuleType("claude_agent_sdk")
        sdk_module.ClaudeAgentOptions = FakeClaudeAgentOptions
        sdk_module.ClaudeSDKClient = FakeClaudeSDKClient
        sdk_module.HookMatcher = FakeHookMatcher

        types_module = types.ModuleType("claude_agent_sdk.types")
        types_module.AssistantMessage = type("AssistantMessage", (), {})
        types_module.ResultMessage = FakeResultMessage
        types_module.SystemMessage = type("SystemMessage", (), {})
        types_module.TextBlock = type("TextBlock", (), {})
        types_module.ToolResultBlock = type("ToolResultBlock", (), {})
        types_module.ToolUseBlock = type("ToolUseBlock", (), {})
        types_module.UserMessage = type("UserMessage", (), {})

        monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk_module)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", types_module)

        result = ClaudeCodeAdapter().run(
            AgentTask(
                prompt="make a sandboxed edit",
                cwd=str(sandbox),
                allowed_tools=["Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"],
                timeout_seconds=5,
                params={
                    "claude_code_filesystem_sandbox": {
                        "required": True,
                        "sandbox_dir": sandbox_real,
                    }
                },
            )
        )

        assert result.success is True
        assert captured_options["sandbox"] == {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
        }
        assert captured_options["permission_mode"] == "dontAsk"
        assert Path(captured_options["cwd"]).resolve() == sandbox.resolve()
        assert captured_options["allowed_tools"] == [
            "Read",
            f"Write({sandbox_real}/**)",
            f"Edit({sandbox_real}/**)",
            f"MultiEdit({sandbox_real}/**)",
            f"NotebookEdit({sandbox_real}/**)",
            "Bash(*)",
        ]


class TestHardTimeout:
    def test_run_async_returns_fallback_instead_of_raising(self) -> None:
        async def _hang() -> str:
            await asyncio.sleep(2)
            return "completed"

        sentinel = object()
        # The worker cannot finish within 0.3s, so the fallback is returned.
        result = _run_async(_hang(), timeout=0.3, on_timeout=lambda: sentinel)
        assert result is sentinel

    def test_run_returns_failed_result_on_hard_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class HangingAdapter(ClaudeCodeAdapter):
            async def _arun(self, task: AgentTask) -> AgentRunResult:
                await asyncio.sleep(2)
                return AgentRunResult(trace=NormalizedTrace(trace_id="late"))

        # Drive the hard timeout to ~0.3s (task timeout 0 + patched buffer).
        monkeypatch.setattr(cc, "_HARD_TIMEOUT_BUFFER_S", 0.3)
        result = HangingAdapter().run(AgentTask(prompt="x", timeout_seconds=0))

        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower()
        assert result.trace.status is TraceStatus.ERROR
        assert result.trace.producer == "claude_code"


def test_adapter_is_agent_adapter() -> None:
    from ail.ingest.base import AgentAdapter

    assert isinstance(ClaudeCodeAdapter(), AgentAdapter)
    assert ClaudeCodeAdapter().name == "claude_code"
