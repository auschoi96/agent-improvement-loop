"""Tests for the Codex agent adapter and the native-tracing enable helper.

The rollout-transcript -> ``NormalizedTrace`` mapping is pure and tested directly
with a synthetic rollout (no Codex CLI needed). ``enable_codex_tracing`` is
exercised against a temp ``.codex`` directory. The ``run`` path is tested for
its missing-CLI behavior and for a mocked happy path (subprocess + rollout
discovery stubbed), since the Codex CLI is an external binary.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import ail.ingest.adapters.codex as codex
from ail.ingest.adapters.codex import (
    CODEX_AGENT,
    CodexAdapter,
    CodexTracingError,
    codex_cohort_tags,
    enable_codex_tracing,
    normalize_codex_rollout,
    read_rollout,
)
from ail.ingest.base import AgentAdapter, AgentTask, SpanKind, TraceStatus


def _ts(second: int) -> str:
    return f"2026-06-29T10:00:{second:02d}.000Z"


def _full_rollout() -> list[dict]:
    """A single-session rollout with two tool calls (one failing) and usage."""
    return [
        {
            "timestamp": _ts(0),
            "type": "session_meta",
            "payload": {
                "id": "019f0560-aaaa-bbbb-cccc-000000000001",
                "cwd": "/repo",
                "cli_version": "0.142.0-alpha.4",
                "model_provider": "Databricks",
            },
        },
        {
            "timestamp": _ts(1),
            "type": "turn_context",
            "payload": {"turn_id": "t1", "model": "gpt-5.5"},
        },
        {"timestamp": _ts(1), "type": "event_msg", "payload": {"type": "task_started"}},
        {
            "timestamp": _ts(2),
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "<system injection>"}],
            },
        },
        {
            "timestamp": _ts(2),
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list the repo"}],
            },
        },
        {
            "timestamp": _ts(3),
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_ok",
                "arguments": json.dumps({"command": ["ls"]}),
            },
        },
        {
            "timestamp": _ts(4),
            "type": "event_msg",
            "payload": {"type": "exec_command_end", "call_id": "call_ok", "exit_code": 0},
        },
        {
            "timestamp": _ts(4),
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_ok",
                "output": "README.md\nsrc",
            },
        },
        {
            "timestamp": _ts(5),
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_bad",
                "arguments": json.dumps({"command": ["missing-cmd"]}),
            },
        },
        {
            "timestamp": _ts(6),
            "type": "event_msg",
            "payload": {
                "type": "exec_command_end",
                "call_id": "call_bad",
                "status": "failed",
                "exit_code": 127,
            },
        },
        {
            "timestamp": _ts(6),
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_bad",
                "output": "command not found",
            },
        },
        {
            "timestamp": _ts(7),
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done — repo listed."}],
            },
        },
        {
            "timestamp": _ts(8),
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 1000,
                        "cached_input_tokens": 400,
                        "output_tokens": 50,
                        "reasoning_output_tokens": 10,
                        "total_tokens": 1050,
                    }
                },
            },
        },
        {"timestamp": _ts(9), "type": "event_msg", "payload": {"type": "task_complete"}},
    ]


class TestNormalizeRollout:
    def test_producer_and_tag(self) -> None:
        trace = normalize_codex_rollout(_full_rollout())
        assert trace.producer == CODEX_AGENT
        assert trace.tags == {"ail.agent": "codex"}

    def test_identity_fields(self) -> None:
        trace = normalize_codex_rollout(_full_rollout(), experiment_id="660599403165942")
        assert trace.trace_id == "019f0560-aaaa-bbbb-cccc-000000000001"
        assert trace.session_id == "019f0560-aaaa-bbbb-cccc-000000000001"
        assert trace.model == "gpt-5.5"
        assert trace.experiment_id == "660599403165942"
        assert trace.metadata["codex.model_provider"] == "Databricks"
        assert trace.metadata["codex.cli_version"] == "0.142.0-alpha.4"

    def test_tool_pairing_and_status(self) -> None:
        trace = normalize_codex_rollout(_full_rollout())
        assert trace.tool_counts == {"shell": 2}
        ok = next(c for c in trace.tool_calls if c.id == "call_ok")
        bad = next(c for c in trace.tool_calls if c.id == "call_bad")
        assert ok.status is TraceStatus.OK
        assert ok.result == "README.md\nsrc"
        assert ok.arguments == {"command": ["ls"]}
        # A non-zero shell exit marks the tool span ERROR (via exec_command_end)...
        assert bad.status is TraceStatus.ERROR

    def test_tool_failure_does_not_fail_the_trace(self) -> None:
        # ...but the agent recovered, so the trace itself stays OK.
        trace = normalize_codex_rollout(_full_rollout())
        assert trace.status is TraceStatus.OK

    def test_spans_mirror_tool_calls(self) -> None:
        trace = normalize_codex_rollout(_full_rollout())
        assert all(s.kind is SpanKind.TOOL for s in trace.spans)
        assert [s.name for s in trace.spans] == ["tool_shell", "tool_shell"]

    def test_token_usage_splits_cache_out_of_input(self) -> None:
        usage = normalize_codex_rollout(_full_rollout()).token_usage
        # Codex input_tokens (1000) is inclusive of cached (400); the cache read
        # is split out so cost is not double-counted, and the authoritative total
        # is preserved.
        assert usage.input_tokens == 600
        assert usage.cache_read_input_tokens == 400
        assert usage.cache_creation_input_tokens == 0
        assert usage.output_tokens == 50
        assert usage.total_tokens == 1050
        assert usage.cache_tokens == 400

    def test_previews_skip_developer_injection(self) -> None:
        trace = normalize_codex_rollout(_full_rollout())
        assert trace.request_preview == "list the repo"
        assert trace.response_preview == "Done — repo listed."

    def test_duration_from_timestamps(self) -> None:
        trace = normalize_codex_rollout(_full_rollout())
        assert trace.execution_duration_ms == 9000

    def test_error_event_marks_trace_error(self) -> None:
        records = _full_rollout()
        records.append({"timestamp": _ts(10), "type": "event_msg", "payload": {"type": "error"}})
        assert normalize_codex_rollout(records).status is TraceStatus.ERROR

    def test_no_token_count_yields_zero_usage(self) -> None:
        records = [r for r in _full_rollout() if r["payload"].get("type") != "token_count"]
        usage = normalize_codex_rollout(records).token_usage
        assert usage.total_tokens == 0

    def test_empty_rollout_gets_generated_trace_id(self) -> None:
        trace = normalize_codex_rollout([])
        assert trace.trace_id.startswith("codex-")
        assert trace.producer == CODEX_AGENT


class TestReadRollout:
    def test_skips_blank_and_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "rollout.jsonl"
        path.write_text(
            json.dumps(_full_rollout()[0])
            + "\n\n"
            + "{not json}\n"
            + json.dumps(_full_rollout()[1])
            + "\n",
            encoding="utf-8",
        )
        records = read_rollout(path)
        assert [r["type"] for r in records] == ["session_meta", "turn_context"]

    def test_normalize_accepts_a_path(self, tmp_path: Path) -> None:
        path = tmp_path / "rollout.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in _full_rollout()) + "\n", encoding="utf-8")
        trace = normalize_codex_rollout(path)
        assert trace.model == "gpt-5.5"
        assert trace.total_tokens == 1050


class TestEnableCodexTracing:
    def test_creates_config_and_tracing_json_when_absent(self, tmp_path: Path) -> None:
        setup = enable_codex_tracing(
            "660599403165942", codex_dir=tmp_path, tracking_uri="databricks"
        )
        assert setup.notify_added is True
        config_text = setup.config_path.read_text()
        assert 'notify = ["mlflow-codex", "notify-hook"]' in config_text
        tracing = json.loads(setup.tracing_config_path.read_text())
        assert tracing == {"trackingUri": "databricks", "experimentId": "660599403165942"}

    def test_notify_prepended_above_table_headers(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('model = "gpt-5.5"\n\n[model_providers.Databricks]\nname = "x"\n')
        enable_codex_tracing("123", codex_dir=tmp_path)
        lines = config.read_text().splitlines()
        notify_idx = next(i for i, ln in enumerate(lines) if ln.startswith("notify"))
        table_idx = next(i for i, ln in enumerate(lines) if ln.startswith("["))
        # TOML requires a top-level key to precede any [table] header.
        assert notify_idx < table_idx
        assert 'model = "gpt-5.5"' in config.read_text()

    def test_idempotent_when_mlflow_notify_present(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('notify = ["mlflow-codex", "notify-hook"]\nmodel = "gpt-5.5"\n')
        setup = enable_codex_tracing("123", codex_dir=tmp_path)
        assert setup.notify_added is False
        # Not duplicated.
        assert config.read_text().count("notify =") == 1

    def test_refuses_to_clobber_foreign_notify(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('notify = ["my-own-notifier"]\n')
        with pytest.raises(CodexTracingError):
            enable_codex_tracing("123", codex_dir=tmp_path)
        # Original left intact.
        assert "my-own-notifier" in config.read_text()

    def test_force_replaces_foreign_notify(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('notify = ["my-own-notifier"]\nmodel = "x"\n')
        setup = enable_codex_tracing("123", codex_dir=tmp_path, force=True)
        assert setup.notify_added is True
        text = config.read_text()
        assert "my-own-notifier" not in text
        assert 'notify = ["mlflow-codex", "notify-hook"]' in text
        assert 'model = "x"' in text

    def test_user_scope_honors_codex_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "codex-home"
        monkeypatch.setenv("CODEX_HOME", str(home))
        setup = enable_codex_tracing("123", scope="user")
        assert setup.config_path == home / "config.toml"
        assert setup.config_path.exists()

    def test_rejects_bad_scope(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            enable_codex_tracing("123", scope="global", codex_dir=tmp_path)


class TestCodexAdapterContract:
    def test_is_agent_adapter_named_codex(self) -> None:
        assert isinstance(CodexAdapter(), AgentAdapter)
        assert CodexAdapter().name == "codex"

    def test_cohort_tags(self) -> None:
        assert codex_cohort_tags() == {"ail.agent": "codex"}

    def test_run_returns_failed_result_when_cli_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(codex.shutil, "which", lambda _cmd: None)
        monkeypatch.setattr(codex.os.path, "isfile", lambda _p: False)
        result = CodexAdapter(command="codex-not-here").run(AgentTask(prompt="hi"))
        assert result.success is False
        assert result.error is not None
        assert "not on PATH" in result.error
        assert result.trace.status is TraceStatus.ERROR
        assert result.trace.producer == CODEX_AGENT
        assert result.trace.tags == {"ail.agent": "codex"}


class TestCodexAdapterRun:
    def test_run_captures_rollout_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sessions = tmp_path / "sessions" / "2026" / "06" / "29"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout-2026-06-29T10-00-00-019f0560.jsonl"

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            # Codex writes the rollout and the last-agent-message file.
            rollout.write_text(
                "\n".join(json.dumps(r) for r in _full_rollout()) + "\n", encoding="utf-8"
            )
            last = cmd[cmd.index("--output-last-message") + 1]
            Path(last).write_text("Done — repo listed.", encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(codex.shutil, "which", lambda _cmd: "/usr/local/bin/codex")
        monkeypatch.setattr(codex.subprocess, "run", fake_run)

        adapter = CodexAdapter(codex_home=tmp_path)
        result = adapter.run(AgentTask(prompt="list the repo", model="gpt-5.5"))

        assert result.success is True
        assert result.output_text == "Done — repo listed."
        assert result.session_id == "019f0560-aaaa-bbbb-cccc-000000000001"
        assert result.trace.producer == CODEX_AGENT
        assert result.trace.tags == {"ail.agent": "codex"}
        assert result.trace.total_tokens == 1050

    def test_run_fails_when_no_rollout_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "sessions").mkdir()

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            return types.SimpleNamespace(returncode=0, stdout="", stderr="boom")

        monkeypatch.setattr(codex.shutil, "which", lambda _cmd: "/usr/local/bin/codex")
        monkeypatch.setattr(codex.subprocess, "run", fake_run)

        result = CodexAdapter(codex_home=tmp_path).run(AgentTask(prompt="x"))
        assert result.success is False
        assert result.error is not None
        assert "no rollout transcript" in result.error
