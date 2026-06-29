"""Claude Code :class:`~ail.ingest.base.AgentAdapter`.

HARVEST
-------
Source: ``databricks-solutions/ai-dev-kit``
        ``.test/src/skill_test/agent/executor.py`` (the ``ClaudeSDKClient``
        runner ~L449-L706, event capture, the
        ``_build_trace_metrics`` mapper ~L82, the agent-env / MCP helpers, and
        the ``_run_in_fresh_loop`` sync wrapper ~L709).
Commit: c4947868f06fbfbb8cb666cbfba15888127b8a3a
License: see PROVENANCE.md (Databricks "DB license").

CHANGES FROM UPSTREAM
---------------------
* Repackaged the free ``run_agent`` / ``run_agent_sync_wrapper`` functions as a
  :class:`ClaudeCodeAdapter` implementing :class:`AgentAdapter`.
* ``_build_trace_metrics`` (which produced the Claude-specific ``TraceMetrics``)
  is reimplemented as ``_build_normalized_trace`` producing the
  producer-agnostic :class:`~ail.ingest.base.NormalizedTrace`, so adapter-
  captured traces match :class:`TraceSource` output exactly.
* Dropped the ai-dev-kit-internal ``SkillTestConfig`` coupling from the MLflow
  Stop hook; tracing is configured from env vars + the experiment passed to the
  adapter. The settings-file search was generalized away from ``.test/`` paths.
* The Claude Agent SDK is imported lazily; if it is absent, :meth:`run` returns
  a failed :class:`AgentRunResult` rather than raising (matches upstream).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedSpan,
    NormalizedTrace,
    SpanKind,
    TokenUsage,
    ToolCall,
    TraceStatus,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Env var prefixes forwarded into the agent subprocess (harvested).
_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_", "DATABRICKS_", "MLFLOW_")
# Internal Claude Code vars that must not leak into the child process.
_SKIP_ENV_KEYS = {
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY",
}
_DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]


@dataclass(slots=True)
class AgentEvent:
    """A captured event from the agent execution stream (harvested)."""

    type: str  # tool_use, tool_result, text, assistant_turn, result, system, error
    timestamp: datetime
    data: dict[str, Any]


class ClaudeCodeAdapter(AgentAdapter):
    """Run Claude Code via the Claude Agent SDK and capture a normalized trace.

    Args:
        mlflow_experiment: Optional MLflow experiment to log the run's trace to
            via a Stop hook (best-effort, fire-and-forget). When ``None``,
            tracing is skipped and the trace is built purely from streamed
            events.
        default_allowed_tools: Tools to allow when a task does not specify any
            and no MCP servers are configured.
    """

    name = "claude_code"

    def __init__(
        self,
        mlflow_experiment: str | None = None,
        default_allowed_tools: list[str] | None = None,
    ) -> None:
        self.mlflow_experiment = mlflow_experiment
        self.default_allowed_tools = default_allowed_tools or list(_DEFAULT_ALLOWED_TOOLS)

    # -- AgentAdapter interface -------------------------------------------

    def run(self, task: AgentTask) -> AgentRunResult:
        """Run ``task`` synchronously (over a dedicated event loop)."""
        return _run_in_fresh_loop(self._arun(task))

    async def _arun(self, task: AgentTask) -> AgentRunResult:
        """Async core: stream the SDK, capture events, build a trace."""
        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
                ClaudeSDKClient,
                HookMatcher,
            )
            from claude_agent_sdk.types import (
                AssistantMessage,
                ResultMessage,
                SystemMessage,
                TextBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
            )
        except ImportError:
            return AgentRunResult(
                trace=NormalizedTrace(
                    trace_id=f"error-{uuid.uuid4().hex}",
                    status=TraceStatus.ERROR,
                    producer=self.name,
                ),
                success=False,
                error=(
                    "claude-agent-sdk not installed. "
                    "Install with: pip install 'claude-agent-sdk>=0.1.39'"
                ),
            )

        session_id = str(uuid.uuid4())
        events: list[AgentEvent] = []
        response_parts: list[str] = []

        mcp_config = _load_mcp_config()
        allowed_tools = task.allowed_tools
        if allowed_tools is None:
            allowed_tools = None if mcp_config else list(self.default_allowed_tools)

        env = _get_agent_env()
        if task.model:
            env["ANTHROPIC_MODEL"] = task.model
        env.pop("CLAUDECODE", None)  # don't let the child think it is nested

        if mcp_config:
            mcp_env = {k: v for k, v in env.items() if k.startswith("DATABRICKS_")}
            for server_cfg in mcp_config.values():
                if "env" not in server_cfg and mcp_env:
                    server_cfg["env"] = mcp_env

        hooks: dict[str, Any] = {}
        if self.mlflow_experiment:
            hook = _build_mlflow_stop_hook(self.mlflow_experiment, self.name)
            if hook is not None:
                hooks["Stop"] = [HookMatcher(hooks=[hook])]

        stderr_lines: list[str] = []

        def _stderr_callback(line: str) -> None:
            stripped = line.strip()
            if stripped:
                stderr_lines.append(stripped)

        options = ClaudeAgentOptions(
            cwd=task.cwd or os.getcwd(),
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            mcp_servers=mcp_config or {},
            system_prompt=task.system_prompt or "",
            setting_sources=[],  # inject our own context; no ambient project skills
            env=env,
            hooks=hooks or None,
            stderr=_stderr_callback,
        )

        start = time.monotonic()
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(task.prompt)
                async for msg in client.receive_response():
                    now = datetime.now(UTC)
                    if time.monotonic() - start > task.timeout_seconds:
                        events.append(
                            AgentEvent(
                                "error", now, {"message": f"Timeout after {task.timeout_seconds}s"}
                            )
                        )
                        break

                    if isinstance(msg, AssistantMessage):
                        usage_data: dict[str, int] = {}
                        if getattr(msg, "usage", None):
                            usage = msg.usage
                            usage_data = {
                                "input_tokens": getattr(usage, "input_tokens", 0),
                                "output_tokens": getattr(usage, "output_tokens", 0),
                                "cache_creation_input_tokens": getattr(
                                    usage, "cache_creation_input_tokens", 0
                                ),
                                "cache_read_input_tokens": getattr(
                                    usage, "cache_read_input_tokens", 0
                                ),
                            }
                        events.append(AgentEvent("assistant_turn", now, {"usage": usage_data}))

                        for block in getattr(msg, "content", []):
                            if isinstance(block, TextBlock):
                                response_parts.append(block.text)
                                events.append(AgentEvent("text", now, {"text": block.text}))
                            elif isinstance(block, ToolUseBlock):
                                events.append(
                                    AgentEvent(
                                        "tool_use",
                                        now,
                                        {
                                            "id": block.id,
                                            "name": block.name,
                                            "input": block.input
                                            if isinstance(block.input, dict)
                                            else {},
                                        },
                                    )
                                )
                            elif isinstance(block, ToolResultBlock):
                                events.append(
                                    AgentEvent("tool_result", now, _tool_result_data(block))
                                )

                    elif isinstance(msg, UserMessage):
                        for block in getattr(msg, "content", []):
                            if isinstance(block, ToolResultBlock):
                                events.append(
                                    AgentEvent("tool_result", now, _tool_result_data(block))
                                )

                    elif isinstance(msg, ResultMessage):
                        session_id = getattr(msg, "session_id", session_id)
                        events.append(
                            AgentEvent(
                                "result",
                                now,
                                {
                                    "session_id": session_id,
                                    "duration_ms": getattr(msg, "duration_ms", None),
                                },
                            )
                        )

                    elif isinstance(msg, SystemMessage):
                        events.append(
                            AgentEvent(
                                "system",
                                now,
                                {
                                    "subtype": getattr(msg, "subtype", ""),
                                    "data": getattr(msg, "data", {}),
                                },
                            )
                        )
        except Exception as e:  # noqa: BLE001 - capture, don't raise, per AgentAdapter contract
            detail = "; ".join(stderr_lines[-5:]) if stderr_lines else "no stderr"
            logger.error("Claude agent run failed: %s | stderr: %s", e, detail)
            events.append(
                AgentEvent("error", datetime.now(UTC), {"message": f"{e} | stderr: {detail}"})
            )

        duration_ms = int((time.monotonic() - start) * 1000)
        model = task.model or os.environ.get("ANTHROPIC_MODEL")
        trace = self._build_normalized_trace(events, session_id, model)
        has_error = any(e.type == "error" for e in events)

        return AgentRunResult(
            trace=trace,
            output_text="\n".join(response_parts),
            success=not has_error,
            error=next((e.data.get("message") for e in events if e.type == "error"), None),
            session_id=session_id,
            duration_ms=duration_ms,
            raw=events,
        )

    def _build_normalized_trace(
        self,
        events: list[AgentEvent],
        session_id: str,
        model: str | None,
    ) -> NormalizedTrace:
        """Build a :class:`NormalizedTrace` from captured stream events.

        Reimplements upstream ``_build_trace_metrics`` against the normalized
        record: ``tool_use``/``tool_result`` pairs become :class:`ToolCall`s
        (and ``TOOL`` spans), ``assistant_turn`` usage is summed into the trace
        :class:`TokenUsage`, and trace timing comes from the first/last events.
        """
        tool_calls: dict[str, ToolCall] = {}
        ordered_tools: list[ToolCall] = []
        usage = TokenUsage()
        start_time: datetime | None = None
        end_time: datetime | None = None

        for event in events:
            if start_time is None:
                start_time = event.timestamp
            end_time = event.timestamp

            if event.type == "tool_use":
                tc = ToolCall(
                    id=str(event.data.get("id") or uuid.uuid4().hex),
                    name=str(event.data.get("name", "unknown")),
                    arguments=event.data.get("input", {}) or {},
                    status=TraceStatus.IN_PROGRESS,
                    span_id=str(event.data.get("id") or ""),
                    start_time=event.timestamp,
                )
                tool_calls[tc.id] = tc
                ordered_tools.append(tc)

            elif event.type == "tool_result":
                pending = tool_calls.get(str(event.data.get("tool_use_id", "")))
                if pending is not None:
                    content = event.data.get("content", "")
                    pending.result = content if isinstance(content, str) else str(content)
                    is_error = bool(event.data.get("is_error"))
                    pending.status = TraceStatus.ERROR if is_error else TraceStatus.OK

            elif event.type == "assistant_turn":
                turn = event.data.get("usage", {}) or {}
                usage = usage + TokenUsage(
                    input_tokens=int(turn.get("input_tokens", 0) or 0),
                    output_tokens=int(turn.get("output_tokens", 0) or 0),
                    cache_creation_input_tokens=int(
                        turn.get("cache_creation_input_tokens", 0) or 0
                    ),
                    cache_read_input_tokens=int(turn.get("cache_read_input_tokens", 0) or 0),
                )

            elif event.type == "result":
                end_time = event.timestamp

        spans = [
            NormalizedSpan(
                span_id=tc.span_id or tc.id,
                name=f"tool_{tc.name}",
                kind=SpanKind.TOOL,
                status=tc.status,
                start_time=tc.start_time,
                inputs=tc.arguments,
                outputs=tc.result,
            )
            for tc in ordered_tools
        ]

        has_error = any(e.type == "error" for e in events)
        duration_ms = None
        if start_time and end_time:
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return NormalizedTrace(
            trace_id=session_id,
            status=TraceStatus.ERROR if has_error else TraceStatus.OK,
            producer=self.name,
            model=model,
            session_id=session_id,
            request_time=start_time,
            end_time=end_time,
            execution_duration_ms=duration_ms,
            token_usage=usage,
            spans=spans,
            tool_calls=ordered_tools,
        )


def _tool_result_data(block: Any) -> dict[str, Any]:
    return {
        "tool_use_id": getattr(block, "tool_use_id", ""),
        "content": getattr(block, "content", ""),
        "is_error": getattr(block, "is_error", False),
    }


# ---------------------------------------------------------------------------
# Subprocess env / MCP config helpers (harvested from executor.py, generalized)
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Walk up from this file to the nearest directory containing ``.git``."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def _resolve_env_refs(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references from ``os.environ``."""

    def _replace(m: re.Match[str]) -> str:
        var = m.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(var, m.group(0))

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _get_agent_env() -> dict[str, str]:
    """Build env vars for the agent subprocess.

    Loads optional FMAPI settings from ``.ail/agent_settings.json`` or
    ``.claude/agent_settings.json`` (``{"env": {...}}`` with ``${VAR}``
    interpolation), then overlays matching-prefix env vars. Internal Claude Code
    vars are stripped so the child does not think it is nested.
    """
    env: dict[str, str] = {}
    repo_root = _find_repo_root()
    for p in (
        repo_root / ".ail" / "agent_settings.json",
        repo_root / ".claude" / "agent_settings.json",
    ):
        if p.exists():
            try:
                file_env = json.loads(p.read_text()).get("env", {})
                for k, v in file_env.items():
                    if isinstance(v, str):
                        env[k] = _resolve_env_refs(v)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load agent settings %s: %s", p, e)
            break

    for key, value in os.environ.items():
        if key in _SKIP_ENV_KEYS:
            continue
        if value and any(key.startswith(prefix) for prefix in _ENV_PREFIXES):
            env[key] = value

    for k in _SKIP_ENV_KEYS:
        env.pop(k, None)

    env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    env.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "600000")
    return env


def _load_mcp_config() -> dict[str, Any]:
    """Load MCP server config from ``.mcp.json`` at the repo root, if present."""
    mcp_json = _find_repo_root() / ".mcp.json"
    if not mcp_json.exists():
        return {}
    try:
        data = json.loads(mcp_json.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    root = str(_find_repo_root())
    resolved: dict[str, Any] = {}
    for name, cfg in data.get("mcpServers", {}).items():
        out: dict[str, Any] = {}
        for key, val in cfg.items():
            if key == "defer_loading":
                continue
            if isinstance(val, str):
                out[key] = val.replace("${CLAUDE_PLUGIN_ROOT}", root)
            elif isinstance(val, list):
                out[key] = [
                    v.replace("${CLAUDE_PLUGIN_ROOT}", root) if isinstance(v, str) else v
                    for v in val
                ]
            else:
                out[key] = val
        if out:
            resolved[name] = out
    return resolved


def _build_mlflow_stop_hook(experiment: str, producer: str) -> Any:
    """Build a best-effort MLflow Stop hook that logs the run's transcript.

    Simplified from upstream: configured from ``experiment`` + ambient
    ``MLFLOW_*`` env vars (no ai-dev-kit ``SkillTestConfig``). Returns ``None``
    if ``mlflow.claude_code.tracing`` is unavailable, in which case the trace is
    still built from streamed events.
    """
    try:
        import mlflow
        from mlflow.claude_code.tracing import process_transcript, setup_mlflow
    except ImportError:
        logger.warning("mlflow.claude_code.tracing unavailable — run will not be logged to MLflow.")
        return None

    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
    os.environ["MLFLOW_CLAUDE_TRACING_ENABLED"] = "true"
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(experiment)
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow experiment '%s' not accessible: %s", experiment, e)
        return None

    async def _stop_hook(
        input_data: dict[str, Any], tool_use_id: Any, context: Any
    ) -> dict[str, bool]:
        session_id = input_data.get("session_id")
        transcript_path = input_data.get("transcript_path")

        async def _upload() -> None:
            try:
                setup_mlflow()
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, process_transcript, transcript_path, session_id),
                    timeout=60.0,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Background trace upload failed (session=%s): %s", session_id, e)

        asyncio.ensure_future(_upload())
        return {"continue": True}

    return _stop_hook


def _run_in_fresh_loop(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion on a dedicated thread + event loop.

    Harvested from upstream: isolating the loop in its own thread avoids anyio
    cancel-scope and subprocess transport cleanup errors when the SDK client
    tears down.
    """
    result_holder: dict[str, Any] = {}

    def _target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder["value"] = loop.run_until_complete(coro)
        except Exception as e:  # noqa: BLE001
            result_holder["error"] = e
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                try:
                    loop.run_until_complete(
                        asyncio.wait_for(loop.shutdown_default_executor(), timeout=5.0)
                    )
                except (TimeoutError, RuntimeError):
                    pass
            except Exception:  # noqa: BLE001
                pass
            # Silence "Event loop is closed" noise from subprocess transport
            # __del__ running during GC after the loop closes (harmless).
            setattr(loop, "_check_closed", lambda: None)  # noqa: B010
            loop.close()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_target)
        future.result(timeout=600)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        raise
    else:
        pool.shutdown(wait=True)

    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["value"]
