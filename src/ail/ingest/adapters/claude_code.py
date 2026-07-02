"""Claude Code :class:`~ail.ingest.base.AgentAdapter`.

Drives the Claude Agent SDK (public ``claude-agent-sdk`` package) against an
:class:`~ail.ingest.base.AgentTask`, captures the agent's streamed events, and
projects them onto a :class:`~ail.ingest.base.NormalizedTrace` — the identical
shape an :class:`~ail.ingest.base.TraceSource` yields — so a frozen task suite
replayed through this adapter is comparable apples-to-apples with traces read
back from MLflow.

When an MLflow experiment is supplied, a Claude Code ``Stop`` hook logs the full
conversation transcript to Databricks-managed MLflow using the public
``mlflow.claude_code.tracing`` helpers (``setup_mlflow`` + ``process_transcript``).

The Claude Agent SDK is an optional dependency, imported lazily. If it is
absent, :meth:`ClaudeCodeAdapter.run` returns a failed
:class:`~ail.ingest.base.AgentRunResult` rather than raising — agent unavailability
is an ordinary failure under the :class:`~ail.ingest.base.AgentAdapter` contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, fields, is_dataclass
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

# Token counters carried on every assistant turn — the public Anthropic
# Messages API ``usage`` schema.
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# Env var families the spawned agent needs to authenticate (model + workspace +
# tracing). Forwarded from the parent process into the child.
_FORWARDED_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_", "DATABRICKS_", "MLFLOW_")
# Internal Claude Code session vars that would confuse a freshly spawned child.
_INTERNAL_ENV_KEYS = frozenset(
    {
        "CLAUDE_CODE_SSE_PORT",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY",
    }
)
_DEFAULT_ALLOWED_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")
_FILESYSTEM_SANDBOX_PARAM = "claude_code_filesystem_sandbox"

# Grace period added on top of the task's own timeout before the worker thread
# is abandoned. The in-stream timeout (``task.timeout_seconds``) handles the
# normal case; this hard ceiling only fires if the SDK wedges so badly that the
# in-stream check never runs, and it yields a failed result rather than raising.
_HARD_TIMEOUT_BUFFER_S = 60

# Public MLflow Claude-tracing env var names (read by ``setup_mlflow``).
_ENV_TRACING_ENABLED = "MLFLOW_CLAUDE_TRACING_ENABLED"
_ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
_ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"

# Agent-settings file (optional): supplies extra env (e.g. FMAPI routing) with
# ``${VAR}`` / ``${VAR:-default}`` interpolation under an ``{"env": {...}}`` key.
_AGENT_SETTINGS_FILES = (
    Path(".ail") / "agent_settings.json",
    Path(".claude") / "agent_settings.json",
)
_PLUGIN_ROOT_TOKEN = "${CLAUDE_PLUGIN_ROOT}"


@dataclass(slots=True)
class AgentEvent:
    """One observation captured from the agent's execution stream.

    ``type`` is one of ``assistant_turn``, ``text``, ``tool_use``,
    ``tool_result``, ``result``, ``system`` or ``error``; ``data`` carries the
    type-specific payload.
    """

    type: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)


class ClaudeCodeAdapter(AgentAdapter):
    """Run Claude Code through the Claude Agent SDK and capture a normalized trace.

    Args:
        mlflow_experiment: Optional MLflow experiment name. When set, a ``Stop``
            hook logs the run's transcript to Databricks-managed MLflow
            (best-effort, fire-and-forget). When ``None``, the trace is built
            solely from the captured event stream.
        default_allowed_tools: Tools allowed when a task specifies none and no
            MCP servers are configured.
    """

    name = "claude_code"

    def __init__(
        self,
        mlflow_experiment: str | None = None,
        default_allowed_tools: list[str] | None = None,
    ) -> None:
        self.mlflow_experiment = mlflow_experiment
        self.default_allowed_tools = (
            list(default_allowed_tools)
            if default_allowed_tools is not None
            else list(_DEFAULT_ALLOWED_TOOLS)
        )

    # -- AgentAdapter interface -------------------------------------------

    def run(self, task: AgentTask) -> AgentRunResult:
        """Execute ``task`` synchronously and return its captured trace.

        The async SDK session is driven to completion on a dedicated event loop
        (see :func:`_run_async`), keeping ``run`` synchronous per the
        :class:`~ail.ingest.base.AgentAdapter` contract. If the session blows
        past its hard timeout, a failed :class:`~ail.ingest.base.AgentRunResult`
        is returned rather than an exception raised.
        """
        hard_timeout = task.timeout_seconds + _HARD_TIMEOUT_BUFFER_S
        return _run_async(
            self._arun(task),
            timeout=hard_timeout,
            on_timeout=lambda: self._timeout_result(hard_timeout),
        )

    def _timeout_result(self, timeout: float) -> AgentRunResult:
        """Failed result for a run that exceeded its hard timeout."""
        message = f"Claude agent run exceeded its hard timeout of {timeout:g}s"
        logger.error(message)
        return AgentRunResult(
            trace=NormalizedTrace(
                trace_id=f"timeout-{uuid.uuid4().hex}",
                status=TraceStatus.ERROR,
                producer=self.name,
            ),
            success=False,
            error=message,
        )

    async def _arun(self, task: AgentTask) -> AgentRunResult:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher
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
            return self._sdk_unavailable_result()

        events: list[AgentEvent] = []
        response_text: list[str] = []
        session_id = str(uuid.uuid4())

        mcp_servers = _load_mcp_servers()
        allowed_tools = task.allowed_tools
        if allowed_tools is None and not mcp_servers:
            allowed_tools = list(self.default_allowed_tools)
        try:
            filesystem_sandbox_dir = _required_filesystem_sandbox_dir(task)
        except ValueError as exc:
            return self._sdk_sandbox_unavailable_result(str(exc))
        if filesystem_sandbox_dir is not None:
            reason = _sdk_filesystem_sandbox_unavailable(ClaudeAgentOptions)
            if reason is not None:
                return self._sdk_sandbox_unavailable_result(reason)
            task_cwd = os.path.realpath(task.cwd or os.getcwd())
            if task_cwd != filesystem_sandbox_dir:
                return self._sdk_sandbox_unavailable_result(
                    "preview requested Claude SDK filesystem sandboxing for "
                    f"{filesystem_sandbox_dir!r}, but the task cwd resolves to {task_cwd!r}"
                )
            allowed_tools = _scope_write_tools_to_sandbox(allowed_tools or [], filesystem_sandbox_dir)

        env = _agent_env()
        if task.model:
            env["ANTHROPIC_MODEL"] = task.model
        env.pop("CLAUDECODE", None)  # the child must not believe it is nested
        _attach_databricks_env(mcp_servers, env)

        hooks = self._build_hooks(HookMatcher)
        stderr_tail: list[str] = []

        options_kwargs: dict[str, Any] = {
            "cwd": task.cwd or os.getcwd(),
            "allowed_tools": allowed_tools,
            "permission_mode": (
                "dontAsk" if filesystem_sandbox_dir is not None else "bypassPermissions"
            ),
            "mcp_servers": mcp_servers or {},
            "system_prompt": task.system_prompt or "",
            "setting_sources": [],  # no ambient project skills; context is injected explicitly
            "env": env,
            "hooks": hooks or None,
            "stderr": lambda line: _capture_stderr(stderr_tail, line),
        }
        if filesystem_sandbox_dir is not None:
            options_kwargs["sandbox"] = _claude_sdk_sandbox_settings(filesystem_sandbox_dir)
        options = ClaudeAgentOptions(**options_kwargs)

        started = time.monotonic()
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(task.prompt)
                async for message in client.receive_response():
                    if time.monotonic() - started > task.timeout_seconds:
                        events.append(
                            AgentEvent(
                                "error",
                                datetime.now(UTC),
                                {"message": f"Timeout after {task.timeout_seconds}s"},
                            )
                        )
                        break

                    if isinstance(message, AssistantMessage):
                        now = datetime.now(UTC)
                        events.append(
                            AgentEvent(
                                "assistant_turn",
                                now,
                                {"usage": _usage_payload(getattr(message, "usage", None))},
                            )
                        )
                        for block in getattr(message, "content", None) or []:
                            if isinstance(block, TextBlock):
                                response_text.append(block.text)
                                events.append(AgentEvent("text", now, {"text": block.text}))
                            elif isinstance(block, ToolUseBlock):
                                tool_input = block.input if isinstance(block.input, dict) else {}
                                events.append(
                                    AgentEvent(
                                        "tool_use",
                                        now,
                                        {"id": block.id, "name": block.name, "input": tool_input},
                                    )
                                )
                            elif isinstance(block, ToolResultBlock):
                                events.append(AgentEvent("tool_result", now, _tool_result(block)))
                    elif isinstance(message, UserMessage):
                        now = datetime.now(UTC)
                        for block in getattr(message, "content", None) or []:
                            if isinstance(block, ToolResultBlock):
                                events.append(AgentEvent("tool_result", now, _tool_result(block)))
                    elif isinstance(message, ResultMessage):
                        session_id = getattr(message, "session_id", None) or session_id
                        events.append(
                            AgentEvent(
                                "result",
                                datetime.now(UTC),
                                {
                                    "session_id": session_id,
                                    "duration_ms": getattr(message, "duration_ms", None),
                                },
                            )
                        )
                    elif isinstance(message, SystemMessage):
                        events.append(
                            AgentEvent(
                                "system",
                                datetime.now(UTC),
                                {
                                    "subtype": getattr(message, "subtype", ""),
                                    "data": getattr(message, "data", {}),
                                },
                            )
                        )
        except Exception as exc:  # noqa: BLE001 - capture per AgentAdapter contract
            tail = "; ".join(stderr_tail[-5:]) if stderr_tail else "no stderr"
            logger.error("Claude agent run failed: %s | stderr: %s", exc, tail)
            events.append(
                AgentEvent("error", datetime.now(UTC), {"message": f"{exc} | stderr: {tail}"})
            )

        wall_ms = int((time.monotonic() - started) * 1000)
        model = task.model or os.environ.get("ANTHROPIC_MODEL")
        trace = self._build_normalized_trace(events, session_id, model)
        first_error = next((e.data.get("message") for e in events if e.type == "error"), None)

        return AgentRunResult(
            trace=trace,
            output_text="\n".join(response_text),
            success=first_error is None,
            error=first_error,
            session_id=session_id,
            duration_ms=wall_ms,
            raw=events,
        )

    def _build_hooks(self, hook_matcher_cls: Any) -> dict[str, Any]:
        if not self.mlflow_experiment:
            return {}
        hook = _mlflow_stop_hook(self.mlflow_experiment)
        if hook is None:
            return {}
        return {"Stop": [hook_matcher_cls(hooks=[hook])]}

    def _sdk_unavailable_result(self) -> AgentRunResult:
        return AgentRunResult(
            trace=NormalizedTrace(
                trace_id=f"error-{uuid.uuid4().hex}",
                status=TraceStatus.ERROR,
                producer=self.name,
            ),
            success=False,
            error=(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install 'claude-agent-sdk>=0.1.39'"
            ),
        )

    def _sdk_sandbox_unavailable_result(self, reason: str) -> AgentRunResult:
        return AgentRunResult(
            trace=NormalizedTrace(
                trace_id=f"error-{uuid.uuid4().hex}",
                status=TraceStatus.ERROR,
                producer=self.name,
            ),
            success=False,
            error=(
                "Claude Agent SDK filesystem sandboxing is required for this preview, "
                f"but it cannot be enabled ({reason}); refusing to run unsandboxed."
            ),
        )

    def _build_normalized_trace(
        self,
        events: list[AgentEvent],
        session_id: str,
        model: str | None,
    ) -> NormalizedTrace:
        """Fold captured events into a :class:`NormalizedTrace`.

        ``tool_use``/``tool_result`` events are paired by tool-use id into
        :class:`ToolCall` records (each also surfaced as a ``TOOL``
        :class:`NormalizedSpan`); per-turn ``usage`` is summed into the trace
        :class:`TokenUsage`; and the trace inherits ``ERROR`` status if any
        ``error`` event was captured.
        """
        tools_by_id: dict[str, ToolCall] = {}
        ordered_tools: list[ToolCall] = []
        usage = TokenUsage()
        start_time: datetime | None = None
        end_time: datetime | None = None

        for event in events:
            start_time = start_time or event.timestamp
            end_time = event.timestamp

            if event.type == "tool_use":
                call = ToolCall(
                    id=str(event.data.get("id") or uuid.uuid4().hex),
                    name=str(event.data.get("name") or "unknown"),
                    arguments=dict(event.data.get("input") or {}),
                    status=TraceStatus.IN_PROGRESS,
                    span_id=str(event.data.get("id") or ""),
                    start_time=event.timestamp,
                )
                tools_by_id[call.id] = call
                ordered_tools.append(call)
            elif event.type == "tool_result":
                self._apply_tool_result(tools_by_id, event)
            elif event.type == "assistant_turn":
                usage = usage + _usage_from_payload(event.data.get("usage"))

        spans = [
            NormalizedSpan(
                span_id=call.span_id or call.id,
                name=f"tool_{call.name}",
                kind=SpanKind.TOOL,
                status=call.status,
                start_time=call.start_time,
                inputs=call.arguments,
                outputs=call.result,
            )
            for call in ordered_tools
        ]

        has_error = any(event.type == "error" for event in events)
        return NormalizedTrace(
            trace_id=session_id,
            status=TraceStatus.ERROR if has_error else TraceStatus.OK,
            producer=self.name,
            model=model,
            session_id=session_id,
            request_time=start_time,
            end_time=end_time,
            execution_duration_ms=_trace_duration_ms(events, start_time, end_time),
            token_usage=usage,
            spans=spans,
            tool_calls=ordered_tools,
        )

    @staticmethod
    def _apply_tool_result(tools_by_id: dict[str, ToolCall], event: AgentEvent) -> None:
        call = tools_by_id.get(str(event.data.get("tool_use_id") or ""))
        if call is None:
            return
        content = event.data.get("content", "")
        call.result = content if isinstance(content, str) else str(content)
        call.status = TraceStatus.ERROR if event.data.get("is_error") else TraceStatus.OK


# ---------------------------------------------------------------------------
# Event extraction helpers
# ---------------------------------------------------------------------------


def _tool_result(block: Any) -> dict[str, Any]:
    """Extract the fields we keep from a public ``ToolResultBlock``."""
    return {
        "tool_use_id": getattr(block, "tool_use_id", ""),
        "content": getattr(block, "content", ""),
        "is_error": bool(getattr(block, "is_error", False)),
    }


def _usage_payload(usage: Any) -> dict[str, int]:
    """Read the public Anthropic ``usage`` dict off an assistant message.

    The SDK exposes ``AssistantMessage.usage`` as a plain mapping, so the
    counters are read by key (not attribute). Missing keys default to zero.
    """
    if not isinstance(usage, dict):
        return {}
    return {key: _as_int(usage.get(key)) for key in _USAGE_KEYS}


def _usage_from_payload(payload: Any) -> TokenUsage:
    data = payload if isinstance(payload, dict) else {}
    return TokenUsage(
        input_tokens=_as_int(data.get("input_tokens")),
        output_tokens=_as_int(data.get("output_tokens")),
        cache_creation_input_tokens=_as_int(data.get("cache_creation_input_tokens")),
        cache_read_input_tokens=_as_int(data.get("cache_read_input_tokens")),
    )


def _trace_duration_ms(
    events: list[AgentEvent], start_time: datetime | None, end_time: datetime | None
) -> int | None:
    """Trace duration: the SDK-reported result duration if present, else wall time."""
    reported = next(
        (e.data.get("duration_ms") for e in events if e.type == "result"),
        None,
    )
    if isinstance(reported, (int, float)):
        return int(reported)
    if start_time and end_time:
        return int((end_time - start_time).total_seconds() * 1000)
    return None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _capture_stderr(buffer: list[str], line: str) -> None:
    stripped = line.strip()
    if stripped:
        buffer.append(stripped)


def _required_filesystem_sandbox_dir(task: AgentTask) -> str | None:
    """Return the required Claude SDK filesystem sandbox dir for preview tasks."""
    raw = task.params.get(_FILESYSTEM_SANDBOX_PARAM)
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("required") is not True:
        raise ValueError(
            f"{_FILESYSTEM_SANDBOX_PARAM} must be a dict with required=True"
        )
    sandbox_dir = raw.get("sandbox_dir")
    if not isinstance(sandbox_dir, str) or not sandbox_dir.strip():
        raise ValueError(f"{_FILESYSTEM_SANDBOX_PARAM}.sandbox_dir is required")
    return os.path.realpath(sandbox_dir)


def _sdk_filesystem_sandbox_unavailable(options_cls: Any) -> str | None:
    """Validate that the installed SDK exposes the native sandbox option we require."""
    if not is_dataclass(options_cls):
        return "ClaudeAgentOptions is not a dataclass with inspectable fields"
    option_fields = {field.name for field in fields(options_cls)}
    missing = {"sandbox", "permission_mode", "allowed_tools", "cwd"} - option_fields
    if missing:
        return "ClaudeAgentOptions is missing " + ", ".join(sorted(missing))
    return None


def _claude_sdk_sandbox_settings(sandbox_dir: str | None) -> dict[str, Any] | None:
    if sandbox_dir is None:
        return None
    return {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
    }


def _scope_write_tools_to_sandbox(allowed_tools: list[str], sandbox_dir: str) -> list[str]:
    scoped: list[str] = []
    for tool in allowed_tools:
        name = tool.split("(", 1)[0]
        if name in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
            rule = f"{name}({sandbox_dir}/**)"
        elif name == "Bash":
            rule = "Bash(*)"
        else:
            rule = tool
        if rule not in scoped:
            scoped.append(rule)
    return scoped


# ---------------------------------------------------------------------------
# Subprocess environment + MCP configuration (public Claude Code conventions)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Nearest ancestor directory containing ``.git``; cwd if none is found."""
    here = Path(__file__).resolve().parent
    for directory in (here, *here.parents):
        if (directory / ".git").exists():
            return directory
    return Path.cwd()


def _expand_env_refs(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references against ``os.environ``."""

    def _sub(match: re.Match[str]) -> str:
        ref = match.group(1)
        if ":-" in ref:
            name, default = ref.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(ref, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", _sub, value)


def _agent_env() -> dict[str, str]:
    """Assemble the environment for the spawned agent.

    Starts from an optional agent-settings file (FMAPI routing, etc.), then
    overlays every parent env var whose name matches a forwarded prefix,
    dropping internal Claude Code session vars. A couple of stream-stability
    defaults are added last.
    """
    env: dict[str, str] = {}

    root = _repo_root()
    for relative in _AGENT_SETTINGS_FILES:
        path = root / relative
        if not path.exists():
            continue
        try:
            settings_env = json.loads(path.read_text()).get("env", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read agent settings %s: %s", path, exc)
            break
        for key, value in settings_env.items():
            if isinstance(value, str):
                env[key] = _expand_env_refs(value)
        break

    for key, value in os.environ.items():
        if key in _INTERNAL_ENV_KEYS or not value:
            continue
        if any(key.startswith(prefix) for prefix in _FORWARDED_ENV_PREFIXES):
            env[key] = value

    for key in _INTERNAL_ENV_KEYS:
        env.pop(key, None)

    env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    env.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "600000")
    return env


def _attach_databricks_env(mcp_servers: dict[str, Any], env: dict[str, str]) -> None:
    """Give MCP servers that declare no ``env`` the Databricks credentials."""
    databricks_env = {k: v for k, v in env.items() if k.startswith("DATABRICKS_")}
    if not databricks_env:
        return
    for server in mcp_servers.values():
        if isinstance(server, dict) and "env" not in server:
            server["env"] = dict(databricks_env)


def _load_mcp_servers() -> dict[str, Any]:
    """Load MCP server definitions from ``.mcp.json`` at the repo root, if present.

    Reads the public Claude Code ``{"mcpServers": {...}}`` format and resolves
    the ``${CLAUDE_PLUGIN_ROOT}`` placeholder to the repo root.
    """
    config_path = _repo_root() / ".mcp.json"
    if not config_path.exists():
        return {}
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    root = str(_repo_root())
    servers: dict[str, Any] = {}
    for name, definition in config.get("mcpServers", {}).items():
        resolved = {
            key: _resolve_plugin_root(value, root)
            for key, value in definition.items()
            if key != "defer_loading"
        }
        if resolved:
            servers[name] = resolved
    return servers


def _resolve_plugin_root(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace(_PLUGIN_ROOT_TOKEN, root)
    if isinstance(value, list):
        return [v.replace(_PLUGIN_ROOT_TOKEN, root) if isinstance(v, str) else v for v in value]
    return value


# ---------------------------------------------------------------------------
# MLflow Stop hook (Databricks-managed MLflow via public mlflow.claude_code)
# ---------------------------------------------------------------------------


def _mlflow_stop_hook(experiment: str) -> Any:
    """Build a best-effort ``Stop`` hook that logs the transcript to MLflow.

    Configures the public ``mlflow.claude_code.tracing`` env contract
    (tracing-enabled flag + Databricks tracking URI + experiment), then returns
    an async hook that runs ``setup_mlflow`` and ``process_transcript`` off the
    event loop. Returns ``None`` if the tracing helpers are unavailable or the
    experiment cannot be reached — in which case the trace is still built from
    the captured event stream.
    """
    try:
        import mlflow
        from mlflow.claude_code.tracing import process_transcript, setup_mlflow
    except ImportError:
        logger.warning("mlflow.claude_code.tracing unavailable; the run will not be logged.")
        return None

    os.environ[_ENV_TRACING_ENABLED] = "true"
    os.environ.setdefault(_ENV_TRACKING_URI, "databricks")
    os.environ[_ENV_EXPERIMENT_NAME] = experiment
    try:
        mlflow.set_tracking_uri(os.environ[_ENV_TRACKING_URI])
        mlflow.set_experiment(experiment)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow experiment %r not accessible: %s", experiment, exc)
        return None

    async def _stop_hook(
        input_data: dict[str, Any], tool_use_id: Any, context: Any
    ) -> dict[str, Any]:
        session_id = input_data.get("session_id")
        transcript_path = input_data.get("transcript_path")

        async def _log_transcript() -> None:
            try:
                setup_mlflow()
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, process_transcript, transcript_path, session_id),
                    timeout=60.0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transcript logging failed (session=%s): %s", session_id, exc)

        asyncio.ensure_future(_log_transcript())
        return {}

    return _stop_hook


# ---------------------------------------------------------------------------
# Synchronous bridge to the async SDK session
# ---------------------------------------------------------------------------


def _run_async(
    coro: Coroutine[Any, Any, _T], *, timeout: float, on_timeout: Callable[[], _T]
) -> _T:
    """Drive ``coro`` to completion on a fresh event loop in a worker thread.

    The Claude Agent SDK spawns and supervises a subprocess over anyio. Running
    the whole session on its own loop in a dedicated thread keeps that lifecycle
    self-contained, so subprocess/transport teardown cannot race the caller's
    loop (the source of spurious "Event loop is closed" / cancel-scope noise),
    and lets the public :meth:`ClaudeCodeAdapter.run` stay synchronous.

    If the worker is still running after ``timeout`` seconds, the call does not
    raise: it abandons the (daemon) worker and returns ``on_timeout()`` so a
    wedged session surfaces as a failed result rather than a propagating
    exception.
    """
    outcome: dict[str, Any] = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            outcome["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001 - re-raised on the caller thread
            outcome["error"] = exc
        finally:
            _shutdown_loop(loop)

    worker = threading.Thread(target=_worker, name="claude-code-adapter", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return on_timeout()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel stragglers and close ``loop`` quietly."""
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:  # noqa: BLE001 - teardown is best-effort
        pass
    finally:
        # Subprocess transports may call back into the loop from __del__ during
        # GC after close; neutralizing the closed-check silences that harmless noise.
        setattr(loop, "_check_closed", lambda: None)  # noqa: B010
        loop.close()
