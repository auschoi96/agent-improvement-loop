"""Codex (OpenAI Codex CLI) onboarding for the ingest seam.

This module mirrors the Claude Code adapter
(:mod:`ail.ingest.adapters.claude_code`) for the **codex-native CLI harness**,
and it carries the two responsibilities that adapter does:

1. **Trace capture** (:func:`normalize_codex_rollout` / :class:`CodexAdapter`):
   read a Codex *rollout transcript* — the JSONL file Codex writes per session
   under ``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*-<session>.jsonl`` — and
   project it onto a :class:`~ail.ingest.base.NormalizedTrace`, the identical
   shape a :class:`~ail.ingest.base.TraceSource` yields. Captured traces are
   tagged ``ail.agent = codex`` (the :mod:`ail.cohorts` convention) so they are
   separable from ``claude_code`` traces in a shared experiment.

2. **Enabling MLflow's native Codex integration** (:func:`enable_codex_tracing`):
   Codex does **not** autolog into MLflow from Python. MLflow's native
   integration is the public Node ``@mlflow/codex`` package, which registers a
   Codex ``notify`` hook (``notify = ["mlflow-codex", "notify-hook"]`` in
   ``config.toml``) that reads the same rollout JSONL and logs ``codex_conversation``
   ``AGENT`` → ``llm_call`` / ``tool_*`` spans to MLflow. This helper writes that
   hook plus the ``mlflow-tracing.json`` config (``trackingUri`` / ``experimentId``)
   so live Codex sessions land in a target experiment — including Databricks-managed
   MLflow (``trackingUri = "databricks"``), which the stock ``mlflow-codex setup``
   refuses to write because its validator only accepts ``http(s)`` URLs.

The rollout schema is the tagged ``RolloutItem`` enum defined in Codex's
``codex-rs/protocol/src/protocol.rs``; this module reads the same fields the
``@mlflow/codex`` package reads (``response_item`` ``function_call`` /
``function_call_output`` pairs, ``event_msg`` ``token_count`` usage, model from
``session_meta`` / ``turn_context``).

The Codex CLI is an external dependency (a binary on ``PATH``, not a Python
package). If it is absent, :meth:`CodexAdapter.run` returns a failed
:class:`~ail.ingest.base.AgentRunResult` rather than raising — agent
unavailability is an ordinary failure under the
:class:`~ail.ingest.base.AgentAdapter` contract.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ail.cohorts import TAG_AGENT
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

__all__ = [
    "CODEX_AGENT",
    "CODEX_TRACE_NAME",
    "CodexAdapter",
    "CodexTracingError",
    "CodexTracingSetup",
    "codex_cohort_tags",
    "enable_codex_tracing",
    "normalize_codex_rollout",
    "read_rollout",
]

#: Stable producer/agent identifier recorded on captured traces
#: (:attr:`NormalizedTrace.producer`) and used as the ``ail.agent`` tag value.
CODEX_AGENT = "codex"

#: Root ``AGENT`` span name the native ``@mlflow/codex`` integration emits.
#: Recognizing this trace name is how a Codex trace read back from MLflow can be
#: attributed to Codex even before any ``ail.agent`` tag is applied.
CODEX_TRACE_NAME = "codex_conversation"

# Rollout JSONL record types (the outer ``type`` of each line).
_REC_SESSION_META = "session_meta"
_REC_TURN_CONTEXT = "turn_context"
_REC_RESPONSE_ITEM = "response_item"
_REC_EVENT_MSG = "event_msg"

# ``event_msg`` payload subtypes we read.
_EVT_TOKEN_COUNT = "token_count"
_EVT_EXEC_END = "exec_command_end"
# Stream/agent-level failures that mark the whole trace ERROR. A *tool* failure
# (a non-zero shell exit) marks only that tool's span — the agent often recovers
# — so it is intentionally not in this set.
_EVT_ERRORS = frozenset({"error", "stream_error", "turn_aborted", "turn_failed"})

# ``response_item`` payload subtypes (OpenAI Responses-API item shapes, as the
# ``@mlflow/codex`` package reads them).
_ITEM_MESSAGE = "message"
_ITEM_FUNCTION_CALL = "function_call"
_ITEM_FUNCTION_CALL_OUTPUT = "function_call_output"

# Native ``@mlflow/codex`` notify-hook wiring written into ``config.toml``.
_NOTIFY_HOOK = ["mlflow-codex", "notify-hook"]
_NOTIFY_LINE = f"notify = {json.dumps(_NOTIFY_HOOK)}"
_CONFIG_FILENAME = "config.toml"
_TRACING_CONFIG_FILENAME = "mlflow-tracing.json"
_CODEX_DIRNAME = ".codex"
_CONFIG_BANNER = "# Added by ail.enable_codex_tracing — forwards each Codex turn to MLflow Tracing."


# ---------------------------------------------------------------------------
# Rollout transcript -> NormalizedTrace (the Python-native trace-capture path)
# ---------------------------------------------------------------------------


def read_rollout(path: str | Path) -> list[dict[str, Any]]:
    """Read a Codex rollout JSONL file into a list of records.

    Blank lines and lines that fail to parse as JSON are skipped (a partially
    written tail line should not abort ingestion).
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def normalize_codex_rollout(
    records: Iterable[Mapping[str, Any]] | str | Path,
    *,
    experiment_id: str | None = None,
    trace_id: str | None = None,
) -> NormalizedTrace:
    """Project a Codex rollout transcript onto a :class:`NormalizedTrace`.

    Accepts either an already-parsed sequence of rollout records or a path to a
    rollout JSONL file. The result is tagged ``ail.agent = codex`` and carries
    ``producer = "codex"``, so it slots into the same cohorts / L0 metrics /
    judges that consume MLflow-read traces.

    Args:
        records: Parsed rollout records, or a path to the rollout JSONL file.
        experiment_id: Optional experiment id to stamp on the trace.
        trace_id: Optional explicit trace id; defaults to the session id, then a
            generated ``codex-<uuid>``.
    """
    if isinstance(records, (str, Path)):
        rollout = read_rollout(records)
    else:
        rollout = [dict(record) for record in records]

    session_id = _session_id(rollout)
    model = _model(rollout)
    tool_calls = _tool_calls(rollout)
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
        for call in tool_calls
    ]
    start_time, end_time = _time_bounds(rollout)
    request_preview, response_preview = _previews(rollout)

    metadata: dict[str, Any] = {}
    cli_version = _session_meta_field(rollout, "cli_version")
    if cli_version:
        metadata["codex.cli_version"] = cli_version
    provider = _session_meta_field(rollout, "model_provider")
    if provider:
        metadata["codex.model_provider"] = provider

    return NormalizedTrace(
        trace_id=trace_id or session_id or f"codex-{uuid.uuid4().hex}",
        status=_status(rollout),
        producer=CODEX_AGENT,
        model=model,
        session_id=session_id,
        experiment_id=experiment_id,
        request_time=start_time,
        end_time=end_time,
        execution_duration_ms=_duration_ms(start_time, end_time),
        token_usage=_token_usage(rollout),
        spans=spans,
        tool_calls=tool_calls,
        request_preview=request_preview,
        response_preview=response_preview,
        tags={TAG_AGENT: CODEX_AGENT},
        metadata=metadata,
    )


def codex_cohort_tags() -> dict[str, str]:
    """The cohort tag set identifying a Codex trace (``{"ail.agent": "codex"}``).

    Pass to :func:`ail.ingest.mlflow_source.apply_trace_tags` to backfill the
    tag onto Codex traces that the native ``@mlflow/codex`` integration logged
    (it has no custom-tag hook of its own).
    """
    return {TAG_AGENT: CODEX_AGENT}


# -- rollout field extraction (pure helpers) --------------------------------


def _payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _records_of(records: list[dict[str, Any]], record_type: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("type") == record_type]


def _session_meta_field(records: list[dict[str, Any]], key: str) -> str | None:
    for record in _records_of(records, _REC_SESSION_META):
        value = _payload(record).get(key)
        if value:
            return str(value)
    return None


def _session_id(records: list[dict[str, Any]]) -> str | None:
    return _session_meta_field(records, "id")


def _model(records: list[dict[str, Any]]) -> str | None:
    # ``turn_context`` carries the model actually used for the turn (honors an
    # in-session ``/model`` switch); fall back to ``session_meta``.
    model: str | None = None
    for record in _records_of(records, _REC_TURN_CONTEXT):
        candidate = _payload(record).get("model")
        if candidate:
            model = str(candidate)
    return model or _session_meta_field(records, "model")


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    """Decode a ``function_call.arguments`` JSON string into a dict."""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments:
        return {}
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return {"raw": arguments}
    return decoded if isinstance(decoded, dict) else {"input": decoded}


def _output_text(output: Any) -> str:
    """Render a ``function_call_output.output`` into text."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, str):
            return content
    return json.dumps(output, default=str) if output is not None else ""


def _tool_statuses(records: list[dict[str, Any]]) -> dict[str, bool]:
    """Map ``call_id`` -> failed, from ``exec_command_end`` events.

    Mirrors ``@mlflow/codex``'s ``buildToolStatuses``: Codex reports a failed
    shell command with ``status == "failed"`` (or a non-zero ``exit_code``).
    Non-exec tools emit no ``exec_command_end`` and so stay in their default
    state.
    """
    failed: dict[str, bool] = {}
    for record in _records_of(records, _REC_EVENT_MSG):
        payload = _payload(record)
        if payload.get("type") != _EVT_EXEC_END:
            continue
        call_id = payload.get("call_id")
        if not call_id:
            continue
        exit_code = payload.get("exit_code")
        is_failed = payload.get("status") == "failed" or (
            isinstance(exit_code, int) and exit_code != 0
        )
        failed[str(call_id)] = bool(is_failed)
    return failed


def _tool_calls(records: list[dict[str, Any]]) -> list[ToolCall]:
    """Pair ``function_call`` / ``function_call_output`` items into ToolCalls."""
    statuses = _tool_statuses(records)
    by_id: dict[str, ToolCall] = {}
    ordered: list[ToolCall] = []

    for record in records:
        if record.get("type") != _REC_RESPONSE_ITEM:
            continue
        payload = _payload(record)
        item_type = payload.get("type")
        if item_type == _ITEM_FUNCTION_CALL:
            call_id = str(payload.get("call_id") or payload.get("id") or uuid.uuid4().hex)
            call = ToolCall(
                id=call_id,
                name=str(payload.get("name") or "unknown"),
                arguments=_parse_arguments(payload.get("arguments")),
                status=TraceStatus.IN_PROGRESS,
                span_id=call_id,
                start_time=_record_time(record),
            )
            by_id[call_id] = call
            ordered.append(call)
        elif item_type == _ITEM_FUNCTION_CALL_OUTPUT:
            existing = by_id.get(str(payload.get("call_id") or ""))
            if existing is not None:
                existing.result = _output_text(payload.get("output"))
                existing.status = TraceStatus.OK

    for call_id, is_failed in statuses.items():
        existing = by_id.get(call_id)
        if existing is not None and is_failed:
            existing.status = TraceStatus.ERROR
    return ordered


def _token_usage(records: list[dict[str, Any]]) -> TokenUsage:
    """Build :class:`TokenUsage` from the last ``token_count`` event.

    Codex's cumulative ``total_token_usage`` reports ``input_tokens`` *inclusive*
    of ``cached_input_tokens`` (cache reads are a subset of input), unlike the
    Anthropic schema this repo's :class:`TokenUsage` mirrors, where cache reads
    are counted apart from fresh input. To keep the L0 cost metric from pricing
    the cached tokens twice (once as input, once as cache-read), the cached count
    is split out of ``input_tokens`` here. Codex's authoritative ``total_tokens``
    is preserved via the total override so the split does not change the total.
    """
    latest: dict[str, Any] | None = None
    for record in _records_of(records, _REC_EVENT_MSG):
        payload = _payload(record)
        if payload.get("type") != _EVT_TOKEN_COUNT:
            continue
        info = payload.get("info")
        if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
            latest = info["total_token_usage"]
    if latest is None:
        return TokenUsage()

    input_tokens = _as_int(latest.get("input_tokens"))
    cached = _as_int(latest.get("cached_input_tokens"))
    cache_read = min(cached, input_tokens)
    fresh_input = input_tokens - cache_read
    output_tokens = _as_int(latest.get("output_tokens"))
    reported_total = latest.get("total_tokens")
    total_override = _as_int(reported_total) if reported_total is not None else None

    return TokenUsage(
        input_tokens=fresh_input,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cache_read,
        _total_tokens=total_override,
    )


def _status(records: list[dict[str, Any]]) -> TraceStatus:
    for record in _records_of(records, _REC_EVENT_MSG):
        if _payload(record).get("type") in _EVT_ERRORS:
            return TraceStatus.ERROR
    return TraceStatus.OK


def _message_text(payload: Mapping[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("text")
    ]
    return "".join(parts)


def _previews(records: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """First user prompt and last assistant message, for trace previews."""
    request: str | None = None
    response: str | None = None
    for record in records:
        if record.get("type") != _REC_RESPONSE_ITEM:
            continue
        payload = _payload(record)
        if payload.get("type") != _ITEM_MESSAGE:
            continue
        role = payload.get("role")
        text = _message_text(payload)
        if not text:
            continue
        if role == "user" and request is None:
            request = text
        elif role == "assistant":
            response = text
    return request, response


def _record_time(record: Mapping[str, Any]) -> datetime | None:
    return _iso_to_dt(record.get("timestamp"))


def _time_bounds(records: list[dict[str, Any]]) -> tuple[datetime | None, datetime | None]:
    times = [t for t in (_record_time(r) for r in records) if t is not None]
    if not times:
        return None, None
    return times[0], times[-1]


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if start and end:
        return int((end - start).total_seconds() * 1000)
    return None


def _iso_to_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Enabling MLflow's native Codex integration (@mlflow/codex notify hook)
# ---------------------------------------------------------------------------


class CodexTracingError(RuntimeError):
    """Raised when Codex tracing cannot be configured without clobbering config."""


@dataclass(frozen=True, slots=True)
class CodexTracingSetup:
    """Result of :func:`enable_codex_tracing`.

    ``config_path`` / ``tracing_config_path`` are the files written;
    ``notify_added`` is whether the ``notify`` line was newly written (``False``
    when it was already present); ``scope`` is ``"user"`` or ``"project"``.
    """

    config_path: Path
    tracing_config_path: Path
    notify_added: bool
    scope: str


def enable_codex_tracing(
    experiment_id: str,
    *,
    tracking_uri: str = "databricks",
    scope: str = "user",
    codex_dir: str | Path | None = None,
    force: bool = False,
) -> CodexTracingSetup:
    """Configure MLflow's native Codex (``@mlflow/codex``) tracing integration.

    Writes, under the resolved ``.codex`` directory:

    * ``config.toml`` — prepends ``notify = ["mlflow-codex", "notify-hook"]`` so
      Codex forwards each turn to the hook. Top-level ``notify`` is prepended
      (not appended) so it stays above any ``[table]`` headers, as TOML requires.
      If a ``notify`` entry already exists it is **not** clobbered: an existing
      ``mlflow-codex`` entry is left as-is (idempotent); a *different* ``notify``
      raises :class:`CodexTracingError` unless ``force=True``.
    * ``mlflow-tracing.json`` — ``{"trackingUri": ..., "experimentId": ...}``,
      the config the hook reads (env vars > project json > user json). Written
      directly so ``trackingUri = "databricks"`` is accepted; the stock
      ``mlflow-codex setup`` rejects non-``http(s)`` URIs.

    The ``@mlflow/codex`` npm package must be installed for the hook to run at
    Codex turn-completion (``npm install -g @mlflow/codex``); this helper only
    writes the configuration.

    Args:
        experiment_id: Target MLflow experiment id (e.g. ``"660599403165942"``).
        tracking_uri: MLflow tracking URI. ``"databricks"`` targets
            Databricks-managed MLflow.
        scope: ``"user"`` writes ``~/.codex`` (or ``$CODEX_HOME``); ``"project"``
            writes ``./.codex`` in the current directory.
        codex_dir: Explicit ``.codex`` directory, overriding ``scope`` resolution
            (used by tests and non-standard layouts).
        force: Overwrite an existing non-mlflow ``notify`` entry instead of raising.

    Returns:
        A :class:`CodexTracingSetup` describing what was written.
    """
    if scope not in ("user", "project"):
        raise ValueError(f"scope must be 'user' or 'project', got {scope!r}")

    directory = _resolve_codex_dir(scope, codex_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / _CONFIG_FILENAME
    tracing_config_path = directory / _TRACING_CONFIG_FILENAME

    notify_added = _ensure_notify_hook(config_path, force=force)
    tracing_config_path.write_text(
        json.dumps({"trackingUri": tracking_uri, "experimentId": str(experiment_id)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return CodexTracingSetup(
        config_path=config_path,
        tracing_config_path=tracing_config_path,
        notify_added=notify_added,
        scope=scope,
    )


def _resolve_codex_dir(scope: str, codex_dir: str | Path | None) -> Path:
    if codex_dir is not None:
        return Path(codex_dir)
    if scope == "project":
        return Path.cwd() / _CODEX_DIRNAME
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home)
    return Path.home() / _CODEX_DIRNAME


def _ensure_notify_hook(config_path: Path, *, force: bool) -> bool:
    """Write the ``notify`` line into ``config.toml``; return whether it was added.

    Returns ``False`` when an ``mlflow-codex`` notify hook is already present.
    """
    if not config_path.exists():
        config_path.write_text(f"{_CONFIG_BANNER}\n{_NOTIFY_LINE}\n", encoding="utf-8")
        return True

    content = config_path.read_text(encoding="utf-8")
    existing = _existing_notify_line(content)
    if existing is not None:
        if "mlflow-codex" in existing:
            return False
        if not force:
            raise CodexTracingError(
                f"{config_path} already has a `notify = ...` entry "
                f"({existing.strip()!r}); update it to {_NOTIFY_LINE!r} manually, "
                "or pass force=True to overwrite it."
            )
        replaced = "\n".join(
            _NOTIFY_LINE if _is_notify_assignment(line) else line for line in content.splitlines()
        )
        trailing = "\n" if content.endswith("\n") else ""
        config_path.write_text(replaced + trailing, encoding="utf-8")
        return True

    # Prepend so the top-level key stays above any [table] header.
    config_path.write_text(f"{_CONFIG_BANNER}\n{_NOTIFY_LINE}\n\n{content}", encoding="utf-8")
    return True


def _is_notify_assignment(line: str) -> bool:
    """Whether ``line`` is a top-level ``notify = ...`` TOML assignment."""
    stripped = line.strip()
    if not stripped.startswith("notify"):
        return False
    return stripped[len("notify") :].lstrip().startswith("=")


def _existing_notify_line(content: str) -> str | None:
    for line in content.splitlines():
        if _is_notify_assignment(line):
            return line
    return None


# ---------------------------------------------------------------------------
# AgentAdapter: drive the Codex CLI and capture its rollout as a trace
# ---------------------------------------------------------------------------


class CodexAdapter(AgentAdapter):
    """Run the Codex CLI against a task and capture its rollout as a trace.

    Drives ``codex exec`` non-interactively, then reads the rollout transcript
    Codex writes for that session and normalizes it via
    :func:`normalize_codex_rollout` — yielding a
    :class:`~ail.ingest.base.NormalizedTrace` tagged ``ail.agent = codex``, the
    same shape a :class:`~ail.ingest.base.TraceSource` yields, so a frozen task
    suite replayed through this adapter is comparable apples-to-apples with
    traces read back from MLflow.

    Args:
        command: Codex executable name or path. Defaults to ``"codex"``.
        sandbox: ``codex exec --sandbox`` policy. Defaults to ``"workspace-write"``.
        codex_home: Explicit ``CODEX_HOME`` whose ``sessions/`` holds rollouts.
            Defaults to ``$CODEX_HOME`` then ``~/.codex``.
        extra_args: Additional raw args inserted before the prompt.
    """

    name = CODEX_AGENT

    def __init__(
        self,
        command: str = "codex",
        *,
        sandbox: str = "workspace-write",
        codex_home: str | Path | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.command = command
        self.sandbox = sandbox
        self.codex_home = Path(codex_home) if codex_home is not None else None
        self.extra_args = list(extra_args) if extra_args is not None else []

    def run(self, task: AgentTask) -> AgentRunResult:
        """Execute ``task`` via ``codex exec`` and capture its rollout trace."""
        binary = shutil.which(self.command) or (
            self.command if os.path.isfile(self.command) else None
        )
        if binary is None:
            return self._cli_unavailable_result()

        sessions_dir = self._sessions_dir()
        since = time.time()
        last_message_path = Path(tempfile.gettempdir()) / f"codex-last-{uuid.uuid4().hex}.txt"
        args = self._build_args(task, last_message_path)

        try:
            completed = subprocess.run(  # noqa: S603 - args are constructed, not shell
                [binary, *args],
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                cwd=task.cwd or None,
            )
        except subprocess.TimeoutExpired:
            return self._failed_result(f"codex exec exceeded {task.timeout_seconds}s")
        except OSError as exc:
            return self._failed_result(f"codex exec failed to start: {exc}")

        output_text = _read_text(last_message_path) or (completed.stdout or "").strip()
        rollout_path = _latest_rollout(sessions_dir, since)
        if rollout_path is None:
            return self._failed_result(
                "codex exec produced no rollout transcript "
                f"(searched {sessions_dir}); stderr: {(completed.stderr or '').strip()[:500]}",
                output_text=output_text,
            )

        trace = normalize_codex_rollout(rollout_path)
        success = completed.returncode == 0 and trace.status is not TraceStatus.ERROR
        return AgentRunResult(
            trace=trace,
            output_text=output_text,
            success=success,
            error=None if success else (completed.stderr or "").strip()[:500] or "codex run failed",
            session_id=trace.session_id,
            duration_ms=trace.execution_duration_ms,
            raw=str(rollout_path),
        )

    def _build_args(self, task: AgentTask, last_message_path: Path) -> list[str]:
        args = ["exec", "--skip-git-repo-check", "--sandbox", self.sandbox]
        args += ["--output-last-message", str(last_message_path)]
        if task.model:
            args += ["--model", task.model]
        if task.cwd:
            args += ["--cd", task.cwd]
        args += self.extra_args
        args.append(task.prompt)
        return args

    def _sessions_dir(self) -> Path:
        home = self.codex_home
        if home is None:
            env_home = os.environ.get("CODEX_HOME")
            home = Path(env_home) if env_home else Path.home() / _CODEX_DIRNAME
        return home / "sessions"

    def _cli_unavailable_result(self) -> AgentRunResult:
        return self._failed_result(
            f"Codex CLI ({self.command!r}) is not on PATH. "
            "Install it (e.g. `npm install -g @openai/codex`) or pass command=<path>."
        )

    def _failed_result(self, message: str, *, output_text: str = "") -> AgentRunResult:
        logger.error("Codex adapter run failed: %s", message)
        return AgentRunResult(
            trace=NormalizedTrace(
                trace_id=f"error-{uuid.uuid4().hex}",
                status=TraceStatus.ERROR,
                producer=CODEX_AGENT,
                tags={TAG_AGENT: CODEX_AGENT},
            ),
            output_text=output_text,
            success=False,
            error=message,
        )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _latest_rollout(sessions_dir: Path, since: float) -> Path | None:
    """Newest ``rollout-*.jsonl`` under ``sessions_dir`` modified at/after ``since``.

    Codex writes one rollout per session under ``sessions/YYYY/MM/DD/``; the
    adapter runs one session at a time, so the newest rollout touched after the
    run started is this run's transcript.
    """
    if not sessions_dir.is_dir():
        return None
    candidates = [
        path
        for path in sessions_dir.rglob("rollout-*.jsonl")
        if path.stat().st_mtime >= since - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
