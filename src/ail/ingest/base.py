"""Reusability seam: producer-agnostic trace ingestion + agent execution.

This module is the spine of the "works for any agent" promise. It defines a
**common, normalized trace record** plus two abstractions:

* :class:`TraceSource` — pull traces out of a tracking backend and normalize
  each one into a :class:`NormalizedTrace`, *regardless of which agent
  produced it*. The concrete MLflow implementation lives in
  :mod:`ail.ingest.mlflow_source`.
* :class:`AgentAdapter` — run an agent against a task input and capture its
  execution as a :class:`NormalizedTrace`. Shipped adapters live in
  :mod:`ail.ingest.adapters` (Claude Code today; Codex next).

Design rules for this file:

* **No heavy dependencies.** Everything here is stdlib only (dataclasses,
  enum, datetime, abc, typing). MLflow, the Databricks SDK and agent SDKs are
  imported only by the concrete implementations. This keeps the seam itself
  trivially importable and forces producer-specific assumptions to live
  *outside* the contract.
* **The normalized record is the contract.** Downstream layers (L0 metrics,
  judges, the optimizer, the leaderboard) consume :class:`NormalizedTrace` and
  never touch a producer-specific shape. Adding a new agent means implementing
  these two interfaces — nothing downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "TraceStatus",
    "SpanKind",
    "TokenUsage",
    "ToolCall",
    "NormalizedSpan",
    "NormalizedTrace",
    "TraceSource",
    "AgentTask",
    "AgentRunResult",
    "AgentAdapter",
]


class TraceStatus(StrEnum):
    """Normalized terminal status of a trace, independent of producer.

    Producers report status in different vocabularies (MLflow ``TraceState``,
    SDK result codes, exit codes); they all collapse onto these values.
    """

    OK = "OK"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: Any) -> TraceStatus:
        """Best-effort map an arbitrary status value onto a member.

        Accepts enum members, their ``.value``, raw strings (case-insensitive)
        and common synonyms (``SUCCESS``/``OK``, ``FAILED``/``ERROR``).
        Anything unrecognized becomes :attr:`UNKNOWN` rather than raising — an
        unparseable status should never crash ingestion.
        """
        if isinstance(value, cls):
            return value
        raw = getattr(value, "value", value)
        text = str(raw).strip().upper()
        if text in ("OK", "SUCCESS", "SUCCEEDED", "COMPLETED", "STATE_OK"):
            return cls.OK
        if text in ("ERROR", "FAILED", "FAILURE", "STATE_ERROR"):
            return cls.ERROR
        if text in ("IN_PROGRESS", "RUNNING", "PENDING", "STATE_IN_PROGRESS"):
            return cls.IN_PROGRESS
        return cls.UNKNOWN


class SpanKind(StrEnum):
    """Normalized span kind, independent of producer.

    Mirrors the categories MLflow exposes plus an ``UNKNOWN`` fallback. Tool
    spans are the ones promoted to :class:`ToolCall`.
    """

    AGENT = "AGENT"
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    PARSER = "PARSER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: Any) -> SpanKind:
        """Map an arbitrary span-type value onto a member (``UNKNOWN`` fallback)."""
        if isinstance(value, cls):
            return value
        raw = getattr(value, "value", value)
        text = str(raw).strip().upper()
        try:
            return cls(text)
        except ValueError:
            return cls.UNKNOWN


@dataclass(slots=True)
class TokenUsage:
    """Token accounting for a trace or span.

    Cache fields are kept separate because cached reads are billed and
    weighted differently from fresh input tokens — the L0 cost metric needs
    both. ``total_tokens`` defaults to ``input + output`` when not supplied
    explicitly by the producer.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    _total_tokens: int | None = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        """Total billed tokens, preferring a producer-supplied total."""
        if self._total_tokens is not None:
            return self._total_tokens
        return self.input_tokens + self.output_tokens

    @property
    def cache_tokens(self) -> int:
        """Total cache-related tokens (creation + read)."""
        return self.cache_creation_input_tokens + self.cache_read_input_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two usages (used to aggregate span usage into a trace total)."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation extracted from a trace.

    This is the producer-agnostic view of "the agent used a tool". For MLflow
    traces it comes from ``TOOL`` spans; for adapter-captured runs it comes
    from streamed ``tool_use``/``tool_result`` events. ``name`` is the bare
    tool name (e.g. ``Read``, ``Bash``, ``mcp__databricks__execute_sql``) — not
    a producer-prefixed span name.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: TraceStatus = TraceStatus.UNKNOWN
    span_id: str | None = None
    start_time: datetime | None = None

    @property
    def is_mcp(self) -> bool:
        """Whether this is an MCP tool call (``mcp__server__tool``)."""
        return self.name.startswith("mcp__")

    @property
    def mcp_server(self) -> str | None:
        """The MCP server name, or ``None`` for non-MCP tools."""
        if not self.is_mcp:
            return None
        parts = self.name.split("__")
        return parts[1] if len(parts) >= 2 else None


@dataclass(slots=True)
class NormalizedSpan:
    """A single span in normalized form.

    A faithful-but-minimal projection of a producer span: enough to reconstruct
    the call tree, attribute tokens/timing, and locate tool/LLM activity,
    without binding to any producer's span schema.
    """

    span_id: str
    name: str
    kind: SpanKind = SpanKind.UNKNOWN
    parent_id: str | None = None
    status: TraceStatus = TraceStatus.UNKNOWN
    start_time: datetime | None = None
    end_time: datetime | None = None
    inputs: Any = None
    outputs: Any = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Span wall-clock duration in milliseconds, if both ends are known."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000.0
        return None


@dataclass(slots=True)
class NormalizedTrace:
    """The common trace record every producer normalizes to.

    This is the single shape consumed by everything downstream (L0 metrics,
    judges, optimizer, leaderboard). It carries the data those layers need —
    spans, tool calls, token usage, model, status, timestamps — plus enough
    provenance (``producer``, ``session_id``, ``experiment_id``, ``tags``,
    ``metadata``) to slice and group traces. ``raw`` is an escape hatch holding
    the original producer object for callers that need a field not yet promoted
    into this contract; downstream code should avoid depending on it.
    """

    trace_id: str
    status: TraceStatus = TraceStatus.UNKNOWN
    producer: str | None = None
    model: str | None = None
    session_id: str | None = None
    experiment_id: str | None = None
    request_time: datetime | None = None
    end_time: datetime | None = None
    execution_duration_ms: int | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    spans: list[NormalizedSpan] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    request_preview: str | None = None
    response_preview: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        """Total tokens for the trace."""
        return self.token_usage.total_tokens

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration in seconds.

        Prefers the producer-reported ``execution_duration_ms`` and falls back
        to ``end_time - request_time``.
        """
        if self.execution_duration_ms is not None:
            return self.execution_duration_ms / 1000.0
        if self.request_time and self.end_time:
            return (self.end_time - self.request_time).total_seconds()
        return None

    @property
    def total_tool_calls(self) -> int:
        """Number of tool calls in the trace."""
        return len(self.tool_calls)

    @property
    def tool_counts(self) -> dict[str, int]:
        """Map of tool name -> invocation count (insertion-ordered)."""
        return dict(Counter(tc.name for tc in self.tool_calls))


class TraceSource(ABC):
    """Pull traces from a tracking backend and normalize them.

    A ``TraceSource`` is producer-agnostic: it knows how to read a tracking
    backend (MLflow) and turn whatever it finds into :class:`NormalizedTrace`
    records, *without assuming which agent wrote them*. Implementations must not
    hardwire a producer's conventions (no "look in ``~/.claude/projects``", no
    "assume ``mlflow autolog claude``"). Producer detection, when possible, is
    best-effort and recorded on :attr:`NormalizedTrace.producer` — never
    required.

    Implementations provide :meth:`iter_traces` (streaming) and
    :meth:`get_trace` (single lookup); :meth:`fetch_traces` is a convenience
    that materializes the stream.
    """

    @abstractmethod
    def iter_traces(
        self,
        *,
        experiment_id: str,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
    ) -> Iterator[NormalizedTrace]:
        """Yield normalized traces from an experiment.

        Args:
            experiment_id: The experiment/location to read traces from.
            filter_string: Optional backend filter (e.g. an MLflow search
                filter such as ``"status = 'OK'"``). Passed through verbatim.
            max_results: Optional cap on the number of traces. ``None`` means
                "all available".
            order_by: Optional backend ordering clause (e.g.
                ``["timestamp_ms DESC"]``).

        Yields:
            One :class:`NormalizedTrace` per backend trace.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        """Fetch and normalize a single trace by id.

        Returns ``None`` if no trace with that id exists.
        """
        raise NotImplementedError

    def fetch_traces(
        self,
        *,
        experiment_id: str,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
    ) -> list[NormalizedTrace]:
        """Materialize :meth:`iter_traces` into a list (convenience wrapper)."""
        return list(
            self.iter_traces(
                experiment_id=experiment_id,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
            )
        )


@dataclass(slots=True)
class AgentTask:
    """A single unit of work to run an agent on.

    This is the stable, agent-agnostic input contract. ``prompt`` is the task
    text; the remaining fields are common knobs every coding agent exposes in
    some form. Anything producer-specific goes in ``params`` so the signature
    of :meth:`AgentAdapter.run` never has to change to support a new agent.
    """

    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    allowed_tools: list[str] | None = None
    cwd: str | None = None
    timeout_seconds: int = 300
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:
    """The outcome of running an agent on an :class:`AgentTask`.

    The ``trace`` is the whole point: a :class:`NormalizedTrace` captured from
    the run, identical in shape to what a :class:`TraceSource` produces, so the
    same downstream metrics/judges apply whether a trace was *replayed live*
    via an adapter or *read back* from the tracking backend.
    """

    trace: NormalizedTrace
    output_text: str = ""
    success: bool = True
    error: str | None = None
    session_id: str | None = None
    duration_ms: int | None = None
    raw: Any = field(default=None, repr=False)


class AgentAdapter(ABC):
    """Run an agent on a task and capture its trace.

    Implementations wrap a specific agent runtime (the Claude Agent SDK, a
    Codex runner, an HTTP endpoint) and present the same :meth:`run` method.
    The returned :class:`AgentRunResult` always carries a
    :class:`NormalizedTrace`, so the evaluation harness can replay a frozen
    task suite through any adapter and compare results apples-to-apples.

    :meth:`run` is synchronous by contract because not every agent runtime is
    async; async runtimes (like the Claude SDK) implement ``run`` over an
    internal event loop.
    """

    #: Short, stable identifier for the agent this adapter drives
    #: (e.g. ``"claude_code"``, ``"codex"``). Recorded on captured traces as
    #: :attr:`NormalizedTrace.producer`.
    name: str = "agent"

    @abstractmethod
    def run(self, task: AgentTask) -> AgentRunResult:
        """Execute ``task`` and return the result with a captured trace.

        Implementations should not raise for ordinary agent failures (timeouts,
        tool errors, a missing SDK): capture the failure on the returned
        :class:`AgentRunResult` (``success=False`` with ``error`` set) and
        still return a :class:`NormalizedTrace` reflecting whatever was
        observed. Reserve exceptions for programmer errors (bad arguments).
        """
        raise NotImplementedError
