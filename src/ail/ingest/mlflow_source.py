"""Databricks-managed MLflow implementation of :class:`~ail.ingest.base.TraceSource`.

This source reads traces through MLflow's public Traces API
(:func:`mlflow.search_traces` / :func:`mlflow.get_trace`) and projects each
``mlflow.entities.Trace`` onto the producer-agnostic
:class:`~ail.ingest.base.NormalizedTrace`. It makes no assumption about which
agent produced a trace: the producer and model are *detected* on a best-effort
basis and recorded on the normalized record, never required.

Authentication targets **Databricks-managed MLflow**. Following the public
MLflow + Databricks SDK configuration model, the tracking URI is ``databricks``
and the model-registry URI is ``databricks-uc``; the active Databricks CLI
profile (``DATABRICKS_CONFIG_PROFILE`` or the ``profile`` argument) selects the
workspace whose credentials the SDK then resolves.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from ail.ingest.base import (
    NormalizedSpan,
    NormalizedTrace,
    SpanKind,
    TokenUsage,
    ToolCall,
    TraceSource,
    TraceStatus,
)

# Public MLflow trace-metadata / tag / span-attribute keys. Centralized so the
# normalization logic reads from one documented vocabulary.
_MD_SESSION = "mlflow.trace.session"
_MD_TOKEN_USAGE = "mlflow.trace.tokenUsage"
_MD_CLAUDE_VERSION = "mlflow.claude_code_version"
_TAG_TRACE_NAME = "mlflow.traceName"
_ATTR_MODEL = "model"
_ATTR_SPAN_TOKEN_USAGE = "mlflow.chat.tokenUsage"
_ATTR_TOOL_NAME = "tool_name"
_ATTR_TOOL_ID = "tool_id"

# A producer is recognized as Claude Code when its traces carry this trace name.
_CLAUDE_TRACE_NAME = "claude_code_conversation"
# Span names produced by Claude Code tracing are prefixed; strip it for the
# bare tool name when an explicit ``tool_name`` attribute is absent.
_TOOL_SPAN_PREFIX = "tool_"

_NS_PER_SECOND = 1_000_000_000
_MS_PER_SECOND = 1_000


class MLflowTraceSource(TraceSource):
    """Read and normalize traces from a Databricks-managed MLflow backend.

    Args:
        tracking_uri: MLflow tracking URI. Defaults to ``"databricks"`` so
            traces are read from Databricks-managed MLflow.
        registry_uri: MLflow model-registry URI. Defaults to ``"databricks-uc"``
            (Unity Catalog).
        profile: Optional Databricks CLI profile name. When given it is exported
            as ``DATABRICKS_CONFIG_PROFILE`` so the SDK authenticates against
            that workspace.
    """

    def __init__(
        self,
        tracking_uri: str = "databricks",
        registry_uri: str = "databricks-uc",
        profile: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.profile = profile
        self._configured = False

    # -- backend configuration --------------------------------------------

    def _configure(self) -> None:
        """Point MLflow at the configured Databricks workspace (idempotent)."""
        if self._configured:
            return
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "MLflowTraceSource requires mlflow. Install it with: pip install 'mlflow>=3.14,<4'"
            ) from exc

        if self.profile:
            os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", self.profile)
        self._resolve_workspace_host()

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        self._configured = True

    @staticmethod
    def _resolve_workspace_host() -> None:
        """Export ``DATABRICKS_HOST`` from the active CLI profile if unset.

        MLflow's ``databricks`` tracking URI authenticates through the
        Databricks SDK, which reads the workspace from a CLI profile in
        ``~/.databrickscfg``. Resolving the host up front via
        :class:`~databricks.sdk.WorkspaceClient` makes the target workspace
        explicit. This is best-effort: if the SDK is missing or the profile is
        unusable, we leave any ambient MLflow/Databricks auth untouched.
        """
        profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
        if not profile or os.environ.get("DATABRICKS_HOST"):
            return
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError:
            return
        try:
            host = WorkspaceClient(profile=profile).config.host
        except Exception:  # noqa: BLE001 - unusable profile: defer to ambient auth
            return
        if host:
            os.environ["DATABRICKS_HOST"] = host

    # -- TraceSource interface --------------------------------------------

    def iter_traces(
        self,
        *,
        experiment_id: str,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
    ) -> Iterator[NormalizedTrace]:
        import mlflow

        self._configure()
        # ``locations`` is the current public way to scope a trace search to one
        # or more experiments (a plain list of experiment ids).
        traces = mlflow.search_traces(
            locations=[experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            return_type="list",
        )
        for trace in traces:
            # UC-table-backed traces may not echo their experiment id back, so
            # supply it from the query as a fallback.
            yield normalize_trace(trace, experiment_id_hint=experiment_id)

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        import mlflow

        self._configure()
        trace = mlflow.get_trace(trace_id)
        return None if trace is None else normalize_trace(trace)


# ---------------------------------------------------------------------------
# Trace normalization: ``mlflow.entities.Trace`` -> ``NormalizedTrace``.
#
# Written against MLflow's public ``Trace`` shape (``trace.info`` +
# ``trace.data.spans``), so it behaves identically for traces fetched live and
# for traces rebuilt via ``Trace.from_dict`` (the offline test path). Every
# read is defensive: a missing or oddly typed field degrades to a sensible
# default instead of raising, because ingestion runs over a heterogeneous,
# real-world corpus.
# ---------------------------------------------------------------------------


def normalize_trace(trace: Any, experiment_id_hint: str | None = None) -> NormalizedTrace:
    """Project an MLflow ``Trace`` onto a :class:`NormalizedTrace`.

    Args:
        trace: An ``mlflow.entities.Trace`` (or any object exposing the same
            ``.info`` / ``.data.spans`` surface).
        experiment_id_hint: Experiment id to record when the trace itself does
            not carry one.

    Returns:
        The normalized record, with producer/model detected best-effort.
    """
    info = trace.info
    metadata: dict[str, Any] = dict(getattr(info, "trace_metadata", None) or {})
    tags: dict[str, str] = {str(k): str(v) for k, v in (getattr(info, "tags", None) or {}).items()}

    raw_spans = list(getattr(getattr(trace, "data", None), "spans", None) or [])
    spans = [_normalize_span(span) for span in raw_spans]
    tool_calls = [_tool_call_from_span(span) for span in spans if span.kind is SpanKind.TOOL]

    started_ms = _as_int(getattr(info, "timestamp_ms", None))
    duration_ms = _as_int(getattr(info, "execution_time_ms", None))
    request_time = _epoch_ms_to_dt(started_ms)
    end_time = _epoch_ms_to_dt(started_ms + duration_ms) if started_ms and duration_ms else None

    return NormalizedTrace(
        trace_id=_first_str(getattr(info, "trace_id", None), getattr(info, "request_id", None)),
        status=TraceStatus.coerce(getattr(info, "state", None) or getattr(info, "status", None)),
        producer=_detect_producer(tags, metadata),
        model=_detect_model(spans),
        session_id=metadata.get(_MD_SESSION),
        experiment_id=_str_or_none(getattr(info, "experiment_id", None)) or experiment_id_hint,
        request_time=request_time,
        end_time=end_time,
        execution_duration_ms=duration_ms,
        token_usage=_resolve_token_usage(info, metadata, spans),
        spans=spans,
        tool_calls=tool_calls,
        request_preview=_str_or_none(getattr(info, "request_preview", None)),
        response_preview=_str_or_none(getattr(info, "response_preview", None)),
        tags=tags,
        metadata=metadata,
        raw=trace,
    )


def _normalize_span(span: Any) -> NormalizedSpan:
    attributes: dict[str, Any] = dict(getattr(span, "attributes", None) or {})
    model = _str_or_none(attributes.get(_ATTR_MODEL)) or _str_or_none(
        getattr(span, "model_name", None)
    )
    return NormalizedSpan(
        span_id=str(getattr(span, "span_id", "") or ""),
        name=str(getattr(span, "name", "") or ""),
        kind=SpanKind.coerce(getattr(span, "span_type", None)),
        parent_id=_str_or_none(getattr(span, "parent_id", None)),
        status=_span_status(span),
        start_time=_epoch_ns_to_dt(getattr(span, "start_time_ns", None)),
        end_time=_epoch_ns_to_dt(getattr(span, "end_time_ns", None)),
        inputs=getattr(span, "inputs", None),
        outputs=getattr(span, "outputs", None),
        model=model,
        token_usage=_token_usage_from_dict(_maybe_json(attributes.get(_ATTR_SPAN_TOKEN_USAGE))),
        attributes=attributes,
    )


def _tool_call_from_span(span: NormalizedSpan) -> ToolCall:
    """Promote a normalized ``TOOL`` span to a :class:`ToolCall`."""
    attributes = span.attributes
    name = attributes.get(_ATTR_TOOL_NAME) or _strip_tool_prefix(span.name)
    outputs = span.outputs
    return ToolCall(
        id=str(attributes.get(_ATTR_TOOL_ID) or span.span_id),
        name=str(name),
        arguments=_as_dict(_maybe_json(span.inputs)),
        result=None if outputs is None else str(outputs),
        status=span.status,
        span_id=span.span_id,
        start_time=span.start_time,
    )


def _strip_tool_prefix(span_name: str) -> str:
    """Drop the ``tool_`` span-name prefix when no explicit tool name is given."""
    if span_name.startswith(_TOOL_SPAN_PREFIX):
        return span_name[len(_TOOL_SPAN_PREFIX) :]
    return span_name


# -- token usage -----------------------------------------------------------


def _resolve_token_usage(
    info: Any, metadata: dict[str, Any], spans: list[NormalizedSpan]
) -> TokenUsage:
    """Resolve trace token usage from the most authoritative source available.

    Order of preference:

    1. The MLflow 3 trace-level ``info.token_usage`` field.
    2. The ``mlflow.trace.tokenUsage`` trace-metadata entry (a JSON string).
    3. The sum of per-span ``mlflow.chat.tokenUsage`` values.
    """
    from_trace = _token_usage_from_dict(getattr(info, "token_usage", None))
    if from_trace is not None:
        return from_trace

    from_metadata = _token_usage_from_dict(_maybe_json(metadata.get(_MD_TOKEN_USAGE)))
    if from_metadata is not None:
        return from_metadata

    span_usages = [span.token_usage for span in spans if span.token_usage is not None]
    if not span_usages:
        return TokenUsage()
    total = TokenUsage()
    for usage in span_usages:
        total = total + usage
    return total


def _token_usage_from_dict(data: Any) -> TokenUsage | None:
    """Build :class:`TokenUsage` from a usage mapping, or ``None`` if not a dict.

    Keys follow the public Anthropic Messages API ``usage`` schema. A
    producer-supplied ``total_tokens`` is preserved as the authoritative total.
    """
    if not isinstance(data, dict):
        return None
    reported_total = data.get("total_tokens")
    return TokenUsage(
        input_tokens=_as_int(data.get("input_tokens")),
        output_tokens=_as_int(data.get("output_tokens")),
        cache_creation_input_tokens=_as_int(data.get("cache_creation_input_tokens")),
        cache_read_input_tokens=_as_int(data.get("cache_read_input_tokens")),
        _total_tokens=None if reported_total is None else int(reported_total),
    )


# -- producer / model detection (best-effort, never required) --------------


def _detect_producer(tags: dict[str, str], metadata: dict[str, Any]) -> str | None:
    """Best-effort producer detection. ``None`` means "unknown".

    Detection only powers slicing/grouping; downstream code must function
    whether or not a producer was identified.
    """
    looks_like_claude = (
        _MD_CLAUDE_VERSION in metadata or tags.get(_TAG_TRACE_NAME) == _CLAUDE_TRACE_NAME
    )
    return "claude_code" if looks_like_claude else None


def _detect_model(spans: list[NormalizedSpan]) -> str | None:
    """Pick a representative model: the first LLM span's model, else any span's."""
    for span in spans:
        if span.kind is SpanKind.LLM and span.model:
            return span.model
    for span in spans:
        if span.model:
            return span.model
    return None


# -- small, defensive helpers ----------------------------------------------


def _span_status(span: Any) -> TraceStatus:
    status = getattr(span, "status", None)
    return TraceStatus.coerce(getattr(status, "status_code", status))


def _maybe_json(value: Any) -> Any:
    """Decode a JSON string to its Python value; pass non-strings through."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a value into a dict: dicts pass through, ``None`` -> ``{}``, else wrapped."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"input": value}


def _as_int(value: Any) -> int:
    """Best-effort int coercion; ``None`` and unparseable values become ``0``."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _first_str(*candidates: Any) -> str:
    """First non-empty candidate as a string, or ``""`` if none qualify."""
    for candidate in candidates:
        text = _str_or_none(candidate)
        if text is not None:
            return text
    return ""


def _epoch_ms_to_dt(ms: Any) -> datetime | None:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / _MS_PER_SECOND, tz=UTC)


def _epoch_ns_to_dt(ns: Any) -> datetime | None:
    if not isinstance(ns, (int, float)) or ns <= 0:
        return None
    return datetime.fromtimestamp(ns / _NS_PER_SECOND, tz=UTC)
