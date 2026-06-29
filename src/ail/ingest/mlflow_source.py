"""MLflow-backed :class:`~ail.ingest.base.TraceSource` (producer-agnostic).

HARVEST
-------
Source: ``databricks-solutions/ai-dev-kit``
        ``.test/src/skill_test/trace/mlflow_integration.py`` (the
        ``mlflow.search_traces(...)`` ingestion, ~L130) and
        ``_configure_mlflow`` auth setup (~L20).
Commit: c4947868f06fbfbb8cb666cbfba15888127b8a3a
License: see PROVENANCE.md (Databricks "DB license").

CHANGES FROM UPSTREAM
---------------------
* Refactored the free functions into a :class:`TraceSource` implementation.
* Dropped all Claude-Code hardwiring: no ``~/.claude/projects/*.jsonl`` local
  fallback, no ``mlflow autolog claude`` status probing, no "trace.jsonl"
  artifact download. This source reads only the MLflow Traces API
  (``search_traces`` / ``get_trace``) and works for any producer that logs
  traces to MLflow.
* Replaced the Claude-specific ``TraceMetrics`` output with the
  producer-agnostic :class:`~ail.ingest.base.NormalizedTrace`. Producer is
  *detected* (best-effort) rather than assumed.
* Token usage now reads MLflow 3's trace-level ``token_usage`` with fallbacks
  to trace metadata and span-level usage.
"""

from __future__ import annotations

import json
import os
import warnings
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

# Trace-metadata / span-attribute keys MLflow uses. Centralized so the
# producer-agnostic parsing has a single place to look.
_META_SESSION = "mlflow.trace.session"
_META_TOKEN_USAGE = "mlflow.trace.tokenUsage"
_META_CLAUDE_VERSION = "mlflow.claude_code_version"
_TAG_TRACE_NAME = "mlflow.traceName"
_SPAN_ATTR_MODEL = "model"
_SPAN_ATTR_TOKEN_USAGE = "mlflow.chat.tokenUsage"
_SPAN_ATTR_TOOL_NAME = "tool_name"
_SPAN_ATTR_TOOL_ID = "tool_id"


class MLflowTraceSource(TraceSource):
    """Read and normalize traces from an MLflow tracking backend.

    Authentication mirrors the upstream pattern: a Databricks CLI profile (via
    ``DATABRICKS_CONFIG_PROFILE`` or the ``profile`` argument) configures the
    SDK, and the tracking/registry URIs default to Databricks. Nothing here is
    specific to a particular agent — any trace in the experiment is normalized
    the same way.

    Args:
        tracking_uri: MLflow tracking URI (default ``"databricks"``).
        registry_uri: MLflow registry URI (default ``"databricks-uc"``).
        profile: Optional Databricks CLI profile. If given it is exported as
            ``DATABRICKS_CONFIG_PROFILE`` before MLflow is configured.
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

    # -- backend setup -----------------------------------------------------

    def _configure(self) -> None:
        """Configure MLflow auth once (harvested from ``_configure_mlflow``).

        Producer-agnostic: it only wires up Databricks auth + tracking URIs. It
        does not assume any autolog integration is active.
        """
        if self._configured:
            return
        try:
            import mlflow
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "mlflow is required for MLflowTraceSource. Install with: pip install 'mlflow>=3.0'"
            ) from e

        if self.profile:
            os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", self.profile)

        profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
        if profile and not os.environ.get("DATABRICKS_HOST"):
            try:
                from databricks.sdk import WorkspaceClient

                w = WorkspaceClient(profile=profile)
                os.environ["DATABRICKS_HOST"] = w.config.host
            except Exception:
                # databricks-sdk missing or profile unusable: fall back to
                # whatever ambient MLflow auth is configured.
                pass

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        self._configured = True

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

        with warnings.catch_warnings():
            # ``experiment_ids`` is deprecated in favor of ``locations`` in
            # recent MLflow but remains the most broadly compatible argument.
            warnings.simplefilter("ignore", FutureWarning)
            traces = mlflow.search_traces(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                return_type="list",
            )

        for trace in traces:
            # UC-table-backed traces do not echo their experiment id, so
            # backfill it from the query context.
            yield normalize_trace(trace, experiment_id_hint=experiment_id)

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        import mlflow

        self._configure()
        trace = mlflow.get_trace(trace_id)
        if trace is None:
            return None
        return normalize_trace(trace)


# ---------------------------------------------------------------------------
# Normalization: MLflow ``Trace`` -> ``NormalizedTrace``
#
# Operates on MLflow ``Trace`` objects so it behaves identically for traces
# fetched live and traces reconstructed via ``Trace.from_dict`` (the test
# fixture path). Everything is defensive: a missing/odd field degrades to a
# sensible default rather than raising, because ingestion runs over a live,
# heterogeneous corpus.
# ---------------------------------------------------------------------------


def normalize_trace(trace: Any, experiment_id_hint: str | None = None) -> NormalizedTrace:
    """Map an MLflow ``Trace`` onto a :class:`NormalizedTrace`.

    Args:
        trace: An ``mlflow.entities.Trace`` (or anything exposing the same
            ``.info`` / ``.data.spans`` shape).
        experiment_id_hint: Experiment id to record when the trace does not
            carry one itself (UC-table-backed traces do not echo it back).

    Returns:
        The normalized record, producer-detected on a best-effort basis.
    """
    info = trace.info
    metadata: dict[str, Any] = dict(getattr(info, "trace_metadata", None) or {})
    tags: dict[str, str] = {str(k): str(v) for k, v in (getattr(info, "tags", None) or {}).items()}

    spans = list(getattr(getattr(trace, "data", None), "spans", None) or [])
    normalized_spans = [_normalize_span(s) for s in spans]
    tool_calls = [
        _tool_call_from_span(raw, norm)
        for raw, norm in zip(spans, normalized_spans, strict=True)
        if norm.kind is SpanKind.TOOL
    ]

    request_time = _ms_to_dt(getattr(info, "timestamp_ms", None))
    duration_ms = getattr(info, "execution_time_ms", None)
    end_time = None
    if request_time is not None and isinstance(duration_ms, int):
        end_time = _ms_to_dt(getattr(info, "timestamp_ms", 0) + duration_ms)

    return NormalizedTrace(
        trace_id=str(getattr(info, "trace_id", "") or getattr(info, "request_id", "")),
        status=TraceStatus.coerce(getattr(info, "state", None) or getattr(info, "status", None)),
        producer=_infer_producer(tags, metadata),
        model=_infer_model(normalized_spans, metadata),
        session_id=metadata.get(_META_SESSION),
        experiment_id=_str_or_none(getattr(info, "experiment_id", None)) or experiment_id_hint,
        request_time=request_time,
        end_time=end_time,
        execution_duration_ms=duration_ms if isinstance(duration_ms, int) else None,
        token_usage=_extract_token_usage(info, metadata, normalized_spans),
        spans=normalized_spans,
        tool_calls=tool_calls,
        request_preview=_str_or_none(getattr(info, "request_preview", None)),
        response_preview=_str_or_none(getattr(info, "response_preview", None)),
        tags=tags,
        metadata=metadata,
        raw=trace,
    )


def _normalize_span(span: Any) -> NormalizedSpan:
    attrs: dict[str, Any] = dict(getattr(span, "attributes", None) or {})
    kind = SpanKind.coerce(getattr(span, "span_type", None))
    return NormalizedSpan(
        span_id=str(getattr(span, "span_id", "") or ""),
        name=str(getattr(span, "name", "") or ""),
        kind=kind,
        parent_id=_str_or_none(getattr(span, "parent_id", None)),
        status=_span_status(span),
        start_time=_ns_to_dt(getattr(span, "start_time_ns", None)),
        end_time=_ns_to_dt(getattr(span, "end_time_ns", None)),
        inputs=getattr(span, "inputs", None),
        outputs=getattr(span, "outputs", None),
        model=attrs.get(_SPAN_ATTR_MODEL) or _str_or_none(getattr(span, "model_name", None)),
        token_usage=_token_usage_from_dict(_maybe_json(attrs.get(_SPAN_ATTR_TOKEN_USAGE))),
        attributes=attrs,
    )


def _tool_call_from_span(raw_span: Any, norm: NormalizedSpan) -> ToolCall:
    attrs = norm.attributes
    name = attrs.get(_SPAN_ATTR_TOOL_NAME)
    if not name:
        # Span names are commonly prefixed, e.g. ``tool_Bash`` -> ``Bash``.
        name = norm.name[5:] if norm.name.startswith("tool_") else norm.name
    arguments = _as_dict(_maybe_json(norm.inputs))
    outputs = norm.outputs
    return ToolCall(
        id=str(attrs.get(_SPAN_ATTR_TOOL_ID) or norm.span_id),
        name=str(name),
        arguments=arguments,
        result=None if outputs is None else str(outputs),
        status=norm.status,
        span_id=norm.span_id,
        start_time=norm.start_time,
    )


# -- token usage -----------------------------------------------------------


def _extract_token_usage(
    info: Any, metadata: dict[str, Any], spans: list[NormalizedSpan]
) -> TokenUsage:
    """Resolve trace token usage, most-authoritative source first.

    1. MLflow 3 trace-level ``info.token_usage``.
    2. ``mlflow.trace.tokenUsage`` in trace metadata (JSON string).
    3. Sum of per-span ``mlflow.chat.tokenUsage``.
    """
    trace_usage = getattr(info, "token_usage", None)
    usage = _token_usage_from_dict(trace_usage)
    if usage is not None:
        return usage

    usage = _token_usage_from_dict(_maybe_json(metadata.get(_META_TOKEN_USAGE)))
    if usage is not None:
        return usage

    total = TokenUsage()
    found = False
    for span in spans:
        if span.token_usage is not None:
            total = total + span.token_usage
            found = True
    return total if found else TokenUsage()


def _token_usage_from_dict(data: Any) -> TokenUsage | None:
    if not isinstance(data, dict):
        return None
    return TokenUsage(
        input_tokens=int(data.get("input_tokens", 0) or 0),
        output_tokens=int(data.get("output_tokens", 0) or 0),
        cache_creation_input_tokens=int(data.get("cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0) or 0),
        _total_tokens=(int(data["total_tokens"]) if data.get("total_tokens") is not None else None),
    )


# -- producer / model detection (best-effort, never required) --------------


def _infer_producer(tags: dict[str, str], metadata: dict[str, Any]) -> str | None:
    """Best-effort producer detection. Returns ``None`` when unknown.

    Detection is a convenience for slicing/grouping; downstream code must work
    regardless of whether a producer was identified.
    """
    if _META_CLAUDE_VERSION in metadata or tags.get(_TAG_TRACE_NAME) == "claude_code_conversation":
        return "claude_code"
    return None


def _infer_model(spans: list[NormalizedSpan], metadata: dict[str, Any]) -> str | None:
    for span in spans:
        if span.kind is SpanKind.LLM and span.model:
            return span.model
    for span in spans:
        if span.model:
            return span.model
    return None


# -- small helpers ---------------------------------------------------------


def _span_status(span: Any) -> TraceStatus:
    status = getattr(span, "status", None)
    code = getattr(status, "status_code", status)
    return TraceStatus.coerce(code)


def _maybe_json(value: Any) -> Any:
    """Parse a value as JSON if it is a string; otherwise return it unchanged."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"input": value}


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _ms_to_dt(ms: Any) -> datetime | None:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def _ns_to_dt(ns: Any) -> datetime | None:
    if not isinstance(ns, (int, float)) or ns <= 0:
        return None
    return datetime.fromtimestamp(ns / 1_000_000_000.0, tz=UTC)
