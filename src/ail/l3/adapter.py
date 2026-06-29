"""MLflow trace → OpenInference/OTLP ``SpanRecord`` JSONL — the HALO input adapter.

HALO (``halo-engine``) indexes a **flat** trace file: one JSON object per line,
each a single span in the OpenTelemetry/OpenInference ``SpanRecord`` shape
(``trace_id``, ``span_id``, ``parent_span_id``, ``name``, ``kind``,
``start_time``, ``end_time``, ``status``, ``resource``, ``scope``,
``attributes``). This module projects a producer-agnostic
:class:`~ail.ingest.base.NormalizedTrace` onto exactly that shape so HALO's
byte-offset index, navigation tools, and renderer can read the trace.

Two layers, deliberately split so the mapping is testable without a tracking
backend:

* :func:`normalized_trace_to_span_records` — pure, in-memory:
  :class:`~ail.ingest.base.NormalizedTrace` → ``list[dict]`` of SpanRecords.
  No MLflow, no I/O. This is the heart of the adapter and where the
  OpenInference attribute conventions live.
* :func:`mlflow_trace_to_otlp_jsonl` — pulls a single trace through the
  :mod:`ail.ingest` seam (the public MLflow Traces API, agent-agnostic) and
  writes the JSONL file HALO consumes.

**Attribute conventions.** HALO's index and renderer key off OpenInference
semantic conventions (verified against ``halo-engine``'s
``trace_index_builder`` / ``trace_store``): ``openinference.span.kind`` for the
semantic kind, ``llm.model_name`` / ``inference.llm.model_name`` for the model,
``inference.llm.input_tokens`` / ``inference.llm.output_tokens`` (ints) for
usage, ``inference.agent_name`` / ``inference.agent_id`` for agent identity, and
``input.value`` / ``output.value`` (plus ``tool.name`` / ``tool.parameters``)
for the content the reviewer reads. We emit those keys derived from the
normalized fields rather than passing MLflow's raw span attributes through, so
the file HALO sees is clean and self-describing.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
from ail.ingest.mlflow_source import MLflowTraceSource

__all__ = [
    "OtlpExport",
    "normalized_trace_to_span_records",
    "write_span_records_jsonl",
    "mlflow_trace_to_otlp_jsonl",
]

# Instrumentation scope recorded on every emitted span: this adapter is the
# "instrumentation" that produced the HALO input file.
_SCOPE_NAME = "ail.l3.otlp_adapter"
_SCOPE_VERSION = "1"

# OTel status codes HALO expects (it special-cases ``STATUS_CODE_ERROR``).
_OTEL_STATUS = {
    TraceStatus.OK: "STATUS_CODE_OK",
    TraceStatus.ERROR: "STATUS_CODE_ERROR",
    TraceStatus.IN_PROGRESS: "STATUS_CODE_UNSET",
    TraceStatus.UNKNOWN: "STATUS_CODE_UNSET",
}

# Top-level OTel ``kind`` is informational for HALO (its logic reads the
# semantic ``openinference.span.kind`` attribute instead), so a single neutral
# value is correct for spans reconstructed from a tracking backend.
_OTEL_SPAN_KIND = "SPAN_KIND_INTERNAL"

# NormalizedTrace SpanKind → OpenInference ``span.kind``. OpenInference has no
# PARSER kind; PARSER/UNKNOWN map to the generic CHAIN so the value is always a
# valid OpenInference kind.
_OPENINFERENCE_KIND = {
    SpanKind.AGENT: "AGENT",
    SpanKind.LLM: "LLM",
    SpanKind.TOOL: "TOOL",
    SpanKind.CHAIN: "CHAIN",
    SpanKind.RETRIEVER: "RETRIEVER",
    SpanKind.RERANKER: "RERANKER",
    SpanKind.EMBEDDING: "EMBEDDING",
    SpanKind.PARSER: "CHAIN",
    SpanKind.UNKNOWN: "CHAIN",
}


@dataclass(slots=True)
class OtlpExport:
    """Result of writing a trace's spans to a HALO-readable JSONL file."""

    path: Path
    trace_id: str
    n_spans: int


def _iso(dt: datetime | None) -> str:
    """ISO-8601 timestamp, or ``""`` when unknown.

    HALO compares ``start_time`` / ``end_time`` **lexicographically** to find a
    trace's bounds, so a zero-padded ISO-8601 string (which sorts in
    chronological order) is the right wire form; an empty string sorts before
    any real timestamp and is a safe "unknown".
    """
    return dt.isoformat() if dt is not None else ""


def _to_text(value: Any) -> str:
    """Render a span input/output payload as text for HALO to read.

    Strings pass through; everything else is JSON-encoded so structured
    payloads (tool args, message lists) stay legible. Not truncated here — HALO
    head-caps oversized attributes itself when it returns spans to the model.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _token_attrs(usage: TokenUsage | None) -> dict[str, int]:
    """OpenInference token-usage attributes (only non-zero counts, as ints)."""
    if usage is None:
        return {}
    attrs: dict[str, int] = {}
    if usage.input_tokens:
        attrs["inference.llm.input_tokens"] = int(usage.input_tokens)
    if usage.output_tokens:
        attrs["inference.llm.output_tokens"] = int(usage.output_tokens)
    return attrs


def _span_attributes(
    span: NormalizedSpan,
    *,
    tool_call: ToolCall | None,
    trace_producer: str | None,
) -> dict[str, Any]:
    """Build the OpenInference ``attributes`` map for one span."""
    attrs: dict[str, Any] = {"openinference.span.kind": _OPENINFERENCE_KIND[span.kind]}

    if span.model:
        attrs["llm.model_name"] = span.model
        attrs["inference.llm.model_name"] = span.model
    attrs.update(_token_attrs(span.token_usage))

    if span.kind is SpanKind.AGENT:
        attrs["inference.agent_name"] = span.name or (trace_producer or "agent")
        attrs["inference.agent_id"] = span.span_id

    if tool_call is not None:
        attrs["tool.name"] = tool_call.name
        if tool_call.arguments:
            attrs["tool.parameters"] = _to_text(tool_call.arguments)

    if span.inputs is not None:
        attrs["input.value"] = _to_text(span.inputs)
    if span.outputs is not None:
        attrs["output.value"] = _to_text(span.outputs)

    return attrs


def _span_record(
    span: NormalizedSpan,
    *,
    trace: NormalizedTrace,
    tool_call: ToolCall | None,
) -> dict[str, Any]:
    """Project one :class:`NormalizedSpan` onto a HALO ``SpanRecord`` dict."""
    return {
        "trace_id": trace.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_id or "",
        "trace_state": "",
        "name": span.name,
        "kind": _OTEL_SPAN_KIND,
        "start_time": _iso(span.start_time),
        "end_time": _iso(span.end_time),
        "status": {"code": _OTEL_STATUS[span.status], "message": ""},
        "resource": {
            "attributes": {
                "service.name": trace.producer or "unknown",
                "ail.mlflow.trace_id": trace.trace_id,
                **({"ail.mlflow.session_id": trace.session_id} if trace.session_id else {}),
            }
        },
        "scope": {"name": _SCOPE_NAME, "version": _SCOPE_VERSION},
        "attributes": _span_attributes(span, tool_call=tool_call, trace_producer=trace.producer),
    }


def normalized_trace_to_span_records(trace: NormalizedTrace) -> list[dict[str, Any]]:
    """Project a :class:`NormalizedTrace` onto a list of HALO ``SpanRecord`` dicts.

    One dict per span, in the trace's span order. ``TOOL`` spans are enriched
    with the resolved tool name/arguments/result from the trace's
    :attr:`~ail.ingest.base.NormalizedTrace.tool_calls` (matched by span id), so
    the bare ``Read`` / ``Bash`` / ``mcp__...`` names — not the producer-prefixed
    span names — reach HALO. Pure and side-effect-free: validate or serialize
    the result however you like.
    """
    tools_by_span: dict[str, ToolCall] = {
        tc.span_id: tc for tc in trace.tool_calls if tc.span_id is not None
    }
    return [
        _span_record(span, trace=trace, tool_call=tools_by_span.get(span.span_id))
        for span in trace.spans
    ]


def write_span_records_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write SpanRecord dicts to ``path`` as JSONL (one span per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")
    return out


def mlflow_trace_to_otlp_jsonl(
    trace_id: str,
    experiment_id: str | None = None,
    *,
    path: str | Path | None = None,
    source: TraceSource | None = None,
    profile: str | None = None,
) -> OtlpExport:
    """Pull one trace and write it as the OTLP/OpenInference JSONL HALO indexes.

    Reads the trace through the agent-agnostic :mod:`ail.ingest` seam (the public
    MLflow Traces API), then writes the flat SpanRecord JSONL.

    Args:
        trace_id: The subject trace to export.
        experiment_id: The experiment the trace lives in. Not required to fetch a
            trace by id, but recorded on the returned :class:`OtlpExport`'s file
            name when ``path`` is omitted and available to callers for context.
        path: Destination JSONL path. When ``None``, a temp file is created
            (``ail-l3-<trace>.jsonl``) and its path returned — the caller owns
            cleanup.
        source: Trace source to read from. Defaults to a
            :class:`~ail.ingest.mlflow_source.MLflowTraceSource` (Databricks-managed
            MLflow). Inject a fake in tests to avoid a live backend.
        profile: Databricks CLI profile selecting the workspace (forwarded to the
            default :class:`MLflowTraceSource`; ignored when ``source`` is given).

    Returns:
        An :class:`OtlpExport` with the written path, the resolved trace id, and
        the span count.

    Raises:
        LookupError: If no trace with ``trace_id`` exists.
    """
    src = source if source is not None else MLflowTraceSource(profile=profile)
    trace = src.get_trace(trace_id)
    if trace is None:
        raise LookupError(f"no trace found for trace_id={trace_id!r}")

    records = normalized_trace_to_span_records(trace)

    if path is None:
        safe = trace_id.replace("/", "_").replace(":", "_")
        fd, tmp = tempfile.mkstemp(prefix=f"ail-l3-{safe[-40:]}-", suffix=".jsonl")
        os.close(fd)
        path = tmp

    out = write_span_records_jsonl(records, path)
    return OtlpExport(path=out, trace_id=trace.trace_id, n_spans=len(records))
