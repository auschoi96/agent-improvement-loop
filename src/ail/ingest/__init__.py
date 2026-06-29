"""Trace ingestion + agent execution: the reusability seam.

The two abstractions in :mod:`ail.ingest.base` (``TraceSource`` and
``AgentAdapter``) are what make the improvement loop work for *any* agent
that can be traced into MLflow and run against a task input. Concrete
implementations live in :mod:`ail.ingest.mlflow_source` and
:mod:`ail.ingest.adapters`.
"""

from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedSpan,
    NormalizedTrace,
    SpanKind,
    TokenUsage,
    ToolCall,
    TraceSource,
    TraceStatus,
)

__all__ = [
    "AgentAdapter",
    "AgentRunResult",
    "AgentTask",
    "NormalizedSpan",
    "NormalizedTrace",
    "SpanKind",
    "TokenUsage",
    "ToolCall",
    "TraceSource",
    "TraceStatus",
]
