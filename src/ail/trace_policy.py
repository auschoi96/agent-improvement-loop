"""Shared policy for separating subject traces from framework-internal traces."""

from __future__ import annotations

from ail.ingest.base import NormalizedTrace

INTERNAL_TRACE_NAMES = frozenset(
    {
        "rlm_review",
        "l3_halo_review",
        "ail_judge_backfill",
        "ail_companion_planner",
        "ail_memory_distiller",
    }
)


def is_internal_trace(trace: NormalizedTrace) -> bool:
    """Return whether a trace was emitted by the improvement framework itself."""
    tags = getattr(trace, "tags", None) or {}
    metadata = getattr(trace, "metadata", None) or {}
    spans = getattr(trace, "spans", None) or []
    trace_name = str(tags.get("mlflow.traceName", "") or "")
    if trace_name in INTERNAL_TRACE_NAMES:
        return True
    if str(tags.get("ail.internal", "")).strip().lower() == "true":
        return True
    if metadata.get("ail.l3.subject_trace_id"):
        return True
    return any(
        any(str(key).startswith("ail.l3.") for key in (getattr(span, "attributes", None) or {}))
        for span in spans
    )


def subject_traces(traces: list[NormalizedTrace]) -> list[NormalizedTrace]:
    """Remove framework-internal traces from a materialized trace collection."""
    return [trace for trace in traces if not is_internal_trace(trace)]
