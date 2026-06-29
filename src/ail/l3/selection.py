"""Pick which traces to send to the L3 reviewer.

L3 is the expensive tier — HALO is a multi-agent, recursive review that spends
real tokens per trace — so it must not run on every trace
(``docs/ARCHITECTURE.md`` §3: L3 is for *discovery*, not the leaderboard). This
module selects the biggest / most-interesting traces: the long-tail sessions
where token waste actually lives (the reference corpus has a low median with
outliers near 549K and 943K tokens).

Two knobs, combinable: a ``min_tokens`` floor and a ``top_n`` cap. Selection is
ranked by total tokens descending, so "review the 5 biggest" and "review every
trace over 200K tokens" are both one call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ail.ingest.base import NormalizedTrace, TraceSource, TraceStatus
from ail.ingest.mlflow_source import MLflowTraceSource

__all__ = ["TraceSelection", "select_traces_to_review", "select_from_experiment"]


@dataclass(slots=True)
class TraceSelection:
    """One trace chosen for review, with the figures that justified the choice."""

    trace_id: str
    total_tokens: int
    total_tool_calls: int
    model: str | None = None
    experiment_id: str | None = None


def select_traces_to_review(
    traces: Iterable[NormalizedTrace],
    *,
    top_n: int | None = None,
    min_tokens: int | None = None,
    status: TraceStatus | None = TraceStatus.OK,
) -> list[TraceSelection]:
    """Rank and filter ``traces`` into the set worth an L3 review.

    Args:
        traces: Normalized traces to choose from.
        top_n: Keep at most this many (the largest). ``None`` means no cap.
        min_tokens: Drop traces below this many total tokens. ``None`` means no
            floor.
        status: Keep only traces with this status. Defaults to
            :attr:`~ail.ingest.base.TraceStatus.OK` (don't burn an expensive
            review on a trace that errored out); pass ``None`` to keep all.

    Returns:
        :class:`TraceSelection` records, sorted by ``total_tokens`` descending and
        capped to ``top_n``.
    """
    selections = [
        TraceSelection(
            trace_id=trace.trace_id,
            total_tokens=trace.total_tokens,
            total_tool_calls=trace.total_tool_calls,
            model=trace.model,
            experiment_id=trace.experiment_id,
        )
        for trace in traces
        if (status is None or trace.status is status)
        and (min_tokens is None or trace.total_tokens >= min_tokens)
    ]
    selections.sort(key=lambda s: s.total_tokens, reverse=True)
    return selections[:top_n] if top_n is not None else selections


def select_from_experiment(
    experiment_id: str,
    *,
    top_n: int | None = 5,
    min_tokens: int | None = None,
    status: TraceStatus | None = TraceStatus.OK,
    source: TraceSource | None = None,
    profile: str | None = None,
    max_results: int | None = None,
) -> list[TraceSelection]:
    """Pull an experiment's traces through the ingest seam and select the biggest.

    A convenience over :func:`select_traces_to_review` that fetches the corpus
    first. Defaults to the 5 largest ``OK`` traces — a sensible "review the
    interesting ones" default that never silently fans out to the whole corpus.
    """
    src = source if source is not None else MLflowTraceSource(profile=profile)
    traces = src.iter_traces(experiment_id=experiment_id, max_results=max_results)
    return select_traces_to_review(traces, top_n=top_n, min_tokens=min_tokens, status=status)
