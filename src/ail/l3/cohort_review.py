"""Cohort batch-runner: L3-review a tag-defined slice of traces and roll up assets.

A single :func:`review_trace` reviews one subject trace; a *cohort* review runs it
over a **tag-defined slice** of an experiment (:mod:`ail.cohorts`) and produces the
aggregate the optimization loop actually acts on: a deduped, **recurrence-ranked**
list of recommended assets. An asset that recurs across many traces is the
highest-value Phase-2 target (build it once, help every trace), so recurrence is
the ranking key.

The flow, reusing the existing seams rather than re-implementing them:

1. **Select** the cohort — push the cohort's equality filter into the trace search
   (:meth:`ail.cohorts.Cohort.to_mlflow_filter`), enforce the full filter in memory
   (:meth:`~ail.cohorts.Cohort.select`, the source of truth), then rank/cap with the
   L3 size selection (:func:`ail.l3.selection.select_traces_to_review`) — L3 is
   expensive, so we review the biggest/most-interesting traces.
2. **Review** each selected trace with :func:`ail.l3.reviewer.review_trace` — its
   **own** reviewer trace (token isolation) and its per-guideline / assets / overall
   assessments attached to the *subject* trace. A trace whose review is degenerate
   (:class:`ail.l3.parser.HaloReportParseError`) — or that otherwise errors — is
   recorded as a ``review_failed`` skip, **never** a fabricated pass (fail-closed).
3. **Aggregate** the recommended assets across the successful verdicts into a
   :class:`~ail.l3.contract.CohortReviewReport`.

The MLflow v4 (UC-backed) trace store reads through a SQL warehouse, so
``sql_warehouse_id`` is surfaced as ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` (the same
plumbing :mod:`ail.publish` / :mod:`ail.compare.monitoring` use) before the cohort
search runs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ail.cohorts import Cohort, TagFilter
from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.ingest.base import TraceSource, TraceStatus
from ail.l3.contract import (
    CohortReviewReport,
    HaloReviewVerdict,
    RankedAsset,
    TraceReviewOutcome,
)
from ail.l3.reviewer import review_trace
from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric
from ail.l3.selection import select_traces_to_review

__all__ = ["aggregate_assets", "review_cohort"]

#: How many distinct sample rationales / benefits to keep per aggregated asset, and
#: how many evidence span ids — enough to audit the recommendation without letting
#: the report grow unbounded with the cohort size.
_MAX_SAMPLES = 5
_MAX_EVIDENCE_SPANS = 20

#: Accepted shapes for ``review_cohort``'s ``tag_filter``: a first-class
#: :class:`~ail.cohorts.Cohort`, a bare :class:`~ail.cohorts.TagFilter`, or the
#: ``{key: value}`` mapping :meth:`TagFilter.from_mapping` understands.
TagFilterSpec = Cohort | TagFilter | Mapping[str, "str | Iterable[str] | None"]


def _as_cohort(location: str, tag_filter: TagFilterSpec) -> Cohort:
    """Normalize the flexible ``tag_filter`` argument into a named :class:`Cohort`."""
    if isinstance(tag_filter, Cohort):
        return tag_filter
    tf = tag_filter if isinstance(tag_filter, TagFilter) else TagFilter.from_mapping(tag_filter)
    return Cohort(name=f"l3-cohort@{location}", tag_filter=tf)


def _render_filter(tag_filter: TagFilter) -> str:
    """A human-readable description of every clause (not just the pushable subset)."""
    parts: list[str] = []
    for clause in tag_filter.clauses:
        if not clause.values:
            parts.append(f"{clause.key} present")
        elif len(clause.values) == 1:
            (value,) = tuple(clause.values)
            parts.append(f"{clause.key}={value}")
        else:
            parts.append(f"{clause.key} in {{{', '.join(sorted(clause.values))}}}")
    return " AND ".join(parts) if parts else "(no filter: whole experiment)"


@dataclass(slots=True)
class _AssetAccumulator:
    """Running aggregate for one ``(asset_type, normalized title)`` across the cohort."""

    asset_type: str
    title: str
    trace_ids: list[str] = field(default_factory=list)
    occurrences: int = 0
    rationales: list[str] = field(default_factory=list)
    expected_benefits: list[str] = field(default_factory=list)
    evidence_span_ids: list[str] = field(default_factory=list)

    def add(self, trace_id: str, rationale: str, benefit: str, evidence: list[str]) -> None:
        self.occurrences += 1
        if trace_id not in self.trace_ids:
            self.trace_ids.append(trace_id)
        _append_capped(self.rationales, rationale, _MAX_SAMPLES)
        _append_capped(self.expected_benefits, benefit, _MAX_SAMPLES)
        for span_id in evidence:
            _append_capped(self.evidence_span_ids, span_id, _MAX_EVIDENCE_SPANS)


def _append_capped(items: list[str], value: str, cap: int) -> None:
    """Append a non-empty, not-already-present ``value`` while under ``cap``."""
    if value and value not in items and len(items) < cap:
        items.append(value)


def _asset_key(asset_type: str, title: str) -> tuple[str, str]:
    """Dedup key: asset type + case/whitespace-normalized title."""
    return (asset_type, " ".join(title.lower().split()))


def aggregate_assets(verdicts: Iterable[HaloReviewVerdict]) -> list[RankedAsset]:
    """Dedupe and recurrence-rank recommended assets across a cohort's verdicts.

    Assets are grouped by ``(asset_type, normalized title)``; an asset's
    ``n_traces`` is the number of **distinct** subject traces that recommended it
    (the recurrence signal). The result is ranked by ``n_traces`` (then total
    ``occurrences``, then type/title for a stable order) and assigned a 1-based
    ``rank`` — the most-recurring asset first.
    """
    acc: dict[tuple[str, str], _AssetAccumulator] = {}
    order: list[tuple[str, str]] = []
    for verdict in verdicts:
        for asset in verdict.recommended_assets:
            key = _asset_key(asset.asset_type, asset.title)
            entry = acc.get(key)
            if entry is None:
                entry = _AssetAccumulator(asset_type=asset.asset_type, title=asset.title.strip())
                acc[key] = entry
                order.append(key)
            entry.add(
                verdict.subject_trace_id,
                asset.rationale,
                asset.expected_benefit,
                asset.evidence_span_ids,
            )

    # Stable rank: most distinct traces first, then most total recommendations,
    # then a deterministic type/title tie-break. Iterating ``order`` (first-seen
    # order) through Python's stable ``sorted`` keeps ties in encounter order.
    ranked = sorted(
        (acc[key] for key in order),
        key=lambda e: (-len(e.trace_ids), -e.occurrences, e.asset_type, e.title.lower()),
    )
    return [
        RankedAsset(
            asset_type=entry.asset_type,  # type: ignore[arg-type]
            title=entry.title,
            rank=i,
            n_traces=len(entry.trace_ids),
            occurrences=entry.occurrences,
            trace_ids=entry.trace_ids,
            expected_benefits=entry.expected_benefits,
            rationales=entry.rationales,
            evidence_span_ids=entry.evidence_span_ids,
        )
        for i, entry in enumerate(ranked, start=1)
    ]


def review_cohort(
    location: str,
    tag_filter: TagFilterSpec,
    *,
    rubric: ReviewRubric = DEFAULT_RUBRIC,
    judge_model: str,
    sql_warehouse_id: str | None = None,
    profile: str | None = None,
    source: TraceSource | None = None,
    top_n: int | None = None,
    min_tokens: int | None = None,
    status: TraceStatus | None = TraceStatus.OK,
    max_results: int | None = None,
    attach: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    reviewer_experiment_id: str | None = None,
    max_turns: int | None = None,
    temperature: float | None = None,
    use_responses_api: bool = False,
    generated_at: str | None = None,
) -> CohortReviewReport:
    """Review every selected trace in a tag-defined cohort and aggregate the assets.

    Args:
        location: The experiment id whose traces the cohort is drawn from.
        tag_filter: The cohort definition — a :class:`~ail.cohorts.Cohort`, a
            :class:`~ail.cohorts.TagFilter`, or a ``{tag_key: value}`` mapping
            (see :meth:`TagFilter.from_mapping`; a value may be a string, an
            iterable of strings, or ``None`` for presence-only).
        rubric: The review rubric (the user's five guidelines by default). Drives
            every per-trace review. Named ``rubric`` to match
            :func:`ail.l3.reviewer.review_trace`; ``DEFAULT_RUBRIC`` is "the five
            guidelines".
        judge_model: Databricks FMAPI chat model HALO runs on for every review.
        sql_warehouse_id: SQL warehouse the v4 trace search/read uses (surfaced as
            ``MLFLOW_TRACING_SQL_WAREHOUSE_ID``). The calling identity still needs
            ``CAN_USE`` on it.
        profile: Databricks CLI profile selecting the workspace.
        source: Trace source to select from. Defaults to a Databricks
            :class:`~ail.ingest.mlflow_source.MLflowTraceSource`; inject a fake in
            tests.
        top_n / min_tokens / status: L3 size selection — keep at most ``top_n`` of
            the largest traces, drop those under ``min_tokens``, and (by default)
            only review ``OK`` traces. ``top_n=None`` reviews every matching trace.
        max_results: Cap on traces *scanned* from the backend before cohort/size
            filtering.
        attach: When ``True`` (default), attach each review's assessments to its
            subject trace.
        base_url / api_key / reviewer_experiment_id / max_turns / temperature /
            use_responses_api: Passed through to :func:`review_trace`.
        generated_at: ISO-8601 stamp for the report (defaults to now, UTC).

    Returns:
        A :class:`~ail.l3.contract.CohortReviewReport` with per-trace outcomes
        (``reviewed`` / ``review_failed`` — fail-closed, never a fake pass) and the
        deduped, recurrence-ranked recommended-asset roll-up.
    """
    stamp = generated_at or datetime.now(UTC).isoformat()
    cohort = _as_cohort(location, tag_filter)

    # The cohort search reads through the v4 trace store's SQL warehouse; surface
    # it before any read (review_trace re-asserts it per review).
    if sql_warehouse_id:
        os.environ[TRACING_WAREHOUSE_ENV] = sql_warehouse_id

    src = source if source is not None else _default_source(profile)

    # Push the cohort's equality filter into the backend scan, then enforce the
    # full cohort filter in memory (the source of truth), then L3-size-select.
    # This mirrors MLflowTraceSource.iter_cohort_traces' pushdown + post-filter,
    # but over the base TraceSource interface (fetch_traces + Cohort.select) so the
    # runner works with any injected source — including the test fakes — not only
    # the concrete MLflow source that carries the cohort-aware methods.
    scanned = src.fetch_traces(
        experiment_id=location,
        filter_string=cohort.to_mlflow_filter(),
        max_results=max_results,
    )
    cohort_traces = cohort.select(scanned)
    selections = select_traces_to_review(
        cohort_traces, top_n=top_n, min_tokens=min_tokens, status=status
    )

    extra: dict[str, object] = {}
    if max_turns is not None:
        extra["max_turns"] = max_turns

    outcomes: list[TraceReviewOutcome] = []
    verdicts: list[HaloReviewVerdict] = []
    for selection in selections:
        try:
            verdict = review_trace(
                selection.trace_id,
                experiment_id=location,
                model=judge_model,
                rubric=rubric,
                base_url=base_url,
                api_key=api_key,
                profile=profile,
                sql_warehouse_id=sql_warehouse_id,
                reviewer_experiment_id=reviewer_experiment_id,
                attach=attach,
                source=src,
                temperature=temperature,
                use_responses_api=use_responses_api,
                **extra,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - one bad trace must not kill the batch
            # Fail-closed: a degenerate/unparseable review (or any per-trace error)
            # is recorded as a SKIP, never a fabricated pass.
            outcomes.append(
                TraceReviewOutcome(
                    trace_id=selection.trace_id,
                    status="review_failed",
                    total_tokens=selection.total_tokens,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        verdicts.append(verdict)
        outcomes.append(
            TraceReviewOutcome(
                trace_id=selection.trace_id,
                status="reviewed",
                reviewer_trace_id=verdict.reviewer_trace_id,
                total_tokens=selection.total_tokens,
                token_efficiency=verdict.token_efficiency,
                token_waste_score=verdict.token_waste_score,
                n_recommended_assets=len(verdict.recommended_assets),
            )
        )

    ranked_assets = aggregate_assets(verdicts)
    n_reviewed = sum(1 for o in outcomes if o.status == "reviewed")
    n_failed = len(outcomes) - n_reviewed

    notes: list[str] = []
    if not selections:
        notes.append("cohort matched no traces to review (collecting / empty cohort)")
    if n_failed:
        notes.append(f"{n_failed} trace(s) recorded as review_failed (skipped, not a pass)")

    return CohortReviewReport(
        rubric_id=rubric.rubric_id,
        cohort_name=cohort.name,
        location=location,
        tag_filter=_render_filter(cohort.tag_filter),
        judge_model=judge_model,
        guideline_ids=list(rubric.guideline_ids()),
        n_selected=len(selections),
        n_reviewed=n_reviewed,
        n_failed=n_failed,
        outcomes=outcomes,
        ranked_assets=ranked_assets,
        generated_at=stamp,
        notes=notes,
    )


def _default_source(profile: str | None) -> TraceSource:
    """Build the default Databricks trace source (lazy import keeps core lean)."""
    from ail.ingest.mlflow_source import MLflowTraceSource

    return MLflowTraceSource(profile=profile)
