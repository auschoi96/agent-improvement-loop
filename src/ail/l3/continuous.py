"""Event-triggered L3/RLM runner for newly-arrived traces.

Databricks Jobs supplies the arrival trigger (table update on the UC-backed
MLflow trace table). This module owns the cost guards inside each firing:

* skip any trace that already carries an ``rlm_*`` assessment;
* deterministically sample trace ids, then rank the sampled set by token count;
* cap the number of HALO reviews performed by one run; and
* delegate every selected trace to :func:`ail.l3.reviewer.review_trace`, preserving
  the existing fail-closed parser/attach behavior.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.ingest.base import NormalizedTrace, TraceSource, TraceStatus
from ail.l3.contract import HaloReviewVerdict, TraceReviewOutcome
from ail.l3.reviewer import (
    ASSETS_FEEDBACK_NAME,
    GUIDELINE_FEEDBACK_PREFIX,
    OVERALL_FEEDBACK_NAME,
    REVIEW_SPAN_NAME,
    review_trace,
)
from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric
from ail.l3.selection import TraceSelection, select_traces_to_review

__all__ = [
    "ContinuousRlmRunReport",
    "REVIEW_FAILED_FEEDBACK_NAME",
    "has_rlm_assessment",
    "sample_trace_id",
    "select_unreviewed_traces",
    "run_continuous_rlm",
]

#: Honest, distinct marker attached to a SUBJECT trace when a HALO review raised
#: (fail-closed: no verdict, no score written). It records ONLY that a review was
#: *attempted here and did not complete* — it is not a fabricated verdict and
#: carries no quality score. The idempotency guard treats it as "already handled"
#: so a permanently-failing trace is not re-sampled and re-reviewed on every
#: table_update firing (which would burn HALO cost until it aged out of the scan
#: window). Shares the ``rlm_`` prefix so it also reads as RLM-owned feedback.
REVIEW_FAILED_FEEDBACK_NAME = "rlm_review_failed"


@dataclass(slots=True)
class ContinuousRlmRunReport:
    """Summary for one arrival-triggered RLM job run."""

    experiment_id: str
    judge_model: str
    n_scanned: int
    n_already_reviewed: int
    n_reviewer_traces_skipped: int
    n_sampled_out: int
    n_selected: int
    n_reviewed: int
    n_failed: int
    sample_rate: float
    max_reviews: int
    outcomes: list[TraceReviewOutcome]


def has_rlm_assessment(trace: NormalizedTrace) -> bool:
    """Whether a trace already has any RLM/HALO feedback attached.

    The MLflow trace read surface carries assessments on ``trace.raw.info``. The
    helper is deliberately shape-tolerant so tests can use light fakes and so a
    future MLflow entity tweak fails closed toward "already reviewed" only when a
    real ``rlm_*`` name is visible.
    """
    raw = trace.raw
    info = getattr(raw, "info", None)
    assessments = getattr(info, "assessments", None) if info is not None else None
    for assessment in list(assessments or []):
        name = str(getattr(assessment, "name", "") or "")
        if _is_rlm_assessment_name(name):
            return True
    return False


def is_rlm_reviewer_trace(trace: NormalizedTrace) -> bool:
    """Whether ``trace`` is HALO's own reviewer trace, not a subject trace."""
    if trace.metadata.get("ail.l3.subject_trace_id"):
        return True
    for span in trace.spans:
        if span.name == REVIEW_SPAN_NAME:
            return True
        if any(str(key).startswith("ail.l3.") for key in span.attributes):
            return True
    return False


def sample_trace_id(trace_id: str, sample_rate: float) -> bool:
    """Deterministically decide whether ``trace_id`` is in the configured sample."""
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    digest = hashlib.sha256(trace_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < sample_rate


def select_unreviewed_traces(
    traces: list[NormalizedTrace],
    *,
    max_reviews: int,
    sample_rate: float,
    min_tokens: int | None = None,
    status: TraceStatus | None = TraceStatus.OK,
) -> tuple[list[TraceSelection], int, int, int]:
    """Choose the bounded, sampled, not-yet-reviewed subset for this firing.

    Returns ``(selections, n_already_reviewed, n_reviewer_traces_skipped,
    n_sampled_out)``. Selection remains token-prioritized by reusing the existing
    L3 size selector after idempotency and sampling have removed traces this run
    must not review.
    """
    if max_reviews < 1:
        return (
            [],
            sum(1 for t in traces if has_rlm_assessment(t)),
            sum(1 for t in traces if is_rlm_reviewer_trace(t)),
            0,
        )

    unreviewed: list[NormalizedTrace] = []
    n_already_reviewed = 0
    n_reviewer_traces_skipped = 0
    for trace in traces:
        if has_rlm_assessment(trace):
            n_already_reviewed += 1
        elif is_rlm_reviewer_trace(trace):
            n_reviewer_traces_skipped += 1
        else:
            unreviewed.append(trace)

    sampled: list[NormalizedTrace] = []
    n_sampled_out = 0
    for trace in unreviewed:
        if sample_trace_id(trace.trace_id, sample_rate):
            sampled.append(trace)
        else:
            n_sampled_out += 1

    selections = select_traces_to_review(
        sampled, top_n=max_reviews, min_tokens=min_tokens, status=status
    )
    return (selections, n_already_reviewed, n_reviewer_traces_skipped, n_sampled_out)


def run_continuous_rlm(
    experiment_id: str,
    *,
    judge_model: str,
    sql_warehouse_id: str | None = None,
    source: TraceSource | None = None,
    profile: str | None = None,
    max_results: int | None = 100,
    max_reviews: int = 2,
    sample_rate: float = 0.10,
    min_tokens: int | None = 50_000,
    status: TraceStatus | None = TraceStatus.OK,
    rubric: ReviewRubric = DEFAULT_RUBRIC,
    attach: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    reviewer_experiment_id: str | None = None,
    max_turns: int | None = None,
    temperature: float | None = None,
    use_responses_api: bool = False,
) -> ContinuousRlmRunReport:
    """Run one near-real-time RLM pass over newly-arrived trace candidates."""
    if not 0 <= sample_rate <= 1:
        raise ValueError("--sample-rate must be between 0 and 1")
    if max_reviews < 1:
        raise ValueError("--max-reviews must be at least 1")
    if sql_warehouse_id:
        os.environ[TRACING_WAREHOUSE_ENV] = sql_warehouse_id

    src = source if source is not None else _default_source(profile)
    traces = src.fetch_traces(
        experiment_id=experiment_id,
        max_results=max_results,
        order_by=["timestamp_ms DESC"],
    )
    (
        selections,
        n_already_reviewed,
        n_reviewer_traces_skipped,
        n_sampled_out,
    ) = select_unreviewed_traces(
        traces,
        max_reviews=max_reviews,
        sample_rate=sample_rate,
        min_tokens=min_tokens,
        status=status,
    )

    extra: dict[str, object] = {}
    if max_turns is not None:
        extra["max_turns"] = max_turns

    outcomes: list[TraceReviewOutcome] = []
    for selection in selections:
        try:
            verdict: HaloReviewVerdict = review_trace(
                selection.trace_id,
                experiment_id=experiment_id,
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
        except Exception as exc:  # noqa: BLE001 - one trace must not kill the run
            error = f"{type(exc).__name__}: {exc}"
            marker_note = ""
            if attach:
                # Attach the honest failure marker so this trace is skipped on later
                # firings. If even the marker attach fails, degrade to the prior
                # behavior (it may be retried next run) rather than killing the run.
                try:
                    _mark_review_failed(selection.trace_id, error=error, judge_model=judge_model)
                except Exception as mark_exc:  # noqa: BLE001
                    marker_note = (
                        f" (failure-marker attach failed: {type(mark_exc).__name__}: {mark_exc})"
                    )
            outcomes.append(
                TraceReviewOutcome(
                    trace_id=selection.trace_id,
                    status="review_failed",
                    total_tokens=selection.total_tokens,
                    error=f"{error}{marker_note}",
                )
            )
            continue

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

    n_reviewed = sum(1 for o in outcomes if o.status == "reviewed")
    return ContinuousRlmRunReport(
        experiment_id=experiment_id,
        judge_model=judge_model,
        n_scanned=len(traces),
        n_already_reviewed=n_already_reviewed,
        n_reviewer_traces_skipped=n_reviewer_traces_skipped,
        n_sampled_out=n_sampled_out,
        n_selected=len(selections),
        n_reviewed=n_reviewed,
        n_failed=len(outcomes) - n_reviewed,
        sample_rate=sample_rate,
        max_reviews=max_reviews,
        outcomes=outcomes,
    )


def _is_rlm_assessment_name(name: str) -> bool:
    # REVIEW_FAILED_FEEDBACK_NAME is listed explicitly (belt-and-suspenders) even
    # though it already matches the GUIDELINE_FEEDBACK_PREFIX check, so the
    # failed-review skip survives a future change to that prefix.
    return name in {
        OVERALL_FEEDBACK_NAME,
        ASSETS_FEEDBACK_NAME,
        REVIEW_FAILED_FEEDBACK_NAME,
    } or name.startswith(GUIDELINE_FEEDBACK_PREFIX)


def _mark_review_failed(trace_id: str, *, error: str, judge_model: str) -> None:
    """Attach the honest "review attempted and failed" marker to a subject trace.

    Mirrors the reviewer's ``mlflow.log_feedback`` attach seam, but writes a
    ``CODE``-sourced boolean (``rlm_review_failed = True``) with the error text as
    the rationale — deliberately *not* an ``LLM_JUDGE`` verdict and with no quality
    score. Its only job is to make :func:`has_rlm_assessment` skip this trace on
    later firings so a failing review is not retried unboundedly.
    """
    import mlflow
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    mlflow.log_feedback(
        trace_id=trace_id,
        name=REVIEW_FAILED_FEEDBACK_NAME,
        value=True,
        source=AssessmentSource(
            source_type=AssessmentSourceType.CODE,
            source_id="ail.l3.continuous",
        ),
        rationale=error,
        metadata={
            "ail.l3.marker": "review_failed",
            "ail.l3.judge_model": judge_model,
        },
    )


def _default_source(profile: str | None) -> TraceSource:
    from ail.ingest.mlflow_source import MLflowTraceSource

    return MLflowTraceSource(profile=profile)
