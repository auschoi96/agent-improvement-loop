"""Stage 1 — capture candidate cases from traces.

Turns :class:`~ail.ingest.base.NormalizedTrace` records (the producer-agnostic
output of the ingestion seam) into :class:`~ail.groundtruth.schema.GroundTruthCase`
candidates. Capture extracts two things only:

* the **task input** (the prompt the agent was given), and
* the **provenance** (a :class:`~ail.groundtruth.schema.Source` pointing back at
  the trace).

Capture deliberately does **not** fill :class:`~ail.groundtruth.schema.Expectations`
— there is no LLM in this module and no expected output is invented. It also
leaves :attr:`~ail.groundtruth.schema.GroundTruthCase.candidate_response` for the
:mod:`~ail.groundtruth.execute` stage to produce by actually running an agent.
The trace's *historical* response is preserved in ``metadata`` purely as
reviewer context (clearly labelled "observed", never "expected").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ail.groundtruth.schema import (
    GroundTruthCase,
    GroundTruthError,
    Source,
    SourceKind,
    TaskInput,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ail.ingest.base import NormalizedTrace

__all__ = ["CaptureError", "candidate_from_trace", "capture_candidates"]

#: Default prefix for a generated case id, joined to the originating trace id so
#: re-capturing the same trace yields the same case id (capture is idempotent).
CASE_ID_PREFIX = "gt"


class CaptureError(GroundTruthError):
    """A trace could not be turned into a candidate case."""


def _case_id(trace: NormalizedTrace, prefix: str) -> str:
    return f"{prefix}::{trace.trace_id}"


def candidate_from_trace(
    trace: NormalizedTrace,
    *,
    case_id: str | None = None,
    case_id_prefix: str = CASE_ID_PREFIX,
    regression_intent: str = "",
) -> GroundTruthCase:
    """Build a single CANDIDATE case from a trace.

    The trace's request preview becomes the task prompt and the trace itself
    becomes the case's required provenance source. Expectations are left empty
    (filled later by a human); ``regression_intent`` defaults to blank and is
    likewise authored by a human at review time — passing it here is only a
    convenience for seeding a draft, never a synthesized expectation.

    Raises:
        CaptureError: if the trace has no usable prompt (no request preview).
    """
    prompt = (trace.request_preview or "").strip()
    if not prompt:
        raise CaptureError(
            f"trace {trace.trace_id!r} has no request preview to use as a task prompt"
        )

    source = Source(
        kind=SourceKind.TRACE,
        ref=trace.trace_id,
        locator=trace.session_id,
        note=f"producer={trace.producer} model={trace.model} status={trace.status.value}",
    )

    metadata: dict[str, object] = {
        "captured_from": "trace",
        "experiment_id": trace.experiment_id,
        "producer": trace.producer,
        "model": trace.model,
        # The agent's *historical* output, kept only as reviewer context. It is
        # labelled "observed" — it is NOT an expected output and is never copied
        # into Expectations.
        "observed_response_preview": trace.response_preview,
    }

    return GroundTruthCase(
        case_id=case_id or _case_id(trace, case_id_prefix),
        task_input=TaskInput(prompt=prompt, model=trace.model),
        sources=[source],
        regression_intent=regression_intent,
        metadata=metadata,
    )


def capture_candidates(
    traces: Iterable[NormalizedTrace],
    *,
    case_id_prefix: str = CASE_ID_PREFIX,
    skip_invalid: bool = True,
) -> list[GroundTruthCase]:
    """Capture a batch of candidate cases from traces.

    Args:
        traces: normalized traces (from any :class:`~ail.ingest.base.TraceSource`).
        case_id_prefix: prefix for generated case ids.
        skip_invalid: if ``True`` (default), traces with no usable prompt are
            skipped; if ``False``, the first such trace raises
            :class:`CaptureError`.

    Returns:
        Candidate cases with empty expectations, ready for the execute and
        review stages. De-duplicated by case id (idempotent re-capture).
    """
    seen: set[str] = set()
    cases: list[GroundTruthCase] = []
    for trace in traces:
        try:
            case = candidate_from_trace(trace, case_id_prefix=case_id_prefix)
        except CaptureError:
            if skip_invalid:
                continue
            raise
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        cases.append(case)
    return cases
