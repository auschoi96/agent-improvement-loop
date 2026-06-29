"""Stage 3 — the human gate. A reviewer fills expectations and approves.

This is the only module in the package that ever writes a non-empty
:class:`~ail.groundtruth.schema.Expectations`, and it does so **only** from
expectations a caller passes in — there is no model, no generation, no
synthesis here. The function signature *is* the gate:

* ``reviewer`` is required and must be a non-empty human identity. There is no
  "system" reviewer.
* approving requires the caller to supply :class:`Expectations` that are
  actually filled, a non-blank ``regression_intent``, and a destination
  ``target_pool``.
* the only decisions are :attr:`~ail.groundtruth.schema.ReviewStatus.APPROVED`
  and :attr:`~ail.groundtruth.schema.ReviewStatus.REJECTED`; you cannot
  "review" a case back into the ``CANDIDATE`` state.

Nothing here auto-accepts. A captured/executed case stays a candidate until a
human calls :func:`apply_review` with content they authored.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ail.groundtruth.schema import (
    Expectations,
    GroundTruthCase,
    GroundTruthError,
    Pool,
    ReviewRecord,
    ReviewStatus,
)

__all__ = [
    "ReviewError",
    "apply_review",
    "approve_case",
    "reject_case",
    "needs_review",
    "pending_cases",
]


class ReviewError(GroundTruthError):
    """A review was malformed (e.g. approving with empty expectations)."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def apply_review(
    case: GroundTruthCase,
    *,
    reviewer: str,
    decision: ReviewStatus,
    expectations: Expectations | None = None,
    regression_intent: str | None = None,
    target_pool: Pool | None = None,
    comment: str = "",
    decided_at: str | None = None,
) -> GroundTruthCase:
    """Apply a human review verdict, returning a new reviewed case.

    Args:
        case: the candidate (or already-reviewed) case being decided.
        reviewer: the human reviewer's identity. Required and non-empty.
        decision: ``APPROVED`` or ``REJECTED``.
        expectations: the human-authored expectations. **Required and non-empty
            to approve.** Ignored shape-wise for rejection (kept if supplied).
        regression_intent: human-authored statement of the regression this case
            guards. Required (non-blank) to approve. If omitted, the case's
            existing ``regression_intent`` is kept.
        target_pool: the destination pool. Required to approve.
        comment: free-form reviewer comment.
        decided_at: ISO-8601 timestamp; defaults to now.

    Raises:
        ReviewError: if the verdict is malformed (no reviewer, bad decision, or
            an approval missing expectations / regression_intent / target_pool).
    """
    if not reviewer or not reviewer.strip():
        raise ReviewError("a non-empty human reviewer identity is required")
    if decision not in (ReviewStatus.APPROVED, ReviewStatus.REJECTED):
        raise ReviewError(f"decision must be APPROVED or REJECTED, not {decision.value!r}")

    new_intent = case.regression_intent if regression_intent is None else regression_intent
    new_expectations = case.expectations if expectations is None else expectations
    new_pool = case.target_pool if target_pool is None else target_pool

    if decision is ReviewStatus.APPROVED:
        if not new_expectations.is_filled():
            raise ReviewError(
                "cannot approve a case with empty expectations — a human must "
                "author the expected behaviour first"
            )
        if not new_intent.strip():
            raise ReviewError("cannot approve a case with a blank regression_intent")
        if new_pool is None:
            raise ReviewError("cannot approve a case without choosing a target pool")

    review = ReviewRecord(
        status=decision,
        reviewer=reviewer,
        decided_at=decided_at or _utc_now_iso(),
        comment=comment,
    )
    return case.model_copy(
        update={
            "expectations": new_expectations,
            "regression_intent": new_intent,
            "target_pool": new_pool,
            "review": review,
        }
    )


def approve_case(
    case: GroundTruthCase,
    *,
    reviewer: str,
    expectations: Expectations,
    regression_intent: str,
    target_pool: Pool,
    comment: str = "",
    decided_at: str | None = None,
) -> GroundTruthCase:
    """Convenience wrapper over :func:`apply_review` for the approve path.

    Forces the caller to pass the expectations, regression intent, and pool that
    approval requires, so the human gate cannot be cleared by accident.
    """
    return apply_review(
        case,
        reviewer=reviewer,
        decision=ReviewStatus.APPROVED,
        expectations=expectations,
        regression_intent=regression_intent,
        target_pool=target_pool,
        comment=comment,
        decided_at=decided_at,
    )


def reject_case(
    case: GroundTruthCase,
    *,
    reviewer: str,
    comment: str = "",
    decided_at: str | None = None,
) -> GroundTruthCase:
    """Convenience wrapper over :func:`apply_review` for the reject path."""
    return apply_review(
        case,
        reviewer=reviewer,
        decision=ReviewStatus.REJECTED,
        comment=comment,
        decided_at=decided_at,
    )


def needs_review(case: GroundTruthCase) -> bool:
    """Whether a case is still an unreviewed candidate awaiting a human."""
    return case.review.status is ReviewStatus.CANDIDATE


def pending_cases(cases: list[GroundTruthCase]) -> list[GroundTruthCase]:
    """The subset of ``cases`` still awaiting human review."""
    return [c for c in cases if needs_review(c)]
