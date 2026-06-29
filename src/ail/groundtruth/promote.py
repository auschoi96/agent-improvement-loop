"""Stage 4 — promote approved cases into a frozen pool.

Promotion is a **separate, explicit** step. Capture and execute never promote;
approving never promotes. Only :func:`promote_approved` writes into a frozen
pool, and only for cases that have cleared the human gate
(:meth:`~ail.groundtruth.schema.GroundTruthCase.is_promotable`).

It enforces the wall invariants:

* **Human-gated.** A case that is not approved-with-expectations is skipped
  (or, in ``strict`` mode, raises). You cannot promote a candidate.
* **Pools never mix.** A case is written to exactly one pool — the pool it was
  approved for — and only if its id is not already present in a *different*
  pool. A cross-pool collision raises :class:`PoolConflictError`.
* **The Task Suite is off-limits.** Promoting ground truth into
  :attr:`~ail.groundtruth.schema.Pool.TASK_SUITE` raises
  :class:`TaskSuiteProtectedError`. The Task Suite is the *held-out benchmark*
  the optimizer is measured against (``docs/ARCHITECTURE.md`` §2); feeding it
  from the same bootstrap loop that the agent and its judges learn from is
  exactly the co-adaptation we exist to prevent. The ground-truth bootstrap
  therefore targets only the *labelled* pools (``ALIGNMENT_SET`` /
  ``HUMAN_ANCHOR``). Populating the Task Suite is a **separate, future API/wave**
  (it is curated from task inputs, not from labelled ground truth — see
  ``docs/MILESTONE-1.md`` §1a, Wave 1b), deliberately not reachable from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ail.groundtruth.schema import (
    GroundTruthCase,
    GroundTruthError,
    GroundTruthSet,
    Pool,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ail.groundtruth.store import GroundTruthStore

__all__ = [
    "PromotionError",
    "PoolConflictError",
    "TaskSuiteProtectedError",
    "PromotionResult",
    "promote_approved",
]


class PromotionError(GroundTruthError):
    """Promotion was refused (e.g. a non-promotable case in strict mode)."""


class PoolConflictError(PromotionError):
    """A case id would land in two different pools — the wall must stay disjoint."""


class TaskSuiteProtectedError(PromotionError):
    """Refused an attempt to promote ground truth into the held-out Task Suite.

    The Task Suite is the frozen benchmark the optimizer is judged against; it
    must never be fed by the ground-truth bootstrap loop (anti-co-adaptation).
    """


@dataclass(slots=True)
class PromotionResult:
    """Outcome of a :func:`promote_approved` call."""

    pool: Pool
    set_name: str
    promoted: list[str] = field(default_factory=list)  # case ids written
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (case_id, reason)

    @property
    def n_promoted(self) -> int:
        return len(self.promoted)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def promote_approved(
    cases: Iterable[GroundTruthCase],
    *,
    pool: Pool,
    store: GroundTruthStore,
    set_name: str | None = None,
    strict: bool = False,
) -> PromotionResult:
    """Persist the human-approved subset of ``cases`` into ``pool``.

    Args:
        cases: candidate/reviewed cases to consider for promotion.
        pool: the destination frozen pool.
        store: where pools are persisted (enforces per-pool isolation).
        set_name: name for the pool's set; defaults to the pool value.
        strict: if ``True``, a non-promotable case (or one approved for a
            different pool) raises instead of being skipped.

    Returns:
        A :class:`PromotionResult` listing promoted and skipped case ids.

    Raises:
        TaskSuiteProtectedError: if ``pool`` is the held-out Task Suite.
        PoolConflictError: if a case id is already stored in a different pool.
        PromotionError: in ``strict`` mode, on the first non-promotable case.
    """
    if pool is Pool.TASK_SUITE:
        raise TaskSuiteProtectedError(
            "refusing to promote ground truth into the held-out Task Suite; it is the "
            "frozen benchmark and must not be fed by the bootstrap loop (Task Suite "
            "population is a separate future API — see promote.py module docs)"
        )

    result = PromotionResult(pool=pool, set_name=set_name or pool.value)

    # Cross-pool disjointness: a case id already living elsewhere may not be
    # duplicated into this pool.
    cross_pool = {cid: p for cid, p in store.case_pool_index().items() if p is not pool}

    existing = store.load(pool)
    by_id: dict[str, GroundTruthCase] = {c.case_id: c for c in existing.cases}

    for case in cases:
        blockers = case.promotion_blockers()
        if blockers:
            reason = "; ".join(blockers)
            if strict:
                raise PromotionError(f"case {case.case_id!r} is not promotable: {reason}")
            result.skipped.append((case.case_id, reason))
            continue

        if case.target_pool is not pool:
            reason = f"approved for pool {case.target_pool} not {pool.value!r}"
            if strict:
                raise PromotionError(f"case {case.case_id!r}: {reason}")
            result.skipped.append((case.case_id, reason))
            continue

        other = cross_pool.get(case.case_id)
        if other is not None:
            raise PoolConflictError(
                f"case {case.case_id!r} is already in pool {other.value!r}; "
                f"refusing to also write it to {pool.value!r}"
            )

        by_id[case.case_id] = case  # idempotent: re-promoting replaces in place
        result.promoted.append(case.case_id)

    updated = GroundTruthSet(
        pool=pool,
        name=result.set_name,
        created_at=existing.created_at or _utc_now_iso(),
        cases=list(by_id.values()),
    )
    store.save(updated)
    return result
