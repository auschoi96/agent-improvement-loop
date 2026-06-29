"""Our own ground-truth contract — a versioned, self-validating Pydantic schema.

This module is **clean-room original work**. It is *not* a copy of SkillForge's
``GroundTruthV5`` (whose license is undeclared — see ``PROVENANCE.md``) nor of
any ``ai-dev-kit`` GRP code. We admired the *shape* of those contracts
conceptually — in particular that every case carries its provenance and states
why it exists — and re-derived an equivalent contract from scratch for this
repository. The two fields we deliberately make first-class are:

* :attr:`GroundTruthCase.sources` — **required** provenance/citations. A case
  with no provenance is not a ground-truth case; it is a guess.
* :attr:`GroundTruthCase.regression_intent` — a human-authored statement of
  *what regression this case guards against*. A promoted case must say why it
  is worth keeping.

The anti-co-adaptation invariant the whole package exists to protect lives in
the type system here:

> :class:`Expectations` (the expected behaviour a judge/test checks against) is
> **never** populated by :mod:`~ail.groundtruth.capture` or
> :mod:`~ail.groundtruth.execute`. It starts empty and is filled **only** by a
> human in :mod:`~ail.groundtruth.approve`. There is no LLM synthesis of
> expected outputs anywhere in this package.

:class:`CandidateResponse` is the agent's *own* output captured for the
reviewer to judge — it is explicitly **not** an expected output. Recording what
the agent produced is fine; inventing what it *should* produce is not.

Everything is a Pydantic v2 model with ``extra="forbid"`` (drift is loud) and
``frozen=True`` (a case is evolved by building a new, validated instance, never
by mutating expectations in place), so the contract round-trips through JSON
(``model_dump_json`` / ``model_validate_json``) with no custom serialization.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The Pool identity is the one canonical vocabulary type, shared with
# ``ail.judges`` so the storage layer and the judge layer can never disagree on
# "which pool". Re-exported from this schema module (``from
# ail.groundtruth.schema import Pool``) for the package's existing consumers.
from ail.pools import Pool

#: Version of the ground-truth contract. Bump the minor for additive,
#: backward-compatible fields; bump the major for breaking shape changes.
SCHEMA_VERSION = "ail.groundtruth/v1"


class GroundTruthError(Exception):
    """Base class for ground-truth contract/pipeline errors."""


class _Frozen(BaseModel):
    """Base for every model: forbid unknown fields and freeze instances.

    ``extra="forbid"`` makes schema drift fail loudly instead of silently
    dropping data. ``frozen=True`` means a case cannot be mutated in place — the
    pipeline evolves a case by constructing a new validated instance — which is
    what structurally prevents :class:`Expectations` from being quietly written
    by a stage that has no business filling them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceKind(StrEnum):
    """Where a piece of provenance points."""

    TRACE = "trace"  # an MLflow / normalized trace the candidate was derived from
    DOCUMENT = "document"  # a spec, doc, ticket, or other citation
    DATASET = "dataset"  # a row in an existing dataset
    HUMAN = "human"  # a human-provided reference / observation
    OTHER = "other"


class Source(_Frozen):
    """A single provenance/citation entry.

    Every :class:`GroundTruthCase` requires at least one. This is the answer to
    "where did this case come from?" — a trace id, a doc URL, a dataset row.
    Provenance is *not* an expected output, so :mod:`~ail.groundtruth.capture`
    fills it freely; it is the grounding that makes a candidate auditable.
    """

    kind: SourceKind
    ref: str  # trace id, URL, file path, dataset id, …
    locator: str | None = None  # span id, line range, cell, session id, …
    note: str = ""


class ReviewStatus(StrEnum):
    """Lifecycle state of a case's human review."""

    CANDIDATE = "candidate"  # captured/executed, awaiting a human
    APPROVED = "approved"  # a human filled expectations and approved it
    REJECTED = "rejected"  # a human rejected it


class TaskInput(_Frozen):
    """The agent-agnostic task a case exercises.

    Mirrors the knobs of :class:`ail.ingest.base.AgentTask` (the execution-time
    input contract) as a validated, JSON-serializable model. :mod:`execute`
    bridges this into an ``AgentTask`` to run the agent.
    """

    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Expectations(_Frozen):
    """Human-authored expected behaviour — **empty until a reviewer fills it**.

    This is the object a judge or programmatic check evaluates a future agent
    response against. It is the single most safety-critical field in the
    package: if it were ever synthesized by a model, the agent and its judge
    would co-adapt and scores would climb while real quality stalled.

    Therefore **no stage other than** :func:`ail.groundtruth.approve.apply_review`
    ever sets a non-default value here. Capture and execute leave it at the
    empty default ``{}`` — ``# filled by reviewer``.
    """

    expected_response: str | None = None
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    rubric: str | None = None
    assertions: list[str] = Field(default_factory=list)
    reviewer_notes: str = ""

    def is_filled(self) -> bool:
        """Whether a human has authored any expectation at all.

        An empty :class:`Expectations` (the capture/execute default) returns
        ``False``; promotion refuses any case for which this is ``False``.
        """
        return bool(
            self.expected_response
            or self.must_include
            or self.must_not_include
            or self.rubric
            or self.assertions
        )


class CandidateResponse(_Frozen):
    """The agent's **own** output for a case, captured for the reviewer.

    Recorded by :mod:`~ail.groundtruth.execute` (a fresh run) or carried from a
    historical trace. This is *not* an expected output — it is what the agent
    actually produced, shown to the human so they can author
    :class:`Expectations`. Storing it never implies it is correct.
    """

    output_text: str
    producer: str | None = None  # which agent produced it (adapter name)
    model: str | None = None
    trace_id: str | None = None
    success: bool = True
    error: str | None = None
    duration_ms: int | None = None
    captured_at: str | None = None  # ISO-8601


class ReviewRecord(_Frozen):
    """The human review verdict attached to a case."""

    status: ReviewStatus = ReviewStatus.CANDIDATE
    reviewer: str | None = None  # human identity; required to approve/reject
    decided_at: str | None = None  # ISO-8601
    comment: str = ""


class GroundTruthCase(_Frozen):
    """One versioned ground-truth case.

    A case moves through the pipeline as four explicit stages, each producing a
    *new* instance (frozen): captured -> executed -> reviewed -> promoted. The
    review fields (:attr:`expectations`, :attr:`regression_intent`,
    :attr:`target_pool`, :attr:`review`) are the only ones a human touches, and
    they gate promotion via :meth:`promotion_blockers`.
    """

    case_id: str
    schema_version: str = SCHEMA_VERSION
    task_input: TaskInput
    # REQUIRED provenance — a case must cite where it came from (>= 1 source).
    sources: list[Source] = Field(min_length=1)
    # Human-authored at the approve stage, exactly like `expectations`: blank on
    # a fresh candidate, authored by the reviewer, and enforced non-blank at the
    # frozen-pool boundary (see `validate_pool_membership` / promotion_blockers).
    # It is therefore *required to promote*, not required to capture — a base
    # min_length on a field that legitimately starts empty for candidates would
    # be inconsistent with the candidate lifecycle.
    regression_intent: str = ""
    expectations: Expectations = Field(default_factory=Expectations)  # {} filled by reviewer
    candidate_response: CandidateResponse | None = None
    review: ReviewRecord = Field(default_factory=ReviewRecord)
    target_pool: Pool | None = None  # destination pool, chosen by the human at review time
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def promotion_blockers(self) -> list[str]:
        """Reasons this case may **not** be promoted (empty list == promotable).

        Encodes the human gate as data: a case is promotable only when a human
        has approved it, named themselves as the reviewer, filled expectations,
        stated a regression intent, and chosen a destination pool.
        """
        blockers: list[str] = []
        if self.review.status is not ReviewStatus.APPROVED:
            blockers.append(f"review status is {self.review.status.value}, not approved")
        if not self.review.reviewer:
            blockers.append("no human reviewer recorded")
        if not self.expectations.is_filled():
            blockers.append("expectations are empty (must be authored by a human)")
        if not self.regression_intent.strip():
            blockers.append("regression_intent is blank")
        if self.target_pool is None:
            blockers.append("no target pool selected")
        if not self.sources:
            blockers.append("no provenance sources")
        return blockers

    def is_promotable(self) -> bool:
        """Whether the case has cleared the human gate and can be promoted."""
        return not self.promotion_blockers()


def validate_pool_membership(pool: Pool, cases: Sequence[GroundTruthCase]) -> None:
    """Assert every case may legitimately live in a frozen ``pool`` (or raise).

    This is the single definition of "what is allowed in a pool", reused by the
    :class:`GroundTruthSet` validator *and* the persistence boundary
    (:meth:`ail.groundtruth.store.GroundTruthStore.save`) so the human gate
    cannot be sidestepped by constructing a set or writing a file directly. A
    case is admissible only if it:

    * targets exactly this pool (``target_pool is pool`` — no mixing, and a
      ``target_pool`` of ``None`` is rejected),
    * has cleared the human gate (:meth:`GroundTruthCase.is_promotable` — i.e.
      approved by a named human, with filled expectations and a non-blank
      regression intent), and
    * has a case id unique within the pool.

    The held-out :attr:`Pool.TASK_SUITE` is special-cased: a ground-truth pool
    set for it must be **empty**. The Task Suite is the benchmark the optimizer
    is judged against and is never populated by this bootstrap loop; populating
    it is a separate future API (see :mod:`ail.groundtruth.promote`).
    """
    if pool is Pool.TASK_SUITE and len(cases) > 0:
        raise GroundTruthError(
            "the Task Suite is the held-out benchmark and is never populated by the "
            "ground-truth bootstrap; a TASK_SUITE pool set must be empty"
        )
    seen: set[str] = set()
    for case in cases:
        if case.target_pool is not pool:
            target = None if case.target_pool is None else case.target_pool.value
            raise GroundTruthError(
                f"case {case.case_id!r} targets pool {target!r} but would be stored in "
                f"pool {pool.value!r} (pools are never mixed; a pooled case must target it)"
            )
        blockers = case.promotion_blockers()
        if blockers:
            raise GroundTruthError(
                f"case {case.case_id!r} has not cleared the human gate and cannot be "
                f"pooled: {'; '.join(blockers)}"
            )
        if case.case_id in seen:
            raise GroundTruthError(f"duplicate case_id {case.case_id!r} in pool {pool.value!r}")
        seen.add(case.case_id)


class GroundTruthSet(_Frozen):
    """A versioned, named collection of **approved** cases for **one** pool.

    The on-disk / persisted unit of a frozen pool. Constructing one runs
    :func:`validate_pool_membership`, so a ``GroundTruthSet`` can only ever hold
    cases that target this pool *and* have cleared the human gate — there is no
    way to build (and therefore no way to persist) a pool set containing an
    unapproved candidate or a pool-mismatched case.
    """

    schema_version: str = SCHEMA_VERSION
    pool: Pool
    name: str
    created_at: str | None = None
    cases: list[GroundTruthCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_pool_membership(self) -> GroundTruthSet:
        validate_pool_membership(self.pool, self.cases)
        return self

    def case_ids(self) -> set[str]:
        """The set of case ids contained in this pool."""
        return {c.case_id for c in self.cases}
