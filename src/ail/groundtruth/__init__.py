"""Ground-truth bootstrap — capture -> execute -> human-approve -> promote (GRP).

A human-anchored pipeline for bootstrapping the gold sets the evaluation wall
needs (see ``docs/ARCHITECTURE.md`` §2, §8). It exists to create labelled cases
*without* the co-adaptation failure mode: the expected behaviour a judge checks
against is authored by a **human**, never synthesized by a model.

Four explicit stages, each producing a new frozen
:class:`~ail.groundtruth.schema.GroundTruthCase`:

1. :func:`~ail.groundtruth.capture.capture_candidates` — derive candidate cases
   (task input + provenance) from normalized traces. No expectations.
2. :func:`~ail.groundtruth.execute.execute_candidate` — run the target agent to
   capture *its own* candidate response for the reviewer. No expectations.
3. :func:`~ail.groundtruth.approve.apply_review` — the **human gate**: a person
   fills :class:`~ail.groundtruth.schema.Expectations` and approves/rejects.
   The only place expectations are ever written.
4. :func:`~ail.groundtruth.promote.promote_approved` — a **separate** explicit
   step that persists only approved cases into a frozen, disjoint pool.

**Schema provenance:** this package's contract is clean-room original work. It
was *not* copied from SkillForge's ``GroundTruthV5`` or any ``ai-dev-kit`` GRP
code — only the conceptual contract (provenance-required, intent-stated,
human-approved) was reimplemented. See ``PROVENANCE.md``.
"""

from __future__ import annotations

from ail.groundtruth.approve import (
    ReviewError,
    apply_review,
    approve_case,
    needs_review,
    pending_cases,
    reject_case,
)
from ail.groundtruth.capture import CaptureError, candidate_from_trace, capture_candidates
from ail.groundtruth.execute import execute_candidate, log_candidate_run
from ail.groundtruth.promote import (
    PoolConflictError,
    PromotionError,
    PromotionResult,
    promote_approved,
)
from ail.groundtruth.schema import (
    SCHEMA_VERSION,
    CandidateResponse,
    Expectations,
    GroundTruthCase,
    GroundTruthError,
    GroundTruthSet,
    Pool,
    ReviewRecord,
    ReviewStatus,
    Source,
    SourceKind,
    TaskInput,
)
from ail.groundtruth.store import (
    GroundTruthStore,
    JsonGroundTruthStore,
    dump_cases,
    load_cases,
    read_review_queue,
    write_review_queue,
)

__all__ = [
    # schema
    "SCHEMA_VERSION",
    "CandidateResponse",
    "Expectations",
    "GroundTruthCase",
    "GroundTruthError",
    "GroundTruthSet",
    "Pool",
    "ReviewRecord",
    "ReviewStatus",
    "Source",
    "SourceKind",
    "TaskInput",
    # capture
    "CaptureError",
    "candidate_from_trace",
    "capture_candidates",
    # execute
    "execute_candidate",
    "log_candidate_run",
    # approve (human gate)
    "ReviewError",
    "apply_review",
    "approve_case",
    "reject_case",
    "needs_review",
    "pending_cases",
    # promote
    "PromotionError",
    "PoolConflictError",
    "PromotionResult",
    "promote_approved",
    # store
    "GroundTruthStore",
    "JsonGroundTruthStore",
    "dump_cases",
    "load_cases",
    "write_review_queue",
    "read_review_queue",
]
