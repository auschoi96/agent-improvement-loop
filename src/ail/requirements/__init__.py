"""Requirements intake — the loop's **front door** (slice 1, backend engine).

Where :mod:`ail.goals.compiler` maps one natural-language sentence to *one*
objective, this package turns a user's free-form requirements **blob** into a set
of evaluation **dimensions** and composes them into a single
:class:`~ail.goals.compiler.CompiledGoal`. It is the step that decides *which
judges exist* for an agent and *what to prioritize*, and it does so
**propose-then-confirm**: nothing is authored or persisted until a human confirms
the plan.

The engine is three small, injectable, offline-testable steps:

1. **Extract** (:mod:`ail.requirements.extractor`) — an injectable LLM (the same
   seam shape as :func:`ail.goals.compile_goal`) reads the requirements blob and
   returns N distinct :class:`~ail.requirements.extractor.RequirementDimension`
   records ``{name, description, user_priority, metric}``. Fail-closed:
   unparseable/empty output raises, dimensions are never fabricated.
2. **Route + compose** (:mod:`ail.requirements.composer`) — each dimension is
   routed via :func:`ail.goals.allowlist.is_l0_metric` /
   :func:`~ail.goals.allowlist.is_judge` to a deterministic **L0 metric** (no
   judge) or a **``{{ trace }}`` MemAlign judge** (:func:`ail.judges.author_judge`),
   and the priorities compose a :class:`~ail.goals.compiler.CompiledGoal` (the
   highest-priority dimension is the primary objective, the rest are guardrails).
   The result is a :class:`~ail.requirements.composer.RequirementsPlan` proposal
   that lists exactly which judges *would* be authored and which metrics are
   deterministic — and authors/persists nothing until :meth:`confirm`.
3. **Persist** (:mod:`ail.requirements.persistence`) — the *confirmed* goal is
   written to a UC Delta table the optimization loop reads (closing the
   intake→loop bridge), using the same additive ``IF NOT EXISTS`` bootstrap
   machinery the rest of the framework uses.
"""

from __future__ import annotations

from ail.requirements.composer import (
    DimensionKind,
    DimensionRole,
    PlanExecution,
    PlannedDimension,
    RequirementsNotConfirmedError,
    RequirementsPlan,
    RequirementsRoutingError,
    build_plan,
    execute_plan,
    plan_requirements,
)
from ail.requirements.extractor import (
    RequirementDimension,
    RequirementsExtractionError,
    build_extractor_system_prompt,
    extract_dimensions,
)
from ail.requirements.persistence import (
    COMPILED_GOAL_COLUMNS,
    COMPILED_GOAL_TABLE,
    compiled_goal_persister,
    load_persisted_goal,
    persist_compiled_goal,
)

__all__ = [
    # extractor
    "RequirementDimension",
    "RequirementsExtractionError",
    "build_extractor_system_prompt",
    "extract_dimensions",
    # composer
    "DimensionKind",
    "DimensionRole",
    "PlannedDimension",
    "RequirementsPlan",
    "PlanExecution",
    "RequirementsRoutingError",
    "RequirementsNotConfirmedError",
    "build_plan",
    "plan_requirements",
    "execute_plan",
    # persistence
    "COMPILED_GOAL_TABLE",
    "COMPILED_GOAL_COLUMNS",
    "persist_compiled_goal",
    "load_persisted_goal",
    "compiled_goal_persister",
]
