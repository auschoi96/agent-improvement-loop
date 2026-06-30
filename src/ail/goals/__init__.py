"""Pillar 2: compile a natural-language goal into a validated ``CompiledGoal``.

``docs/ARCHITECTURE.md`` §4 — the loop begins with a natural-language
optimization goal that this lane turns into a structured *objective + target +
guardrails*. :func:`compile_goal` drives an injectable LLM to propose the
mapping and **strictly** validates it against the schema and the
:mod:`ail.goals.allowlist` (an unmapped metric fails loud, never silently
invented). The result is always unconfirmed — a human must
:meth:`~ail.goals.compiler.CompiledGoal.confirm` it before it drives
optimization.

The :class:`~ail.goals.compiler.CompiledGoal` satisfies the
:class:`ail.readiness.GoalView` Protocol, so the readiness module gates it
directly. This package imports the shared modules (``ail.metrics``,
``ail.judges``, ``ail.cohorts``, ``ail.readiness``) but never modifies them.
"""

from ail.goals.allowlist import (
    ALLOWLIST,
    JUDGE_METRICS,
    L0_OBJECTIVE_METRICS,
    is_judge,
    is_known_metric,
    is_l0_metric,
)
from ail.goals.compiler import (
    CompiledGoal,
    GoalCompileError,
    GoalContractError,
    GoalDirection,
    GoalProposerLLM,
    GoalTarget,
    Guardrail,
    UnmappedMetricError,
    compile_goal,
)

__all__ = [
    # allowlist
    "ALLOWLIST",
    "JUDGE_METRICS",
    "L0_OBJECTIVE_METRICS",
    "is_judge",
    "is_known_metric",
    "is_l0_metric",
    # compiler
    "CompiledGoal",
    "GoalDirection",
    "GoalTarget",
    "Guardrail",
    "GoalProposerLLM",
    "compile_goal",
    # errors
    "GoalCompileError",
    "UnmappedMetricError",
    "GoalContractError",
]
