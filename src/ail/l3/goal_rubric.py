"""Derive a goal-steered :class:`~ail.l3.rubric.ReviewRubric` from the compiled user goal.

The L3 reviewer scores a trace against a :class:`~ail.l3.rubric.ReviewRubric`, whose
``objective`` is baked into HALO's prompt: HALO is told to *"judge each guideline, and
make every recommendation, in service of that objective"*
(:func:`ail.l3.reviewer.build_review_prompt`). By default that objective is
:data:`ail.l3.rubric.DEFAULT_OBJECTIVE` (same task quality, fewer tokens, lower
latency). This module re-points it at **what the user actually asked for** — the
compiled optimization goal (``docs/ARCHITECTURE.md`` §4, :mod:`ail.goals`) — so the
review focuses on the user's objective and guardrails instead of a fixed rubric.

This is a thin *adapter* between two existing, stable pieces — it imports
:mod:`ail.goals` (the source of truth for what a goal is) and :mod:`ail.l3.rubric`
(the source of truth for what a rubric is) and modifies neither. It deliberately lives
outside :mod:`ail.l3.rubric` (which is intentionally stdlib-only) and is *not* imported
by ``ail.l3.__init__``, so ``import ail.l3`` never pulls the heavier goals/pydantic
stack; the runner and the job import it directly.

The guideline set, score scale, and asset directive are inherited unchanged from the
base rubric (the four efficiency/clarity dimensions apply to any objective) — only the
steering ``objective`` and the recorded ``rubric_id`` change, so a consumer can still
tell which rubric a verdict was scored against.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric

if TYPE_CHECKING:
    from ail.goals.compiler import CompiledGoal

__all__ = [
    "goal_rubric_id",
    "render_goal_objective",
    "rubric_from_goal",
]


def goal_rubric_id(goal: CompiledGoal) -> str:
    """Stable rubric id recorded on every verdict a goal-steered review produces.

    Encodes the objective metric + direction so a reader of the attached
    ``rlm_*`` assessments can tell the review was steered by the user's goal (and by
    which one), distinct from the default ``ail.l3.default/v1``.
    """
    return f"ail.l3.goal/{goal.objective_metric}-{goal.direction}/v1"


def _guardrail_clauses(goal: CompiledGoal) -> str:
    """Render the goal's guardrails as a natural-language constraint clause.

    Judge and deterministic-metric guardrails alike become a human-readable phrase
    so HALO weighs the constraints the user cares about (e.g. "not regressing
    correctness") when scoring and recommending. Empty when the goal has no
    guardrails.
    """
    parts: list[str] = []
    for g in goal.guardrails:
        name = g.name.replace("_", " ")
        if g.threshold is not None and g.must_not_regress:
            parts.append(f"not regressing {name} (guardrail {g.threshold:g})")
        elif g.threshold is not None:
            parts.append(f"holding {name} to its {g.threshold:g} guardrail")
        elif g.must_not_regress:
            parts.append(f"not regressing {name}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def render_goal_objective(goal: CompiledGoal) -> str:
    """Render the compiled goal as the infinitive objective phrase HALO's prompt reads.

    The result slots into both *"The objective of this review is to <objective>"* and
    *"assets ... that would let the agent <objective>"* in
    :func:`ail.l3.reviewer.build_review_prompt`, so it must read as an infinitive
    action phrase (e.g. ``reduce the agent's total tokens by 30% while not regressing
    correctness``). A relative target renders as a percentage movement; an absolute
    target as the level to reach.
    """
    metric = goal.objective_metric.replace("_", " ")
    verb = "reduce" if goal.direction == "minimize" else "increase"
    target = goal.target
    if target.kind == "relative":
        core = f"{verb} the agent's {metric} by {abs(target.value) * 100:g}%"
    else:
        core = f"{verb} the agent's {metric} to {target.value:g}"
    clauses = _guardrail_clauses(goal)
    return f"{core} while {clauses}" if clauses else core


def rubric_from_goal(goal: CompiledGoal, *, base: ReviewRubric = DEFAULT_RUBRIC) -> ReviewRubric:
    """Build a goal-steered :class:`~ail.l3.rubric.ReviewRubric` from a compiled goal.

    Inherits ``base``'s guidelines, score scale, and asset directive unchanged, and
    only re-points ``objective`` at the user's goal (:func:`render_goal_objective`)
    and stamps a goal-derived ``rubric_id`` (:func:`goal_rubric_id`). Callers that
    have no goal keep passing :data:`ail.l3.rubric.DEFAULT_RUBRIC` — this is the
    opt-in, goal-present path; ``DEFAULT_RUBRIC`` stays the fallback.
    """
    return replace(
        base,
        rubric_id=goal_rubric_id(goal),
        objective=render_goal_objective(goal),
    )
