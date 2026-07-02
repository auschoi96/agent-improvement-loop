"""Tests for the goal-derived review rubric (:mod:`ail.l3.goal_rubric`).

Pure/offline: a :class:`~ail.goals.compiler.CompiledGoal` is built directly (no LLM,
no MLflow, no HALO) and rendered into a :class:`~ail.l3.rubric.ReviewRubric`. The
prompt assertion proves the goal — not the fixed default objective — is what HALO
would be steered by (goal-derived-rubric-vs-default).
"""

from __future__ import annotations

from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.l3.goal_rubric import goal_rubric_id, render_goal_objective, rubric_from_goal
from ail.l3.reviewer import build_review_prompt
from ail.l3.rubric import DEFAULT_OBJECTIVE, DEFAULT_RUBRIC


def _goal(
    *,
    objective_metric: str = "total_tokens",
    direction: str = "minimize",
    value: float = -0.30,
    kind: str = "relative",
    guardrails: tuple[Guardrail, ...] = (),
) -> CompiledGoal:
    return CompiledGoal(
        objective_metric=objective_metric,
        direction=direction,  # type: ignore[arg-type]
        target=GoalTarget(value=value, kind=kind),  # type: ignore[arg-type]
        guardrails=guardrails,
        cohort="claude_code",
    )


class TestRenderObjective:
    def test_relative_minimize_renders_percentage_reduction(self) -> None:
        assert render_goal_objective(_goal()) == "reduce the agent's total tokens by 30%"

    def test_relative_maximize_renders_percentage_increase(self) -> None:
        goal = _goal(objective_metric="total_usd", direction="maximize", value=0.10)
        assert render_goal_objective(goal) == "increase the agent's total usd by 10%"

    def test_absolute_target_renders_level_to_reach(self) -> None:
        goal = _goal(direction="minimize", value=0.50, kind="absolute")
        assert render_goal_objective(goal) == "reduce the agent's total tokens to 0.5"

    def test_guardrails_appended_as_constraint_clause(self) -> None:
        goal = _goal(
            guardrails=(
                Guardrail(name="correctness", kind="judge", threshold=4.0),
                Guardrail(name="total_tokens", kind="metric", must_not_regress=True),
            )
        )
        objective = render_goal_objective(goal)
        assert objective.startswith("reduce the agent's total tokens by 30% while ")
        assert "holding correctness to its 4 guardrail" in objective
        assert "not regressing total tokens" in objective
        assert " and " in objective  # multiple clauses joined

    def test_must_not_regress_only_guardrail(self) -> None:
        goal = _goal(guardrails=(Guardrail(name="correctness", kind="judge", threshold=4.0),))
        # judge guardrail with threshold => "holding ... to its N guardrail"
        assert render_goal_objective(goal).endswith("while holding correctness to its 4 guardrail")


class TestGoalRubricId:
    def test_encodes_metric_and_direction(self) -> None:
        assert goal_rubric_id(_goal()) == "ail.l3.goal/total_tokens-minimize/v1"
        assert (
            goal_rubric_id(_goal(objective_metric="total_usd", direction="maximize", value=0.1))
            == "ail.l3.goal/total_usd-maximize/v1"
        )


class TestRubricFromGoal:
    def test_reuses_base_guidelines_scale_and_assets_only_reobjectives(self) -> None:
        rubric = rubric_from_goal(_goal())
        # The steering objective + id change...
        assert rubric.objective == "reduce the agent's total tokens by 30%"
        assert rubric.objective != DEFAULT_OBJECTIVE
        assert rubric.rubric_id == "ail.l3.goal/total_tokens-minimize/v1"
        # ...but the guideline set, score scale, and asset directive are inherited.
        assert rubric.guideline_ids() == DEFAULT_RUBRIC.guideline_ids()
        assert rubric.recommend_assets == DEFAULT_RUBRIC.recommend_assets
        assert (rubric.score_min, rubric.score_max) == (
            DEFAULT_RUBRIC.score_min,
            DEFAULT_RUBRIC.score_max,
        )

    def test_default_rubric_left_untouched(self) -> None:
        rubric_from_goal(_goal())
        # replace() returns a new frozen instance; the module-level default is unchanged.
        assert DEFAULT_RUBRIC.objective == DEFAULT_OBJECTIVE
        assert DEFAULT_RUBRIC.rubric_id == "ail.l3.default/v1"

    def test_prompt_is_steered_by_goal_not_default_objective(self) -> None:
        """The whole point: the goal objective — not DEFAULT_OBJECTIVE — reaches HALO."""
        goal_prompt = build_review_prompt("tr-1", rubric_from_goal(_goal()))
        default_prompt = build_review_prompt("tr-1", DEFAULT_RUBRIC)
        assert "reduce the agent's total tokens by 30%" in goal_prompt
        assert DEFAULT_OBJECTIVE not in goal_prompt
        # The default rubric's prompt carries the fixed objective (contrast).
        assert DEFAULT_OBJECTIVE in default_prompt
        assert "reduce the agent's total tokens by 30%" not in default_prompt

    def test_custom_base_rubric_is_honored(self) -> None:
        from dataclasses import replace

        base = replace(DEFAULT_RUBRIC, recommend_assets=False, score_max=10)
        rubric = rubric_from_goal(_goal(), base=base)
        assert rubric.recommend_assets is False
        assert rubric.score_max == 10
        assert rubric.objective == "reduce the agent's total tokens by 30%"
