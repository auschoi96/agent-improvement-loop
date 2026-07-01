"""Decision-rule tests: each feedback signal maps to the right action kind.

Pure functions over typed feedback + the compiled goal — no MLflow, no agent, no
warehouse. Covers item (d) of the lane-2 plan: every signal routes to its
documented action kind, and every gate (recurrence, objective-already-met, judge
trust, the goal's own threshold, dominance, regression) fail-closes when unmet.
"""

from __future__ import annotations

from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.loop.decision_rules import (
    DecisionThresholds,
    FeedbackBundle,
    JudgeDimensionSignal,
    PostApplyRegressionSignal,
    RedundantReadSignal,
    RlmAssetSignal,
    decide,
    decide_judge_dimension,
    decide_post_apply_regression,
    decide_redundant_read,
    decide_rlm_asset,
    objective_target_met,
)
from ail.loop.proposals import ActionKind, RiskClass, TriggerKind

# -- goal builders ---------------------------------------------------------


def _token_goal(*, guardrails: tuple[Guardrail, ...] = ()) -> CompiledGoal:
    """A minimize-total_tokens goal (relative −30% target), optionally with judges."""
    return CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        guardrails=guardrails,
        cohort="claude_code",
    )


def _modularity_goal() -> CompiledGoal:
    """A token goal that also guards a ``modularity`` judge with a threshold of 4.0."""
    return _token_goal(guardrails=(Guardrail(name="modularity", kind="judge", threshold=4.0),))


# -- (d) each signal -> the right action kind ------------------------------


def test_rlm_additive_asset_recurring_maps_to_metric_view() -> None:
    signal = RlmAssetSignal(
        asset_type="metric_view",
        title="token waste by tool",
        n_traces=5,
        rank=1,
        trace_ids=("t1", "t2", "t3"),
    )
    decision = decide_rlm_asset(signal, _token_goal(), objective_met=False)
    assert decision is not None
    assert decision.action_kind is ActionKind.METRIC_VIEW
    assert decision.risk_class is RiskClass.ADDITIVE_ASSET  # informational
    assert decision.trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET
    assert decision.trigger.asset_type == "metric_view"
    assert decision.trigger.trace_refs == ["t1", "t2", "t3"]


def test_redundant_read_dominant_maps_to_skill_update() -> None:
    signal = RedundantReadSignal(
        tool="read_file",
        repeated_target="config.py",
        occurrences=12,
        dominant=True,
        estimated_wasted_tokens=4000,
        trace_ids=("t1",),
    )
    decision = decide_redundant_read(signal, _token_goal())
    assert decision is not None
    assert decision.action_kind is ActionKind.SKILL_UPDATE
    assert decision.risk_class is RiskClass.AGENT_CHANGE
    assert decision.trigger.kind is TriggerKind.REDUNDANT_READ_PATTERN


def test_judge_dimension_below_threshold_and_trusted_maps_to_gepa_prompt() -> None:
    signal = JudgeDimensionSignal(
        judge_name="modularity",
        dimension="modularity",
        score=2.0,
        trusted=True,
        trace_ids=("t1", "t2"),
    )
    decision = decide_judge_dimension(signal, _modularity_goal())
    assert decision is not None
    assert decision.action_kind is ActionKind.GEPA_PROMPT
    assert decision.risk_class is RiskClass.AGENT_CHANGE
    assert decision.trigger.kind is TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD
    # the certifying judge travels on the trigger so the controller's gate can check it
    assert decision.trigger.judge_name == "modularity"
    assert decision.trigger.threshold == 4.0
    assert decision.trigger.observed_value == 2.0


def test_post_apply_regression_maps_to_revert() -> None:
    signal = PostApplyRegressionSignal(
        agent_version="v2",
        predecessor_version="v1",
        objective_metric="total_tokens",
        regressed=True,
        risk_class=RiskClass.ADDITIVE_ASSET,
    )
    decision = decide_post_apply_regression(signal)
    assert decision is not None
    assert decision.action_kind is ActionKind.REVERT
    # revert inherits the reverted change's risk class
    assert decision.risk_class is RiskClass.ADDITIVE_ASSET
    assert decision.trigger.kind is TriggerKind.POST_APPLY_REGRESSION


# -- the negative gates (fail-closed) --------------------------------------


def test_rlm_non_additive_asset_does_not_fire() -> None:
    # a skill / prompt_change recommendation is handled by the other rules, not here
    signal = RlmAssetSignal(asset_type="skill", title="x", n_traces=9)
    assert decide_rlm_asset(signal, _token_goal(), objective_met=False) is None


def test_rlm_asset_below_recurrence_floor_does_not_fire() -> None:
    th = DecisionThresholds(min_asset_recurrence_traces=5)
    signal = RlmAssetSignal(asset_type="metric_view", title="x", n_traces=4)
    assert decide_rlm_asset(signal, _token_goal(), objective_met=False, thresholds=th) is None


def test_rlm_asset_does_not_fire_when_objective_already_met() -> None:
    signal = RlmAssetSignal(asset_type="metric_view", title="x", n_traces=9)
    assert decide_rlm_asset(signal, _token_goal(), objective_met=True) is None


def test_redundant_read_not_dominant_does_not_fire() -> None:
    signal = RedundantReadSignal(
        tool="read_file", repeated_target="a", occurrences=99, dominant=False
    )
    assert decide_redundant_read(signal, _token_goal()) is None


def test_redundant_read_below_occurrence_floor_does_not_fire() -> None:
    th = DecisionThresholds(min_redundant_occurrences=10)
    signal = RedundantReadSignal(
        tool="read_file", repeated_target="a", occurrences=9, dominant=True
    )
    assert decide_redundant_read(signal, _token_goal(), thresholds=th) is None


def test_judge_dimension_distrusted_judge_does_not_fire() -> None:
    signal = JudgeDimensionSignal(
        judge_name="modularity", dimension="modularity", score=1.0, trusted=False
    )
    assert decide_judge_dimension(signal, _modularity_goal()) is None


def test_judge_dimension_above_threshold_does_not_fire() -> None:
    signal = JudgeDimensionSignal(
        judge_name="modularity", dimension="modularity", score=4.5, trusted=True
    )
    assert decide_judge_dimension(signal, _modularity_goal()) is None


def test_judge_dimension_no_goal_threshold_does_not_fire() -> None:
    # the goal names no threshold for this judge -> cannot decide (fail closed)
    signal = JudgeDimensionSignal(
        judge_name="modularity", dimension="modularity", score=0.0, trusted=True
    )
    assert decide_judge_dimension(signal, _token_goal()) is None


def test_post_apply_no_regression_does_not_fire() -> None:
    signal = PostApplyRegressionSignal(
        agent_version="v2",
        predecessor_version="v1",
        objective_metric="total_tokens",
        regressed=False,
    )
    assert decide_post_apply_regression(signal) is None


# -- objective_target_met (the goal IS the threshold) ----------------------


def test_objective_target_met_absolute_minimize() -> None:
    goal = CompiledGoal(
        objective_metric="total_usd",
        direction="minimize",
        target=GoalTarget(value=0.50, kind="absolute"),
        cohort="c",
    )
    assert objective_target_met(goal, observed=0.40) is True
    assert objective_target_met(goal, observed=0.60) is False


def test_objective_target_met_relative_needs_baseline() -> None:
    goal = _token_goal()  # relative -30%
    # baseline 1000 -> bar 700; minimize: 650 meets, 800 does not
    assert objective_target_met(goal, observed=650.0, baseline=1000.0) is True
    assert objective_target_met(goal, observed=800.0, baseline=1000.0) is False
    # no baseline for a relative target -> undecidable
    assert objective_target_met(goal, observed=650.0) is None
    # no observed value -> undecidable
    assert objective_target_met(goal, observed=None) is None


# -- decide() routes a whole feedback bundle -------------------------------


def test_decide_routes_each_signal_in_a_bundle() -> None:
    feedback = FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,  # bar 700; 900 > 700 -> objective NOT met
        rlm_assets=(RlmAssetSignal(asset_type="metric_view", title="x", n_traces=5),),
        redundant_reads=(
            RedundantReadSignal(tool="read", repeated_target="a", occurrences=8, dominant=True),
        ),
        judge_dimensions=(
            JudgeDimensionSignal(
                judge_name="modularity", dimension="modularity", score=2.0, trusted=True
            ),
        ),
        post_apply_regressions=(
            PostApplyRegressionSignal(
                agent_version="v2",
                predecessor_version="v1",
                objective_metric="total_tokens",
                regressed=True,
            ),
        ),
    )
    decisions = decide(feedback, _modularity_goal())
    kinds = [d.action_kind for d in decisions]
    assert kinds == [
        ActionKind.METRIC_VIEW,
        ActionKind.SKILL_UPDATE,
        ActionKind.GEPA_PROMPT,
        ActionKind.REVERT,
    ]
