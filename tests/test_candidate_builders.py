"""Tests for the first real candidate→prove path: the token-efficiency skill install.

Covers the builder's guarantees with no live MLflow / agent / warehouse:

* a token-efficiency ``SKILL_UPDATE`` decision (+ no pending proposal, + a
  token-reduction goal) → a :class:`Candidate` carrying the *proven*
  :func:`ail.optimize.lever.token_efficiency_intervention`;
* the **cost guard**: an already-pending target, an already-built-this-cycle target,
  and an unavailable pending-check each yield ``None`` (fail-closed toward not
  spending the expensive frozen-suite proof);
* fail-closed ``None`` for every unwired action kind / non-token-reduction goal;
* end-to-end through the unchanged controller pipeline (:func:`ail.loop.controller.run_cycle`)
  with a mocked prover: a proven artifact + a passing gate → exactly one PENDING
  proposal; a not-proven artifact → zero; no waste signal → zero (Lane A emits no
  decision to consume);
* the entrypoint's pending-proposal fetch fails closed (``None``) on a read error.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ail.compare.contract import Recommendation
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.jobs import optimization_cycle as oc
from ail.loop.candidate_builders import (
    is_token_reduction_goal,
    token_efficiency_candidate_builder,
    token_efficiency_skill_change,
)
from ail.loop.controller import Candidate, run_cycle
from ail.loop.decision_rules import Decision, FeedbackBundle, RedundantReadSignal
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProposalStatus,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
    derive_proposal_id,
)
from ail.optimize.lever import SkillInjectionIntervention, token_efficiency_intervention
from ail.optimize.phase2 import L1Outcome, Phase2Artifact, TaskOutcome
from ail.publish import DEFAULT_CATALOG, DEFAULT_SCHEMA
from ail.readiness.contract import (
    EvalHealth,
    Gate,
    GateName,
    JudgeHealth,
    ReadinessStatus,
    ReadinessTier,
)
from ail.registry import Agent

# -- fixtures --------------------------------------------------------------


def _agent() -> Agent:
    return Agent(agent_name="claude_code", experiment_id="660599403165942")


def _token_goal() -> CompiledGoal:
    return CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        guardrails=(Guardrail(name="modularity", kind="judge", threshold=4.0),),
        cohort="claude_code",
    ).confirm()


def _skill_decision() -> Decision:
    """A Lane-A redundant-read → SKILL_UPDATE decision (the token-efficiency decision)."""
    return Decision(
        ActionKind.SKILL_UPDATE,
        default_risk_class(ActionKind.SKILL_UPDATE),
        TriggerSignal(
            kind=TriggerKind.REDUNDANT_READ_PATTERN,
            summary="redundant reads dominate the L0 waste diagnosis",
            metric="total_tokens",
        ),
    )


def _proposal_id_for(agent: Agent) -> str:
    return derive_proposal_id(
        agent_name=agent.agent_name,
        action_kind=ActionKind.SKILL_UPDATE,
        change=token_efficiency_skill_change(),
    )


def _proven_artifact() -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=3,
        n_promote=3,
        realized_token_savings_absolute=1200.0,
        realized_token_savings_pct=35.4,
        outcomes=[
            TaskOutcome(
                task_id=f"p{i}", recommendation=Recommendation.PROMOTE, l1_outcome=L1Outcome.PASSED
            )
            for i in range(3)
        ],
    )


def _blocked_artifact() -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=2,
        n_promote=0,
        n_block=2,
        outcomes=[TaskOutcome(task_id="b", recommendation=Recommendation.BLOCK)],
    )


def _ready() -> ReadinessStatus:
    return ReadinessStatus(
        cohort_name="claude_code",
        objective_metric="total_tokens",
        requires_quality=True,
        guardrail_names=["modularity"],
        trace_count=80,
        tier=ReadinessTier.READY_TO_PROVE,
        gates=[Gate(name=GateName.TRACE_PROVE, passed=True, reason="enough traces")],
        reasons=[],
        eval_health=EvalHealth(
            cohort_name="claude_code",
            scored_coverage=0.9,
            judges=[
                JudgeHealth(
                    judge_name="modularity",
                    measured=True,
                    agreement_rate=0.82,
                    distrusted=False,
                    reason="trusted",
                )
            ],
        ),
    )


# ==========================================================================
# The builder: token-efficiency skill install
# ==========================================================================


def test_builds_token_efficiency_candidate_when_not_pending() -> None:
    agent = _agent()
    build = token_efficiency_candidate_builder(pending_proposal_ids=frozenset())
    candidate = build(_skill_decision(), goal=_token_goal(), agent=agent)

    assert candidate is not None
    # the prover_input is the proven token-efficiency SkillInjectionIntervention
    assert isinstance(candidate.prover_input, SkillInjectionIntervention)
    assert candidate.prover_input.skill == token_efficiency_intervention().skill
    # the change installs the skill (matching kind the lane-3 apply engine expects)
    assert candidate.change.kind is ChangeKind.SKILL_DIFF
    assert candidate.change.diff.strip()
    # id the controller would assign matches what the cost guard keys on
    assert derive_proposal_id(
        agent_name=agent.agent_name, action_kind=ActionKind.SKILL_UPDATE, change=candidate.change
    ) == _proposal_id_for(agent)


def test_cost_guard_skips_already_pending_target() -> None:
    agent = _agent()
    pending = frozenset({_proposal_id_for(agent)})
    build = token_efficiency_candidate_builder(pending_proposal_ids=pending)
    assert build(_skill_decision(), goal=_token_goal(), agent=agent) is None


def test_cost_guard_skips_when_pending_check_unavailable() -> None:
    # Fail-closed toward NOT spending: the pending check could not run (None).
    build = token_efficiency_candidate_builder(pending_proposal_ids=None)
    assert build(_skill_decision(), goal=_token_goal(), agent=_agent()) is None


def test_cost_guard_skips_second_build_in_same_cycle() -> None:
    # Two SKILL_UPDATE decisions in one cycle collapse to one proof: the second build
    # is skipped (the id was already built this cycle) so the proof runs at most once.
    agent = _agent()
    build = token_efficiency_candidate_builder(pending_proposal_ids=frozenset())
    first = build(_skill_decision(), goal=_token_goal(), agent=agent)
    second = build(_skill_decision(), goal=_token_goal(), agent=agent)
    assert first is not None
    assert second is None


def test_none_for_unwired_action_kinds() -> None:
    agent = _agent()
    goal = _token_goal()
    build = token_efficiency_candidate_builder(pending_proposal_ids=frozenset())
    for ak, trigger_kind in [
        (ActionKind.METRIC_VIEW, TriggerKind.RLM_RECOMMENDED_ASSET),
        (ActionKind.GEPA_PROMPT, TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD),
        (ActionKind.INSTRUCTION_UPDATE, TriggerKind.AGENT_PLANNER),
        (ActionKind.REVERT, TriggerKind.POST_APPLY_REGRESSION),
    ]:
        decision = Decision(
            ak, default_risk_class(ak), TriggerSignal(kind=trigger_kind, summary="x")
        )
        assert build(decision, goal=goal, agent=agent) is None


def test_none_for_skill_update_with_non_redundant_read_trigger() -> None:
    # A SKILL_UPDATE not triggered by the genuine redundant-read waste signal must NOT
    # be hijacked into the token-efficiency skill: the intervention cannot faithfully
    # prove a change the triggering evidence did not call for (e.g. a Lane B planner
    # proposal intending some *other* skill). Fail-closed -> None.
    agent = _agent()
    goal = _token_goal()
    build = token_efficiency_candidate_builder(pending_proposal_ids=frozenset())
    for trigger_kind in [
        TriggerKind.AGENT_PLANNER,
        TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD,
        TriggerKind.RLM_RECOMMENDED_ASSET,
    ]:
        decision = Decision(
            ActionKind.SKILL_UPDATE,
            default_risk_class(ActionKind.SKILL_UPDATE),
            TriggerSignal(kind=trigger_kind, summary="a different skill", metric="total_tokens"),
        )
        assert build(decision, goal=goal, agent=agent) is None
    # sanity: the genuine redundant-read-triggered decision still builds
    assert build(_skill_decision(), goal=goal, agent=agent) is not None


def test_none_for_non_token_reduction_goal() -> None:
    # The skill is proven on total_tokens; a goal it does not speak to → no candidate.
    agent = _agent()
    build = token_efficiency_candidate_builder(pending_proposal_ids=frozenset())
    wrong_direction = CompiledGoal(
        objective_metric="total_tokens",
        direction="maximize",
        target=GoalTarget(value=0.30, kind="relative"),
        cohort="claude_code",
    ).confirm()
    other_metric = CompiledGoal(
        objective_metric="total_usd",
        direction="minimize",
        target=GoalTarget(value=-0.10, kind="relative"),
        cohort="claude_code",
    ).confirm()
    assert not is_token_reduction_goal(wrong_direction)
    assert not is_token_reduction_goal(other_metric)
    assert build(_skill_decision(), goal=wrong_direction, agent=agent) is None
    assert build(_skill_decision(), goal=other_metric, agent=agent) is None


def test_token_efficiency_change_is_deterministic() -> None:
    # Stability underpins the cost guard: the same install → the same id across cycles.
    assert token_efficiency_skill_change().diff == token_efficiency_skill_change().diff
    a, b = _agent(), _agent()
    assert _proposal_id_for(a) == _proposal_id_for(b)


# ==========================================================================
# End-to-end through the unchanged controller pipeline (mocked prover + gate)
# ==========================================================================


def _feedback_with_waste() -> FeedbackBundle:
    # A dominant, recurring redundant-read pattern → Lane A decide() emits SKILL_UPDATE.
    return FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,
        redundant_reads=(
            RedundantReadSignal(
                tool="Read",
                repeated_target="/x",
                occurrences=5,
                dominant=True,
                trace_ids=("t1", "t2"),
            ),
        ),
    )


def test_end_to_end_proven_and_gated_emits_one_pending_proposal() -> None:
    agent = _agent()
    seen_prover_inputs: list[Any] = []

    def _prover(candidate: Candidate, *, goal: CompiledGoal, agent: Agent) -> Phase2Artifact:
        seen_prover_inputs.append(candidate.prover_input)
        return _proven_artifact()

    result = run_cycle(
        agent,
        _token_goal(),
        feedback_source=_feedback_with_waste,
        candidate_builder=token_efficiency_candidate_builder(pending_proposal_ids=frozenset()),
        prover=_prover,
        gate=lambda *, goal, agent: _ready(),
        now="2026-07-01T00:00:00+00:00",
    )

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.action_kind is ActionKind.SKILL_UPDATE
    assert proposal.change.kind is ChangeKind.SKILL_DIFF
    assert proposal.proof.proved_improvement and proposal.proof.correctness_held
    assert proposal.trigger.kind is TriggerKind.REDUNDANT_READ_PATTERN
    # the proof actually ran over the token-efficiency intervention (not a fabrication)
    assert len(seen_prover_inputs) == 1
    assert isinstance(seen_prover_inputs[0], SkillInjectionIntervention)
    assert seen_prover_inputs[0].skill == token_efficiency_intervention().skill


def test_end_to_end_not_proven_emits_zero_proposals() -> None:
    result = run_cycle(
        _agent(),
        _token_goal(),
        feedback_source=_feedback_with_waste,
        candidate_builder=token_efficiency_candidate_builder(pending_proposal_ids=frozenset()),
        prover=lambda c, *, goal, agent: _blocked_artifact(),
        gate=lambda *, goal, agent: _ready(),
        now="2026-07-01T00:00:00+00:00",
    )
    assert result.proposals == ()
    assert any("not proven" in s.reason for s in result.skipped)


def test_end_to_end_no_waste_signal_emits_zero_proposals() -> None:
    # No redundant-read waste → Lane A decide() emits no SKILL_UPDATE → nothing to build.
    calls: list[Candidate] = []

    result = run_cycle(
        _agent(),
        _token_goal(),
        feedback_source=lambda: FeedbackBundle(
            objective_metric_value=900.0, objective_baseline_value=1000.0
        ),
        candidate_builder=token_efficiency_candidate_builder(pending_proposal_ids=frozenset()),
        prover=lambda c, *, goal, agent: calls.append(c) or _proven_artifact(),
        gate=lambda *, goal, agent: _ready(),
        now="2026-07-01T00:00:00+00:00",
    )
    assert result.proposals == ()
    assert result.skipped == ()  # no decision fired at all
    assert calls == []  # the expensive prover was never invoked


# ==========================================================================
# Entrypoint cost-guard input: fetch pending proposal ids (fail-closed)
# ==========================================================================


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        catalog=DEFAULT_CATALOG, schema=DEFAULT_SCHEMA, profile=None, warehouse_id="wh1"
    )


def test_fetch_pending_proposal_ids_returns_ids(monkeypatch: Any) -> None:
    import ail.loop.apply_service as apply_service
    import ail.publish as publish

    monkeypatch.setattr(publish, "_build_workspace_client", lambda profile: object())
    monkeypatch.setattr(
        apply_service,
        "_query_rows",
        lambda client, wh, sql: [
            {"proposal_id": "abc"},
            {"proposal_id": "def"},
            {"proposal_id": None},  # dropped
        ],
    )
    ids = oc._fetch_pending_proposal_ids(_agent(), _args())
    assert ids == frozenset({"abc", "def"})


def test_fetch_pending_proposal_ids_fails_closed_to_none(monkeypatch: Any) -> None:
    import ail.loop.apply_service as apply_service
    import ail.publish as publish

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("warehouse unreachable")

    monkeypatch.setattr(publish, "_build_workspace_client", lambda profile: object())
    monkeypatch.setattr(apply_service, "_query_rows", _boom)
    # A read failure → None (the fail-closed "unavailable, do not spend" sentinel).
    assert oc._fetch_pending_proposal_ids(_agent(), _args()) is None
