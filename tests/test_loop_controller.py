"""Loop-controller tests — fail-closed, propose-only, with all seams faked.

No live MLflow / agent / warehouse: the feedback source, candidate builder,
prover, and gate are all injected fakes. Covers the lane-2 plan items:

* (a) a proven + gated candidate → exactly one pending proposal with the right
  why / proof / gate payload;
* (b) a non-improving candidate (no PROMOTE, or a correctness regression) →
  ZERO proposals (fail-closed);
* (c) an ungated state (readiness not met, or a distrusted certifying judge) →
  ZERO proposals;
* (e) ``risk_class`` is set but never causes an auto-apply — the controller calls
  no apply/alias function (there is none), and both an additive-asset and an
  agent-change proposal come out merely *pending*;
* (f) the publish step writes agent-scoped rows.
"""

from __future__ import annotations

import json

import pytest
from databricks.sdk.service.sql import StatementState

from ail.compare.contract import Recommendation
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.loop.controller import Candidate, CycleResult, evaluate_gate, run_cycle
from ail.loop.decision_rules import (
    FeedbackBundle,
    JudgeDimensionSignal,
    RlmAssetSignal,
)
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProposalStatus,
    ProposedChange,
    RiskClass,
    TriggerKind,
)
from ail.loop.publish_proposals import (
    PROPOSAL_COLUMNS,
    publish_agent_proposals,
    publish_proposals,
)
from ail.optimize.phase2 import L1Outcome, Phase2Artifact, TaskOutcome
from ail.readiness.contract import (
    EvalHealth,
    Gate,
    GateName,
    JudgeHealth,
    ReadinessStatus,
    ReadinessTier,
)
from ail.registry import Agent

# -- fixtures / builders ---------------------------------------------------


def _agent() -> Agent:
    return Agent(agent_name="claude_code", experiment_id="660599403165942")


def _goal() -> CompiledGoal:
    """A confirmed minimize-tokens goal that also guards a trusted ``modularity`` judge."""
    return CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        guardrails=(Guardrail(name="modularity", kind="judge", threshold=4.0),),
        cohort="claude_code",
    ).confirm()


def _proven_artifact(*, regressed: bool = False, n_promote: int = 3) -> Phase2Artifact:
    outcomes = [
        TaskOutcome(
            task_id=f"p{i}", recommendation=Recommendation.PROMOTE, l1_outcome=L1Outcome.PASSED
        )
        for i in range(n_promote)
    ]
    if regressed:
        outcomes.append(
            TaskOutcome(
                task_id="reg", recommendation=Recommendation.BLOCK, l1_outcome=L1Outcome.REGRESSED
            )
        )
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=n_promote + (1 if regressed else 0),
        n_promote=n_promote,
        realized_token_savings_absolute=1200.0,
        realized_token_savings_pct=35.4,
        outcomes=outcomes,
    )


def _blocked_artifact() -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=4,
        n_promote=0,
        n_block=4,
        outcomes=[
            TaskOutcome(task_id=f"b{i}", recommendation=Recommendation.BLOCK) for i in range(4)
        ],
    )


def _ready(*, modularity_distrusted: bool = False) -> ReadinessStatus:
    judges = [
        JudgeHealth(
            judge_name="modularity",
            measured=not modularity_distrusted,
            agreement_rate=None if modularity_distrusted else 0.82,
            distrusted=modularity_distrusted,
            reason="unmeasured" if modularity_distrusted else "trusted",
        )
    ]
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
            n_distrusted_judges=1 if modularity_distrusted else 0,
            distrusted_judges=["modularity"] if modularity_distrusted else [],
            judges=judges,
        ),
    )


def _not_ready() -> ReadinessStatus:
    return ReadinessStatus(
        cohort_name="claude_code",
        objective_metric="total_tokens",
        trace_count=5,
        tier=ReadinessTier.BASELINE_ONLY,
        gates=[Gate(name=GateName.TRACE_PROVE, passed=False, reason="need 45 more traces")],
        reasons=["need 45 more traces"],
        eval_health=EvalHealth(cohort_name="claude_code"),
    )


def _build_candidate(decision, *, goal, agent) -> Candidate:  # type: ignore[no-untyped-def]
    """A candidate builder that produces a matching change per action kind."""
    ak = decision.action_kind
    if ak is ActionKind.METRIC_VIEW:
        change = ProposedChange(
            kind=ChangeKind.METRIC_VIEW_SQL,
            summary="token waste view",
            sql="CREATE OR REPLACE VIEW c.s.v WITH METRICS LANGUAGE YAML AS $$ ... $$",
        )
    elif ak is ActionKind.SKILL_UPDATE:
        change = ProposedChange(
            kind=ChangeKind.SKILL_DIFF, summary="read-cache skill", diff="--- a\n+++ b"
        )
    elif ak is ActionKind.GEPA_PROMPT:
        change = ProposedChange(
            kind=ChangeKind.EVOLVED_BODY_REF,
            summary="gepa-evolved skill",
            evolved_body_ref="prompts:/c.s.p/4",
        )
    elif ak is ActionKind.INSTRUCTION_UPDATE:
        change = ProposedChange(kind=ChangeKind.INSTRUCTION_DIFF, summary="instr", diff="--- a")
    else:  # REVERT
        change = ProposedChange(kind=ChangeKind.REVERT_REF, summary="revert", revert_target="v1")
    return Candidate(change=change, prover_input=ak.value)


def _feedback_metric_view() -> FeedbackBundle:
    return FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,  # bar 700; 900 > 700 -> objective not met
        rlm_assets=(
            RlmAssetSignal(
                asset_type="metric_view",
                title="token waste by tool",
                n_traces=5,
                rank=1,
                trace_ids=("t1", "t2", "t3"),
            ),
        ),
    )


def _feedback_judge() -> FeedbackBundle:
    return FeedbackBundle(
        judge_dimensions=(
            JudgeDimensionSignal(
                judge_name="modularity",
                dimension="modularity",
                score=2.0,
                trusted=True,
                trace_ids=("t9",),
            ),
        ),
    )


def _run(
    *,
    feedback: FeedbackBundle,
    artifact: Phase2Artifact,
    readiness: ReadinessStatus,
    builder=_build_candidate,  # type: ignore[no-untyped-def]
) -> CycleResult:
    return run_cycle(
        _agent(),
        _goal(),
        feedback_source=lambda: feedback,
        candidate_builder=builder,
        prover=lambda candidate, *, goal, agent: artifact,
        gate=lambda *, goal, agent: readiness,
        now="2026-06-30T00:00:00+00:00",
    )


# -- (a) a proven + gated candidate -> exactly one pending proposal --------


def test_proven_and_gated_emits_one_pending_proposal() -> None:
    result = _run(feedback=_feedback_metric_view(), artifact=_proven_artifact(), readiness=_ready())
    assert len(result.proposals) == 1
    assert result.skipped == ()
    p = result.proposals[0]
    assert p.status is ProposalStatus.PENDING
    assert p.agent_name == "claude_code"
    assert p.action_kind is ActionKind.METRIC_VIEW
    assert p.risk_class is RiskClass.ADDITIVE_ASSET
    # why
    assert p.trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET
    assert p.trigger.trace_refs == ["t1", "t2", "t3"]
    # what
    assert p.change.kind is ChangeKind.METRIC_VIEW_SQL
    assert p.change.sql is not None
    # proof
    assert p.proof.proved_improvement is True
    assert p.proof.correctness_held is True
    assert p.proof.realized_savings_pct == 35.4
    assert p.proof.suite_content_hash == "deadbeef"
    # gate
    assert p.gate_status.gated is True
    assert p.gate_status.readiness_tier == "ready_to_prove"
    assert p.created_at == "2026-06-30T00:00:00+00:00"


def test_judge_signal_proven_and_gated_emits_gepa_prompt_proposal() -> None:
    result = _run(feedback=_feedback_judge(), artifact=_proven_artifact(), readiness=_ready())
    assert len(result.proposals) == 1
    p = result.proposals[0]
    assert p.action_kind is ActionKind.GEPA_PROMPT
    assert p.risk_class is RiskClass.AGENT_CHANGE
    assert p.trigger.judge_name == "modularity"
    # the certifying judge's agreement is surfaced on the gate payload
    assert p.gate_status.judge_agreement == 0.82


# -- (b) a non-improving candidate -> ZERO proposals (fail-closed) ---------


def test_no_promote_yields_zero_proposals() -> None:
    result = _run(
        feedback=_feedback_metric_view(), artifact=_blocked_artifact(), readiness=_ready()
    )
    assert result.proposals == ()
    assert len(result.skipped) == 1
    assert "not proven" in result.skipped[0].reason


def test_correctness_regression_yields_zero_proposals() -> None:
    result = _run(
        feedback=_feedback_metric_view(),
        artifact=_proven_artifact(regressed=True),
        readiness=_ready(),
    )
    assert result.proposals == ()
    assert "correctness_held=False" in result.skipped[0].reason


# -- (c) an ungated state -> ZERO proposals --------------------------------


def test_readiness_not_met_yields_zero_proposals() -> None:
    result = _run(
        feedback=_feedback_metric_view(), artifact=_proven_artifact(), readiness=_not_ready()
    )
    assert result.proposals == ()
    assert "readiness not met" in result.skipped[0].reason


def test_distrusted_certifying_judge_yields_zero_proposals() -> None:
    # readiness tier is READY_TO_PROVE, but the trigger's certifying judge is distrusted
    result = _run(
        feedback=_feedback_judge(),
        artifact=_proven_artifact(),
        readiness=_ready(modularity_distrusted=True),
    )
    assert result.proposals == ()
    assert "distrusted" in result.skipped[0].reason


# -- candidate builder produced nothing -> skip (fail-closed) --------------


def test_candidate_builder_returning_none_yields_zero_proposals() -> None:
    result = _run(
        feedback=_feedback_metric_view(),
        artifact=_proven_artifact(),
        readiness=_ready(),
        builder=lambda decision, *, goal, agent: None,
    )
    assert result.proposals == ()
    assert "no candidate" in result.skipped[0].reason


# -- per-decision fault isolation: one failure does not torpedo the others -


def test_one_decision_error_does_not_drop_the_others() -> None:
    # Two decisions this cycle: a metric_view (additive) and a gepa_prompt (judge),
    # both of which would prove + gate. The prover raises ONLY for the gepa_prompt
    # candidate (e.g. an MLflow/network timeout in the frozen-suite run). The earlier,
    # already-proven metric_view proposal must survive; the failing decision must be a
    # fail-closed skip carrying the error (never a proposal, never a crashed cycle).
    feedback = FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,
        rlm_assets=(RlmAssetSignal(asset_type="metric_view", title="v", n_traces=5),),
        judge_dimensions=(
            JudgeDimensionSignal(
                judge_name="modularity", dimension="modularity", score=2.0, trusted=True
            ),
        ),
    )

    def _flaky_prover(candidate, *, goal, agent):  # type: ignore[no-untyped-def]
        if candidate.prover_input == ActionKind.GEPA_PROMPT.value:
            raise RuntimeError("frozen-suite prove timed out")
        return _proven_artifact()

    result = run_cycle(
        _agent(),
        _goal(),
        feedback_source=lambda: feedback,
        candidate_builder=_build_candidate,
        prover=_flaky_prover,
        gate=lambda *, goal, agent: _ready(),
        now="2026-06-30T00:00:00+00:00",
    )

    # the metric_view decision still produced its proposal despite the later failure
    assert [p.action_kind for p in result.proposals] == [ActionKind.METRIC_VIEW]
    # the failing gepa_prompt decision is a fail-closed skip carrying the error
    errored = [s for s in result.skipped if s.action_kind == ActionKind.GEPA_PROMPT.value]
    assert len(errored) == 1
    assert "errored" in errored[0].reason
    assert "RuntimeError" in errored[0].reason
    assert "frozen-suite prove timed out" in errored[0].reason


# -- the controller refuses an unconfirmed goal ----------------------------


def test_run_cycle_refuses_unconfirmed_goal() -> None:
    unconfirmed = CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        cohort="claude_code",
    )
    assert unconfirmed.human_confirmed is False
    with pytest.raises(ValueError, match="unconfirmed goal"):
        run_cycle(
            _agent(),
            unconfirmed,
            feedback_source=lambda: _feedback_metric_view(),
            candidate_builder=_build_candidate,
            prover=lambda candidate, *, goal, agent: _proven_artifact(),
            gate=lambda *, goal, agent: _ready(),
        )


# -- (e) risk_class is informational; the controller never auto-applies ----


def test_risk_class_set_but_no_auto_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tripwire: if the controller ever reached into the apply capabilities, these fire.
    import ail.optimize.prompt_registry as pr

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("the controller must never apply a change (register/alias)")

    monkeypatch.setattr(pr, "register_prompt_body", _boom)
    monkeypatch.setattr(pr, "register_gepa_candidate", _boom)

    # both an additive-asset (metric_view) and an agent-change (gepa_prompt) signal,
    # both proven + gated -> two pending proposals, neither applied.
    feedback = FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,
        rlm_assets=(RlmAssetSignal(asset_type="metric_view", title="v", n_traces=5),),
        judge_dimensions=(
            JudgeDimensionSignal(
                judge_name="modularity", dimension="modularity", score=2.0, trusted=True
            ),
        ),
    )
    result = _run(feedback=feedback, artifact=_proven_artifact(), readiness=_ready())

    by_kind = {p.action_kind: p for p in result.proposals}
    assert set(by_kind) == {ActionKind.METRIC_VIEW, ActionKind.GEPA_PROMPT}
    assert by_kind[ActionKind.METRIC_VIEW].risk_class is RiskClass.ADDITIVE_ASSET
    assert by_kind[ActionKind.GEPA_PROMPT].risk_class is RiskClass.AGENT_CHANGE
    # the risk class did NOT change the outcome: both are merely pending
    assert all(p.status is ProposalStatus.PENDING for p in result.proposals)
    # and there is no apply seam on the controller at all
    import ail.loop.controller as controller

    assert not hasattr(controller, "apply")
    assert "apply" not in run_cycle.__code__.co_varnames


# -- evaluate_gate unit ----------------------------------------------------


def test_evaluate_gate_readiness_and_judge() -> None:
    gated, reasons = evaluate_gate(_ready())
    assert gated is True and reasons == []

    gated, reasons = evaluate_gate(_not_ready())
    assert gated is False and "readiness not met" in reasons[0]

    gated, reasons = evaluate_gate(_ready(modularity_distrusted=True), judge_name="modularity")
    assert gated is False and "distrusted" in reasons[0]

    # a judge with no measurement in the cohort cannot certify either
    gated, reasons = evaluate_gate(_ready(), judge_name="security")
    assert gated is False and "no measurement" in reasons[0]


# -- (f) the publish step writes agent-scoped rows -------------------------


class _FakeStatus:
    def __init__(self, state: StatementState) -> None:
        self.state = state
        self.error = None


class _FakeResp:
    def __init__(self, state: StatementState) -> None:
        self.statement_id = "stmt-1"
        self.status = _FakeStatus(state)


class _FakeStatementExecution:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_statement(self, *, warehouse_id, statement, wait_timeout=None):  # type: ignore[no-untyped-def]
        self.statements.append(statement)
        return _FakeResp(StatementState.SUCCEEDED)

    def get_statement(self, statement_id):  # type: ignore[no-untyped-def]
        return _FakeResp(StatementState.SUCCEEDED)


class _FakeClient:
    def __init__(self) -> None:
        self.statement_execution = _FakeStatementExecution()


def _two_proposals() -> list:  # type: ignore[type-arg]
    proposals = list(
        _run(
            feedback=_feedback_metric_view(), artifact=_proven_artifact(), readiness=_ready()
        ).proposals
    )
    proposals += list(
        _run(feedback=_feedback_judge(), artifact=_proven_artifact(), readiness=_ready()).proposals
    )
    assert len(proposals) == 2
    return proposals


def test_publish_agent_proposals_appends_without_replacing_queue() -> None:
    proposals = _two_proposals()
    client = _FakeClient()
    n = publish_agent_proposals(
        proposals,
        agent_name="claude_code",
        experiment_id="660599403165942",
        client=client,
        warehouse_id="wh",
        catalog="cat",
        schema="sch",
    )
    assert n == 2
    stmts = client.statement_execution.statements
    assert not any("REPLACE WHERE" in statement for statement in stmts)
    inserts = [statement for statement in stmts if "INSERT INTO" in statement]
    assert len(inserts) == 2
    staged = "\n".join(inserts)
    for p in proposals:
        assert p.proposal_id in staged
    assert all("WHERE NOT EXISTS" in statement for statement in inserts)


def test_publish_agent_proposals_rejects_mixed_agents() -> None:
    proposals = _two_proposals()
    other = proposals[0].model_copy(update={"agent_name": "other_agent"})
    client = _FakeClient()
    with pytest.raises(ValueError, match="scoped to agent 'claude_code'"):
        publish_agent_proposals(
            [*proposals, other],
            agent_name="claude_code",
            experiment_id="660599403165942",
            client=client,
            warehouse_id="wh",
        )


def test_publish_proposals_appends_each_agent_independently() -> None:
    a = _two_proposals()[0]  # claude_code
    b = a.model_copy(update={"agent_name": "other_agent"})
    client = _FakeClient()
    written = publish_proposals(
        [a, b], warehouse_id="wh", catalog="cat", schema="sch", client=client
    )
    assert written == {"claude_code": 1, "other_agent": 1}
    inserts = [s for s in client.statement_execution.statements if "INSERT INTO" in s]
    assert any("agent_name = 'claude_code'" in s for s in inserts)
    assert any("agent_name = 'other_agent'" in s for s in inserts)
    assert not any("REPLACE WHERE" in s for s in client.statement_execution.statements)


def test_proposal_row_matches_column_order() -> None:
    from ail.loop.publish_proposals import _proposal_row

    p = _two_proposals()[0]
    row = _proposal_row(p, generated_at="2026-06-30T00:00:00+00:00")
    assert len(row) == len(PROPOSAL_COLUMNS)
    # the JSON-encoded trace refs round-trip
    refs_idx = PROPOSAL_COLUMNS.index("trigger_trace_refs")
    assert json.loads(row[refs_idx]) == p.trigger.trace_refs
