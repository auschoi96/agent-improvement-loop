"""Proposed-action model tests: typed, inert, fail-closed proof extraction.

No MLflow / agent / warehouse. Covers the proposal record itself: the
action↔change validators, the proof extracted from a frozen-suite artifact
(including the fail-closed regression / no-PROMOTE cases), the gate-status
projection, deterministic ids, and JSON round-tripping.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ail.compare.contract import Recommendation
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    ProofSummary,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    RiskClass,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
    derive_proposal_id,
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

# -- builders --------------------------------------------------------------


def _artifact(
    *,
    n_promote: int,
    n_block: int = 0,
    n_errored: int = 0,
    regressed: bool = False,
    savings_pct: float | None = 12.5,
    savings_abs: float = 1000.0,
    objective: str = "total_tokens",
    suite_hash: str = "deadbeef",
) -> Phase2Artifact:
    outcomes: list[TaskOutcome] = []
    for i in range(n_promote):
        outcomes.append(
            TaskOutcome(
                task_id=f"p{i}",
                recommendation=Recommendation.PROMOTE,
                l1_outcome=L1Outcome.PASSED,
            )
        )
    if regressed:
        outcomes.append(
            TaskOutcome(
                task_id="reg",
                recommendation=Recommendation.BLOCK,
                l1_outcome=L1Outcome.REGRESSED,
            )
        )
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash=suite_hash,
        objective_metric=objective,
        n_tasks=n_promote + n_block + n_errored + (1 if regressed else 0),
        n_promote=n_promote,
        n_block=n_block,
        n_errored=n_errored,
        realized_token_savings_absolute=savings_abs,
        realized_token_savings_pct=savings_pct,
        outcomes=outcomes,
    )


def _readiness(
    *,
    tier: ReadinessTier,
    judges: list[JudgeHealth] | None = None,
    scored_coverage: float = 0.8,
) -> ReadinessStatus:
    judges = judges or []
    distrusted = [j.judge_name for j in judges if j.distrusted]
    return ReadinessStatus(
        cohort_name="claude_code",
        objective_metric="total_tokens",
        tier=tier,
        gates=[
            Gate(name=GateName.TRACE_PROVE, passed=tier == ReadinessTier.READY_TO_PROVE, reason="")
        ],
        reasons=[] if tier == ReadinessTier.READY_TO_PROVE else ["need more traces"],
        eval_health=EvalHealth(
            cohort_name="claude_code",
            scored_coverage=scored_coverage,
            n_distrusted_judges=len(distrusted),
            distrusted_judges=distrusted,
            judges=judges,
        ),
    )


def _metric_view_change() -> ProposedChange:
    return ProposedChange(
        kind=ChangeKind.METRIC_VIEW_SQL,
        summary="token waste by tool",
        sql="CREATE OR REPLACE VIEW c.s.v WITH METRICS LANGUAGE YAML AS $$ ... $$",
    )


# -- ProposedChange / ProposedAction validators ----------------------------


def test_change_requires_its_payload() -> None:
    with pytest.raises(ValidationError, match="must set a non-empty 'sql'"):
        ProposedChange(kind=ChangeKind.METRIC_VIEW_SQL, summary="x")
    with pytest.raises(ValidationError, match="must set a non-empty 'evolved_body_ref'"):
        ProposedChange(kind=ChangeKind.EVOLVED_BODY_REF, summary="x")


def test_action_must_match_change_kind() -> None:
    # a metric_view action carrying a skill diff is rejected
    with pytest.raises(ValidationError, match="requires a change of kind 'metric_view_sql'"):
        ProposedAction(
            proposal_id="x",
            agent_name="claude_code",
            action_kind=ActionKind.METRIC_VIEW,
            risk_class=RiskClass.ADDITIVE_ASSET,
            objective_metric="total_tokens",
            goal_cohort="claude_code",
            trigger=TriggerSignal(kind=TriggerKind.RLM_RECOMMENDED_ASSET, summary="x"),
            change=ProposedChange(kind=ChangeKind.SKILL_DIFF, summary="x", diff="--- a"),
            proof=ProofSummary(objective_metric="total_tokens"),
            gate_status=GateStatus(readiness_tier="ready_to_prove"),
        )


def test_default_risk_class_mapping() -> None:
    assert default_risk_class(ActionKind.METRIC_VIEW) is RiskClass.ADDITIVE_ASSET
    assert default_risk_class(ActionKind.SKILL_UPDATE) is RiskClass.AGENT_CHANGE
    assert default_risk_class(ActionKind.GEPA_PROMPT) is RiskClass.AGENT_CHANGE
    assert default_risk_class(ActionKind.INSTRUCTION_UPDATE) is RiskClass.AGENT_CHANGE
    assert default_risk_class(ActionKind.REVERT) is RiskClass.AGENT_CHANGE


# -- ProofSummary.from_phase2_artifact (fail-closed) -----------------------


def test_proof_from_artifact_proven_and_correctness_held() -> None:
    proof = ProofSummary.from_phase2_artifact(_artifact(n_promote=3, savings_pct=35.4))
    assert proof.proved_improvement is True
    assert proof.correctness_held is True
    assert proof.n_promote == 3
    assert proof.realized_savings_pct == 35.4
    assert proof.suite_content_hash == "deadbeef"
    assert proof.suite_version == "v1-seed"


def test_proof_no_promote_is_not_an_improvement() -> None:
    proof = ProofSummary.from_phase2_artifact(_artifact(n_promote=0, n_block=4))
    assert proof.proved_improvement is False
    assert proof.correctness_held is False


def test_proof_with_regression_does_not_hold_correctness() -> None:
    # one task PROMOTEs but another REGRESSED correctness -> correctness not held
    proof = ProofSummary.from_phase2_artifact(_artifact(n_promote=2, regressed=True))
    assert proof.proved_improvement is True
    assert proof.correctness_held is False


# -- GateStatus projection -------------------------------------------------


def test_gate_status_surfaces_certifying_judge_agreement() -> None:
    judges = [
        JudgeHealth(judge_name="modularity", measured=True, agreement_rate=0.82, distrusted=False),
        JudgeHealth(judge_name="correctness", measured=False, distrusted=True),
    ]
    readiness = _readiness(tier=ReadinessTier.READY_TO_PROVE, judges=judges, scored_coverage=0.9)
    gate_status = GateStatus.from_readiness(
        readiness, gated=True, reasons=[], judge_name="modularity"
    )
    assert gate_status.readiness_tier == "ready_to_prove"
    assert gate_status.can_prove_improvement is True
    assert gate_status.judge_agreement == 0.82
    assert gate_status.scored_coverage == 0.9
    assert gate_status.n_distrusted_judges == 1  # the unmeasured correctness judge
    assert gate_status.gated is True


def test_gate_status_judge_agreement_none_when_no_judge() -> None:
    readiness = _readiness(tier=ReadinessTier.BASELINE_ONLY)
    gate_status = GateStatus.from_readiness(
        readiness, gated=False, reasons=["readiness not met"], judge_name=None
    )
    assert gate_status.judge_agreement is None
    assert gate_status.gated is False
    assert gate_status.reasons == ["readiness not met"]


# -- ids -------------------------------------------------------------------


def test_proposal_id_is_deterministic_and_content_addressed() -> None:
    a = derive_proposal_id(
        agent_name="claude_code", action_kind=ActionKind.METRIC_VIEW, change=_metric_view_change()
    )
    b = derive_proposal_id(
        agent_name="claude_code", action_kind=ActionKind.METRIC_VIEW, change=_metric_view_change()
    )
    assert a == b  # same content -> same id (idempotent publish)
    other = derive_proposal_id(
        agent_name="claude_code",
        action_kind=ActionKind.METRIC_VIEW,
        change=ProposedChange(
            kind=ChangeKind.METRIC_VIEW_SQL, summary="other", sql="CREATE VIEW different ..."
        ),
    )
    assert a != other  # different SQL -> different id


# -- the full record round-trips through JSON ------------------------------


def test_proposed_action_round_trips_through_json() -> None:
    change = _metric_view_change()
    proposal = ProposedAction(
        proposal_id=derive_proposal_id(
            agent_name="claude_code", action_kind=ActionKind.METRIC_VIEW, change=change
        ),
        agent_name="claude_code",
        action_kind=ActionKind.METRIC_VIEW,
        risk_class=default_risk_class(ActionKind.METRIC_VIEW),
        objective_metric="total_tokens",
        goal_cohort="claude_code",
        trigger=TriggerSignal(
            kind=TriggerKind.RLM_RECOMMENDED_ASSET, summary="x", n_traces=5, trace_refs=["t1"]
        ),
        change=change,
        proof=ProofSummary.from_phase2_artifact(_artifact(n_promote=2)),
        gate_status=GateStatus.from_readiness(
            _readiness(tier=ReadinessTier.READY_TO_PROVE), gated=True, reasons=[]
        ),
        created_at="2026-06-30T00:00:00+00:00",
    )
    assert proposal.status is ProposalStatus.PENDING
    restored = ProposedAction.model_validate_json(proposal.model_dump_json())
    assert restored == proposal
