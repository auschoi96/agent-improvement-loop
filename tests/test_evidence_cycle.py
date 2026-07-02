"""Evidence-first cycle tests — decoupled from proving, fail-closed, propose-only.

The local companion planner lane (``docs/PRODUCT_ARCHITECTURE.md`` §3/§7):
:func:`ail.loop.evidence_cycle.run_evidence_cycle` emits a PENDING proposal on
**evidence + gate alone** — no frozen-suite prover, no :class:`ProofSummary`. These
tests pin the properties the reviewer checks:

* (a) a built + gated candidate → exactly one PENDING proposal carrying its evidence
  (why) and gate status, with ``proof=None`` — and **the prover is never invoked**;
* (b) a Lane-B planner failure contributes zero B decisions but **Lane A still runs
  and emits** (fail-closed, preserved from :mod:`ail.loop.planner`);
* (c) an ungated state (readiness not met, or a distrusted certifying judge) → ZERO
  proposals;
* (d) **no fabricated evidence**: an unreadable feedback source propagates (the cycle
  never invents a "why" to propose on);
* (e) the real :func:`evidence_candidate_builder` maps the redundant-read skill
  decision (and declines everything else, fail-closed);
* (f) **schema / read-compatibility**: an evidence-only (``proof=None``) proposal
  flattens to a row with NULL ``proof_*`` columns that the app's SELECT-only reader
  (:func:`ail.loop.apply_service._row_to_proposal`) reconstructs without error.

No live MLflow / agent / warehouse: every seam is an injected fake.
"""

from __future__ import annotations

import pytest

from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.loop.candidate_builders import evidence_candidate_builder
from ail.loop.controller import Candidate
from ail.loop.decision_rules import (
    Decision,
    FeedbackBundle,
    JudgeDimensionSignal,
    RedundantReadSignal,
    RlmAssetSignal,
)
from ail.loop.evidence_cycle import EvidenceCycleResult, run_evidence_cycle
from ail.loop.planner import PlanParseError
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProposalStatus,
    ProposedChange,
    RiskClass,
    TriggerKind,
)
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
    """A confirmed minimize-tokens goal guarding a trusted ``modularity`` judge."""
    return CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        guardrails=(Guardrail(name="modularity", kind="judge", threshold=4.0),),
        cohort="claude_code",
    ).confirm()


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
    """A fake builder producing a matching change per action kind (prover_input unused)."""
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
    return Candidate(change=change, prover_input=None)


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


def _feedback_redundant() -> FeedbackBundle:
    return FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,
        redundant_reads=(
            RedundantReadSignal(
                tool="Read",
                repeated_target="/x",
                occurrences=5,
                dominant=True,
                estimated_wasted_tokens=4200,
                trace_ids=("t1", "t2"),
            ),
        ),
    )


def _planner_empty(feedback: FeedbackBundle, goal: CompiledGoal, agent: Agent) -> list[Decision]:
    """A Lane-B planner that proposes nothing cleanly (keeps a test about Lane A)."""
    return []


def _planner_raises(feedback: FeedbackBundle, goal: CompiledGoal, agent: Agent) -> list[Decision]:
    """A Lane-B planner that fails closed (parse error / model failure)."""
    raise PlanParseError("planner produced no usable plan (test)")


def _run(
    *,
    feedback: FeedbackBundle,
    readiness: ReadinessStatus,
    builder=_build_candidate,  # type: ignore[no-untyped-def]
    planner=_planner_empty,  # type: ignore[no-untyped-def]
) -> EvidenceCycleResult:
    return run_evidence_cycle(
        _agent(),
        _goal(),
        feedback_source=lambda: feedback,
        candidate_builder=builder,
        gate=lambda *, goal, agent: readiness,
        planner=planner,
        now="2026-07-02T00:00:00+00:00",
    )


# -- (a) built + gated -> one PENDING proposal, evidence-only, NO prover ----


def test_built_and_gated_emits_one_evidence_only_proposal() -> None:
    ecr = _run(feedback=_feedback_metric_view(), readiness=_ready())
    assert len(ecr.result.proposals) == 1
    assert ecr.result.skipped == ()
    p = ecr.result.proposals[0]
    assert p.status is ProposalStatus.PENDING
    assert p.agent_name == "claude_code"
    assert p.action_kind is ActionKind.METRIC_VIEW
    assert p.risk_class is RiskClass.ADDITIVE_ASSET
    # evidence (the why) travels
    assert p.trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET
    assert p.trigger.trace_refs == ["t1", "t2", "t3"]
    # what
    assert p.change.kind is ChangeKind.METRIC_VIEW_SQL
    # EVIDENCE-FIRST: no frozen-suite proof at all
    assert p.proof is None
    # gate is real + unweakened
    assert p.gate_status.gated is True
    assert p.gate_status.readiness_tier == "ready_to_prove"
    assert p.created_at == "2026-07-02T00:00:00+00:00"
    # plan provenance
    assert ecr.plan.n_from_a == 1
    assert ecr.plan.n_from_b == 0
    assert ecr.plan.planner_error is None


def test_prover_is_never_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point of the lane: proving is decoupled from proposing. Trip both the
    # frozen-suite comparison and the proof-summary extraction; neither may be touched.
    import ail.loop.proposals as proposals_mod
    import ail.optimize.phase2 as phase2

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("the evidence-first planner must NEVER prove")

    monkeypatch.setattr(phase2, "run_phase2_comparison", _boom)
    monkeypatch.setattr(proposals_mod.ProofSummary, "from_phase2_artifact", staticmethod(_boom))

    ecr = _run(feedback=_feedback_metric_view(), readiness=_ready())
    assert len(ecr.result.proposals) == 1
    assert ecr.result.proposals[0].proof is None
    # structural: run_evidence_cycle takes no prover seam at all
    assert "prover" not in run_evidence_cycle.__code__.co_varnames


# -- (b) Lane-B failure contributes zero B, Lane A still emits (fail-closed) -


def test_lane_b_failure_preserves_lane_a() -> None:
    ecr = _run(feedback=_feedback_metric_view(), readiness=_ready(), planner=_planner_raises)
    # Lane A's metric_view decision still produced its proposal
    assert [p.action_kind for p in ecr.result.proposals] == [ActionKind.METRIC_VIEW]
    # Lane B contributed nothing, and its failure is recorded (not fabricated away)
    assert ecr.plan.n_from_a == 1
    assert ecr.plan.n_from_b == 0
    assert ecr.plan.planner_error is not None
    assert "no usable plan" in ecr.plan.planner_error


def test_lane_b_llm_call_error_also_preserves_lane_a() -> None:
    def _planner_boom(feedback, goal, agent):  # type: ignore[no-untyped-def]
        raise RuntimeError("model endpoint 503")

    ecr = _run(feedback=_feedback_metric_view(), readiness=_ready(), planner=_planner_boom)
    assert [p.action_kind for p in ecr.result.proposals] == [ActionKind.METRIC_VIEW]
    assert ecr.plan.planner_error is not None
    assert "503" in ecr.plan.planner_error


# -- (c) ungated -> ZERO proposals -----------------------------------------


def test_readiness_not_met_yields_zero_proposals() -> None:
    ecr = _run(feedback=_feedback_metric_view(), readiness=_not_ready())
    assert ecr.result.proposals == ()
    assert len(ecr.result.skipped) == 1
    assert "readiness not met" in ecr.result.skipped[0].reason


def test_distrusted_certifying_judge_yields_zero_proposals() -> None:
    # tier is READY_TO_PROVE, but the judge-dimension trigger's certifying judge is distrusted
    ecr = _run(feedback=_feedback_judge(), readiness=_ready(modularity_distrusted=True))
    assert ecr.result.proposals == ()
    assert len(ecr.result.skipped) == 1
    assert "distrusted" in ecr.result.skipped[0].reason


def test_candidate_builder_returning_none_yields_zero_proposals() -> None:
    ecr = _run(
        feedback=_feedback_metric_view(),
        readiness=_ready(),
        builder=lambda decision, *, goal, agent: None,
    )
    assert ecr.result.proposals == ()
    assert "no candidate" in ecr.result.skipped[0].reason


# -- (d) no fabricated evidence: unreadable feedback propagates -------------


def test_unreadable_feedback_propagates_no_proposal() -> None:
    def _broken_feedback() -> FeedbackBundle:
        raise RuntimeError("trace store unreachable")

    with pytest.raises(RuntimeError, match="trace store unreachable"):
        run_evidence_cycle(
            _agent(),
            _goal(),
            feedback_source=_broken_feedback,
            candidate_builder=_build_candidate,
            gate=lambda *, goal, agent: _ready(),
            planner=_planner_empty,
        )


def test_refuses_unconfirmed_goal() -> None:
    unconfirmed = CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        cohort="claude_code",
    )
    assert unconfirmed.human_confirmed is False
    with pytest.raises(ValueError, match="unconfirmed goal"):
        run_evidence_cycle(
            _agent(),
            unconfirmed,
            feedback_source=_feedback_metric_view,
            candidate_builder=_build_candidate,
            gate=lambda *, goal, agent: _ready(),
        )


# -- per-decision fault isolation ------------------------------------------


def test_one_decision_error_does_not_drop_the_others() -> None:
    # metric_view (additive) + gepa_prompt (judge). The builder raises ONLY for the
    # gepa_prompt decision; the metric_view proposal must survive, the failing one is a
    # fail-closed skip carrying the error.
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

    def _flaky_builder(decision, *, goal, agent):  # type: ignore[no-untyped-def]
        if decision.action_kind is ActionKind.GEPA_PROMPT:
            raise RuntimeError("builder blew up")
        return _build_candidate(decision, goal=goal, agent=agent)

    ecr = _run(feedback=feedback, readiness=_ready(), builder=_flaky_builder)
    assert [p.action_kind for p in ecr.result.proposals] == [ActionKind.METRIC_VIEW]
    errored = [s for s in ecr.result.skipped if s.action_kind == ActionKind.GEPA_PROMPT.value]
    assert len(errored) == 1
    assert "errored" in errored[0].reason
    assert "RuntimeError" in errored[0].reason


# -- (e) the real evidence_candidate_builder -------------------------------


def test_evidence_candidate_builder_maps_redundant_read_skill() -> None:
    # Lane A's redundant-read rule -> SKILL_UPDATE(REDUNDANT_READ_PATTERN); the real
    # builder installs the token-efficiency skill (no cost guard, no prove).
    ecr = _run(
        feedback=_feedback_redundant(),
        readiness=_ready(),
        builder=evidence_candidate_builder(),
    )
    assert len(ecr.result.proposals) == 1
    p = ecr.result.proposals[0]
    assert p.action_kind is ActionKind.SKILL_UPDATE
    assert p.change.kind is ChangeKind.SKILL_DIFF
    assert p.change.diff  # the token-efficiency skill section, as a diff addition
    assert "token-efficiency" in p.change.summary.lower()
    assert p.proof is None


def test_evidence_candidate_builder_declines_non_redundant_read() -> None:
    # An RLM metric_view decision is NOT the redundant-read skill target -> None -> skip.
    ecr = _run(
        feedback=_feedback_metric_view(),
        readiness=_ready(),
        builder=evidence_candidate_builder(),
    )
    assert ecr.result.proposals == ()
    assert "no candidate" in ecr.result.skipped[0].reason


# -- (f) schema / read-compatibility ---------------------------------------


def test_evidence_only_row_has_null_proof_and_full_width() -> None:
    from ail.loop.publish_proposals import PROPOSAL_COLUMNS, _proposal_row

    p = _run(feedback=_feedback_metric_view(), readiness=_ready()).result.proposals[0]
    assert p.proof is None
    row = _proposal_row(p, generated_at="2026-07-02T00:00:00+00:00")
    # additive-only: width still matches the (unchanged) column set
    assert len(row) == len(PROPOSAL_COLUMNS)
    # every proof_* column is NULL for an evidence-only proposal
    for col in PROPOSAL_COLUMNS:
        if col.startswith("proof_"):
            assert row[PROPOSAL_COLUMNS.index(col)] is None
    # the why + gate still populate
    assert row[PROPOSAL_COLUMNS.index("trigger_kind")] == TriggerKind.RLM_RECOMMENDED_ASSET.value
    assert row[PROPOSAL_COLUMNS.index("gate_gated")] is True


def test_evidence_only_row_round_trips_through_app_reader() -> None:
    # The app reads agent_proposed_actions SELECT-only; lane 3b reconstructs a proposal
    # from the flat row via ail.loop.apply_service._row_to_proposal. An evidence-only
    # (proof=None) row must reconstruct WITHOUT error (NULL proof columns -> a zeroed
    # ProofSummary), so the read path stays compatible.
    from ail.loop.apply_service import _row_to_proposal
    from ail.loop.publish_proposals import PROPOSAL_COLUMNS, _proposal_row

    p = _run(feedback=_feedback_metric_view(), readiness=_ready()).result.proposals[0]
    row = dict(
        zip(
            PROPOSAL_COLUMNS,
            _proposal_row(p, generated_at="2026-07-02T00:00:00+00:00"),
            strict=True,
        )
    )
    reconstructed = _row_to_proposal(row)
    assert reconstructed.proposal_id == p.proposal_id
    assert reconstructed.action_kind is ActionKind.METRIC_VIEW
    assert reconstructed.change.kind is ChangeKind.METRIC_VIEW_SQL
    assert reconstructed.trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET
    # a NULL-proof row reads back as a non-improving zeroed proof (no crash)
    assert reconstructed.proof is not None
    assert reconstructed.proof.proved_improvement is False
    assert reconstructed.gate_status.gated is True


def test_app_selected_columns_are_all_present() -> None:
    # The two app queries (ail-self-optimizer/config/queries/{proposed_actions,recent_activity}.sql)
    # SELECT only these columns; keeping the schema additive-only means they still resolve.
    from ail.loop.publish_proposals import PROPOSAL_COLUMNS

    app_selected = {
        # proposed_actions.sql
        "proposal_id",
        "agent_name",
        "status",
        "action_kind",
        "risk_class",
        "objective_metric",
        "created_at",
        "trigger_kind",
        "trigger_summary",
        "trigger_metric",
        "trigger_observed_value",
        "trigger_threshold",
        "trigger_n_traces",
        "trigger_judge_name",
        "change_kind",
        "change_summary",
        "change_sql",
        "change_diff",
        "change_evolved_body_ref",
        "change_revert_target",
        "proof_proved_improvement",
        "proof_correctness_held",
        "proof_realized_savings_pct",
        "proof_n_promote",
        "proof_n_block",
        "proof_suite_version",
        "gate_readiness_tier",
        "gate_gated",
        "gate_judge_agreement",
        "gate_scored_coverage",
        "gate_n_distrusted_judges",
        # recent_activity.sql (adds generated_at; trigger_summary already listed above)
        "generated_at",
    }
    missing = app_selected - set(PROPOSAL_COLUMNS)
    assert missing == set()
