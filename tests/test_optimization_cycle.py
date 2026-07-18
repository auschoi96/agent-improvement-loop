"""Unified optimization-cycle tests — injectable seams, no live MLflow/agent/warehouse.

Covers the piece-2/piece-3 hard constraints:

* **In-cycle RLM reuses the existing reviewer + doesn't crash the cycle**: the
  default RLM step delegates to :func:`ail.l3.continuous.run_continuous_rlm` with the
  existing sampling knobs (no new scheme); a total review failure is recorded and the
  cycle still runs, proves, gates, and publishes.
* **One cadence over one set**: the RLM step runs *before* the feedback is read.
* **Propose-only + idempotent publish**: the cycle publishes PENDING proposals (and
  publishes the empty set as a queue-preserving no-op) — it applies nothing.
* **Real feedback assembly (pure mappers)**: reading back attached RLM verdicts,
  recurrence-ranking assets, and mapping the L0 redundancy diagnosis.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ail.compare.contract import Recommendation
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.ingest.base import NormalizedTrace, TokenUsage, ToolCall, TraceStatus
from ail.jobs import optimization_cycle as oc
from ail.l3.continuous import ContinuousRlmRunReport
from ail.l3.contract import AssetRecommendation, HaloReviewVerdict
from ail.l3.reviewer import OVERALL_FEEDBACK_NAME
from ail.loop.controller import Candidate
from ail.loop.decision_rules import Decision, FeedbackBundle, RlmAssetSignal
from ail.loop.proposals import ChangeKind, ProposalStatus, ProposedChange
from ail.metrics.contract import (
    AggregateMetrics,
    L0MetricsReport,
    RepeatedCall,
    TokenBreakdown,
    TokenStats,
    ToolRedundancy,
)
from ail.metrics.l0_deterministic import _sum_cost
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
    return CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        guardrails=(Guardrail(name="modularity", kind="judge", threshold=4.0),),
        cohort="claude_code",
    ).confirm()


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


def _metric_view_candidate(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate:
    return Candidate(
        change=ProposedChange(
            kind=ChangeKind.METRIC_VIEW_SQL,
            summary="token waste view",
            sql="CREATE OR REPLACE VIEW c.s.v WITH METRICS LANGUAGE YAML AS $$ ... $$",
        ),
        prover_input="metric_view",
    )


def _feedback_metric_view() -> FeedbackBundle:
    return FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,
        rlm_assets=(
            RlmAssetSignal(
                asset_type="metric_view", title="token waste by tool", n_traces=5, rank=1
            ),
        ),
    )


def _rlm_report() -> ContinuousRlmRunReport:
    return ContinuousRlmRunReport(
        experiment_id="660599403165942",
        judge_model="databricks-claude-sonnet-4-6",
        n_scanned=10,
        n_already_reviewed=3,
        n_reviewer_traces_skipped=0,
        n_sampled_out=5,
        n_selected=2,
        n_reviewed=2,
        n_failed=0,
        sample_rate=0.10,
        max_reviews=2,
        outcomes=[],
    )


def _no_planner(feedback: FeedbackBundle, goal: CompiledGoal, agent: Agent) -> list[Decision]:
    # A Lane-B planner that proposes nothing (keeps these tests about the cadence,
    # not the planner — the planner has its own suite).
    from ail.loop.planner import PlanParseError

    raise PlanParseError("no plan (test)")


# ==========================================================================
# Orchestration: in-cycle RLM → propose → publish
# ==========================================================================


def test_rlm_runs_before_feedback_then_publishes() -> None:
    calls: list[str] = []
    published: list[Any] = []

    def _rlm() -> ContinuousRlmRunReport:
        calls.append("rlm")
        return _rlm_report()

    def _feedback() -> FeedbackBundle:
        calls.append("feedback")
        return _feedback_metric_view()

    def _publish(proposals: list[Any]) -> int:
        published.append(proposals)
        return len(proposals)

    report = oc.run_optimization_cycle(
        _agent(),
        _goal(),
        rlm_step=_rlm,
        feedback_source=_feedback,
        candidate_builder=_metric_view_candidate,
        prover=lambda c, *, goal, agent: _proven_artifact(),
        gate=lambda *, goal, agent: _ready(),
        planner=_no_planner,
        publish_fn=_publish,
        now="2026-07-01T00:00:00+00:00",
    )

    assert calls == ["rlm", "feedback"]  # one cadence, one set: review THEN plan
    assert report.rlm_error is None
    assert report.rlm is not None and report.rlm.n_reviewed == 2
    assert len(report.cycle.proposals) == 1
    assert report.cycle.proposals[0].status is ProposalStatus.PENDING
    assert report.n_published == 1
    assert len(published) == 1 and len(published[0]) == 1


def test_review_error_recorded_but_cycle_continues() -> None:
    # THE in-cycle-RLM fail-closed regression: a total review failure must not crash
    # the cycle or fabricate a verdict — it is recorded and the cycle still proposes.
    published: list[Any] = []

    def _rlm() -> ContinuousRlmRunReport:
        raise RuntimeError("trace store unreachable")

    report = oc.run_optimization_cycle(
        _agent(),
        _goal(),
        rlm_step=_rlm,
        feedback_source=_feedback_metric_view,
        candidate_builder=_metric_view_candidate,
        prover=lambda c, *, goal, agent: _proven_artifact(),
        gate=lambda *, goal, agent: _ready(),
        planner=_no_planner,
        publish_fn=lambda proposals: published.append(proposals) or len(proposals),
        now="2026-07-01T00:00:00+00:00",
    )

    assert report.rlm is None
    assert report.rlm_error is not None
    assert "RuntimeError" in report.rlm_error
    assert "trace store unreachable" in report.rlm_error
    # the cycle still ran and produced its proposal
    assert len(report.cycle.proposals) == 1
    assert report.n_published == 1


def test_zero_proposals_still_calls_queue_preserving_publish() -> None:
    # A blocked prove -> zero proposals; publish is still called (with the empty set)
    # while the queue-preserving publisher leaves existing rows untouched.
    published: list[Any] = []

    report = oc.run_optimization_cycle(
        _agent(),
        _goal(),
        rlm_step=_rlm_report,
        feedback_source=_feedback_metric_view,
        candidate_builder=_metric_view_candidate,
        prover=lambda c, *, goal, agent: _blocked_artifact(),
        gate=lambda *, goal, agent: _ready(),
        planner=_no_planner,
        publish_fn=lambda proposals: published.append(proposals) or len(proposals),
    )

    assert report.cycle.proposals == ()
    assert report.n_published == 0
    assert published == [[]]  # called once, with the empty set


def test_default_rlm_step_reuses_continuous_reviewer(monkeypatch: Any) -> None:
    # The in-cycle RLM step delegates to the EXISTING reviewer with the EXISTING
    # sampling knobs — no reimplementation, no new sampling scheme.
    captured: dict[str, Any] = {}

    def _fake_run(experiment: str, **kwargs: Any) -> ContinuousRlmRunReport:
        captured["experiment"] = experiment
        captured.update(kwargs)
        return _rlm_report()

    monkeypatch.setattr(oc, "run_continuous_rlm", _fake_run)
    args = SimpleNamespace(
        experiment="660599403165942",
        judge_model="databricks-claude-sonnet-4-6",
        warehouse_id="wh1",
        max_results=100,
        max_reviews=2,
        sample_rate=0.10,
        min_tokens=50_000,
        reviewer_experiment="",
        max_turns=40,
        temperature=None,
    )
    step = oc._default_rlm_step(args)
    report = step()

    assert report.n_reviewed == 2
    assert captured["experiment"] == "660599403165942"
    assert captured["judge_model"] == "databricks-claude-sonnet-4-6"
    assert captured["sample_rate"] == 0.10
    assert captured["max_reviews"] == 2
    assert captured["min_tokens"] == 50_000
    assert captured["max_results"] == 100


# ==========================================================================
# Feedback assembly (pure mappers): read the now-fresh cohort feedback
# ==========================================================================


def _verdict(trace_id: str, *, asset_title: str = "token waste by tool") -> HaloReviewVerdict:
    return HaloReviewVerdict(
        subject_trace_id=trace_id,
        token_efficiency="fair",
        token_waste_score=40,
        summary="some waste",
        recommended_assets=[AssetRecommendation(asset_type="metric_view", title=asset_title)],
    )


def _trace_with_verdict(
    trace_id: str, verdict: HaloReviewVerdict | None, **kw: Any
) -> NormalizedTrace:
    assessments = []
    if verdict is not None:
        assessments.append(
            SimpleNamespace(
                name=OVERALL_FEEDBACK_NAME,
                metadata={"verdict_json": verdict.model_dump_json()},
            )
        )
    raw = SimpleNamespace(info=SimpleNamespace(assessments=assessments))
    return NormalizedTrace(trace_id=trace_id, status=TraceStatus.OK, raw=raw, **kw)


def test_verdict_from_trace_reads_attached_verdict() -> None:
    v = _verdict("t1")
    got = oc.verdict_from_trace(_trace_with_verdict("t1", v))
    assert got is not None
    assert got.subject_trace_id == "t1"
    assert [a.title for a in got.recommended_assets] == ["token waste by tool"]


def test_verdict_from_trace_none_when_absent_or_malformed() -> None:
    assert oc.verdict_from_trace(_trace_with_verdict("t1", None)) is None
    bad = SimpleNamespace(
        info=SimpleNamespace(
            assessments=[
                SimpleNamespace(name=OVERALL_FEEDBACK_NAME, metadata={"verdict_json": "{"})
            ]
        )
    )
    assert oc.verdict_from_trace(NormalizedTrace(trace_id="t2", raw=bad)) is None


def test_ranked_assets_from_traces_recurrence_ranks() -> None:
    traces = [
        _trace_with_verdict("t1", _verdict("t1")),
        _trace_with_verdict("t2", _verdict("t2")),
        _trace_with_verdict("t3", None),  # no verdict -> contributes nothing
    ]
    ranked = oc.ranked_assets_from_traces(traces)
    assert len(ranked) == 1
    assert ranked[0].n_traces == 2
    assert ranked[0].rank == 1
    signals = oc.rlm_asset_signals(ranked)
    assert signals[0].asset_type == "metric_view"
    assert signals[0].n_traces == 2


def _l0_report(repeats: list[RepeatedCall], *, total_tokens: int = 1234) -> L0MetricsReport:
    return L0MetricsReport(
        aggregate=AggregateMetrics(
            n_traces=1,
            tokens=TokenBreakdown(total_tokens=total_tokens),
            token_stats=TokenStats(),
            cost=_sum_cost([]),
            redundancy=ToolRedundancy(repeated_calls=repeats),
        )
    )


def test_redundant_reads_from_l0_marks_dominant_and_filters() -> None:
    report = _l0_report(
        [
            RepeatedCall(tool="Read", identity="/a", count=5, signature_kind="path"),
            RepeatedCall(tool="Read", identity="/b", count=3, signature_kind="path"),
            RepeatedCall(tool="Bash", identity="setup", count=1, signature_kind="shell"),
        ]
    )
    signals = oc.redundant_reads_from_l0(report, min_occurrences=2, dominant_top_n=1)
    # count=1 dropped (below min); the two remaining are sorted by count desc
    assert [s.repeated_target for s in signals] == ["/a", "/b"]
    assert [s.occurrences for s in signals] == [5, 3]
    assert [s.dominant for s in signals] == [True, False]  # only the top is dominant


def test_build_feedback_bundle_composes_real_sources() -> None:
    # Two traces: repeated Read of the same path (L0 redundancy) + an attached RLM
    # verdict recommending a metric_view. total_tokens is the L0 objective value.
    def _mk(trace_id: str) -> NormalizedTrace:
        return _trace_with_verdict(
            trace_id,
            _verdict(trace_id),
            token_usage=TokenUsage(input_tokens=600, output_tokens=400),
            tool_calls=[
                ToolCall(id=f"{trace_id}-1", name="Read", arguments={"file_path": "/x"}),
                ToolCall(id=f"{trace_id}-2", name="Read", arguments={"file_path": "/x"}),
                ToolCall(id=f"{trace_id}-3", name="Read", arguments={"file_path": "/x"}),
            ],
        )

    traces = [_mk("t1"), _mk("t2")]
    bundle = oc.build_feedback_bundle(
        traces, objective_metric="total_tokens", objective_baseline=3000.0
    )
    assert bundle.objective_metric_value == 2000.0  # (600+400) * 2 traces
    assert bundle.objective_baseline_value == 3000.0
    # the recurring metric_view recommendation surfaced across both traces
    assert len(bundle.rlm_assets) == 1
    assert bundle.rlm_assets[0].asset_type == "metric_view"
    assert bundle.rlm_assets[0].n_traces == 2
    # the repeated Read of /x surfaced as a redundant-read signal
    assert any(s.repeated_target == "/x" for s in bundle.redundant_reads)
    # no fabricated judge / regression signals
    assert bundle.judge_dimensions == ()
    assert bundle.post_apply_regressions == ()
