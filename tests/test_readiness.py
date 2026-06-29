"""Tests for the readiness gating + eval-health surface (:mod:`ail.readiness`).

All offline, fixtures only — readiness is a pure function of a
:class:`~ail.cohorts.Cohort` (identity) and the measured
:class:`~ail.readiness.ReadinessFacts`, so no test touches MLflow, a model, or a
live workspace. The ``GoalView`` Protocol is satisfied by a tiny local stub
(:class:`FakeGoal`) so these tests never import the parallel goals lane.

The headline invariant under test is **fail-closed**: missing data (no traces, no
labels, an unmeasured judge, zero coverage) must yield a *not-ready* tier with the
reason spelled out — never a ready tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ail.cohorts import Cohort
from ail.judges import ScorePair, compute_agreement
from ail.readiness import (
    EvalHealth,
    Gate,
    GateName,
    GoalView,
    JudgeFact,
    ReadinessFacts,
    ReadinessStatus,
    ReadinessThresholds,
    ReadinessTier,
    compute_eval_health,
    compute_readiness,
)

# ---------------------------------------------------------------------------
# Test doubles + fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeGoal:
    """A structural stand-in for the goals lane's ``CompiledGoal``.

    Carries exactly the three members :class:`GoalView` requires, so it satisfies
    the Protocol without importing ``ail.goals`` (which does not exist in this
    lane).
    """

    objective_metric: str
    requires_quality: bool
    guardrail_names: tuple[str, ...] = field(default_factory=tuple)


COST_GOAL = FakeGoal(objective_metric="total_tokens", requires_quality=False)
QUALITY_GOAL = FakeGoal(
    objective_metric="correctness",
    requires_quality=True,
    guardrail_names=("correctness",),
)


@pytest.fixture
def cohort() -> Cohort:
    return Cohort.by_agent("claude_code")


def _trusted_judge(name: str = "correctness", *, n_scored: int) -> JudgeFact:
    """A judge measured at perfect agreement (well above the floor)."""
    pairs = [ScorePair(item_id=str(i), human_value="yes", judge_value="yes") for i in range(10)]
    report = compute_agreement(pairs, judge_name=name)
    return JudgeFact.from_agreement_report(report, n_scored_traces=n_scored)


def _healthy_quality_facts(*, trace_count: int) -> ReadinessFacts:
    """Facts where every quality gate passes (frozen suite, labels, judge, coverage)."""
    return ReadinessFacts(
        trace_count=trace_count,
        label_count=30,
        frozen_suite_present=True,
        n_scored_traces=trace_count,
        judge_runs=trace_count,
        judge_run_successes=trace_count,
        judges=(_trusted_judge(n_scored=trace_count),),
    )


# ---------------------------------------------------------------------------
# GoalView Protocol
# ---------------------------------------------------------------------------


class TestGoalView:
    def test_stub_satisfies_protocol(self) -> None:
        assert isinstance(COST_GOAL, GoalView)
        assert isinstance(QUALITY_GOAL, GoalView)

    def test_protocol_members_readable(self) -> None:
        assert QUALITY_GOAL.objective_metric == "correctness"
        assert QUALITY_GOAL.requires_quality is True
        assert tuple(QUALITY_GOAL.guardrail_names) == ("correctness",)


# ---------------------------------------------------------------------------
# The four headline scenarios
# ---------------------------------------------------------------------------


class TestReadinessScenarios:
    def test_zero_traces_is_collecting(self, cohort: Cohort) -> None:
        """An empty cohort (0 traces) => COLLECTING, regardless of goal."""
        for goal in (COST_GOAL, QUALITY_GOAL):
            status = compute_readiness(cohort, goal, ReadinessFacts(trace_count=0))
            assert status.tier is ReadinessTier.COLLECTING
            assert status.can_prove_improvement is False
            # the not-ready reason names the trace shortfall
            assert any("more trace" in r for r in status.reasons)

    def test_quality_goal_with_no_labels_is_not_ready(self, cohort: Cohort) -> None:
        """Enough traces but 0 labels + a quality goal => not ready, with a labels reason."""
        facts = ReadinessFacts(trace_count=60, label_count=0, frozen_suite_present=True)
        status = compute_readiness(cohort, QUALITY_GOAL, facts)

        assert status.tier is ReadinessTier.BASELINE_ONLY
        assert status.can_prove_improvement is False
        labels_gate = status.gate_for(GateName.HUMAN_LABELS)
        assert labels_gate is not None and labels_gate.passed is False
        assert any("human label" in r for r in status.reasons)

    def test_healthy_cost_goal_with_enough_traces_is_ready_to_prove(self, cohort: Cohort) -> None:
        """A healthy token/cost goal with enough traces => READY_TO_PROVE.

        A deterministic goal needs no quality gates — trace count alone clears it.
        """
        status = compute_readiness(cohort, COST_GOAL, ReadinessFacts(trace_count=60))

        assert status.tier is ReadinessTier.READY_TO_PROVE
        assert status.can_prove_improvement is True
        assert status.reasons == []
        # quality gates are not even evaluated for a deterministic goal
        assert status.gate_for(GateName.HUMAN_LABELS) is None
        assert status.gate_for(GateName.JUDGE_TRUSTED) is None

    def test_unmeasured_judge_distrusted_and_zero_coverage_not_ready(self, cohort: Cohort) -> None:
        """An unmeasured judge => distrusted, and 0 scored-coverage => not ready."""
        unmeasured = JudgeFact(judge_name="correctness", agreement_rate=None, n_scored_traces=0)
        facts = ReadinessFacts(
            trace_count=60,
            label_count=30,
            frozen_suite_present=True,
            n_scored_traces=0,
            judge_runs=0,
            judges=(unmeasured,),
        )
        status = compute_readiness(cohort, QUALITY_GOAL, facts)

        assert status.tier is ReadinessTier.BASELINE_ONLY
        assert status.can_prove_improvement is False

        # distrusted-by-default surfaces in eval-health
        assert status.eval_health.n_distrusted_judges == 1
        assert status.eval_health.distrusted_judges == ["correctness"]
        assert status.eval_health.scored_coverage == 0.0

        # both the judge-trust and coverage gates fail with reasons
        judge_gate = status.gate_for(GateName.JUDGE_TRUSTED)
        coverage_gate = status.gate_for(GateName.SCORED_COVERAGE)
        assert judge_gate is not None and judge_gate.passed is False
        assert coverage_gate is not None and coverage_gate.passed is False
        assert any("distrusted" in r for r in status.reasons)
        assert any("scored-coverage" in r for r in status.reasons)


# ---------------------------------------------------------------------------
# Tier ladder
# ---------------------------------------------------------------------------


class TestTierLadder:
    def test_below_baseline_is_collecting(self, cohort: Cohort) -> None:
        status = compute_readiness(cohort, COST_GOAL, ReadinessFacts(trace_count=5))
        assert status.tier is ReadinessTier.COLLECTING

    def test_cost_goal_baseline_only_between_floors(self, cohort: Cohort) -> None:
        """>= baseline but < prove floor => BASELINE_ONLY for a deterministic goal."""
        status = compute_readiness(cohort, COST_GOAL, ReadinessFacts(trace_count=20))
        assert status.tier is ReadinessTier.BASELINE_ONLY
        assert status.can_prove_improvement is False

    def test_quality_goal_ready_for_quality_below_prove_floor(self, cohort: Cohort) -> None:
        """All quality gates pass but < prove floor => READY_FOR_QUALITY."""
        status = compute_readiness(cohort, QUALITY_GOAL, _healthy_quality_facts(trace_count=40))
        assert status.tier is ReadinessTier.READY_FOR_QUALITY
        assert status.can_prove_improvement is False
        assert status.reasons == [status.gate_for(GateName.TRACE_PROVE).reason]  # type: ignore[union-attr]

    def test_quality_goal_ready_to_prove_at_prove_floor(self, cohort: Cohort) -> None:
        status = compute_readiness(cohort, QUALITY_GOAL, _healthy_quality_facts(trace_count=60))
        assert status.tier is ReadinessTier.READY_TO_PROVE
        assert status.can_prove_improvement is True
        assert status.reasons == []

    def test_thresholds_are_adjustable(self, cohort: Cohort) -> None:
        """The ladder is configurable, not buried constants (doc §2)."""
        lax = ReadinessThresholds(baseline_min_traces=2, prove_min_traces=5)
        status = compute_readiness(cohort, COST_GOAL, ReadinessFacts(trace_count=6), thresholds=lax)
        assert status.tier is ReadinessTier.READY_TO_PROVE


# ---------------------------------------------------------------------------
# Judge trust (distrusted-by-default)
# ---------------------------------------------------------------------------


class TestJudgeTrust:
    def test_unmeasured_judge_is_distrusted_by_default(self) -> None:
        jf = JudgeFact(judge_name="j", agreement_rate=None)
        assert jf.measured is False
        assert jf.is_distrusted is True

    def test_below_floor_judge_is_distrusted(self) -> None:
        jf = JudgeFact(judge_name="j", agreement_rate=0.4, agreement_floor=0.7)
        assert jf.measured is True
        assert jf.is_distrusted is True

    def test_at_floor_judge_is_trusted(self) -> None:
        jf = JudgeFact(judge_name="j", agreement_rate=0.7, agreement_floor=0.7)
        assert jf.is_distrusted is False

    def test_from_agreement_report_carries_distrust(self) -> None:
        """A below-floor agreement report yields a distrusted JudgeFact."""
        pairs = [
            ScorePair(item_id=str(i), human_value="yes", judge_value=("no" if i < 7 else "yes"))
            for i in range(10)
        ]
        report = compute_agreement(pairs, judge_name="correctness")
        assert report.distrusted is True
        jf = JudgeFact.from_agreement_report(report, n_scored_traces=30)
        assert jf.is_distrusted is True

    def test_quality_goal_with_only_distrusted_judges_not_ready(self, cohort: Cohort) -> None:
        facts = ReadinessFacts(
            trace_count=60,
            label_count=30,
            frozen_suite_present=True,
            n_scored_traces=60,
            judge_runs=60,
            judge_run_successes=60,
            judges=(JudgeFact(judge_name="correctness", agreement_rate=0.4),),
        )
        status = compute_readiness(cohort, QUALITY_GOAL, facts)
        assert status.tier is ReadinessTier.BASELINE_ONLY
        gate = status.gate_for(GateName.JUDGE_TRUSTED)
        assert gate is not None and gate.passed is False

    def test_quality_goal_with_no_judges_fails_closed(self, cohort: Cohort) -> None:
        """No judges at all => judge-trust gate fails (distrusted by default)."""
        facts = ReadinessFacts(
            trace_count=60, label_count=30, frozen_suite_present=True, n_scored_traces=60
        )
        status = compute_readiness(cohort, QUALITY_GOAL, facts)
        gate = status.gate_for(GateName.JUDGE_TRUSTED)
        assert gate is not None and gate.passed is False
        assert "no calibrated judge" in gate.reason


# ---------------------------------------------------------------------------
# Eval-health surface
# ---------------------------------------------------------------------------


class TestEvalHealth:
    def test_fields_on_empty_cohort(self, cohort: Cohort) -> None:
        health = compute_eval_health(cohort, ReadinessFacts(trace_count=0))
        assert health.cohort_name == cohort.name
        assert health.n_traces == 0
        assert health.scored_coverage == 0.0
        assert health.judge_run_success_rate is None
        assert health.n_judges == 0
        assert health.n_distrusted_judges == 0

    def test_scored_coverage_fraction(self, cohort: Cohort) -> None:
        health = compute_eval_health(cohort, ReadinessFacts(trace_count=40, n_scored_traces=10))
        assert health.scored_coverage == 0.25

    def test_judge_run_success_rate(self, cohort: Cohort) -> None:
        facts = ReadinessFacts(
            trace_count=10,
            judge_runs=10,
            judge_run_successes=8,
            judges=(JudgeFact(judge_name="j", agreement_rate=0.9),),
        )
        health = compute_eval_health(cohort, facts)
        assert health.judge_run_success_rate == 0.8

    def test_no_judge_runs_rate_is_none_not_full(self, cohort: Cohort) -> None:
        """Zero judge runs => success rate is None (did-not-evaluate), not 100%."""
        facts = ReadinessFacts(
            trace_count=10, judge_runs=0, judges=(JudgeFact(judge_name="j", agreement_rate=None),)
        )
        health = compute_eval_health(cohort, facts)
        assert health.judge_run_success_rate is None
        assert any("undefined" in n for n in health.notes)

    def test_distrusted_count_and_per_judge_detail(self, cohort: Cohort) -> None:
        facts = ReadinessFacts(
            trace_count=20,
            n_scored_traces=20,
            judges=(
                JudgeFact(judge_name="trusted", agreement_rate=0.9, n_scored_traces=20),
                JudgeFact(judge_name="unmeasured", agreement_rate=None),
                JudgeFact(judge_name="below", agreement_rate=0.3),
            ),
        )
        health = compute_eval_health(cohort, facts)
        assert health.n_judges == 3
        assert health.n_distrusted_judges == 2
        assert set(health.distrusted_judges) == {"unmeasured", "below"}
        per = {j.judge_name: j for j in health.judges}
        assert per["trusted"].distrusted is False
        assert per["trusted"].coverage == 1.0
        assert per["unmeasured"].measured is False


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


class TestContract:
    def test_status_round_trips_through_json(self, cohort: Cohort) -> None:
        status = compute_readiness(cohort, QUALITY_GOAL, _healthy_quality_facts(trace_count=60))
        restored = ReadinessStatus.model_validate_json(status.model_dump_json())
        assert restored == status
        # the convenience flag is derived from the serialized tier
        assert restored.tier is ReadinessTier.READY_TO_PROVE
        assert restored.can_prove_improvement is True

    def test_models_forbid_unknown_fields(self) -> None:
        for model in (ReadinessStatus, EvalHealth, Gate):
            assert model.model_config.get("extra") == "forbid"

    def test_gate_carries_name_pass_and_reason(self, cohort: Cohort) -> None:
        status = compute_readiness(cohort, COST_GOAL, ReadinessFacts(trace_count=0))
        gate = status.gate_for(GateName.TRACE_BASELINE)
        assert isinstance(gate, Gate)
        assert gate.passed is False
        assert gate.reason


# ---------------------------------------------------------------------------
# Facts validation
# ---------------------------------------------------------------------------


class TestFactsValidation:
    def test_scored_cannot_exceed_traces(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            ReadinessFacts(trace_count=5, n_scored_traces=6)

    def test_successes_cannot_exceed_runs(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            ReadinessFacts(trace_count=5, judge_runs=2, judge_run_successes=3)

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            ReadinessFacts(trace_count=-1)
        with pytest.raises(ValueError, match=">= 0"):
            JudgeFact(judge_name="j", n_scored_traces=-1)
