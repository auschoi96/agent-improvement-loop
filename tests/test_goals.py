"""Tests for the goal compiler (:mod:`ail.goals`).

All offline: every test injects a **mocked LLM** (a callable returning a canned
string), so no live model is ever called. The headline guarantees under test:

* a goal is **strictly** validated against the schema + the allowlist — an
  unmapped metric fails loud, never silently invented;
* the **human gate** holds — a compiled goal is unconfirmed until :meth:`confirm`;
* the **readiness contract** holds — a :class:`CompiledGoal` is accepted by
  :func:`ail.readiness.compute_readiness` as a ``GoalView`` and its
  ``judge_trusted`` gate keys off the goal's ``guardrail_names``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ail.cohorts import Cohort
from ail.goals import (
    ALLOWLIST,
    JUDGE_METRICS,
    L0_OBJECTIVE_METRICS,
    CompiledGoal,
    GoalCompileError,
    GoalContractError,
    GoalProposerLLM,
    UnmappedMetricError,
    compile_goal,
)
from ail.judges import ScorePair, compute_agreement
from ail.readiness import (
    GateName,
    GoalView,
    JudgeFact,
    ReadinessFacts,
    ReadinessTier,
    compute_readiness,
)

# ---------------------------------------------------------------------------
# Mocked LLM + fixtures
# ---------------------------------------------------------------------------


def mock_llm(payload: dict | str, *, fence: bool = False) -> GoalProposerLLM:
    """A mock proposer returning a fixed response (JSON dict or raw string)."""
    text = json.dumps(payload) if isinstance(payload, dict) else payload
    if fence:
        text = f"```json\n{text}\n```"

    def _llm(*, system: str, user: str) -> str:
        return text

    return _llm


COST_PROPOSAL = {
    "objective_metric": "total_tokens",
    "direction": "minimize",
    "target": {"value": -0.30, "kind": "relative"},
    "guardrails": [{"name": "correctness", "kind": "judge", "must_not_regress": True}],
}
QUALITY_PROPOSAL = {
    "objective_metric": "correctness",
    "direction": "maximize",
    "target": {"value": 0.95, "kind": "absolute"},
    "guardrails": [{"name": "correctness", "kind": "judge", "threshold": 4}],
}


@pytest.fixture
def cohort() -> Cohort:
    return Cohort.by_agent("claude_code")


def _trusted_judge(name: str, *, n_scored: int = 60) -> JudgeFact:
    pairs = [ScorePair(item_id=str(i), human_value="yes", judge_value="yes") for i in range(10)]
    return JudgeFact.from_agreement_report(
        compute_agreement(pairs, judge_name=name), n_scored_traces=n_scored
    )


def _quality_facts(*judges: JudgeFact, trace_count: int = 60) -> ReadinessFacts:
    """Facts where the universal + quality data gates pass; ``judges`` set trust."""
    return ReadinessFacts(
        trace_count=trace_count,
        label_count=30,
        frozen_suite_present=True,
        n_scored_traces=trace_count,
        judge_runs=trace_count,
        judge_run_successes=trace_count,
        judges=judges,
    )


# ---------------------------------------------------------------------------
# Allowlist: derived from reality
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_l0_metrics_are_real_contract_fields(self) -> None:
        """Every L0 allowlist name is a real field on an ail.metrics.contract model."""
        from ail.metrics.contract import (
            AggregateMetrics,
            CostAggregate,
            TokenBreakdown,
            ToolRedundancy,
            TraceMetrics,
        )

        fields = set()
        for model in (
            TokenBreakdown,
            CostAggregate,
            ToolRedundancy,
            TraceMetrics,
            AggregateMetrics,
        ):
            fields |= set(model.model_fields)
        for name in L0_OBJECTIVE_METRICS:
            assert name in fields, f"{name} is not a real L0 contract field"

    def test_judge_names_come_from_the_registry(self) -> None:
        """Judge names are exactly the built-in scorer registry keys."""
        from ail.judges.scorers import DEFAULT_SCORERS

        assert JUDGE_METRICS == frozenset(DEFAULT_SCORERS)
        assert JUDGE_METRICS == {"correctness", "modularity", "groundedness", "token_efficiency"}

    def test_allowlist_is_the_union(self) -> None:
        assert ALLOWLIST == frozenset(L0_OBJECTIVE_METRICS) | JUDGE_METRICS

    def test_system_prompt_lists_the_allowlist(self) -> None:
        """The prompt is grounded in the live allowlist, not a hardcoded copy."""
        from ail.goals.compiler import _build_system_prompt

        prompt = _build_system_prompt()
        for name in ALLOWLIST:
            assert name in prompt


# ---------------------------------------------------------------------------
# Valid goals
# ---------------------------------------------------------------------------


class TestValidGoals:
    def test_token_cost_goal(self, cohort: Cohort) -> None:
        goal = compile_goal(
            "cut tokens ~30% without hurting correctness", cohort, llm=mock_llm(COST_PROPOSAL)
        )
        assert goal.objective_metric == "total_tokens"
        assert goal.direction == "minimize"
        assert goal.target.value == -0.30
        assert goal.target.kind == "relative"
        # a judge guardrail is involved, so quality is required and named
        assert goal.requires_quality is True
        assert tuple(goal.guardrail_names) == ("correctness",)
        assert goal.human_confirmed is False

    def test_quality_goal_objective_is_a_judge(self, cohort: Cohort) -> None:
        """A judged objective is required to also appear among the judge guardrails."""
        goal = compile_goal(
            "maximize correctness", cohort, llm=mock_llm(QUALITY_PROPOSAL, fence=True)
        )
        assert goal.objective_metric == "correctness"
        assert goal.requires_quality is True
        assert tuple(goal.guardrail_names) == ("correctness",)

    def test_pure_deterministic_goal_requires_no_quality(self, cohort: Cohort) -> None:
        """An L0 objective with only metric guardrails needs no judged signal."""
        proposal = {
            "objective_metric": "total_usd",
            "direction": "minimize",
            "target": {"value": -0.20, "kind": "relative"},
            "guardrails": [
                {"name": "duration_seconds", "kind": "metric", "must_not_regress": True}
            ],
        }
        goal = compile_goal("cut cost 20%, keep latency", cohort, llm=mock_llm(proposal))
        assert goal.requires_quality is False
        assert tuple(goal.guardrail_names) == ()

    def test_absolute_target(self, cohort: Cohort) -> None:
        proposal = {
            "objective_metric": "redundancy_rate",
            "direction": "minimize",
            "target": {"value": 0.05, "kind": "absolute"},
            "guardrails": [],
        }
        goal = compile_goal("get redundancy under 5%", cohort, llm=mock_llm(proposal))
        assert goal.target.kind == "absolute"
        assert goal.target.value == 0.05

    def test_metric_guardrails_do_not_leak_into_guardrail_names(self, cohort: Cohort) -> None:
        """guardrail_names exposes only judges (what readiness's judge gate consumes)."""
        proposal = {
            "objective_metric": "total_tokens",
            "direction": "minimize",
            "target": {"value": -0.25, "kind": "relative"},
            "guardrails": [
                {"name": "correctness", "kind": "judge", "must_not_regress": True},
                {"name": "total_usd", "kind": "metric", "must_not_regress": True},
            ],
        }
        goal = compile_goal("cut tokens, keep cost+quality", cohort, llm=mock_llm(proposal))
        # both guardrails are tracked...
        assert {g.name for g in goal.guardrails} == {"correctness", "total_usd"}
        # ...but only the judge is exposed to readiness
        assert tuple(goal.guardrail_names) == ("correctness",)


# ---------------------------------------------------------------------------
# Bound to a cohort
# ---------------------------------------------------------------------------


class TestCohortBinding:
    def test_bound_to_cohort_object(self, cohort: Cohort) -> None:
        goal = compile_goal("cut tokens", cohort, llm=mock_llm(COST_PROPOSAL))
        assert goal.cohort is cohort
        assert goal.cohort_name == cohort.name == "claude_code"

    def test_bound_to_cohort_name(self) -> None:
        goal = compile_goal("cut tokens", "nightly-regression", llm=mock_llm(COST_PROPOSAL))
        assert goal.cohort == "nightly-regression"
        assert goal.cohort_name == "nightly-regression"

    def test_llm_cannot_choose_the_cohort(self, cohort: Cohort) -> None:
        """The caller owns the cohort; a proposal that sets one fails loud."""
        bad = {**COST_PROPOSAL, "cohort": "attacker-chosen"}
        with pytest.raises(GoalCompileError, match="must not set 'cohort'"):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------


class TestHumanGate:
    def test_defaults_unconfirmed(self, cohort: Cohort) -> None:
        goal = compile_goal("cut tokens", cohort, llm=mock_llm(COST_PROPOSAL))
        assert goal.human_confirmed is False

    def test_confirm_flips_and_is_immutable(self, cohort: Cohort) -> None:
        goal = compile_goal("cut tokens", cohort, llm=mock_llm(COST_PROPOSAL))
        confirmed = goal.confirm()
        assert confirmed.human_confirmed is True
        # the original is untouched — confirmation produces a new record
        assert goal.human_confirmed is False
        assert confirmed is not goal

    def test_llm_cannot_self_confirm(self, cohort: Cohort) -> None:
        bad = {**COST_PROPOSAL, "human_confirmed": True}
        with pytest.raises(GoalCompileError, match="must not set 'human_confirmed'"):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))


# ---------------------------------------------------------------------------
# Fail loud: unmapped metrics + bad contracts
# ---------------------------------------------------------------------------


class TestFailLoud:
    def test_unmapped_objective_via_compile(self, cohort: Cohort) -> None:
        bad = {**COST_PROPOSAL, "objective_metric": "hallucination_rate"}
        with pytest.raises(UnmappedMetricError, match="not a known metric"):
            compile_goal("reduce hallucinations", cohort, llm=mock_llm(bad))

    def test_unmapped_objective_via_direct_construction(self) -> None:
        """Direct construction is guarded too — the same typed error, not a generic one."""
        with pytest.raises(UnmappedMetricError, match="not a known metric"):
            CompiledGoal(
                objective_metric="bogus_metric",
                direction="minimize",
                target={"value": -0.1, "kind": "relative"},
                cohort="c",
            )

    def test_judge_guardrail_name_must_be_a_judge(self, cohort: Cohort) -> None:
        bad = {
            **COST_PROPOSAL,
            "guardrails": [{"name": "total_tokens", "kind": "judge", "must_not_regress": True}],
        }
        with pytest.raises(UnmappedMetricError, match="kind 'judge' but is not a registered judge"):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))

    def test_metric_guardrail_name_must_be_an_l0_metric(self, cohort: Cohort) -> None:
        bad = {
            **COST_PROPOSAL,
            "guardrails": [{"name": "correctness", "kind": "metric", "must_not_regress": True}],
        }
        with pytest.raises(UnmappedMetricError, match="kind 'metric' but is not a known L0 metric"):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))

    def test_judged_objective_not_in_guardrails_fails(self, cohort: Cohort) -> None:
        """A judged objective with no matching guardrail breaks the readiness contract."""
        bad = {
            "objective_metric": "groundedness",
            "direction": "maximize",
            "target": {"value": 0.10, "kind": "relative"},
            "guardrails": [],
        }
        with pytest.raises(GoalContractError, match="must also be listed as a guardrail"):
            compile_goal("be more grounded", cohort, llm=mock_llm(bad))

    def test_guardrail_that_constrains_nothing_fails(self, cohort: Cohort) -> None:
        bad = {
            **COST_PROPOSAL,
            "guardrails": [{"name": "correctness", "kind": "judge"}],
        }
        with pytest.raises(GoalContractError, match="constrains nothing"):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))

    def test_relative_target_sign_must_match_direction(self, cohort: Cohort) -> None:
        bad = {
            **COST_PROPOSAL,
            "target": {"value": 0.30, "kind": "relative"},
        }  # minimize + positive
        with pytest.raises(GoalContractError, match="negative relative target"):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))

    def test_zero_relative_target_fails(self, cohort: Cohort) -> None:
        bad = {**COST_PROPOSAL, "target": {"value": 0.0, "kind": "relative"}}
        with pytest.raises(GoalContractError, match="non-zero"):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))

    def test_unknown_field_is_a_schema_error(self, cohort: Cohort) -> None:
        bad = {**COST_PROPOSAL, "surprise": 1}
        with pytest.raises(ValidationError):
            compile_goal("cut tokens", cohort, llm=mock_llm(bad))

    def test_non_json_llm_output_fails_loud(self, cohort: Cohort) -> None:
        with pytest.raises(GoalCompileError, match="did not return valid JSON"):
            compile_goal("cut tokens", cohort, llm=mock_llm("I cannot help with that."))

    def test_json_with_surrounding_prose_is_recovered(self, cohort: Cohort) -> None:
        """A model that wraps the object in prose still parses (outermost {...})."""
        raw = "Here is the goal:\n" + json.dumps(COST_PROPOSAL) + "\nHope that helps!"
        goal = compile_goal("cut tokens", cohort, llm=mock_llm(raw))
        assert goal.objective_metric == "total_tokens"

    def test_empty_nl_text_fails(self, cohort: Cohort) -> None:
        with pytest.raises(GoalCompileError, match="non-empty"):
            compile_goal("   ", cohort, llm=mock_llm(COST_PROPOSAL))


# ---------------------------------------------------------------------------
# Default LLM seam (no live call)
# ---------------------------------------------------------------------------


class TestDefaultProposer:
    def test_no_llm_and_no_endpoint_fails_loud(
        self, cohort: Cohort, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no injected llm and no configured endpoint, fail loud — never a live guess."""
        monkeypatch.delenv("AIL_GOAL_LLM_ENDPOINT", raising=False)
        with pytest.raises(GoalCompileError, match="no default endpoint configured"):
            compile_goal("cut tokens", cohort)


# ---------------------------------------------------------------------------
# Readiness GoalView compatibility (the cross-lane contract)
# ---------------------------------------------------------------------------


class TestReadinessContract:
    def test_compiled_goal_is_a_goalview(self, cohort: Cohort) -> None:
        goal = compile_goal("cut tokens", cohort, llm=mock_llm(COST_PROPOSAL))
        assert isinstance(goal, GoalView)
        # the three Protocol members are present and the right shape
        assert isinstance(goal.objective_metric, str)
        assert isinstance(goal.requires_quality, bool)
        assert tuple(goal.guardrail_names) == ("correctness",)

    def test_quality_goal_accepted_by_compute_readiness(self, cohort: Cohort) -> None:
        """A quality CompiledGoal feeds straight into compute_readiness and is READY_TO_PROVE.

        The judge_trusted gate keys off the goal's guardrail_names: the goal names
        'correctness', the facts carry a trusted 'correctness' judge, so the gate
        passes and the cohort can prove an improvement.
        """
        goal = compile_goal("cut tokens, keep correctness", cohort, llm=mock_llm(COST_PROPOSAL))
        facts = _quality_facts(_trusted_judge("correctness"))

        status = compute_readiness(cohort, goal, facts)

        assert status.tier is ReadinessTier.READY_TO_PROVE
        assert status.can_prove_improvement is True
        assert status.guardrail_names == ["correctness"]
        assert status.requires_quality is True
        judge_gate = status.gate_for(GateName.JUDGE_TRUSTED)
        assert judge_gate is not None and judge_gate.passed is True

    def test_judge_gate_keys_off_guardrail_names_no_substitution(self, cohort: Cohort) -> None:
        """The goal requires 'correctness'; a trusted *different* judge must not stand in.

        This is the proof that readiness keys the judge_trusted gate off the
        compiled goal's guardrail_names specifically — not any trusted judge.
        """
        goal = compile_goal("cut tokens, keep correctness", cohort, llm=mock_llm(COST_PROPOSAL))
        # only a trusted 'groundedness' judge is present — not the required 'correctness'
        facts = _quality_facts(_trusted_judge("groundedness"))

        status = compute_readiness(cohort, goal, facts)

        judge_gate = status.gate_for(GateName.JUDGE_TRUSTED)
        assert judge_gate is not None and judge_gate.passed is False
        assert "correctness" in judge_gate.reason
        assert status.can_prove_improvement is False

    def test_objective_as_judge_required_in_readiness(self, cohort: Cohort) -> None:
        """A correctness-objective goal makes readiness require the correctness judge."""
        goal = compile_goal("maximize correctness", cohort, llm=mock_llm(QUALITY_PROPOSAL))
        # distrusted correctness judge => gate fails
        facts = _quality_facts(JudgeFact(judge_name="correctness", agreement_rate=0.4))
        status = compute_readiness(cohort, goal, facts)
        judge_gate = status.gate_for(GateName.JUDGE_TRUSTED)
        assert judge_gate is not None and judge_gate.passed is False
        assert status.tier is ReadinessTier.BASELINE_ONLY

    def test_deterministic_goal_skips_quality_gates_in_readiness(self, cohort: Cohort) -> None:
        """A pure token/cost goal needs no judge — readiness evaluates no quality gates."""
        proposal = {
            "objective_metric": "total_tokens",
            "direction": "minimize",
            "target": {"value": -0.30, "kind": "relative"},
            "guardrails": [{"name": "total_usd", "kind": "metric", "must_not_regress": True}],
        }
        goal = compile_goal("cut tokens, keep cost", cohort, llm=mock_llm(proposal))
        facts = ReadinessFacts(trace_count=60, frozen_suite_present=True)
        status = compute_readiness(cohort, goal, facts)

        assert status.requires_quality is False
        assert status.tier is ReadinessTier.READY_TO_PROVE
        assert status.gate_for(GateName.JUDGE_TRUSTED) is None


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


class TestContractShape:
    def test_models_forbid_unknown_fields(self) -> None:
        from ail.goals.compiler import GoalTarget, Guardrail

        for model in (CompiledGoal, GoalTarget, Guardrail):
            assert model.model_config.get("extra") == "forbid"

    def test_compiled_goal_is_frozen(self, cohort: Cohort) -> None:
        goal = compile_goal("cut tokens", cohort, llm=mock_llm(COST_PROPOSAL))
        with pytest.raises(ValidationError):
            goal.objective_metric = "total_usd"  # type: ignore[misc]
