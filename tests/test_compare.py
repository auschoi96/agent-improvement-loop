"""Tests for the candidate-vs-baseline comparison harness (:mod:`ail.compare`).

All tests are offline. The two model-touching seams are mocked:

* the **agent** is a ``ScriptedAdapter`` that returns a pre-built
  :class:`~ail.ingest.base.AgentRunResult` (baseline vs candidate keyed off a
  marker the intervention writes into the task), so no agent runtime is needed;
* the **correctness judge** is a ``FakeJudge`` (a duck-typed callable with a
  scripted verdict), so the guardrail is exercised with no model call.

The fixture Task Suite is a tiny in-memory :class:`~ail.groundtruth.schema.GroundTruthCase`;
nothing depends on the real frozen suite being populated. The three mandated
decision cases are covered (PROMOTE on a token drop with correctness held; BLOCK
on a token drop with correctness regressed; no false PROMOTE on no token change),
plus fail-closed scoring, the optional L1 guardrail, suite immutability, and the
monitoring-warehouse helper.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import pytest

from ail.compare import (
    CORRECTNESS_GUARDRAIL,
    INTERIM_JUDGE_NOTE,
    PROGRAMMATIC_GUARDRAIL,
    CallableIntervention,
    ComparisonConfig,
    ComparisonResult,
    ProgrammaticSignal,
    Recommendation,
    compare_candidate,
    configure_monitoring_warehouse,
)
from ail.compare.monitoring import MONITORING_WAREHOUSE_TAG, TRACING_WAREHOUSE_ENV
from ail.groundtruth.schema import (
    Expectations,
    GroundTruthCase,
    Pool,
    Source,
    SourceKind,
    TaskInput,
)
from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedTrace,
    TokenUsage,
    ToolCall,
    TraceStatus,
)

# ---------------------------------------------------------------------------
# Test doubles + fixtures
# ---------------------------------------------------------------------------

_CANDIDATE_MARKER = "variant"


class _Feedback:
    """Duck-typed stand-in for an MLflow ``Feedback`` (exposes ``.value``)."""

    def __init__(self, value: Any, rationale: str = "") -> None:
        self.value = value
        self.rationale = rationale


class FakeJudge:
    """A scripted correctness judge: maps a run's ``outputs`` to a verdict.

    ``responses`` maps an outputs value to whatever the judge "returns" (a raw
    ``"yes"``/``"no"``, a ``_Feedback``, ``None``, ...). ``raise_on`` names
    outputs values for which the call raises (to exercise fail-closed scoring).
    """

    name = "correctness"

    def __init__(self, responses: dict[str, Any], *, raise_on: set[str] | None = None) -> None:
        self.responses = responses
        self.raise_on = raise_on or set()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, inputs: Any = None, outputs: Any = None, expectations: Any = None) -> Any:
        self.calls.append({"inputs": inputs, "outputs": outputs, "expectations": expectations})
        if outputs in self.raise_on:
            raise RuntimeError(f"judge boom on {outputs!r}")
        return self.responses[outputs]


class ScriptedAdapter(AgentAdapter):
    """Returns a pre-built result for the baseline task and another for the candidate.

    The candidate task is told apart by the marker the intervention writes into
    ``task.params`` — so this exercises the real WITH/WITHOUT execution path
    (two ``run`` calls with different tasks) rather than faking the split.
    """

    name = "scripted"

    def __init__(self, baseline: AgentRunResult, candidate: AgentRunResult) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.tasks: list[AgentTask] = []

    def run(self, task: AgentTask) -> AgentRunResult:
        self.tasks.append(task)
        if task.params.get(_CANDIDATE_MARKER) == "candidate":
            return self.candidate
        return self.baseline


def _mark_candidate(task: AgentTask) -> AgentTask:
    """A pure intervention: tag the task as the candidate variant (new task)."""
    return replace(task, params={**task.params, _CANDIDATE_MARKER: "candidate"})


def use_tool_intervention() -> CallableIntervention:
    return CallableIntervention(name="point-agent-at-metric-view-tool", transform=_mark_candidate)


def _case(prompt: str = "What is 6 times 7?", case_id: str = "suite-1") -> GroundTruthCase:
    """A tiny in-memory frozen Task-Suite case (task input + expectations)."""
    return GroundTruthCase(
        case_id=case_id,
        task_input=TaskInput(prompt=prompt),
        sources=[Source(kind=SourceKind.HUMAN, ref="human-anchor-row-1")],
        expectations=Expectations(expected_response="42", must_include=["42"]),
        regression_intent="guards correctness while reducing tokens",
        target_pool=Pool.TASK_SUITE,
    )


def _run(
    *,
    trace_id: str,
    output: str,
    input_tokens: int,
    output_tokens: int = 0,
    tool_calls: list[ToolCall] | None = None,
    success: bool = True,
    model: str | None = "claude-opus-4-8",
) -> AgentRunResult:
    trace = NormalizedTrace(
        trace_id=trace_id,
        status=TraceStatus.OK if success else TraceStatus.ERROR,
        producer="scripted",
        model=model,
        token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        tool_calls=tool_calls or [],
    )
    return AgentRunResult(trace=trace, output_text=output, success=success)


def _reads(path: str, n: int) -> list[ToolCall]:
    """``n`` byte-identical Read calls of ``path`` (redundant after the first)."""
    return [ToolCall(id=f"r{i}", name="Read", arguments={"file_path": path}) for i in range(n)]


# ---------------------------------------------------------------------------
# The three mandated decision cases
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_token_drop_with_correctness_held_promotes(self) -> None:
        baseline = _run(trace_id="base", output="long baseline answer: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="tight candidate answer: 42", input_tokens=60_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"long baseline answer: 42": "yes", "tight candidate answer: 42": "yes"})

        result = compare_candidate(
            _case(),
            adapter,
            intervention=use_tool_intervention(),
            correctness_judge=judge,
            generated_at="2026-06-29T00:00:00+00:00",
        )

        assert result.recommendation is Recommendation.PROMOTE
        assert result.objective_met is True
        assert result.guardrails_passed is True
        tokens = result.delta_for("total_tokens")
        assert tokens is not None
        assert tokens.delta_absolute == -40_000.0
        assert tokens.delta_pct == -40.0
        assert tokens.improved is True
        guardrail = result.guardrail_for(CORRECTNESS_GUARDRAIL)
        assert guardrail is not None and guardrail.passed and guardrail.regressed is False
        assert result.intervention == "point-agent-at-metric-view-tool"
        assert result.baseline_trace_id == "base" and result.candidate_trace_id == "cand"

    def test_token_drop_but_correctness_regressed_blocks(self) -> None:
        baseline = _run(trace_id="base", output="correct baseline: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="cheaper but wrong: 41", input_tokens=55_000)
        adapter = ScriptedAdapter(baseline, candidate)
        # Baseline was correct; the cheaper candidate is wrong -> a regression.
        judge = FakeJudge({"correct baseline: 42": "yes", "cheaper but wrong: 41": "no"})

        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )

        assert result.recommendation is Recommendation.BLOCK
        # The objective (token drop) WAS met; the guardrail is what blocks.
        assert result.objective_met is True
        assert result.guardrails_passed is False
        guardrail = result.guardrail_for(CORRECTNESS_GUARDRAIL)
        assert guardrail is not None
        assert guardrail.passed is False and guardrail.regressed is True
        assert guardrail.baseline_value == "yes" and guardrail.candidate_value == "no"
        assert any("REGRESSED" in r for r in result.reasons)

    def test_no_token_change_does_not_falsely_promote(self) -> None:
        # No intervention -> the same baseline task runs twice -> identical tokens.
        baseline = _run(trace_id="base", output="same answer: 42", input_tokens=80_000)
        adapter = ScriptedAdapter(baseline, baseline)
        judge = FakeJudge({"same answer: 42": "yes"})  # correctness held (and irrelevant)

        result = compare_candidate(_case(), adapter, intervention=None, correctness_judge=judge)

        assert result.recommendation is not Recommendation.PROMOTE
        assert result.recommendation is Recommendation.BLOCK
        assert result.objective_met is False
        # Correctness did not regress; the block is purely "objective not met".
        assert result.guardrails_passed is True
        tokens = result.delta_for("total_tokens")
        assert tokens is not None and tokens.delta_absolute == 0.0 and tokens.improved is False
        assert any("objective NOT met" in r for r in result.reasons)
        assert result.intervention is None


# ---------------------------------------------------------------------------
# Guardrail behaviour
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_fail_closed_when_candidate_unscorable(self) -> None:
        # A real token drop, but the judge cannot score the candidate -> no promote.
        baseline = _run(trace_id="base", output="baseline: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="candidate: 42", input_tokens=50_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"baseline: 42": "yes", "candidate: 42": None})

        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )

        assert result.objective_met is True
        assert result.recommendation is Recommendation.BLOCK
        guardrail = result.guardrail_for(CORRECTNESS_GUARDRAIL)
        assert guardrail is not None and guardrail.passed is False
        assert guardrail.regressed is False  # un-measurable, not a measured regression
        assert "failing closed" in guardrail.reason

    def test_fail_closed_when_judge_raises(self) -> None:
        baseline = _run(trace_id="base", output="baseline: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="candidate: 42", input_tokens=50_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"baseline: 42": "yes"}, raise_on={"candidate: 42"})

        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        assert result.recommendation is Recommendation.BLOCK
        guardrail = result.guardrail_for(CORRECTNESS_GUARDRAIL)
        assert guardrail is not None and guardrail.passed is False

    def test_correctness_held_via_feedback_object(self) -> None:
        # The judge may return an MLflow ``Feedback``; coercion must unwrap it.
        baseline = _run(trace_id="base", output="baseline: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="candidate: 42", input_tokens=70_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge(
            {"baseline: 42": _Feedback("yes"), "candidate: 42": _Feedback("yes", "looks right")}
        )
        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        assert result.recommendation is Recommendation.PROMOTE
        guardrail = result.guardrail_for(CORRECTNESS_GUARDRAIL)
        assert guardrail is not None and guardrail.baseline_value == "yes"

    def test_baseline_already_wrong_is_not_a_regression(self) -> None:
        # If the baseline was wrong and the candidate is also wrong, correctness did
        # not regress: the non-regression guardrail passes (it guards against the
        # intervention making things worse, not pre-existing deficiency).
        baseline = _run(trace_id="base", output="wrong baseline", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="still wrong", input_tokens=40_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"wrong baseline": "no", "still wrong": "no"})
        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        guardrail = result.guardrail_for(CORRECTNESS_GUARDRAIL)
        assert guardrail is not None and guardrail.passed is True and guardrail.regressed is False
        # Objective met + guardrail held -> promote (correctness did not get worse).
        assert result.recommendation is Recommendation.PROMOTE

    def test_correctness_guardrail_is_flagged_interim(self) -> None:
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        guardrail = result.guardrail_for(CORRECTNESS_GUARDRAIL)
        assert guardrail is not None
        assert guardrail.interim is True
        assert guardrail.interim_note == INTERIM_JUDGE_NOTE
        assert guardrail.judge_name == "correctness"

    def test_programmatic_l1_regression_blocks_even_with_token_drop(self) -> None:
        baseline = _run(trace_id="base", output="baseline: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="candidate: 42", input_tokens=50_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"baseline: 42": "yes", "candidate: 42": "yes"})

        def check(result: AgentRunResult) -> ProgrammaticSignal:
            # Tests passed at baseline, fail for the candidate -> an L1 regression.
            passed = result.trace.trace_id != "cand"
            return ProgrammaticSignal(name="pytest", passed=passed, details="suite")

        result = compare_candidate(
            _case(),
            adapter,
            intervention=use_tool_intervention(),
            correctness_judge=judge,
            programmatic_check=check,
        )
        assert result.objective_met is True
        assert result.recommendation is Recommendation.BLOCK
        l1 = result.guardrail_for(PROGRAMMATIC_GUARDRAIL)
        assert l1 is not None and l1.passed is False and l1.regressed is True
        assert l1.interim is False  # L1 is an objective signal, not an interim judge

    def test_programmatic_l1_held_allows_promote(self) -> None:
        baseline = _run(trace_id="base", output="baseline: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="candidate: 42", input_tokens=50_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"baseline: 42": "yes", "candidate: 42": "yes"})
        result = compare_candidate(
            _case(),
            adapter,
            intervention=use_tool_intervention(),
            correctness_judge=judge,
            programmatic_check=lambda r: ProgrammaticSignal(name="pytest", passed=True),
        )
        assert result.recommendation is Recommendation.PROMOTE
        assert len(result.guardrails) == 2


# ---------------------------------------------------------------------------
# Objective threshold, deltas, and execution
# ---------------------------------------------------------------------------


class TestObjectiveAndDeltas:
    def test_reduction_below_threshold_does_not_meet_objective(self) -> None:
        # 40% drop, but a 50% minimum is required -> objective not met -> BLOCK.
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        result = compare_candidate(
            _case(),
            adapter,
            intervention=use_tool_intervention(),
            correctness_judge=judge,
            config=ComparisonConfig(min_token_reduction_pct=50.0),
        )
        assert result.objective_met is False
        assert result.guardrails_passed is True
        assert result.recommendation is Recommendation.BLOCK

    def test_cost_objective_metric(self) -> None:
        baseline = _run(trace_id="base", output="b: 42", input_tokens=1_000_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=400_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        result = compare_candidate(
            _case(),
            adapter,
            intervention=use_tool_intervention(),
            correctness_judge=judge,
            config=ComparisonConfig(objective_metric="total_usd"),
        )
        cost = result.delta_for("total_usd")
        assert cost is not None and cost.baseline == 5.0 and cost.candidate == 2.0
        assert result.objective_met is True
        assert result.recommendation is Recommendation.PROMOTE

    def test_unknown_objective_metric_raises(self) -> None:
        adapter = ScriptedAdapter(
            _run(trace_id="b", output="x", input_tokens=10),
            _run(trace_id="c", output="x", input_tokens=5),
        )
        with pytest.raises(ValueError, match="objective_metric"):
            compare_candidate(
                _case(),
                adapter,
                correctness_judge=FakeJudge({"x": "yes"}),
                config=ComparisonConfig(objective_metric="nonsense"),
            )

    def test_redundancy_delta_is_computed(self) -> None:
        baseline = _run(
            trace_id="base", output="b: 42", input_tokens=100_000, tool_calls=_reads("a.py", 4)
        )
        candidate = _run(
            trace_id="cand", output="c: 42", input_tokens=60_000, tool_calls=_reads("a.py", 1)
        )
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        redundancy = result.delta_for("redundancy_rate")
        assert redundancy is not None
        assert redundancy.baseline == 0.75 and redundancy.candidate == 0.0  # 3/4 vs 0/1
        assert redundancy.improved is True

    def test_unpriced_model_flags_cost_delta(self) -> None:
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000, model="mystery-llm")
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000, model="mystery-llm")
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        assert any("cost delta is partial" in n for n in result.notes)

    def test_runs_agent_twice_with_and_without_intervention(self) -> None:
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        assert len(adapter.tasks) == 2
        # First run is the baseline (no marker); second is the candidate (marked).
        assert adapter.tasks[0].params.get(_CANDIDATE_MARKER) is None
        assert adapter.tasks[1].params.get(_CANDIDATE_MARKER) == "candidate"

    def test_failed_run_is_noted(self) -> None:
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000, success=False)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        assert any("reported failure" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Frozen-suite contract + serialization
# ---------------------------------------------------------------------------


class TestFrozenSuiteAndContract:
    def test_does_not_mutate_the_suite_case(self) -> None:
        case = _case()
        before = case.model_dump()
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})

        compare_candidate(
            case, adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )

        # The frozen Task-Suite case is byte-for-byte unchanged: the harness reads
        # it, never writes/re-pools/persists it. (It is also pydantic-frozen, so a
        # write would raise — this asserts no copy with edits leaked back either.)
        assert case.model_dump() == before
        assert case.candidate_response is None
        assert case.expectations.expected_response == "42"

    def test_judge_sees_task_request_and_human_expectations(self) -> None:
        # The judge's inputs are the task request (identical for both runs); only
        # the agent's outputs differ. Expectations come from the human-authored
        # case, never synthesized.
        case = _case(prompt="What is 6 times 7?")
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        compare_candidate(
            case, adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        assert {c["inputs"] for c in judge.calls} == {"What is 6 times 7?"}
        assert {c["outputs"] for c in judge.calls} == {"b: 42", "c: 42"}
        assert all(c["expectations"]["expected_response"] == "42" for c in judge.calls)

    def test_result_round_trips_through_json(self) -> None:
        baseline = _run(trace_id="base", output="b: 42", input_tokens=100_000)
        candidate = _run(trace_id="cand", output="c: 42", input_tokens=60_000)
        adapter = ScriptedAdapter(baseline, candidate)
        judge = FakeJudge({"b: 42": "yes", "c: 42": "yes"})
        result = compare_candidate(
            _case(), adapter, intervention=use_tool_intervention(), correctness_judge=judge
        )
        reloaded = ComparisonResult.model_validate_json(result.model_dump_json())
        assert reloaded == result

    def test_contract_forbids_unknown_fields(self) -> None:
        with pytest.raises(Exception, match="extra"):
            ComparisonResult.model_validate({"task_id": "t", "bogus_field": 1})


# ---------------------------------------------------------------------------
# Monitoring SQL warehouse helper
# ---------------------------------------------------------------------------


class _FakeClient:
    """Records ``set_experiment_tag`` calls (duck-typed ``MlflowClient``)."""

    def __init__(self) -> None:
        self.tags: list[tuple[str, str, str]] = []

    def set_experiment_tag(self, experiment_id: str, key: str, value: str) -> None:
        self.tags.append((experiment_id, key, value))


class TestConfigureMonitoringWarehouse:
    def test_sets_experiment_tag_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
        client = _FakeClient()
        cfg = configure_monitoring_warehouse("exp-123", "wh-abc", client=client)

        assert client.tags == [("exp-123", MONITORING_WAREHOUSE_TAG, "wh-abc")]
        assert os.environ[TRACING_WAREHOUSE_ENV] == "wh-abc"
        assert cfg.experiment_id == "exp-123" and cfg.warehouse_id == "wh-abc"
        assert cfg.tag_key == MONITORING_WAREHOUSE_TAG

    def test_set_env_false_leaves_environment_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
        client = _FakeClient()
        configure_monitoring_warehouse("exp-1", "wh-1", client=client, set_env=False)
        assert TRACING_WAREHOUSE_ENV not in os.environ
        assert client.tags == [("exp-1", MONITORING_WAREHOUSE_TAG, "wh-1")]

    @pytest.mark.parametrize("experiment_id,warehouse_id", [("", "wh"), ("exp", ""), ("  ", "wh")])
    def test_rejects_blank_inputs(self, experiment_id: str, warehouse_id: str) -> None:
        with pytest.raises(ValueError):
            configure_monitoring_warehouse(experiment_id, warehouse_id, client=_FakeClient())
