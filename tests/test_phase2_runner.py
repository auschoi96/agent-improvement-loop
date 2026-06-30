"""Tests for the Phase-2 comparison runner (:mod:`ail.optimize.phase2`).

Fully offline. The agent is a ``SuiteAdapter`` that returns pre-built traces
(baseline vs candidate told apart by the injected skill marker in the system
prompt). L1 verification is exercised with real but trivial shell commands
(``true`` / ``false`` / a stateful script / a missing binary) so the
:func:`~ail.optimize.phase2.make_command_check` path runs end-to-end without any
network, model, or live agent.

The fail-closed checklist is asserted explicitly: a crashed candidate BLOCKS and
is never counted as a token win; a task with no verification BLOCKS; an
un-runnable verification BLOCKS; a comparison that raises BLOCKS. None of these
read as a pass or inflate the realized savings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ail.compare import Recommendation
from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedTrace,
    TokenUsage,
    TraceStatus,
)
from ail.optimize import VerifySpec, case_from_task, run_phase2_comparison
from ail.optimize.phase2 import L1Outcome, Phase2Artifact
from ail.task_suite.schema import Difficulty, Task, TaskCategory, TaskSuite

_STAMP = "2026-06-29T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Test doubles + fixtures
# ---------------------------------------------------------------------------


class Arm:
    """A pre-built run outcome for one arm (baseline or candidate)."""

    def __init__(self, tokens: int, *, success: bool = True, output: str = "done") -> None:
        self.tokens = tokens
        self.success = success
        self.output = output


class SuiteAdapter(AgentAdapter):
    """Return a scripted result per task, baseline vs candidate keyed on the skill marker."""

    name = "scripted-suite"

    def __init__(self, plans: dict[str, tuple[Arm, Arm]]) -> None:
        self.plans = plans  # prompt -> (baseline arm, candidate arm)
        self.seen: list[AgentTask] = []

    def run(self, task: AgentTask) -> AgentRunResult:
        self.seen.append(task)
        baseline_arm, candidate_arm = self.plans[task.prompt]
        is_candidate = "<skill" in (task.system_prompt or "")
        arm = candidate_arm if is_candidate else baseline_arm
        trace_id = ("cand" if is_candidate else "base") + f"-{task.prompt[:6]}"
        trace = NormalizedTrace(
            trace_id=trace_id,
            status=TraceStatus.OK if arm.success else TraceStatus.ERROR,
            producer=self.name,
            model="claude-opus-4-8",
            token_usage=TokenUsage(input_tokens=arm.tokens),
        )
        return AgentRunResult(
            trace=trace,
            output_text=arm.output,
            success=arm.success,
            error=None if arm.success else "candidate crashed",
        )


def _task(prompt: str, task_id: str = "ts-001") -> Task:
    return Task(
        task_id=task_id,
        prompt=prompt,
        category=TaskCategory.REPEATED_TARGET_BOILERPLATE,
        source_trace_id=f"src-{task_id}",
        difficulty=Difficulty.MEDIUM,
    )


def _suite(*tasks: Task) -> TaskSuite:
    return TaskSuite(version="test-v1", tasks=tuple(tasks)).freeze()


def _pass_spec() -> dict[str, VerifySpec]:
    return {"ts-001": VerifySpec(name="ok", command=["true"])}


# ---------------------------------------------------------------------------
# The Task -> case bridge
# ---------------------------------------------------------------------------


class TestCaseBridge:
    def test_case_from_task_carries_prompt_and_empty_expectations(self) -> None:
        task = _task("do the work", task_id="ts-009")
        case = case_from_task(task)
        assert case.case_id == "ts-009"
        assert case.task_input.prompt == "do the work"
        # The frozen suite has no human expectations — exactly why correctness is L1.
        assert case.expectations.is_filled() is False
        assert case.sources[0].ref == "src-ts-009"
        assert case.tags["category"] == TaskCategory.REPEATED_TARGET_BOILERPLATE.value


# ---------------------------------------------------------------------------
# Happy path + provenance
# ---------------------------------------------------------------------------


class TestPromote:
    def test_token_drop_with_l1_pass_promotes(self) -> None:
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        assert artifact.n_tasks == 1 and artifact.n_promote == 1 and artifact.n_block == 0
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.PROMOTE
        assert out.objective_met is True and out.token_improved is True
        assert out.token_delta_absolute == -40_000.0 and out.token_delta_pct == -40.0
        assert out.l1_outcome is L1Outcome.PASSED and out.l1_verification_configured is True
        assert out.baseline_succeeded and out.candidate_succeeded
        # Realized savings counted over the promoted task.
        assert artifact.realized_token_savings_absolute == 40_000.0
        assert artifact.realized_token_savings_pct == 40.0

    def test_records_run_provenance(self) -> None:
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=_pass_spec(),
            experiment="exp-name",
            profile="dais-demo",
            warehouse_id="wh-123",
            generated_at=_STAMP,
        )
        assert artifact.experiment == "exp-name"
        assert artifact.profile == "dais-demo"
        assert artifact.warehouse_id == "wh-123"
        assert artifact.suite_version == "test-v1"
        assert artifact.suite_content_hash  # the frozen suite's hash, recorded
        assert artifact.baseline_config == "baseline-no-asset"
        assert artifact.candidate_config == "candidate-token-efficiency-skill"

    def test_runs_baseline_and_candidate_arms(self) -> None:
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        # Two runs: baseline (no skill) then candidate (skill injected).
        assert len(adapter.seen) == 2
        assert "<skill" not in (adapter.seen[0].system_prompt or "")
        assert "<skill" in (adapter.seen[1].system_prompt or "")

    def test_artifact_round_trips_through_json(self) -> None:
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        reloaded = Phase2Artifact.model_validate_json(artifact.model_dump_json())
        assert reloaded == artifact


# ---------------------------------------------------------------------------
# Fail-closed checklist
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_crashed_candidate_is_blocked_not_a_token_win(self) -> None:
        # MANDATED: a crashed candidate uses ~0 tokens *because it did nothing*. Its
        # apparent token reduction must BLOCK and never count as a win.
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(500, success=False))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.token_improved is True  # tokens did fall...
        assert out.objective_met is True
        assert out.recommendation is Recommendation.BLOCK  # ...but it is BLOCKED
        assert out.candidate_succeeded is False
        assert any("execution" in r for r in out.blocking_reasons)
        # The crash is NOT aggregated into realized savings.
        assert artifact.n_promote == 0 and artifact.n_block == 1
        assert artifact.realized_token_savings_absolute == 0.0
        assert artifact.realized_token_savings_pct is None

    def test_no_verification_configured_blocks(self) -> None:
        # No L1 check => no correctness signal => fail closed (never a silent pass).
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=None,  # nothing configured
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.l1_outcome is L1Outcome.NOT_CONFIGURED
        assert out.l1_verification_configured is False
        assert artifact.realized_token_savings_absolute == 0.0
        assert any("no L1 verification configured" in n for n in artifact.notes)

    def test_unrunnable_verification_blocks(self) -> None:
        # A verification command that cannot be launched => errored => fail closed,
        # even though tokens dropped. A broken verifier never reads as "passed".
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs={
                "ts-001": VerifySpec(name="missing", command=["no-such-binary-xyz-12345"])
            },
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.objective_met is True
        assert out.recommendation is Recommendation.BLOCK
        assert out.l1_outcome is L1Outcome.NO_VERDICT
        assert artifact.realized_token_savings_absolute == 0.0

    def test_l1_regression_blocks(self, tmp_path: Path) -> None:
        # A stateful verifier that passes on the first (baseline) call and fails on
        # the second (candidate) call => the lever broke a check that passed => BLOCK.
        script = tmp_path / "verify.sh"
        marker = tmp_path / "marker"
        script.write_text(
            f'#!/bin/sh\nif [ -e "{marker}" ]; then exit 1; fi\n: > "{marker}"\nexit 0\n',
            encoding="utf-8",
        )
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs={"ts-001": VerifySpec(name="regress", command=["sh", str(script)])},
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.l1_outcome is L1Outcome.REGRESSED
        assert artifact.realized_token_savings_absolute == 0.0

    def test_comparison_that_raises_is_blocked_not_a_pass(self) -> None:
        # An adapter that raises (programmer-ish error mid-run) must record a blocked,
        # errored outcome for that task — never a pass — and not abort the whole run.
        class BoomAdapter(AgentAdapter):
            name = "boom"

            def run(self, task: AgentTask) -> AgentRunResult:
                raise RuntimeError("adapter exploded")

        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=BoomAdapter(),
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.error is not None and "adapter exploded" in out.error
        assert out.comparison is None
        assert out.objective_met is False and out.token_improved is False
        assert artifact.n_errored == 1 and artifact.n_promote == 0
        assert artifact.realized_token_savings_absolute == 0.0

    def test_failed_baseline_blocks(self) -> None:
        # A failed baseline makes the comparison untrustworthy => BLOCK.
        adapter = SuiteAdapter({"p1": (Arm(100_000, success=False), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.baseline_succeeded is False


# ---------------------------------------------------------------------------
# Multi-task aggregation + config validation
# ---------------------------------------------------------------------------


class TestAggregationAndConfig:
    def test_mixed_outcomes_only_count_promotes_in_realized_savings(self) -> None:
        plans = {
            "p1": (Arm(100_000), Arm(60_000)),  # promote (-40k)
            "p2": (Arm(80_000), Arm(500, success=False)),  # crash -> block, not counted
            "p3": (Arm(50_000), Arm(30_000)),  # block: no verification configured
        }
        adapter = SuiteAdapter(plans)
        suite = _suite(
            _task("p1", "ts-001"),
            _task("p2", "ts-002"),
            _task("p3", "ts-003"),
        )
        artifact = run_phase2_comparison(
            suite=suite,
            adapter=adapter,
            verify_specs={
                "ts-001": VerifySpec(name="ok", command=["true"]),
                "ts-002": VerifySpec(name="ok", command=["true"]),
                # ts-003 deliberately unverified
            },
            generated_at=_STAMP,
        )
        assert artifact.n_tasks == 3
        assert artifact.n_promote == 1
        assert artifact.n_block == 2
        # Only the single promoted task's delta is realized.
        assert artifact.realized_token_savings_absolute == 40_000.0
        assert artifact.realized_baseline_tokens == 100_000.0
        assert artifact.realized_candidate_tokens == 60_000.0

    def test_task_id_filter_runs_subset(self) -> None:
        adapter = SuiteAdapter(
            {"p1": (Arm(100_000), Arm(60_000)), "p2": (Arm(80_000), Arm(40_000))}
        )
        suite = _suite(_task("p1", "ts-001"), _task("p2", "ts-002"))
        artifact = run_phase2_comparison(
            suite=suite,
            adapter=adapter,
            verify_specs={
                "ts-001": VerifySpec(name="ok", command=["true"]),
                "ts-002": VerifySpec(name="ok", command=["true"]),
            },
            task_ids={"ts-002"},
            generated_at=_STAMP,
        )
        assert artifact.n_tasks == 1
        assert artifact.outcomes[0].task_id == "ts-002"

    def test_objective_threshold_blocks_small_reduction(self) -> None:
        from ail.compare import ComparisonConfig

        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(95_000))})  # only 5% drop
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            verify_specs=_pass_spec(),
            config=ComparisonConfig(min_token_reduction_pct=10.0),
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.objective_met is False
        assert out.recommendation is Recommendation.BLOCK

    def test_candidate_without_intervention_raises(self) -> None:
        from ail.optimize import BASELINE

        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        with pytest.raises(ValueError, match="no intervention"):
            run_phase2_comparison(
                suite=_suite(_task("p1")),
                adapter=adapter,
                candidate=BASELINE,  # not a candidate
                verify_specs=_pass_spec(),
            )

    def test_baseline_with_intervention_raises(self) -> None:
        from ail.optimize import CANDIDATE

        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        with pytest.raises(ValueError, match="carries an intervention"):
            run_phase2_comparison(
                suite=_suite(_task("p1")),
                adapter=adapter,
                baseline=CANDIDATE,  # baseline must have no asset
                verify_specs=_pass_spec(),
            )
