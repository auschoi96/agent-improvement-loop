"""Tests for the Phase A-0 advisory-memory spike (:mod:`ail.memory`).

Fully offline. No live model call, no real agent arm: the intervention and
provenance guard are exercised directly, the loader against in-memory parsed
entries, and the end-to-end harness flow with a scripted ``SuiteAdapter`` (the
same mocking pattern as ``tests/test_phase2_runner.py``) that tells baseline from
candidate apart by the injected ``<learnings`` marker.

The fail-closed contract is asserted explicitly: empty memory makes ``apply`` a
no-op returning the *same* task (identical to the baseline), and the provenance
wall raises when the memory source overlaps the frozen suite.
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
from ail.l3.contract import RankedAsset
from ail.memory import (
    MEMORY_CANDIDATE,
    MemoryInjectionIntervention,
    assert_memory_disjoint_from_suite,
    build_memory_learnings,
    format_advisory_line,
    load_ranked_assets,
    select_top_k,
)
from ail.memory.config import build_memory_candidate
from ail.optimize import VerifySpec, run_phase2_comparison
from ail.optimize.lever import LeverConfig
from ail.pools import PoolOverlapError
from ail.task_suite.schema import Difficulty, Task, TaskCategory, TaskSuite

_STAMP = "2026-06-29T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


def _asset(
    title: str,
    *,
    occurrences: int,
    rank: int = 1,
    trace_ids: list[str] | None = None,
    expected_benefits: list[str] | None = None,
    rationales: list[str] | None = None,
) -> RankedAsset:
    return RankedAsset(
        asset_type="skill",
        title=title,
        rank=rank,
        n_traces=len(trace_ids or []),
        occurrences=occurrences,
        trace_ids=trace_ids or [],
        expected_benefits=expected_benefits or [],
        rationales=rationales or [],
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


def _memory_candidate(*learnings: str) -> LeverConfig:
    """A LeverConfig carrying a memory intervention (bypasses the disk-loaded default)."""
    return LeverConfig(
        name="candidate-advisory-memory",
        intervention=MemoryInjectionIntervention(learnings=tuple(learnings)),
    )


class Arm:
    """A pre-built run outcome for one arm (baseline or candidate)."""

    def __init__(self, tokens: int, *, success: bool = True, output: str = "done") -> None:
        self.tokens = tokens
        self.success = success
        self.output = output


class SuiteAdapter(AgentAdapter):
    """Scripted result per task; baseline vs candidate keyed on the memory marker."""

    name = "scripted-suite"

    def __init__(self, plans: dict[str, tuple[Arm, Arm]]) -> None:
        self.plans = plans
        self.seen: list[AgentTask] = []

    def run(self, task: AgentTask) -> AgentRunResult:
        self.seen.append(task)
        baseline_arm, candidate_arm = self.plans[task.prompt]
        is_candidate = "<learnings" in (task.system_prompt or "")
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
            error=None if arm.success else "arm crashed",
        )


# ---------------------------------------------------------------------------
# Intervention: additive injection + fail-closed no-op
# ---------------------------------------------------------------------------


class TestInterventionIsolation:
    def test_injects_into_empty_system_prompt_and_nothing_else(self) -> None:
        iv = MemoryInjectionIntervention(learnings=("- cache reads", "- batch commands"))
        task = AgentTask(prompt="implement X", model="m", allowed_tools=["Read"])
        out = iv.apply(task)
        assert out is not task  # pure: a new task
        assert out.system_prompt is not None
        assert out.system_prompt.startswith('<learnings source="prior-sessions">')
        assert "- cache reads" in out.system_prompt and "- batch commands" in out.system_prompt
        # Every other field is identical — memory is the only controlled difference.
        assert out.prompt == task.prompt
        assert out.model == task.model
        assert out.allowed_tools == task.allowed_tools
        assert out.cwd == task.cwd
        assert out.timeout_seconds == task.timeout_seconds
        assert out.params == task.params

    def test_appends_to_existing_system_prompt(self) -> None:
        iv = MemoryInjectionIntervention(learnings=("- do less",))
        task = AgentTask(prompt="p", system_prompt="You are careful.")
        out = iv.apply(task)
        assert out.system_prompt is not None
        assert out.system_prompt.startswith("You are careful.")
        assert "<learnings" in out.system_prompt

    def test_does_not_mutate_input_task(self) -> None:
        iv = MemoryInjectionIntervention(learnings=("- x",))
        task = AgentTask(prompt="p")
        iv.apply(task)
        assert task.system_prompt is None  # untouched

    def test_empty_memory_is_a_noop_returning_same_task(self) -> None:
        # Fail-closed: no learnings ⇒ the candidate is byte-identical to the baseline.
        iv = MemoryInjectionIntervention(learnings=())
        task = AgentTask(prompt="p", system_prompt="prior")
        out = iv.apply(task)
        assert out is task  # the SAME object, not merely equal
        assert out.system_prompt == "prior"

    def test_empty_memory_noop_leaves_none_system_prompt(self) -> None:
        iv = MemoryInjectionIntervention()
        task = AgentTask(prompt="p")
        out = iv.apply(task)
        assert out is task
        assert out.system_prompt is None


# ---------------------------------------------------------------------------
# Source loader / formatter
# ---------------------------------------------------------------------------


class TestSource:
    def test_load_ranked_assets_from_parsed_entries(self) -> None:
        parsed = [
            {
                "asset_type": "skill",
                "title": "Retry policy",
                "rank": 1,
                "n_traces": 2,
                "occurrences": 2,
                "trace_ids": ["t1", "t2"],
                "expected_benefits": ["fewer retries"],
                "rationales": ["saw repeated retries"],
            }
        ]
        assets = load_ranked_assets(parsed=parsed)
        assert len(assets) == 1
        assert isinstance(assets[0], RankedAsset)
        assert assets[0].title == "Retry policy"
        assert assets[0].occurrences == 2

    def test_absent_report_yields_no_learnings(self, tmp_path: Path) -> None:
        # Fail-closed: a missing source is an empty list (no-op == baseline), not a crash.
        missing = tmp_path / "does-not-exist.json"
        assert load_ranked_assets(missing) == []
        assert build_memory_learnings(load_ranked_assets(missing)) == ()

    def test_select_top_k_by_occurrences_desc(self) -> None:
        assets = [
            _asset("low", occurrences=1, rank=3),
            _asset("high", occurrences=9, rank=2),
            _asset("mid", occurrences=5, rank=1),
        ]
        top = select_top_k(assets, k=2)
        assert [a.title for a in top] == ["high", "mid"]  # ranked by occurrences, not rank

    def test_select_top_k_zero_selects_none(self) -> None:
        assert select_top_k([_asset("x", occurrences=1)], k=0) == []

    def test_select_top_k_tiebreak_is_deterministic(self) -> None:
        # Equal occurrences ⇒ break on report rank, then title.
        assets = [
            _asset("b", occurrences=1, rank=2),
            _asset("a", occurrences=1, rank=1),
        ]
        assert [a.title for a in select_top_k(assets, k=2)] == ["a", "b"]

    def test_format_advisory_line_prefers_expected_benefit(self) -> None:
        asset = _asset(
            "Retry policy", occurrences=2, expected_benefits=["cuts retries"], rationales=["r"]
        )
        assert format_advisory_line(asset) == "- Retry policy — cuts retries"

    def test_format_advisory_line_falls_back_to_rationale(self) -> None:
        asset = _asset("Cache", occurrences=1, rationales=["re-read files a lot"])
        assert format_advisory_line(asset) == "- Cache — re-read files a lot"

    def test_format_advisory_line_title_only_when_no_detail(self) -> None:
        assert format_advisory_line(_asset("Bare", occurrences=1)) == "- Bare"

    def test_build_memory_learnings_is_top_k_lines(self) -> None:
        assets = [
            _asset("high", occurrences=9, expected_benefits=["big win"]),
            _asset("low", occurrences=1, expected_benefits=["small win"]),
        ]
        assert build_memory_learnings(assets, k=1) == ("- high — big win",)


# ---------------------------------------------------------------------------
# MEMORY_CANDIDATE wiring
# ---------------------------------------------------------------------------


class TestMemoryCandidate:
    def test_candidate_wires_a_memory_intervention(self) -> None:
        assert isinstance(MEMORY_CANDIDATE, LeverConfig)
        assert MEMORY_CANDIDATE.asset_enabled is True
        assert MEMORY_CANDIDATE.intervention is not None
        assert isinstance(MEMORY_CANDIDATE.intervention, MemoryInjectionIntervention)
        assert MEMORY_CANDIDATE.name == "candidate-advisory-memory"

    def test_build_memory_candidate_absent_report_is_noop(self) -> None:
        # With no report on disk the candidate is a fail-closed no-op (no learnings).
        cfg = build_memory_candidate("no-such-report.json")
        assert isinstance(cfg.intervention, MemoryInjectionIntervention)
        assert cfg.intervention.learnings == ()


# ---------------------------------------------------------------------------
# Provenance wall (teaching-to-the-test guard)
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_disjoint_memory_passes(self) -> None:
        suite = _suite(_task("do work", task_id="ts-001"))
        assets = [_asset("skill", occurrences=1, trace_ids=["organic-trace-abc"])]
        assert_memory_disjoint_from_suite(assets=assets, suite=suite)  # no raise

    def test_overlap_on_source_trace_id_raises(self) -> None:
        suite = _suite(_task("do work", task_id="ts-001"))  # source_trace_id="src-ts-001"
        assets = [_asset("leaky", occurrences=1, trace_ids=["src-ts-001"])]
        with pytest.raises(PoolOverlapError, match="teaching to the test"):
            assert_memory_disjoint_from_suite(assets=assets, suite=suite)

    def test_overlap_on_task_id_raises(self) -> None:
        suite = _suite(_task("do work", task_id="ts-001"))
        assets = [_asset("leaky", occurrences=1, trace_ids=["ts-001"])]
        with pytest.raises(PoolOverlapError):
            assert_memory_disjoint_from_suite(assets=assets, suite=suite)


# ---------------------------------------------------------------------------
# End-to-end through the unchanged Phase-2 harness (mocked adapter, no live call)
# ---------------------------------------------------------------------------


class TestHarnessFlow:
    def test_memory_reaches_only_the_candidate_arm(self) -> None:
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            candidate=_memory_candidate("- cache reads"),
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        # Two arms ran; the candidate carries the marker, the baseline does not.
        marked = ["<learnings" in (t.system_prompt or "") for t in adapter.seen]
        assert marked.count(True) == 1
        assert marked.count(False) == 1

    def test_token_drop_with_l1_pass_promotes(self) -> None:
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(60_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            candidate=_memory_candidate("- do less"),
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        assert artifact.n_promote == 1
        assert artifact.outcomes[0].recommendation is Recommendation.PROMOTE
        assert artifact.realized_token_savings_absolute == 40_000

    def test_empty_memory_candidate_cannot_promote(self) -> None:
        # No learnings ⇒ candidate == baseline ⇒ no token reduction ⇒ BLOCK (never a fake win).
        adapter = SuiteAdapter({"p1": (Arm(100_000), Arm(100_000))})
        artifact = run_phase2_comparison(
            suite=_suite(_task("p1")),
            adapter=adapter,
            candidate=_memory_candidate(),  # empty memory
            verify_specs=_pass_spec(),
            generated_at=_STAMP,
        )
        assert artifact.n_promote == 0
        assert artifact.outcomes[0].recommendation is Recommendation.BLOCK
        # Both arms saw the identical (marker-free) prompt.
        assert all("<learnings" not in (t.system_prompt or "") for t in adapter.seen)
