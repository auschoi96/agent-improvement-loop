"""Tests for the GRP pipeline: capture -> execute -> approve -> promote.

Includes the Wave 1a acceptance round-trip (capture a candidate, simulate human
approval, promote, reload from the pool) and the explicit anti-co-adaptation
assertions: no stage other than the human gate ever fills expectations, and
there is no expected-output synthesis surface anywhere in the package.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ail.groundtruth as gt
from ail.groundtruth.approve import apply_review, approve_case, pending_cases, reject_case
from ail.groundtruth.capture import CaptureError, capture_candidates
from ail.groundtruth.execute import execute_candidate
from ail.groundtruth.promote import PoolConflictError, PromotionError, promote_approved
from ail.groundtruth.schema import (
    Expectations,
    GroundTruthCase,
    Pool,
    ReviewStatus,
)
from ail.groundtruth.store import JsonGroundTruthStore, read_review_queue, write_review_queue
from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedTrace,
    TraceStatus,
)


class FakeAdapter(AgentAdapter):
    """Deterministic offline agent: returns a fixed output, records the task."""

    name = "fake"

    def __init__(self, output: str = "def add(a, b):\n    return a + b", success: bool = True):
        self.output = output
        self.success = success
        self.last_task: AgentTask | None = None

    def run(self, task: AgentTask) -> AgentRunResult:
        self.last_task = task
        trace = NormalizedTrace(
            trace_id="tr-exec-1",
            status=TraceStatus.OK if self.success else TraceStatus.ERROR,
            producer=self.name,
            model="claude-sonnet-4-6",
        )
        return AgentRunResult(
            trace=trace,
            output_text=self.output,
            success=self.success,
            error=None if self.success else "boom",
            duration_ms=123,
        )


def _trace(trace_id: str = "tr-1", prompt: str = "Write an add(a, b) function") -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=trace_id,
        status=TraceStatus.OK,
        producer="claude_code",
        model="claude-opus-4-8",
        session_id="sess-1",
        experiment_id="660599403165942",
        request_preview=prompt,
        response_preview="def add(a, b): return a + b",
    )


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


class TestCapture:
    def test_captures_task_and_provenance_without_expectations(self) -> None:
        [case] = capture_candidates([_trace()])
        assert case.task_input.prompt == "Write an add(a, b) function"
        assert case.sources[0].ref == "tr-1"
        assert case.sources[0].kind.value == "trace"
        # The capture stage never authors expectations.
        assert case.expectations.is_filled() is False
        assert case.review.status is ReviewStatus.CANDIDATE
        assert case.candidate_response is None

    def test_observed_response_kept_as_context_not_expectation(self) -> None:
        [case] = capture_candidates([_trace()])
        # The historical response is preserved as reviewer context, clearly
        # labelled observed — and crucially NOT promoted into expectations.
        assert case.metadata["observed_response_preview"] == "def add(a, b): return a + b"
        assert case.expectations.is_filled() is False

    def test_skips_traces_without_prompt(self) -> None:
        no_prompt = NormalizedTrace(trace_id="tr-empty", request_preview=None)
        assert capture_candidates([no_prompt, _trace()]) == capture_candidates([_trace()])

    def test_strict_capture_raises_on_empty_prompt(self) -> None:
        no_prompt = NormalizedTrace(trace_id="tr-empty", request_preview="")
        with pytest.raises(CaptureError):
            capture_candidates([no_prompt], skip_invalid=False)

    def test_recapture_is_idempotent(self) -> None:
        cases = capture_candidates([_trace(), _trace()])
        assert len(cases) == 1


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_records_agent_own_output_without_expectations(self) -> None:
        [case] = capture_candidates([_trace()])
        adapter = FakeAdapter()
        executed = execute_candidate(case, adapter)

        assert executed.candidate_response is not None
        assert executed.candidate_response.output_text == adapter.output
        assert executed.candidate_response.producer == "fake"
        assert executed.candidate_response.success is True
        # Executing the agent must not invent expectations or approve anything.
        assert executed.expectations.is_filled() is False
        assert executed.review.status is ReviewStatus.CANDIDATE

    def test_runs_the_captured_prompt(self) -> None:
        [case] = capture_candidates([_trace()])
        adapter = FakeAdapter()
        execute_candidate(case, adapter)
        assert adapter.last_task is not None
        assert adapter.last_task.prompt == case.task_input.prompt

    def test_failed_run_is_recorded_not_raised(self) -> None:
        [case] = capture_candidates([_trace()])
        executed = execute_candidate(case, FakeAdapter(success=False))
        assert executed.candidate_response is not None
        assert executed.candidate_response.success is False
        assert executed.candidate_response.error == "boom"


# ---------------------------------------------------------------------------
# approve (human gate)
# ---------------------------------------------------------------------------


class TestApprove:
    def test_cannot_approve_without_reviewer(self) -> None:
        [case] = capture_candidates([_trace()])
        with pytest.raises(gt.ReviewError):
            approve_case(
                case,
                reviewer="",
                expectations=Expectations(expected_response="x"),
                regression_intent="why",
                target_pool=Pool.ALIGNMENT_SET,
            )

    def test_cannot_approve_with_empty_expectations(self) -> None:
        [case] = capture_candidates([_trace()])
        with pytest.raises(gt.ReviewError):
            apply_review(
                case,
                reviewer="austin",
                decision=ReviewStatus.APPROVED,
                expectations=Expectations(),  # empty
                regression_intent="why",
                target_pool=Pool.ALIGNMENT_SET,
            )

    def test_cannot_approve_with_blank_regression_intent(self) -> None:
        [case] = capture_candidates([_trace()])
        with pytest.raises(gt.ReviewError):
            approve_case(
                case,
                reviewer="austin",
                expectations=Expectations(expected_response="3"),
                regression_intent="   ",
                target_pool=Pool.ALIGNMENT_SET,
            )

    def test_approval_fills_expectations_and_marks_pool(self) -> None:
        [case] = capture_candidates([_trace()])
        approved = approve_case(
            case,
            reviewer="austin",
            expectations=Expectations(must_include=["return"]),
            regression_intent="guards add()",
            target_pool=Pool.HUMAN_ANCHOR,
            comment="lgtm",
        )
        assert approved.review.status is ReviewStatus.APPROVED
        assert approved.review.reviewer == "austin"
        assert approved.expectations.is_filled() is True
        assert approved.target_pool is Pool.HUMAN_ANCHOR
        assert approved.is_promotable() is True

    def test_rejection_does_not_require_expectations(self) -> None:
        [case] = capture_candidates([_trace()])
        rejected = reject_case(case, reviewer="austin", comment="off topic")
        assert rejected.review.status is ReviewStatus.REJECTED
        assert rejected.is_promotable() is False

    def test_pending_filters_unreviewed(self) -> None:
        [case] = capture_candidates([_trace()])
        approved = approve_case(
            case,
            reviewer="austin",
            expectations=Expectations(expected_response="3"),
            regression_intent="why",
            target_pool=Pool.ALIGNMENT_SET,
        )
        assert pending_cases([case]) == [case]
        assert pending_cases([approved]) == []


# ---------------------------------------------------------------------------
# promote (separate, explicit, pool-disjoint)
# ---------------------------------------------------------------------------


def _approved(case: GroundTruthCase, pool: Pool, *, reviewer: str = "austin") -> GroundTruthCase:
    return approve_case(
        case,
        reviewer=reviewer,
        expectations=Expectations(must_include=["return"]),
        regression_intent="guards add()",
        target_pool=pool,
    )


class TestPromote:
    def test_promotes_only_approved_cases(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        [candidate] = capture_candidates([_trace("tr-a")])
        approved = _approved(capture_candidates([_trace("tr-b")])[0], Pool.ALIGNMENT_SET)

        result = promote_approved([candidate, approved], pool=Pool.ALIGNMENT_SET, store=store)

        assert result.n_promoted == 1
        assert result.promoted == [approved.case_id]
        assert result.n_skipped == 1
        assert candidate.case_id == result.skipped[0][0]

    def test_strict_mode_raises_on_non_promotable(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        [candidate] = capture_candidates([_trace()])
        with pytest.raises(PromotionError):
            promote_approved([candidate], pool=Pool.ALIGNMENT_SET, store=store, strict=True)

    def test_skips_case_approved_for_a_different_pool(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        approved = _approved(capture_candidates([_trace()])[0], Pool.HUMAN_ANCHOR)
        result = promote_approved([approved], pool=Pool.ALIGNMENT_SET, store=store)
        assert result.n_promoted == 0
        assert result.n_skipped == 1

    def test_never_mixes_pools(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        # Same case id approved for two different pools, promoted to the first.
        anchor = _approved(capture_candidates([_trace("tr-x")])[0], Pool.HUMAN_ANCHOR)
        promote_approved([anchor], pool=Pool.HUMAN_ANCHOR, store=store)

        clash = _approved(capture_candidates([_trace("tr-x")])[0], Pool.ALIGNMENT_SET)
        with pytest.raises(PoolConflictError):
            promote_approved([clash], pool=Pool.ALIGNMENT_SET, store=store)

        # The alignment pool stays empty; the anchor pool keeps its one case.
        assert store.load(Pool.ALIGNMENT_SET).cases == []
        assert store.load(Pool.HUMAN_ANCHOR).case_ids() == {anchor.case_id}

    def test_promotion_is_idempotent(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        approved = _approved(capture_candidates([_trace()])[0], Pool.ALIGNMENT_SET)
        promote_approved([approved], pool=Pool.ALIGNMENT_SET, store=store)
        promote_approved([approved], pool=Pool.ALIGNMENT_SET, store=store)
        assert len(store.load(Pool.ALIGNMENT_SET).cases) == 1


# ---------------------------------------------------------------------------
# Acceptance: full round-trip
# ---------------------------------------------------------------------------


def test_round_trip_capture_execute_approve_promote_reload(tmp_path: Path) -> None:
    """Wave 1a acceptance: capture -> execute -> human approve -> promote -> reload."""
    store = JsonGroundTruthStore(tmp_path / "pools")

    # 1. capture from a trace
    [candidate] = capture_candidates([_trace("tr-roundtrip")])

    # 2. execute the agent to capture its own response (offline FakeAdapter)
    executed = execute_candidate(candidate, FakeAdapter())
    assert executed.candidate_response is not None

    # The candidate is parked in a review queue for a human, off the frozen wall.
    queue_path = write_review_queue([executed], tmp_path / "queue.json")
    [from_queue] = read_review_queue(queue_path)
    assert from_queue == executed
    assert from_queue.expectations.is_filled() is False  # still unlabelled

    # 3. a human fills expectations and approves (the gate)
    approved = approve_case(
        from_queue,
        reviewer="austin",
        expectations=Expectations(
            must_include=["def add", "return"],
            rubric="defines an add() that returns the sum",
        ),
        regression_intent="guards the canonical add() example",
        target_pool=Pool.ALIGNMENT_SET,
    )

    # 4. separate, explicit promotion into the frozen pool
    result = promote_approved([approved], pool=Pool.ALIGNMENT_SET, store=store)
    assert result.n_promoted == 1

    # reload as a GroundTruthSet and confirm the labelled case survived
    reloaded = store.load(Pool.ALIGNMENT_SET)
    assert reloaded.pool is Pool.ALIGNMENT_SET
    assert reloaded.case_ids() == {approved.case_id}
    restored = reloaded.cases[0]
    assert restored.expectations.must_include == ["def add", "return"]
    assert restored.regression_intent == "guards the canonical add() example"
    assert restored.review.reviewer == "austin"
    assert restored == approved


# ---------------------------------------------------------------------------
# Anti-co-adaptation: no LLM synthesis of expected outputs anywhere
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(gt.__file__).parent
#: Surfaces that could synthesize an expected output. Plain ``mlflow`` logging is
#: allowed (the execute stage audits captures); ``mlflow.genai`` / judges are not.
_FORBIDDEN_SYNTH_TOKENS = [
    "make_judge",
    "mlflow.genai",
    "chat.completions",
    "messages.create",
    "litellm",
    "import anthropic",
    "import openai",
    "ai_gen(",
    "ai_query(",
]
#: How expectations get *written* onto a case (kwarg or model-update key).
_EXPECTATION_WRITE_PATTERNS = ["expectations=", '"expectations":']


def _src(name: str) -> str:
    return (_PACKAGE_DIR / name).read_text()


def test_no_expected_output_synthesis_surface_in_package() -> None:
    for path in _PACKAGE_DIR.glob("*.py"):
        text = path.read_text()
        for token in _FORBIDDEN_SYNTH_TOKENS:
            assert token not in text, f"{path.name} references synthesis surface {token!r}"


def test_only_the_human_gate_writes_expectations() -> None:
    # Capture/execute/promote never write expectations onto a case…
    for name in ("capture", "execute", "promote"):
        src = _src(f"{name}.py")
        for pat in _EXPECTATION_WRITE_PATTERNS:
            assert pat not in src, f"{name}.py writes expectations ({pat!r}) — only approve may"

    # …while the human gate is exactly where expectations are written.
    assert any(pat in _src("approve.py") for pat in _EXPECTATION_WRITE_PATTERNS)


def test_capture_and_execute_leave_expectations_empty() -> None:
    """Behavioural mirror of the structural guard above."""
    [candidate] = capture_candidates([_trace()])
    assert candidate.expectations.is_filled() is False
    executed = execute_candidate(candidate, FakeAdapter())
    assert executed.expectations.is_filled() is False
    # And an unlabelled candidate can never be promoted, by construction.
    assert executed.is_promotable() is False
