"""Tests for the GRP pipeline: capture -> execute -> approve -> promote.

Includes the Wave 1a acceptance round-trip (capture a candidate, simulate human
approval, promote, reload from the pool) and the explicit anti-co-adaptation
assertions: no stage other than the human gate ever fills expectations, and
there is no expected-output synthesis surface anywhere in the package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ail.groundtruth as gt
from ail.groundtruth.approve import apply_review, approve_case, pending_cases, reject_case
from ail.groundtruth.capture import CaptureError, capture_candidates
from ail.groundtruth.execute import execute_candidate
from ail.groundtruth.promote import (
    PoolConflictError,
    PromotionError,
    TaskSuiteProtectedError,
    promote_approved,
)
from ail.groundtruth.schema import (
    Expectations,
    GroundTruthCase,
    GroundTruthSet,
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

    def test_promote_into_task_suite_is_forbidden(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        # A human can approve a case targeting the Task Suite, but the held-out
        # benchmark must never be fed by the bootstrap loop.
        approved = _approved(capture_candidates([_trace()])[0], Pool.TASK_SUITE)
        with pytest.raises(TaskSuiteProtectedError):
            promote_approved([approved], pool=Pool.TASK_SUITE, store=store)
        assert not (tmp_path / "task_suite.json").exists()


class TestStoreEnforcesHumanGate:
    """The persistence boundary cannot be used to bypass approve/promote."""

    def test_save_rejects_unapproved_case_even_via_model_construct(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        [candidate] = capture_candidates([_trace()])  # unapproved, target_pool is None
        # model_construct skips the GroundTruthSet validator — the store.save()
        # boundary must still refuse it.
        bad = GroundTruthSet.model_construct(
            pool=Pool.ALIGNMENT_SET, name="alignment_set", cases=[candidate]
        )
        with pytest.raises(gt.GroundTruthError):
            store.save(bad)
        assert not (tmp_path / "alignment_set.json").exists()

    def test_save_rejects_pooled_but_unapproved_case(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        candidate = capture_candidates([_trace()])[0].model_copy(
            update={"target_pool": Pool.ALIGNMENT_SET}  # pool tagged, never approved
        )
        bad = GroundTruthSet.model_construct(
            pool=Pool.ALIGNMENT_SET, name="alignment_set", cases=[candidate]
        )
        with pytest.raises(gt.GroundTruthError):
            store.save(bad)
        assert not (tmp_path / "alignment_set.json").exists()

    def test_save_rejects_pool_mismatch(self, tmp_path: Path) -> None:
        store = JsonGroundTruthStore(tmp_path)
        approved = _approved(capture_candidates([_trace()])[0], Pool.HUMAN_ANCHOR)
        bad = GroundTruthSet.model_construct(
            pool=Pool.ALIGNMENT_SET, name="alignment_set", cases=[approved]
        )
        with pytest.raises(gt.GroundTruthError):
            store.save(bad)


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
#
# These guards are AST-based (not string grep), so they cannot be slipped past
# with dict(expectations=...), **kwargs spread, setattr, split-string keys, or
# helper indirection — and they catch model-generation imports/calls reached by
# any name, including indirect forms a grep would miss.
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(gt.__file__).parent
_PACKAGE_FILES = sorted(_PACKAGE_DIR.glob("*.py"))
#: The human gate is the ONLY stage permitted to write expectations onto a case.
_EXPECTATION_WRITERS_ALLOWED = {"approve.py"}

#: Import roots that pull in a model-generation or arbitrary-HTTP client.
_FORBIDDEN_IMPORT_SINGLE = {
    "openai",
    "anthropic",
    "cohere",
    "litellm",
    "httpx",
    "requests",
    "aiohttp",
}
#: Multi-component import prefixes (e.g. mlflow is fine, mlflow.genai is not).
_FORBIDDEN_IMPORT_PREFIXES = ("mlflow.genai", "google.generativeai", "google.genai")
#: Dotted call/attribute chains that indicate generation or raw HTTP egress.
_FORBIDDEN_CALL_CHAINS = (
    ".genai",
    "messages.create",
    "responses.create",
    "chat.completions",
    "make_judge",
    "requests.post",
    "requests.get",
    "httpx.post",
    "httpx.get",
)


def _const_str(node: ast.AST) -> str | None:
    """Constant-fold a node to a string, resolving ``"ex" + "pectations"``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _const_str(node.left), _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _dotted(node: ast.AST) -> str | None:
    """Flatten a Name/Attribute/Call chain to a dotted path (``a.b.create``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return None


class _ExpectationWriteFinder(ast.NodeVisitor):
    """Flag any node that writes a value into an ``expectations`` attr/key."""

    TARGET = "expectations"

    def __init__(self) -> None:
        self.hits: list[str] = []

    def visit_keyword(self, node: ast.keyword) -> None:  # foo(expectations=...)
        if node.arg == self.TARGET:
            self.hits.append(f"keyword arg {self.TARGET}= (line {node.value.lineno})")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:  # {"expectations": ...} / model_copy(update=...)
        for key in node.keys:
            if key is not None and _const_str(key) == self.TARGET:
                self.hits.append(f"dict key {self.TARGET!r} (line {node.lineno})")
        self.generic_visit(node)

    def _check_target(self, target: ast.AST, lineno: int) -> None:
        if isinstance(target, ast.Attribute) and target.attr == self.TARGET:
            self.hits.append(f"attribute assignment .{self.TARGET} (line {lineno})")
        if isinstance(target, ast.Subscript) and _const_str(target.slice) == self.TARGET:
            self.hits.append(f"subscript assignment [{self.TARGET!r}] (line {lineno})")

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # setattr(obj, "expectations", value)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and _const_str(node.args[1]) == self.TARGET
        ):
            self.hits.append(f"setattr(…, {self.TARGET!r}, …) (line {node.lineno})")
        self.generic_visit(node)


class _SynthesisSurfaceFinder(ast.NodeVisitor):
    """Collect every import and dotted call chain in a module."""

    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.chains: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.update(alias.name for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module)
            self.imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _dotted(node)
        if chain:
            self.chains.append(chain)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _dotted(node.func)
        if chain:
            self.chains.append(chain)
        self.generic_visit(node)


def _is_forbidden_import(name: str) -> bool:
    if name.split(".")[0] in _FORBIDDEN_IMPORT_SINGLE:
        return True
    return any(name == p or name.startswith(p + ".") for p in _FORBIDDEN_IMPORT_PREFIXES)


def _synthesis_hits(tree: ast.AST) -> list[str]:
    finder = _SynthesisSurfaceFinder()
    finder.visit(tree)
    hits = [f"import {i}" for i in finder.imports if _is_forbidden_import(i)]
    hits += [f"call {c}" for c in finder.chains if any(s in c for s in _FORBIDDEN_CALL_CHAINS)]
    return hits


def test_no_expectation_writes_outside_human_gate_ast() -> None:
    for path in _PACKAGE_FILES:
        if path.name in _EXPECTATION_WRITERS_ALLOWED:
            continue
        finder = _ExpectationWriteFinder()
        finder.visit(ast.parse(path.read_text()))
        assert finder.hits == [], f"{path.name} writes expectations outside the gate: {finder.hits}"


def test_human_gate_is_the_expectation_writer_ast() -> None:
    finder = _ExpectationWriteFinder()
    finder.visit(ast.parse((_PACKAGE_DIR / "approve.py").read_text()))
    assert finder.hits, "expected approve.py to be the one place expectations are written"


def test_no_model_generation_surface_in_package_ast() -> None:
    for path in _PACKAGE_FILES:
        hits = _synthesis_hits(ast.parse(path.read_text()))
        assert hits == [], f"{path.name} references a model-generation/network surface: {hits}"


# --- self-tests: prove the guards FAIL if synthesis is introduced indirectly ---

_INDIRECT_EXPECTATION_WRITES = [
    "GroundTruthCase(expectations=x)",  # keyword arg
    'case.model_copy(update={"expectations": x})',  # dict literal key
    'case.model_copy(update={"ex" + "pectations": x})',  # split-string key
    "case.expectations = x",  # attribute assignment
    'd["expectations"] = x',  # subscript assignment
    'setattr(case, "expectations", x)',  # setattr
    'GroundTruthCase(**{"expectations": x})',  # **kwargs spread via dict literal
]


@pytest.mark.parametrize("snippet", _INDIRECT_EXPECTATION_WRITES)
def test_expectation_write_finder_catches_indirect_forms(snippet: str) -> None:
    finder = _ExpectationWriteFinder()
    finder.visit(ast.parse(snippet))
    assert finder.hits, f"guard failed to flag indirect expectation write: {snippet!r}"


_SYNTH_SURFACE_SNIPPETS = [
    "import openai",
    "from openai import OpenAI",
    "import anthropic",
    "import httpx",
    "import requests",
    "from mlflow.genai import judges",
    "from mlflow.genai.judges import make_judge",
    "x = client.messages.create(model='m')",
    "x = client.responses.create()",
    "x = c.chat.completions.create()",
    "x = make_judge('j')",
    "requests.post(url)",
    "httpx.post(url)",
]


@pytest.mark.parametrize("snippet", _SYNTH_SURFACE_SNIPPETS)
def test_synthesis_surface_finder_catches_each(snippet: str) -> None:
    assert _synthesis_hits(ast.parse(snippet)), f"guard failed to flag surface: {snippet!r}"


def test_plain_mlflow_logging_is_allowed_by_guard() -> None:
    # The execute stage's audit logging (plain mlflow) must NOT trip the guard.
    allowed = "import mlflow\nmlflow.start_run()\nmlflow.log_text(t, 'x.txt')\nmlflow.set_tags({})"
    assert _synthesis_hits(ast.parse(allowed)) == []


def test_capture_and_execute_leave_expectations_empty() -> None:
    """Behavioural mirror of the structural guards above."""
    [candidate] = capture_candidates([_trace()])
    assert candidate.expectations.is_filled() is False
    executed = execute_candidate(candidate, FakeAdapter())
    assert executed.expectations.is_filled() is False
    # And an unlabelled candidate can never be promoted, by construction.
    assert executed.is_promotable() is False
