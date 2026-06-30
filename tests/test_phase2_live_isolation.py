"""Tests for the live Phase-2 harness: per-arm isolation + tamper-proof verify.

Fully offline — **no live agent, no network**. The agent is a
``FileWritingAdapter`` that *actually writes files* into ``task.cwd`` (the per-arm
workspace the harness sets), so the isolation and verification machinery is
exercised end-to-end against the real filesystem. Verification uses real but tiny
deterministic checks (a python script; one test uses ``pytest``) restored from a
fixture, so the arm-aware, tamper-proof path runs for real.

The mandated checklist (see the task contract) is covered explicitly:

* **non-contamination** — a candidate-arm edit never appears in the baseline
  workspace and vice-versa;
* **arm-aware verify** — baseline passes + candidate passes (fewer tokens) →
  PROMOTE; baseline passes + candidate breaks the check → REGRESSED → BLOCK;
* **tamper-proof** — an adapter that overwrites/deletes the verify test is still
  judged against the *pristine* test (a faked test cannot fake a pass);
* **cleanup** — the per-arm workspaces are torn down even when the adapter raises.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ail.compare import NO_LLM_JUDGE, ArmWorkspaces, Recommendation, compare_candidate
from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedTrace,
    TokenUsage,
    TraceStatus,
)
from ail.optimize import (
    CANDIDATE,
    VerifySpec,
    case_from_task,
    load_fixture,
    run_phase2_comparison,
)
from ail.optimize.phase2 import L1Outcome
from ail.task_suite.schema import Difficulty, Task, TaskCategory, TaskSuite

_STAMP = "2026-06-29T00:00:00+00:00"

# A cwd-relative L1 check (the tmp-fixture analogue of the committed example):
# passes iff ``solution.txt`` (as the arm left it) contains exactly "CORRECT".
_CHECK_SCRIPT = (
    "import sys\n"
    "from pathlib import Path\n"
    "p = Path('solution.txt')\n"
    "text = p.read_text(encoding='utf-8') if p.exists() else ''\n"
    "sys.exit(0 if text.strip() == 'CORRECT' else 1)\n"
)
# Matches the committed example fixture's verify/check.py expectation.
_EXAMPLE_SOLUTION = "def add(a, b):\n    return a + b\n"


# ---------------------------------------------------------------------------
# Test doubles + fixture helpers
# ---------------------------------------------------------------------------


@dataclass
class ArmPlan:
    """What the fake agent does for one arm: files to write + token/exec outcome.

    ``writes`` maps a workspace-relative path to its new content, or to ``None``
    to delete that file (to script tampering / deletion). ``raises`` makes the
    adapter blow up mid-run (to exercise cleanup-on-exception).
    """

    writes: dict[str, str | None] = field(default_factory=dict)
    tokens: int = 0
    success: bool = True
    output: str = "done"
    raises: bool = False


class FileWritingAdapter(AgentAdapter):
    """A FAKE adapter that ACTUALLY edits files in ``task.cwd`` (no live agent).

    Tells the candidate arm apart from the baseline by the skill marker the real
    CANDIDATE intervention injects into the system prompt, then applies that arm's
    :class:`ArmPlan` to ``task.cwd`` — the per-arm workspace the harness set.
    Records ``(cwd, is_candidate)`` per call so a test can assert on isolation and
    on workspace teardown.
    """

    name = "file-writer"

    def __init__(self, *, baseline: ArmPlan, candidate: ArmPlan) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.seen: list[tuple[str | None, bool]] = []

    def run(self, task: AgentTask) -> AgentRunResult:
        is_candidate = "<skill" in (task.system_prompt or "")
        plan = self.candidate if is_candidate else self.baseline
        self.seen.append((task.cwd, is_candidate))
        if plan.raises:
            raise RuntimeError("adapter exploded mid-run")
        cwd = Path(task.cwd) if task.cwd else Path.cwd()
        for rel, content in plan.writes.items():
            target = cwd / rel
            if content is None:
                target.unlink(missing_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        trace = NormalizedTrace(
            trace_id=("cand" if is_candidate else "base"),
            status=TraceStatus.OK if plan.success else TraceStatus.ERROR,
            producer=self.name,
            model="claude-opus-4-8",
            token_usage=TokenUsage(input_tokens=plan.tokens),
        )
        return AgentRunResult(
            trace=trace,
            output_text=plan.output,
            success=plan.success,
            error=None if plan.success else "arm failed",
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
    return TaskSuite(version="test-iso-v1", tasks=tuple(tasks)).freeze()


def _make_fixture(
    root: Path,
    task_id: str,
    *,
    seed_files: dict[str, str],
    verify_files: dict[str, str],
) -> Path:
    """Author a throwaway live fixture under ``<root>/eval/phase2_fixtures/<id>``."""
    base = root / "eval" / "phase2_fixtures" / task_id
    for sub, files in (("seed", seed_files), ("verify", verify_files)):
        for rel, content in files.items():
            path = base / sub / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return base


def _script_spec(task_id: str) -> dict[str, VerifySpec]:
    return {task_id: VerifySpec(name="check", command=[sys.executable, "verify/check.py"])}


# ---------------------------------------------------------------------------
# (a) NON-CONTAMINATION — arms never see each other's edits
# ---------------------------------------------------------------------------


class TestNonContamination:
    def test_candidate_edits_never_land_in_the_baseline_workspace(self, tmp_path: Path) -> None:
        base_ws = tmp_path / "base"
        cand_ws = tmp_path / "cand"
        for ws in (base_ws, cand_ws):
            ws.mkdir()
            (ws / "seed.txt").write_text("seed", encoding="utf-8")

        adapter = FileWritingAdapter(
            baseline=ArmPlan(writes={"base_only.txt": "B"}, tokens=100_000),
            candidate=ArmPlan(writes={"cand_only.txt": "C"}, tokens=60_000),
        )
        compare_candidate(
            case_from_task(_task("do the work")),
            adapter,
            intervention=CANDIDATE.intervention,
            correctness_judge=NO_LLM_JUDGE,
            workspace=ArmWorkspaces(baseline_cwd=str(base_ws), candidate_cwd=str(cand_ws)),
            generated_at=_STAMP,
        )

        # Each arm's edit lands only in its own workspace.
        assert (base_ws / "base_only.txt").exists()
        assert not (base_ws / "cand_only.txt").exists()
        assert (cand_ws / "cand_only.txt").exists()
        assert not (cand_ws / "base_only.txt").exists()
        # The two arms ran in DISTINCT directories (not one shared cwd).
        assert {cwd for cwd, _ in adapter.seen} == {str(base_ws), str(cand_ws)}
        baseline_cwd = next(cwd for cwd, is_cand in adapter.seen if not is_cand)
        candidate_cwd = next(cwd for cwd, is_cand in adapter.seen if is_cand)
        assert baseline_cwd != candidate_cwd


# ---------------------------------------------------------------------------
# (b) ARM-AWARE verify — baseline's verdict vs candidate's verdict, per workspace
# ---------------------------------------------------------------------------


class TestArmAwareVerification:
    def test_both_pass_and_candidate_cheaper_promotes(self, tmp_path: Path) -> None:
        _make_fixture(
            tmp_path,
            "ts-iso",
            seed_files={"solution.txt": "TODO"},
            verify_files={"check.py": _CHECK_SCRIPT},
        )
        adapter = FileWritingAdapter(
            baseline=ArmPlan(writes={"solution.txt": "CORRECT"}, tokens=100_000),
            candidate=ArmPlan(writes={"solution.txt": "CORRECT"}, tokens=60_000),
        )
        artifact = run_phase2_comparison(
            suite=_suite(_task("solve it", "ts-iso")),
            adapter=adapter,
            verify_specs=_script_spec("ts-iso"),
            fixtures_root=tmp_path,
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.PROMOTE
        assert out.l1_outcome is L1Outcome.PASSED
        assert out.baseline_succeeded and out.candidate_succeeded
        assert artifact.realized_token_savings_absolute == 40_000.0

    def test_candidate_breaks_verify_is_regression_and_blocks(self, tmp_path: Path) -> None:
        # Baseline writes a passing solution; the candidate writes a failing one.
        # The check passed at baseline and fails for the candidate => REGRESSED.
        _make_fixture(
            tmp_path,
            "ts-iso",
            seed_files={"solution.txt": "TODO"},
            verify_files={"check.py": _CHECK_SCRIPT},
        )
        adapter = FileWritingAdapter(
            baseline=ArmPlan(writes={"solution.txt": "CORRECT"}, tokens=100_000),
            candidate=ArmPlan(writes={"solution.txt": "WRONG"}, tokens=60_000),
        )
        artifact = run_phase2_comparison(
            suite=_suite(_task("solve it", "ts-iso")),
            adapter=adapter,
            verify_specs=_script_spec("ts-iso"),
            fixtures_root=tmp_path,
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.l1_outcome is L1Outcome.REGRESSED
        # A token drop measured on a regressed candidate is never a realized win.
        assert out.token_improved is True
        assert artifact.realized_token_savings_absolute == 0.0


# ---------------------------------------------------------------------------
# (c) TAMPER-PROOF — a deleted/overwritten verify test cannot fake a pass
# ---------------------------------------------------------------------------


class TestTamperProof:
    def test_overwriting_the_verify_check_cannot_fake_a_pass(self, tmp_path: Path) -> None:
        # The candidate writes a WRONG solution AND overwrites the check with an
        # always-pass script. The harness restores the pristine check first, so the
        # wrong solution is still caught: REGRESSED, not a faked pass.
        _make_fixture(
            tmp_path,
            "ts-tamper",
            seed_files={"solution.txt": "TODO"},
            verify_files={"check.py": _CHECK_SCRIPT},
        )
        adapter = FileWritingAdapter(
            baseline=ArmPlan(writes={"solution.txt": "CORRECT"}, tokens=100_000),
            candidate=ArmPlan(
                writes={
                    "solution.txt": "WRONG",
                    "verify/check.py": "import sys\nsys.exit(0)\n",  # tamper: always pass
                },
                tokens=60_000,
            ),
        )
        artifact = run_phase2_comparison(
            suite=_suite(_task("solve it", "ts-tamper")),
            adapter=adapter,
            verify_specs=_script_spec("ts-tamper"),
            fixtures_root=tmp_path,
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.l1_outcome is L1Outcome.REGRESSED

    def test_deleting_the_verify_check_cannot_fake_a_pass(self, tmp_path: Path) -> None:
        # The candidate deletes the (restored) check dir contents; restore puts the
        # pristine check back, so the wrong solution is still caught.
        _make_fixture(
            tmp_path,
            "ts-del",
            seed_files={"solution.txt": "TODO"},
            verify_files={"check.py": _CHECK_SCRIPT},
        )
        adapter = FileWritingAdapter(
            baseline=ArmPlan(writes={"solution.txt": "CORRECT"}, tokens=100_000),
            candidate=ArmPlan(
                writes={"solution.txt": "WRONG", "verify/check.py": None},  # try to delete it
                tokens=60_000,
            ),
        )
        artifact = run_phase2_comparison(
            suite=_suite(_task("solve it", "ts-del")),
            adapter=adapter,
            verify_specs=_script_spec("ts-del"),
            fixtures_root=tmp_path,
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.l1_outcome is L1Outcome.REGRESSED

    def test_pytest_verifier_path_restores_tampered_test(self, tmp_path: Path) -> None:
        # Demonstrates the canonical pytest verifier AND tamper-proofing against a
        # pytest test file: the candidate rewrites the test to always pass, but the
        # pristine test is restored and catches the WRONG solution.
        pytest_test = (
            "from pathlib import Path\n\n\n"
            "def test_solution() -> None:\n"
            "    assert Path('solution.txt').read_text(encoding='utf-8').strip() == 'CORRECT'\n"
        )
        _make_fixture(
            tmp_path,
            "ts-pytest",
            seed_files={"solution.txt": "TODO"},
            verify_files={"test_solution.py": pytest_test},
        )
        adapter = FileWritingAdapter(
            baseline=ArmPlan(writes={"solution.txt": "CORRECT"}, tokens=100_000),
            candidate=ArmPlan(
                writes={
                    "solution.txt": "WRONG",
                    "verify/test_solution.py": "def test_ok() -> None:\n    assert True\n",
                },
                tokens=60_000,
            ),
        )
        artifact = run_phase2_comparison(
            suite=_suite(_task("solve it", "ts-pytest")),
            adapter=adapter,
            verify_specs={
                "ts-pytest": VerifySpec(
                    name="pytest",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "verify",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                    ],
                )
            },
            fixtures_root=tmp_path,
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.l1_outcome is L1Outcome.REGRESSED


# ---------------------------------------------------------------------------
# (d) CLEANUP — workspaces torn down even when the adapter raises
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_workspaces_removed_even_when_adapter_raises(self, tmp_path: Path) -> None:
        _make_fixture(
            tmp_path,
            "ts-boom",
            seed_files={"solution.txt": "TODO"},
            verify_files={"check.py": _CHECK_SCRIPT},
        )
        adapter = FileWritingAdapter(
            baseline=ArmPlan(raises=True),  # blows up on the baseline run
            candidate=ArmPlan(writes={"solution.txt": "CORRECT"}, tokens=60_000),
        )
        artifact = run_phase2_comparison(
            suite=_suite(_task("solve it", "ts-boom")),
            adapter=adapter,
            verify_specs=_script_spec("ts-boom"),
            fixtures_root=tmp_path,
            generated_at=_STAMP,
        )
        # The raise is captured as a blocked, errored outcome — never a pass.
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.BLOCK
        assert out.error is not None and "adapter exploded" in out.error
        assert artifact.n_errored == 1 and artifact.n_promote == 0
        # The per-arm workspaces (and their shared parent) were torn down anyway.
        recorded_cwd = adapter.seen[0][0]
        assert recorded_cwd is not None
        assert not Path(recorded_cwd).exists()
        assert not Path(recorded_cwd).parent.exists()


# ---------------------------------------------------------------------------
# Loader + the committed throwaway example fixture (end-to-end)
# ---------------------------------------------------------------------------


class TestFixtureLoaderAndExample:
    def test_load_fixture_discovers_committed_example(self) -> None:
        fixture = load_fixture("example-token-task")
        assert fixture is not None
        assert fixture.seed_dir.is_dir()
        assert fixture.has_verify

    def test_load_fixture_returns_none_for_unknown_task(self) -> None:
        assert load_fixture("no-such-task-zzz-99999") is None

    def test_committed_example_fixture_runs_end_to_end(self) -> None:
        # Uses repo discovery (no fixtures_root) against the committed example.
        adapter = FileWritingAdapter(
            baseline=ArmPlan(writes={"solution.py": _EXAMPLE_SOLUTION}, tokens=100_000),
            candidate=ArmPlan(writes={"solution.py": _EXAMPLE_SOLUTION}, tokens=60_000),
        )
        artifact = run_phase2_comparison(
            suite=_suite(_task("implement add", "example-token-task")),
            adapter=adapter,
            verify_specs=_script_spec("example-token-task"),
            generated_at=_STAMP,
        )
        out = artifact.outcomes[0]
        assert out.recommendation is Recommendation.PROMOTE
        assert out.l1_outcome is L1Outcome.PASSED
        assert artifact.realized_token_savings_absolute == 40_000.0
