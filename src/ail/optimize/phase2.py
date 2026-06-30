"""Phase-2 runner: baseline-vs-candidate on the frozen suite, L1-gated, fail-closed.

This is the orchestration the live comparison driver (and the
``scripts/run_phase2_comparison.py`` CLI) runs: for each task in the **frozen**
Task Suite, run :func:`ail.compare.compare_candidate` with the
:data:`~ail.optimize.lever.CANDIDATE` token-efficiency intervention, gate
correctness on a deterministic **L1 programmatic** check (``NO_LLM_JUDGE`` — no
uncalibrated judge in the decision path), and fold the per-task results into a
single :class:`Phase2Artifact` carrying, per task, the **L0 token delta**, the
**L1 programmatic correctness outcome**, and the harness **decision**.

The whole module reuses the existing comparison machinery — it adds **no** new
scoring logic. It only (1) bridges a frozen :class:`~ail.task_suite.schema.Task`
into the read-only :class:`~ail.groundtruth.schema.GroundTruthCase` the harness
consumes, (2) turns a caller-supplied verification command into the L1
:class:`~ail.compare.ProgrammaticSignal` the harness gates on, and (3) shapes the
output artifact.

**Fail-closed, everywhere.** Every path that means "did not run / errored / no
data" maps to ``BLOCK`` and is never counted as a token win:

* a crashed/failed/timed-out run is blocked by the harness execution guardrail
  (its ~0-token "reduction" never reads as success);
* a task with **no** verification configured has no correctness signal, so the
  harness fails closed (a failed correctness guardrail → ``BLOCK``);
* an L1 verification that **could not run** yields ``errored`` →
  :data:`~ail.compare.ProgrammaticSignal.errored`, which fails the programmatic
  guardrail closed (a broken verifier never reads as "passed");
* a comparison that **raises** for a task is recorded as a blocked, errored
  outcome — never a pass;
* the artifact's *realized* token savings are summed over **PROMOTE** tasks only,
  so a blocked task's token delta is never aggregated into the headline.

This module performs **no** network I/O and runs no live agent: the live, costly
run is driven by the CLI against a real workspace. Unit tests drive it with a
mocked adapter and recorded L1 signals.
"""

from __future__ import annotations

import subprocess
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ail.compare import (
    EXECUTION_GUARDRAIL,
    NO_LLM_JUDGE,
    PROGRAMMATIC_GUARDRAIL,
    ComparisonConfig,
    ComparisonResult,
    GuardrailCheck,
    ProgrammaticCheck,
    ProgrammaticSignal,
    Recommendation,
    compare_candidate,
)
from ail.groundtruth.schema import GroundTruthCase, Source, SourceKind, TaskInput
from ail.ingest.base import AgentAdapter, AgentRunResult
from ail.optimize.lever import BASELINE, CANDIDATE, LeverConfig
from ail.task_suite.schema import Task, TaskSuite

__all__ = [
    "PHASE2_SCHEMA_VERSION",
    "L1Outcome",
    "VerifySpec",
    "TaskOutcome",
    "Phase2Artifact",
    "case_from_task",
    "make_command_check",
    "run_phase2_comparison",
]

#: Version of the Phase-2 comparison artifact contract.
PHASE2_SCHEMA_VERSION = "ail.optimize.phase2/v1"

#: The L0 token metric whose reduction is the objective (an emitted MetricDelta).
_TOKEN_METRIC = "total_tokens"


# ---------------------------------------------------------------------------
# Frozen suite Task -> read-only GroundTruthCase bridge
# ---------------------------------------------------------------------------


def case_from_task(task: Task) -> GroundTruthCase:
    """Bridge a frozen :class:`~ail.task_suite.schema.Task` into a harness case.

    :func:`ail.compare.compare_candidate` consumes a
    :class:`~ail.groundtruth.schema.GroundTruthCase` (task input + provenance);
    the frozen suite stores leaner :class:`~ail.task_suite.schema.Task` records.
    This builds a fresh, read-only case from one: the prompt becomes the task
    input, the source trace becomes provenance, and **expectations stay empty** —
    the frozen suite carries no human-authored expectations, which is exactly why
    the correctness guardrail must be the deterministic L1 check, not an LLM judge
    (see :data:`ail.compare.NO_LLM_JUDGE`). The case is never persisted or
    promoted; ``target_pool`` is left ``None``.
    """
    return GroundTruthCase(
        case_id=task.task_id,
        task_input=TaskInput(prompt=task.prompt),
        sources=[
            Source(
                kind=SourceKind.TRACE,
                ref=task.source_trace_id,
                note=f"frozen Task Suite reconstruction (category={task.category.value})",
            )
        ],
        tags={"category": task.category.value, "difficulty": task.difficulty.value},
    )


# ---------------------------------------------------------------------------
# L1 programmatic verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifySpec:
    """A per-task L1 verification command (tests/build/verifiable outcome).

    The orchestrator supplies these **outside** the frozen suite (the suite is a
    sealed benchmark and carries no commands). The command runs after a run and
    passes on exit code 0. A run with **no** ``VerifySpec`` has no correctness
    signal, so the harness fails closed (``BLOCK``) for that task.
    """

    name: str
    command: list[str] | str
    cwd: str | None = None
    shell: bool = False
    timeout_seconds: int = 600


def make_command_check(spec: VerifySpec) -> ProgrammaticCheck:
    """Build an L1 :data:`~ail.compare.ProgrammaticCheck` from a :class:`VerifySpec`.

    The returned check runs ``spec.command`` (in ``spec.cwd``) and reports
    ``passed = (exit code == 0)``. Crucially it **fails closed on a no-verdict**:
    if the command cannot be launched, times out, or crashes, the signal is marked
    :attr:`~ail.compare.ProgrammaticSignal.errored` so the programmatic guardrail
    blocks rather than mistaking an un-runnable verifier for "no regression".

    The signal is independent of the agent run's own success (the execution
    guardrail handles a crashed run); the verification reflects the *task's*
    verifiable outcome.
    """

    def check(_result: AgentRunResult) -> ProgrammaticSignal:
        try:
            proc = subprocess.run(
                spec.command,
                cwd=spec.cwd,
                shell=spec.shell,
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ProgrammaticSignal(
                name=spec.name,
                passed=False,
                details=f"verification could not run: {exc}",
                errored=True,
            )
        passed = proc.returncode == 0
        tail = (proc.stderr or proc.stdout or "").strip()
        details = f"exit={proc.returncode}"
        if tail and not passed:
            details += f"; {tail[-300:]}"
        return ProgrammaticSignal(name=spec.name, passed=passed, details=details, errored=False)

    return check


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------


class _Contract(BaseModel):
    """Base for the artifact models: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class L1Outcome(StrEnum):
    """The L1 programmatic correctness outcome for one task, derived from the harness.

    ``PASSED`` / ``FAILED_BOTH`` / ``REGRESSED`` are the three verdicts of a check
    that *ran*; ``NO_VERDICT`` and ``NOT_CONFIGURED`` are the two fail-closed
    "no correctness signal" states. Only ``PASSED`` and ``FAILED_BOTH`` clear the
    guardrail (``FAILED_BOTH`` is a pre-existing deficiency, not a regression the
    intervention caused); the rest contribute a ``BLOCK``.
    """

    PASSED = "passed"  # verification held: candidate verifies (not worse than baseline)
    FAILED_BOTH = "failed_both"  # ran, failed on both arms -> not a regression (guardrail passes)
    REGRESSED = "regressed"  # passed at baseline, fails for candidate -> the lever broke it
    NO_VERDICT = "no_verdict"  # verification could not run, or the comparison errored
    NOT_CONFIGURED = "not_configured"  # no verification supplied -> no correctness signal


class TaskOutcome(_Contract):
    """One task's baseline-vs-candidate result: L0 token delta + L1 outcome + decision."""

    task_id: str
    category: str = ""
    difficulty: str = ""
    recommendation: Recommendation = Recommendation.BLOCK
    objective_met: bool = False
    guardrails_passed: bool = False

    # L0 token delta (the objective; lower is better).
    baseline_total_tokens: float = 0.0
    candidate_total_tokens: float = 0.0
    token_delta_absolute: float = 0.0
    token_delta_pct: float | None = None
    token_improved: bool = False

    # L1 programmatic correctness (the guardrail).
    l1_outcome: L1Outcome = L1Outcome.NOT_CONFIGURED
    l1_verification_configured: bool = False

    # Execution success of each arm.
    baseline_succeeded: bool = False
    candidate_succeeded: bool = False

    # Provenance + detail.
    baseline_trace_id: str | None = None
    candidate_trace_id: str | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    error: str | None = None  # set iff the comparison itself raised for this task
    comparison: ComparisonResult | None = None

    @property
    def promoted(self) -> bool:
        """Whether this task's candidate cleared the gate (objective met + guardrails)."""
        return self.recommendation is Recommendation.PROMOTE


class Phase2Artifact(_Contract):
    """The artifact one Phase-2 comparison run produces, round-trippable through JSON.

    Realized token savings (:attr:`realized_token_savings_absolute` /
    :attr:`realized_token_savings_pct`) are summed over **PROMOTE** tasks only —
    a blocked or crashed task's token delta is *never* counted as a win.
    """

    schema_version: str = PHASE2_SCHEMA_VERSION
    generated_at: str | None = None
    suite_version: str = ""  # content label, e.g. "v1-seed"
    suite_content_hash: str = ""
    baseline_config: str = BASELINE.name
    candidate_config: str = CANDIDATE.name
    objective_metric: str = _TOKEN_METRIC
    min_token_reduction_pct: float = 0.0
    experiment: str | None = None
    profile: str | None = None
    warehouse_id: str | None = None

    n_tasks: int = 0
    n_promote: int = 0
    n_block: int = 0
    n_errored: int = 0

    realized_baseline_tokens: float = 0.0
    realized_candidate_tokens: float = 0.0
    realized_token_savings_absolute: float = 0.0
    realized_token_savings_pct: float | None = None

    outcomes: list[TaskOutcome] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result -> outcome mapping
# ---------------------------------------------------------------------------


def _l1_outcome(prog: GuardrailCheck | None) -> L1Outcome:
    """Derive the :class:`L1Outcome` from the programmatic guardrail (no string-sniffing).

    ``prog is None`` means no verification was configured (with ``NO_LLM_JUDGE``
    the harness then emits a failed correctness guardrail) → ``NOT_CONFIGURED``.
    Otherwise the guardrail's flags decide: ``regressed`` → ``REGRESSED``;
    ``not passed`` (and not regressed) is the errored/no-verdict branch →
    ``NO_VERDICT``; ``passed`` splits on the candidate signal into ``PASSED``
    (candidate verifies) vs ``FAILED_BOTH`` (failed on both arms, not a
    regression).
    """
    if prog is None:
        return L1Outcome.NOT_CONFIGURED
    if prog.regressed:
        return L1Outcome.REGRESSED
    if not prog.passed:
        return L1Outcome.NO_VERDICT
    return L1Outcome.PASSED if bool(prog.candidate_value) else L1Outcome.FAILED_BOTH


def _outcome_from_result(
    task: Task, result: ComparisonResult, *, verification_configured: bool
) -> TaskOutcome:
    """Fold a :class:`ComparisonResult` into a :class:`TaskOutcome`."""
    tokens = result.delta_for(_TOKEN_METRIC)
    execution = result.guardrail_for(EXECUTION_GUARDRAIL)
    prog = result.guardrail_for(PROGRAMMATIC_GUARDRAIL)
    # Reuse the harness's own justification strings (they name the failing guardrail).
    blocking = list(result.reasons) if result.recommendation is Recommendation.BLOCK else []
    return TaskOutcome(
        task_id=task.task_id,
        category=task.category.value,
        difficulty=task.difficulty.value,
        recommendation=result.recommendation,
        objective_met=result.objective_met,
        guardrails_passed=result.guardrails_passed,
        baseline_total_tokens=tokens.baseline if tokens else 0.0,
        candidate_total_tokens=tokens.candidate if tokens else 0.0,
        token_delta_absolute=tokens.delta_absolute if tokens else 0.0,
        token_delta_pct=tokens.delta_pct if tokens else None,
        token_improved=bool(tokens.improved) if tokens else False,
        l1_outcome=_l1_outcome(prog),
        l1_verification_configured=verification_configured,
        baseline_succeeded=bool(execution.baseline_value) if execution else False,
        candidate_succeeded=bool(execution.candidate_value) if execution else False,
        baseline_trace_id=result.baseline_trace_id,
        candidate_trace_id=result.candidate_trace_id,
        blocking_reasons=blocking,
        error=None,
        comparison=result,
    )


def _errored_outcome(
    task: Task, exc: BaseException, *, verification_configured: bool
) -> TaskOutcome:
    """A fail-closed outcome for a task whose comparison **raised**.

    Recorded as ``BLOCK`` with the error captured and **no** claimed token win or
    execution success — a comparison that did not complete is never a pass.
    """
    return TaskOutcome(
        task_id=task.task_id,
        category=task.category.value,
        difficulty=task.difficulty.value,
        recommendation=Recommendation.BLOCK,
        objective_met=False,
        guardrails_passed=False,
        l1_outcome=L1Outcome.NO_VERDICT,
        l1_verification_configured=verification_configured,
        baseline_succeeded=False,
        candidate_succeeded=False,
        blocking_reasons=[f"comparison raised: {type(exc).__name__}: {exc}"],
        error=f"{type(exc).__name__}: {exc}",
        comparison=None,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_phase2_comparison(
    *,
    suite: TaskSuite,
    adapter: AgentAdapter,
    candidate: LeverConfig = CANDIDATE,
    baseline: LeverConfig = BASELINE,
    verify_specs: Mapping[str, VerifySpec] | None = None,
    config: ComparisonConfig | None = None,
    experiment: str | None = None,
    profile: str | None = None,
    warehouse_id: str | None = None,
    task_ids: Collection[str] | None = None,
    generated_at: str | None = None,
) -> Phase2Artifact:
    """Run baseline-vs-candidate across the frozen suite and emit a :class:`Phase2Artifact`.

    Args:
        suite: The frozen Task Suite (read only; never mutated).
        adapter: The agent adapter to run each task through (e.g. the Claude Code
            adapter). Run twice per task by the harness — baseline then candidate.
        candidate: The CANDIDATE config (asset enabled). Must carry an
            intervention; a baseline-shaped config here is a programmer error.
        baseline: The BASELINE config (no asset). Must **not** carry an
            intervention; recorded for provenance (the harness runs the baseline
            arm as the un-intervened task).
        verify_specs: Per-task L1 verification commands keyed by ``task_id``. A
            task with no entry has no correctness signal and fails closed (BLOCK).
        config: Objective metric + reduction threshold. Defaults to a strict
            ``total_tokens`` reduction with a 0% floor.
        experiment, profile, warehouse_id: Recorded on the artifact for provenance
            (the CLI does the actual workspace wiring; this function does no I/O).
        task_ids: Optional subset of task ids to run; ``None`` runs all.
        generated_at: ISO-8601 stamp recorded on the artifact and each comparison
            (caller-supplied so a run is reproducible/deterministic in tests).

    Returns:
        A :class:`Phase2Artifact`.

    Raises:
        ValueError: if ``candidate`` has no intervention, or ``baseline`` has one.
    """
    if candidate.intervention is None:
        raise ValueError(
            f"candidate config {candidate.name!r} has no intervention; it is not a candidate "
            "(the candidate is the arm with the asset enabled)"
        )
    if baseline.intervention is not None:
        raise ValueError(
            f"baseline config {baseline.name!r} carries an intervention; the baseline must run "
            "the task with no asset"
        )

    cfg = config or ComparisonConfig(objective_metric=_TOKEN_METRIC)
    stamp = generated_at or datetime.now(UTC).isoformat()
    specs = dict(verify_specs or {})
    selected = [t for t in suite.tasks if task_ids is None or t.task_id in task_ids]

    outcomes: list[TaskOutcome] = []
    for task in selected:
        spec = specs.get(task.task_id)
        check = make_command_check(spec) if spec is not None else None
        try:
            result = compare_candidate(
                case_from_task(task),
                adapter,
                intervention=candidate.intervention,
                correctness_judge=NO_LLM_JUDGE,
                programmatic_check=check,
                config=cfg,
                generated_at=stamp,
            )
        except Exception as exc:  # noqa: BLE001 - a single task error must not abort the run
            outcomes.append(_errored_outcome(task, exc, verification_configured=spec is not None))
            continue
        outcomes.append(
            _outcome_from_result(task, result, verification_configured=spec is not None)
        )

    return _assemble_artifact(
        suite=suite,
        candidate=candidate,
        baseline=baseline,
        cfg=cfg,
        outcomes=outcomes,
        experiment=experiment,
        profile=profile,
        warehouse_id=warehouse_id,
        generated_at=stamp,
        specs=specs,
        selected_count=len(selected),
    )


def _assemble_artifact(
    *,
    suite: TaskSuite,
    candidate: LeverConfig,
    baseline: LeverConfig,
    cfg: ComparisonConfig,
    outcomes: list[TaskOutcome],
    experiment: str | None,
    profile: str | None,
    warehouse_id: str | None,
    generated_at: str,
    specs: Mapping[str, VerifySpec],
    selected_count: int,
) -> Phase2Artifact:
    """Aggregate per-task outcomes into the artifact (realized savings over PROMOTE only)."""
    promoted = [o for o in outcomes if o.recommendation is Recommendation.PROMOTE]
    errored = [o for o in outcomes if o.error is not None]

    realized_baseline = round(sum(o.baseline_total_tokens for o in promoted), 6)
    realized_candidate = round(sum(o.candidate_total_tokens for o in promoted), 6)
    realized_savings = round(realized_baseline - realized_candidate, 6)
    realized_pct = (
        round(100.0 * realized_savings / realized_baseline, 4) if realized_baseline > 0 else None
    )

    notes = [
        "Correctness is gated by deterministic L1 programmatic checks (NO_LLM_JUDGE); no "
        "uncalibrated LLM judge is in the decision path.",
        "Realized token savings are summed over PROMOTE tasks only; a blocked or crashed "
        "task's token delta is never counted as a win.",
    ]
    unverified = [o.task_id for o in outcomes if not o.l1_verification_configured]
    if unverified:
        notes.append(
            f"{len(unverified)} task(s) had no L1 verification configured and were blocked "
            f"(fail-closed, no correctness signal): {', '.join(unverified)}"
        )

    return Phase2Artifact(
        generated_at=generated_at,
        suite_version=suite.version,
        suite_content_hash=suite.content_hash,
        baseline_config=baseline.name,
        candidate_config=candidate.name,
        objective_metric=cfg.objective_metric,
        min_token_reduction_pct=cfg.min_token_reduction_pct,
        experiment=experiment,
        profile=profile,
        warehouse_id=warehouse_id,
        n_tasks=selected_count,
        n_promote=len(promoted),
        n_block=sum(1 for o in outcomes if o.recommendation is Recommendation.BLOCK),
        n_errored=len(errored),
        realized_baseline_tokens=realized_baseline,
        realized_candidate_tokens=realized_candidate,
        realized_token_savings_absolute=realized_savings,
        realized_token_savings_pct=realized_pct,
        outcomes=outcomes,
        notes=notes,
    )
