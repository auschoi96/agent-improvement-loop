"""The candidate-vs-baseline comparison harness — the reusable core of Phase 2.

This is the "evaluate candidate vs baseline" box of the loop
(``docs/ARCHITECTURE.md`` §4), built as a standalone, agent-agnostic unit so the
loop controller (and tests) can call it directly. Given one frozen Task-Suite
task, an :class:`~ail.ingest.base.AgentAdapter`, and an optional
:class:`Intervention`, it:

1. runs the agent **without** the intervention (the baseline) and **with** it
   (the candidate), capturing a :class:`~ail.ingest.base.NormalizedTrace` from
   each run via the adapter (the live-ingest seam — a captured trace is identical
   in shape to one a :class:`~ail.ingest.base.TraceSource` reads back);
2. computes deterministic **L0** metrics for both traces
   (:func:`ail.metrics.l0_deterministic.compute_trace_metrics`) and the per-metric
   deltas (tokens / cost / redundancy);
3. applies the **guardrail** — the anti-co-adaptation gate — and
4. emits a structured :class:`~ail.compare.contract.ComparisonResult` with an
   overall ``PROMOTE`` / ``BLOCK`` recommendation.

**The objective is the L0 reduction; the guardrail is correctness.** Token/cost
reduction is read straight off L0 — deterministic and un-gameable (no model in
the loop, nothing the agent can inflate). A candidate is recommended ``PROMOTE``
only when it achieves that reduction **and** no guardrail regressed. The
correctness guardrail is **non-regression** relative to the baseline: a candidate
is blocked if it makes correctness *worse*, not because the baseline was already
imperfect.

**Fail closed on execution failure.** Before correctness, an execution-success
guardrail blocks the promotion unless **both** the baseline and the candidate
ran to success. This closes a "fake-good" hole: a crashed candidate uses few
tokens precisely because it did nothing, so its apparent token "reduction" must
never count as an improvement, and a failed baseline makes the comparison itself
untrustworthy. A failed run is a first-class fail-closed condition, not a note —
alongside the un-scorable-guardrail case.

**Interim guardrail (documented, not faked).** The correctness guardrail uses the
**BASE** correctness judge from :mod:`ail.judges.scorers` (plus any available L1
programmatic signal) TODAY, because the reference experiment has zero human
labels and MemAlign has nothing to align against (``docs/ARCHITECTURE.md`` §8).
This base judge is **not** MemAlign-aligned and **not** judge-vs-human calibrated;
it is recorded as ``interim`` on every :class:`~ail.compare.contract.GuardrailCheck`
(see :data:`~ail.compare.contract.INTERIM_JUDGE_NOTE`). It switches to the
MemAlign-aligned, Human-Anchor-audited judge once labels exist — we do not fake
alignment by pretending an unaligned judge is calibrated.

**L1-only correctness (:data:`NO_LLM_JUDGE`).** Pass :data:`NO_LLM_JUDGE` as the
``correctness_judge`` to take the LLM judge out of the decision path entirely and
let the deterministic **L1 programmatic** check (tests/build pass) *be* the
correctness guardrail. This is the honest mode for the frozen Task Suite: its
tasks are :class:`~ail.task_suite.schema.Task` reconstructions with **no
human-authored expectations** (a TASK_SUITE ground-truth set must be empty), so a
judge would have nothing to score against and an uncalibrated judge would be
distrusted by the readiness wall anyway. Fail-closed is preserved: with
:data:`NO_LLM_JUDGE` and no ``programmatic_check`` there is no correctness signal
at all, so a **failed** correctness guardrail is emitted and the recommendation is
``BLOCK``.

**Frozen-suite contract.** The harness only ever *reads* the task case — it never
writes to, mutates, re-pools, or trains against the Task Suite. The case is a
frozen pydantic model and the harness derives fresh
:class:`~ail.ingest.base.AgentTask` objects from it without touching it; the
:class:`~ail.compare.contract.ComparisonResult` is a separate artifact. A test
asserts the input case is byte-for-byte unchanged after a run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ail.compare.contract import (
    INTERIM_JUDGE_NOTE,
    ComparisonResult,
    GuardrailCheck,
    MetricDelta,
    Recommendation,
)
from ail.groundtruth.schema import GroundTruthCase
from ail.ingest.base import AgentAdapter, AgentRunResult, AgentTask
from ail.judges.agreement import coerce_score
from ail.metrics.l0_deterministic import compute_trace_metrics

if TYPE_CHECKING:
    from mlflow.genai.judges import Judge

    from ail.metrics.contract import PriceBookEntry, TraceMetrics
    from ail.pools import ScoreValue

__all__ = [
    "Intervention",
    "CallableIntervention",
    "ProgrammaticSignal",
    "ProgrammaticCheck",
    "ArmVerifier",
    "ArmWorkspaces",
    "ComparisonConfig",
    "NO_LLM_JUDGE",
    "compare_candidate",
]

#: Name of the execution-success guardrail check on the result (fails closed when
#: either agent run did not succeed).
EXECUTION_GUARDRAIL = "execution"
#: Name of the correctness guardrail check on the result.
CORRECTNESS_GUARDRAIL = "correctness"
#: Name of the (optional) L1 programmatic guardrail check on the result.
PROGRAMMATIC_GUARDRAIL = "programmatic"


class _NoLLMJudge:
    """The type of the :data:`NO_LLM_JUDGE` sentinel (one instance, opaque)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "NO_LLM_JUDGE"


#: Pass as ``correctness_judge`` to run the correctness guardrail on the **L1
#: programmatic signal alone**, with **no LLM judge in the decision path**.
#:
#: This is the honest mode for the frozen Task Suite, whose tasks carry **no
#: human-authored expectations** (a :class:`~ail.task_suite.schema.TaskSuite`
#: holds :class:`~ail.task_suite.schema.Task` reconstructions, and a TASK_SUITE
#: ground-truth set must be empty — see
#: :func:`ail.groundtruth.schema.validate_pool_membership`). With nothing to score
#: against, an LLM judge would be un-scorable on every task and fail closed
#: everywhere; and a judge built on zero human labels is uncalibrated and would be
#: distrusted by the readiness wall (``docs/ARCHITECTURE.md`` §8). So the
#: deterministic, externally-verifiable L1 check (tests/build pass) **is** the
#: correctness guardrail.
#:
#: Fail-closed contract: ``NO_LLM_JUDGE`` does **not** weaken the gate. When it is
#: set and no ``programmatic_check`` is supplied, there is no correctness signal
#: at all, so :func:`compare_candidate` emits a **failed** correctness guardrail
#: (``passed=False``) and the recommendation is ``BLOCK`` — a promotion is never
#: certified without a correctness guardrail.
NO_LLM_JUDGE: _NoLLMJudge = _NoLLMJudge()


class Intervention(ABC):
    """A transform applied to a baseline :class:`~ail.ingest.base.AgentTask`.

    An intervention is *how* the candidate differs from the baseline — a skill
    update that points the agent at a new tool, a changed system prompt, an
    added allowed tool. It is **not** a change to the task being asked: the
    user's request (and the ground-truth expectations a judge checks against) are
    identical for baseline and candidate, so the comparison isolates the effect
    of the intervention alone.

    Implementations must be **pure**: :meth:`apply` returns a *new* ``AgentTask``
    and must not mutate the one it is given (the harness reuses the baseline task
    to run the baseline). :class:`CallableIntervention` is the convenience form.
    """

    #: Short, stable identifier recorded on the
    #: :class:`~ail.compare.contract.ComparisonResult`.
    name: str = "intervention"

    @abstractmethod
    def apply(self, task: AgentTask) -> AgentTask:
        """Return a new ``AgentTask`` reflecting this intervention."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class CallableIntervention(Intervention):
    """An :class:`Intervention` backed by a function ``AgentTask -> AgentTask``.

    The general form: wrap any task transform without subclassing. The function
    must return a new task (e.g. via :func:`dataclasses.replace`) rather than
    mutating its argument. Constructed by keyword
    (``CallableIntervention(name=..., transform=...)``) so the required
    ``transform`` is unaffected by the base class's default ``name``.
    """

    name: str = "intervention"
    transform: Callable[[AgentTask], AgentTask]

    def apply(self, task: AgentTask) -> AgentTask:
        return self.transform(task)


@dataclass(frozen=True, slots=True)
class ProgrammaticSignal:
    """An L1 programmatic pass/fail for one run (tests / lint / typecheck / build).

    The objective L1 tier of the layered metrics (``docs/ARCHITECTURE.md`` §3):
    an externally-run, verifiable signal. The harness does not run these checks
    itself (there is no L1 runner in scope here); a caller supplies a
    :data:`ProgrammaticCheck` that derives one of these from an
    :class:`~ail.ingest.base.AgentRunResult`.

    ``errored`` distinguishes "the check **ran** and reported a fail"
    (``passed=False, errored=False``) from "the check could **not** produce a
    verdict" (``errored=True`` — the verification command could not be launched,
    timed out, or crashed). The two must not be conflated: a check that *ran* and
    failed is real data the guardrail acts on — a **failed baseline** is not a valid
    anchor (the objective is "same quality for fewer tokens", and a token reduction
    against a baseline that never passed proves nothing), so it fails closed, while
    a failed candidate against a passing baseline is a regression. A check that
    produced **no verdict** is missing data and likewise fails the guardrail closed
    — an unmeasured guardrail never certifies a promotion (see
    :func:`_programmatic_guardrail`).
    """

    name: str
    passed: bool
    details: str = ""
    errored: bool = False


#: A caller-supplied L1 check: derive a :class:`ProgrammaticSignal` from a run's
#: result. The harness applies the *same* check to the baseline and the candidate
#: so the programmatic guardrail compares like with like. **Arm-blind**: the check
#: sees only the run result, not which workspace produced it — fine for a mock
#: trace, but for a file-mutating task use :data:`ArmVerifier` via
#: :class:`ArmWorkspaces` so the verdict is measured in the arm's own workspace.
ProgrammaticCheck = Callable[[AgentRunResult], ProgrammaticSignal]

#: An **arm-aware** L1 verification callable: given one arm's workspace path and
#: that arm's run result, produce a :class:`ProgrammaticSignal`. Unlike
#: :data:`ProgrammaticCheck` (arm-blind — the same closure run against both
#: results), this receives the arm's own ``cwd``, so the live runner can restore
#: the pristine verify files into *that* workspace and run the check *there* — the
#: baseline verdict reflects the baseline's edits and the candidate verdict the
#: candidate's, never a single fixed-cwd run.
ArmVerifier = Callable[[str, AgentRunResult], ProgrammaticSignal]


@dataclass(frozen=True, slots=True)
class ArmWorkspaces:
    """Per-arm isolated workspaces (+ optional arm-aware verifier) for one comparison.

    For a **real, file-mutating** coding task the two arms must edit *separate*
    copies of the starting repo state: ``baseline_cwd`` and ``candidate_cwd`` are
    distinct directories the harness sets as each arm's
    :attr:`~ail.ingest.base.AgentTask.cwd` (via :func:`dataclasses.replace`), so
    the candidate's edits can never land on top of the baseline's and vice-versa
    (no cross-arm contamination).

    ``verify`` is the arm-aware L1 check run *after* each arm's agent run, in that
    arm's workspace (see :data:`ArmVerifier`). ``None`` leaves correctness to the
    LLM judge or, under :data:`NO_LLM_JUDGE` with no verifier, fails closed — the
    same no-signal rule a missing ``programmatic_check`` triggers. Passing both a
    ``programmatic_check`` and a ``verify`` to :func:`compare_candidate` is a
    programmer error (two different L1 seams) and raises.
    """

    baseline_cwd: str
    candidate_cwd: str
    verify: ArmVerifier | None = None


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Knobs for the comparison.

    Args:
        objective_metric: The L0 metric whose reduction is the objective. Must be
            one of the emitted :class:`~ail.compare.contract.MetricDelta` metrics
            (default ``"total_tokens"``; ``"total_usd"`` for a cost objective).
        min_token_reduction_pct: Minimum reduction (in percent of the baseline)
            required for :attr:`~ail.compare.contract.ComparisonResult.objective_met`.
            The default ``0.0`` means *any strict reduction* counts; raise it so a
            candidate cannot promote on noise. A no-change candidate never meets
            the objective regardless of this value.
    """

    objective_metric: str = "total_tokens"
    min_token_reduction_pct: float = 0.0


# ---------------------------------------------------------------------------
# Task bridging (frozen suite case -> executable AgentTask)
# ---------------------------------------------------------------------------


def _baseline_task(case: GroundTruthCase) -> AgentTask:
    """Build the baseline :class:`AgentTask` from a case's task input.

    The same ``TaskInput -> AgentTask`` mapping the ground-truth execute stage
    uses (:func:`ail.groundtruth.execute._task_from_case`); kept local so the
    comparison harness does not reach into another stage's private helper, and a
    fresh task is built for each run so no two runs alias a mutable ``params``.
    """
    ti = case.task_input
    return AgentTask(
        prompt=ti.prompt,
        system_prompt=ti.system_prompt,
        model=ti.model,
        params=dict(ti.params),
    )


# ---------------------------------------------------------------------------
# L0 deltas
# ---------------------------------------------------------------------------


def _delta(metric: str, unit: str, baseline: float, candidate: float) -> MetricDelta:
    """One lower-is-better :class:`MetricDelta` (tokens/cost/redundancy are all so)."""
    b = float(baseline)
    c = float(candidate)
    delta_abs = round(c - b, 6)
    delta_pct = round(100.0 * (c - b) / b, 4) if b != 0 else None
    return MetricDelta(
        metric=metric,
        unit=unit,
        lower_is_better=True,
        baseline=b,
        candidate=c,
        delta_absolute=delta_abs,
        delta_pct=delta_pct,
        improved=delta_abs < 0,  # strict: a tie is not an improvement
    )


def _build_deltas(
    baseline: TraceMetrics, candidate: TraceMetrics
) -> tuple[list[MetricDelta], list[str]]:
    """Per-metric baseline-vs-candidate deltas for the L0 tokens/cost/redundancy."""
    notes: list[str] = []
    # Token deltas iterate the TokenBreakdown fields rather than hardcoding names,
    # so a future ail.metrics token category is picked up automatically instead of
    # being silently dropped. (Cost and redundancy are structured, not flat token
    # counts, so they stay explicit below.)
    base_tokens = baseline.tokens.model_dump()
    cand_tokens = candidate.tokens.model_dump()
    deltas = [_delta(name, "tokens", base_tokens[name], cand_tokens[name]) for name in base_tokens]
    deltas += [
        _delta("total_tool_calls", "calls", baseline.total_tool_calls, candidate.total_tool_calls),
        _delta(
            "redundancy_rate",
            "rate",
            baseline.redundancy.redundancy_rate,
            candidate.redundancy.redundancy_rate,
        ),
        _delta("total_usd", "usd", baseline.cost.total_usd, candidate.cost.total_usd),
    ]
    # Cost is only honest when both sides were priced; flag an unpriced side so a
    # reader never mistakes a $0 delta for "no cost change" (see L0 cost contract).
    if not (baseline.cost.priced and candidate.cost.priced):
        notes.append(
            "cost delta is partial: "
            f"baseline priced={baseline.cost.priced}, candidate priced={candidate.cost.priced}; "
            "an unpriced side contributes $0 and the total_usd delta understates the true change"
        )
    # Duration is producer-reported and may be absent; only compare when both have it.
    if baseline.duration_seconds is not None and candidate.duration_seconds is not None:
        deltas.append(
            _delta(
                "duration_seconds",
                "seconds",
                baseline.duration_seconds,
                candidate.duration_seconds,
            )
        )
    return deltas, notes


def _objective_met(delta: MetricDelta, min_reduction_pct: float) -> bool:
    """Whether a lower-is-better objective fell by at least ``min_reduction_pct``.

    Requires a **strict** reduction first (so a no-change candidate never meets
    the objective), then that the reduction clears the threshold. When the
    baseline is 0 the percentage is undefined and a reduction is impossible, so
    the objective is not met.
    """
    if delta.delta_absolute >= 0:
        return False
    if delta.delta_pct is None:  # baseline 0 -> reduction impossible (already handled above)
        return False
    return abs(delta.delta_pct) >= min_reduction_pct


# ---------------------------------------------------------------------------
# Guardrails (anti-co-adaptation gate)
# ---------------------------------------------------------------------------


def _execution_guardrail(
    baseline_result: AgentRunResult, candidate_result: AgentRunResult
) -> GuardrailCheck:
    """Fail closed unless BOTH agent runs succeeded — the most fundamental guardrail.

    A comparison is only trustworthy when the baseline *and* the candidate
    actually ran to success. A crashed or failed candidate can post a spurious
    token "reduction" — it did less work, or nothing — which must **never** count
    as an improvement; a failed baseline makes the whole comparison untrustworthy.
    Either way the recommendation must be ``BLOCK``. This sits first in the
    guardrail list and, like the un-scorable correctness branch, is a first-class
    fail-closed decision input rather than a mere note. ``regressed`` marks the
    specific "baseline succeeded, candidate broke" case; a failed baseline is
    un-comparable, not a measured regression.
    """
    baseline_ok = bool(baseline_result.success)
    candidate_ok = bool(candidate_result.success)
    passed = baseline_ok and candidate_ok
    if passed:
        reason = "both agent runs succeeded"
    else:
        failed = []
        if not baseline_ok:
            failed.append(f"baseline failed ({baseline_result.error or 'no detail'})")
        if not candidate_ok:
            failed.append(f"candidate failed ({candidate_result.error or 'no detail'})")
        reason = (
            "agent run failure ("
            + "; ".join(failed)
            + "); failing closed — a failed run's token change is not a real improvement, and a "
            "failed baseline makes the comparison untrustworthy"
        )
    return GuardrailCheck(
        name=EXECUTION_GUARDRAIL,
        passed=passed,
        reason=reason,
        baseline_value=baseline_ok,
        candidate_value=candidate_ok,
        regressed=baseline_ok and not candidate_ok,
        judge_name=None,
        interim=False,
        interim_note=None,
    )


# Map a correctness verdict (categorical yes/no, bool, or 0/1) to an order so
# "did it regress" is a comparison. Anything unrecognized is None (un-scorable),
# which fails the guardrail closed.
_CORRECTNESS_YES = {"yes", "true", "correct", "pass", "1"}
_CORRECTNESS_NO = {"no", "false", "incorrect", "fail", "0"}


def _correctness_rank(value: ScoreValue | None) -> int | None:
    """Rank a correctness verdict: correct -> 1, incorrect -> 0, un-scorable -> None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value >= 1 else 0
    text = str(value).strip().casefold()
    if text in _CORRECTNESS_YES:
        return 1
    if text in _CORRECTNESS_NO:
        return 0
    return None


def _score_correctness(
    judge: Judge, *, inputs: Any, outputs: Any, expectations: Any
) -> tuple[ScoreValue | None, str | None]:
    """Call the correctness judge for one run; capture failure rather than raise.

    Mirrors :func:`ail.judges.agreement._score_one`: a judge call that raises is
    captured as an error (the run is treated as un-scorable, which fails the
    guardrail closed) so one bad run never aborts the comparison.
    """
    try:
        result = judge(inputs=inputs, outputs=outputs, expectations=expectations)
        return coerce_score(result), None
    except Exception as exc:  # noqa: BLE001 - a judge failure must not abort the comparison
        return None, str(exc)


def _correctness_guardrail(
    judge: Judge,
    *,
    case: GroundTruthCase,
    baseline_result: AgentRunResult,
    candidate_result: AgentRunResult,
) -> GuardrailCheck:
    """The correctness non-regression guardrail, scored by the BASE judge (interim).

    Scores baseline and candidate against the case's human-authored expectations
    with the **same** task request as ``inputs`` (only the agent's response
    differs), then passes iff the candidate did not score *worse* than the
    baseline. Fails closed when either side cannot be scored — an unmeasured
    guardrail must never certify a promotion (the anti-co-adaptation fail-closed
    rule, mirroring :mod:`ail.judges.agreement`).
    """
    inputs = case.task_input.prompt
    expectations = case.expectations.model_dump()
    judge_name = getattr(judge, "name", "correctness")

    baseline_value, baseline_err = _score_correctness(
        judge, inputs=inputs, outputs=baseline_result.output_text, expectations=expectations
    )
    candidate_value, candidate_err = _score_correctness(
        judge, inputs=inputs, outputs=candidate_result.output_text, expectations=expectations
    )

    base_rank = _correctness_rank(baseline_value)
    cand_rank = _correctness_rank(candidate_value)

    if base_rank is None or cand_rank is None:
        problems = []
        if base_rank is None:
            problems.append(f"baseline un-scorable ({baseline_err or baseline_value!r})")
        if cand_rank is None:
            problems.append(f"candidate un-scorable ({candidate_err or candidate_value!r})")
        reason = (
            "correctness could not be measured ("
            + "; ".join(problems)
            + "); failing closed — an unmeasured guardrail never certifies a promotion"
        )
        return GuardrailCheck(
            name=CORRECTNESS_GUARDRAIL,
            passed=False,
            reason=reason,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            regressed=False,
            judge_name=judge_name,
            interim=True,
            interim_note=INTERIM_JUDGE_NOTE,
        )

    regressed = cand_rank < base_rank
    if regressed:
        reason = (
            f"correctness REGRESSED: baseline={baseline_value!r} -> candidate={candidate_value!r}; "
            "the intervention made correctness worse"
        )
    elif cand_rank > base_rank:
        reason = (
            f"correctness improved: baseline={baseline_value!r} -> candidate={candidate_value!r}"
        )
    else:
        reason = f"correctness held at {candidate_value!r} (no regression)"
    return GuardrailCheck(
        name=CORRECTNESS_GUARDRAIL,
        passed=not regressed,
        reason=reason,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        regressed=regressed,
        judge_name=judge_name,
        interim=True,
        interim_note=INTERIM_JUDGE_NOTE,
    )


def _no_correctness_signal_guardrail() -> GuardrailCheck:
    """A failed correctness guardrail standing in for *no correctness signal at all*.

    Emitted when the LLM judge is disabled (:data:`NO_LLM_JUDGE`) **and** no L1
    ``programmatic_check`` was supplied: there is then nothing measuring
    correctness, so the gate fails closed rather than letting a token reduction
    promote unguarded. An unmeasured guardrail never certifies a promotion.
    """
    return GuardrailCheck(
        name=CORRECTNESS_GUARDRAIL,
        passed=False,
        reason=(
            "no correctness signal: the LLM judge is disabled (NO_LLM_JUDGE) and no L1 "
            "programmatic check was supplied; failing closed — a promotion is never certified "
            "without a correctness guardrail"
        ),
        baseline_value=None,
        candidate_value=None,
        regressed=False,
        judge_name=None,
        interim=False,
        interim_note=None,
    )


def _programmatic_guardrail(
    baseline: ProgrammaticSignal, candidate: ProgrammaticSignal
) -> GuardrailCheck:
    """Optional L1 guardrail anchored on a PASSING baseline: fail closed otherwise.

    The objective is "same quality for fewer tokens", so the baseline's token count
    only anchors a reduction if the baseline actually **passed** its L1 check. A
    baseline that did not pass is not a valid anchor: a token drop measured against
    a failed baseline proves nothing, so it fails closed (``BLOCK``) regardless of
    the candidate — independently of whether the candidate also failed. This is the
    same fail-closed treatment the execution guardrail applies to a failed baseline;
    a failed baseline that *also* fails for the candidate must never read as "fails
    for both, no regression" and certify a promotion.

    Fails closed first on a **no-verdict** signal: if either side's check could
    not run (:attr:`ProgrammaticSignal.errored`), there is no measurement to
    compare, so the guardrail fails regardless of the ``passed`` flags — a broken
    or un-runnable verification is missing data, not a verdict.

    With a passing baseline as the anchor, the candidate must not break what passed:
    a check that passed at baseline and fails for the candidate is a regression and
    ``BLOCK``s.
    """
    if baseline.errored or candidate.errored:
        problems = []
        if baseline.errored:
            problems.append(
                f"baseline check '{baseline.name}' produced no verdict ({baseline.details})"
            )
        if candidate.errored:
            problems.append(
                f"candidate check '{candidate.name}' produced no verdict ({candidate.details})"
            )
        reason = (
            "L1 check produced no verdict ("
            + "; ".join(problems)
            + "); failing closed — an unmeasured guardrail never certifies a promotion"
        )
        return GuardrailCheck(
            name=PROGRAMMATIC_GUARDRAIL,
            passed=False,
            reason=reason,
            baseline_value=baseline.passed,
            candidate_value=candidate.passed,
            regressed=False,
            judge_name=None,
            interim=False,
            interim_note=None,
        )
    if not baseline.passed:
        reason = (
            f"L1 '{baseline.name}' did not pass at baseline; failing closed — a failed baseline "
            "is not a valid anchor, so a token reduction against it proves nothing"
            + (f" ({baseline.details})" if baseline.details else "")
        )
        return GuardrailCheck(
            name=PROGRAMMATIC_GUARDRAIL,
            passed=False,
            reason=reason,
            baseline_value=baseline.passed,
            candidate_value=candidate.passed,
            regressed=False,
            judge_name=None,
            interim=False,
            interim_note=None,
        )
    # Baseline passed: it is a valid anchor. Only a candidate that breaks what
    # passed is a regression.
    regressed = baseline.passed and not candidate.passed
    if regressed:
        reason = f"L1 '{candidate.name}' REGRESSED: passed at baseline, fails for candidate" + (
            f" ({candidate.details})" if candidate.details else ""
        )
    else:
        reason = f"L1 '{candidate.name}' passes for the candidate"
    return GuardrailCheck(
        name=PROGRAMMATIC_GUARDRAIL,
        passed=not regressed,
        reason=reason,
        baseline_value=baseline.passed,
        candidate_value=candidate.passed,
        regressed=regressed,
        judge_name=None,
        interim=False,
        interim_note=None,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_programmatic_signals(
    *,
    programmatic_check: ProgrammaticCheck | None,
    workspace: ArmWorkspaces | None,
    baseline_result: AgentRunResult,
    candidate_result: AgentRunResult,
) -> tuple[ProgrammaticSignal, ProgrammaticSignal] | None:
    """The (baseline, candidate) L1 signals, arm-aware when a workspace verifier is set.

    Arm-aware (live) verification wins when a workspace carries a verifier: each
    arm is verified in its **own** workspace (``baseline_cwd`` / ``candidate_cwd``),
    so the baseline signal reflects the baseline's edits and the candidate signal
    the candidate's — never one fixed-cwd run that ignores which arm produced the
    state. Otherwise the legacy arm-blind ``programmatic_check`` (the same callable
    applied to both results) is used; ``None`` means no L1 signal at all. The two
    signals feed the *unchanged* :func:`_programmatic_guardrail`, so the fail-closed
    semantics are identical regardless of which seam produced them.
    """
    if workspace is not None and workspace.verify is not None:
        return (
            workspace.verify(workspace.baseline_cwd, baseline_result),
            workspace.verify(workspace.candidate_cwd, candidate_result),
        )
    if programmatic_check is not None:
        return (programmatic_check(baseline_result), programmatic_check(candidate_result))
    return None


def compare_candidate(
    case: GroundTruthCase,
    adapter: AgentAdapter,
    *,
    intervention: Intervention | None = None,
    correctness_judge: Judge | _NoLLMJudge | None = None,
    programmatic_check: ProgrammaticCheck | None = None,
    workspace: ArmWorkspaces | None = None,
    config: ComparisonConfig | None = None,
    pricebook: dict[str, PriceBookEntry] | None = None,
    generated_at: str | None = None,
) -> ComparisonResult:
    """Run baseline vs candidate on one frozen task and recommend PROMOTE / BLOCK.

    Args:
        case: A frozen Task-Suite case (task input + human-authored expectations
            + provenance). **Read only** — never mutated, re-pooled, or persisted
            back; the comparison result is a separate artifact.
        adapter: The agent to run. Called once for the baseline task and once for
            the candidate task; each :class:`~ail.ingest.base.AgentRunResult`
            carries the captured :class:`~ail.ingest.base.NormalizedTrace`.
        intervention: The change under test. ``None`` runs the same baseline task
            twice (e.g. to measure noise); with no token reduction that yields
            ``BLOCK`` (no false promotion), which the tests assert.
        correctness_judge: The guardrail judge. Defaults to the **base**
            correctness judge (:func:`ail.judges.scorers.make_correctness_judge`)
            built with ``temperature=0`` for reproducible scoring. INTERIM and not
            MemAlign-aligned — see the module docstring. Injectable so tests pass a
            scripted judge with no model call. Pass :data:`NO_LLM_JUDGE` to run
            **no** LLM judge and let the L1 ``programmatic_check`` be the
            correctness guardrail — the honest mode for the frozen Task Suite,
            whose tasks carry no human-authored expectations to score against. With
            ``NO_LLM_JUDGE`` and no ``programmatic_check`` the gate fails closed
            (a failed correctness guardrail, ``BLOCK``).
        programmatic_check: Optional **arm-blind** L1 check applied to **both**
            runs to add a programmatic non-regression guardrail. ``None`` omits it
            (unless ``NO_LLM_JUDGE`` is set with no ``workspace`` verifier either,
            in which case its absence fails closed). For a file-mutating task use
            ``workspace`` instead — passing both raises.
        workspace: Optional :class:`ArmWorkspaces` giving each arm its **own**
            ``cwd`` (so a file-mutating agent edits isolated copies — no cross-arm
            contamination) and an optional **arm-aware** verifier run in each arm's
            workspace. When its ``verify`` is set it is the programmatic guardrail
            source (mutually exclusive with ``programmatic_check``); its absence
            under ``NO_LLM_JUDGE`` fails closed exactly like a missing
            ``programmatic_check``.
        config: Objective metric + reduction threshold knobs.
        pricebook: Optional L0 price-book override (passed to
            :func:`ail.metrics.l0_deterministic.compute_trace_metrics`).
        generated_at: ISO-8601 stamp recorded on the result (defaults to now).

    Returns:
        A :class:`~ail.compare.contract.ComparisonResult`.

    Raises:
        ValueError: if ``config.objective_metric`` is not one of the emitted
            metric deltas.
    """
    cfg = config or ComparisonConfig()

    if programmatic_check is not None and workspace is not None and workspace.verify is not None:
        raise ValueError(
            "pass either programmatic_check (arm-blind) or workspace.verify (arm-aware), "
            "not both: they are two different L1 verification seams and supplying both is "
            "ambiguous about which produces the programmatic guardrail"
        )

    # Build a fresh task per run so the intervention can never alias / mutate the
    # baseline's task (interventions are contractually pure, but do not rely on it).
    baseline_task = _baseline_task(case)
    candidate_task = _baseline_task(case)
    if intervention is not None:
        # ``replace()`` with no changes hands the intervention an independent copy.
        candidate_task = intervention.apply(replace(candidate_task))
    if workspace is not None:
        # Per-arm isolation: each arm runs in its OWN workspace, so a file-mutating
        # candidate edits a separate copy and the arms never see each other's edits.
        baseline_task = replace(baseline_task, cwd=workspace.baseline_cwd)
        candidate_task = replace(candidate_task, cwd=workspace.candidate_cwd)

    baseline_result = adapter.run(baseline_task)
    candidate_result = adapter.run(candidate_task)

    baseline_metrics = compute_trace_metrics(baseline_result.trace, pricebook=pricebook)
    candidate_metrics = compute_trace_metrics(candidate_result.trace, pricebook=pricebook)

    deltas, notes = _build_deltas(baseline_metrics, candidate_metrics)

    objective_delta = next((d for d in deltas if d.metric == cfg.objective_metric), None)
    if objective_delta is None:
        raise ValueError(
            f"objective_metric {cfg.objective_metric!r} is not an emitted metric; "
            f"choose one of {[d.metric for d in deltas]}"
        )
    objective_met = _objective_met(objective_delta, cfg.min_token_reduction_pct)

    # Resolve the two L1 signals once, from whichever verification seam is wired
    # (arm-aware workspace verifier or legacy arm-blind check); ``None`` means no
    # L1 signal at all. They feed the unchanged _programmatic_guardrail below.
    prog_signals = _resolve_programmatic_signals(
        programmatic_check=programmatic_check,
        workspace=workspace,
        baseline_result=baseline_result,
        candidate_result=candidate_result,
    )

    # Execution success is the most fundamental gate (a failed run is untrustworthy
    # and its token "savings" are not real), so it leads the guardrail list.
    guardrails = [_execution_guardrail(baseline_result, candidate_result)]
    if isinstance(correctness_judge, _NoLLMJudge):
        # No LLM judge in the decision path: the L1 programmatic check is the
        # correctness guardrail. With no L1 signal at all (no check and no
        # arm-aware verifier) there is no correctness signal, so fail closed with
        # an explicit failed guardrail.
        if prog_signals is None:
            guardrails.append(_no_correctness_signal_guardrail())
    else:
        judge = correctness_judge if correctness_judge is not None else _default_correctness_judge()
        guardrails.append(
            _correctness_guardrail(
                judge,
                case=case,
                baseline_result=baseline_result,
                candidate_result=candidate_result,
            )
        )
    if prog_signals is not None:
        guardrails.append(_programmatic_guardrail(*prog_signals))

    guardrails_passed = all(g.passed for g in guardrails)
    # PROMOTE requires the objective met AND every guardrail passed — which now
    # includes execution success, so a crashed candidate can never be promoted.
    promote = objective_met and guardrails_passed
    recommendation = Recommendation.PROMOTE if promote else Recommendation.BLOCK

    reasons = _reasons(
        promote=promote,
        objective_met=objective_met,
        objective_delta=objective_delta,
        guardrails=guardrails,
    )

    return ComparisonResult(
        task_id=case.case_id,
        intervention=intervention.name if intervention is not None else None,
        objective_metric=cfg.objective_metric,
        objective_met=objective_met,
        guardrails_passed=guardrails_passed,
        recommendation=recommendation,
        reasons=reasons,
        baseline_trace_id=baseline_result.trace.trace_id,
        candidate_trace_id=candidate_result.trace.trace_id,
        deltas=deltas,
        guardrails=guardrails,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        notes=notes,
    )


def _reasons(
    *,
    promote: bool,
    objective_met: bool,
    objective_delta: MetricDelta,
    guardrails: list[GuardrailCheck],
) -> list[str]:
    """The human-readable justification for the recommendation."""
    pct = "n/a" if objective_delta.delta_pct is None else f"{objective_delta.delta_pct:+.2f}%"
    obj_phrase = (
        f"{objective_delta.metric} {objective_delta.baseline:g} -> "
        f"{objective_delta.candidate:g} ({pct})"
    )
    if promote:
        reasons = [f"objective met: {obj_phrase}"]
        reasons += [f"guardrail '{g.name}' passed: {g.reason}" for g in guardrails]
        return reasons
    reasons = []
    if not objective_met:
        reasons.append(f"objective NOT met: {obj_phrase} (no qualifying reduction)")
    reasons += [f"guardrail '{g.name}' failed: {g.reason}" for g in guardrails if not g.passed]
    return reasons


def _default_correctness_judge() -> Judge:
    """Build the interim BASE correctness guardrail judge (temperature 0).

    Lazy: importing/calling this pulls MLflow, so it is only invoked when a caller
    did not inject a judge. Built with ``temperature=0`` so the guardrail verdict
    is reproducible. NOT MemAlign-aligned — see the module docstring.
    """
    from ail.judges.scorers import make_correctness_judge

    return make_correctness_judge(inference_params={"temperature": 0.0})
