"""The measured **facts** readiness consumes — pure, stdlib-only inputs.

Readiness is a pure function of a few numbers about a cohort, kept as small frozen
dataclasses so the gating logic is trivially testable with fixtures and never has
to touch MLflow, a model, or the cohort's traces directly. The caller is
responsible for *producing* these facts for a cohort — e.g. ``cohort.select`` then
:func:`ail.metrics.compute_l0` for the trace count, the ground-truth store for the
label count, :mod:`ail.task_suite` for frozen-suite presence, and
:mod:`ail.judges.agreement` for per-judge agreement (see
:meth:`JudgeFact.from_agreement_report`).

The **distrusted-by-default** rule of ``docs/READINESS_AND_TRUST.md`` §3 lives
here: a :class:`JudgeFact` with no measured agreement (``agreement_rate is None``)
is :attr:`~JudgeFact.is_distrusted` — an unmeasured judge is never trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

from ail.judges.agreement import DEFAULT_FLOOR
from ail.judges.contract import AgreementReport

__all__ = ["JudgeFact", "ReadinessFacts"]


@dataclass(frozen=True, slots=True)
class JudgeFact:
    """What readiness knows about one judge's trust and coverage over a cohort.

    Args:
        judge_name: Identifier of the judge.
        agreement_rate: Judge-vs-human agreement on the Human Anchor, or ``None``
            when the judge has **not** been measured against humans. ``None`` means
            unmeasured, which is distrusted by default.
        agreement_floor: The floor the rate is measured against (defaults to
            :data:`ail.judges.agreement.DEFAULT_FLOOR`); at or above it the judge
            is trusted, below it distrusted.
        distrusted: An explicit trust verdict when one is already known (e.g. taken
            from an :class:`~ail.judges.contract.AgreementReport`). When ``None``
            the verdict is derived from ``agreement_rate``/``agreement_floor``.
        insufficient_data: Set when agreement could not be measured on enough items
            (mirrors :attr:`AgreementReport.insufficient_data`); forces distrust.
        n_scored_traces: How many of the cohort's traces this judge produced a real
            verdict for (the per-judge coverage numerator).
    """

    judge_name: str
    agreement_rate: float | None = None
    agreement_floor: float = DEFAULT_FLOOR
    distrusted: bool | None = None
    insufficient_data: bool = False
    n_scored_traces: int = 0

    def __post_init__(self) -> None:
        if self.n_scored_traces < 0:
            raise ValueError(f"n_scored_traces must be >= 0, got {self.n_scored_traces}")

    @property
    def measured(self) -> bool:
        """Whether this judge was actually measured against humans.

        ``False`` for an unmeasured judge (no ``agreement_rate``) or one flagged
        ``insufficient_data`` — both of which read as distrusted.
        """
        return self.agreement_rate is not None and not self.insufficient_data

    @property
    def is_distrusted(self) -> bool:
        """Fail-closed trust verdict: unmeasured ⇒ distrusted, below floor ⇒ distrusted.

        An explicit :attr:`distrusted` wins when set (e.g. carried from an
        agreement report). Otherwise an unmeasured judge (or one with insufficient
        data) is distrusted, and a measured judge is distrusted iff its rate is
        below the floor.
        """
        if self.distrusted is not None:
            return self.distrusted
        rate = self.agreement_rate
        if rate is None or self.insufficient_data:
            return True
        return rate < self.agreement_floor

    @classmethod
    def from_agreement_report(
        cls, report: AgreementReport, *, n_scored_traces: int = 0
    ) -> JudgeFact:
        """Build a :class:`JudgeFact` from a judge's :class:`AgreementReport`.

        Carries the agreement module's own ``distrusted``/``insufficient_data``
        verdict through verbatim — readiness ties to that concept rather than
        re-deriving trust. ``n_scored_traces`` (how many cohort traces this judge
        scored) is supplied separately because the agreement report measures the
        Human Anchor, not the cohort.
        """
        return cls(
            judge_name=report.judge_name,
            agreement_rate=report.agreement_rate,
            agreement_floor=report.floor,
            distrusted=report.distrusted,
            insufficient_data=report.insufficient_data,
            n_scored_traces=n_scored_traces,
        )


@dataclass(frozen=True, slots=True)
class ReadinessFacts:
    """The cohort-level numbers readiness gates on.

    Args:
        trace_count: Number of traces in the cohort (``0`` is the empty / collecting
            case). The caller produces this via ``cohort.select(traces)``.
        label_count: Number of human labels available to calibrate a judge for this
            goal — the hard gate for a quality claim.
        frozen_suite_present: Whether a frozen Task Suite exists to compare against.
        n_scored_traces: Number of cohort traces carrying at least one real judge
            verdict (the cohort scored-coverage numerator).
        judge_runs: Total judge invocations attempted on the cohort.
        judge_run_successes: How many of those invocations produced a verdict
            without erroring (powers the judge-run success rate).
        judges: Per-judge agreement/coverage facts (see :class:`JudgeFact`).
    """

    trace_count: int = 0
    label_count: int = 0
    frozen_suite_present: bool = False
    n_scored_traces: int = 0
    judge_runs: int = 0
    judge_run_successes: int = 0
    judges: tuple[JudgeFact, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("trace_count", self.trace_count),
            ("label_count", self.label_count),
            ("n_scored_traces", self.n_scored_traces),
            ("judge_runs", self.judge_runs),
            ("judge_run_successes", self.judge_run_successes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.n_scored_traces > self.trace_count:
            raise ValueError(
                f"n_scored_traces ({self.n_scored_traces}) cannot exceed "
                f"trace_count ({self.trace_count})"
            )
        if self.judge_run_successes > self.judge_runs:
            raise ValueError(
                f"judge_run_successes ({self.judge_run_successes}) cannot exceed "
                f"judge_runs ({self.judge_runs})"
            )
