"""Compute per-cohort, per-goal readiness and the eval-health surface.

The two public functions implement ``docs/READINESS_AND_TRUST.md`` §2–§3:

* :func:`compute_eval_health` — the eval-health / coverage surface for a cohort:
  scored-coverage %, judge-run success rate, and the count of distrusted judges
  (an unmeasured judge is distrusted by default).
* :func:`compute_readiness` — per-cohort, per-goal readiness: evaluate each data
  gate, derive a :class:`~ail.readiness.contract.ReadinessTier`, and return a
  fail-closed :class:`~ail.readiness.contract.ReadinessStatus` that never reads
  green on missing data.

Both are pure functions of a :class:`~ail.cohorts.Cohort` (identity only — the
name everything keys off) and the measured :class:`~ail.readiness.facts.ReadinessFacts`.
Thresholds live in :class:`ReadinessThresholds` so the §2 ladder is **visible and
adjustable**, not buried constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from ail.cohorts import Cohort
from ail.readiness.contract import (
    EvalHealth,
    Gate,
    GateName,
    JudgeHealth,
    ReadinessStatus,
    ReadinessTier,
)
from ail.readiness.facts import JudgeFact, ReadinessFacts
from ail.readiness.goal import GoalView

__all__ = ["ReadinessThresholds", "compute_eval_health", "compute_readiness"]


@dataclass(frozen=True, slots=True)
class ReadinessThresholds:
    """The readiness ladder's gate thresholds — defaults that are meant to be tuned.

    ``docs/READINESS_AND_TRUST.md`` §2 is explicit that these are heuristics, not
    laws, and "must be visible and adjustable, not buried constants" — hence a
    frozen dataclass a caller can override per goal/risk tolerance, mirroring
    :class:`ail.judges.agreement.AgreementConfig`.

    Args:
        baseline_min_traces: Traces needed before an L0 baseline + waste diagnosis
            is meaningful (§2: ~10–20). Below it the cohort is *collecting*.
        prove_min_traces: Traces needed for statistical power to *prove* an
            improvement (§2: ~50+ — token usage is heavy-tailed, so a "reduction"
            on a handful of traces is noise).
        quality_min_labels: Human labels needed to calibrate a judge enough to
            trust it for a quality claim (§2 ladder: ≥~20 for a first trustworthy
            signal; raise toward ~30–50 to prove). Labels are the hard gate.
        scored_coverage_floor: Minimum fraction of cohort traces that must carry a
            real judge verdict before quality numbers count (§3 risk #2: low
            coverage ⇒ no claim).
    """

    baseline_min_traces: int = 10
    prove_min_traces: int = 50
    quality_min_labels: int = 20
    scored_coverage_floor: float = 0.5


def compute_eval_health(
    cohort: Cohort,
    facts: ReadinessFacts,
    *,
    thresholds: ReadinessThresholds | None = None,
    generated_at: str | None = None,
) -> EvalHealth:
    """Build the eval-health / coverage surface for ``cohort`` from ``facts``.

    Reports the fraction of traces actually scored (not merely "judges
    configured"), the judge-run success rate (``None`` when no runs were
    recorded — the fail-loud "did not evaluate" signal), and the count of
    distrusted judges. Every judge that has not been measured against humans, or
    is below its agreement floor, is counted as distrusted (never as trusted).
    """
    th = thresholds or ReadinessThresholds()
    n = facts.trace_count

    judges = [_judge_health(jf, n_traces=n) for jf in facts.judges]
    distrusted = [j.judge_name for j in judges if j.distrusted]
    scored_coverage = facts.n_scored_traces / n if n else 0.0
    run_rate = facts.judge_run_successes / facts.judge_runs if facts.judge_runs else None

    notes: list[str] = []
    if n == 0:
        notes.append("empty cohort: no traces to score (collecting)")
    elif facts.n_scored_traces == 0:
        notes.append(
            "no traces carry a real judge verdict (scored-coverage 0%): "
            "judges may be registered but not running"
        )
    if facts.judge_runs == 0 and facts.judges:
        notes.append("no judge runs recorded: success rate is undefined, not 100%")
    if distrusted:
        notes.append(
            f"{len(distrusted)} distrusted judge(s): {', '.join(distrusted)} "
            "(unmeasured or below agreement floor) — verdicts do not count toward claims"
        )

    return EvalHealth(
        cohort_name=cohort.name,
        n_traces=n,
        n_scored_traces=facts.n_scored_traces,
        scored_coverage=round(scored_coverage, 6),
        coverage_floor=th.scored_coverage_floor,
        judge_runs=facts.judge_runs,
        judge_run_successes=facts.judge_run_successes,
        judge_run_success_rate=round(run_rate, 6) if run_rate is not None else None,
        n_judges=len(judges),
        n_distrusted_judges=len(distrusted),
        distrusted_judges=distrusted,
        judges=judges,
        generated_at=generated_at,
        notes=notes,
    )


def _judge_health(fact: JudgeFact, *, n_traces: int) -> JudgeHealth:
    """Project one :class:`JudgeFact` onto the :class:`JudgeHealth` contract."""
    distrusted = fact.is_distrusted
    coverage = fact.n_scored_traces / n_traces if n_traces else 0.0
    if not fact.measured:
        reason = "unmeasured against humans → distrusted by default"
    elif distrusted:
        reason = f"agreement {fact.agreement_rate} below floor {fact.agreement_floor}"
    else:
        reason = f"agreement {fact.agreement_rate} at/above floor {fact.agreement_floor}"
    return JudgeHealth(
        judge_name=fact.judge_name,
        measured=fact.measured,
        agreement_rate=fact.agreement_rate,
        agreement_floor=fact.agreement_floor,
        distrusted=distrusted,
        n_scored_traces=fact.n_scored_traces,
        coverage=round(coverage, 6),
        reason=reason,
    )


def compute_readiness(
    cohort: Cohort,
    goal: GoalView,
    facts: ReadinessFacts,
    *,
    thresholds: ReadinessThresholds | None = None,
    generated_at: str | None = None,
) -> ReadinessStatus:
    """Compute fail-closed readiness for ``cohort`` against ``goal``.

    Evaluates the trace gates (every goal) and, for a goal whose
    :attr:`~ail.readiness.goal.GoalView.requires_quality` is set, the quality
    gates (frozen suite, human labels, a trusted judge, scored-coverage). Derives
    the tier from which gates pass and returns a
    :class:`~ail.readiness.contract.ReadinessStatus` whose ``reasons`` enumerate
    every unmet gate. When any gate is unmet the tier is a not-ready tier — the
    function never returns a ready tier on missing data (the refusal is the
    feature).
    """
    th = thresholds or ReadinessThresholds()
    eval_health = compute_eval_health(cohort, facts, thresholds=th, generated_at=generated_at)

    gates: list[Gate] = [
        _trace_baseline_gate(facts.trace_count, th),
        _trace_prove_gate(facts.trace_count, th),
    ]
    if goal.requires_quality:
        gates.append(_frozen_suite_gate(facts.frozen_suite_present))
        gates.append(_human_labels_gate(facts.label_count, th))
        gates.append(_judge_trusted_gate(goal, facts))
        gates.append(_scored_coverage_gate(eval_health, th))

    tier = _derive_tier(goal, facts, gates, th)
    reasons = [g.reason for g in gates if not g.passed]

    notes: list[str] = []
    if tier == ReadinessTier.READY_TO_PROVE:
        notes.append(
            "data sufficient to prove an improvement; the comparison harness still "
            "requires candidate runs on the frozen Task Suite before promoting"
        )

    return ReadinessStatus(
        cohort_name=cohort.name,
        objective_metric=goal.objective_metric,
        requires_quality=goal.requires_quality,
        guardrail_names=list(goal.guardrail_names),
        trace_count=facts.trace_count,
        tier=tier,
        gates=gates,
        reasons=reasons,
        eval_health=eval_health,
        generated_at=generated_at,
        notes=notes,
    )


# -- gates -----------------------------------------------------------------


def _trace_baseline_gate(trace_count: int, th: ReadinessThresholds) -> Gate:
    need = th.baseline_min_traces
    passed = trace_count >= need
    reason = (
        f"have {trace_count} trace(s) (>= {need} to baseline)"
        if passed
        else f"need {need - trace_count} more trace(s) to baseline "
        f"(have {trace_count}, need {need})"
    )
    return Gate(name=GateName.TRACE_BASELINE, passed=passed, reason=reason)


def _trace_prove_gate(trace_count: int, th: ReadinessThresholds) -> Gate:
    need = th.prove_min_traces
    passed = trace_count >= need
    reason = (
        f"have {trace_count} trace(s) (>= {need} for statistical power)"
        if passed
        else f"need {need - trace_count} more trace(s) for statistical power to prove "
        f"improvement (have {trace_count}, need {need})"
    )
    return Gate(name=GateName.TRACE_PROVE, passed=passed, reason=reason)


def _frozen_suite_gate(present: bool) -> Gate:
    reason = "frozen Task Suite present" if present else "no frozen Task Suite to compare against"
    return Gate(name=GateName.FROZEN_SUITE, passed=present, reason=reason)


def _human_labels_gate(label_count: int, th: ReadinessThresholds) -> Gate:
    need = th.quality_min_labels
    passed = label_count >= need
    reason = (
        f"have {label_count} human label(s) (>= {need} to calibrate the judge)"
        if passed
        else f"need {need - label_count} more human label(s) to calibrate the judge "
        f"(have {label_count}, need {need})"
    )
    return Gate(name=GateName.HUMAN_LABELS, passed=passed, reason=reason)


def _judge_trusted_gate(goal: GoalView, facts: ReadinessFacts) -> Gate:
    """A quality goal needs at least one relevant, trusted judge.

    Relevant judges are those whose name is in the goal's guardrails (if any
    match); otherwise every judge in the facts. The gate passes only when that set
    is non-empty **and** none of its judges are distrusted — an unmeasured judge
    is distrusted by default, so a goal with no judges (or only unmeasured ones)
    fails closed.
    """
    guardrails = set(goal.guardrail_names)
    relevant = [jf for jf in facts.judges if jf.judge_name in guardrails] or list(facts.judges)
    if not relevant:
        return Gate(
            name=GateName.JUDGE_TRUSTED,
            passed=False,
            reason="no calibrated judge: judges are distrusted by default until measured",
        )
    distrusted = [jf.judge_name for jf in relevant if jf.is_distrusted]
    if distrusted:
        return Gate(
            name=GateName.JUDGE_TRUSTED,
            passed=False,
            reason=f"judge(s) {', '.join(distrusted)} distrusted "
            "(unmeasured against humans or below agreement floor)",
        )
    trusted = [jf.judge_name for jf in relevant]
    return Gate(
        name=GateName.JUDGE_TRUSTED,
        passed=True,
        reason=f"judge(s) {', '.join(trusted)} measured and trusted",
    )


def _scored_coverage_gate(eval_health: EvalHealth, th: ReadinessThresholds) -> Gate:
    floor = th.scored_coverage_floor
    coverage = eval_health.scored_coverage
    passed = coverage >= floor
    reason = (
        f"scored-coverage {coverage:.0%} (>= floor {floor:.0%})"
        if passed
        else f"scored-coverage {coverage:.0%} below floor {floor:.0%}"
    )
    return Gate(name=GateName.SCORED_COVERAGE, passed=passed, reason=reason)


# -- tier derivation -------------------------------------------------------


def _derive_tier(
    goal: GoalView,
    facts: ReadinessFacts,
    gates: list[Gate],
    th: ReadinessThresholds,
) -> ReadinessTier:
    """Map the gate results onto a tier, fail-closed.

    Below the baseline trace floor (0 traces is the limiting case) the cohort is
    always *collecting*. Above it, a token/cost goal climbs to *ready-to-prove* on
    trace count alone; a quality goal must additionally pass every quality gate to
    reach *ready-for-quality*, and then clear the prove trace floor to reach
    *ready-to-prove*.
    """
    if facts.trace_count < th.baseline_min_traces:
        return ReadinessTier.COLLECTING

    has_power = facts.trace_count >= th.prove_min_traces

    if not goal.requires_quality:
        return ReadinessTier.READY_TO_PROVE if has_power else ReadinessTier.BASELINE_ONLY

    quality_gate_names = {
        GateName.FROZEN_SUITE,
        GateName.HUMAN_LABELS,
        GateName.JUDGE_TRUSTED,
        GateName.SCORED_COVERAGE,
    }
    quality_ready = all(g.passed for g in gates if g.name in quality_gate_names)
    if not quality_ready:
        return ReadinessTier.BASELINE_ONLY
    return ReadinessTier.READY_TO_PROVE if has_power else ReadinessTier.READY_FOR_QUALITY
