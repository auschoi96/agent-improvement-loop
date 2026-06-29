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

    Evaluates the universal gates (the trace counts and the **frozen Task Suite** —
    no improvement, deterministic or judged, can be proven without a paired
    baseline-vs-candidate benchmark) and, for a goal whose
    :attr:`~ail.readiness.goal.GoalView.requires_quality` is set, the quality gates
    (human labels, a trusted judge, scored-coverage). Derives the tier from which
    gates pass and returns a :class:`~ail.readiness.contract.ReadinessStatus` whose
    ``reasons`` enumerate every unmet gate. When any gate is unmet the tier is a
    not-ready tier — the function never returns a ready tier on missing data (the
    refusal is the feature).
    """
    th = thresholds or ReadinessThresholds()
    eval_health = compute_eval_health(cohort, facts, thresholds=th, generated_at=generated_at)

    # Universal gates apply to every goal: a deterministic token/cost claim is as
    # dependent on a frozen paired benchmark as a judged one (docs/ARCHITECTURE.md
    # §4 — comparison runs on the FROZEN Task Suite only).
    gates: list[Gate] = [
        _trace_baseline_gate(facts.trace_count, th),
        _trace_prove_gate(facts.trace_count, th),
        _frozen_suite_gate(facts.frozen_suite_present),
    ]
    if goal.requires_quality:
        gates.append(_human_labels_gate(facts.label_count, th))
        gates.append(_judge_trusted_gate(goal, facts))
        gates.append(_scored_coverage_gate(eval_health, th))

    tier = _derive_tier(goal, gates)
    reasons = [g.reason for g in gates if not g.passed]

    notes: list[str] = []
    if tier == ReadinessTier.READY_TO_PROVE:
        notes.append(
            "data sufficient to prove an improvement; the comparison harness still "
            "must execute candidate runs against the frozen Task Suite before promoting"
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


def _required_judge_names(goal: GoalView) -> set[str]:
    """The exact set of judges a quality goal depends on — its named guardrails.

    Fail-closed by construction: the gate is checked against *exactly* these
    judges, never any unrelated judge that happens to be present. In this
    architecture the objective_metric is the deterministic L0 reduction and the
    guardrails are the judges (``docs/ARCHITECTURE.md`` §3); a deployer whose
    objective is itself a judged metric must list that judge among
    ``guardrail_names`` so it is required here. An empty set means the quality
    goal named no judge at all — which the gate refuses to certify (a quality
    claim with no calibrated judge is unfounded).
    """
    return set(goal.guardrail_names)


def _judge_trusted_gate(goal: GoalView, facts: ReadinessFacts) -> Gate:
    """A quality goal's required guardrail judges must each be present and trusted.

    Evaluated against *exactly* the goal's required judges (see
    :func:`_required_judge_names`) — never substituting an unrelated judge. Each
    required judge must be present in ``facts.judges`` **and** trusted (measured,
    not distrusted, not below floor). A required judge missing from the facts, or
    unmeasured/below-floor, fails the gate with a clear reason; a quality goal that
    names no judge fails closed. This is the wall: a trusted ``latency`` judge can
    never stand in for an unmeasured ``security`` judge the goal actually needs.
    """
    required = _required_judge_names(goal)
    if not required:
        return Gate(
            name=GateName.JUDGE_TRUSTED,
            passed=False,
            reason="quality goal names no guardrail judge; a quality claim needs a "
            "calibrated judge — cannot certify (fail closed)",
        )
    by_name = {jf.judge_name: jf for jf in facts.judges}
    missing = sorted(name for name in required if name not in by_name)
    distrusted = sorted(
        name for name in required if name in by_name and by_name[name].is_distrusted
    )
    if missing or distrusted:
        problems: list[str] = []
        if missing:
            problems.append(f"no measurement for required judge(s) {', '.join(missing)}")
        if distrusted:
            problems.append(
                f"required judge(s) {', '.join(distrusted)} distrusted "
                "(unmeasured against humans or below agreement floor)"
            )
        return Gate(name=GateName.JUDGE_TRUSTED, passed=False, reason="; ".join(problems))
    return Gate(
        name=GateName.JUDGE_TRUSTED,
        passed=True,
        reason=f"required judge(s) {', '.join(sorted(required))} measured and trusted",
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


# Gate names whose passing defines each tier, keyed off the gates actually built.
# A frozen Task Suite is universal — proving any improvement needs a paired
# benchmark — so it sits in the prove set for every goal, not just quality ones.
_PROVE_GATES = (GateName.TRACE_PROVE, GateName.FROZEN_SUITE)
_QUALITY_GATES = (
    GateName.FROZEN_SUITE,
    GateName.HUMAN_LABELS,
    GateName.JUDGE_TRUSTED,
    GateName.SCORED_COVERAGE,
)


def _derive_tier(goal: GoalView, gates: list[Gate]) -> ReadinessTier:
    """Map the gate results onto a tier, fail-closed.

    Below the baseline trace floor (0 traces is the limiting case) the cohort is
    *collecting*. To *prove* an improvement (any goal) the trace-power and frozen-
    suite gates must pass; a quality goal must additionally clear every quality
    gate to reach *ready-for-quality*, then the prove floor for *ready-to-prove*.

    Reads gate results by name through :func:`_gate_passed`, which raises if a
    required gate was never evaluated — so a refactor that drops a gate fails loud
    rather than silently reading as green.
    """
    by_name = {g.name: g for g in gates}

    if not _gate_passed(by_name, GateName.TRACE_BASELINE):
        return ReadinessTier.COLLECTING

    quality_gates = _QUALITY_GATES if goal.requires_quality else ()
    prove_gates = _PROVE_GATES + quality_gates
    if all(_gate_passed(by_name, name) for name in prove_gates):
        return ReadinessTier.READY_TO_PROVE

    if quality_gates and all(_gate_passed(by_name, name) for name in quality_gates):
        return ReadinessTier.READY_FOR_QUALITY

    return ReadinessTier.BASELINE_ONLY


def _gate_passed(by_name: dict[GateName, Gate], name: GateName) -> bool:
    """Whether the gate named ``name`` passed; raise loud if it was not evaluated.

    Tier derivation must never reference a gate that ``compute_readiness`` did not
    build — that would let a dropped gate read as a silent pass. A missing gate is
    a programming error in this module, so it raises rather than defaulting.
    """
    gate = by_name.get(name)
    if gate is None:
        raise RuntimeError(
            f"required readiness gate {name.value!r} was not evaluated — "
            "bug in compute_readiness (a gate was dropped from the built set)"
        )
    return gate.passed
