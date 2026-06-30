"""``ail-readiness`` — a one-command onboarding **preflight**.

A first-time user runs ``ail-readiness <experiment_id>`` and sees, in one table,
exactly how far they are from each data gate the loop enforces — *before* they
invest in labeling, alignment, or a comparison run. It answers the only question
that matters on day one: "what is unlocked now, and what do I still need to
connect?"

It is a thin **driver** over the already-built readiness module — it reuses, and
never reimplements, the gate logic:

* The facts come from the live experiment via the existing ingest seam
  (:class:`ail.ingest.mlflow_source.MLflowTraceSource`, i.e.
  ``mlflow.search_traces``) plus the human/judge **assessments** carried on each
  trace (the same ``trace.info.assessments`` shape :mod:`ail.judges.labeling`
  reads) and the frozen-suite loader (:func:`ail.task_suite.load_task_suite`).
* The verdicts come from :func:`ail.readiness.compute_readiness` with the default
  :class:`~ail.readiness.ReadinessThresholds`. Every PASS/FAIL marker and every
  "need N more …" message printed here is the readiness module's own output — the
  CLI only lays it out.

**Fail-closed and honest.** A *not-ready* result is a normal, successful run
(exit ``0``): the refusal is the product. The CLI exits **non-zero** only when it
cannot gather the facts at all (auth / trace-store access), and in that case it
prints an actionable error naming the profile and warehouse — it never prints a
fabricated "ready" on error.

Two readiness lenses are shown because the framework unlocks them separately
(``docs/CONNECT_YOUR_AGENT.md`` §2): the **deterministic** token/cost prove path
(traces + a frozen suite; no labels) and the **MemAlign-judge** quality path
(human labels + a trusted judge + scored-coverage). The user's ``--goal`` drives
the headline tier; the token-efficiency judge drives the quality lens.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ail.cohorts import Cohort, TagFilter
from ail.publish import REFERENCE_EXPERIMENT
from ail.readiness import (
    Gate,
    GateName,
    GoalView,
    JudgeFact,
    ReadinessFacts,
    ReadinessStatus,
    ReadinessThresholds,
    compute_readiness,
)

if TYPE_CHECKING:
    from ail.ingest.base import NormalizedTrace

#: The MemAlign-aligned token-efficiency judge name (a registered scorer, see
#: :data:`ail.judges.scorers.DEFAULT_SCORERS`). It is the judge the quality lens
#: probes: the "MemAlign judge" the readiness ladder unlocks at >=20 labels.
TOKEN_EFFICIENCY_JUDGE = "token_efficiency"

#: Default deterministic objective when the user states no ``--goal``: prove a
#: token reduction. Matches the Phase-2 headline metric (``total_tokens``).
DEFAULT_OBJECTIVE_METRIC = "total_tokens"

#: Human label that goes with :data:`GateName` for the printed table.
_GATE_LABELS: dict[GateName, str] = {
    GateName.TRACE_BASELINE: "baseline / diagnosis (RLM)",
    GateName.TRACE_PROVE: "prove an improvement",
    GateName.FROZEN_SUITE: "frozen Task Suite",
    GateName.HUMAN_LABELS: "MemAlign labels",
    GateName.JUDGE_TRUSTED: "judge trusted",
    GateName.SCORED_COVERAGE: "scored-coverage",
}

#: Stable display order for the gate table.
_GATE_ORDER: tuple[GateName, ...] = (
    GateName.TRACE_BASELINE,
    GateName.TRACE_PROVE,
    GateName.FROZEN_SUITE,
    GateName.HUMAN_LABELS,
    GateName.JUDGE_TRUSTED,
    GateName.SCORED_COVERAGE,
)


class PreflightAccessError(RuntimeError):
    """The preflight could not read the experiment's traces (auth / permission).

    Raised by the live :func:`gather_facts` path when ``mlflow.search_traces`` (or
    workspace auth) fails. It carries an **actionable** message — which profile and
    warehouse were used, and that the UC trace store needs ``CAN_USE`` — so the
    user fixes the access rather than seeing a fabricated "ready". :func:`main`
    turns it into a non-zero exit with no ready line.
    """


@dataclass(frozen=True)
class PreflightGoal:
    """A minimal :class:`~ail.readiness.GoalView` for the preflight.

    Structural stand-in for the goals lane's ``CompiledGoal`` (it carries exactly
    the three members the Protocol requires), so the preflight can compute
    readiness without importing :mod:`ail.goals` on the default, offline path. A
    ``--goal`` string still compiles through the real goals lane.
    """

    objective_metric: str
    requires_quality: bool = False
    guardrail_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreflightResult:
    """Everything the renderer needs — the readiness module's own output.

    ``status`` is the readiness for the **user's** goal (the headline tier, e.g.
    the deterministic token-win prove path). ``quality_status`` is the readiness
    for the token-efficiency **MemAlign-judge** lens; it equals ``status`` when the
    user's goal is itself a quality goal. ``gates`` is the full six-gate set drawn
    from ``quality_status`` (a quality goal evaluates every gate), so the table
    always shows all four headline gates plus judge-trust and frozen-suite.
    """

    status: ReadinessStatus
    quality_status: ReadinessStatus
    facts: ReadinessFacts
    thresholds: ReadinessThresholds
    experiment_id: str
    cohort_label: str
    objective_metric: str

    @property
    def gates(self) -> list[Gate]:
        return list(self.quality_status.gates)


# A facts source maps (experiment_id, cohort) -> the measured ReadinessFacts.
# Injected in tests with a fake that returns canned facts (or raises) so no test
# ever touches MLflow — the same seam ``register_scorers`` / ``publish_job`` use.
FactsSource = Callable[[str, Cohort], ReadinessFacts]


# ---------------------------------------------------------------------------
# Fact gathering (the live path; injected away in tests)
# ---------------------------------------------------------------------------


def gather_facts(
    experiment_id: str,
    cohort: Cohort,
    *,
    profile: str | None = None,
    warehouse_id: str | None = None,
    source: Any | None = None,
    suite_present: bool | None = None,
) -> ReadinessFacts:
    """Measure the cohort's :class:`~ail.readiness.ReadinessFacts` from the live experiment.

    Reuses the ingest seam: traces are counted via
    :meth:`MLflowTraceSource.fetch_cohort_traces` (``mlflow.search_traces``), human
    labels and judge verdicts are read from each trace's ``info.assessments`` (the
    shape :mod:`ail.judges.labeling` uses), and frozen-suite presence comes from
    :func:`ail.task_suite.load_task_suite`. Judges discovered on the traces are
    recorded **unmeasured** (``agreement_rate=None``) — the preflight does not run
    a Human-Anchor agreement pass, so a judge is distrusted by default here; trust
    is certified by the alignment cadence (Stage 3), not by a preflight.

    Args:
        experiment_id: MLflow experiment to read.
        cohort: The cohort whose traces are counted (an all-matching cohort means
            the whole experiment).
        profile: Databricks CLI profile selecting the workspace.
        warehouse_id: SQL warehouse the UC trace store is backed by; recorded so
            an access error can name it. Exported as ``AIL_WAREHOUSE_ID`` for any
            downstream that reads it.
        source: Injectable trace source exposing ``fetch_cohort_traces`` (defaults
            to :class:`MLflowTraceSource`); the seam tests stub out.
        suite_present: Override frozen-suite detection (mainly for tests); when
            ``None`` the on-disk loader is consulted.

    Raises:
        PreflightAccessError: trace-store access / auth failed — never a fake ready.
    """
    if warehouse_id:
        os.environ.setdefault("AIL_WAREHOUSE_ID", warehouse_id)

    if source is None:
        from ail.ingest.mlflow_source import MLflowTraceSource

        source = MLflowTraceSource(profile=profile)

    try:
        traces = source.fetch_cohort_traces(cohort, experiment_id=experiment_id)
    except Exception as exc:  # noqa: BLE001 - any read failure becomes an actionable error
        raise PreflightAccessError(_access_hint(experiment_id, profile, warehouse_id, exc)) from exc

    label_count, n_scored, judges = _count_assessments(traces)
    present = _frozen_suite_present() if suite_present is None else suite_present

    return ReadinessFacts(
        trace_count=len(traces),
        label_count=label_count,
        frozen_suite_present=present,
        n_scored_traces=n_scored,
        judges=judges,
    )


def _access_hint(
    experiment_id: str, profile: str | None, warehouse_id: str | None, exc: Exception
) -> str:
    """Build the actionable message for a trace-store access failure."""
    prof = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE") or "(default/ambient)"
    wh = warehouse_id or os.environ.get("AIL_WAREHOUSE_ID") or "(none supplied)"
    return (
        f"could not read traces for experiment {experiment_id!r} "
        f"(profile={prof}, warehouse_id={wh}): {type(exc).__name__}: {exc}. "
        "Check the Databricks profile points at the right workspace and that the "
        "identity has CAN_USE on the SQL warehouse backing the UC trace store "
        "(and CAN_VIEW on the experiment). No readiness verdict was produced."
    )


def _count_assessments(
    traces: Sequence[NormalizedTrace],
) -> tuple[int, int, tuple[JudgeFact, ...]]:
    """Count human labels, scored traces, and per-judge coverage from assessments.

    A **human label** is a ``HUMAN``-sourced assessment (the calibration pool). A
    trace is **scored** when it carries at least one non-human (judge) verdict. Per
    judge name we record how many traces it scored; agreement is left unmeasured
    (distrusted by default — the fail-closed rule of ``READINESS_AND_TRUST.md`` §3).
    """
    label_count = 0
    scored_traces = 0
    per_judge: dict[str, int] = {}
    for trace in traces:
        raw = getattr(trace, "raw", None)
        info = getattr(raw, "info", None)
        assessments = list(getattr(info, "assessments", None) or [])
        trace_has_judge = False
        for assessment in assessments:
            name = str(getattr(assessment, "name", "") or "")
            if _is_human(assessment):
                label_count += 1
            elif name:
                trace_has_judge = True
                per_judge[name] = per_judge.get(name, 0) + 1
        if trace_has_judge:
            scored_traces += 1

    judges = tuple(
        JudgeFact(judge_name=name, n_scored_traces=count) for name, count in per_judge.items()
    )
    return label_count, scored_traces, judges


def _is_human(assessment: Any) -> bool:
    """Whether an assessment is HUMAN-sourced (mirrors ``ail.judges.labeling._is_human``)."""
    source = getattr(assessment, "source", None)
    return str(getattr(source, "source_type", "")) == "HUMAN"


def _frozen_suite_present() -> bool:
    """Whether a sealed, frozen Task Suite is loadable (fail-closed: absent on any error)."""
    try:
        from ail.task_suite import DEFAULT_ARTIFACT_VERSION, load_task_suite

        load_task_suite(DEFAULT_ARTIFACT_VERSION)
        return True
    except Exception:  # noqa: BLE001 - missing/unsealed/unfetchable suite => not present
        return False


# ---------------------------------------------------------------------------
# Readiness computation (pure reuse of ail.readiness)
# ---------------------------------------------------------------------------


def build_cohort(cohort_tag: str | None, experiment_id: str) -> Cohort:
    """The cohort to gate on: the ``ail.agent`` tag value, else the whole experiment."""
    if cohort_tag:
        return Cohort.by_agent(cohort_tag)
    # An empty TagFilter matches every trace — "the whole experiment".
    return Cohort(name=f"experiment:{experiment_id}", tag_filter=TagFilter())


def build_goal(objective_metric: str) -> PreflightGoal:
    """The user's headline goal: a deterministic prove path for ``objective_metric``."""
    return PreflightGoal(objective_metric=objective_metric, requires_quality=False)


def _quality_goal() -> PreflightGoal:
    """The MemAlign-judge lens: a quality goal guarded by the token-efficiency judge."""
    return PreflightGoal(
        objective_metric=TOKEN_EFFICIENCY_JUDGE,
        requires_quality=True,
        guardrail_names=(TOKEN_EFFICIENCY_JUDGE,),
    )


def evaluate(
    experiment_id: str,
    *,
    cohort_tag: str | None = None,
    goal: GoalView | None = None,
    facts_source: FactsSource,
    thresholds: ReadinessThresholds | None = None,
    generated_at: str | None = None,
) -> PreflightResult:
    """Gather facts and compute readiness — the orchestration the renderer consumes.

    Computes the headline goal's readiness and, separately, the token-efficiency
    quality lens (unless the headline goal is itself a quality goal, in which case
    the two coincide). Both are the readiness module's own output; nothing here
    re-derives a tier or a gate.
    """
    th = thresholds or ReadinessThresholds()
    cohort = build_cohort(cohort_tag, experiment_id)
    headline_goal = goal or build_goal(DEFAULT_OBJECTIVE_METRIC)

    facts = facts_source(experiment_id, cohort)

    status = compute_readiness(
        cohort, headline_goal, facts, thresholds=th, generated_at=generated_at
    )
    if headline_goal.requires_quality:
        quality_status = status
    else:
        quality_status = compute_readiness(
            cohort, _quality_goal(), facts, thresholds=th, generated_at=generated_at
        )

    return PreflightResult(
        status=status,
        quality_status=quality_status,
        facts=facts,
        thresholds=th,
        experiment_id=experiment_id,
        cohort_label=cohort.name,
        objective_metric=headline_goal.objective_metric,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_READY = "READY"
_NOT_READY = "NOT READY"


def render(result: PreflightResult) -> str:
    """Render the human-readable preflight: header, gate table, tiers, and summary."""
    facts = result.facts
    th = result.thresholds
    eh = result.quality_status.eval_health

    lines: list[str] = []
    lines.append(f"Readiness preflight — experiment {result.experiment_id!r}")
    lines.append(f"  cohort: {result.cohort_label}    objective: {result.objective_metric}")
    lines.append(
        f"  thresholds: baseline>={th.baseline_min_traces} traces, "
        f"prove>={th.prove_min_traces} traces, labels>={th.quality_min_labels}, "
        f"coverage>={th.scored_coverage_floor:.0%}"
    )
    lines.append("")

    header = f"  {'GATE':<28}{'HAVE':<12}{'NEED':<12}{'GAP':<8}STATUS"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    by_name = {g.name: g for g in result.gates}
    for name in _GATE_ORDER:
        gate = by_name.get(name)
        if gate is None:
            continue
        have, need, gap = _cells(name, gate, facts, th, eh)
        marker = _READY if gate.passed else _NOT_READY
        lines.append(f"  {_GATE_LABELS[name]:<28}{have:<12}{need:<12}{gap:<8}{marker}")
    lines.append("")

    lines.append(
        f"  Prove a '{result.objective_metric}' win (deterministic): {result.status.tier.value}"
    )
    lines.append(f"  MemAlign judge (trustworthy quality): {result.quality_status.tier.value}")
    lines.append("")

    details = _unmet_reasons(result.gates)
    if details:
        lines.append("  What you still need (from the readiness module):")
        lines.extend(f"    - {reason}" for reason in details)
        lines.append("")

    lines.append(f"  Unlocked now: {_summary(result)}")
    return "\n".join(lines)


def _cells(
    name: GateName,
    gate: Gate,
    facts: ReadinessFacts,
    th: ReadinessThresholds,
    eval_health: Any,
) -> tuple[str, str, str]:
    """The (have, need, gap) display cells for one gate (verdict stays ``gate.passed``)."""
    if name is GateName.TRACE_BASELINE:
        return _numeric(facts.trace_count, th.baseline_min_traces)
    if name is GateName.TRACE_PROVE:
        return _numeric(facts.trace_count, th.prove_min_traces)
    if name is GateName.HUMAN_LABELS:
        return _numeric(facts.label_count, th.quality_min_labels)
    if name is GateName.SCORED_COVERAGE:
        coverage = eval_health.scored_coverage
        floor = th.scored_coverage_floor
        gap = "0%" if gate.passed else f"{floor - coverage:.0%}"
        return f"{coverage:.0%}", f"{floor:.0%}", gap
    # Boolean gates (frozen suite, judge trusted): the verdict is the value.
    return ("yes" if gate.passed else "no", "yes", "—")


def _numeric(have: int, need: int) -> tuple[str, str, str]:
    return str(have), str(need), str(max(0, need - have))


def _unmet_reasons(gates: Sequence[Gate]) -> list[str]:
    """The readiness module's own reason strings for the unmet gates, in table order."""
    by_name = {g.name: g for g in gates}
    out: list[str] = []
    for name in _GATE_ORDER:
        gate = by_name.get(name)
        if gate is not None and not gate.passed:
            out.append(gate.reason)
    return out


def _summary(result: PreflightResult) -> str:
    """One line of per-capability status, derived from the individual gates.

    e.g. "RLM+diagnosis: READY; MemAlign judge: need 12 more labels; prove a token
    win: need 35 more traces". Each clause is gate-driven, so it can never report a
    capability ready that the readiness module did not pass.
    """
    by_name = {g.name: g for g in result.gates}
    facts = result.facts
    th = result.thresholds
    parts: list[str] = []

    baseline = by_name.get(GateName.TRACE_BASELINE)
    if baseline is not None:
        parts.append(
            "RLM+diagnosis: READY"
            if baseline.passed
            else f"RLM+diagnosis: need {max(0, th.baseline_min_traces - facts.trace_count)} "
            "more traces"
        )

    labels = by_name.get(GateName.HUMAN_LABELS)
    judge = by_name.get(GateName.JUDGE_TRUSTED)
    if labels is not None:
        if not labels.passed:
            parts.append(
                f"MemAlign judge: need {max(0, th.quality_min_labels - facts.label_count)} "
                "more labels"
            )
        elif judge is not None and not judge.passed:
            parts.append("MemAlign judge: labels in — align + measure the judge (Stage 3)")
        else:
            parts.append("MemAlign judge: READY")

    prove = by_name.get(GateName.TRACE_PROVE)
    frozen = by_name.get(GateName.FROZEN_SUITE)
    if prove is not None and frozen is not None:
        if prove.passed and frozen.passed:
            parts.append(f"prove a {result.objective_metric} win: READY")
        elif not prove.passed:
            parts.append(
                f"prove a {result.objective_metric} win: need "
                f"{max(0, th.prove_min_traces - facts.trace_count)} more traces"
            )
        else:
            parts.append(f"prove a {result.objective_metric} win: need a frozen Task Suite")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_facts_source(profile: str | None, warehouse_id: str | None) -> FactsSource:
    """The production facts source: the live :func:`gather_facts` over MLflow."""

    def _source(experiment_id: str, cohort: Cohort) -> ReadinessFacts:
        return gather_facts(experiment_id, cohort, profile=profile, warehouse_id=warehouse_id)

    return _source


def _compile_goal(goal_text: str, cohort: Cohort) -> GoalView:
    """Compile a natural-language ``--goal`` through the real goals lane (reuse).

    Lazy-imported so the default, offline path never pulls the goals lane (or its
    LLM proposer). Compilation failures propagate as a clear error to :func:`main`.
    """
    from ail.goals.compiler import compile_goal

    return compile_goal(goal_text, cohort)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-readiness",
        description=(
            "Onboarding preflight: show, per data gate, how far an experiment is from "
            "unlocking baseline/RLM, a MemAlign judge, a provable improvement, and "
            "scored-coverage. Reuses ail.readiness; never prints a fabricated 'ready'."
        ),
    )
    parser.add_argument(
        "experiment_id",
        nargs="?",
        default=REFERENCE_EXPERIMENT,
        help="MLflow experiment id to inspect (defaults to the reference experiment).",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        help="Databricks CLI profile selecting the workspace.",
    )
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID"),
        help="SQL warehouse backing the UC trace store (named in access errors).",
    )
    parser.add_argument(
        "--cohort-tag",
        default=None,
        help="Filter to one agent by its ail.agent tag value (default: whole experiment).",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Optional natural-language goal; compiled via ail.goals (may call an LLM). "
        "Omit for the default deterministic token-win readiness.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    goal: GoalView | None = None
    if args.goal:
        cohort = build_cohort(args.cohort_tag, args.experiment_id)
        try:
            goal = _compile_goal(args.goal, cohort)
        except Exception as exc:  # noqa: BLE001 - surface a clear, non-zero compile failure
            print(f"[ail-readiness] could not compile --goal: {exc}", file=sys.stderr)
            return 2

    try:
        result = evaluate(
            args.experiment_id,
            cohort_tag=args.cohort_tag,
            goal=goal,
            facts_source=_default_facts_source(args.profile, args.warehouse_id),
        )
    except PreflightAccessError as exc:
        print(f"[ail-readiness] {exc}", file=sys.stderr)
        return 1

    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
