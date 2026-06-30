"""Goal-parameterized decision rules: a detected feedback signal → a candidate action.

These are the pure functions behind step 2 of the loop (``docs/LOOP_CONTROLLER.md``
§"Decision rules"): given a typed feedback signal and the **compiled goal**, decide
*which* action kind addresses it — or decide nothing. They map the four
illustrative rules in the design:

* an **RLM/HALO recommended asset** of an additive type, recurring across ≥ N
  traces, while the goal's objective is still unmet → a **metric view**
  (:func:`decide_rlm_asset`);
* a **redundant-read / boilerplate pattern** that dominates the L0 waste diagnosis
  → a **skill update** (:func:`decide_redundant_read`);
* a **judge dimension below the goal's threshold**, *and* that judge is trusted →
  **GEPA** prompt evolution (:func:`decide_judge_dimension`);
* a registered version's **post-apply regression** → a **revert**
  (:func:`decide_post_apply_regression`).

**No magic thresholds.** Every bar comes from the typed goal (its target and its
guardrail thresholds) or from :class:`DecisionThresholds` — a visible, adjustable
frozen dataclass, mirroring :class:`ail.readiness.ReadinessThresholds` — never a
constant buried in a function body. The rules are *pure*: they read their inputs
and return a :class:`Decision` (or ``None``); they run no agent, touch no MLflow,
and emit nothing themselves — the controller sequences them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ail.goals.compiler import CompiledGoal
from ail.loop.proposals import (
    ActionKind,
    RiskClass,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
)

__all__ = [
    "DecisionThresholds",
    "RlmAssetSignal",
    "RedundantReadSignal",
    "JudgeDimensionSignal",
    "PostApplyRegressionSignal",
    "FeedbackBundle",
    "Decision",
    "objective_target_met",
    "decide_rlm_asset",
    "decide_redundant_read",
    "decide_judge_dimension",
    "decide_post_apply_regression",
    "decide",
]

#: Additive-asset RLM recommendation types that the asset generator
#: (:mod:`ail.optimize.assets`) can turn into a deployable metric view. A
#: recommendation of one of these (and recurring enough) maps to a metric_view
#: action; a ``skill`` / ``prompt_change`` recommendation is handled by the
#: redundant-read / judge-dimension rules, not here.
_ADDITIVE_ASSET_TYPES = frozenset({"metric_view", "tool", "semantic_layer", "data_pipeline"})


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Decision-rule bars that have no home on the goal or readiness — adjustable.

    Mirrors :class:`ail.readiness.ReadinessThresholds`: heuristics, not laws, kept
    visible and tunable rather than buried as constants. Only the genuinely
    decision-specific knobs live here; the objective target and the judge-dimension
    threshold come from the compiled goal itself.

    Args:
        min_asset_recurrence_traces: The ``N`` in "an RLM asset recurring across ≥
            N traces". A recommendation seen on fewer distinct traces is not yet a
            recurring pattern worth proposing an asset for.
        min_redundant_occurrences: How many times a redundant-read pattern must
            repeat before it counts as *dominant* enough to propose a skill update.
    """

    min_asset_recurrence_traces: int = 3
    min_redundant_occurrences: int = 3


@dataclass(frozen=True, slots=True)
class RlmAssetSignal:
    """An RLM/HALO recommended asset, recurrence-ranked across a cohort.

    Built from a :class:`ail.l3.contract.RankedAsset` (the cohort roll-up of L3
    review): ``asset_type`` / ``title`` / ``rank`` / ``n_traces`` come straight off
    it, and ``trace_ids`` are the distinct subject traces that recommended it.
    """

    asset_type: str
    title: str
    n_traces: int
    rank: int = 0
    trace_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RedundantReadSignal:
    """A redundant-read / boilerplate pattern from the L0 waste diagnosis.

    Built from a :class:`ail.l3.contract.RedundancyFinding` (or the L0 diagnosis
    rows): a repeated read/shell target the agent hit ``occurrences`` times.
    ``dominant`` marks that this pattern leads the waste diagnosis — only a
    dominant, recurring pattern is worth a skill update.
    """

    tool: str | None
    repeated_target: str | None
    occurrences: int
    dominant: bool = False
    estimated_wasted_tokens: int | None = None
    trace_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class JudgeDimensionSignal:
    """A judged quality dimension scored below par, with its judge's trust verdict.

    ``judge_name`` is the judge whose dimension fired — it must also be trusted
    (``trusted``: judge-vs-human agreement at/above floor) for the rule to act, and
    it is the judge whose trust the controller's gate later requires. ``dimension``
    is the human-facing dimension name (e.g. ``"modularity"``); ``score`` is its
    measured value.
    """

    judge_name: str
    dimension: str
    score: float
    trusted: bool
    trace_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PostApplyRegressionSignal:
    """A registered version whose real post-apply impact regressed vs its predecessor.

    Carries the ``risk_class`` of the change being reverted so the revert proposal
    inherits the right blast-radius label (reverting an additive asset is itself
    additive; reverting an agent change is an agent change).
    """

    agent_version: str
    predecessor_version: str
    objective_metric: str
    regressed: bool
    risk_class: RiskClass = RiskClass.AGENT_CHANGE
    trace_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeedbackBundle:
    """The typed feedback one cycle gathers about an agent/cohort.

    Pure data — the (injectable) feedback source produces it from the L3 cohort
    review, the L0 waste diagnosis, the judges, and the post-apply monitoring.
    ``objective_metric_value`` is the cohort's current value of the goal's
    objective; ``objective_baseline_value`` is the baseline a *relative* goal
    target is measured against (needed only to evaluate a relative target).
    """

    objective_metric_value: float | None = None
    objective_baseline_value: float | None = None
    rlm_assets: tuple[RlmAssetSignal, ...] = ()
    redundant_reads: tuple[RedundantReadSignal, ...] = ()
    judge_dimensions: tuple[JudgeDimensionSignal, ...] = ()
    post_apply_regressions: tuple[PostApplyRegressionSignal, ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    """A fired rule's verdict: which action to build, its risk class, and the why.

    The :attr:`trigger` is the evidence payload that becomes the proposal's "why";
    :attr:`risk_class` is informational (it never authorizes an apply).
    """

    action_kind: ActionKind
    risk_class: RiskClass
    trigger: TriggerSignal


def objective_target_met(
    goal: CompiledGoal,
    *,
    observed: float | None,
    baseline: float | None = None,
) -> bool | None:
    """Whether ``observed`` already meets ``goal``'s target — ``None`` if undecidable.

    The bar comes entirely from the goal (no magic number): an **absolute** target
    is the level itself; a **relative** target is ``baseline * (1 + value)`` and
    therefore needs ``baseline`` to evaluate. Direction decides the comparison
    (``minimize`` ⇒ at/below the bar; ``maximize`` ⇒ at/above). Returns ``None``
    when it cannot be decided (no observed value, or a relative target with no
    baseline) — the caller treats *unknown* as *not met* (conservative: it may
    still propose, but the proof gate is what actually protects shipping).
    """
    if observed is None:
        return None
    target = goal.target
    if target.kind == "absolute":
        bar = target.value
    else:
        if baseline is None:
            return None
        bar = baseline * (1.0 + target.value)
    if goal.direction == "minimize":
        return observed <= bar
    return observed >= bar


def decide_rlm_asset(
    signal: RlmAssetSignal,
    goal: CompiledGoal,
    *,
    objective_met: bool | None,
    thresholds: DecisionThresholds | None = None,
) -> Decision | None:
    """RLM additive asset recurring across ≥ N traces + objective unmet → metric_view.

    Fires only for an *additive* asset type the generator can build
    (:data:`_ADDITIVE_ASSET_TYPES`), recurring across at least
    ``thresholds.min_asset_recurrence_traces`` distinct traces, while the goal's
    objective is **not** already met (``objective_met`` is ``False`` or unknown).
    A skill / prompt_change recommendation, or one that recurs too little, or a
    goal already at target, returns ``None`` (fail-closed: no candidate action).
    """
    th = thresholds or DecisionThresholds()
    if signal.asset_type not in _ADDITIVE_ASSET_TYPES:
        return None
    if signal.n_traces < th.min_asset_recurrence_traces:
        return None
    if objective_met is True:
        return None
    trigger = TriggerSignal(
        kind=TriggerKind.RLM_RECOMMENDED_ASSET,
        summary=(
            f"RLM recommended a {signal.asset_type!r} asset ({signal.title!r}) recurring "
            f"across {signal.n_traces} trace(s); goal objective {goal.objective_metric!r} "
            "not yet met"
        ),
        metric=goal.objective_metric,
        n_traces=signal.n_traces,
        trace_refs=list(signal.trace_ids),
        asset_type=signal.asset_type,
        source_rank=signal.rank,
    )
    return Decision(ActionKind.METRIC_VIEW, default_risk_class(ActionKind.METRIC_VIEW), trigger)


def decide_redundant_read(
    signal: RedundantReadSignal,
    goal: CompiledGoal,
    *,
    thresholds: DecisionThresholds | None = None,
) -> Decision | None:
    """A dominant, recurring redundant-read pattern → skill_update.

    Fires when the pattern dominates the waste diagnosis (``signal.dominant``) and
    repeats at least ``thresholds.min_redundant_occurrences`` times — the
    read-cache / context-compaction skill target. Otherwise ``None``.
    """
    th = thresholds or DecisionThresholds()
    if not signal.dominant or signal.occurrences < th.min_redundant_occurrences:
        return None
    target = signal.repeated_target or signal.tool or "repeated target"
    trigger = TriggerSignal(
        kind=TriggerKind.REDUNDANT_READ_PATTERN,
        summary=(
            f"redundant-read pattern dominates the L0 waste diagnosis: {target!r} repeated "
            f"{signal.occurrences} time(s)"
            + (
                f" (~{signal.estimated_wasted_tokens} wasted tokens)"
                if signal.estimated_wasted_tokens is not None
                else ""
            )
        ),
        metric=goal.objective_metric,
        observed_value=float(signal.occurrences),
        n_traces=len(signal.trace_ids),
        trace_refs=list(signal.trace_ids),
    )
    return Decision(ActionKind.SKILL_UPDATE, default_risk_class(ActionKind.SKILL_UPDATE), trigger)


def _judge_guardrail_threshold(goal: CompiledGoal, judge_name: str) -> float | None:
    """The goal's guardrail threshold for ``judge_name``, or ``None`` if it sets none."""
    for g in goal.guardrails:
        if g.kind == "judge" and g.name == judge_name and g.threshold is not None:
            return g.threshold
    return None


def decide_judge_dimension(
    signal: JudgeDimensionSignal,
    goal: CompiledGoal,
) -> Decision | None:
    """A trusted judge's dimension below the goal's threshold → gepa_prompt.

    The threshold comes from the goal's guardrail for that judge (no magic number);
    if the goal names no threshold for this judge, the rule cannot decide and
    returns ``None`` (fail-closed — a quality claim needs the goal's own bar). The
    judge must also be ``trusted`` (judge-vs-human agreement at/above floor): an
    untrusted judge's verdict cannot trigger an agent change.
    """
    threshold = _judge_guardrail_threshold(goal, signal.judge_name)
    if threshold is None:
        return None
    if not signal.trusted:
        return None
    if signal.score >= threshold:
        return None
    trigger = TriggerSignal(
        kind=TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD,
        summary=(
            f"trusted judge {signal.judge_name!r} scored dimension {signal.dimension!r} at "
            f"{signal.score} (below goal threshold {threshold})"
        ),
        metric=signal.dimension,
        observed_value=signal.score,
        threshold=threshold,
        n_traces=len(signal.trace_ids),
        trace_refs=list(signal.trace_ids),
        judge_name=signal.judge_name,
    )
    return Decision(ActionKind.GEPA_PROMPT, default_risk_class(ActionKind.GEPA_PROMPT), trigger)


def decide_post_apply_regression(signal: PostApplyRegressionSignal) -> Decision | None:
    """A registered version's post-apply regression → revert.

    Fires when the version regressed vs its predecessor. The revert proposal
    inherits the regressed change's ``risk_class`` (reverting an additive asset is
    additive; reverting an agent change is an agent change).
    """
    if not signal.regressed:
        return None
    trigger = TriggerSignal(
        kind=TriggerKind.POST_APPLY_REGRESSION,
        summary=(
            f"version {signal.agent_version!r} regressed vs predecessor "
            f"{signal.predecessor_version!r} on {signal.objective_metric!r}"
        ),
        metric=signal.objective_metric,
        n_traces=len(signal.trace_ids),
        trace_refs=list(signal.trace_ids),
    )
    return Decision(ActionKind.REVERT, signal.risk_class, trigger)


def decide(
    feedback: FeedbackBundle,
    goal: CompiledGoal,
    *,
    thresholds: DecisionThresholds | None = None,
) -> list[Decision]:
    """Run every applicable rule over ``feedback`` and collect the fired decisions.

    Pure: maps each detected signal through its rule and returns the decisions in
    a stable order (assets, then redundant reads, then judge dimensions, then
    regressions). A signal that no rule fires on contributes nothing — the
    controller then proves and gates each decision before any proposal exists.
    """
    th = thresholds or DecisionThresholds()
    objective_met = objective_target_met(
        goal,
        observed=feedback.objective_metric_value,
        baseline=feedback.objective_baseline_value,
    )
    decisions: list[Decision] = []
    for asset in feedback.rlm_assets:
        d = decide_rlm_asset(asset, goal, objective_met=objective_met, thresholds=th)
        if d is not None:
            decisions.append(d)
    for rr in feedback.redundant_reads:
        d = decide_redundant_read(rr, goal, thresholds=th)
        if d is not None:
            decisions.append(d)
    for jd in feedback.judge_dimensions:
        d = decide_judge_dimension(jd, goal)
        if d is not None:
            decisions.append(d)
    for reg in feedback.post_apply_regressions:
        d = decide_post_apply_regression(reg)
        if d is not None:
            decisions.append(d)
    return decisions
