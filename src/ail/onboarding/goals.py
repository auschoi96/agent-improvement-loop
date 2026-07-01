"""The onboarding wizard's **fixed goal catalog** and its data-gate requirements.

The "Add an agent" wizard (``docs/ONBOARDING_WIZARD.md`` §46, §3) offers exactly
four goals and no free-text: **Token efficiency · Latency · Accuracy · Cost**.
This module is the *single source of truth* for two facts the app must not
re-derive in TypeScript (the two-tier discipline of ``docs/OBSERVABILITY_APP.md``):

1. **goal → scorer** — the RESOLVED design note (§53–58, §92): *Accuracy* needs a
   MemAlign-aligned **judge** (the human-calibrated ``correctness`` scorer);
   *Latency* and *Cost* are **deterministic L0** numbers (an LLM judge cannot
   measure them better than the exact figure — no fake judge is stood up for
   them); *Token efficiency* is **hybrid** — a deterministic L0 token reduction
   plus an *optional* quality-per-token judge (``token_efficiency``).
2. **goal → data gates** — which readiness gates a chosen goal must clear before
   the loop will act. This is computed by **reusing** :func:`ail.readiness.compute_readiness`
   against a zero-facts (collecting) baseline, so every threshold and every
   "need N more …" string is the readiness module's own output. The floors
   (``50`` traces to prove, ``20`` labels for a judged goal, ``0.5`` scored-coverage)
   live only in :class:`ail.readiness.ReadinessThresholds` — never copied here.

Every metric/judge name below is validated at import against the real allowlist
(:mod:`ail.goals.allowlist`, itself anchored to :mod:`ail.metrics.contract` and
:data:`ail.judges.scorers.DEFAULT_SCORERS`), so a scorer rename fails the import
loud rather than letting the wizard name a scorer that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ail.cohorts import Cohort, TagFilter
from ail.goals.allowlist import is_judge, is_l0_metric
from ail.judges.scorers import CORRECTNESS, TOKEN_EFFICIENCY
from ail.readiness import (
    Gate,
    GateName,
    ReadinessFacts,
    ReadinessStatus,
    ReadinessThresholds,
    compute_readiness,
)

__all__ = [
    "GoalKey",
    "ScorerKind",
    "GoalSpec",
    "GOAL_CATALOG",
    "GoalOption",
    "GateRequirement",
    "GoalRequirement",
    "RequirementsResult",
    "build_requirements",
    "build_judge_config",
    "UnknownGoalError",
]


class UnknownGoalError(ValueError):
    """A requested goal key is not one of the four fixed wizard options."""


class GoalKey(StrEnum):
    """The four (and only four) goals the wizard offers — a fixed dropdown."""

    TOKEN_EFFICIENCY = "token_efficiency"
    LATENCY = "latency"
    ACCURACY = "accuracy"
    COST = "cost"


class ScorerKind(StrEnum):
    """How a goal's progress is measured — the resolved design mapping (§53–58)."""

    #: A deterministic L0 metric (:mod:`ail.metrics`); no LLM judge — measuring it
    #: with a judge would only reparrot the exact number (Latency, Cost).
    DETERMINISTIC_L0 = "deterministic_l0"
    #: A human-calibrated MemAlign-aligned judge (Accuracy → ``correctness``).
    MEMALIGN_JUDGE = "memalign_judge"
    #: A deterministic L0 objective plus an *optional* quality judge (Token
    #: efficiency → ``total_tokens`` + optional ``token_efficiency``).
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """One fixed goal's definition — a :class:`~ail.readiness.GoalView` by structure.

    Carries exactly the three members readiness consumes
    (:attr:`objective_metric`, :attr:`requires_quality`, :attr:`guardrail_names`)
    so it can drive :func:`ail.readiness.compute_readiness` directly, plus the
    display + scorer-mapping fields the wizard renders.

    Args:
        key: The stable goal identifier (the dropdown value).
        label: Human label shown in the dropdown.
        objective_metric: The metric whose movement is this goal's objective — a
            real L0 metric name or a registered judge name (validated at import).
        scorer_kind: How progress is measured (see :class:`ScorerKind`).
        requires_quality: Whether proving this goal needs a *judged* quality
            signal — drives whether readiness evaluates the label/judge/coverage
            gates. ``True`` only for Accuracy.
        guardrail_names: The judge(s) a quality goal's ``judge_trusted`` gate
            requires (Accuracy → ``("correctness",)``).
        optional_quality_judge: A judge the user *may* additionally enable for a
            deterministic/hybrid goal (Token efficiency → ``token_efficiency``);
            enabling it would add the quality gates. ``None`` when there is none.
        description: One-line explanation of the scorer choice, shown in the UI.
    """

    key: GoalKey
    label: str
    objective_metric: str
    scorer_kind: ScorerKind
    requires_quality: bool
    guardrail_names: tuple[str, ...] = ()
    optional_quality_judge: str | None = None
    description: str = ""

    @property
    def scorer(self) -> str:
        """A short human name of the scorer this goal maps to."""
        if self.scorer_kind is ScorerKind.MEMALIGN_JUDGE:
            return f"MemAlign judge ({', '.join(self.guardrail_names)})"
        if self.scorer_kind is ScorerKind.HYBRID:
            opt = self.optional_quality_judge
            return f"deterministic L0 ({self.objective_metric})" + (
                f" + optional judge ({opt})" if opt else ""
            )
        return f"deterministic L0 ({self.objective_metric})"


#: The catalog, in dropdown order. Each objective/judge name is validated below.
GOAL_CATALOG: dict[GoalKey, GoalSpec] = {
    GoalKey.TOKEN_EFFICIENCY: GoalSpec(
        key=GoalKey.TOKEN_EFFICIENCY,
        label="Token efficiency",
        objective_metric="total_tokens",
        scorer_kind=ScorerKind.HYBRID,
        requires_quality=False,
        optional_quality_judge=TOKEN_EFFICIENCY.name,
        description=(
            "Deterministic L0 token-reduction objective, correctness held. Hybrid: "
            "you may also enable the quality-per-token judge (needs labels)."
        ),
    ),
    GoalKey.LATENCY: GoalSpec(
        key=GoalKey.LATENCY,
        label="Latency",
        objective_metric="duration_seconds",
        scorer_kind=ScorerKind.DETERMINISTIC_L0,
        requires_quality=False,
        description=(
            "Deterministic L0 duration — measured exactly from the trace, no judge "
            "(a judge cannot measure latency better than the number)."
        ),
    ),
    GoalKey.ACCURACY: GoalSpec(
        key=GoalKey.ACCURACY,
        label="Accuracy",
        objective_metric="correctness",
        scorer_kind=ScorerKind.MEMALIGN_JUDGE,
        requires_quality=True,
        guardrail_names=(CORRECTNESS.name,),
        description=(
            "A human-calibrated MemAlign-aligned judge (correctness). Needs human "
            "labels to align the judge before any accuracy claim is trusted."
        ),
    ),
    GoalKey.COST: GoalSpec(
        key=GoalKey.COST,
        label="Cost",
        objective_metric="total_usd",
        scorer_kind=ScorerKind.DETERMINISTIC_L0,
        requires_quality=False,
        description=(
            "Deterministic L0 estimated cost — measured from tokens, no judge "
            "(a judge cannot measure cost better than the number)."
        ),
    ),
}

# Anchor every catalog name to the real sources (mirrors ail.goals.allowlist's own
# import-time check): an L0 objective must be a known L0 metric, a judge objective
# and any guardrail/optional judge must be a registered scorer. A rename upstream
# fails THIS import loud rather than letting the wizard offer a phantom scorer.
for _spec in GOAL_CATALOG.values():
    if _spec.scorer_kind is ScorerKind.MEMALIGN_JUDGE:
        if not is_judge(_spec.objective_metric):  # pragma: no cover - guards drift
            raise RuntimeError(
                f"goal {_spec.key!r} objective {_spec.objective_metric!r} is not a "
                "registered judge — ail.judges.scorers drifted."
            )
    elif not is_l0_metric(_spec.objective_metric):  # pragma: no cover - guards drift
        raise RuntimeError(
            f"goal {_spec.key!r} objective {_spec.objective_metric!r} is not a known "
            "L0 metric — ail.metrics.contract / ail.goals.allowlist drifted."
        )
    for _judge in (*_spec.guardrail_names, *filter(None, (_spec.optional_quality_judge,))):
        if not is_judge(_judge):  # pragma: no cover - guards drift
            raise RuntimeError(
                f"goal {_spec.key!r} names judge {_judge!r} that is not registered in "
                "ail.judges.scorers.DEFAULT_SCORERS."
            )


def resolve_goal(key: str) -> GoalSpec:
    """Return the :class:`GoalSpec` for ``key`` or raise :class:`UnknownGoalError`."""
    try:
        return GOAL_CATALOG[GoalKey(key)]
    except ValueError as exc:
        valid = ", ".join(k.value for k in GoalKey)
        raise UnknownGoalError(f"unknown goal {key!r}; must be one of: {valid}") from exc


# ---------------------------------------------------------------------------
# JSON-shaped results the wizard renders (pydantic; extra fields forbidden)
# ---------------------------------------------------------------------------


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoalOption(_Contract):
    """One catalog entry, as the dropdown renders it (no gate detail)."""

    key: str
    label: str
    objective_metric: str
    scorer: str
    scorer_kind: str
    requires_quality: bool
    guardrail_judges: list[str] = Field(default_factory=list)
    optional_quality_judge: str | None = None
    description: str = ""

    @classmethod
    def from_spec(cls, spec: GoalSpec) -> GoalOption:
        return cls(
            key=spec.key.value,
            label=spec.label,
            objective_metric=spec.objective_metric,
            scorer=spec.scorer,
            scorer_kind=spec.scorer_kind.value,
            requires_quality=spec.requires_quality,
            guardrail_judges=list(spec.guardrail_names),
            optional_quality_judge=spec.optional_quality_judge,
            description=spec.description,
        )


class GateRequirement(_Contract):
    """One readiness gate a chosen goal must clear — the module's own reason text.

    ``needed`` is :attr:`ail.readiness.Gate.reason` computed against a zero-facts
    baseline, so it reads e.g. "need 50 more trace(s) …" with the real threshold.
    ``threshold`` is the raw number from :class:`ReadinessThresholds` (``None`` for
    boolean gates like the frozen suite / judge-trusted).
    """

    name: str
    label: str
    needed: str
    threshold: float | None = None


class GoalRequirement(_Contract):
    """A chosen goal's scorer mapping + the exact gates it needs (Page 3 source)."""

    key: str
    label: str
    objective_metric: str
    scorer: str
    scorer_kind: str
    requires_quality: bool
    requires_labels: bool
    guardrail_judges: list[str] = Field(default_factory=list)
    optional_quality_judge: str | None = None
    gates: list[GateRequirement] = Field(default_factory=list)


class Thresholds(_Contract):
    """The readiness floors, surfaced verbatim from :class:`ReadinessThresholds`."""

    baseline_min_traces: int
    prove_min_traces: int
    quality_min_labels: int
    scored_coverage_floor: float


class RequirementsResult(_Contract):
    """The wizard's goal catalog + the data-gate requirements for a selection.

    Returned by the ``requirements`` action. ``catalog`` populates the dropdown;
    ``selected``/``union_gates``/``requires_labels`` populate the data-gate page
    for the goals the user picked (empty selection → catalog + thresholds only).
    """

    outcome: str = "requirements"
    thresholds: Thresholds
    catalog: list[GoalOption]
    selected: list[GoalRequirement] = Field(default_factory=list)
    union_gates: list[GateRequirement] = Field(default_factory=list)
    requires_labels: bool = False


# ---------------------------------------------------------------------------
# Gate surfacing — pure reuse of ail.readiness.compute_readiness
# ---------------------------------------------------------------------------

_GATE_LABELS: dict[GateName, str] = {
    GateName.TRACE_BASELINE: "Traces to baseline",
    GateName.TRACE_PROVE: "Traces to prove an improvement",
    GateName.FROZEN_SUITE: "Frozen Task Suite present",
    GateName.HUMAN_LABELS: "Human labels to align the judge",
    GateName.JUDGE_TRUSTED: "Judge measured & trusted",
    GateName.SCORED_COVERAGE: "Scored-coverage floor",
}

# The all-matching cohort used only to drive compute_readiness; its identity does
# not affect which gates are built or their thresholds (gates key off the goal +
# facts, not the cohort name).
_ONBOARDING_COHORT = Cohort(name="onboarding", tag_filter=TagFilter())


def _threshold_for(name: GateName, th: ReadinessThresholds) -> float | None:
    return {
        GateName.TRACE_BASELINE: float(th.baseline_min_traces),
        GateName.TRACE_PROVE: float(th.prove_min_traces),
        GateName.HUMAN_LABELS: float(th.quality_min_labels),
        GateName.SCORED_COVERAGE: th.scored_coverage_floor,
    }.get(name)


def _gate_requirement(gate: Gate, th: ReadinessThresholds) -> GateRequirement:
    return GateRequirement(
        name=gate.name.value,
        label=_GATE_LABELS.get(gate.name, gate.name.value),
        needed=gate.reason,
        threshold=_threshold_for(gate.name, th),
    )


def _readiness_for(spec: GoalSpec, th: ReadinessThresholds) -> ReadinessStatus:
    """The gate set a goal needs, from a zero-facts (collecting) baseline.

    Zero facts means every applicable gate is present and *failing*, so its
    :attr:`~ail.readiness.Gate.reason` spells out the full requirement ("need 50
    more traces", "need 20 more human labels"). Reusing the readiness module here
    keeps the wizard's floors in lockstep with the code that enforces them.
    """
    return compute_readiness(_ONBOARDING_COHORT, spec, ReadinessFacts(), thresholds=th)


def build_requirements(goal_keys: list[str] | None = None) -> RequirementsResult:
    """Build the goal catalog and, for ``goal_keys``, their data-gate requirements.

    ``goal_keys`` are validated against the fixed catalog (an unknown key raises
    :class:`UnknownGoalError`). The per-goal gates and the union across the
    selection are the readiness module's own output; ``requires_labels`` is true
    iff any chosen goal is judged (needs the ``human_labels`` gate).
    """
    th = ReadinessThresholds()
    catalog = [GoalOption.from_spec(GOAL_CATALOG[k]) for k in GoalKey]

    selected: list[GoalRequirement] = []
    union: dict[str, GateRequirement] = {}
    requires_labels = False
    for key in goal_keys or []:
        spec = resolve_goal(key)
        status = _readiness_for(spec, th)
        gates = [_gate_requirement(g, th) for g in status.gates]
        goal_needs_labels = any(g.name == GateName.HUMAN_LABELS.value for g in gates)
        requires_labels = requires_labels or goal_needs_labels
        selected.append(
            GoalRequirement(
                key=spec.key.value,
                label=spec.label,
                objective_metric=spec.objective_metric,
                scorer=spec.scorer,
                scorer_kind=spec.scorer_kind.value,
                requires_quality=spec.requires_quality,
                requires_labels=goal_needs_labels,
                guardrail_judges=list(spec.guardrail_names),
                optional_quality_judge=spec.optional_quality_judge,
                gates=gates,
            )
        )
        for gate in gates:
            union.setdefault(gate.name, gate)

    return RequirementsResult(
        thresholds=Thresholds(
            baseline_min_traces=th.baseline_min_traces,
            prove_min_traces=th.prove_min_traces,
            quality_min_labels=th.quality_min_labels,
            scored_coverage_floor=th.scored_coverage_floor,
        ),
        catalog=catalog,
        selected=selected,
        union_gates=list(union.values()),
        requires_labels=requires_labels,
    )


def build_judge_config(goal_keys: list[str]) -> dict[str, object]:
    """The opaque per-agent ``judge_config`` the registry carries for these goals.

    Structured + JSON-serializable (the registry stores it as ``judge_config_json``;
    the judges lane interprets it). Records, per chosen goal, its objective metric,
    scorer kind, whether it is judged, and the guardrail/optional judge names — the
    resolved goal→scorer mapping, so nothing downstream has to re-derive it.
    """
    if not goal_keys:
        raise ValueError("at least one goal is required to register an agent")
    scorers: dict[str, object] = {}
    for key in goal_keys:
        spec = resolve_goal(key)
        scorers[spec.key.value] = {
            "objective_metric": spec.objective_metric,
            "scorer_kind": spec.scorer_kind.value,
            "requires_quality": spec.requires_quality,
            "guardrail_judges": list(spec.guardrail_names),
            "optional_quality_judge": spec.optional_quality_judge,
        }
    return {"goals": list(dict.fromkeys(goal_keys)), "scorers": scorers}
