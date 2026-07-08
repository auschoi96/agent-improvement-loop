"""Route extracted dimensions and compose them into one ``CompiledGoal``.

The second step of the intake engine. It takes the
:class:`~ail.requirements.extractor.RequirementDimension` set and does two things:

**Route** — for each dimension it decides, via the goal allowlist
(:func:`ail.goals.allowlist.is_l0_metric` / :func:`~ail.goals.allowlist.is_judge`),
whether the dimension is a **deterministic L0 metric** (latency, cost, tokens,
tool-call count, redundancy — measured exactly, *no judge*) or a **quality /
subjective** dimension that needs a ``{{ trace }}`` MemAlign judge authored for it
(:func:`ail.judges.author_judge`). The extractor's ``metric`` field is only a
*candidate*: a non-null value that is not actually an L0 metric fails closed here,
so a mis-mapped metric can never masquerade as deterministic.

**Compose** — the user's priorities compose a single
:class:`ail.goals.compiler.CompiledGoal`: the **highest-priority** dimension becomes
the primary objective, every other dimension becomes a guardrail (a judge guardrail
for a quality dimension, a deterministic ``must_not_regress`` guardrail for an L0
one). Because a composed goal may reference a judge the intake is about to author —
one not yet in the static built-in allowlist — composition runs inside
:func:`ail.goals.allowlist.judge_allowlist` so those authored names validate (GAP B).

**Propose-then-confirm.** :func:`plan_requirements` / :func:`build_plan` return a
:class:`RequirementsPlan` *proposal* that lists exactly which judges *would* be
authored and which metrics are deterministic — and **authors nothing and persists
nothing**. Side effects happen only in :func:`execute_plan`, which refuses unless
the plan has been :meth:`~RequirementsPlan.confirm`\\ed. A dimension the user did not
intend is never auto-authored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ail.cohorts import Cohort
from ail.goals.allowlist import is_l0_metric, judge_allowlist
from ail.goals.compiler import (
    CompiledGoal,
    GoalDirection,
    GoalTarget,
    Guardrail,
)
from ail.judges.authoring import normalize_judge_name
from ail.requirements.extractor import (
    GoalProposerLLM,
    RequirementDimension,
    extract_dimensions,
)

__all__ = [
    "DimensionKind",
    "DimensionRole",
    "PlannedDimension",
    "RequirementsPlan",
    "PlanExecution",
    "JudgeAuthor",
    "RequirementsRoutingError",
    "RequirementsNotConfirmedError",
    "build_plan",
    "plan_requirements",
    "execute_plan",
]

#: How a dimension is measured after routing.
DimensionKind = Literal["deterministic_l0", "memalign_judge"]
#: A dimension's role in the composed goal.
DimensionRole = Literal["objective", "guardrail"]

#: The default relative target magnitude per direction, used for the composed goal's
#: primary objective. These are conservative *placeholders* the human reviews and
#: confirms in the propose-then-confirm flow (the arg-based loop already defaults a
#: relative ``-0.30`` token target); the goal schema requires a signed relative
#: target whose sign matches the direction, so a non-zero default is needed.
_DEFAULT_RELATIVE_TARGET: dict[GoalDirection, float] = {"minimize": -0.30, "maximize": 0.10}


class RequirementsRoutingError(Exception):
    """A dimension could not be routed to a deterministic metric or a judge.

    Raised when the LLM's candidate ``metric`` names something that is not a real L0
    metric (fail-closed: a mis-mapped metric is refused, never treated as
    deterministic), or when two quality dimensions normalize to the same judge name
    (an ambiguous duplicate that would author one judge for two intents).
    """


class RequirementsNotConfirmedError(Exception):
    """:func:`execute_plan` was called on a plan that was not :meth:`confirm`\\ed.

    The propose-then-confirm gate: no judge is authored and nothing is persisted
    until a human confirms the plan.
    """


class JudgeAuthor(Protocol):
    """The injectable judge-authoring seam (mirrors :func:`ail.judges.author_judge`).

    Any callable taking the dimension's ``name``/``description`` and the target
    ``experiment_id`` satisfies it. Production uses :func:`ail.judges.author_judge`
    (which guarantees a ``{{ trace }}`` judge whose label-schema name equals the
    judge name); tests inject a spy so :func:`execute_plan` is fully offline.
    """

    def __call__(self, name: str, description: str, *, experiment_id: str) -> Any: ...


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannedDimension(_Contract):
    """One routed dimension: how it will be measured and its role in the goal.

    Args:
        name: The dimension's name (from :class:`RequirementDimension`).
        description: What the dimension means (the judge rubric source for a quality
            dimension).
        user_priority: The user's priority (``1`` = highest).
        kind: ``"deterministic_l0"`` (an exact metric, no judge) or
            ``"memalign_judge"`` (a ``{{ trace }}`` judge to author).
        role: ``"objective"`` for the highest-priority dimension, ``"guardrail"``
            for the rest.
        metric: The L0 metric name, set iff ``kind == "deterministic_l0"``.
        judge_name: The canonical judge name (:func:`normalize_judge_name`), set iff
            ``kind == "memalign_judge"`` — this is the exact judge
            :func:`ail.judges.author_judge` would create.
        direction: The optimization direction implied by the dimension —
            ``"minimize"`` for the (lower-is-better) L0 metrics, ``"maximize"`` for a
            quality judge (higher score is better).
    """

    name: str
    description: str
    user_priority: int
    kind: DimensionKind
    role: DimensionRole
    metric: str | None
    judge_name: str | None
    direction: GoalDirection


@dataclass(frozen=True, slots=True)
class RequirementsPlan:
    """The propose-then-confirm plan: the composed goal + what it would author.

    A pure proposal — building it authors no judge and persists nothing. It lists
    the routed dimensions (:attr:`dimensions`), the composed but **unconfirmed**
    :attr:`goal`, and the exact split of :attr:`judges_to_author` vs
    :attr:`deterministic_metrics`. Call :meth:`confirm` (the human gate) before
    :func:`execute_plan` will author judges or persist anything.

    A frozen dataclass (not a pydantic model) on purpose: it *holds* an already-
    validated :class:`~ail.goals.compiler.CompiledGoal` as-is, so it never triggers
    a re-validation of the goal outside the authored-judge allowlist context in
    which the goal was composed.
    """

    dimensions: tuple[PlannedDimension, ...]
    goal: CompiledGoal
    confirmed: bool = False

    @property
    def judges_to_author(self) -> tuple[PlannedDimension, ...]:
        """The quality dimensions a confirm+execute would author a judge for."""
        return tuple(d for d in self.dimensions if d.kind == "memalign_judge")

    @property
    def deterministic_metrics(self) -> tuple[PlannedDimension, ...]:
        """The dimensions measured by a deterministic L0 metric (no judge)."""
        return tuple(d for d in self.dimensions if d.kind == "deterministic_l0")

    @property
    def objective(self) -> PlannedDimension:
        """The primary (highest-priority) dimension — the goal's objective."""
        return next(d for d in self.dimensions if d.role == "objective")

    def confirm(self) -> RequirementsPlan:
        """Return a confirmed copy (the human gate), with the goal itself confirmed.

        Returns a new plan with ``confirmed=True`` and its :attr:`goal` replaced by
        :meth:`goal.confirm() <ail.goals.compiler.CompiledGoal.confirm>` (which uses
        ``model_copy`` — no re-validation), so the goal composed under the authored-
        judge allowlist stays valid without re-entering that context.
        """
        return replace(self, confirmed=True, goal=self.goal.confirm())

    def describe(self) -> str:
        """A one-line, human-readable summary of the plan (for operator surfacing)."""
        judges = ", ".join(d.judge_name or d.name for d in self.judges_to_author) or "none"
        metrics = ", ".join(d.metric or d.name for d in self.deterministic_metrics) or "none"
        obj = self.objective
        obj_scorer = obj.judge_name if obj.kind == "memalign_judge" else obj.metric
        return (
            f"objective={self.goal.objective_metric} ({obj_scorer}, {self.goal.direction}); "
            f"judges to author=[{judges}]; deterministic metrics=[{metrics}]; "
            f"confirmed={self.confirmed}"
        )


@dataclass(frozen=True, slots=True)
class PlanExecution:
    """The result of :func:`execute_plan`: what was authored / persisted.

    ``authored`` holds whatever the injected :class:`JudgeAuthor` returned per judge
    dimension (an :class:`ail.judges.authoring.AuthoredJudge` in production);
    ``authored_names`` are the canonical judge names authored, in plan order.
    ``persisted`` is ``True`` iff a persister was supplied and the goal was written.
    ``goal`` is the confirmed goal that was authored/persisted against.
    """

    goal: CompiledGoal
    authored: tuple[Any, ...]
    authored_names: tuple[str, ...]
    persisted: bool


def _route(
    dimension: RequirementDimension,
) -> tuple[DimensionKind, str | None, str | None, GoalDirection]:
    """Route one dimension to ``(kind, metric, judge_name, direction)``.

    A non-null candidate ``metric`` must be a real L0 metric
    (:func:`ail.goals.allowlist.is_l0_metric`) — otherwise fail closed. A null
    ``metric`` is a quality dimension: author a ``{{ trace }}`` judge named by
    :func:`normalize_judge_name`. L0 metrics are lower-is-better, so the derived
    direction is ``minimize``; a quality judge is higher-is-better, so ``maximize``.
    """
    if dimension.metric is not None:
        if not is_l0_metric(dimension.metric):
            raise RequirementsRoutingError(
                f"dimension {dimension.name!r} proposed metric {dimension.metric!r}, which "
                "is not a deterministic L0 metric (see ail.goals.allowlist.L0_OBJECTIVE_METRICS). "
                "Refusing to treat it as deterministic (fail-closed); a quality dimension must "
                "leave 'metric' null so a judge is authored."
            )
        return "deterministic_l0", dimension.metric, None, "minimize"
    return "memalign_judge", None, normalize_judge_name(dimension.name), "maximize"


def build_plan(
    dimensions: Sequence[RequirementDimension],
    cohort: Cohort | str,
    *,
    known_judges: Iterable[str] = (),
) -> RequirementsPlan:
    """Route ``dimensions`` and compose them into a :class:`RequirementsPlan`.

    The highest-priority dimension (lowest ``user_priority``, ties broken by input
    order) is the primary objective; the rest are guardrails. A judged objective is
    also added as its own judge guardrail (the :class:`CompiledGoal` contract). The
    composed goal is validated inside :func:`ail.goals.allowlist.judge_allowlist` so
    the judges this plan would author — plus any ``known_judges`` already registered —
    validate even though they are not in the static built-in set.

    Args:
        dimensions: The extracted dimensions (at least one).
        cohort: The cohort the composed goal is bound to (a :class:`ail.cohorts.Cohort`
            or its name) — the caller owns this, never the LLM.
        known_judges: Judge names already registered for the agent (e.g. from
            :func:`ail.goals.allowlist.sourced_judge_names`) that a dimension may
            reference without this plan re-authoring them.

    Raises:
        RequirementsRoutingError: A dimension has an empty set, a mis-mapped metric,
            or two quality dimensions collide on one judge name.
    """
    if not dimensions:
        raise RequirementsRoutingError("at least one dimension is required to compose a goal.")

    planned_raw: list[
        tuple[int, RequirementDimension, DimensionKind, str | None, str | None, GoalDirection]
    ] = []
    seen_judges: dict[str, str] = {}
    for idx, dim in enumerate(dimensions):
        kind, metric, judge_name, direction = _route(dim)
        if judge_name is not None:
            if judge_name in seen_judges:
                raise RequirementsRoutingError(
                    f"dimensions {seen_judges[judge_name]!r} and {dim.name!r} both normalize to "
                    f"judge name {judge_name!r}; give them distinct names (ambiguous duplicate)."
                )
            seen_judges[judge_name] = dim.name
        planned_raw.append((idx, dim, kind, metric, judge_name, direction))

    # Primary = highest priority (lowest user_priority), ties broken by input order.
    primary_idx = min(planned_raw, key=lambda r: (r[1].user_priority, r[0]))[0]

    planned: list[PlannedDimension] = []
    for idx, dim, kind, metric, judge_name, direction in planned_raw:
        planned.append(
            PlannedDimension(
                name=dim.name,
                description=dim.description,
                user_priority=dim.user_priority,
                kind=kind,
                role="objective" if idx == primary_idx else "guardrail",
                metric=metric,
                judge_name=judge_name,
                direction=direction,
            )
        )

    authored_judge_names = frozenset(d.judge_name for d in planned if d.judge_name)
    extra_judges = authored_judge_names | frozenset(known_judges)
    with judge_allowlist(extra_judges):
        goal = _compose_goal(planned, cohort)

    return RequirementsPlan(dimensions=tuple(planned), goal=goal)


def _compose_goal(planned: Sequence[PlannedDimension], cohort: Cohort | str) -> CompiledGoal:
    """Compose the primary objective + guardrails into an unconfirmed ``CompiledGoal``.

    Must be called inside a :func:`ail.goals.allowlist.judge_allowlist` context that
    admits every judge name referenced, so validation of an authored judge passes.
    """
    primary = next(d for d in planned if d.role == "objective")

    if primary.kind == "memalign_judge":
        assert primary.judge_name is not None  # routing guarantees this
        objective_metric = primary.judge_name
    else:
        assert primary.metric is not None  # routing guarantees this
        objective_metric = primary.metric
    direction = primary.direction

    guardrails: list[Guardrail] = []
    seen: set[str] = set()

    # A judged objective must ALSO be a judge guardrail (the readiness contract:
    # readiness does no objective-as-judge auto-detection).
    if primary.kind == "memalign_judge" and primary.judge_name is not None:
        guardrails.append(Guardrail(name=primary.judge_name, kind="judge", must_not_regress=True))
        seen.add(primary.judge_name)

    for dim in planned:
        if dim.role == "objective":
            continue
        gkind: Literal["metric", "judge"]
        if dim.kind == "memalign_judge":
            assert dim.judge_name is not None
            name, gkind = dim.judge_name, "judge"
        else:
            assert dim.metric is not None
            name, gkind = dim.metric, "metric"
        # Dedupe, and never add a metric guardrail that duplicates the objective metric.
        if name in seen or name == objective_metric:
            continue
        guardrails.append(Guardrail(name=name, kind=gkind, must_not_regress=True))
        seen.add(name)

    return CompiledGoal(
        objective_metric=objective_metric,
        direction=direction,
        target=GoalTarget(value=_DEFAULT_RELATIVE_TARGET[direction], kind="relative"),
        guardrails=tuple(guardrails),
        cohort=cohort,
        human_confirmed=False,
    )


def plan_requirements(
    nl_text: str,
    cohort: Cohort | str,
    *,
    llm: GoalProposerLLM,
    known_judges: Iterable[str] = (),
) -> RequirementsPlan:
    """Extract dimensions from ``nl_text`` and compose them into a plan proposal.

    Convenience over :func:`ail.requirements.extractor.extract_dimensions` +
    :func:`build_plan`. Authors nothing and persists nothing — the returned plan is
    a proposal that must be :meth:`~RequirementsPlan.confirm`\\ed before
    :func:`execute_plan` acts.
    """
    dimensions = extract_dimensions(nl_text, llm=llm)
    return build_plan(dimensions, cohort, known_judges=known_judges)


def execute_plan(
    plan: RequirementsPlan,
    *,
    experiment_id: str,
    author: JudgeAuthor | None = None,
    persist: Callable[[CompiledGoal], None] | None = None,
) -> PlanExecution:
    """Author the plan's judges and persist its goal — only after :meth:`confirm`.

    The single side-effecting entry point. It **refuses** with
    :class:`RequirementsNotConfirmedError` unless ``plan.confirmed`` is ``True``, so a
    judge is never authored and nothing is persisted for an unconfirmed plan. For a
    confirmed plan it authors one ``{{ trace }}`` judge per quality dimension (in plan
    order) via ``author`` (defaulting to :func:`ail.judges.author_judge`), then — if a
    ``persist`` callable is supplied — writes the confirmed goal.

    Judges are authored **before** the goal is persisted, so the persisted goal only
    ever references judges that were just created.

    Args:
        plan: A confirmed :class:`RequirementsPlan`.
        experiment_id: The MLflow experiment the judges are authored against.
        author: The judge-authoring seam; ``None`` lazily uses
            :func:`ail.judges.author_judge` (which needs ``databricks-agents`` to
            register). Inject a spy in tests.
        persist: A callable that writes the confirmed goal (e.g. from
            :func:`ail.requirements.persistence.compiled_goal_persister`); ``None``
            skips persistence.

    Returns:
        A :class:`PlanExecution` recording the authored judges and whether the goal
        was persisted.
    """
    if not plan.confirmed:
        raise RequirementsNotConfirmedError(
            "refusing to author judges / persist the goal: the plan is not confirmed. "
            "Review it and call RequirementsPlan.confirm() first (propose-then-confirm)."
        )

    author_fn: JudgeAuthor
    if author is not None:
        author_fn = author
    else:
        from ail.judges.authoring import author_judge

        author_fn = author_judge

    authored: list[Any] = []
    authored_names: list[str] = []
    for dim in plan.judges_to_author:
        assert dim.judge_name is not None
        authored.append(author_fn(dim.name, dim.description, experiment_id=experiment_id))
        authored_names.append(dim.judge_name)

    persisted = False
    if persist is not None:
        persist(plan.goal)
        persisted = True

    return PlanExecution(
        goal=plan.goal,
        authored=tuple(authored),
        authored_names=tuple(authored_names),
        persisted=persisted,
    )
