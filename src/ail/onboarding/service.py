"""The onboarding wizard's **server side** — the Python the app's authenticated
routes invoke (``docs/ONBOARDING_WIZARD.md`` slice 1).

The app is a thin AppKit (Node/React) app; the authenticated ``server/plugins/onboarding``
routes resolve the actor from the platform identity headers and hand a JSON action
to this module (the same bridge shape as :mod:`ail.loop.apply_service`: a single
CLI reads a JSON action on stdin and prints a typed JSON result on stdout). It does
a small set of things, **reusing** existing framework capabilities rather than
reimplementing any:

* ``requirements`` — the fixed goal catalog + the data gates a selection needs,
  straight from :mod:`ail.onboarding.goals` (which reuses :func:`ail.readiness.compute_readiness`).
* ``validate_experiment`` — is an experiment *fresh* (empty of prior AIL state)?
  via :func:`ail.onboarding.experiment.validate_experiment`, fail-closed.
* ``create_experiment`` — create a fresh MLflow experiment, fail-closed with an
  honest permission error + documented prerequisite when the SP cannot.
* ``register_agent`` — write the new agent to the ``agent_registry`` UC table by
  **reusing** :func:`ail.publish_versions.publish_registry` (never reimplemented),
  so it appears in the app's existing AgentSwitcher.
* ``preview_requirements`` — run the free-form requirements-intake engine
  (:func:`ail.requirements.plan_requirements`, with the SAME LLM seam
  :func:`ail.goals.compile_goal` uses) and return a **structured, machine-readable
  preview** of the routed plan (which dimensions become authored ``{{trace}}`` judges
  vs. deterministic L0 metrics, the objective + guardrails, and the *suggested*
  target). A pure proposal — it authors nothing and persists nothing.
* ``confirm_requirements`` — re-derive the same plan from the requirements text,
  apply the human's explicit ``objective_target``, and **reuse**
  :func:`ail.requirements.execute_plan` to author the judges (via the existing
  :func:`ail.judges.author_judge` path), register the exact dimensions as MLflow
  custom code scorers, and persist the confirmed goal (via
  :func:`ail.requirements.compiled_goal_persister`). Fail-closed: ``execute_plan``
  refuses unless the plan is confirmed *and* the goal is ``human_confirmed``.

**Fail-closed / no fabrication everywhere.** An empty actor, a non-fresh
experiment, a denied create, a name/experiment collision, or any infra failure all
yield a ``refused``/``error`` result — never a fabricated "created"/"registered".
The actor is the **authenticated** identity the route passes; it is never trusted
from the browser and (defence-in-depth) an empty actor is refused here too.

**Unit-testable with no live write.** The pure orchestration (:func:`register_agent`
and the ``run_*`` helpers driven with injected clients) runs against fakes; the live
wiring (:func:`run_action`, :func:`main`) is a thin composition on top.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ail.goals.compiler import (
    CompiledGoal,
    GoalCompileError,
    GoalContractError,
    GoalProposerLLM,
    _default_databricks_proposer,
)
from ail.onboarding.experiment import (
    ExperimentAccessError,
    ExperimentClient,
    ExperimentPermissionError,
    UcTraceLocation,
    build_experiment_client,
)
from ail.onboarding.experiment import create_experiment as create_experiment_probe
from ail.onboarding.experiment import validate_experiment as validate_experiment_probe
from ail.onboarding.goals import (
    GOAL_CATALOG,
    GoalKey,
    RequirementsResult,
    UnknownGoalError,
    build_judge_config,
    build_requirements,
)
from ail.publish import DEFAULT_CATALOG, DEFAULT_SCHEMA
from ail.publish_versions import REGISTRY_TABLE, publish_registry
from ail.registry import Agent, AgentRegistry
from ail.requirements import (
    RequirementsExtractionError,
    RequirementsNotConfirmedError,
    RequirementsPlan,
    RequirementsRoutingError,
    compiled_goal_persister,
    execute_plan,
    plan_requirements,
)
from ail.requirements.composer import JudgeAuthor

__all__ = [
    "OnboardingOutcome",
    "ValidationResult",
    "CreationResult",
    "RegisterResult",
    "PreviewedDimension",
    "SuggestedTarget",
    "RequirementsPreviewResult",
    "RequirementsConfirmResult",
    "BootstrapResult",
    "ErrorResult",
    "compiled_goal_to_goal_config",
    "register_agent",
    "run_requirements",
    "run_preview_requirements",
    "run_confirm_requirements",
    "run_validate",
    "run_create",
    "run_register",
    "run_action",
    "load_registered_agents",
    "main",
]


class OnboardingOutcome(StrEnum):
    """The outcome the app surfaces for a wizard action."""

    REQUIREMENTS = "requirements"
    VALIDATED = "validated"
    CREATED = "created"
    REGISTERED = "registered"
    #: A free-form requirements PREVIEW — the routed plan, authored/persisted nothing.
    REQUIREMENTS_PREVIEW = "requirements_preview"
    #: A confirmed free-form requirements intake — judges authored + goal persisted.
    REQUIREMENTS_CONFIRMED = "requirements_confirmed"
    #: A fail-closed decision-level refusal (collision, bad input, empty actor) —
    #: nothing was written; surface :attr:`refused_reason`.
    REFUSED = "refused"
    #: An infrastructure / access error — never a fabricated success.
    ERROR = "error"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidationResult(_Contract):
    """The experiment-freshness verdict (page 1). ``fresh`` is the honest gate."""

    outcome: OnboardingOutcome
    experiment_id: str
    name: str = ""
    exists: bool = False
    fresh: bool = False
    trace_count: int = 0
    trace_count_capped: bool = False
    already_registered: bool = False
    registered_as: str | None = None
    reasons: list[str] = Field(default_factory=list)
    actor: str = ""
    error: str | None = None


class CreationResult(_Contract):
    """The experiment-creation result (page 1). ``prerequisite`` names the deploy grant.

    On the CREATED path we hand the experiment back **ready to use**:
    ``experiment_url`` is the workspace deep-link (``{host}/ml/experiments/{id}``,
    host resolved LIVE from the active profile — ``""`` when it can't be resolved,
    fail-soft), and ``tracing_hint`` is a copy-paste snippet pointing an agent's
    MLflow tracing at the new experiment. Both are empty on any non-CREATED outcome.
    """

    outcome: OnboardingOutcome
    experiment_id: str = ""
    name: str = ""
    experiment_url: str = ""
    tracing_hint: str = ""
    annotations_table: str = ""
    actor: str = ""
    error: str | None = None
    prerequisite: str | None = None


class RegisterResult(_Contract):
    """The agent-registration result (page 4)."""

    outcome: OnboardingOutcome
    agent_name: str
    experiment_id: str
    goals: list[str] = Field(default_factory=list)
    judge_config: dict[str, Any] | None = None
    registered_code_scorers: list[str] = Field(default_factory=list)
    actor: str = ""
    refused_reason: str | None = None
    error: str | None = None


class PreviewedDimension(_Contract):
    """One routed dimension in a requirements PREVIEW (machine-readable).

    A JSON-serializable projection of :class:`ail.requirements.PlannedDimension` — the
    client renders it verbatim (two-tier: Python owns the routing facts).
    """

    name: str
    description: str
    user_priority: int
    #: ``"deterministic_l0"`` (an exact metric, no judge) or ``"memalign_judge"``.
    kind: str
    #: ``"objective"`` (the primary dimension) or ``"guardrail"``.
    role: str
    metric: str | None = None
    judge_name: str | None = None
    #: ``"minimize"`` (L0 metric) or ``"maximize"`` (quality judge).
    direction: str


class SuggestedTarget(_Contract):
    """The composed objective target — a **suggestion** the human must set/acknowledge.

    ``is_suggestion`` is always ``True`` on a preview: the value is the composer's
    signed relative default, surfaced so the wizard can pre-fill an editable field
    labelled "adjust before confirming". It is never a confirmed/approved value.
    """

    value: float
    kind: str
    is_suggestion: bool = True


class RequirementsPreviewResult(_Contract):
    """The free-form requirements PREVIEW (``preview_requirements``).

    The plan's human-readable :meth:`~ail.requirements.RequirementsPlan.describe`
    summary plus machine fields per dimension and the suggested objective target.
    Authored nothing, persisted nothing — a pure proposal for human review.
    """

    outcome: OnboardingOutcome = OnboardingOutcome.REQUIREMENTS_PREVIEW
    requirements_text: str = ""
    cohort: str = ""
    agent_name: str = ""
    describe: str = ""
    objective_metric: str = ""
    direction: str = ""
    requires_quality: bool = False
    dimensions: list[PreviewedDimension] = Field(default_factory=list)
    judges_to_author: list[str] = Field(default_factory=list)
    deterministic_metrics: list[str] = Field(default_factory=list)
    suggested_target: SuggestedTarget | None = None
    actor: str = ""


class RequirementsConfirmResult(_Contract):
    """The confirmed free-form requirements intake (``confirm_requirements``).

    Honest outcome: ``requirements_confirmed`` with the judges that were authored and
    whether the goal was persisted; ``refused`` (anonymous actor / no explicit
    target) or ``error`` (extraction/routing/contract/infra failure) otherwise — a
    confirm is never fabricated.
    """

    outcome: OnboardingOutcome
    agent_name: str = ""
    experiment_id: str = ""
    cohort: str = ""
    objective_metric: str = ""
    objective_target: float | None = None
    authored_judges: list[str] = Field(default_factory=list)
    registered_code_scorers: list[str] = Field(default_factory=list)
    persisted: bool = False
    #: The confirmed goal serialized to the registry ``goal_config`` shape (the keys
    #: :func:`ail.jobs.continuous_rlm._knobs_from_goal_config` reads). The wizard threads
    #: this onto the ``register_agent`` payload so a requirements-confirmed goal steers
    #: the continuous-RLM lane; ``None`` on any non-success outcome.
    goal_config: dict[str, Any] | None = None
    actor: str = ""
    refused_reason: str | None = None
    error: str | None = None


class ErrorResult(_Contract):
    """A dispatch-level error (unknown/malformed action)."""

    outcome: OnboardingOutcome = OnboardingOutcome.ERROR
    action: str = ""
    error: str


class BootstrapResult(_Contract):
    outcome: OnboardingOutcome
    agent_name: str = ""
    experiment_id: str = ""
    reviewer_experiment_id: str = ""
    annotations_table: str = ""
    experiment_url: str = ""
    tracing_hint: str = ""
    authored_judges: list[str] = Field(default_factory=list)
    goal_config: dict[str, Any] | None = None
    actor: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Pure orchestration (injected clients → unit-testable, no live write)
# ---------------------------------------------------------------------------


def compiled_goal_to_goal_config(goal: CompiledGoal) -> dict[str, Any]:
    """Serialize a :class:`~ail.goals.compiler.CompiledGoal` to the registry ``goal_config``.

    Maps to the exact keys the continuous-RLM lane consumes
    (:func:`ail.jobs.continuous_rlm._knobs_from_goal_config`): ``objective_metric``,
    ``goal_direction``, ``goal_target``, ``goal_target_kind``, and ``guardrail_judge``
    (one ``'name'`` or ``'name:threshold'`` spec per **judge** guardrail — the same
    ``'name:threshold'`` shape :func:`ail.jobs.continuous_rlm._build_rubric` decodes).
    Deterministic-metric guardrails are intentionally **not** serialized: the RLM's
    ``guardrail_judge`` knob is judge-only (a metric guardrail there would be
    reconstructed as ``kind='judge'`` and fail closed). All values are plain JSON
    scalars/lists so :func:`ail.publish_versions.publish_registry` can ``json.dumps``
    the dict onto the ``goal_config_json`` column.

    Pure: no I/O, no side effects — a wizard-confirmed requirements goal flows through
    this onto the ``register_agent`` payload so ``confirm → register → RLM`` steers.
    """
    judge_specs = [
        f"{g.name}:{g.threshold}" if g.threshold is not None else g.name
        for g in goal.guardrails
        if g.kind == "judge"
    ]
    return {
        "objective_metric": goal.objective_metric,
        "goal_direction": str(goal.direction),
        "goal_target": goal.target.value,
        "goal_target_kind": str(goal.target.kind),
        "guardrail_judge": judge_specs,
    }


def register_agent(
    *,
    agent_name: str,
    experiment_id: str,
    goal_keys: list[str],
    actor: str,
    client: Any,
    warehouse_id: str,
    existing_agents: dict[str, str],
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    generated_at: str | None = None,
    goal_config: dict[str, Any] | None = None,
    reviewer_experiment_id: str | None = None,
    annotations_table: str | None = None,
    target_workspace: str | None = None,
) -> RegisterResult:
    """Register one agent by **reusing** :func:`ail.publish_versions.publish_registry`.

    Fail-closed before writing: an empty actor, a blank name/experiment, an unknown
    goal, or a collision (the name is taken, or the experiment is already claimed by
    another agent) all ``REFUSED`` — nothing is written. Otherwise it builds the
    typed :class:`ail.registry.Agent` (its ``judge_config`` is the resolved
    goal→scorer mapping) and hands the registry to ``publish_registry`` (the composite
    ``agent_name`` REPLACE, so other agents are untouched). A write failure inside
    ``publish_registry`` propagates to the caller, which surfaces it as ``ERROR`` —
    never a fabricated ``registered``.

    The optional ``goal_config`` / ``annotations_table`` / ``target_workspace`` are set
    on the :class:`~ail.registry.Agent` verbatim (the write path already persists all
    three columns). They are what make a registered agent fully functional across the
    loop: ``goal_config`` steers the continuous-RLM lane
    (:func:`ail.jobs.continuous_rlm._knobs_from_goal_config`), ``annotations_table`` is
    the OTEL table the memory-distiller job reads, and ``target_workspace`` is the
    repo/path the open-ended executor edits. ``None`` for any of them is a legitimate
    *registered-but-not-fully-functional* state (the fixed-catalog path leaves
    ``goal_config`` ``None`` → RLM neutral; an agent without the other two is skipped by
    the memory job / fails closed in the executor, by design — never fabricated).
    """
    name = agent_name.strip()
    exp = experiment_id.strip()
    if not actor.strip():
        return _register_refused(
            name,
            exp,
            goal_keys,
            actor,
            "refusing an anonymous registration — no authenticated actor identity",
        )
    if not name:
        return _register_refused(name, exp, goal_keys, actor, "an agent name is required")
    if not exp:
        return _register_refused(name, exp, goal_keys, actor, "an experiment id is required")
    if goal_keys:
        try:
            judge_config = build_judge_config(goal_keys)
        except (UnknownGoalError, ValueError) as exc:
            return _register_refused(name, exp, goal_keys, actor, str(exc))
    elif goal_config:
        judge_config = {"source": "free_form_requirements"}
    else:
        return _register_refused(
            name, exp, goal_keys, actor, "select a goal or confirm free-form requirements"
        )

    if name in existing_agents:
        return _register_refused(
            name, exp, goal_keys, actor, f"an agent named {name!r} is already registered"
        )
    owner = next((n for n, e in existing_agents.items() if e == exp), None)
    if owner is not None:
        return _register_refused(
            name,
            exp,
            goal_keys,
            actor,
            f"experiment {exp} is already registered to agent {owner!r} (one agent per experiment)",
        )

    goals = list(dict.fromkeys(goal_keys))
    labels = ", ".join(GOAL_CATALOG[GoalKey(k)].label for k in goals)
    agent = Agent(
        agent_name=name,
        experiment_id=exp,
        reviewer_experiment_id=reviewer_experiment_id,
        description=f"Registered via the onboarding wizard by {actor}; goals: {labels}.",
        judge_config=judge_config,
        goal_config=goal_config,
        annotations_table=annotations_table,
        target_workspace=target_workspace,
    )
    publish_registry(
        AgentRegistry(agents=[agent]),
        client=client,
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
        generated_at=generated_at,
    )
    return RegisterResult(
        outcome=OnboardingOutcome.REGISTERED,
        agent_name=name,
        experiment_id=exp,
        goals=goals,
        judge_config=judge_config,
        actor=actor,
    )


def _register_refused(
    agent_name: str, experiment_id: str, goal_keys: list[str], actor: str, reason: str
) -> RegisterResult:
    return RegisterResult(
        outcome=OnboardingOutcome.REFUSED,
        agent_name=agent_name,
        experiment_id=experiment_id,
        goals=list(goal_keys),
        actor=actor,
        refused_reason=reason,
    )


# ---------------------------------------------------------------------------
# Registry-claim reader (authoritative uniqueness check; tolerant of a fresh table)
# ---------------------------------------------------------------------------


class _RegistryTableMissing(RuntimeError):
    """The ``agent_registry`` table does not exist yet (fresh workspace)."""


def load_registered_agents(
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> dict[str, str]:
    """Read ``agent_name -> experiment_id`` from the ``agent_registry`` UC table.

    The authoritative source for both the freshness "already-claimed" check and the
    registration collision check. Tolerant of a **not-yet-created** table (a fresh
    workspace has none until the first publish) — that is an empty registry, not an
    error. Any *other* failure (permission, warehouse down) propagates so the caller
    fails closed rather than treating "cannot read" as "no agents".
    """
    fqn = f"`{catalog}`.`{schema}`.{REGISTRY_TABLE}"
    stmt = f"SELECT agent_name, experiment_id FROM {fqn}"
    try:
        rows = _query_rows(client, warehouse_id, stmt)
    except _RegistryTableMissing:
        return {}
    out: dict[str, str] = {}
    for row in rows:
        name = row.get("agent_name")
        exp = row.get("experiment_id")
        if name:
            out[str(name)] = "" if exp is None else str(exp)
    return out


def _query_rows(client: Any, warehouse_id: str, statement: str) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as ``{column: value}`` dicts (all as strings).

    Mirrors the statement-execution wait loop used elsewhere in the framework
    (:mod:`ail.publish`). A "table/view not found" failure is raised as
    :class:`_RegistryTableMissing` (a fresh workspace, not a real error); any other
    non-success is a hard :class:`RuntimeError`.
    """
    import time

    from databricks.sdk.service.sql import StatementState

    resp = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
    )
    statement_id = resp.statement_id
    state = resp.status.state if resp.status else None
    while state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1.0)
        resp = client.statement_execution.get_statement(statement_id)
        state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        detail = ""
        if resp.status and resp.status.error:
            detail = resp.status.error.message or ""
        low = detail.lower()
        if "table_or_view_not_found" in low or "does not exist" in low or "cannot be found" in low:
            raise _RegistryTableMissing(detail)
        raise RuntimeError(f"statement {state}: {detail}\nSQL head: {statement[:200]}")

    manifest = resp.manifest
    columns = [c.name for c in manifest.schema.columns] if manifest and manifest.schema else []
    data = resp.result.data_array if resp.result and resp.result.data_array else []
    return [dict(zip(columns, row, strict=False)) for row in data]


# ---------------------------------------------------------------------------
# Live wiring — build seams, call the pure orchestration, fail-closed
# ---------------------------------------------------------------------------


def run_requirements(goal_keys: list[str] | None = None) -> RequirementsResult | ErrorResult:
    """The goal catalog + a selection's data-gate requirements (no workspace call)."""
    try:
        return build_requirements(goal_keys)
    except UnknownGoalError as exc:
        return ErrorResult(action="requirements", error=str(exc))


#: The requirements-engine failures that are honest, expected user-facing outcomes
#: (a garbage blob, a mis-mapped metric, a wrong-sign target) rather than infra
#: bugs — surfaced as an ERROR/ErrorResult carrying the engine's own message,
#: never a fabricated plan/confirm.
_REQUIREMENTS_ENGINE_ERRORS = (
    RequirementsExtractionError,
    RequirementsRoutingError,
    RequirementsNotConfirmedError,
    GoalCompileError,
    GoalContractError,
    ValueError,
)


def _preview_from_plan(
    plan: RequirementsPlan,
    *,
    requirements_text: str,
    cohort: str,
    agent_name: str,
    actor: str,
) -> RequirementsPreviewResult:
    """Project a (proposal) :class:`RequirementsPlan` into the JSON preview contract."""
    dimensions = [
        PreviewedDimension(
            name=d.name,
            description=d.description,
            user_priority=d.user_priority,
            kind=d.kind,
            role=d.role,
            metric=d.metric,
            judge_name=d.judge_name,
            direction=d.direction,
        )
        for d in plan.dimensions
    ]
    target = plan.goal.target
    return RequirementsPreviewResult(
        requirements_text=requirements_text,
        cohort=cohort,
        agent_name=agent_name,
        describe=plan.describe(),
        objective_metric=plan.goal.objective_metric,
        direction=plan.goal.direction,
        requires_quality=plan.goal.requires_quality,
        dimensions=dimensions,
        judges_to_author=[d.judge_name for d in plan.judges_to_author if d.judge_name],
        deterministic_metrics=[d.metric for d in plan.deterministic_metrics if d.metric],
        suggested_target=SuggestedTarget(value=target.value, kind=target.kind),
        actor=actor,
    )


def run_preview_requirements(
    requirements_text: str,
    *,
    cohort: str = "",
    agent_name: str = "",
    actor: str = "",
    llm: GoalProposerLLM | None = None,
    known_judges: Iterable[str] = (),
) -> RequirementsPreviewResult | ErrorResult:
    """PREVIEW a free-form requirements blob — extract + route + compose, no side effects.

    Runs the Slice-1 intake engine (:func:`ail.requirements.plan_requirements`) with
    the SAME LLM seam :func:`ail.goals.compile_goal` uses (``llm=None`` lazily builds
    the Databricks proposer; tests inject a mock) and returns a structured,
    JSON-serializable preview of the routed plan for human review. A pure proposal —
    it authors **no** judge and persists **nothing**.

    Fail-closed: a blank blob, an extraction/routing/compile failure, or an
    unconfigured LLM endpoint is an honest :class:`ErrorResult`, never a fabricated
    plan.
    """
    text = requirements_text.strip() if requirements_text else ""
    if not text:
        return ErrorResult(
            action="preview_requirements", error="requirements text must be a non-empty string"
        )
    resolved_cohort = cohort.strip() or agent_name.strip()
    if not resolved_cohort:
        return ErrorResult(
            action="preview_requirements",
            error="a cohort or agent_name is required to compose the plan",
        )
    try:
        proposer = llm if llm is not None else _default_databricks_proposer()
        plan = plan_requirements(text, resolved_cohort, llm=proposer, known_judges=known_judges)
    except _REQUIREMENTS_ENGINE_ERRORS as exc:
        return ErrorResult(action="preview_requirements", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a live LLM/infra failure is an honest error, never a fake plan
        return ErrorResult(action="preview_requirements", error=f"{type(exc).__name__}: {exc}")
    return _preview_from_plan(
        plan,
        requirements_text=text,
        cohort=resolved_cohort,
        agent_name=agent_name.strip(),
        actor=actor,
    )


def run_confirm_requirements(
    requirements_text: str,
    *,
    objective_target: float | None,
    experiment_id: str,
    agent_name: str,
    cohort: str = "",
    actor: str = "",
    profile: str | None = None,
    warehouse_id: str | None = None,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    llm: GoalProposerLLM | None = None,
    author: JudgeAuthor | None = None,
    persist: Callable[[CompiledGoal], None] | None = None,
    deterministic_registrar: Callable[[str, list[str]], list[str]] | None = None,
    known_judges: Iterable[str] = (),
) -> RequirementsConfirmResult:
    """CONFIRM a free-form requirements intake — author judges + persist the goal.

    Re-derives the SAME plan from ``requirements_text`` (deterministic given the same
    text + LLM), applies the human's explicit ``objective_target`` via
    :meth:`RequirementsPlan.confirm`, then **reuses** :func:`ail.requirements.execute_plan`
    to author one ``{{trace}}`` judge per quality dimension (via the existing
    :func:`ail.judges.author_judge` path — ``author=None`` uses it), register one
    self-contained MLflow custom code scorer per deterministic dimension, and
    persist the confirmed goal (``persist=None`` builds the per-agent
    :func:`ail.requirements.compiled_goal_persister`; tests inject spies).

    Fail-closed:

    * an anonymous actor or a missing ``objective_target`` is a ``refused`` (the
      human must set/acknowledge the target the wizard pre-filled from the suggestion);
    * an extraction/routing/wrong-sign-target failure is an honest ``error``;
    * ``execute_plan`` itself refuses unless the plan is confirmed AND
      ``goal.human_confirmed`` — so nothing is authored/persisted for an unconfirmed
      plan, and any infra failure is an ``error``, never a fabricated confirm.
    """
    resolved_cohort = cohort.strip() or agent_name.strip()

    def _fail(
        outcome: OnboardingOutcome, *, error: str | None = None, refused: str | None = None
    ) -> RequirementsConfirmResult:
        # Never echo a non-finite target (NaN/inf) back — it is invalid JSON and would
        # misrepresent what was refused; a rejected target reads back as "no target".
        echoed = (
            objective_target
            if objective_target is not None and math.isfinite(objective_target)
            else None
        )
        return RequirementsConfirmResult(
            outcome=outcome,
            agent_name=agent_name.strip(),
            experiment_id=experiment_id.strip(),
            cohort=resolved_cohort,
            objective_target=echoed,
            actor=actor,
            error=error,
            refused_reason=refused,
        )

    if not actor.strip():
        return _fail(
            OnboardingOutcome.REFUSED,
            refused="refusing an anonymous confirmation — no authenticated actor identity",
        )
    # Fail closed on a missing OR non-finite target (defense-in-depth: the service
    # cannot assume its only caller is the TS route). JSON admits NaN/Infinity, and
    # the CompiledGoal sign checks (value < 0 / value > 0) are BOTH False for NaN, so
    # a non-finite target would otherwise slip past plan.confirm()/execute_plan() and
    # author judges + persist a goal carrying a meaningless target. Refuse, never coerce.
    if objective_target is None or not math.isfinite(objective_target):
        return _fail(
            OnboardingOutcome.REFUSED,
            refused=(
                "a finite, explicit objective target is required to confirm — set or acknowledge "
                "the suggested target before confirming (propose-then-confirm); NaN/inf are refused"
            ),
        )
    text = requirements_text.strip() if requirements_text else ""
    if not text:
        return _fail(OnboardingOutcome.ERROR, error="requirements text must be a non-empty string")
    if not experiment_id.strip():
        return _fail(OnboardingOutcome.ERROR, error="an experiment id is required to author judges")
    if not agent_name.strip():
        return _fail(OnboardingOutcome.ERROR, error="an agent name is required to persist the goal")

    # Resolve the persist seam. Tests inject `persist`; the live path builds the
    # per-agent persister, which needs a SQL warehouse to write agent_compiled_goals
    # (mirrors run_register's warehouse resolution). Judges are authored via the live
    # author_judge path (author=None) exactly as execute_plan already defaults.
    persist_fn = persist
    if persist_fn is None:
        wh = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
        if not wh:
            return _fail(
                OnboardingOutcome.ERROR,
                error="no SQL warehouse id (set DATABRICKS_WAREHOUSE_ID) — cannot persist the goal",
            )
        try:
            from ail.publish import _build_workspace_client

            client = _build_workspace_client(profile)
            persist_fn = compiled_goal_persister(
                agent_name=agent_name.strip(),
                client=client,
                warehouse_id=wh,
                catalog=catalog,
                schema=schema,
                requirements_text=text,
            )
        except Exception as exc:  # noqa: BLE001 - client build failure is an honest ERROR
            return _fail(OnboardingOutcome.ERROR, error=f"{type(exc).__name__}: {exc}")

    try:
        proposer = llm if llm is not None else _default_databricks_proposer()
        plan = plan_requirements(text, resolved_cohort, llm=proposer, known_judges=known_judges)
        confirmed = plan.confirm(objective_target=objective_target)
        metric_names = [d.metric for d in confirmed.deterministic_metrics if d.metric]
        if deterministic_registrar is None:
            from ail.metrics.mlflow_scorers import register_deterministic_scorers

            registered_code_scorers = register_deterministic_scorers(
                experiment_id.strip(),
                metric_names,
                profile=profile,
                warehouse_id=warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID"),
            )
        else:
            registered_code_scorers = deterministic_registrar(experiment_id.strip(), metric_names)
        execution = execute_plan(
            confirmed, experiment_id=experiment_id.strip(), author=author, persist=persist_fn
        )
    except _REQUIREMENTS_ENGINE_ERRORS as exc:
        return _fail(OnboardingOutcome.ERROR, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - any infra failure is an honest ERROR, never a fake confirm
        return _fail(OnboardingOutcome.ERROR, error=f"{type(exc).__name__}: {exc}")

    return RequirementsConfirmResult(
        outcome=OnboardingOutcome.REQUIREMENTS_CONFIRMED,
        agent_name=agent_name.strip(),
        experiment_id=experiment_id.strip(),
        cohort=resolved_cohort,
        objective_metric=execution.goal.objective_metric,
        objective_target=objective_target,
        authored_judges=list(execution.authored_names),
        registered_code_scorers=registered_code_scorers,
        persisted=execution.persisted,
        # Surface the confirmed goal in the registry goal_config shape so the wizard can
        # thread it onto the register_agent payload (confirm → register → RLM steers).
        goal_config=compiled_goal_to_goal_config(execution.goal),
        actor=actor,
    )


def run_validate(
    experiment_id: str,
    *,
    actor: str = "",
    profile: str | None = None,
    warehouse_id: str | None = None,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    experiment_client: ExperimentClient | None = None,
    claimed_experiment_ids: dict[str, str] | None = None,
) -> ValidationResult:
    """Validate freshness live (fail-closed: an access error is ERROR, never fresh)."""
    exp = experiment_id.strip()
    if not exp:
        return ValidationResult(
            outcome=OnboardingOutcome.ERROR,
            experiment_id="",
            actor=actor,
            error="an experiment id is required to validate freshness",
        )
    try:
        client = experiment_client or build_experiment_client(profile)
        # A 'fresh' verdict requires BOTH the MLflow-traces emptiness check AND the
        # registry-claims check to have actually RUN. When claims are injected
        # (tests / a caller that already resolved them) that check has run; on the
        # live path it needs a warehouse to read agent_registry. If it CANNOT run —
        # no warehouse configured, or a registry-read permission/authority error —
        # we must NOT assume "no claims" and report fresh: that would fabricate an
        # unverified 'not claimed' state. Return an honest undetermined error naming
        # the prerequisite instead (fail-closed, no fabricated freshness).
        claimed = claimed_experiment_ids
        if claimed is None:
            wh = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
            if not wh:
                return ValidationResult(
                    outcome=OnboardingOutcome.ERROR,
                    experiment_id=exp,
                    actor=actor,
                    error=(
                        "cannot verify the experiment is unclaimed — no SQL warehouse "
                        "configured to read the agent_registry. Set DATABRICKS_WAREHOUSE_ID "
                        "(and ensure the app SP can read agent_registry) so the registry-claim "
                        "check can run; refusing to report a freshness verdict it could not verify"
                    ),
                )
            # _claimed_experiments raises (RuntimeError / ExperimentAccessError) on a
            # registry-read failure — caught below and surfaced as an honest ERROR,
            # never a fresh verdict. A genuinely-absent table is an empty registry
            # (the check ran and found no claims), which is a legitimate 'not claimed'.
            claimed = _claimed_experiments(wh, profile, catalog, schema)
        validation = validate_experiment_probe(exp, client=client, claimed_experiment_ids=claimed)
    except (ExperimentAccessError, RuntimeError) as exc:
        return ValidationResult(
            outcome=OnboardingOutcome.ERROR, experiment_id=exp, actor=actor, error=str(exc)
        )
    return ValidationResult(
        outcome=OnboardingOutcome.VALIDATED,
        experiment_id=validation.experiment_id,
        name=validation.name,
        exists=validation.exists,
        fresh=validation.fresh,
        trace_count=validation.trace_count,
        trace_count_capped=validation.trace_count_capped,
        already_registered=validation.already_registered,
        registered_as=validation.registered_as,
        reasons=validation.reasons,
        actor=actor,
    )


def run_create(
    name: str,
    *,
    actor: str,
    profile: str | None = None,
    experiment_client: ExperimentClient | None = None,
    trace_catalog: str | None = None,
    trace_schema: str | None = None,
    trace_table_prefix: str | None = None,
    allow_existing: bool = False,
) -> CreationResult:
    """Create a fresh experiment live (fail-closed; permission error is honest ERROR)."""
    if not actor.strip():
        return CreationResult(
            outcome=OnboardingOutcome.REFUSED,
            name=name,
            actor=actor,
            error="refusing an anonymous experiment creation — no authenticated actor identity",
        )
    clean = name.strip()
    if not clean:
        return CreationResult(
            outcome=OnboardingOutcome.ERROR,
            actor=actor,
            error="an experiment name is required to create one",
        )
    try:
        client = experiment_client or build_experiment_client(profile)
        location = None
        if trace_catalog and trace_schema:
            prefix = _trace_table_prefix(trace_table_prefix or clean)
            location = UcTraceLocation(trace_catalog, trace_schema, prefix)
        creation = create_experiment_probe(
            clean,
            client=client,
            trace_location=location,
            allow_existing=allow_existing,
        )
    except ExperimentPermissionError as exc:
        return CreationResult(
            outcome=OnboardingOutcome.ERROR,
            name=clean,
            actor=actor,
            error=str(exc),
            prerequisite="app service principal needs experiment-create authority in the workspace",
        )
    except (ExperimentAccessError, ValueError) as exc:
        return CreationResult(
            outcome=OnboardingOutcome.ERROR, name=clean, actor=actor, error=str(exc)
        )
    # The create SUCCEEDED — hand it back ready to use. The URL is a convenience:
    # fail-soft to '' if the host can't be resolved (never fail a real creation for it).
    return CreationResult(
        outcome=OnboardingOutcome.CREATED,
        experiment_id=creation.experiment_id,
        name=creation.name,
        experiment_url=_experiment_url(client, creation.experiment_id),
        tracing_hint=_tracing_hint(creation.experiment_id),
        annotations_table=location.annotations_table if location else "",
        actor=actor,
    )


def _ensure_bootstrap_experiment(
    name: str,
    *,
    actor: str,
    profile: str | None,
    catalog: str,
    trace_schema: str,
    table_prefix: str,
) -> CreationResult:
    """Create the deterministic Quick Connect experiment, or resume a partial attempt."""
    client = build_experiment_client(profile)
    clean = name.strip()
    target = clean
    if not clean.startswith("/"):
        home = client.workspace_home()
        if not home:
            return CreationResult(
                outcome=OnboardingOutcome.ERROR,
                name=clean,
                actor=actor,
                error="could not resolve the workspace home for Quick Connect",
            )
        target = f"{home.rstrip('/')}/{clean.strip('/')}"
    try:
        existing = client.get_experiment_by_name(target)
    except ExperimentAccessError as exc:
        return CreationResult(
            outcome=OnboardingOutcome.ERROR, name=target, actor=actor, error=str(exc)
        )
    location = UcTraceLocation(catalog, trace_schema, table_prefix)
    if existing is not None:
        return CreationResult(
            outcome=OnboardingOutcome.CREATED,
            experiment_id=existing.experiment_id,
            name=existing.name,
            experiment_url=_experiment_url(client, existing.experiment_id),
            tracing_hint=_tracing_hint(existing.experiment_id),
            annotations_table=location.annotations_table,
            actor=actor,
        )
    return run_create(
        target,
        actor=actor,
        profile=profile,
        experiment_client=client,
        trace_catalog=catalog,
        trace_schema=trace_schema,
        trace_table_prefix=table_prefix,
    )


def _ensure_baseline_judges(
    experiment_id: str, *, profile: str | None, warehouse_id: str | None = None
) -> list[str]:
    """Ensure the full trace-native baseline judge suite and matching label targets."""
    import mlflow
    from mlflow.genai.label_schemas import (
        InputCategorical,
        InputNumeric,
        create_label_schema,
        get_label_schema,
    )
    from mlflow.utils.databricks_utils import is_databricks_uri

    from ail.judges.registration import (
        _configure_databricks,
        create_aligned_scorer,
        list_registered_scorers,
    )
    from ail.judges.scorers import DEFAULT_SCORERS

    _configure_databricks(profile=profile, tracking_uri="databricks", registry_uri="databricks-uc")
    mlflow.set_experiment(experiment_id=experiment_id)
    if warehouse_id:
        mlflow.set_experiment_tag("mlflow.monitoring.sqlWarehouseId", warehouse_id)
    schema_scope = (
        {} if is_databricks_uri(mlflow.get_tracking_uri()) else {"experiment_id": experiment_id}
    )
    existing = {scorer.name for scorer in list_registered_scorers(experiment_id, profile=profile)}
    ensured: list[str] = []
    for name, spec in DEFAULT_SCORERS.items():
        try:
            get_label_schema(name, **schema_scope)
        except Exception:  # noqa: BLE001 - absent schemas are created below
            label_input: Any
            if name in {"correctness", "groundedness"}:
                label_input = InputCategorical(options=["yes", "no"])
            else:
                label_input = InputNumeric(min_value=1, max_value=5)
            create_label_schema(
                name=name,
                type="feedback",
                input=label_input,
                instruction=(
                    f"Review this trace for {name}. Record the verdict and cite the decisive "
                    "request, tool result, or final-response evidence in the comment."
                ),
                enable_comment=True,
                title=name.replace("_", " "),
                **schema_scope,
            )
        if name not in existing:
            create_aligned_scorer(
                spec,
                experiment_id=experiment_id,
                sampling_rate=1.0,
                profile=profile,
            )
        ensured.append(name)
    return ensured


def run_bootstrap(
    *,
    agent_name: str,
    requirements_text: str,
    actor: str,
    target_workspace: str,
    profile: str | None,
    warehouse_id: str | None,
    catalog: str,
    schema: str,
    trace_schema: str,
) -> BootstrapResult:
    name = agent_name.strip()
    workspace = target_workspace.strip()
    if not name or not requirements_text.strip() or not actor.strip() or not workspace:
        return BootstrapResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=name,
            actor=actor,
            error=(
                "agent_name, requirements_text, authenticated actor, and a local companion "
                "target_workspace are required"
            ),
        )
    prefix = _trace_table_prefix(name)
    subject = _ensure_bootstrap_experiment(
        f"{name}-traces",
        actor=actor,
        profile=profile,
        catalog=catalog,
        trace_schema=trace_schema,
        table_prefix=prefix,
    )
    if subject.outcome != OnboardingOutcome.CREATED:
        return BootstrapResult(
            outcome=OnboardingOutcome.ERROR, agent_name=name, actor=actor, error=subject.error
        )
    reviewer = _ensure_bootstrap_experiment(
        f"{name}-ail-internal",
        actor=actor,
        profile=profile,
        catalog=catalog,
        trace_schema=trace_schema,
        table_prefix=f"{prefix}_internal",
    )
    if reviewer.outcome != OnboardingOutcome.CREATED:
        return BootstrapResult(
            outcome=OnboardingOutcome.ERROR, agent_name=name, actor=actor, error=reviewer.error
        )
    preview = run_preview_requirements(requirements_text, cohort=name, agent_name=name, actor=actor)
    if not isinstance(preview, RequirementsPreviewResult) or preview.suggested_target is None:
        return BootstrapResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=name,
            actor=actor,
            error=getattr(preview, "error", "requirements preview did not produce a target"),
        )
    confirmed = run_confirm_requirements(
        requirements_text,
        objective_target=preview.suggested_target.value,
        experiment_id=subject.experiment_id,
        agent_name=name,
        cohort=name,
        actor=actor,
        profile=profile,
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
    )
    if confirmed.outcome != OnboardingOutcome.REQUIREMENTS_CONFIRMED:
        return BootstrapResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=name,
            actor=actor,
            error=confirmed.error or confirmed.refused_reason,
        )
    try:
        baseline_judges = _ensure_baseline_judges(
            subject.experiment_id, profile=profile, warehouse_id=warehouse_id
        )
    except Exception as exc:  # noqa: BLE001 - provisioning failure is an honest bootstrap error
        return BootstrapResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=name,
            actor=actor,
            error=f"could not provision baseline judges: {type(exc).__name__}: {exc}",
        )
    metric_goal = {
        "total_tokens": "token_efficiency",
        "duration_seconds": "latency",
        "total_usd": "cost",
        "correctness": "accuracy",
    }.get(confirmed.objective_metric, "accuracy")
    registered = run_register(
        agent_name=name,
        experiment_id=subject.experiment_id,
        reviewer_experiment_id=reviewer.experiment_id,
        goal_keys=[metric_goal],
        actor=actor,
        profile=profile,
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
        goal_config=confirmed.goal_config,
        annotations_table=subject.annotations_table,
        target_workspace=workspace,
    )
    if registered.outcome != OnboardingOutcome.REGISTERED:
        return BootstrapResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=name,
            actor=actor,
            error=registered.error or registered.refused_reason,
        )
    return BootstrapResult(
        outcome=OnboardingOutcome.REGISTERED,
        agent_name=name,
        experiment_id=subject.experiment_id,
        reviewer_experiment_id=reviewer.experiment_id,
        annotations_table=subject.annotations_table,
        experiment_url=subject.experiment_url,
        tracing_hint=subject.tracing_hint,
        authored_judges=list(dict.fromkeys([*confirmed.authored_judges, *baseline_judges])),
        goal_config=confirmed.goal_config,
        actor=actor,
    )


def _trace_table_prefix(value: str) -> str:
    leaf = value.rstrip("/").rsplit("/", 1)[-1]
    prefix = re.sub(r"[^a-z0-9_]+", "_", leaf.lower().replace("-", "_"))
    prefix = prefix.strip("_")[:48]
    if not prefix:
        raise ValueError("a trace table prefix must contain at least one letter or number")
    if prefix[0].isdigit():
        prefix = f"agent_{prefix}"
    return prefix


def _experiment_url(client: ExperimentClient, experiment_id: str) -> str:
    """Compose ``{host}/ml/experiments/{id}`` (host resolved LIVE; ``""`` if unresolvable).

    Reuses the ``WorkspaceClient(profile).config.host`` pattern (behind the injectable
    :meth:`~ail.onboarding.experiment.ExperimentClient.workspace_host` seam so it stays
    offline-testable). Fail-soft: any resolution failure yields ``""`` — the URL is a
    convenience surfaced to the user, never a reason to fail a creation that succeeded.
    """
    try:
        host = client.workspace_host()
    except Exception:  # noqa: BLE001 - the URL is a convenience; never crash the create
        return ""
    host = (host or "").rstrip("/")
    return f"{host}/ml/experiments/{experiment_id}" if host else ""


def _tracing_hint(experiment_id: str) -> str:
    """A copy-paste snippet pointing an agent's MLflow tracing at ``experiment_id``."""
    return (
        f"mlflow.set_experiment(experiment_id='{experiment_id}')  "
        "# then enable autolog (e.g. mlflow.langchain.autolog()) to route traces here"
    )


def run_register(
    *,
    agent_name: str,
    experiment_id: str,
    goal_keys: list[str],
    actor: str,
    profile: str | None = None,
    warehouse_id: str | None = None,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    goal_config: dict[str, Any] | None = None,
    reviewer_experiment_id: str | None = None,
    annotations_table: str | None = None,
    target_workspace: str | None = None,
) -> RegisterResult:
    """Register an agent live (fail-closed; a write failure is ERROR, never registered).

    Threads the optional extended registry fields (``goal_config`` /
    ``annotations_table`` / ``target_workspace``) to :func:`register_agent`, which sets
    them on the persisted :class:`~ail.registry.Agent`. Type validation of these fields
    happens in :func:`run_action` (the payload boundary); here they are already typed.
    """
    if not actor.strip():
        return _register_refused(
            agent_name,
            experiment_id,
            goal_keys,
            actor,
            "refusing an anonymous registration — no authenticated actor identity",
        )
    wh = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not wh:
        return RegisterResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=agent_name,
            experiment_id=experiment_id,
            goals=list(goal_keys),
            actor=actor,
            error="no SQL warehouse id (set DATABRICKS_WAREHOUSE_ID) — cannot write the registry",
        )
    try:
        from ail.publish import _build_workspace_client

        client = _build_workspace_client(profile)
        existing = load_registered_agents(
            client=client, warehouse_id=wh, catalog=catalog, schema=schema
        )
        from ail.goals.allowlist import is_l0_metric
        from ail.metrics.mlflow_scorers import register_deterministic_scorers

        deterministic_metrics: list[str] = []
        for key in goal_keys:
            try:
                objective = GOAL_CATALOG[GoalKey(key)].objective_metric
            except (KeyError, ValueError):
                continue
            if is_l0_metric(objective):
                deterministic_metrics.append(objective)
        configured_objective = (goal_config or {}).get("objective_metric")
        if isinstance(configured_objective, str) and is_l0_metric(configured_objective):
            deterministic_metrics.append(configured_objective)
        registered_code_scorers = register_deterministic_scorers(
            experiment_id.strip(),
            deterministic_metrics,
            profile=profile,
            warehouse_id=wh,
        )
        _ensure_baseline_judges(experiment_id.strip(), profile=profile, warehouse_id=wh)
        result = register_agent(
            agent_name=agent_name,
            experiment_id=experiment_id,
            goal_keys=goal_keys,
            actor=actor,
            client=client,
            warehouse_id=wh,
            existing_agents=existing,
            catalog=catalog,
            schema=schema,
            goal_config=goal_config,
            reviewer_experiment_id=reviewer_experiment_id,
            annotations_table=annotations_table,
            target_workspace=target_workspace,
        )
        result = result.model_copy(update={"registered_code_scorers": registered_code_scorers})
        if result.outcome is OnboardingOutcome.REGISTERED:
            # Add this agent's own *_otel_spans table to the arrival-triggered RLM
            # job's watched-table list so the job WAKES on the new agent's traces
            # (the job body already reviews every registered agent; only its trigger
            # is table-scoped). Best-effort: a reconcile failure must NOT undo a
            # successful registration — the four cron jobs already cover the agent,
            # and the deploy-time heal (ail.jobs.bootstrap_grants) re-reconciles the
            # whole registry on the next bundle deploy. See ail.jobs.rlm_trigger.
            _reconcile_rlm_trigger_after_register(
                client,
                agent_name=agent_name,
                experiment_id=experiment_id,
                annotations_table=annotations_table,
            )
        return result
    except Exception as exc:  # noqa: BLE001 - any infra failure is an honest ERROR, never a fake register
        return RegisterResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=agent_name,
            experiment_id=experiment_id,
            goals=list(goal_keys),
            actor=actor,
            error=f"{type(exc).__name__}: {exc}",
        )


def _reconcile_rlm_trigger_after_register(
    client: Any,
    *,
    agent_name: str,
    experiment_id: str,
    annotations_table: str | None,
) -> None:
    """Best-effort: add the agent's spans table to arrival-driven job triggers.

    Reads the RLM and judge-backfill job ids wired by the onboarding resource.
    Absent/blank/non-numeric ids skip quietly. Any Jobs API failure is logged and
    swallowed because the daily recovery sweep still covers durable registrations.
    """
    from ail.jobs.rlm_trigger import reconcile_rlm_trigger_tables

    agent = Agent(
        agent_name=agent_name,
        experiment_id=experiment_id,
        annotations_table=annotations_table,
    )
    trigger_jobs = (
        ("RLM", "AIL_RLM_JOB_ID"),
        ("judge backfill", "AIL_JUDGE_BACKFILL_JOB_ID"),
    )
    for label, env_name in trigger_jobs:
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        try:
            job_id = int(raw)
        except ValueError:
            print(
                f"[ail.onboarding] {env_name}={raw!r} is not an int; "
                f"skipping {label} trigger reconcile (agent is registered).",
                file=sys.stderr,
            )
            continue
        try:
            result = reconcile_rlm_trigger_tables(client, rlm_job_id=job_id, agents=[agent])
        except Exception as exc:  # noqa: BLE001 - reconcile is best-effort
            print(
                f"[ail.onboarding] {label} trigger reconcile failed for agent={agent_name} "
                "(agent IS registered; daily recovery still covers it): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        if result.updated:
            print(
                f"[ail.onboarding] {label} trigger now watches {', '.join(result.added)} "
                f"for agent={agent_name}"
            )
        elif result.underivable:
            print(
                f"[ail.onboarding] agent={agent_name} has no derivable spans table "
                f"(no annotations_table); {label} trigger unchanged.",
                file=sys.stderr,
            )


def _claimed_experiments(
    warehouse_id: str, profile: str | None, catalog: str, schema: str
) -> dict[str, str]:
    """``experiment_id -> owning agent_name`` for the freshness "already-claimed" check."""
    from ail.publish import _build_workspace_client

    client = _build_workspace_client(profile)
    by_agent = load_registered_agents(
        client=client, warehouse_id=warehouse_id, catalog=catalog, schema=schema
    )
    return {exp: name for name, exp in by_agent.items() if exp}


# ---------------------------------------------------------------------------
# Dispatch + CLI (the bridge invokes `python -m ail.onboarding.service`)
# ---------------------------------------------------------------------------


def _coerce_goal_config(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Coerce a payload ``goal_config`` (dict | JSON-object string | absent) → dict|None.

    Returns ``(value, error)``: ``error`` is a fail-closed refusal reason for anything
    that is neither absent/``None``, a JSON object, nor a JSON string that parses to an
    object — never a silently-dropped field or a crash. A blank string is treated as
    absent (``None``). A ``bool``/scalar/array is refused (it is not a goal mapping).
    """
    if raw is None:
        return None, None
    if isinstance(raw, bool):  # a bool is an int; it is never a goal mapping
        return None, "goal_config must be a JSON object (mapping), not a boolean"
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, None
        try:
            parsed = json.loads(text)
        except ValueError:
            return None, "goal_config must be a JSON object or a JSON-object string"
        if not isinstance(parsed, dict):
            return None, "goal_config must be a JSON object (mapping), not a scalar/array"
        return parsed, None
    return None, (
        f"goal_config must be a JSON object (mapping) or string, not a {type(raw).__name__}"
    )


def _coerce_optional_str(raw: Any, field: str) -> tuple[str | None, str | None]:
    """Coerce an optional string payload field → str|None, fail-closed on a non-string.

    Returns ``(value, error)``: a blank string is treated as absent (``None``, i.e. the
    field was left unset); a non-string value is a fail-closed refusal reason, never a
    coerced/dropped value. A ``bool`` is a non-string and is refused.
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        text = raw.strip()
        return (text or None), None
    return None, f"{field} must be a string (a fully-qualified name), not a {type(raw).__name__}"


def run_action(payload: dict[str, Any]) -> BaseModel:
    """Dispatch one JSON action to its live handler; unknown/malformed → ERROR."""
    action = str(payload.get("action") or "")
    actor = str(payload.get("actor") or "")
    profile = payload.get("profile")
    warehouse_id = payload.get("warehouse_id")
    catalog = str(payload.get("catalog") or DEFAULT_CATALOG)
    schema = str(payload.get("schema") or DEFAULT_SCHEMA)
    goals = payload.get("goals")
    goal_keys = [str(g) for g in goals] if isinstance(goals, list) else []

    if action == "requirements":
        return run_requirements(goal_keys or None)
    if action == "bootstrap_agent":
        return run_bootstrap(
            agent_name=str(payload.get("agent_name") or ""),
            requirements_text=str(payload.get("requirements_text") or ""),
            actor=actor,
            target_workspace=str(payload.get("target_workspace") or ""),
            profile=profile,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
            trace_schema=str(payload.get("trace_schema") or "mlflow_traces"),
        )
    if action == "preview_requirements":
        return run_preview_requirements(
            str(payload.get("requirements_text") or ""),
            cohort=str(payload.get("cohort") or ""),
            agent_name=str(payload.get("agent_name") or ""),
            actor=actor,
        )
    if action == "confirm_requirements":
        raw_target = payload.get("objective_target")
        # Coerce a JSON numeric to a finite float. A bool is an int subclass — reject it
        # so a stray `true` is never a 1.0 target. json.loads also admits NaN/Infinity/
        # -Infinity and a huge integer overflows float; all of those become None here so
        # confirm_requirements refuses honestly rather than crashing or letting a
        # non-finite target slip past the sign checks (NaN < 0 and NaN > 0 are both False).
        target: float | None = None
        if isinstance(raw_target, (int, float)) and not isinstance(raw_target, bool):
            try:
                coerced = float(raw_target)
            except (OverflowError, ValueError):
                coerced = None
            if coerced is not None and math.isfinite(coerced):
                target = coerced
        return run_confirm_requirements(
            str(payload.get("requirements_text") or ""),
            objective_target=target,
            experiment_id=str(payload.get("experiment_id") or ""),
            agent_name=str(payload.get("agent_name") or ""),
            cohort=str(payload.get("cohort") or ""),
            actor=actor,
            profile=profile,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
        )
    if action == "validate_experiment":
        return run_validate(
            str(payload.get("experiment_id") or ""),
            actor=actor,
            profile=profile,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
        )
    if action == "create_experiment":
        return run_create(
            str(payload.get("name") or ""),
            actor=actor,
            profile=profile,
            trace_catalog=str(payload.get("trace_catalog") or "") or None,
            trace_schema=str(payload.get("trace_schema") or "") or None,
            trace_table_prefix=str(payload.get("trace_table_prefix") or "") or None,
            allow_existing=payload.get("allow_existing") is True,
        )
    if action == "register_agent":
        name = str(payload.get("agent_name") or "")
        exp = str(payload.get("experiment_id") or "")
        # Parse the extended registry fields fail-closed BEFORE any live write: a bad
        # type is an honest REFUSED (nothing written), never a crash or a silently
        # dropped field. goal_config accepts a JSON object OR a JSON-object string
        # (the browser may relay it either way); annotations_table/target_workspace
        # must be strings. The actor is the authenticated identity the route injected.
        goal_config, gc_err = _coerce_goal_config(payload.get("goal_config"))
        if gc_err is not None:
            return _register_refused(name, exp, goal_keys, actor, gc_err)
        annotations_table, at_err = _coerce_optional_str(
            payload.get("annotations_table"), "annotations_table"
        )
        if at_err is not None:
            return _register_refused(name, exp, goal_keys, actor, at_err)
        target_workspace, tw_err = _coerce_optional_str(
            payload.get("target_workspace"), "target_workspace"
        )
        if tw_err is not None:
            return _register_refused(name, exp, goal_keys, actor, tw_err)
        reviewer_experiment_id, re_err = _coerce_optional_str(
            payload.get("reviewer_experiment_id"), "reviewer_experiment_id"
        )
        if re_err is not None:
            return _register_refused(name, exp, goal_keys, actor, re_err)
        return run_register(
            agent_name=name,
            experiment_id=exp,
            goal_keys=goal_keys,
            actor=actor,
            profile=profile,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
            goal_config=goal_config,
            reviewer_experiment_id=reviewer_experiment_id,
            annotations_table=annotations_table,
            target_workspace=target_workspace,
        )
    return ErrorResult(action=action, error=f"unknown onboarding action {action!r}")


def main(argv: list[str] | None = None) -> int:
    """CLI bridge: read a JSON action on stdin, print a JSON result on stdout.

    The Node/AppKit onboarding route (which authenticates the actor and injects it
    as ``actor`` — never trusted from the browser) invokes this as a subprocess.
    Always prints a parseable result and returns ``0`` for an action-level outcome
    (including a fail-closed REFUSED/ERROR); returns non-zero only when stdin is
    itself unparseable.
    """
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        print(json.dumps({"outcome": "error", "error": f"unparseable stdin: {exc}"}))
        return 2
    if not isinstance(payload, dict):
        print(json.dumps({"outcome": "error", "error": "stdin must be a JSON object"}))
        return 2
    result = run_action(payload)
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
