"""The onboarding wizard's **server side** — the Python the app's authenticated
routes invoke (``docs/ONBOARDING_WIZARD.md`` slice 1).

The app is a thin AppKit (Node/React) app; the authenticated ``server/plugins/onboarding``
routes resolve the actor from the platform identity headers and hand a JSON action
to this module (the same bridge shape as :mod:`ail.loop.apply_service`: a single
CLI reads a JSON action on stdin and prints a typed JSON result on stdout). It does
exactly four things, **reusing** existing framework capabilities rather than
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
import os
import sys
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ail.onboarding.experiment import (
    ExperimentAccessError,
    ExperimentClient,
    ExperimentPermissionError,
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

__all__ = [
    "OnboardingOutcome",
    "ValidationResult",
    "CreationResult",
    "RegisterResult",
    "ErrorResult",
    "register_agent",
    "run_requirements",
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
    """The experiment-creation result (page 1). ``prerequisite`` names the deploy grant."""

    outcome: OnboardingOutcome
    experiment_id: str = ""
    name: str = ""
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
    actor: str = ""
    refused_reason: str | None = None
    error: str | None = None


class ErrorResult(_Contract):
    """A dispatch-level error (unknown/malformed action)."""

    outcome: OnboardingOutcome = OnboardingOutcome.ERROR
    action: str = ""
    error: str


# ---------------------------------------------------------------------------
# Pure orchestration (injected clients → unit-testable, no live write)
# ---------------------------------------------------------------------------


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
    try:
        judge_config = build_judge_config(goal_keys)
    except (UnknownGoalError, ValueError) as exc:
        return _register_refused(name, exp, goal_keys, actor, str(exc))

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
        description=f"Registered via the onboarding wizard by {actor}; goals: {labels}.",
        judge_config=judge_config,
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
        creation = create_experiment_probe(clean, client=client)
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
    return CreationResult(
        outcome=OnboardingOutcome.CREATED,
        experiment_id=creation.experiment_id,
        name=creation.name,
        actor=actor,
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
) -> RegisterResult:
    """Register an agent live (fail-closed; a write failure is ERROR, never registered)."""
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
        return register_agent(
            agent_name=agent_name,
            experiment_id=experiment_id,
            goal_keys=goal_keys,
            actor=actor,
            client=client,
            warehouse_id=wh,
            existing_agents=existing,
            catalog=catalog,
            schema=schema,
        )
    except Exception as exc:  # noqa: BLE001 - any infra failure is an honest ERROR, never a fake register
        return RegisterResult(
            outcome=OnboardingOutcome.ERROR,
            agent_name=agent_name,
            experiment_id=experiment_id,
            goals=list(goal_keys),
            actor=actor,
            error=f"{type(exc).__name__}: {exc}",
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
        return run_create(str(payload.get("name") or ""), actor=actor, profile=profile)
    if action == "register_agent":
        return run_register(
            agent_name=str(payload.get("agent_name") or ""),
            experiment_id=str(payload.get("experiment_id") or ""),
            goal_keys=goal_keys,
            actor=actor,
            profile=profile,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
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
