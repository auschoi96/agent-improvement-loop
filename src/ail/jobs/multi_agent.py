"""Registry-driven multi-agent fan-out — the shared core the scheduled jobs share.

The four scheduled jobs (continuous RLM, auto-align, memory distiller, L0 publish)
each used to run for exactly ONE agent bound via bundle vars. This module is the
DRY core that turns each into a registry-driven MULTI-agent job: one run iterates
EVERY agent read from the UC ``agent_registry`` — the single source of truth, via
:func:`ail.publish_versions.load_registered_agents_full` — and processes each with
**per-agent isolation**: one agent's failure is logged loudly (to stderr, naming
the agent) and recorded, and the loop CONTINUES to the next agent. The aggregate
carries a worst-case return code (non-zero if ANY agent failed), mirroring the
local companion's ``worst_rc`` pattern (:mod:`ail.companion.bootstrap`).

Fail-closed boundary (do NOT weaken): ``load_registered_agents_full`` raises on a
real infra / permission / warehouse error and returns ``[]`` ONLY when the
registry table does not yet exist (a fresh workspace before the first publish).
So this module NEVER wraps the registry read in the per-agent try/except — a real
read failure PROPAGATES (the job fails loud) rather than being swallowed into
"zero agents". The per-agent try/except wraps ONLY the per-agent body. An empty
registry is a clean no-op (exit ``0``), not an error and never fabricated work.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

from ail.publish import _build_workspace_client
from ail.publish_versions import load_registered_agents_full
from ail.registry import Agent

__all__ = [
    "AgentOutcome",
    "MultiAgentResult",
    "load_registered_agents",
    "missing_registry_target",
    "resolve_registered_agent",
    "run_for_each_registered_agent",
]


@dataclass(frozen=True)
class AgentOutcome:
    """One agent's outcome within a multi-agent run."""

    agent_name: str
    rc: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.rc == 0 and self.error is None


@dataclass(frozen=True)
class MultiAgentResult:
    """The aggregate of a multi-agent run — attempts, failures, and worst-case rc."""

    outcomes: tuple[AgentOutcome, ...]

    @property
    def worst_rc(self) -> int:
        """Non-zero iff ANY agent failed (max of the per-agent return codes)."""
        return max((o.rc for o in self.outcomes), default=0)

    @property
    def attempted(self) -> tuple[str, ...]:
        """Names of every agent that was attempted, in registry order."""
        return tuple(o.agent_name for o in self.outcomes)

    @property
    def failures(self) -> tuple[AgentOutcome, ...]:
        return tuple(o for o in self.outcomes if o.rc != 0)

    @property
    def n_agents(self) -> int:
        return len(self.outcomes)

    @property
    def n_failed(self) -> int:
        return len(self.failures)


def run_for_each_registered_agent(
    agents: Sequence[Agent],
    per_agent_fn: Callable[[Agent], int | None],
    *,
    job_name: str,
    stderr: TextIO | None = None,
) -> MultiAgentResult:
    """Run ``per_agent_fn`` for every agent with per-agent isolation; aggregate the result.

    Iterates ``agents`` in registry order and calls ``per_agent_fn(agent)`` for each.
    A ``None`` return is treated as ``0`` (success); a non-zero int is that agent's
    return code; a raised exception is caught, logged LOUDLY to ``stderr`` naming the
    agent (with its traceback), recorded as ``rc=1``, and the loop CONTINUES to the
    next agent — one agent's failure NEVER aborts the others.

    An empty ``agents`` is a clean no-op: it logs "no registered agents" and returns
    an empty result whose :attr:`~MultiAgentResult.worst_rc` is ``0``. The caller
    must NOT call this with agents it could not read — a failed registry read must
    propagate before reaching here (see the module docstring).
    """
    err = stderr if stderr is not None else sys.stderr
    if not agents:
        print(f"[{job_name}] no registered agents in agent_registry — nothing to do (clean no-op).")
        return MultiAgentResult(outcomes=())

    names = ", ".join(a.agent_name for a in agents)
    print(f"[{job_name}] registry mode: {len(agents)} registered agent(s): {names}")

    outcomes: list[AgentOutcome] = []
    for i, agent in enumerate(agents, start=1):
        print(
            f"[{job_name}] ({i}/{len(agents)}) agent={agent.agent_name} "
            f"experiment={agent.experiment_id} — starting"
        )
        try:
            rc = per_agent_fn(agent)
        except Exception as exc:  # noqa: BLE001 - per-agent isolation: one failure must not abort the rest
            print(
                f"[{job_name}] agent={agent.agent_name} FAILED "
                f"(logged and continuing to the next agent): {exc!r}",
                file=err,
            )
            traceback.print_exc(file=err)
            outcomes.append(AgentOutcome(agent_name=agent.agent_name, rc=1, error=str(exc)))
            continue
        rc_int = int(rc or 0)
        if rc_int != 0:
            print(
                f"[{job_name}] agent={agent.agent_name} completed with non-zero rc={rc_int} "
                "(recorded; continuing).",
                file=err,
            )
        outcomes.append(AgentOutcome(agent_name=agent.agent_name, rc=rc_int))

    result = MultiAgentResult(outcomes=tuple(outcomes))
    print(
        f"[{job_name}] done: agents={result.n_agents} failed={result.n_failed} "
        f"worst_rc={result.worst_rc}"
    )
    return result


def load_registered_agents(
    *,
    warehouse_id: str,
    catalog: str,
    schema: str,
    client: Any | None = None,
) -> list[Agent]:
    """Read the UC ``agent_registry`` back as typed :class:`~ail.registry.Agent`\\ s.

    Thin wrapper over :func:`ail.publish_versions.load_registered_agents_full` that
    builds a workspace client from the already-resolved run-as auth when one is not
    injected. The caller MUST have resolved auth (``resolve_job_auth``) first so the
    default client picks up the explicit ``DATABRICKS_HOST``/``DATABRICKS_TOKEN``
    bearer. Fail-closed behavior is inherited unchanged (see module docstring).
    """
    c = client if client is not None else _build_workspace_client(None)
    return load_registered_agents_full(
        client=c, warehouse_id=warehouse_id, catalog=catalog, schema=schema
    )


def resolve_registered_agent(
    agent_name: str,
    *,
    warehouse_id: str,
    catalog: str,
    schema: str,
    client: Any | None = None,
) -> Agent:
    """Resolve ONE registered agent by name from the UC ``agent_registry`` — the single
    source of truth the app writes and the scheduled jobs read.

    The read side the local companion (planner + executor) shares with the scheduled
    multi-agent jobs, so a UI-onboarded agent — present ONLY in UC, never in any YAML —
    is resolvable by name, carrying its ``experiment_id``, ``target_workspace``,
    ``goal_config`` and the rest straight from UC (no local YAML, no CLI-arg guess).

    Reuses the fail-closed :func:`load_registered_agents` unchanged, so the two
    boundaries it inherits are preserved (see the module docstring):

    * a real infra / permission / warehouse error PROPAGATES (never swallowed into
      "no such agent"); and
    * a genuinely **not-yet-created** registry table reads back as an empty registry.

    FAIL-CLOSED on absence: if ``agent_name`` is not among the registered agents this
    raises :class:`KeyError` — it NEVER fabricates a bare :class:`Agent` with a guessed
    experiment / target_workspace. An empty registry (table absent) and "agent not in a
    present table" are distinct states, and the error message names which one occurred,
    but both fail closed here: there is nothing to resolve, so the caller must not plan
    or execute against a guessed identity.
    """
    agents = load_registered_agents(
        warehouse_id=warehouse_id, catalog=catalog, schema=schema, client=client
    )
    for agent in agents:
        if agent.agent_name == agent_name:
            return agent
    fqn = f"{catalog}.{schema}.agent_registry"
    have = ", ".join(a.agent_name for a in agents) or "<none — registry table absent or empty>"
    raise KeyError(
        f"no agent named {agent_name!r} in the UC agent_registry ({fqn}); registered: {have}. "
        "Onboard it in the app (or publish the registry) before the companion can resolve it — "
        "fail-closed: never a guessed experiment / target_workspace."
    )


def missing_registry_target(
    warehouse_id: str | None, catalog: str | None, schema: str | None
) -> list[str]:
    """The workspace-safety vars the registry read needs but is missing (fail-closed).

    Registry mode cannot guess the workspace — the #67/#5 lesson: no baked-in
    workspace defaults. Returns the empty list when all three are present; otherwise
    the names of the missing arg(s), so a caller can fail loud with an actionable
    message rather than reading the wrong (or no) ``agent_registry``.
    """
    return [
        name
        for name, val in (
            ("--warehouse-id", warehouse_id),
            ("--catalog", catalog),
            ("--schema", schema),
        )
        if not val
    ]
