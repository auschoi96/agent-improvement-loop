"""``ail-companion-planner`` — the local companion planner (evidence-first, no proving).

The **local companion** planner lane of ``docs/PRODUCT_ARCHITECTURE.md`` (§4/§7): a
deployer-run process (Claude Agent SDK compute, **not** Databricks serverless) that
reads the judge + RLM + L0 evidence already attached to an agent's traces and writes
**evidence-backed PENDING proposals** into the app's ``agent_proposed_actions`` table.
It is the **evidence-first** planner: it **does not prove** (proving is opt-in Tier-2,
run later on the user's frozen suite when the human wants a measured delta) and it
**does not apply** anything. The human is the gate; this surfaces the evidence.

On one run it:

1. **reads the feedback** for the agent/experiment — the RLM-recommended assets, the
   L0 redundant-read waste patterns, and (when present) judge dimensions + post-apply
   regressions — assembled from the traces by the *existing*
   :func:`ail.jobs.optimization_cycle.build_feedback_bundle`. If the feedback cannot be
   read, it prints an honest error and writes **no** proposal (never a fabricated why);
2. **plans** over that evidence: the deterministic **Lane A** rules
   (:func:`ail.loop.decision_rules.decide`) **and** the LLM-agent **Lane B** planner
   (:func:`ail.loop.planner.agent_planner`), de-duped into one union — Lane A always
   runs and emits even if Lane B fails (:func:`ail.loop.planner.combined_decisions`);
3. **builds a concrete candidate** per decision and **gates on readiness + judge trust
   only** (:func:`ail.loop.evidence_cycle.run_evidence_cycle` → the *unweakened*
   :func:`ail.loop.controller.evaluate_gate`) — **no prover call, no ProofSummary**;
4. **publishes** the resulting PENDING (``proof=None``) proposals to
   ``agent_proposed_actions`` (:func:`ail.loop.publish_proposals.publish_agent_proposals`),
   atomically replacing this agent's slice; and
5. **surfaces to the operator** (structured stdout): the evidence it read, what Lane A
   and Lane B each decided and why, and for each proposal the row it wrote (or, per
   fail-closed skipped decision, why it wrote none).

**Auth — a static token, matched to the experiment's workspace host (hard-won).** The
planner runs as a *long-lived local process*. A ``--profile`` OAuth login refreshes its
token mid-run, and that refresh cannot reliably persist from a background process — the
run then dies mid-flight. So this entrypoint requires a **static** ``DATABRICKS_TOKEN``
(a PAT or a secret-scope token) pinned to ``DATABRICKS_HOST`` (the workspace the
experiment lives in), drops any ambient ``DATABRICKS_CONFIG_PROFILE`` so nothing falls
back to refreshing OAuth, and refuses to run without one.

**Registry-driven (UC ``agent_registry``).** The agent's experiment is resolved by name
from the UC ``agent_registry`` (the SAME table the app writes and the scheduled jobs
read) via the shared :func:`ail.jobs.multi_agent.resolve_registered_agent`, so a
UI-onboarded agent is plannable with just ``--agent`` + the UC connection. ``--experiment``
is an optional local/manual override: when given it wins (no registry read); when omitted
the experiment comes from UC. Fail-closed when neither yields an experiment — the cycle
never plans against an empty/guessed experiment.

**Reuse.** The feedback assembly, goal construction, readiness gate, and publish are
the *same* seams the (paused, prove-before-propose) unified cycle uses
(:mod:`ail.jobs.optimization_cycle`); only the cycle is the evidence-first
:func:`ail.loop.evidence_cycle.run_evidence_cycle` and the candidate builder is the
no-cost-guard :func:`ail.loop.candidate_builders.evidence_candidate_builder`.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Literal

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.goals.compiler import CompiledGoal, GoalCompileError
from ail.jobs import optimization_cycle as oc
from ail.jobs.multi_agent import resolve_registered_agent
from ail.jobs.publish_job import resolve_job_auth
from ail.loop.candidate_builders import evidence_candidate_builder
from ail.loop.decision_rules import FeedbackBundle
from ail.loop.evidence_cycle import EvidenceCycleResult, run_evidence_cycle
from ail.loop.planner import CombinedDecisions, Planner, agent_planner
from ail.loop.proposals import ProposedAction, TriggerKind
from ail.publish import DEFAULT_CATALOG, DEFAULT_SCHEMA
from ail.registry import Agent

__all__ = [
    "resolve_static_auth",
    "resolve_planner_agent",
    "surface_evidence",
    "surface_plan",
    "surface_proposals",
    "load_persisted_goal",
    "resolve_goal",
    "run",
    "main",
]

_TAG = "[ail.companion]"


def resolve_static_auth(args: argparse.Namespace) -> str:
    """Pin a **static** ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``; never a refreshing OAuth.

    Sets ``DATABRICKS_HOST`` from ``--host`` when given, **drops any
    ``DATABRICKS_CONFIG_PROFILE``** so the SDK cannot fall back to a profile's
    mid-run-refreshing OAuth (the failure mode this planner must avoid — it runs as a
    long-lived local process), and requires a static credential to be present:
    ``DATABRICKS_TOKEN`` in the environment, or a ``--token-secret-scope`` /
    ``--token-secret-key`` pair. Fails loud with operator guidance if neither is set,
    rather than silently minting a short-lived OAuth bearer.

    Delegates the actual resolution to the hardened
    :func:`ail.jobs.publish_job.resolve_job_auth` (the ``"env"`` path when a static
    token is already present — no minting, no refresh) and returns its short label.
    """
    if args.host:
        os.environ["DATABRICKS_HOST"] = args.host
    # Hard-won lesson: a --profile OAuth login refreshes mid-run and cannot persist from
    # a background process, breaking a long run. Drop it so nothing falls back to it.
    os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)

    have_env_token = bool(os.environ.get("DATABRICKS_HOST")) and bool(
        os.environ.get("DATABRICKS_TOKEN")
    )
    have_scope = bool(args.token_secret_scope and args.token_secret_key)
    if not have_env_token and not have_scope:
        raise SystemExit(
            f"{_TAG} refusing to run without a STATIC Databricks token matched to the "
            "experiment's workspace host. Set DATABRICKS_HOST (or --host) and DATABRICKS_TOKEN:\n"
            "  export DATABRICKS_HOST=https://<workspace-host>\n"
            "  export DATABRICKS_TOKEN=<pat-or-static-token>\n"
            "(or pass --token-secret-scope/--token-secret-key). Do NOT rely on a --profile OAuth "
            "login: its mid-run token refresh cannot persist from a long-running local process "
            "and will break the run."
        )

    return resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )


# ---------------------------------------------------------------------------
# Operator-facing surfacing (structured stdout): evidence -> plan -> outcome.
# ---------------------------------------------------------------------------


def surface_evidence(feedback: FeedbackBundle) -> None:
    """Print the RLM / L0 / judge / regression evidence the planner READ, verbatim.

    The human-readable "why" behind every proposal starts here: the operator sees the
    exact evidence the plan rested on (never a fabricated summary).
    """
    fb = feedback
    print(f"{_TAG} --- EVIDENCE READ ---")
    print(
        f"{_TAG} objective: value={fb.objective_metric_value} "
        f"baseline={fb.objective_baseline_value}"
    )

    assets = fb.rlm_assets
    print(f"{_TAG} RLM-recommended assets ({len(assets)}):")
    for a in assets:
        print(
            f"{_TAG}   [{a.asset_type}] {a.title!r}: recurred across {a.n_traces} trace(s), "
            f"rank {a.rank}; traces {list(a.trace_ids)}"
        )
    if not assets:
        print(f"{_TAG}   (none)")

    reads = fb.redundant_reads
    print(f"{_TAG} L0 redundant-read / waste patterns ({len(reads)}):")
    for r in reads:
        target = r.repeated_target or r.tool or "repeated target"
        waste = (
            f", ~{r.estimated_wasted_tokens} wasted tokens"
            if r.estimated_wasted_tokens is not None
            else ""
        )
        print(
            f"{_TAG}   {target!r}: repeated {r.occurrences}x{waste}; dominant={r.dominant}; "
            f"traces {list(r.trace_ids)}"
        )
    if not reads:
        print(f"{_TAG}   (none)")

    dims = fb.judge_dimensions
    print(f"{_TAG} judge dimensions below par ({len(dims)}):")
    for j in dims:
        print(
            f"{_TAG}   judge {j.judge_name!r} dimension {j.dimension!r}: score {j.score}, "
            f"trusted={j.trusted}; traces {list(j.trace_ids)}"
        )
    if not dims:
        print(f"{_TAG}   (none)")

    regs = fb.post_apply_regressions
    print(f"{_TAG} post-apply regressions ({len(regs)}):")
    for p in regs:
        print(
            f"{_TAG}   version {p.agent_version!r} vs {p.predecessor_version!r} on "
            f"{p.objective_metric!r}: regressed={p.regressed}; traces {list(p.trace_ids)}"
        )
    if not regs:
        print(f"{_TAG}   (none)")


def surface_plan(plan: CombinedDecisions) -> None:
    """Print what Lane A and Lane B each decided, and why — attributable per decision.

    Lane B's failure (if any) is surfaced honestly as ``planner_error`` rather than
    hidden — Lane A still ran and emitted (fail-closed).
    """
    p = plan
    print(f"{_TAG} --- PLAN (Lane A + Lane B) ---")
    print(
        f"{_TAG} decisions: lane_A={p.n_from_a} lane_B={p.n_from_b} "
        f"deduped={p.n_deduped} planner_error={p.planner_error}"
    )
    for d in p.decisions:
        lane = "B" if d.trigger.kind is TriggerKind.AGENT_PLANNER else "A"
        print(
            f"{_TAG}   [Lane {lane}] {d.action_kind.value} (trigger={d.trigger.kind.value}): "
            f"{d.trigger.summary}"
        )
    if not p.decisions:
        print(f"{_TAG}   (no decisions this cycle)")


def surface_proposals(ecr: EvidenceCycleResult, *, published: dict[str, str]) -> None:
    """Print, per proposal, the row written (or, per skip, the fail-closed reason none was).

    ``published`` maps ``proposal_id`` → a short outcome string (the row's disposition,
    e.g. ``"written"`` or ``"dry-run (not written)"``), so each surfaced proposal shows
    exactly what happened to its row.
    """
    result = ecr.result
    print(f"{_TAG} --- PROPOSALS ({len(result.proposals)} PENDING, evidence-only) ---")
    for p in result.proposals:
        disp = published.get(p.proposal_id, "not written")
        print(
            f"{_TAG}   proposal {p.proposal_id} [{p.action_kind.value}] "
            f"risk={p.risk_class.value} gate_tier={p.gate_status.readiness_tier} "
            f"proof=NONE(evidence-first) -> row {disp}"
        )
        print(f"{_TAG}     why: {p.trigger.summary}")
        print(f"{_TAG}     what: {p.change.summary}")
    if not result.proposals:
        print(f"{_TAG}   (no proposals emitted this cycle)")

    print(f"{_TAG} --- SKIPPED ({len(result.skipped)} fail-closed) ---")
    for s in result.skipped:
        print(f"{_TAG}   {s.action_kind} (trigger={s.trigger_kind}): {s.reason}")
    if not result.skipped:
        print(f"{_TAG}   (none)")


# ---------------------------------------------------------------------------
# Goal load (GAP A): prefer the confirmed intake goal, fall back to CLI args.
# ---------------------------------------------------------------------------


def load_persisted_goal(args: argparse.Namespace) -> CompiledGoal | None:
    """Read this agent's confirmed intake goal from UC, or ``None`` (fall back to args).

    Closes the intake→loop bridge on the read side: the confirmed goal the user
    stated (:mod:`ail.requirements.persistence`) is preferred over the operator's CLI
    ``--objective-metric``/``--goal-*`` flags. Fail-soft — returns ``None`` (so
    :func:`resolve_goal` uses the arg-based goal) when:

    * no static token is set (the read needs one; tests drive :func:`run` without a
      token, so this keeps them offline — the companion's :func:`resolve_static_auth`
      pins one before a real run), or
    * the table / the agent's row is absent (a first run before any intake), or
    * any read/parse error occurs (never fabricates or blocks the cycle).
    """
    if not (os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")):
        return None
    try:
        from ail.publish import _build_workspace_client
        from ail.requirements.persistence import load_persisted_goal as _load

        client = _build_workspace_client(None)  # static env token (resolve_static_auth pinned it)
        return _load(
            agent_name=args.agent,
            client=client,
            warehouse_id=args.warehouse_id,
            catalog=args.catalog,
            schema=args.schema,
        )
    except Exception as exc:  # noqa: BLE001 - a read failure falls back to args, never blocks
        print(
            f"{_TAG} note: could not read a persisted intake goal "
            f"({type(exc).__name__}: {exc}); using the CLI-arg goal."
        )
        return None


def _goal_from_registry(agent: Agent) -> CompiledGoal | None:
    """Build the confirmed goal persisted on the registry row, if complete."""
    config = agent.goal_config or {}
    objective = str(config.get("objective_metric") or "").strip()
    if not objective:
        return None
    from ail.goals.compiler import GoalTarget, Guardrail

    raw_direction = str(config.get("goal_direction") or "minimize")
    direction: Literal["minimize", "maximize"]
    if raw_direction == "minimize":
        direction = "minimize"
    elif raw_direction == "maximize":
        direction = "maximize"
    else:
        return None
    raw_target_kind = str(config.get("goal_target_kind") or "relative")
    target_kind: Literal["relative", "absolute"]
    if raw_target_kind == "relative":
        target_kind = "relative"
    elif raw_target_kind == "absolute":
        target_kind = "absolute"
    else:
        return None

    raw_guardrails = config.get("guardrail_judge") or []
    if isinstance(raw_guardrails, str):
        raw_guardrails = [raw_guardrails]
    guardrails: list[Guardrail] = []
    for raw in raw_guardrails:
        name, _, threshold = str(raw).partition(":")
        if name.strip():
            guardrails.append(
                Guardrail(
                    name=name.strip(),
                    kind="judge",
                    threshold=float(threshold) if threshold.strip() else None,
                )
            )
    try:
        return CompiledGoal(
            objective_metric=objective,
            direction=direction,
            target=GoalTarget(
                value=float(config.get("goal_target", -0.30)),
                kind=target_kind,
            ),
            guardrails=tuple(guardrails),
            cohort=agent.agent_name,
        ).confirm()
    except (GoalCompileError, TypeError, ValueError):
        return None


def resolve_goal(
    args: argparse.Namespace, *, agent: Agent | None = None
) -> tuple[CompiledGoal, str]:
    """Resolve the goal the cycle runs, preferring the confirmed intake goal.

    Returns ``(goal, source)`` where ``source`` is ``"persisted-intake"`` when a
    **confirmed** persisted goal was loaded, else ``"cli-args"`` (the existing
    :func:`ail.jobs.optimization_cycle._build_goal` arg-based path). Only a
    human-confirmed persisted goal is preferred — an unconfirmed one is ignored so
    the controller's confirm gate is never bypassed by a stale write.
    """
    persisted = load_persisted_goal(args)
    if persisted is not None and persisted.human_confirmed:
        return persisted, "persisted-intake"
    if agent is not None:
        registered = _goal_from_registry(agent)
        if registered is not None:
            return registered, "uc-registry"
    return oc._build_goal(args), "cli-args"


# ---------------------------------------------------------------------------
# Agent resolution (GAP): registry-driven experiment, with --experiment as override.
# ---------------------------------------------------------------------------


def resolve_planner_agent(args: argparse.Namespace) -> tuple[Agent, str]:
    """Resolve the planner's :class:`~ail.registry.Agent`, registry-driven by default.

    Returns ``(agent, source)`` where ``source`` is:

    * ``"uc-registry"`` — the default: resolve the agent by name from the UC
      ``agent_registry`` (the SAME table the app writes and the scheduled jobs read) via
      the shared :func:`ail.jobs.multi_agent.resolve_registered_agent`, taking its
      ``experiment_id`` — and carrying ``goal_config`` and the rest — straight from UC.
      So a UI-onboarded agent is plannable with just ``--agent`` + the UC connection
      (``--warehouse-id`` / ``--catalog`` / ``--schema``); or
    * ``"explicit-arg"`` — the local/manual override: when ``--experiment`` is given it
      WINS, and the Agent is built from it directly with **no** registry read (a one-off
      run against an explicit experiment).

    Fail-closed: with neither ``--experiment`` nor a registry match,
    :func:`resolve_registered_agent` raises :class:`KeyError` — the cycle never plans
    against an empty/guessed experiment. A real registry-read infra error propagates.
    """
    if args.experiment:
        return Agent(agent_name=args.agent, experiment_id=args.experiment), "explicit-arg"
    from ail.publish import _build_workspace_client

    client = _build_workspace_client(None)  # static env token (resolve_static_auth pinned it)
    agent = resolve_registered_agent(
        args.agent,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
        client=client,
    )
    return agent, "uc-registry"


# ---------------------------------------------------------------------------
# One evidence-first run: read evidence -> plan/gate/propose -> publish.
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute one evidence-first companion run; return a process exit code.

    Returns ``0`` on a completed run (proposals published, or the empty set published
    to clear a superseded slice — unless ``--dry-run``). Returns ``2`` fail-closed when
    the **agent cannot be resolved** (no ``--experiment`` and not in the registry) or the
    **evidence or readiness could not be read**: it prints an honest error and publishes
    **nothing** (it never clears the agent's slice on an unknown state, and never
    fabricates a proposal or a guessed experiment).
    """
    if args.warehouse_id:
        os.environ[TRACING_WAREHOUSE_ENV] = args.warehouse_id
    try:
        agent, experiment_source = resolve_planner_agent(args)
    except KeyError as exc:
        print(f"{_TAG} ERROR: {exc}; writing NO proposal (fail-closed, no guessed experiment).")
        return 2
    # The reused feedback/gate seams read ``args.experiment``; pin it to the resolved id
    # so a registry-resolved experiment (when --experiment was omitted) is what the cycle
    # actually runs against — not the None it started as.
    args.experiment = agent.experiment_id
    goal, goal_source = resolve_goal(args, agent=agent)

    planner: Planner = (
        partial(agent_planner, model=args.planner_model) if args.planner_model else agent_planner
    )

    print(
        f"{_TAG} agent={agent.agent_name} experiment={agent.experiment_id} "
        f"experiment_source={experiment_source} host={os.environ.get('DATABRICKS_HOST')} "
        f"goal_source={goal_source} objective={goal.objective_metric} "
        f"target={goal.target.kind}:{goal.target.value} confirmed={goal.human_confirmed} "
        f"table={args.catalog}.{args.schema}.agent_proposed_actions dry_run={args.dry_run}"
    )

    # Read the evidence FIRST, fail-closed: a read failure prints an honest error and
    # publishes nothing (never a fabricated why, never a slice-clearing empty publish
    # on an unknown state).
    feedback_source = oc._default_feedback_source(agent, args)
    try:
        feedback = feedback_source()
    except Exception as exc:  # noqa: BLE001 - surface honestly; do not fabricate/clear
        print(
            f"{_TAG} ERROR: could not read the agent's feedback "
            f"({type(exc).__name__}: {exc}); writing NO proposal (fail-closed, no fabricated why)."
        )
        return 2

    surface_evidence(feedback)

    # Plan (Lane A + Lane B) + gate (readiness + judge trust) + propose (proof=None).
    # A gate/readiness read failure or an unconfirmed goal is likewise fail-closed: no
    # publish.
    try:
        ecr = run_evidence_cycle(
            agent,
            goal,
            feedback_source=lambda: feedback,
            candidate_builder=evidence_candidate_builder(),
            gate=oc._default_gate(args),
            planner=planner,
            decision_thresholds=None,
        )
    except ValueError as exc:  # unconfirmed goal (fail-loud precondition)
        print(f"{_TAG} ERROR: {exc}; writing NO proposal.")
        return 2
    except Exception as exc:  # noqa: BLE001 - readiness/gate read failure => fail-closed, no publish
        print(
            f"{_TAG} ERROR: could not compute readiness/gate "
            f"({type(exc).__name__}: {exc}); writing NO proposal (fail-closed)."
        )
        return 2

    surface_plan(ecr.plan)

    proposals = list(ecr.result.proposals)
    published: dict[str, str] = {}
    if args.dry_run:
        for p in proposals:
            published[p.proposal_id] = "dry-run (not written)"
        surface_proposals(ecr, published=published)
        print(f"{_TAG} DRY-RUN: not publishing ({len(proposals)} proposal(s) would be written).")
        return 0

    # Publish: agent-scoped atomic REPLACE of this agent's whole pending slice (so a
    # superseded proposal disappears and the write is idempotent). Published even when
    # empty — to clear a slice whose evidence no longer holds — but ONLY because the
    # cycle above ran successfully (a read failure returned 2 above without reaching here).
    n_written = _publish(proposals, agent=agent, args=args)
    for p in proposals:
        published[p.proposal_id] = "written"

    surface_proposals(ecr, published=published)
    print(
        f"{_TAG} PUBLISHED {n_written} row(s) to "
        f"{args.catalog}.{args.schema}.agent_proposed_actions "
        f"(agent-scoped REPLACE of {agent.agent_name!r}'s pending slice)."
    )
    return 0


def _publish(proposals: list[ProposedAction], *, agent: Agent, args: argparse.Namespace) -> int:
    """Write this agent's PENDING proposals via the existing agent-scoped atomic swap."""
    from ail.loop.publish_proposals import publish_agent_proposals
    from ail.publish import _build_workspace_client

    client = _build_workspace_client(None)  # static env token (resolve_static_auth pinned it)
    return publish_agent_proposals(
        proposals,
        agent_name=agent.agent_name,
        experiment_id=agent.experiment_id,
        client=client,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local companion planner (evidence-first, no proving): read an agent's "
            "judge/RLM/L0 evidence, plan (Lane A + Lane B), gate on readiness + judge trust, "
            "and publish PENDING evidence-backed proposals. Registry-driven: the agent's "
            "experiment is resolved from the UC agent_registry by --agent; --experiment is an "
            "optional local override. Proves nothing; applies nothing."
        )
    )
    parser.add_argument("--agent", default="claude_code", help="Agent name (proposal scope).")
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment id (LOCAL/MANUAL OVERRIDE). Optional: when omitted, the "
        "experiment is resolved from the UC agent_registry by --agent (the registry-driven "
        "default). When given, it OVERRIDES the registry (a one-off run against this experiment). "
        "Fail-closed if neither is available.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("DATABRICKS_HOST"),
        help="Workspace host the experiment lives in (sets DATABRICKS_HOST). A STATIC "
        "DATABRICKS_TOKEN pinned to this host is required; --profile OAuth is refused.",
    )
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full evidence-first cycle and surface the plan + proposals, but "
        "publish NOTHING (preview before writing the agent's slice).",
    )
    # goal (operator-configured; confirmed by --goal-confirmed / AIL_CONFIRM_GOAL)
    parser.add_argument("--objective-metric", default="total_tokens")
    parser.add_argument("--goal-direction", default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--goal-target", type=float, default=-0.30)
    parser.add_argument("--goal-target-kind", default="relative", choices=["relative", "absolute"])
    parser.add_argument(
        "--guardrail-judge",
        action="append",
        default=None,
        help="Judge guardrail as 'name:threshold' (repeatable).",
    )
    parser.add_argument(
        "--objective-baseline",
        type=oc._opt_float,
        default=None,
        help="Baseline a relative objective target is measured against "
        "(empty => treated as not-yet-met, no fabricated baseline).",
    )
    parser.add_argument(
        "--goal-confirmed",
        default=os.environ.get("AIL_CONFIRM_GOAL", "false"),
        help="'true' to mark the operator-configured goal human-confirmed (required to run). "
        "Anything else leaves it unconfirmed and the cycle refuses to run (fail-loud).",
    )
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")
    # The companion never uses a profile (static token only — see resolve_static_auth);
    # the reused optimization_cycle seams read args.profile, so pin it to None here.
    args.profile = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth_path = resolve_static_auth(args)
    print(f"{_TAG} auth={auth_path} host={os.environ.get('DATABRICKS_HOST')}")
    if auth_path == "minted":
        print(
            f"{_TAG} WARNING: auth was MINTED from ambient identity, not a static token. For a "
            "long local run, export a static DATABRICKS_TOKEN pinned to --host instead — a minted "
            "OAuth bearer risks a mid-run refresh that cannot persist from a background process."
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
