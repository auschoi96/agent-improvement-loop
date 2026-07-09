"""Databricks Job entrypoint for the scheduled continuous RLM review.

Fired on a schedule (``resources/continuous_rlm.job.yml``), not a trace-arrival
trigger: the UC-backed trace store is a VIEW, so a ``table_update`` trigger is
infeasible. Each firing runs one bounded :func:`ail.l3.continuous.run_continuous_rlm`
pass — sampling, idempotency, and the fail-closed failed-review marker all live there.

Registry-driven multi-agent: with no ``--experiment`` it runs REGISTRY MODE — it
reads every agent from the UC ``agent_registry`` (via :mod:`ail.jobs.multi_agent`)
and reviews each agent's OWN experiment against that agent's OWN ``goal_config``
(falling back to the bundle's global goal vars when an agent has none), with
per-agent isolation. Passing an explicit ``--experiment`` is the single-agent
override for local/manual runs (review only that experiment with the args' goal).

Two things this wrapper owns:

* **Model + effort.** ``--judge-model`` defaults to ``databricks-gpt-5-5-pro`` — the
  most powerful *viable* HALO judge (Databricks Claude endpoints reject HALO's always-on
  ``parallel_tool_calls``; ``gpt-5-6`` does not exist on the gateway). Effort is
  auto-resolved from the model by :func:`ail.l3.reviewer.resolve_reasoning_effort`
  (so this alias gets ``xhigh`` despite HALO's hyphen-blind prefix check); pass an
  explicit ``--reasoning-effort`` to override.
* **Goal-steering.** When ``--objective-metric`` is set, the review rubric is derived
  from the operator-configured goal (:func:`ail.l3.goal_rubric.rubric_from_goal`) so
  HALO judges and recommends in service of what the user is optimizing for. An empty
  ``--objective-metric`` falls back to the default five-guideline rubric.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from ail.jobs.multi_agent import (
    load_registered_agents,
    missing_registry_target,
    run_for_each_registered_agent,
)
from ail.jobs.publish_job import resolve_job_auth
from ail.l3.continuous import run_continuous_rlm
from ail.l3.reviewer import normalize_reasoning_effort
from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric
from ail.registry import Agent

#: Most powerful VIABLE HALO judge on the Databricks gateway (see module docstring).
DEFAULT_JUDGE_MODEL = "databricks-gpt-5-5-pro"


def _resolve_goal_knobs(
    args: argparse.Namespace, *, goal_config: dict[str, Any] | None, cohort: str
) -> dict[str, Any]:
    """The goal knobs for one review: from the agent's ``goal_config``, else the args.

    Single-agent runs pass ``goal_config=None`` and get the args' goal knobs (the
    current behavior). A registry agent passes its OWN ``goal_config`` (the free-form
    mapping carried on :class:`ail.registry.Agent`); each present key overrides the
    corresponding arg default, and an absent/empty key falls back to the arg — so a
    partially-configured agent still steers on what it set and defaults the rest.

    ``guardrail_judge`` is normalized to a list of ``'name:threshold'`` specs whether
    the registry stored a single string or a list.
    """
    gc = goal_config or {}

    def _val(key: str, fallback: Any) -> Any:
        v = gc.get(key)
        return fallback if v is None or v == "" else v

    guard = gc.get("guardrail_judge")
    if guard is None or guard == "":
        guardrail_specs = [s for s in (args.guardrail_judge or []) if s.strip()]
    elif isinstance(guard, str):
        guardrail_specs = [guard]
    else:  # a list/tuple of specs
        guardrail_specs = [str(s) for s in guard if str(s).strip()]

    return {
        "objective_metric": str(_val("objective_metric", args.objective_metric)),
        "goal_direction": str(_val("goal_direction", args.goal_direction)),
        "goal_target": float(_val("goal_target", args.goal_target)),
        "goal_target_kind": str(_val("goal_target_kind", args.goal_target_kind)),
        "guardrail_specs": guardrail_specs,
        "cohort": cohort,
    }


def _build_rubric(knobs: dict[str, Any]) -> ReviewRubric:
    """Build the review rubric: goal-steered when configured, else the default rubric.

    Mirrors :func:`ail.jobs.optimization_cycle._build_goal` (identical goal knobs),
    then derives a :class:`~ail.l3.rubric.ReviewRubric` from the compiled goal via
    :func:`ail.l3.goal_rubric.rubric_from_goal`. An empty ``objective_metric`` (the
    fallback) keeps :data:`~ail.l3.rubric.DEFAULT_RUBRIC`. The ``CompiledGoal`` is
    fully validated (allowlist + readiness contract), so a misconfigured goal fails
    loud here rather than silently reviewing against a fabricated objective. No
    confirmation gate: this is a read-only review that attaches assessments, not an
    apply.
    """
    if not str(knobs["objective_metric"]).strip():
        return DEFAULT_RUBRIC

    from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
    from ail.l3.goal_rubric import rubric_from_goal

    guardrails: list[Guardrail] = []
    for spec in knobs["guardrail_specs"]:
        if not spec.strip():
            continue  # blank spec (e.g. the empty DAB var default) => no guardrail
        name, _, threshold = spec.partition(":")
        guardrails.append(
            Guardrail(
                name=name.strip(),
                kind="judge",
                threshold=float(threshold) if threshold.strip() else None,
            )
        )
    goal = CompiledGoal(
        objective_metric=knobs["objective_metric"],
        direction=knobs["goal_direction"],
        target=GoalTarget(value=knobs["goal_target"], kind=knobs["goal_target_kind"]),
        guardrails=tuple(guardrails),
        cohort=knobs["cohort"],
    )
    return rubric_from_goal(goal)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one bounded scheduled RLM/HALO pass over recent MLflow traces. "
        "With no --experiment it runs REGISTRY MODE over every agent in agent_registry "
        "(each agent's own experiment + goal_config); pass --experiment to review JUST "
        "that one experiment (single-agent override)."
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="Explicit experiment id => single-agent override (review only that one). "
        "Empty (the default) => registry mode: review every agent in agent_registry.",
    )
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument(
        "--catalog",
        default=os.environ.get("AIL_CATALOG", ""),
        help="UC catalog holding agent_registry (registry mode). Defaults to AIL_CATALOG.",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("AIL_SCHEMA", ""),
        help="UC schema holding agent_registry (registry mode). Defaults to AIL_SCHEMA.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"HALO judge serving endpoint (default {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="",
        type=str.lower,
        choices=["", "none", "auto", "minimal", "low", "medium", "high", "xhigh"],
        help="Explicit HALO reasoning-effort override. Empty / 'none' / 'auto' (any case) "
        "=> no override, auto-resolve from --judge-model (databricks-gpt-5-5-pro => xhigh).",
    )
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--max-reviews", type=int, default=2)
    parser.add_argument("--sample-rate", type=float, default=0.10)
    parser.add_argument("--min-tokens", type=int, default=50_000)
    parser.add_argument("--reviewer-experiment", default="")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=None)
    # goal-steering (same knobs reused by the local companion planner). Empty --objective-metric
    # => the default five-guideline rubric (no goal). The goal is used only to steer
    # the read-only review, so there is no confirmation gate.
    parser.add_argument("--agent", default="claude_code", help="Cohort the goal is bound to.")
    parser.add_argument(
        "--objective-metric",
        default="",
        help="Goal objective metric (empty => default rubric, no goal-steering).",
    )
    parser.add_argument("--goal-direction", default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--goal-target", type=float, default=-0.30)
    parser.add_argument("--goal-target-kind", default="relative", choices=["relative", "absolute"])
    parser.add_argument(
        "--guardrail-judge",
        action="append",
        default=None,
        help="Judge guardrail as 'name:threshold' (repeatable); blank => none.",
    )
    parser.add_argument(
        "--token-secret-scope",
        default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", ""),
        help="Secret scope holding the run-as bearer token. Empty => mint from run-as identity.",
    )
    parser.add_argument(
        "--token-secret-key",
        default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""),
        help="Secret key within --token-secret-scope.",
    )
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")
    return args


def _run_rlm_for(
    args: argparse.Namespace,
    *,
    experiment: str,
    reviewer_experiment: str | None,
    goal_config: dict[str, Any] | None,
    cohort: str,
) -> int:
    """Run one bounded RLM pass over ``experiment`` — the reused single-agent body.

    Goal-steering resolves from ``goal_config`` (the agent's own knobs) with the args
    as the fallback, so each agent is reviewed against ITS objective. Idempotency /
    sampling / the failed-review marker all live in :func:`run_continuous_rlm`
    unchanged and are per-experiment, so one agent's pass never touches another's.
    """
    knobs = _resolve_goal_knobs(args, goal_config=goal_config, cohort=cohort)
    rubric = _build_rubric(knobs)
    # Normalize the effort input HERE (the CLI/job boundary): empty / 'none' / 'auto'
    # mean "no override, auto-resolve" and must become None rather than a literal effort
    # injected into HALO. build_engine_config re-applies this defensively.
    reasoning_effort = normalize_reasoning_effort(args.reasoning_effort)
    print(
        f"[ail.jobs.continuous_rlm] experiment={experiment} cohort={cohort} "
        f"judge_model={args.judge_model} reasoning_effort={reasoning_effort or 'auto'} "
        f"rubric={rubric.rubric_id} sample_rate={args.sample_rate} "
        f"max_reviews={args.max_reviews}"
    )
    report = run_continuous_rlm(
        experiment,
        judge_model=args.judge_model,
        sql_warehouse_id=args.warehouse_id,
        max_results=args.max_results,
        max_reviews=args.max_reviews,
        sample_rate=args.sample_rate,
        min_tokens=args.min_tokens,
        rubric=rubric,
        reviewer_experiment_id=reviewer_experiment,
        max_turns=args.max_turns,
        temperature=args.temperature,
        reasoning_effort=reasoning_effort,
    )
    print(
        "[ail.jobs.continuous_rlm] "
        f"scanned={report.n_scanned} already_reviewed={report.n_already_reviewed} "
        f"reviewer_traces_skipped={report.n_reviewer_traces_skipped} "
        f"sampled_out={report.n_sampled_out} selected={report.n_selected} "
        f"reviewed={report.n_reviewed} failed={report.n_failed}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth_path = resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    print(f"[ail.jobs.continuous_rlm] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')}")

    if args.experiment:
        # Single-agent override: review JUST this experiment with the args' goal knobs
        # (goal_config=None => the args are the goal source, as before).
        return _run_rlm_for(
            args,
            experiment=args.experiment,
            reviewer_experiment=args.reviewer_experiment or None,
            goal_config=None,
            cohort=args.agent,
        )

    # Registry mode: review every agent in agent_registry, each with its own goal.
    missing = missing_registry_target(args.warehouse_id, args.catalog, args.schema)
    if missing:
        print(
            f"[ail.jobs.continuous_rlm] registry mode requires {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    agents = load_registered_agents(
        warehouse_id=args.warehouse_id, catalog=args.catalog, schema=args.schema
    )

    def per_agent(agent: Agent) -> int:
        return _run_rlm_for(
            args,
            experiment=agent.experiment_id,
            # The reviewer's own traces land in (and are skipped from) the agent's
            # experiment — preserving the old single-agent binding (reviewer=experiment)
            # per agent. An explicit --reviewer-experiment still overrides.
            reviewer_experiment=args.reviewer_experiment or agent.experiment_id,
            goal_config=agent.goal_config,
            cohort=agent.agent_name,
        )

    result = run_for_each_registered_agent(agents, per_agent, job_name="ail.jobs.continuous_rlm")
    return result.worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
