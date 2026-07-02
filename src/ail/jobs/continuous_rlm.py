"""Databricks Job entrypoint for the scheduled continuous RLM review.

Fired on a schedule (``resources/continuous_rlm.job.yml``), not a trace-arrival
trigger: the UC-backed trace store is a VIEW, so a ``table_update`` trigger is
infeasible. Each firing runs one bounded :func:`ail.l3.continuous.run_continuous_rlm`
pass — sampling, idempotency, and the fail-closed failed-review marker all live there.

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

from ail.jobs.publish_job import resolve_job_auth
from ail.l3.continuous import run_continuous_rlm
from ail.l3.reviewer import normalize_reasoning_effort
from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric

#: Most powerful VIABLE HALO judge on the Databricks gateway (see module docstring).
DEFAULT_JUDGE_MODEL = "databricks-gpt-5-5-pro"


def _build_rubric(args: argparse.Namespace) -> ReviewRubric:
    """Build the review rubric: goal-steered when configured, else the default rubric.

    Mirrors :func:`ail.jobs.optimization_cycle._build_goal` (identical goal knobs),
    then derives a :class:`~ail.l3.rubric.ReviewRubric` from the compiled goal via
    :func:`ail.l3.goal_rubric.rubric_from_goal`. An empty ``--objective-metric`` (the
    fallback) keeps :data:`~ail.l3.rubric.DEFAULT_RUBRIC`. The ``CompiledGoal`` is
    fully validated (allowlist + readiness contract), so a misconfigured goal fails
    loud here rather than silently reviewing against a fabricated objective. No
    confirmation gate: this is a read-only review that attaches assessments, not an
    apply.
    """
    if not args.objective_metric.strip():
        return DEFAULT_RUBRIC

    from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
    from ail.l3.goal_rubric import rubric_from_goal

    guardrails: list[Guardrail] = []
    for spec in args.guardrail_judge or []:
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
        objective_metric=args.objective_metric,
        direction=args.goal_direction,
        target=GoalTarget(value=args.goal_target, kind=args.goal_target_kind),
        guardrails=tuple(guardrails),
        cohort=args.agent,
    )
    return rubric_from_goal(goal)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one bounded scheduled RLM/HALO pass over recent MLflow traces."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
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
    # goal-steering (same knobs as ail-optimization-cycle). Empty --objective-metric
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth_path = resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    rubric = _build_rubric(args)
    # Normalize the effort input HERE (the CLI/job boundary): empty / 'none' / 'auto'
    # mean "no override, auto-resolve" and must become None rather than a literal effort
    # injected into HALO. build_engine_config re-applies this defensively.
    reasoning_effort = normalize_reasoning_effort(args.reasoning_effort)
    print(
        f"[ail.jobs.continuous_rlm] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')} "
        f"experiment={args.experiment} judge_model={args.judge_model} "
        f"reasoning_effort={reasoning_effort or 'auto'} rubric={rubric.rubric_id} "
        f"sample_rate={args.sample_rate} max_reviews={args.max_reviews}"
    )
    report = run_continuous_rlm(
        args.experiment,
        judge_model=args.judge_model,
        sql_warehouse_id=args.warehouse_id,
        max_results=args.max_results,
        max_reviews=args.max_reviews,
        sample_rate=args.sample_rate,
        min_tokens=args.min_tokens,
        rubric=rubric,
        reviewer_experiment_id=args.reviewer_experiment or None,
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


if __name__ == "__main__":
    raise SystemExit(main())
