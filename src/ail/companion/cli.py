"""Consolidated local companion CLI orchestration.

``python -m ail.companion`` — one local entrypoint for the companion roles.

This module is intentionally a thin CLI/orchestration layer. The work is delegated to
the existing deployer-run modules:

* planning: :mod:`ail.jobs.companion_planner`
* preview/commit execution: :mod:`ail.jobs.agent_executor`
* opt-in frozen-suite proving: :func:`ail.optimize.run_phase2_comparison`
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from ail.compare import ComparisonConfig, configure_monitoring_warehouse
from ail.ingest.adapters.claude_code import ClaudeCodeAdapter
from ail.ingest.adapters.codex import CodexAdapter
from ail.ingest.base import AgentAdapter
from ail.jobs import agent_executor, companion_planner
from ail.loop.verify_service import (
    run_verify_tick,
    select_pending_verify_requests,
    write_verify_result,
)
from ail.optimize import VerifySpec, run_phase2_comparison
from ail.publish import _build_workspace_client
from ail.task_suite.loader import DEFAULT_ARTIFACT_VERSION, load_task_suite

_TAG = "[ail.companion]"


def _refuse_profile(argv: list[str]) -> None:
    if "--profile" in argv or any(a.startswith("--profile=") for a in argv):
        raise SystemExit(
            f"{_TAG} refusing --profile OAuth auth. Export a STATIC DATABRICKS_HOST + "
            "DATABRICKS_TOKEN pinned to the workspace host instead."
        )


def _auth(args: argparse.Namespace) -> None:
    auth_path = companion_planner.resolve_static_auth(args)
    print(f"{_TAG} auth={auth_path} host={os.environ.get('DATABRICKS_HOST')}")


def run_plan(argv: list[str]) -> int:
    """Run the existing evidence-first companion planner once."""
    args = companion_planner._parse_args(argv)
    _auth(args)
    return companion_planner.run(args)


def run_execute(argv: list[str]) -> int:
    """Run the existing executor once: preview pending work and commit approvals."""
    args = agent_executor._parse_args(argv)
    _auth(args)
    return agent_executor.run(args)


def _load_run_plan(path: Path | None) -> dict[str, VerifySpec]:
    if path is None:
        return {}
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: dict[str, VerifySpec] = {}
    for task_id, entry in raw.items():
        if not isinstance(entry, dict) or "command" not in entry:
            raise ValueError(f"run-plan entry {task_id!r} must be a mapping with a 'command'")
        specs[str(task_id)] = VerifySpec(
            name=str(entry.get("name", f"verify-{task_id}")),
            command=entry["command"],
            cwd=entry.get("cwd"),
            shell=bool(entry.get("shell", False)),
            timeout_seconds=int(entry.get("timeout_seconds", 600)),
        )
    return specs


def _build_adapter(args: argparse.Namespace) -> AgentAdapter:
    if args.adapter == "claude_code":
        return ClaudeCodeAdapter(
            mlflow_experiment=args.experiment,
            default_allowed_tools=args.allowed_tools,
        )
    return CodexAdapter(
        command=args.codex_command,
        codex_home=args.codex_home,
        extra_args=args.codex_arg,
    )


def _prove_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ail.companion prove",
        description="Run opt-in Tier-2 frozen-suite verification for the current candidate.",
    )
    parser.add_argument("--suite-version", default=DEFAULT_ARTIFACT_VERSION)
    parser.add_argument("--suite-root", type=Path, default=None)
    parser.add_argument("--run-plan", type=Path, default=None)
    parser.add_argument("--fixtures-root", type=Path, default=None)
    parser.add_argument("--task-id", action="append", default=None, dest="task_ids")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--warehouse-id", default=None)
    parser.add_argument("--objective-metric", default="total_tokens")
    parser.add_argument("--min-reduction-pct", type=float, default=0.0)
    parser.add_argument("--adapter", choices=["claude_code", "codex"], default="claude_code")
    parser.add_argument("--allowed-tool", action="append", default=None, dest="allowed_tools")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-home", type=Path, default=None)
    parser.add_argument("--codex-arg", action="append", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2_companion.json"))
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    return parser


def run_prove(argv: list[str]) -> int:
    """Run the existing frozen-suite Phase-2 prover and write its artifact."""
    parser = _prove_parser()
    args = parser.parse_args(argv)
    if args.warehouse_id and not args.experiment_id:
        parser.error("--warehouse-id requires --experiment-id for monitoring provenance")
    _auth(args)

    if args.warehouse_id:
        configure_monitoring_warehouse(args.experiment_id, args.warehouse_id)

    artifact = run_phase2_comparison(
        suite=load_task_suite(args.suite_version, root=args.suite_root),
        adapter=_build_adapter(args),
        verify_specs=_load_run_plan(args.run_plan),
        config=ComparisonConfig(
            objective_metric=args.objective_metric,
            min_token_reduction_pct=args.min_reduction_pct,
        ),
        experiment=args.experiment,
        profile=None,
        warehouse_id=args.warehouse_id,
        task_ids=args.task_ids,
        fixtures_root=args.fixtures_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"{_TAG} proved {artifact.n_tasks} task(s): {artifact.n_promote} PROMOTE / "
        f"{artifact.n_block} BLOCK / {artifact.n_errored} ERRORED. artifact={args.output}"
    )
    return 2 if artifact.n_block > 0 or artifact.n_errored > 0 else 0


def _poll_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ail.companion poll",
        description=(
            "Bounded UC poll loop: run the existing executor each tick, optionally running "
            "the existing planner on a cadence."
        ),
    )
    parser.add_argument("--agent", default="claude_code")
    parser.add_argument("--registry", default="config/agents.yaml")
    parser.add_argument("--volume-root", default=os.environ.get("AIL_SNAPSHOT_VOLUME"))
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument("--catalog", default=agent_executor.DEFAULT_CATALOG)
    parser.add_argument("--schema", default=agent_executor.DEFAULT_SCHEMA)
    parser.add_argument("--operator", default=os.environ.get("USER") or "ail-agent-executor")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--plan-every",
        type=int,
        default=0,
        metavar="ITERATIONS",
        help="Run planner every N poll iterations; 0 disables planning in the loop.",
    )
    parser.add_argument("--experiment", default=None, help="Planner experiment id.")
    # In-app opt-in Tier-2 "verify on my suite" (L9): each enabled tick, pick up the
    # proposals a reviewer flagged for a frozen-suite proof and run the EXISTING prover
    # (run_phase2_comparison) on them, writing the result back to UC keyed to the
    # proposal. Opt-in: disabled by default (0) — the deployer turns it on.
    parser.add_argument(
        "--verify-every",
        type=int,
        default=0,
        metavar="ITERATIONS",
        help="Run the in-app verify-request handler every N poll iterations; "
        "0 disables 'verify on my suite'.",
    )
    parser.add_argument("--suite-version", default=DEFAULT_ARTIFACT_VERSION)
    parser.add_argument("--suite-root", type=Path, default=None)
    parser.add_argument("--run-plan", type=Path, default=None)
    parser.add_argument("--fixtures-root", type=Path, default=None)
    parser.add_argument("--adapter", choices=["claude_code", "codex"], default="claude_code")
    parser.add_argument("--allowed-tool", action="append", default=None, dest="allowed_tools")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-home", type=Path, default=None)
    parser.add_argument("--codex-arg", action="append", default=None)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--objective-metric", default="total_tokens")
    parser.add_argument("--goal-direction", default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--goal-target", type=float, default=-0.30)
    parser.add_argument("--goal-target-kind", default="relative", choices=["relative", "absolute"])
    parser.add_argument("--guardrail-judge", action="append", default=None)
    parser.add_argument("--objective-baseline", type=companion_planner.oc._opt_float, default=None)
    parser.add_argument("--goal-confirmed", default=os.environ.get("AIL_CONFIRM_GOAL", "false"))
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    return parser


def _add_optional(argv: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _add_repeated(argv: list[str], flag: str, values: list[str] | None) -> None:
    for value in values or []:
        argv.extend([flag, str(value)])


def _executor_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = [
        "--agent",
        args.agent,
        "--registry",
        args.registry,
        "--catalog",
        args.catalog,
        "--schema",
        args.schema,
        "--operator",
        args.operator,
    ]
    _add_optional(argv, "--volume-root", args.volume_root)
    _add_optional(argv, "--host", args.host)
    _add_optional(argv, "--warehouse-id", args.warehouse_id)
    _add_optional(argv, "--model", args.model)
    _add_optional(argv, "--timeout", args.timeout)
    _add_optional(argv, "--token-secret-scope", args.token_secret_scope)
    _add_optional(argv, "--token-secret-key", args.token_secret_key)
    if args.trace:
        argv.append("--trace")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def _planner_argv(args: argparse.Namespace) -> list[str]:
    if not args.experiment:
        raise SystemExit("--experiment is required when --plan-every is enabled")
    argv: list[str] = [
        "--agent",
        args.agent,
        "--experiment",
        args.experiment,
        "--catalog",
        args.catalog,
        "--schema",
        args.schema,
        "--max-results",
        str(args.max_results),
        "--objective-metric",
        args.objective_metric,
        "--goal-direction",
        args.goal_direction,
        "--goal-target",
        str(args.goal_target),
        "--goal-target-kind",
        args.goal_target_kind,
        "--goal-confirmed",
        args.goal_confirmed,
    ]
    _add_optional(argv, "--host", args.host)
    _add_optional(argv, "--warehouse-id", args.warehouse_id)
    _add_optional(argv, "--planner-model", args.planner_model)
    _add_optional(argv, "--objective-baseline", args.objective_baseline)
    _add_repeated(argv, "--guardrail-judge", args.guardrail_judge)
    _add_optional(argv, "--token-secret-scope", args.token_secret_scope)
    _add_optional(argv, "--token-secret-key", args.token_secret_key)
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def _run_verify_once(args: argparse.Namespace, *, client: Any, warehouse_id: str) -> None:
    """Wire the live seams and run one in-app verify-request tick (fail-soft).

    Reuses the existing frozen-suite prover (:func:`run_phase2_comparison`) — proving
    is never reimplemented here. An idle tick only runs the SELECT (cheap); the suite
    and adapter are loaded and the prover runs only when a request is present. Each
    request's terminal state (verified / blocked / errored / no_suite) is written
    honestly by :func:`run_verify_tick`; this outer guard only stops an infra hiccup
    (e.g. a pre-migration table missing the verify_* columns) from crashing the poll.
    """

    def _select() -> list[dict[str, Any]]:
        return select_pending_verify_requests(
            client=client,
            warehouse_id=warehouse_id,
            agent_name=args.agent,
            catalog=args.catalog,
            schema=args.schema,
        )

    def _load_suite() -> Any:
        return load_task_suite(args.suite_version, root=args.suite_root)

    def _prove(suite: Any) -> Any:
        return run_phase2_comparison(
            suite=suite,
            adapter=_build_adapter(args),
            verify_specs=_load_run_plan(args.run_plan),
            config=ComparisonConfig(objective_metric=args.objective_metric),
            experiment=args.experiment,
            profile=None,
            warehouse_id=warehouse_id,
            fixtures_root=args.fixtures_root,
        )

    def _write(**kwargs: Any) -> None:
        write_verify_result(
            client=client,
            warehouse_id=warehouse_id,
            agent_name=args.agent,
            catalog=args.catalog,
            schema=args.schema,
            **kwargs,
        )

    try:
        summary = run_verify_tick(
            agent_name=args.agent,
            select_requested=_select,
            load_suite=_load_suite,
            run_prover=_prove,
            write_result=_write,
        )
    except Exception as exc:  # noqa: BLE001 - a verify infra hiccup must not crash the executor poll
        print(f"{_TAG} verify tick skipped ({type(exc).__name__}: {exc})")
        return
    if summary.n_requested:
        print(
            f"{_TAG} verify tick: {summary.n_requested} request(s) -> "
            f"{summary.n_verified} verified / {summary.n_blocked} blocked / "
            f"{summary.n_errored} errored / {summary.n_no_suite} no-suite"
        )


def run_poll(argv: list[str]) -> int:
    parser = _poll_parser()
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")
    if args.interval_seconds < 0:
        parser.error("--interval-seconds must be >= 0")
    if args.plan_every < 0:
        parser.error("--plan-every must be >= 0")
    if args.verify_every < 0:
        parser.error("--verify-every must be >= 0")

    exec_args = agent_executor._parse_args(_executor_argv(args))
    plan_args = companion_planner._parse_args(_planner_argv(args)) if args.plan_every else None
    _auth(exec_args)

    # Opt-in Tier-2 verify polling: build the workspace client once, up front. Fail
    # closed on a missing warehouse — never silently drop verify requests.
    verify_client: Any | None = None
    verify_warehouse = args.warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if args.verify_every:
        if not verify_warehouse:
            print(
                f"{_TAG} verify disabled — no warehouse id (set --warehouse-id or AIL_WAREHOUSE_ID)"
            )
        else:
            verify_client = _build_workspace_client(None)

    exit_code = 0
    for iteration in range(1, args.max_iterations + 1):
        print(f"{_TAG} poll iteration {iteration}/{args.max_iterations}")
        if plan_args is not None and (iteration - 1) % args.plan_every == 0:
            exit_code = max(exit_code, companion_planner.run(plan_args))
        exit_code = max(exit_code, agent_executor.run(exec_args))
        run_verify = verify_client is not None and (iteration - 1) % args.verify_every == 0
        if run_verify and verify_warehouse:
            _run_verify_once(args, client=verify_client, warehouse_id=verify_warehouse)
        if iteration < args.max_iterations:
            time.sleep(args.interval_seconds)
    return exit_code


def _usage() -> str:
    return (
        "usage: python -m ail.companion {plan,execute,prove,poll,run} ...\n\n"
        "Subcommands:\n"
        "  plan     evidence-first proposal planning (delegates to ail.jobs.companion_planner)\n"
        "  execute  one executor pass: preview pending, commit approved\n"
        "  prove    opt-in frozen-suite Phase-2 verification\n"
        "  poll     bounded loop over executor work, optionally planning on cadence\n"
        "  run      alias for poll\n"
    )


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    _refuse_profile(raw)
    command, rest = raw[0], raw[1:]
    if command == "plan":
        return run_plan(rest)
    if command == "execute":
        return run_execute(rest)
    if command == "prove":
        return run_prove(rest)
    if command in {"poll", "run"}:
        return run_poll(rest)
    raise SystemExit(f"{_usage()}\nunknown subcommand: {command}")
