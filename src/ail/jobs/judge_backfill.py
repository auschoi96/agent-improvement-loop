"""Scheduled registry-driven judge coverage repair."""

from __future__ import annotations

import argparse
import os
import sys

from ail.events import append_memory_event
from ail.jobs.multi_agent import (
    load_registered_agents,
    missing_registry_target,
    run_for_each_registered_agent,
)
from ail.jobs.publish_job import resolve_job_auth
from ail.judges.backfill import run_judge_backfill
from ail.registry import Agent


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="")
    parser.add_argument("--reviewer-experiment", default="")
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument("--catalog", default=os.environ.get("AIL_CATALOG", ""))
    parser.add_argument("--schema", default=os.environ.get("AIL_SCHEMA", ""))
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error("--warehouse-id is required")
    return args


def _run_for(args: argparse.Namespace, agent: Agent) -> int:
    if not agent.reviewer_experiment_id:
        print(
            f"[ail.judge_backfill] agent={agent.agent_name} has no "
            "reviewer_experiment_id; refusing",
            file=sys.stderr,
        )
        return 2
    report = run_judge_backfill(
        agent.experiment_id,
        reviewer_experiment_id=agent.reviewer_experiment_id,
        sql_warehouse_id=args.warehouse_id,
        max_results=args.max_results,
        max_evaluations=args.max_evaluations,
        max_workers=args.max_workers,
    )
    print(
        f"[ail.judge_backfill] agent={agent.agent_name} scanned={report.n_scanned} "
        f"internal_skipped={report.n_internal_skipped} already_covered={report.n_already_covered} "
        f"selected={report.n_selected} evaluated={report.n_evaluated} failed={report.n_failed}"
    )
    for outcome in report.outcomes:
        if outcome.error:
            print(
                f"[ail.judge_backfill] {outcome.trace_id} {outcome.judge_name}: {outcome.error}",
                file=sys.stderr,
            )
    # Emit even for an idempotent no-op. If a prior run wrote assessments but its
    # event append failed, the Jobs retry can still wake memory after coverage is
    # already complete. The memory watermark makes redundant events cheap.
    if args.catalog and args.schema:
        append_memory_event(
            experiment_id=agent.experiment_id,
            source="judge_backfill",
            source_id=agent.agent_name,
            warehouse_id=args.warehouse_id,
            catalog=args.catalog,
            schema=args.schema,
        )
    return 1 if report.n_failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    if args.experiment:
        return _run_for(
            args,
            Agent(
                agent_name="manual",
                experiment_id=args.experiment,
                reviewer_experiment_id=args.reviewer_experiment or None,
            ),
        )
    missing = missing_registry_target(args.warehouse_id, args.catalog, args.schema)
    if missing:
        print(f"[ail.judge_backfill] registry mode requires {', '.join(missing)}", file=sys.stderr)
        return 2
    agents = load_registered_agents(
        warehouse_id=args.warehouse_id, catalog=args.catalog, schema=args.schema
    )
    return run_for_each_registered_agent(
        agents, lambda agent: _run_for(args, agent), job_name="ail.judge_backfill"
    ).worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
