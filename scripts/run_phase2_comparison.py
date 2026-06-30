#!/usr/bin/env python3
"""Drive the Phase-2 token-efficiency comparison (the ORCHESTRATOR's entry point).

Runs the frozen Task Suite through the BASELINE (no asset) and CANDIDATE
(token-efficiency skill) configs with the Claude Code adapter, gates correctness
on deterministic **L1 programmatic** checks (no LLM judge in the decision path),
and writes a :class:`~ail.optimize.phase2.Phase2Artifact` JSON with, per task, the
L0 token delta + the L1 correctness outcome + the harness PROMOTE/BLOCK decision.

This script is a *thin* wrapper: all logic lives in (and is unit-tested in)
:mod:`ail.optimize.phase2`. It is the only layer that does workspace I/O — set the
Databricks profile, optionally wire the monitoring SQL warehouse, build the live
adapter, load the run plan. **It runs the real, costly comparison; do not run it
in CI or a worktree smoke test** (the library function is the tested seam).

Example
-------
    python scripts/run_phase2_comparison.py \
        --suite-version v1 \
        --experiment /Shared/dais-demo-agent-improvement \
        --profile dais-demo \
        --warehouse <sql-warehouse-id> --experiment-id 660599403165942 \
        --run-plan run_plan.yaml \
        --output artifacts/phase2_token_lever.json

The run plan (JSON or YAML) maps task ids to L1 verification commands; a task with
no entry has no correctness signal and is blocked (fail-closed). Example::

    ts-017:
      name: deploy-smoke
      command: ["uv", "run", "deploy/databricks/deploy.py", "--check"]
      cwd: /abs/path/to/checkout
      timeout_seconds: 600
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from ail.compare import ComparisonConfig, configure_monitoring_warehouse
from ail.ingest.adapters.claude_code import ClaudeCodeAdapter
from ail.optimize import VerifySpec, run_phase2_comparison
from ail.task_suite.loader import load_task_suite


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite-version", default="v1", help="frozen Task Suite artifact directory")
    p.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment NAME the adapter logs the run to (Databricks-managed MLflow)",
    )
    p.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile (sets DATABRICKS_CONFIG_PROFILE for the adapter)",
    )
    p.add_argument(
        "--warehouse",
        default=None,
        help="SQL warehouse id for live trace reads (requires --experiment-id)",
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help="numeric MLflow experiment id to tag with the monitoring warehouse",
    )
    p.add_argument(
        "--run-plan",
        type=Path,
        default=None,
        help="JSON/YAML mapping task_id -> L1 verification command (see module docstring)",
    )
    p.add_argument(
        "--task-id",
        action="append",
        default=None,
        dest="task_ids",
        help="restrict to these task ids (repeatable); default runs the whole suite",
    )
    p.add_argument("--objective-metric", default="total_tokens", help="L0 metric to reduce")
    p.add_argument(
        "--min-reduction-pct",
        type=float,
        default=0.0,
        help="minimum %% token reduction for the objective (default: any strict reduction)",
    )
    p.add_argument(
        "--allowed-tool",
        action="append",
        default=None,
        dest="allowed_tools",
        help="tool the agent may use (repeatable); default is the adapter's built-in set",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2_token_lever.json"),
        help="where to write the Phase-2 artifact JSON",
    )
    return p.parse_args(argv)


def _load_run_plan(path: Path | None) -> dict[str, VerifySpec]:
    """Parse a JSON/YAML run plan into ``{task_id: VerifySpec}``."""
    if path is None:
        return {}
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: dict[str, VerifySpec] = {}
    for task_id, entry in raw.items():
        if not isinstance(entry, dict) or "command" not in entry:
            raise ValueError(f"run-plan entry {task_id!r} must be a mapping with a 'command'")
        specs[task_id] = VerifySpec(
            name=str(entry.get("name", f"verify-{task_id}")),
            command=entry["command"],
            cwd=entry.get("cwd"),
            shell=bool(entry.get("shell", False)),
            timeout_seconds=int(entry.get("timeout_seconds", 600)),
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    if args.warehouse:
        if not args.experiment_id:
            raise SystemExit("--warehouse requires --experiment-id (the experiment to tag)")
        configure_monitoring_warehouse(args.experiment_id, args.warehouse)

    suite = load_task_suite(args.suite_version)
    verify_specs = _load_run_plan(args.run_plan)

    adapter = ClaudeCodeAdapter(
        mlflow_experiment=args.experiment,
        default_allowed_tools=args.allowed_tools,
    )

    artifact = run_phase2_comparison(
        suite=suite,
        adapter=adapter,
        verify_specs=verify_specs,
        config=ComparisonConfig(
            objective_metric=args.objective_metric,
            min_token_reduction_pct=args.min_reduction_pct,
        ),
        experiment=args.experiment,
        profile=args.profile,
        warehouse_id=args.warehouse,
        task_ids=args.task_ids,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")

    print(
        f"Phase-2 comparison complete: {artifact.n_promote} PROMOTE / "
        f"{artifact.n_block} BLOCK / {artifact.n_errored} ERRORED over {artifact.n_tasks} task(s).",
        file=sys.stderr,
    )
    print(
        f"Realized token savings (PROMOTE only): "
        f"{artifact.realized_token_savings_absolute:g} tokens "
        f"({artifact.realized_token_savings_pct}%). Artifact: {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
