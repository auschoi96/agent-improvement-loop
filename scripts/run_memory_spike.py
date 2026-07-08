#!/usr/bin/env python3
"""Drive the Phase A-0 advisory-memory spike (BASELINE vs MEMORY_CANDIDATE).

Runs the frozen ``phase2-mini`` suite through :data:`~ail.optimize.lever.BASELINE`
(no asset) and :data:`~ail.memory.config.MEMORY_CANDIDATE` (the top-k RLM learnings
injected as advisory context) with the Claude Code adapter, gating correctness on
deterministic **L1 programmatic** checks (no LLM judge in the decision path) via the
*unchanged* Phase-2 machinery (:func:`ail.optimize.phase2.run_phase2_comparison`).
The only difference from ``scripts/run_phase2_comparison.py`` is the candidate arm:
here it is the advisory-memory intervention instead of the token-efficiency skill.

This is a MINIMAL "prove value before complexity" experiment — no Lakebase, no
memory-writer job, no store, no deployer injection, no organic traces (all
deferred). It answers one question: does the injected memory reduce tokens on the
frozen suite without regressing L1 correctness?

**Guarded, fail-closed to a dry no-op.** It runs the real, costly agent arms only
when ``AIL_LIVE_MEMORY=1`` is set (mirroring ``AIL_LIVE_GEPA`` in
``scripts/run_gepa_optimization.py``). Without it the script performs a **dry
no-op**: it loads the suite, builds the memory candidate, and runs the
teaching-to-the-test provenance guard, then prints what it *would* do and exits —
never spawning an agent arm. It lives under ``scripts/`` so pytest never collects
it; the offline-tested seams are :mod:`ail.memory` and :mod:`ail.optimize.phase2`.

Example (live)
--------------
    AIL_LIVE_MEMORY=1 python scripts/run_memory_spike.py \\
        --suite-version phase2-mini \\
        --report artifacts/rlm_batch_report.json \\
        --top-k 8 \\
        --run-plan run_plan.yaml \\
        --experiment /Shared/dais-demo-agent-improvement \\
        --profile dais-demo \\
        --output artifacts/memory_spike.json

The run plan (JSON or YAML) maps task ids to L1 verification commands — the same
format as ``scripts/run_phase2_comparison.py``. A task with no entry has no
correctness signal and is blocked (fail-closed).
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
from ail.memory import assert_memory_disjoint_from_suite, build_memory_candidate, load_ranked_assets
from ail.optimize import VerifySpec, run_phase2_comparison
from ail.optimize.lever import BASELINE
from ail.optimize.phase2 import Phase2Artifact, TaskOutcome
from ail.task_suite.loader import load_task_suite

_LIVE_ENV = "AIL_LIVE_MEMORY"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite-version", default="phase2-mini", help="frozen Task Suite artifact dir")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="RLM batch report JSON (memory source); default artifacts/rlm_batch_report.json",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="how many learnings (top-k by cross-trace recurrence) to inject",
    )
    p.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment NAME the adapter logs each run to (Databricks-managed MLflow)",
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
        "--fixtures-root",
        type=Path,
        default=None,
        help="root containing eval/phase2_fixtures (per-arm isolation); default is repo discovery",
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
        default=Path("artifacts/memory_spike.json"),
        help="where to write the spike artifact JSON",
    )
    return p.parse_args(argv)


def _load_run_plan(path: Path | None) -> dict[str, VerifySpec]:
    """Parse a JSON/YAML run plan into ``{task_id: VerifySpec}`` (same format as Phase 2)."""
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


def _print_task_line(o: TaskOutcome) -> None:
    """One per-task line: token delta + L1 correctness + PROMOTE/BLOCK (phase2 shape)."""
    pct = "n/a" if o.token_delta_pct is None else f"{o.token_delta_pct:+.2f}%"
    print(
        f"  {o.task_id:<16} {o.recommendation.value:<7} "
        f"tokens {o.baseline_total_tokens:g} -> {o.candidate_total_tokens:g} ({pct}) | "
        f"L1={o.l1_outcome.value}",
        file=sys.stderr,
    )


def _print_summary(artifact: Phase2Artifact, output: Path) -> None:
    for o in artifact.outcomes:
        _print_task_line(o)
    print(
        f"Advisory-memory spike complete: {artifact.n_promote} PROMOTE / "
        f"{artifact.n_block} BLOCK / {artifact.n_errored} ERRORED over {artifact.n_tasks} task(s).",
        file=sys.stderr,
    )
    print(
        f"Realized token savings (PROMOTE only): "
        f"{artifact.realized_token_savings_absolute:g} tokens "
        f"({artifact.realized_token_savings_pct}%). Artifact: {output}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    suite = load_task_suite(args.suite_version)
    candidate = build_memory_candidate(args.report, k=args.top_k)
    n_learnings = len(candidate.intervention.learnings) if candidate.intervention is not None else 0

    # Provenance wall: prove the memory was not distilled from the eval traces
    # (teaching to the test) before running anything — in dry mode too.
    assert_memory_disjoint_from_suite(assets=load_ranked_assets(args.report), suite=suite)

    if n_learnings == 0:
        print(
            "WARNING: no learnings loaded from the RLM report "
            f"({args.report or 'artifacts/rlm_batch_report.json'}); MEMORY_CANDIDATE is a no-op "
            "identical to the baseline. Point --report at the report to inject memory.",
            file=sys.stderr,
        )

    if os.environ.get(_LIVE_ENV) != "1":
        # Dry no-op: everything except the costly agent arms.
        print(
            f"DRY NO-OP (set {_LIVE_ENV}=1 to run the real, costly agent arms). Would run BASELINE "
            f"vs {candidate.name} on suite {suite.version!r} ({len(suite.tasks)} task(s)) with "
            f"{n_learnings} injected learning(s), objective={args.objective_metric}. The offline "
            "seams are ail.memory and ail.optimize.phase2.",
            file=sys.stderr,
        )
        return 0

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    if args.warehouse:
        if not args.experiment_id:
            raise SystemExit("--warehouse requires --experiment-id (the experiment to tag)")
        configure_monitoring_warehouse(args.experiment_id, args.warehouse)

    verify_specs = _load_run_plan(args.run_plan)

    adapter = ClaudeCodeAdapter(
        mlflow_experiment=args.experiment,
        default_allowed_tools=args.allowed_tools,
    )

    artifact = run_phase2_comparison(
        suite=suite,
        adapter=adapter,
        candidate=candidate,
        baseline=BASELINE,
        verify_specs=verify_specs,
        config=ComparisonConfig(
            objective_metric=args.objective_metric,
            min_token_reduction_pct=args.min_reduction_pct,
        ),
        experiment=args.experiment,
        profile=args.profile,
        warehouse_id=args.warehouse,
        task_ids=args.task_ids,
        fixtures_root=args.fixtures_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")

    _print_summary(artifact, args.output)
    return 0 if artifact.n_errored == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
