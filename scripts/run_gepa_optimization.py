#!/usr/bin/env python3
"""Drive the live GEPA agent-optimization loop (the ORCHESTRATOR's entry point).

Stage 5 of the loop: evolve the token-efficiency skill body with GEPA against the
**train split** of the frozen Task Suite — fitness is the harness PROMOTE decision
plus the realized L0 token reduction, fail-closed on execution / L1 correctness —
then validate the selected candidate **and** the seed body on the **held-out
split** through the live harness. Writes a :class:`~ail.optimize.gepa_runner.GepaOptimizationResult`
JSON: the evolved skill body (a CANDIDATE), its held-out result vs the seed's, and
the train/held-out split. It does **not** apply or promote the artifact — promotion
is a separate human step (the human gate).

This script is a *thin* wrapper: all logic lives in (and is unit-tested in)
:mod:`ail.optimize.gepa_runner` and :mod:`ail.optimize.phase2`. It is **live and
costly** — every GEPA fitness evaluation runs the real agent (a baseline arm + a
candidate arm per train task), and the reflection LM is called for every mutation.

**Guarded.** It refuses to run unless ``AIL_LIVE_GEPA=1`` is set, so it can never be
triggered by accident (and it lives under ``scripts/`` so pytest never collects it).
The library function is the offline-tested seam; **do not run this in CI or a
worktree smoke test.**

Reflection LM
-------------
``--reflection-lm`` defaults to the MLflow URI ``databricks:/databricks-claude-sonnet-4-6``;
it is normalized to litellm's ``databricks/<model>`` form at the GEPA boundary, so
the environment must carry Databricks credentials (e.g. ``DATABRICKS_HOST`` +
``DATABRICKS_TOKEN``, or a configured ``--profile``).

Example
-------
    AIL_LIVE_GEPA=1 python scripts/run_gepa_optimization.py \\
        --suite-version phase2-mini \\
        --experiment /Shared/dais-demo-agent-improvement \\
        --profile dais-demo \\
        --run-plan run_plan.yaml \\
        --holdout-id ts-route-05 --holdout-id ts-config-04 \\
        --max-metric-calls 60 \\
        --output artifacts/gepa_candidate.json

The run plan (JSON or YAML) maps task ids to L1 verification commands — the same
format as ``scripts/run_phase2_comparison.py``. A task with no entry has no
correctness signal and scores zero (fail-closed). See ``docs/GEPA_OPTIMIZATION.md``
for the fitness wiring, the train/held-out wall, and the human gate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from ail.compare import configure_monitoring_warehouse
from ail.ingest.adapters.claude_code import ClaudeCodeAdapter
from ail.optimize import VerifySpec
from ail.optimize.gepa_runner import DEFAULT_REFLECTION_LM, GepaConfig, run_gepa_optimization
from ail.task_suite.loader import load_task_suite

_LIVE_ENV = "AIL_LIVE_GEPA"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite-version", default="phase2-mini", help="frozen Task Suite artifact dir")
    p.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment NAME the adapter logs each run to (Databricks-managed MLflow)",
    )
    p.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile (sets DATABRICKS_CONFIG_PROFILE for the adapter + reflect LM)",
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
        "--holdout-id",
        action="append",
        default=None,
        dest="holdout_ids",
        help="held-out task id (repeatable); GEPA never trains on these. Overrides the fraction",
    )
    p.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.4,
        help="fraction of the suite held out for final validation (used if no --holdout-id)",
    )
    p.add_argument(
        "--max-metric-calls",
        type=int,
        default=50,
        help="GEPA total evaluation budget — the dominant cost dial (each call runs the agent)",
    )
    p.add_argument(
        "--max-train-tasks",
        type=int,
        default=None,
        help="cap on how many train tasks GEPA optimizes against (cost bound)",
    )
    p.add_argument(
        "--reflection-lm",
        default=DEFAULT_REFLECTION_LM,
        help="reflection/teacher LM URI for reflective mutation",
    )
    p.add_argument("--seed", type=int, default=0, help="deterministic seed for the split + GEPA")
    p.add_argument("--objective-metric", default="total_tokens", help="L0 metric to reduce")
    p.add_argument(
        "--min-reduction-pct",
        type=float,
        default=0.0,
        help="minimum %% token reduction for the objective (default: any strict reduction)",
    )
    p.add_argument(
        "--seed-skill-file",
        type=Path,
        default=None,
        help="optional file holding the seed skill body; default is the token-efficiency skill",
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
        default=Path("artifacts/gepa_candidate.json"),
        help="where to write the GEPA optimization-result JSON (the CANDIDATE)",
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


def main(argv: list[str] | None = None) -> int:
    if os.environ.get(_LIVE_ENV) != "1":
        raise SystemExit(
            f"refusing to run the live GEPA loop: set {_LIVE_ENV}=1 to confirm. This runs the "
            "real agent for every fitness evaluation and calls the reflection LM — it is costly "
            "and must never run in CI. The offline-tested seam is ail.optimize.gepa_runner."
        )

    args = _parse_args(argv)

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    if args.warehouse:
        if not args.experiment_id:
            raise SystemExit("--warehouse requires --experiment-id (the experiment to tag)")
        configure_monitoring_warehouse(args.experiment_id, args.warehouse)

    suite = load_task_suite(args.suite_version)
    verify_specs = _load_run_plan(args.run_plan)
    seed_body = args.seed_skill_file.read_text(encoding="utf-8") if args.seed_skill_file else None

    adapter = ClaudeCodeAdapter(
        mlflow_experiment=args.experiment,
        default_allowed_tools=args.allowed_tools,
    )

    config = GepaConfig(
        objective_metric=args.objective_metric,
        min_token_reduction_pct=args.min_reduction_pct,
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_metric_calls,
        holdout_fraction=args.holdout_fraction,
        max_train_tasks=args.max_train_tasks,
        seed=args.seed,
    )

    result = run_gepa_optimization(
        suite=suite,
        adapter=adapter,
        seed_skill_body=seed_body,
        verify_specs=verify_specs,
        config=config,
        holdout_task_ids=args.holdout_ids,
        fixtures_root=str(args.fixtures_root) if args.fixtures_root else None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print(
        f"GEPA optimization complete: candidate {'CHANGED' if result.changed else 'UNCHANGED'} "
        f"from seed (GEPA metric calls: {result.gepa_total_metric_calls}, "
        f"candidates: {result.gepa_num_candidates}).",
        file=sys.stderr,
    )
    print(
        "Train ids: "
        + ", ".join(result.train_task_ids)
        + " | held-out ids (validation only): "
        + ", ".join(result.holdout_task_ids),
        file=sys.stderr,
    )
    evolved = result.holdout_evolved
    seed = result.holdout_seed_baseline
    if evolved is not None and seed is not None:
        print(
            "Held-out realized token savings: "
            f"evolved {evolved.realized_token_savings_pct}% "
            f"({evolved.n_promote}/{evolved.n_tasks} PROMOTE) vs "
            f"seed {seed.realized_token_savings_pct}% "
            f"({seed.n_promote}/{seed.n_tasks} PROMOTE); "
            f"delta {result.holdout_savings_delta_pct} pct-points.",
            file=sys.stderr,
        )
    print(
        f"HUMAN GATE: this is a CANDIDATE artifact, NOT promoted. Review {args.output} and "
        "promote separately if the held-out result holds up.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
