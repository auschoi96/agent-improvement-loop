"""``ail-auto-align`` — the scheduled auto-align trigger's Databricks-Job entrypoint.

A thin **driver** over :func:`ail.judges.auto_align.auto_align_scorers`. On each
scheduled firing it, per judged dimension: aligns the judge with MemAlign once
enough human labels exist, re-aligns as more accrue, guards trust with the
agreement floor, and rolls back a regression — turning "a human adds labels" into
"the judge becomes trusted automatically". It reuses, and never reimplements, the
L2 pieces (see :mod:`ail.judges.auto_align`); the entrypoint only resolves the
Job's runtime concerns and prints what happened.

Registry-driven multi-agent: with no ``--experiment`` it runs REGISTRY MODE — it
reads every agent from the UC ``agent_registry`` (via :mod:`ail.jobs.multi_agent`)
and runs the cadence for each agent's OWN experiment, with per-agent isolation
(one agent's alignment failure is logged and the loop continues; the per-judge
watermark is scoped per experiment, so agents never disturb each other). Passing an
explicit ``--experiment`` is the single-agent override for local/manual runs.

Two runtime concerns it owns (both mirrored from :mod:`ail.jobs.publish_job` and
the optimization-cycle job):

* **Auth for the v4 trace store** — reading UC-backed traces rejects profile-only
  OAuth, so :func:`ail.jobs.publish_job.resolve_job_auth` resolves an explicit
  bearer (pre-set env → secret scope → minted from the run-as identity).
* **A SQL warehouse for the trace read** — the v4 trace store serves reads through
  a SQL warehouse; ``--warehouse-id`` is exported as ``MLFLOW_TRACING_SQL_WAREHOUSE_ID``
  so the in-process ``search_traces`` read picks it up (the same variable
  :mod:`ail.compare.monitoring` sets).

Scheduled, not event-triggered: the trace-store tables are views (no table-update
trigger is possible), the same reason the optimization cycle is scheduled.

Fail-closed and honest: a bad argument exits ``2`` with an actionable message; a
missing backend/dependency exits ``1``; and it never prints a fabricated success
— a judge held DISTRUSTED or rolled back is reported as such. The exit code is
non-zero only when a judge's cadence *failed* (raised), never when it correctly
held or rolled back.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.jobs.multi_agent import (
    load_registered_agents,
    missing_registry_target,
    run_for_each_registered_agent,
)
from ail.jobs.publish_job import resolve_job_auth
from ail.judges.agreement import DEFAULT_FLOOR, DEFAULT_MIN_SAMPLES, AgreementConfig
from ail.judges.alignment import MemAlignConfig, build_memalign_optimizer
from ail.judges.auto_align import (
    DEFAULT_LABEL_FLOOR,
    AutoAlignConfig,
    AutoAlignReport,
    auto_align_scorers,
)
from ail.judges.labeling import DEFAULT_ANCHOR_FRACTION
from ail.judges.registration import DEFAULT_SAMPLING_RATE
from ail.judges.scorers import DEFAULT_SCORERS, ScorerSpec
from ail.registry import Agent

if TYPE_CHECKING:
    from mlflow.genai.judges.base import AlignmentOptimizer


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-auto-align",
        description=(
            "Scheduled auto-align trigger: per judged dimension, align the judge with MemAlign "
            "once enough human labels exist, re-align as more accrue, guard trust with the "
            "agreement floor, and roll back a regression. Reuses ail.judges.auto_align."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="Explicit experiment id => single-agent override (align only that one). "
        "Empty (the default) => registry mode: align every agent in agent_registry.",
    )
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID"),
        help="SQL warehouse id for the v4 trace-store read (exported as "
        f"{TRACING_WAREHOUSE_ENV}). Required unless already set in the environment.",
    )
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
        "--judges",
        default="",
        help="Comma-separated judge names to auto-align (default: all built-in scorers). "
        "A judge with fewer than --label-floor labels simply skips.",
    )
    parser.add_argument(
        "--label-floor",
        type=int,
        default=DEFAULT_LABEL_FLOOR,
        help=f"Min human labels before a judge's first alignment (default {DEFAULT_LABEL_FLOOR}).",
    )
    parser.add_argument(
        "--agreement-floor",
        type=float,
        default=DEFAULT_FLOOR,
        help=f"Min judge-vs-human agreement to trust an aligned judge (default {DEFAULT_FLOOR}).",
    )
    parser.add_argument(
        "--min-anchor-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Min scored anchor items below which a measurement is insufficient (fail closed). "
        "Raise it so a tiny anchor cannot certify a judge.",
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for float-label agreement (e.g. 1.0 for a 1-5 graded judge).",
    )
    parser.add_argument(
        "--anchor-fraction",
        type=float,
        default=None,
        help="Fraction of labeled traces held out as the Human Anchor (default: labeling default).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=DEFAULT_SAMPLING_RATE,
        help=f"Sampling rate applied on promotion (default {DEFAULT_SAMPLING_RATE}).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Trace-fetch ceiling when reading labels (default: no cap).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model URI (e.g. 'databricks:/...'); omit for MLflow's default judge model.",
    )
    parser.add_argument(
        "--reflection-lm",
        default=None,
        help="MemAlign guideline-distillation model URI; omit for MLflow's default MemAlign.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="MemAlign episodic-memory embedding model URI (only used with --reflection-lm).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="MemAlign embedding dimension (only used with --reflection-lm).",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Dry run: run the full decision but register nothing and persist no watermark state.",
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
    if not args.warehouse_id and not os.environ.get(TRACING_WAREHOUSE_ENV):
        parser.error(
            "--warehouse-id is required (or set AIL_WAREHOUSE_ID / "
            f"{TRACING_WAREHOUSE_ENV}) so the v4 trace-store read can find a SQL warehouse"
        )
    return args


def _build_config(args: argparse.Namespace) -> AutoAlignConfig:
    agreement = AgreementConfig(
        floor=args.agreement_floor,
        numeric_tolerance=args.numeric_tolerance,
        min_samples=args.min_anchor_samples,
    )
    anchor_fraction = (
        args.anchor_fraction if args.anchor_fraction is not None else DEFAULT_ANCHOR_FRACTION
    )
    return AutoAlignConfig(
        label_floor=args.label_floor,
        agreement=agreement,
        anchor_fraction=anchor_fraction,
        sampling_rate=args.sampling_rate,
        max_results=args.max_results,
    )


def _build_optimizer(args: argparse.Namespace) -> AlignmentOptimizer | None:
    """A configured MemAlign optimizer when a reflection LM is given, else None.

    ``None`` lets ``align_judge`` fall through to MLflow's default MemAlign. When
    ``--reflection-lm`` is set, build one from the model URIs so the reflection /
    embedding models are driven through the gateway explicitly.
    """
    if not args.reflection_lm:
        return None
    config = MemAlignConfig(
        reflection_lm=args.reflection_lm, embedding_model=args.embedding_model or None
    )
    if args.embedding_dim is not None:
        config = replace(config, embedding_dim=args.embedding_dim)
    return build_memalign_optimizer(config)


def _resolve_scorers(names: str) -> dict[str, ScorerSpec]:
    """The scorer specs to auto-align, filtered by ``--judges`` (all by default)."""
    if not names.strip():
        return dict(DEFAULT_SCORERS)
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in DEFAULT_SCORERS]
    if unknown:
        raise ValueError(
            f"unknown judge name(s) {unknown}; known built-in scorers: {sorted(DEFAULT_SCORERS)}"
        )
    return {n: DEFAULT_SCORERS[n] for n in wanted}


def _align_for(
    args: argparse.Namespace,
    *,
    scorers: dict[str, ScorerSpec],
    optimizer: AlignmentOptimizer | None,
    config: AutoAlignConfig,
    experiment: str,
) -> int:
    """Run the auto-align cadence for one experiment — the reused single-agent body.

    ``auto_align_scorers`` is reused unchanged. Its per-judge experiment-tag watermark
    is scoped to ``experiment``, so each agent's cadence advances independently — one
    agent's labels/alignment never disturb another's. Returns non-zero only when a
    judge's cadence *failed*; a correct hold / rollback / skip is a successful run.

    ``profile`` is intentionally not forwarded: :func:`resolve_job_auth` already set
    an explicit ``DATABRICKS_HOST``/``TOKEN`` bearer and dropped the ambient profile,
    so downstream MLflow config must NOT re-add a profile (which would re-open the
    v4-store per-request OAuth fallback — see :mod:`ail.jobs.publish_job`).
    """
    report = auto_align_scorers(
        experiment,
        scorers=scorers,
        config=config,
        optimizer=optimizer,
        model=args.model or None,
        register=not args.no_register,
    )
    _print_report(report)
    return 1 if report.n_failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        scorers = _resolve_scorers(args.judges)
        config = _build_config(args)
        optimizer = _build_optimizer(args)
    except ValueError as exc:
        print(f"[ail-auto-align] invalid request: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"[ail-auto-align] {exc}", file=sys.stderr)
        return 1

    auth_path = resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    if args.warehouse_id:
        # The v4 trace store serves reads through a SQL warehouse; make the
        # in-process search_traces read find it.
        os.environ[TRACING_WAREHOUSE_ENV] = args.warehouse_id
    print(
        f"[ail.jobs.auto_align_job] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')} "
        f"judges={sorted(scorers)} label_floor={config.label_floor} "
        f"agreement_floor={config.agreement.floor} register={not args.no_register}"
    )

    if args.experiment:
        # Single-agent override: align JUST this experiment, exactly as before.
        print(f"[ail.jobs.auto_align_job] single-agent experiment={args.experiment}")
        return _align_for(
            args, scorers=scorers, optimizer=optimizer, config=config, experiment=args.experiment
        )

    # Registry mode: align every agent in agent_registry, each on its own experiment.
    warehouse: str = args.warehouse_id or os.environ.get(TRACING_WAREHOUSE_ENV) or ""
    missing = missing_registry_target(warehouse, args.catalog, args.schema)
    if missing:
        print(
            f"[ail.jobs.auto_align_job] registry mode requires {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    agents = load_registered_agents(
        warehouse_id=warehouse, catalog=args.catalog, schema=args.schema
    )

    def per_agent(agent: Agent) -> int:
        return _align_for(
            args,
            scorers=scorers,
            optimizer=optimizer,
            config=config,
            experiment=agent.experiment_id,
        )

    result = run_for_each_registered_agent(agents, per_agent, job_name="ail.jobs.auto_align_job")
    return result.worst_rc


def _print_report(report: AutoAlignReport) -> None:
    print(
        f"[ail.jobs.auto_align_job] aligned={report.n_aligned} "
        f"rolled_back={report.n_rolled_back} held_distrusted={report.n_held_distrusted} "
        f"skipped={report.n_skipped} failed={report.n_failed}"
    )
    for r in report.results:
        rate = "-" if r.agreement_rate is None else f"{r.agreement_rate:.3f}"
        note = r.notes[0] if r.notes else (r.error or "")
        print(
            f"    {r.judge_name:<20} {r.status.value:<22} labels={r.label_count} "
            f"watermark={r.watermark} agreement={rate}  {note}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
