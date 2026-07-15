#!/usr/bin/env python3
"""Move Claude Code subject traces into isolated Traces-in-UC experiments."""

from __future__ import annotations

import argparse
import os

import mlflow

from ail.jobs.bootstrap_tables import _read_rows
from ail.jobs.multi_agent import load_registered_agents
from ail.onboarding.service import _ensure_baseline_judges, _ensure_bootstrap_experiment
from ail.publish import _build_workspace_client, _execute
from ail.publish_versions import REGISTRY_TABLE, publish_registry
from ail.registry import Agent, AgentRegistry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--old-experiment", required=True)
    parser.add_argument("--subject-name", default="claude-code-traces-uc")
    parser.add_argument("--reviewer-name", default="claude-code-ail-internal-uc")
    parser.add_argument("--trace-catalog", required=True)
    parser.add_argument("--trace-schema", default="mlflow_traces")
    parser.add_argument("--source-prefix", default="cc")
    parser.add_argument("--subject-prefix", default="claude_code")
    parser.add_argument("--reviewer-prefix", default="claude_code_internal")
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--agent-name", default="claude_code")
    parser.add_argument("--target-workspace", required=True)
    parser.add_argument("--skip-judges", action="store_true")
    parser.add_argument("--reset-target", action="store_true")
    return parser.parse_args()


def _fqn(catalog: str, schema: str, table: str) -> str:
    return f"`{catalog}`.`{schema}`.`{table}`"


def _copy_uc_traces(
    client: object,
    warehouse_id: str,
    catalog: str,
    schema: str,
    source_prefix: str,
    target_prefix: str,
    *,
    reset_target: bool,
) -> tuple[int, int, int]:
    source_metadata = _fqn(catalog, schema, f"{source_prefix}_trace_metadata")
    source_spans = _fqn(catalog, schema, f"{source_prefix}_otel_spans")
    source_logs = _fqn(catalog, schema, f"{source_prefix}_otel_logs")
    source_annotations = _fqn(catalog, schema, f"{source_prefix}_otel_annotations")
    target_spans = _fqn(catalog, schema, f"{target_prefix}_otel_spans")
    target_logs = _fqn(catalog, schema, f"{target_prefix}_otel_logs")
    target_metrics = _fqn(catalog, schema, f"{target_prefix}_otel_metrics")
    target_annotations = _fqn(catalog, schema, f"{target_prefix}_otel_annotations")
    target_metadata = _fqn(catalog, schema, f"{target_prefix}_trace_metadata")
    subject = "m.tags['mlflow.traceName'] = 'claude_code_conversation'"

    if reset_target:
        for table in (target_annotations, target_logs, target_metrics, target_spans):
            _execute(client, warehouse_id, f"DELETE FROM {table}")

    _execute(
        client,
        warehouse_id,
        f"""INSERT INTO {target_spans}
        SELECT s.* FROM {source_spans} s
        WHERE EXISTS (
          SELECT 1 FROM {source_metadata} m WHERE m.trace_id = s.trace_id AND {subject}
        ) AND NOT EXISTS (
          SELECT 1 FROM {target_spans} t WHERE t.record_id = s.record_id
        )""",
    )
    _execute(
        client,
        warehouse_id,
        f"""INSERT INTO {target_logs}
        SELECT l.* FROM {source_logs} l
        WHERE EXISTS (
          SELECT 1 FROM {source_metadata} m WHERE m.trace_id = l.trace_id AND {subject}
        ) AND NOT EXISTS (
          SELECT 1 FROM {target_logs} t WHERE t.record_id = l.record_id
        )""",
    )
    _execute(
        client,
        warehouse_id,
        f"""INSERT INTO {target_annotations}
        SELECT a.* FROM {source_annotations} a
        WHERE EXISTS (
          SELECT 1 FROM {source_metadata} m WHERE m.trace_id = a.target_id AND {subject}
        ) AND NOT EXISTS (
          SELECT 1 FROM {target_annotations} t WHERE t.annotation_id = a.annotation_id
        )""",
    )

    source_counts = _read_rows(
        client,
        warehouse_id,
        f"""SELECT COUNT(*) AS traces,
          (SELECT COUNT(*) FROM {source_spans} s WHERE EXISTS (
            SELECT 1 FROM {source_metadata} m WHERE m.trace_id = s.trace_id AND {subject}
          )) AS spans,
          (SELECT COUNT(*) FROM {source_annotations} a WHERE EXISTS (
            SELECT 1 FROM {source_metadata} m WHERE m.trace_id = a.target_id AND {subject}
          )) AS annotations
        FROM {source_metadata} m WHERE {subject}""",
    )[0]
    target_counts = _read_rows(
        client,
        warehouse_id,
        f"""SELECT COUNT(*) AS traces,
          (SELECT COUNT(*) FROM {target_spans}) AS spans,
          (SELECT COUNT(*) FROM {target_annotations}) AS annotations
        FROM {target_metadata}""",
    )[0]
    expected = tuple(int(source_counts[key]) for key in ("traces", "spans", "annotations"))
    actual = tuple(int(target_counts[key]) for key in ("traces", "spans", "annotations"))
    if actual != expected:
        raise RuntimeError(
            f"UC trace copy verification failed: expected={expected} actual={actual}"
        )
    return actual


def main() -> int:
    args = _parse_args()
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = args.warehouse_id
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    legacy = mlflow.set_experiment(experiment_id=args.old_experiment)
    legacy_prefix = getattr(getattr(legacy, "trace_location", None), "table_prefix", None)
    if legacy_prefix and legacy_prefix != args.source_prefix:
        raise RuntimeError(
            f"legacy experiment uses UC prefix {legacy_prefix!r}, not {args.source_prefix!r}"
        )

    subject = _ensure_bootstrap_experiment(
        args.subject_name,
        actor="migration",
        profile=args.profile,
        catalog=args.trace_catalog,
        trace_schema=args.trace_schema,
        table_prefix=args.subject_prefix,
    )
    reviewer = _ensure_bootstrap_experiment(
        args.reviewer_name,
        actor="migration",
        profile=args.profile,
        catalog=args.trace_catalog,
        trace_schema=args.trace_schema,
        table_prefix=args.reviewer_prefix,
    )
    if subject.outcome != "created" or reviewer.outcome != "created":
        raise RuntimeError(
            f"experiment creation failed: subject={subject.error} reviewer={reviewer.error}"
        )

    workspace = _build_workspace_client(args.profile)
    copied, copied_spans, copied_annotations = _copy_uc_traces(
        workspace,
        args.warehouse_id,
        args.trace_catalog,
        args.trace_schema,
        args.source_prefix,
        args.subject_prefix,
        reset_target=args.reset_target,
    )
    if not args.skip_judges:
        _ensure_baseline_judges(
            subject.experiment_id,
            profile=args.profile,
            warehouse_id=args.warehouse_id,
        )

    agents = load_registered_agents(
        warehouse_id=args.warehouse_id, catalog=args.catalog, schema=args.schema
    )
    prior = next((agent for agent in agents if agent.agent_name == args.agent_name), None)
    goal_config = (
        prior.goal_config
        if prior and prior.goal_config
        else {
            "objective_metric": "total_tokens",
            "goal_direction": "minimize",
            "goal_target": -0.30,
            "goal_target_kind": "relative",
            "guardrail_judge": [],
        }
    )
    agent = (
        prior or Agent(agent_name=args.agent_name, experiment_id=subject.experiment_id)
    ).model_copy(
        update={
            "experiment_id": subject.experiment_id,
            "reviewer_experiment_id": reviewer.experiment_id,
            "annotations_table": subject.annotations_table,
            "target_workspace": args.target_workspace,
            "goal_config": goal_config,
        }
    )
    registry_fqn = f"`{args.catalog}`.`{args.schema}`.{REGISTRY_TABLE}"
    try:
        _execute(
            workspace,
            args.warehouse_id,
            f"ALTER TABLE {registry_fqn} ADD COLUMNS (reviewer_experiment_id STRING)",
        )
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower():
            raise
    publish_registry(
        AgentRegistry(agents=[agent]),
        client=workspace,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
    )
    print(
        f"subject_experiment_id={subject.experiment_id}\n"
        f"reviewer_experiment_id={reviewer.experiment_id}\n"
        f"annotations_table={subject.annotations_table}\n"
        f"copied_traces={copied}\ncopied_spans={copied_spans}\n"
        f"copied_annotations={copied_annotations}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
