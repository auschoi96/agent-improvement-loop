"""Serverless Job transport for the deployed app's onboarding engine."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.jobs.publish_job import resolve_job_auth
from ail.onboarding.service import run_action
from ail.publish import _build_workspace_client, _execute, _lit

ONBOARDING_RESULTS_TABLE = "agent_onboarding_results"
ONBOARDING_REQUESTS_TABLE = "agent_onboarding_requests"


def _ddl(catalog: str, schema: str) -> list[str]:
    """Authoritative governed storage for asynchronous onboarding transport."""
    return [
        f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}` "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS `{catalog}`.`{schema}`.{ONBOARDING_REQUESTS_TABLE} (
            request_id STRING,
            actor STRING,
            payload_json STRING,
            run_id BIGINT,
            created_at STRING,
            expires_at STRING,
            consumed_at STRING
        ) USING DELTA""",
        f"""CREATE TABLE IF NOT EXISTS `{catalog}`.`{schema}`.{ONBOARDING_RESULTS_TABLE} (
            request_id STRING,
            outcome STRING,
            result_json STRING,
            recorded_at STRING
        ) USING DELTA""",
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Lakeflow receives only an opaque request id. The user payload is read from a
    # governed UC table and never appears in job parameters or run metadata.
    parser.add_argument("--request-id", "--request_id", dest="request_id", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--trace-schema", default="mlflow_traces")
    parser.add_argument("--goal-llm-endpoint", required=True)
    # The arrival-triggered RLM job id, so a successful registration can add the new
    # agent's *_otel_spans table to that job's table_update trigger (see
    # ail.onboarding.service._reconcile_rlm_trigger_after_register). Empty => the
    # reconcile is skipped (the agent is still registered and covered by the cron jobs).
    parser.add_argument("--rlm-job-id", "--rlm_job_id", dest="rlm_job_id", default="")
    parser.add_argument(
        "--judge-backfill-job-id",
        "--judge_backfill_job_id",
        dest="judge_backfill_job_id",
        default="",
    )
    parser.add_argument("--token-secret-scope", default="")
    parser.add_argument("--token-secret-key", default="")
    return parser.parse_args(argv)


def _wait_for_statement(client: Any, response: Any) -> Any:
    from databricks.sdk.service.sql import StatementState

    state = response.status.state if response.status else None
    while state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1.0)
        response = client.statement_execution.get_statement(response.statement_id)
        state = response.status.state if response.status else None
    if state != StatementState.SUCCEEDED:
        detail = response.status.error.message if response.status and response.status.error else ""
        raise RuntimeError(f"onboarding request read failed: {state}: {detail}")
    return response


def _read_payload(
    *, request_id: str, warehouse_id: str, catalog: str, schema: str
) -> dict[str, object]:
    client = _build_workspace_client(None)
    fqn = f"`{catalog}`.`{schema}`.`{ONBOARDING_REQUESTS_TABLE}`"
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=(
            f"SELECT payload_json FROM {fqn} WHERE request_id = {_lit(request_id)} "
            "AND payload_json IS NOT NULL "
            "AND CAST(expires_at AS TIMESTAMP) > current_timestamp() LIMIT 1"
        ),
        wait_timeout="50s",
    )
    response = _wait_for_statement(client, response)
    data = response.result.data_array if response.result and response.result.data_array else []
    if not data or not data[0] or not data[0][0]:
        raise RuntimeError("onboarding request was not found, expired, or already consumed")
    payload = json.loads(data[0][0])
    if not isinstance(payload, dict):
        raise ValueError("onboarding payload must be a JSON object")
    return payload


def _redact_request(*, request_id: str, warehouse_id: str, catalog: str, schema: str) -> None:
    client = _build_workspace_client(None)
    fqn = f"`{catalog}`.`{schema}`.`{ONBOARDING_REQUESTS_TABLE}`"
    _execute(
        client,
        warehouse_id,
        f"UPDATE {fqn} SET payload_json = NULL, "
        "consumed_at = CAST(current_timestamp() AS STRING) "
        f"WHERE request_id = {_lit(request_id)}",
    )
    _execute(
        client,
        warehouse_id,
        f"DELETE FROM {fqn} WHERE CAST(created_at AS TIMESTAMP) "
        "< current_timestamp() - INTERVAL 7 DAYS",
    )


def _persist_result(
    *, request_id: str, result_json: str, outcome: str, warehouse_id: str, catalog: str, schema: str
) -> None:
    client = _build_workspace_client(None)
    fqn = f"`{catalog}`.`{schema}`.{ONBOARDING_RESULTS_TABLE}"
    _execute(client, warehouse_id, f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    _execute(
        client,
        warehouse_id,
        f"""CREATE TABLE IF NOT EXISTS {fqn} (
            request_id STRING,
            outcome STRING,
            result_json STRING,
            recorded_at STRING
        ) USING DELTA""",
    )
    _execute(client, warehouse_id, f"DELETE FROM {fqn} WHERE request_id = {_lit(request_id)}")
    _execute(
        client,
        warehouse_id,
        f"DELETE FROM {fqn} WHERE CAST(recorded_at AS TIMESTAMP) "
        "< current_timestamp() - INTERVAL 7 DAYS",
    )
    recorded_at = datetime.now(UTC).isoformat()
    _execute(
        client,
        warehouse_id,
        f"INSERT INTO {fqn} VALUES ({_lit(request_id)}, {_lit(outcome)}, "
        f"{_lit(result_json)}, {_lit(recorded_at)})",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    resolve_job_auth(
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    # MLflow's UC trace reader does not infer the warehouse from the Job's task
    # arguments. Surface the same configured warehouse explicitly before any
    # validate/bootstrap action inspects an existing experiment's trace history.
    os.environ[TRACING_WAREHOUSE_ENV] = args.warehouse_id
    os.environ["AIL_GOAL_LLM_ENDPOINT"] = args.goal_llm_endpoint
    # Surface the RLM job id for the registration's best-effort trigger reconcile.
    # Blank stays unset so the reconcile is a quiet no-op on a deploy that omitted it.
    if args.rlm_job_id.strip():
        os.environ["AIL_RLM_JOB_ID"] = args.rlm_job_id.strip()
    if args.judge_backfill_job_id.strip():
        os.environ["AIL_JUDGE_BACKFILL_JOB_ID"] = args.judge_backfill_job_id.strip()
    try:
        payload = _read_payload(
            request_id=args.request_id,
            warehouse_id=args.warehouse_id,
            catalog=args.catalog,
            schema=args.schema,
        )
        payload.update(
            {
                "warehouse_id": args.warehouse_id,
                "catalog": args.catalog,
                "schema": args.schema,
                # These are trusted Job configuration, not app-supplied payload values.
                "trace_catalog": args.catalog,
                "trace_schema": args.trace_schema,
            }
        )
        result = run_action(payload)
        result_json = result.model_dump_json()
        outcome = str(getattr(result, "outcome", "error"))
        _persist_result(
            request_id=args.request_id,
            result_json=result_json,
            outcome=outcome,
            warehouse_id=args.warehouse_id,
            catalog=args.catalog,
            schema=args.schema,
        )
    finally:
        _redact_request(
            request_id=args.request_id,
            warehouse_id=args.warehouse_id,
            catalog=args.catalog,
            schema=args.schema,
        )
    print(result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
