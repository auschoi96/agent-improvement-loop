"""Serverless Job transport for the deployed app's onboarding engine."""

from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import UTC, datetime

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.jobs.publish_job import resolve_job_auth
from ail.onboarding.service import run_action
from ail.publish import _build_workspace_client, _execute, _lit

ONBOARDING_RESULTS_TABLE = "agent_onboarding_results"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Lakeflow forwards top-level job parameters to Python wheel tasks using the
    # parameter names verbatim (``--request_id`` / ``--payload_base64``). Keep the
    # hyphenated forms used by the task's explicit named_parameters as aliases so
    # either transport shape resolves to one canonical argparse destination.
    parser.add_argument("--request-id", "--request_id", dest="request_id", required=True)
    parser.add_argument(
        "--payload-base64", "--payload_base64", dest="payload_base64", required=True
    )
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
    parser.add_argument("--token-secret-scope", default="")
    parser.add_argument("--token-secret-key", default="")
    return parser.parse_args(argv)


def _decode_payload(encoded: str) -> dict[str, object]:
    raw = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("onboarding payload must be a JSON object")
    return payload


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
    payload = _decode_payload(args.payload_base64)
    payload.update(
        {
            "warehouse_id": args.warehouse_id,
            "catalog": args.catalog,
            "schema": args.schema,
            # These are trusted Job configuration, not app-supplied payload values.
            # In particular, app.yaml does not expand bundle `${var.*}` tokens, so
            # allowing those values through would send an invalid UC trace location
            # to MLflow when the wizard creates the isolated reviewer experiment.
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
    print(result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
