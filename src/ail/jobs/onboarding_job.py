"""Serverless Job transport for the deployed app's onboarding engine."""

from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import UTC, datetime

from ail.jobs.publish_job import resolve_job_auth
from ail.onboarding.service import run_action
from ail.publish import _build_workspace_client, _execute, _lit

ONBOARDING_RESULTS_TABLE = "agent_onboarding_results"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--payload-base64", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--goal-llm-endpoint", required=True)
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
    os.environ["AIL_GOAL_LLM_ENDPOINT"] = args.goal_llm_endpoint
    payload = _decode_payload(args.payload_base64)
    payload.update(
        {
            "warehouse_id": args.warehouse_id,
            "catalog": args.catalog,
            "schema": args.schema,
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
