"""Job-task entrypoint for the app's Approve/Reject write-path (deployed transport).

Lane 3b's apply engine (:mod:`ail.loop.apply_service`) is **Python**; the app is
**Node**. In local dev / a self-hosted image the Node route bridges to Python by
running ``python -m ail.loop.apply_service`` as a subprocess
(:func:`ail.loop.apply_service.main`, stdin/stdout). The **deployed Databricks App
image is Node-only** — the ``ail`` wheel is not importable there; it ships as
serverless Jobs (``docs/DEPLOY.md``). So on the deployed image the Approve button
cannot spawn Python. This module is the deployed transport's server side: a
``python_wheel_task`` entrypoint the app **triggers as a Job** (``resources/
apply_service.job.yml``), passing the reviewer's decision as **job named
parameters** instead of stdin. It is a thin param-adapter — it calls the *same*
:func:`ail.loop.apply_service.run_decision`; no business logic lives here.

Getting the result back to the caller (the load-bearing part)
------------------------------------------------------------
A serverless ``python_wheel_task`` does **not** stream stdout back to whoever
triggered it, so the CLI's "print the result on stdout" contract does not survive
this transport. Instead this job writes the engine's real
:class:`~ail.loop.apply_service.ApplyServiceResult` (full JSON) to a small UC Delta
**result table** (:data:`APPLY_RESULTS_TABLE`) keyed by ``(proposal_id,
decided_at)`` under the framework ``catalog.schema``. The Node bridge
(:func:`jobTriggerApplyBridge` in ``server/plugins/approvals/bridge.ts``) reads
that row back **after** the run reaches a terminal SUCCESS state and returns it
verbatim. This is consistent with the framework's two-tier / auditable pattern and
lets the app render the real outcome. ``decided_at`` is server-set per decision, so
the key is unique per decision (a retried trigger of the *same* decision overwrites
its own row; a *different* decision writes a distinct row).

Fail-closed
-----------
The engine's own guards are unchanged: an empty approver, an unknown decision, a
missing/non-pending proposal (already applied/rejected/superseded), or any infra
failure yield a ``REFUSED``/``ERROR`` :class:`ApplyServiceResult` — never a fake
apply. Those are **legitimate decision-level outcomes**: the job records them to the
result table and exits ``0`` so the bridge surfaces them honestly. Only a genuine
adapter failure (auth resolution, or the result-row write itself failing) raises,
so the run ends non-terminal/FAILED and the bridge fails closed (it never fabricates
a success from a run it cannot read back).

Auth mirrors :mod:`ail.jobs.publish_job`: the run-as identity mints a short-lived
bearer when no token-secret-scope is configured (see
:func:`ail.jobs.publish_job.resolve_job_auth`).
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from typing import Any

from ail.jobs.publish_job import resolve_job_auth
from ail.loop.apply_service import ApplyServiceResult, run_decision
from ail.optimize.prompt_registry import (
    DEFAULT_CATALOG,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SCHEMA,
)
from ail.publish import _build_workspace_client, _execute, _lit

__all__ = [
    "APPLY_RESULTS_TABLE",
    "APPLY_RESULT_COLUMNS",
    "write_apply_result",
    "main",
]

#: Out-of-band handoff table: the engine's real ``ApplyServiceResult`` (full JSON),
#: written by this job and read back by the Node bridge after a terminal SUCCESS.
#: Keyed by ``(proposal_id, decided_at)`` — ``decided_at`` is server-set per
#: decision, so the key identifies exactly this decision's run.
APPLY_RESULTS_TABLE = "agent_apply_results"

#: Column order — declared once, reused by the DDL and the INSERT (the
#: :mod:`ail.publish` convention) so the two can never drift. Must match the
#: read-back SELECT in ``server/plugins/approvals/bridge.ts``.
APPLY_RESULT_COLUMNS: list[str] = [
    "proposal_id",
    "decided_at",
    "agent_name",
    "decision",
    "outcome",
    "result_json",
    "recorded_at",
]


def _apply_results_ddl(catalog: str, schema: str) -> list[str]:
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{APPLY_RESULTS_TABLE} (
            proposal_id STRING,
            decided_at STRING,
            agent_name STRING,
            decision STRING,
            outcome STRING,
            result_json STRING,
            recorded_at STRING
        ) USING DELTA
        COMMENT 'Out-of-band ApplyServiceResult JSON the app bridge reads back (lane 3b).'""",
    ]


def write_apply_result(
    result: ApplyServiceResult,
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    recorded_at: str | None = None,
) -> None:
    """Append the engine's real result to ``agent_apply_results`` (create if needed).

    Stores the *verbatim* :meth:`~pydantic.BaseModel.model_dump_json` so the Node
    bridge round-trips the exact :class:`ApplyServiceResult` the engine produced —
    no re-derivation on either side.
    """
    for ddl in _apply_results_ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)
    stamp = recorded_at or datetime.now(UTC).isoformat()
    values = [
        result.proposal_id,
        result.decided_at,
        result.agent_name,
        result.decision.value,
        result.outcome.value,
        result.model_dump_json(),
        stamp,
    ]
    fqn = f"`{catalog}`.`{schema}`.{APPLY_RESULTS_TABLE}"
    cols = ", ".join(APPLY_RESULT_COLUMNS)
    literals = ", ".join(_lit(v) for v in values)
    _execute(client, warehouse_id, f"INSERT INTO {fqn} ({cols}) VALUES ({literals})")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Job transport for the app's authenticated Approve/Reject write-path: "
        "runs ail.loop.apply_service.run_decision from job named parameters and writes the "
        "result to the agent_apply_results table for the app bridge to read back."
    )
    # The decision — supplied by the trigger (job parameters), never hardcoded in
    # the bundle. `approver`/`decided-at` are set SERVER-SIDE by the authenticated
    # route; the job trusts them as-is (the engine still refuses an empty approver).
    parser.add_argument("--proposal-id", default="")
    parser.add_argument("--agent-name", default="")
    parser.add_argument("--decision", default="")
    parser.add_argument("--approver", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--decided-at", default="")
    # Framework wiring — from bundle vars (mirrors ail-publish-job).
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID") or os.environ.get("DATABRICKS_WAREHOUSE_ID"),
        help="SQL warehouse id used to load the proposal, apply, and record the result.",
    )
    parser.add_argument("--catalog", default=os.environ.get("AIL_CATALOG"))
    parser.add_argument("--schema", default=os.environ.get("AIL_SCHEMA"))
    parser.add_argument("--prompt-name", default=DEFAULT_PROMPT_NAME)
    parser.add_argument(
        "--token-secret-scope",
        default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", ""),
        help="Secret scope holding the run-as bearer token (production path). "
        "Empty => mint a short-lived token from the run-as identity.",
    )
    parser.add_argument(
        "--token-secret-key",
        default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""),
        help="Secret key within --token-secret-scope.",
    )
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error(
            "--warehouse-id is required (or set AIL_WAREHOUSE_ID / DATABRICKS_WAREHOUSE_ID)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    """Job entry: resolve auth, decide via the engine, persist the real result.

    Returns ``0`` for any *decision-level* outcome (including a fail-closed
    ``REFUSED``/``ERROR`` — those are honest results the bridge surfaces). Raises
    (non-zero / FAILED run) only on an adapter-level failure the bridge must not
    mistake for a decision: auth resolution failing, or the result-row write itself
    failing (so the bridge, unable to read the row, fails closed rather than
    fabricating a success).
    """
    args = _parse_args(argv)
    auth_path = resolve_job_auth(
        # Empty strings (the bundle default when no scope is configured) mean
        # "not provided" -> fall through to minting.
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )
    decided_at = args.decided_at or datetime.now(UTC).isoformat()
    print(
        f"[ail.jobs.apply_job] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')} "
        f"proposal={args.proposal_id!r} agent={args.agent_name!r} decision={args.decision!r} "
        f"-> {args.catalog}.{args.schema}"
    )

    # The engine re-loads the authoritative pending proposal, re-checks proof + gate,
    # and applies (or refuses). A non-pending proposal (already decided / superseded)
    # loads as None -> REFUSED, so a duplicated/retried trigger never re-applies.
    result = run_decision(
        proposal_id=args.proposal_id,
        agent_name=args.agent_name,
        decision=args.decision,
        approver=args.approver,
        reason=args.reason or None,
        decided_at=decided_at,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
        prompt_name=args.prompt_name,
    )

    # Persist the REAL result out-of-band so the Node bridge can read it back (a
    # serverless wheel task does not stream stdout to the caller). A write failure
    # raises -> the run FAILS -> the bridge fails closed (never a fabricated apply).
    client = _build_workspace_client(None)
    write_apply_result(
        result,
        client=client,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
    )

    # Also print for run-log visibility (not the bridge's retrieval path).
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
