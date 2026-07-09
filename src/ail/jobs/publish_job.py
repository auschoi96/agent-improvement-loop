"""Scheduled-Job entrypoint for the Tier A L0 publish step.

This is the wheel-task entrypoint a Databricks Job invokes on a schedule to keep
the L0 leaderboard *living*: it refreshes the UC Delta tables from the latest
MLflow traces. It does **not** recompute any metric — it resolves auth for the
job's run-as identity and then delegates to :func:`ail.publish.publish`, which is
the single source of truth for the metric logic (via :mod:`ail.metrics`) and for
the atomic, idempotent write.

Registry-driven multi-agent: with no ``--experiment`` it runs REGISTRY MODE — it
reads every agent from the UC ``agent_registry`` (via :mod:`ail.jobs.multi_agent`)
and publishes each agent's own experiment, one per-experiment L0 slice, with
per-agent isolation (one agent's publish failure is logged and the loop continues).
Passing an explicit ``--experiment`` is the single-agent override for local/manual
runs (publish only that experiment, exactly as before).

Why a wrapper exists at all
---------------------------
The L0 metrics are read from MLflow's v4 trace REST store. On the reference
workspace that store **rejects profile-only OAuth** for span ``batchGet`` (a 401)
— the same reason ``ail.publish`` / ``ail.metrics.report`` prefer explicit
``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``. A serverless Job cannot have those env
vars injected from the bundle (the serverless environment spec has no env-var
field), so the *bearer token must be resolved at runtime*. That is the one job-
specific concern this module owns.

Auth resolution order (:func:`resolve_job_auth`)
------------------------------------------------
1. **Pre-set env** — if ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` are already in
   the environment (local runs, CI, or a hand-set Job), use them unchanged.
2. **Secret scope** (``--token-secret-scope`` + ``--token-secret-key``) — read a
   bearer token a Databricks **secret scope**. This is the production-hardened
   path: store the run-as **service principal's** token in a secret scope and
   point the Job at it; nothing sensitive is ever committed or passed as a Job
   parameter. The run-as SP needs ``READ`` on the scope.
3. **Mint from ambient identity** (default) — call ``Config.authenticate()`` to
   mint a short-lived **OAuth bearer** for the Job's run-as identity and pass it
   *explicitly* as ``DATABRICKS_TOKEN``. This sidesteps the v4 store's per-request
   OAuth-resolution bug without storing any long-lived credential, and works
   identically whether the Job runs as a user (the demo) or as a service
   principal (production) — the SP just needs the data grants.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, MutableMapping
from typing import Any

from ail.jobs.multi_agent import (
    load_registered_agents,
    missing_registry_target,
    run_for_each_registered_agent,
)
from ail.publish import DEFAULT_CATALOG, DEFAULT_SCHEMA, publish
from ail.registry import Agent


def _default_workspace_client() -> Any:
    """Build a ``WorkspaceClient`` from the Job's ambient (run-as) credentials."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def _bearer_from_config(config: Any) -> str:
    """Mint a short-lived OAuth bearer token from an SDK ``Config``.

    ``Config.authenticate()`` returns request headers for whatever auth the SDK
    resolved (a user's U2M token locally, the run-as SP's M2M token in a Job).
    We extract the bearer so it can be passed *explicitly* — the v4 trace store
    accepts a bearer header even though it rejects profile-managed OAuth.
    """
    headers = config.authenticate() or {}
    scheme, _, token = headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise RuntimeError(
            "ambient Databricks auth did not yield a bearer token "
            f"(authorization scheme was {scheme!r}); set DATABRICKS_HOST/"
            "DATABRICKS_TOKEN explicitly or provide a token secret scope"
        )
    return token


def resolve_job_auth(
    *,
    token_secret_scope: str | None = None,
    token_secret_key: str | None = None,
    workspace_client_factory: Callable[[], Any] = _default_workspace_client,
    env: MutableMapping[str, str] | None = None,
) -> str:
    """Ensure ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` are a v4-acceptable bearer.

    Mutates ``env`` (defaults to ``os.environ``) so the downstream
    :func:`ail.publish.publish` call picks the explicit-token path. Returns a
    short label of the path taken (``"env"`` / ``"secret-scope"`` / ``"minted"``)
    for logging.
    """
    environ = os.environ if env is None else env

    if environ.get("DATABRICKS_HOST") and environ.get("DATABRICKS_TOKEN"):
        return "env"

    client = workspace_client_factory()
    host = client.config.host
    if not host:
        raise RuntimeError("could not resolve a workspace host from ambient Databricks auth")

    if token_secret_scope and token_secret_key:
        token = client.dbutils.secrets.get(token_secret_scope, token_secret_key)
        path = "secret-scope"
    else:
        token = _bearer_from_config(client.config)
        path = "minted"

    # Drop any ambient profile so MLflow's per-request credential resolution
    # cannot fall back to OAuth (which the v4 store rejects) for some spans while
    # using the explicit bearer for others.
    environ.pop("DATABRICKS_CONFIG_PROFILE", None)
    environ["DATABRICKS_HOST"] = host
    environ["DATABRICKS_TOKEN"] = token
    return path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scheduled Tier A publish: refresh the L0 UC Delta tables from MLflow traces. "
        "With no --experiment it runs REGISTRY MODE over every agent in agent_registry; "
        "pass --experiment to publish JUST that one experiment (single-agent override)."
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="Explicit experiment id => single-agent override (publish only that one). "
        "Empty (the default) => registry mode: publish every agent in agent_registry.",
    )
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID"),
        help="SQL warehouse id used to create and populate the Delta tables.",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--max-results", default=None, type=int)
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
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")
    return args


def _publish_one(args: argparse.Namespace, *, experiment_id: str) -> int:
    """Publish one experiment's L0 slice — the reused single-agent body.

    Per-experiment idempotency is preserved: :func:`ail.publish.publish` REPLACEs
    only ``experiment_id = <this experiment>``, so publishing agent A never disturbs
    agent B's rows in the shared L0 tables.
    """
    publish(
        experiment_id=experiment_id,
        warehouse_id=args.warehouse_id,
        catalog=args.catalog,
        schema=args.schema,
        max_results=args.max_results,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth_path = resolve_job_auth(
        # Empty strings (the bundle default when no scope is configured) mean
        # "not provided" -> fall through to minting.
        token_secret_scope=args.token_secret_scope or None,
        token_secret_key=args.token_secret_key or None,
    )

    if args.experiment:
        # Single-agent override: publish JUST this experiment, exactly as before.
        print(
            f"[ail.jobs.publish_job] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')} "
            f"single-agent experiment={args.experiment} -> {args.catalog}.{args.schema}"
        )
        return _publish_one(args, experiment_id=args.experiment)

    # Registry mode: publish every agent in agent_registry, one L0 slice each.
    missing = missing_registry_target(args.warehouse_id, args.catalog, args.schema)
    if missing:
        print(
            f"[ail.jobs.publish_job] registry mode requires {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    print(
        f"[ail.jobs.publish_job] auth={auth_path} host={os.environ.get('DATABRICKS_HOST')} "
        f"registry mode -> {args.catalog}.{args.schema}"
    )
    agents = load_registered_agents(
        warehouse_id=args.warehouse_id, catalog=args.catalog, schema=args.schema
    )

    def per_agent(agent: Agent) -> int:
        return _publish_one(args, experiment_id=agent.experiment_id)

    result = run_for_each_registered_agent(agents, per_agent, job_name="ail.jobs.publish_job")
    return result.worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
