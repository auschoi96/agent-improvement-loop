"""Tier A — publish the L0 contract to Unity Catalog Delta tables.

This is the **source of truth** for the self-optimization leaderboard. It reuses
the repo's ingestion seam (:class:`ail.ingest.mlflow_source.MLflowTraceSource`)
and metric engine (:func:`ail.metrics.l0_deterministic.compute_l0`) — *no metric
logic is re-implemented here or in SQL* — and writes the resulting
:class:`~ail.metrics.contract.L0MetricsReport` to flat, analytics-friendly Delta
tables. The AppKit app (Tier B) only ``SELECT``s from these tables; Python stays
the authority for every computed number (pricing, redundancy, percentiles).

Tables written to ``<catalog>.<schema>`` (default
``austin_choi_omni_agent_catalog.agent_improvement_loop``):

* ``l0_session_metrics`` — one row per trace: model/producer/status, the token
  breakdown, estimated cost (with a ``cost_priced`` honesty flag), duration,
  tool count, and the strict redundancy rate.
* ``l0_corpus_summary`` — one row per experiment: corpus totals plus the
  bimodal token distribution (median/mean/p90/max/min) and cost rollup.
* ``l0_diagnosis`` — the tool-waste diagnostic: one row per repeated-call
  identity (``shell`` boilerplate re-runs, ``path`` repeated file edits/reads,
  ``args`` exact repeats), each with its repeat count.

The write is **idempotent and atomic per experiment**: each table is created if
missing, the new snapshot is staged in a transient clone, then swapped into the
live table with a single ``INSERT INTO … REPLACE WHERE experiment_id = <id>
SELECT * FROM <staging>`` statement. Because the live table is only ever touched
by that one atomic Delta transaction, a failure before or during staging leaves
the previous complete snapshot intact — a reader never sees empty or partial
data for an experiment that already had one. Re-runs replace; other experiments
are left untouched.

Auth mirrors :mod:`ail.metrics.report`: if ``DATABRICKS_HOST`` and
``DATABRICKS_TOKEN`` are set they are used for both the MLflow trace pull (the
reference experiment's v4 trace store rejects OAuth-profile creds for span
``batchGet``) and the warehouse writes; otherwise a Databricks CLI ``--profile``
is used. The reference experiment is also UC-table-backed, so the v4 trace store
reads it *through a SQL warehouse*: ``publish`` exports its ``warehouse_id`` as
``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` (the same warehouse that backs the writes) so
the read works without extra configuration.

Run::

    python -m ail.publish --experiment 660599403165942 \\
        --warehouse-id <SQL_WAREHOUSE_ID> --profile dais-demo
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from datetime import UTC, datetime
from typing import Any

from ail.ingest.mlflow_source import MLflowTraceSource
from ail.metrics.contract import L0MetricsReport
from ail.metrics.l0_deterministic import compute_l0

REFERENCE_EXPERIMENT = "660599403165942"
DEFAULT_CATALOG = "austin_choi_omni_agent_catalog"
DEFAULT_SCHEMA = "agent_improvement_loop"

SESSION_TABLE = "l0_session_metrics"
SUMMARY_TABLE = "l0_corpus_summary"
DIAGNOSIS_TABLE = "l0_diagnosis"

# INSERT ... VALUES batches. Diagnosis rows can number in the low thousands; a
# few hundred rows per statement keeps each statement comfortably small.
_INSERT_BATCH = 400


# ---------------------------------------------------------------------------
# SQL literal rendering (the only data crossing into SQL is this controlled,
# Python-computed report, but values are still escaped, never interpolated raw).
# ---------------------------------------------------------------------------


def _lit(value: Any) -> str:
    """Render a Python value as a safe Databricks SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Avoid scientific notation surprises; full precision round-trips fine.
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{text}'"


def _row(values: list[Any]) -> str:
    return "(" + ", ".join(_lit(v) for v in values) + ")"


# ---------------------------------------------------------------------------
# Report -> flat rows
# ---------------------------------------------------------------------------

# Column orders are declared once and reused by both the DDL and the INSERTs so
# the two can never drift.
SESSION_COLUMNS: list[str] = [
    "experiment_id",
    "trace_id",
    "session_id",
    "producer",
    "model",
    "status",
    "request_time",
    "duration_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cache_total_tokens",
    "est_cost_usd",
    "cost_priced",
    "total_tool_calls",
    "distinct_tool_calls",
    "redundant_tool_calls",
    "redundancy_rate",
    "generated_at",
]

SUMMARY_COLUMNS: list[str] = [
    "experiment_id",
    "schema_version",
    "generated_at",
    "trace_count",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "cache_total_tokens",
    "median_tokens",
    "mean_tokens",
    "p90_tokens",
    "max_tokens",
    "min_tokens",
    "total_tool_calls",
    "redundancy_rate",
    "total_cost_usd",
    "priced_traces",
    "unpriced_traces",
]

DIAGNOSIS_COLUMNS: list[str] = [
    "experiment_id",
    "trace_id",
    "session_id",
    "model",
    "signature_kind",
    "tool",
    "identity",
    "repeat_count",
    "trace_total_tool_calls",
    "generated_at",
]


def _session_rows(report: L0MetricsReport) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for m in report.traces:
        rows.append(
            [
                report.experiment_id,
                m.trace_id,
                m.session_id,
                m.producer,
                m.model,
                m.status,
                m.request_time,
                m.duration_seconds,
                m.tokens.input_tokens,
                m.tokens.output_tokens,
                m.tokens.total_tokens,
                m.tokens.cache_creation_input_tokens,
                m.tokens.cache_read_input_tokens,
                m.tokens.cache_total_tokens,
                m.cost.total_usd,
                m.cost.priced,
                m.total_tool_calls,
                m.redundancy.distinct_tool_calls,
                m.redundancy.redundant_tool_calls,
                m.redundancy.redundancy_rate,
                report.generated_at,
            ]
        )
    return rows


def _summary_row(report: L0MetricsReport) -> list[Any]:
    agg = report.aggregate
    return [
        report.experiment_id,
        report.schema_version,
        report.generated_at,
        agg.n_traces,
        agg.tokens.input_tokens,
        agg.tokens.output_tokens,
        agg.tokens.total_tokens,
        agg.tokens.cache_total_tokens,
        agg.token_stats.median,
        agg.token_stats.mean,
        agg.token_stats.p90,
        agg.token_stats.max,
        agg.token_stats.min,
        agg.total_tool_calls,
        agg.redundancy.redundancy_rate,
        agg.cost.total_usd,
        agg.cost.priced_traces,
        agg.cost.unpriced_traces,
    ]


def _diagnosis_rows(report: L0MetricsReport) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for m in report.traces:
        for rc in m.redundancy.repeated_calls:
            rows.append(
                [
                    report.experiment_id,
                    m.trace_id,
                    m.session_id,
                    m.model,
                    rc.signature_kind,
                    rc.tool,
                    rc.identity,
                    rc.count,
                    m.total_tool_calls,
                    report.generated_at,
                ]
            )
    return rows


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def _ddl(catalog: str, schema: str) -> list[str]:
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{SESSION_TABLE} (
            experiment_id STRING,
            trace_id STRING,
            session_id STRING,
            producer STRING,
            model STRING,
            status STRING,
            request_time STRING,
            duration_seconds DOUBLE,
            input_tokens BIGINT,
            output_tokens BIGINT,
            total_tokens BIGINT,
            cache_creation_input_tokens BIGINT,
            cache_read_input_tokens BIGINT,
            cache_total_tokens BIGINT,
            est_cost_usd DOUBLE,
            cost_priced BOOLEAN,
            total_tool_calls INT,
            distinct_tool_calls INT,
            redundant_tool_calls INT,
            redundancy_rate DOUBLE,
            generated_at STRING
        ) USING DELTA
        COMMENT 'One row per trace: L0 token/cost/tool metrics. est_cost_usd is an ESTIMATE.'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{SUMMARY_TABLE} (
            experiment_id STRING,
            schema_version STRING,
            generated_at STRING,
            trace_count INT,
            total_input_tokens BIGINT,
            total_output_tokens BIGINT,
            total_tokens BIGINT,
            cache_total_tokens BIGINT,
            median_tokens DOUBLE,
            mean_tokens DOUBLE,
            p90_tokens DOUBLE,
            max_tokens BIGINT,
            min_tokens BIGINT,
            total_tool_calls BIGINT,
            redundancy_rate DOUBLE,
            total_cost_usd DOUBLE,
            priced_traces INT,
            unpriced_traces INT
        ) USING DELTA
        COMMENT 'One row per experiment: corpus totals + token distribution + cost (ESTIMATE).'""",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{DIAGNOSIS_TABLE} (
            experiment_id STRING,
            trace_id STRING,
            session_id STRING,
            model STRING,
            signature_kind STRING,
            tool STRING,
            identity STRING,
            repeat_count INT,
            trace_total_tool_calls INT,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Tool-waste diagnosis: one row per repeated-call identity (shell/path/args).'""",
    ]


# ---------------------------------------------------------------------------
# Statement execution
# ---------------------------------------------------------------------------


def _build_workspace_client(profile: str | None) -> Any:
    """Build a WorkspaceClient, preferring explicit env-token auth.

    Consistent with :mod:`ail.metrics.report`: a PAT in ``DATABRICKS_HOST`` /
    ``DATABRICKS_TOKEN`` is used when present (the path that works against the
    reference workspace's v4 trace store); otherwise the CLI ``profile``.
    """
    from databricks.sdk import WorkspaceClient

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if host and token:
        os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)
        return WorkspaceClient(host=host, token=token)
    if profile:
        return WorkspaceClient(profile=profile)
    return WorkspaceClient()


def _execute(client: Any, warehouse_id: str, statement: str) -> None:
    """Run a single SQL statement on the warehouse and wait for success."""
    from databricks.sdk.service.sql import StatementState

    resp = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
    )
    statement_id = resp.statement_id
    state = resp.status.state if resp.status else None
    while state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1.0)
        resp = client.statement_execution.get_statement(statement_id)
        state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        detail = ""
        if resp.status and resp.status.error:
            detail = f": {resp.status.error.message}"
        raise RuntimeError(f"statement {state}{detail}\nSQL head: {statement[:300]}")


def _insert_batched(
    client: Any,
    warehouse_id: str,
    table_fqn: str,
    columns: list[str],
    rows: list[list[Any]],
) -> int:
    """INSERT rows in batches; returns the number of rows written."""
    if not rows:
        return 0
    col_list = ", ".join(columns)
    written = 0
    for start in range(0, len(rows), _INSERT_BATCH):
        chunk = rows[start : start + _INSERT_BATCH]
        values = ",\n".join(_row(r) for r in chunk)
        _execute(
            client,
            warehouse_id,
            f"INSERT INTO {table_fqn} ({col_list}) VALUES\n{values}",
        )
        written += len(chunk)
    return written


def _atomic_replace_table(
    client: Any,
    warehouse_id: str,
    schema_fqn: str,
    table: str,
    columns: list[str],
    rows: list[list[Any]],
    experiment_id: str,
) -> int:
    """Atomically replace one experiment's rows in ``table`` with ``rows``.

    The new snapshot is loaded into a transient staging clone (batched — those
    writes are not visible to readers of the live table), then swapped in with a
    single ``INSERT INTO … REPLACE WHERE`` Delta transaction. The live table is
    therefore mutated exactly once, atomically: if staging fails the live table
    keeps its prior complete snapshot. Empty ``rows`` is handled correctly — the
    swap removes any prior rows for the experiment and writes nothing.

    Returns the number of rows written.
    """
    main = f"{schema_fqn}.{table}"
    staging = f"{schema_fqn}._stg_{table}"

    # Fresh empty clone — same schema and column order as the live table, so the
    # SELECT * swap below is position-aligned. CREATE OR REPLACE is itself atomic
    # and clears any staging table left behind by an earlier interrupted run.
    _execute(
        client, warehouse_id, f"CREATE OR REPLACE TABLE {staging} AS SELECT * FROM {main} WHERE 1=0"
    )
    try:
        _insert_batched(client, warehouse_id, staging, columns, rows)
        _execute(
            client,
            warehouse_id,
            f"INSERT INTO {main} REPLACE WHERE experiment_id = {_lit(experiment_id)} "
            f"SELECT * FROM {staging}",
        )
    finally:
        # Best-effort cleanup; never mask a failure from the steps above.
        with contextlib.suppress(Exception):
            _execute(client, warehouse_id, f"DROP TABLE IF EXISTS {staging}")
    return len(rows)


# ---------------------------------------------------------------------------
# Source + orchestration
# ---------------------------------------------------------------------------


def _build_source(profile: str | None) -> MLflowTraceSource:
    """MLflow trace source, preferring explicit env-token auth (see module docstring)."""
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)
        return MLflowTraceSource(tracking_uri="databricks")
    if profile:
        return MLflowTraceSource(tracking_uri=f"databricks://{profile}", profile=profile)
    return MLflowTraceSource()


def publish(
    *,
    experiment_id: str,
    warehouse_id: str,
    profile: str | None = None,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    max_results: int | None = None,
    generated_at: str | None = None,
) -> L0MetricsReport:
    """Pull traces, compute L0, and (idempotently) publish to UC Delta tables.

    Returns the computed :class:`~ail.metrics.contract.L0MetricsReport`.
    """
    # The reference experiment is backed by a UC table; MLflow's v4 trace store
    # reads those traces through a SQL warehouse and requires its id in the
    # environment. We already have the warehouse (it backs the Delta writes), so
    # surface it for the read too. ``setdefault`` respects an explicit override.
    os.environ.setdefault("MLFLOW_TRACING_SQL_WAREHOUSE_ID", warehouse_id)

    source = _build_source(profile)
    traces = source.fetch_traces(experiment_id=experiment_id, max_results=max_results)
    stamp = generated_at or datetime.now(UTC).isoformat()
    report = compute_l0(traces, experiment_id=experiment_id, generated_at=stamp)

    client = _build_workspace_client(profile)
    fqn = f"`{catalog}`.`{schema}`"

    # 1. Schema + tables (idempotent).
    for ddl in _ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)

    # 2. Atomically replace each table's slice for this experiment. Each call
    #    mutates its live table exactly once (staging clone -> REPLACE WHERE
    #    swap), so a mid-run failure never leaves empty/partial data.
    n_session = _atomic_replace_table(
        client,
        warehouse_id,
        fqn,
        SESSION_TABLE,
        SESSION_COLUMNS,
        _session_rows(report),
        experiment_id,
    )
    n_summary = _atomic_replace_table(
        client,
        warehouse_id,
        fqn,
        SUMMARY_TABLE,
        SUMMARY_COLUMNS,
        [_summary_row(report)],
        experiment_id,
    )
    n_diag = _atomic_replace_table(
        client,
        warehouse_id,
        fqn,
        DIAGNOSIS_TABLE,
        DIAGNOSIS_COLUMNS,
        _diagnosis_rows(report),
        experiment_id,
    )

    print(
        f"published experiment={experiment_id} to {catalog}.{schema}: "
        f"{SESSION_TABLE}={n_session} rows, {SUMMARY_TABLE}={n_summary} row, "
        f"{DIAGNOSIS_TABLE}={n_diag} rows"
    )
    print(
        f"corpus: n_traces={report.n_traces} "
        f"total_tokens={report.aggregate.tokens.total_tokens:,} "
        f"median={report.aggregate.token_stats.median:,.0f} "
        f"max={report.aggregate.token_stats.max:,} "
        f"total_cost=${report.aggregate.cost.total_usd:,.2f} "
        f"(priced={report.aggregate.cost.priced_traces}, "
        f"unpriced={report.aggregate.cost.unpriced_traces})"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the L0 contract to Unity Catalog Delta tables (Tier A)."
    )
    parser.add_argument("--experiment", default=REFERENCE_EXPERIMENT)
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("AIL_WAREHOUSE_ID"),
        help="SQL warehouse id used to create and populate the Delta tables.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AIL_DATABRICKS_PROFILE", "dais-demo"),
        help="Databricks CLI profile (ignored if DATABRICKS_HOST/DATABRICKS_TOKEN are set).",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--max-results", default=None, type=int)
    args = parser.parse_args(argv)

    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")

    publish(
        experiment_id=args.experiment,
        warehouse_id=args.warehouse_id,
        profile=args.profile,
        catalog=args.catalog,
        schema=args.schema,
        max_results=args.max_results,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
