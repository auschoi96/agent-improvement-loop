"""Tier A — publish pending proposed actions to a unified Unity Catalog table.

This is the write-side seam between the controller (lane 2) and the in-app
approval queue (lane 3, ``docs/LOOP_CONTROLLER.md``). The controller emits pending
:class:`~ail.loop.proposals.ProposedAction`\\ s; this module writes them to one
unified Delta table, ``agent_proposed_actions``, that lane 3 reads **SELECT-only**
to populate the Proposals view (the app stays read-only; lane 3 owns the
authenticated approve/reject write-path).

Writes are queue-preserving: each proposal is inserted only when its
``(agent_name, experiment_id, proposal_id)`` key is absent. A planner firing never
deletes unrelated pending recommendations and never resets decided/history rows.
* **Inert by construction.** A row carries the change body (SQL/diff/ref/target)
  and its proof + gate status — it does **not** apply anything. The apply happens
  only on human approval in lane 3.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from ail.loop.proposals import ProposedAction
from ail.publish import (
    DEFAULT_CATALOG,
    DEFAULT_SCHEMA,
    _build_workspace_client,
    _execute,
    _lit,
)

__all__ = [
    "SCHEMA_VERSION",
    "PROPOSALS_TABLE",
    "PROPOSAL_COLUMNS",
    "publish_agent_proposals",
    "publish_proposals",
    "insert_proposal_if_absent",
]

SCHEMA_VERSION = "ail.loop.proposals/v1.1"

#: The single unified table lane 3's approval queue reads (SELECT-only).
PROPOSALS_TABLE = "agent_proposed_actions"

#: Column order — declared once, reused by the DDL and the INSERTs so the two can
#: never drift (the :mod:`ail.publish` convention).
PROPOSAL_COLUMNS: list[str] = [
    "agent_name",
    "experiment_id",
    "proposal_id",
    "schema_version",
    "status",
    "action_kind",
    "risk_class",
    "objective_metric",
    "goal_cohort",
    # why
    "trigger_kind",
    "trigger_summary",
    "trigger_metric",
    "trigger_observed_value",
    "trigger_threshold",
    "trigger_n_traces",
    "trigger_judge_name",
    "trigger_asset_type",
    "trigger_source_rank",
    "trigger_trace_refs",
    # what
    "change_kind",
    "change_summary",
    "change_sql",
    "change_diff",
    "change_evolved_body_ref",
    "change_revert_target",
    # what — AGENT_TASK payload (additive, nullable): the NL plan, plus the executor-
    # filled (L7b-2) concrete-change preview + produced change-set ref. All NULL for a
    # non-AGENT_TASK proposal, and for an AGENT_TASK the preview/ref are NULL until L7b-2.
    "change_plan",
    "change_preview_diff",
    "change_produced_change_ref",
    # reviewed local-apply contract + companion lifecycle. The hosted app only
    # advances approval; these fields are executed/updated by the local companion.
    "change_local_apply_spec_json",
    "local_apply_status",
    "local_apply_error",
    "local_apply_completed_at",
    "local_apply_pre_change_ref",
    "local_apply_validation_output",
    # proof
    "proof_objective_metric",
    "proof_proved_improvement",
    "proof_correctness_held",
    "proof_realized_savings_absolute",
    "proof_realized_savings_pct",
    "proof_n_promote",
    "proof_n_block",
    "proof_n_errored",
    "proof_suite_content_hash",
    "proof_suite_version",
    # gate
    "gate_readiness_tier",
    "gate_can_prove_improvement",
    "gate_judge_agreement",
    "gate_scored_coverage",
    "gate_n_distrusted_judges",
    "gate_gated",
    "gate_reasons",
    # verify — the opt-in Tier-2 "verify on my suite" request lifecycle (L9). A freshly
    # published proposal is never verify-requested (verify_requested=False, the rest
    # NULL); the app's authenticated verify write-path sets verify_requested/status/by/at,
    # and the companion poll handler drives the frozen-suite proof and writes the RESULT
    # into the existing proof_* columns above (no parallel proof schema) plus an honest
    # terminal verify_status. This is request lifecycle only — the proof itself reuses
    # proof_*; verify_* never carries a fabricated result.
    "verify_requested",
    "verify_status",
    "verify_requested_by",
    "verify_requested_at",
    "verify_completed_at",
    "verify_error",
    # provenance
    "created_at",
    "generated_at",
]


def _proposal_row(p: ProposedAction, *, generated_at: str | None) -> list[Any]:
    """Flatten one :class:`ProposedAction` into a row aligned with ``PROPOSAL_COLUMNS``.

    List-valued fields (trace refs, gate reasons) are stored as JSON arrays so a
    reader gets them back structured; every scalar is written verbatim.

    ``proof`` is optional: an **evidence-first** proposal
    (:func:`ail.loop.evidence_cycle.run_evidence_cycle`) carries ``proof=None`` (it
    rests on its evidence + gate, proving is opt-in Tier-2 — see
    ``docs/PRODUCT_ARCHITECTURE.md`` §3). All ten ``proof_*`` columns are then written
    ``NULL`` (the columns already exist and are nullable, so the DDL and the app's
    SELECT-only read are unchanged — a NULL proof column just reads back as "no
    frozen-suite proof"). A *prove-before-propose* proposal
    (:func:`ail.loop.controller.run_cycle`) still writes its full proof unchanged.
    """
    t = p.trigger
    c = p.change
    pr = p.proof
    g = p.gate_status
    proof_cols: list[Any] = (
        [None] * 10
        if pr is None
        else [
            pr.objective_metric,
            pr.proved_improvement,
            pr.correctness_held,
            pr.realized_savings_absolute,
            pr.realized_savings_pct,
            pr.n_promote,
            pr.n_block,
            pr.n_errored,
            pr.suite_content_hash,
            pr.suite_version,
        ]
    )
    return [
        p.agent_name,
        p.experiment_id,
        p.proposal_id,
        p.schema_version,
        p.status.value,
        p.action_kind.value,
        p.risk_class.value,
        p.objective_metric,
        p.goal_cohort,
        t.kind.value,
        t.summary,
        t.metric,
        t.observed_value,
        t.threshold,
        t.n_traces,
        t.judge_name,
        t.asset_type,
        t.source_rank,
        json.dumps(t.trace_refs),
        c.kind.value,
        c.summary,
        c.sql,
        c.diff,
        c.evolved_body_ref,
        c.revert_target,
        c.plan,
        c.preview_diff,
        c.produced_change_ref,
        c.local_apply_spec.model_dump_json() if c.local_apply_spec is not None else None,
        "awaiting_approval" if c.local_apply_spec is not None else None,
        None,
        None,
        None,
        None,
        *proof_cols,
        g.readiness_tier,
        g.can_prove_improvement,
        g.judge_agreement,
        g.scored_coverage,
        g.n_distrusted_judges,
        g.gated,
        json.dumps(g.reasons),
        # verify lifecycle — a controller-published proposal is never verify-requested.
        # It becomes so only when a reviewer clicks "Verify on my suite" in the app
        # (the authenticated verify write-path); the poll handler then fills proof_* +
        # the terminal verify_status. So publish writes not-requested / all-NULL here.
        False,  # verify_requested
        None,  # verify_status
        None,  # verify_requested_by
        None,  # verify_requested_at
        None,  # verify_completed_at
        None,  # verify_error
        p.created_at,
        generated_at,
    ]


def _ddl(catalog: str, schema: str) -> list[str]:
    # MIGRATION NOTE (existing deployments): this is CREATE TABLE IF NOT EXISTS, so the
    # AGENT_TASK columns (change_plan / change_preview_diff / change_produced_change_ref)
    # land only on a FRESH table — an ALREADY-created ``agent_proposed_actions`` is not
    # ALTERed (same known pattern as the proof_* columns). Before L7b-2 publishes an
    # AGENT_TASK, an operator with a pre-existing table must add those three columns
    # (nullable STRING) once, e.g.:
    #   ALTER TABLE `<catalog>`.`<schema>`.agent_proposed_actions
    #     ADD COLUMNS (change_plan STRING, change_preview_diff STRING,
    #                  change_produced_change_ref STRING);
    # The same applies to the L9 verify_* lifecycle columns below — an operator with a
    # pre-existing table must add them once (nullable), e.g.:
    #   ALTER TABLE `<catalog>`.`<schema>`.agent_proposed_actions
    #     ADD COLUMNS (verify_requested BOOLEAN, verify_status STRING,
    #                  verify_requested_by STRING, verify_requested_at STRING,
    #                  verify_completed_at STRING, verify_error STRING);
    # No auto-ALTER is done here (out of scope). See docs/DEPLOY.md.
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{PROPOSALS_TABLE} (
            agent_name STRING,
            experiment_id STRING,
            proposal_id STRING,
            schema_version STRING,
            status STRING,
            action_kind STRING,
            risk_class STRING,
            objective_metric STRING,
            goal_cohort STRING,
            trigger_kind STRING,
            trigger_summary STRING,
            trigger_metric STRING,
            trigger_observed_value DOUBLE,
            trigger_threshold DOUBLE,
            trigger_n_traces INT,
            trigger_judge_name STRING,
            trigger_asset_type STRING,
            trigger_source_rank INT,
            trigger_trace_refs STRING,
            change_kind STRING,
            change_summary STRING,
            change_sql STRING,
            change_diff STRING,
            change_evolved_body_ref STRING,
            change_revert_target STRING,
            change_plan STRING,
            change_preview_diff STRING,
            change_produced_change_ref STRING,
            change_local_apply_spec_json STRING,
            local_apply_status STRING,
            local_apply_error STRING,
            local_apply_completed_at STRING,
            local_apply_pre_change_ref STRING,
            local_apply_validation_output STRING,
            proof_objective_metric STRING,
            proof_proved_improvement BOOLEAN,
            proof_correctness_held BOOLEAN,
            proof_realized_savings_absolute DOUBLE,
            proof_realized_savings_pct DOUBLE,
            proof_n_promote INT,
            proof_n_block INT,
            proof_n_errored INT,
            proof_suite_content_hash STRING,
            proof_suite_version STRING,
            gate_readiness_tier STRING,
            gate_can_prove_improvement BOOLEAN,
            gate_judge_agreement DOUBLE,
            gate_scored_coverage DOUBLE,
            gate_n_distrusted_judges INT,
            gate_gated BOOLEAN,
            gate_reasons STRING,
            verify_requested BOOLEAN,
            verify_status STRING,
            verify_requested_by STRING,
            verify_requested_at STRING,
            verify_completed_at STRING,
            verify_error STRING,
            created_at STRING,
            generated_at STRING
        ) USING DELTA
        COMMENT 'Pending human-gated proposed actions (why + proof + gate); lane 3 reads SELECT-only.'""",  # noqa: E501
    ]


def publish_agent_proposals(
    proposals: list[ProposedAction],
    *,
    agent_name: str,
    experiment_id: str,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    generated_at: str | None = None,
) -> int:
    """Append **one agent's** proposals to its durable approval queue.

    Existing keys are left untouched so a re-run cannot erase a preview, decision,
    or apply status. An empty list is a no-op; it never clears the queue.

    Returns the number of rows written.
    """
    mismatched = sorted({p.agent_name for p in proposals if p.agent_name != agent_name})
    if mismatched:
        raise ValueError(
            f"publish_agent_proposals is scoped to agent {agent_name!r} but got proposals for "
            f"{mismatched}; publish each agent separately."
        )

    mismatched_experiments = sorted(
        {p.experiment_id for p in proposals if p.experiment_id != experiment_id}
    )
    if mismatched_experiments:
        raise ValueError(
            f"publish_agent_proposals is scoped to experiment {experiment_id!r} but got "
            f"{mismatched_experiments}; publish each experiment independently."
        )

    stamp = generated_at or datetime.now(UTC).isoformat()
    fqn = f"`{catalog}`.`{schema}`"
    for ddl in _ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)

    for proposal in proposals:
        _execute(
            client,
            warehouse_id,
            _insert_if_absent_statement(fqn, proposal, generated_at=stamp),
        )
    return len(proposals)


def publish_proposals(
    proposals: list[ProposedAction],
    *,
    warehouse_id: str,
    profile: str | None = None,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    client: Any | None = None,
    generated_at: str | None = None,
) -> dict[str, int]:
    """Publish proposals for **any number of agents**, preserving every queue.

    Groups by agent/experiment and calls :func:`publish_agent_proposals` for each.
    Builds a workspace client (the
    :mod:`ail.publish` way) when one is not injected. Returns ``{agent_name: rows}``.
    """
    by_agent: dict[tuple[str, str], list[ProposedAction]] = defaultdict(list)
    for p in proposals:
        by_agent[(p.agent_name, p.experiment_id)].append(p)

    ws = client if client is not None else _build_workspace_client(profile)
    written: dict[str, int] = {}
    for (name, experiment_id), agent_proposals in by_agent.items():
        written[name] = publish_agent_proposals(
            agent_proposals,
            agent_name=name,
            experiment_id=experiment_id,
            client=ws,
            warehouse_id=warehouse_id,
            catalog=catalog,
            schema=schema,
            generated_at=generated_at,
        )
    return written


def insert_proposal_if_absent(
    proposal: ProposedAction,
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    generated_at: str | None = None,
) -> bool:
    """Insert one externally-produced proposal without replacing the agent slice.

    The GEPA job runs independently of the controller publisher. Reusing the
    publisher's agent-scoped ``REPLACE`` would erase unrelated pending/decided rows,
    so this path uses a guarded ``INSERT ... WHERE NOT EXISTS`` keyed by
    ``(agent_name, experiment_id, proposal_id)``. Repeating persistence for the same
    MLflow run is therefore idempotent and never resets an approved/applied row.
    """
    stamp = generated_at or datetime.now(UTC).isoformat()
    for ddl in _ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)
    fqn = f"`{catalog}`.`{schema}`"
    _execute(
        client,
        warehouse_id,
        _insert_if_absent_statement(fqn, proposal, generated_at=stamp),
    )
    return True


def _insert_if_absent_statement(fqn: str, proposal: ProposedAction, *, generated_at: str) -> str:
    """Escaped idempotent INSERT for one proposal, preserving an existing row."""
    row = _proposal_row(proposal, generated_at=generated_at)
    columns = ", ".join(PROPOSAL_COLUMNS)
    values = ", ".join(_lit(v) for v in row)
    table = f"{fqn}.{PROPOSALS_TABLE}"
    return (
        f"INSERT INTO {table} ({columns}) SELECT {values} "
        f"WHERE NOT EXISTS (SELECT 1 FROM {table} "
        f"WHERE agent_name = {_lit(proposal.agent_name)} "
        f"AND experiment_id = {_lit(proposal.experiment_id)} "
        f"AND proposal_id = {_lit(proposal.proposal_id)})"
    )
