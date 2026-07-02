"""``ail-agent-executor`` — the local companion runner for the open-ended executor.

The deployer-run companion (``docs/PRODUCT_ARCHITECTURE.md`` §4/§7, Claude Agent SDK
compute — **not** Databricks serverless) that drives lane L7b-2 end-to-end against the
app's ``agent_proposed_actions`` table. On one run it:

1. **polls PENDING ``AGENT_TASK`` proposals** that have no preview yet and calls
   :func:`ail.executor.produce_preview` for each — running the agent in an isolated
   sandbox copy of ``agent.target_workspace`` and recording the concrete
   ``preview_diff`` + ``produced_change_ref`` back onto the proposal row (so the app
   shows the human the real diff). A proposal that already carries a preview is
   **skipped** (never re-previewed — a re-run is non-deterministic and would move the
   diff out from under a reviewer);
2. **polls APPROVED ``AGENT_TASK`` proposals** and calls
   :func:`ail.executor.commit_approved` for each — applying the **stored** produced
   change-set (the exact diff the human approved) to the live workspace via the L6
   snapshot substrate (snapshot-live-first, then apply, then record), advancing the
   row to ``applied``. It **never re-runs the agent** at commit; and
3. **surfaces every step** to the operator (structured stdout): what it previewed /
   committed, and the fail-closed reason for anything it skipped or refused.

**Auth — a static token, matched to the workspace host (the hard-won lesson).** The
runner is a long-lived local process; a ``--profile`` OAuth login refreshes its token
mid-run and cannot persist from a background process. It reuses the companion's
:func:`ail.jobs.companion_planner.resolve_static_auth` — a **static** ``DATABRICKS_TOKEN``
pinned to ``DATABRICKS_HOST``, dropping any ambient ``DATABRICKS_CONFIG_PROFILE``,
refusing to run without one.

**Reuse, not reinvention.** The proposal read side is the *same* flat-row → proposal
mapping lane 3b uses (:func:`ail.loop.apply_service._row_to_proposal` /
``_query_rows`` — the "apply_service reader"); the SQL primitives are
:mod:`ail.publish`'s; the change-set versioning is :mod:`ail.versioning.snapshot`
(via :mod:`ail.executor`); the target workspace + experiment come off the
:class:`ail.registry.Agent` entry.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from typing import Any

from ail.executor import (
    CommitRecordError,
    CommitRefused,
    PreviewError,
    commit_approved,
    produce_preview,
)
from ail.jobs.companion_planner import resolve_static_auth
from ail.loop.apply_service import DECISIONS_TABLE, _query_rows, _row_to_proposal
from ail.loop.proposals import ActionKind, ProposalStatus, ProposedAction
from ail.loop.publish_proposals import PROPOSALS_TABLE
from ail.publish import DEFAULT_CATALOG, DEFAULT_SCHEMA, _build_workspace_client, _execute, _lit
from ail.registry import Agent, load_registry
from ail.versioning import SnapshotError, new_volume_client

__all__ = [
    "COMMITS_TABLE",
    "COMMIT_COLUMNS",
    "list_agent_task_proposals",
    "write_preview",
    "mark_committed",
    "latest_approver",
    "record_commit",
    "run",
    "main",
]

_TAG = "[ail.executor]"

#: Append-only audit of every committed open-ended change: the snapshot refs (the
#: revert point + the approved produced change-set), the file count, the approver, and
#: when. Distinct from ``agent_prompt_lineage`` (prompt-version lineage — an arbitrary
#: file change-set is not a prompt version) and from ``agent_action_decisions`` (the
#: human decision audit); this is the record a revert reads the pre-change snapshot from.
COMMITS_TABLE = "agent_executor_commits"

#: Column order — declared once, reused by the DDL and the INSERT (the
#: :mod:`ail.publish` convention) so the two can never drift.
COMMIT_COLUMNS: list[str] = [
    "agent_name",
    "proposal_id",
    "target_workspace",
    "produced_change_ref",
    "pre_change_ref",
    "n_files",
    "changed_paths",
    "summary",
    "approver",
    "committed_at",
    "recorded_at",
]


# ---------------------------------------------------------------------------
# Persistence — read (apply_service reader) + targeted writes (ail.publish SQL)
# ---------------------------------------------------------------------------


def list_agent_task_proposals(
    client: Any,
    warehouse_id: str,
    *,
    status: ProposalStatus,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> list[ProposedAction]:
    """Read this table's ``AGENT_TASK`` proposals in ``status`` (SELECT-only).

    Reuses the lane-3b "apply_service reader" (``_query_rows`` + ``_row_to_proposal``)
    so the flat-row → :class:`~ail.loop.proposals.ProposedAction` mapping is never
    re-implemented.
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    sql = (
        f"SELECT * FROM {fqn} "
        f"WHERE action_kind = {_lit(ActionKind.AGENT_TASK.value)} "
        f"AND status = {_lit(status.value)}"
    )
    return [_row_to_proposal(row) for row in _query_rows(client, warehouse_id, sql)]


def write_preview(
    client: Any,
    warehouse_id: str,
    *,
    agent_name: str,
    proposal_id: str,
    preview_diff: str,
    produced_change_ref: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    """Record a produced preview onto its still-pending, not-yet-previewed proposal row.

    The ``change_produced_change_ref IS NULL / ''`` guard makes this fail-closed and
    idempotent: it never overwrites a preview a human may already be reviewing.
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    _execute(
        client,
        warehouse_id,
        f"UPDATE {fqn} SET change_preview_diff = {_lit(preview_diff)}, "
        f"change_produced_change_ref = {_lit(produced_change_ref)} "
        f"WHERE agent_name = {_lit(agent_name)} AND proposal_id = {_lit(proposal_id)} "
        f"AND status = {_lit(ProposalStatus.PENDING.value)} "
        "AND (change_produced_change_ref IS NULL OR change_produced_change_ref = '')",
    )


def mark_committed(
    client: Any,
    warehouse_id: str,
    *,
    agent_name: str,
    proposal_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    """Advance an approved proposal to ``applied`` after its change is committed live.

    Scoped to ``status = 'approved'`` so only a still-approved row is advanced (the
    controller re-materializes the pending set on its next cycle; the durable record of
    the applied change is the ``agent_executor_commits`` audit).
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    _execute(
        client,
        warehouse_id,
        f"UPDATE {fqn} SET status = {_lit(ProposalStatus.APPLIED.value)} "
        f"WHERE agent_name = {_lit(agent_name)} AND proposal_id = {_lit(proposal_id)} "
        f"AND status = {_lit(ProposalStatus.APPROVED.value)}",
    )


def latest_approver(
    client: Any,
    warehouse_id: str,
    *,
    agent_name: str,
    proposal_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> str | None:
    """The authenticated approver recorded for this proposal's latest approve decision.

    Read from the lane-3b ``agent_action_decisions`` audit (the app records the
    authenticated approver there on every approve). ``None`` when no approve decision
    is recorded (or the audit is unreadable) — the caller falls back to the configured
    operator identity and surfaces the source.
    """
    fqn = f"`{catalog}`.`{schema}`.{DECISIONS_TABLE}"
    sql = (
        f"SELECT approver FROM {fqn} "
        f"WHERE agent_name = {_lit(agent_name)} AND proposal_id = {_lit(proposal_id)} "
        "AND decision = 'approve' ORDER BY decided_at DESC LIMIT 1"
    )
    try:
        rows = _query_rows(client, warehouse_id, sql)
    except Exception:  # noqa: BLE001 - a missing/unreadable audit is a soft None, not fatal
        return None
    if not rows:
        return None
    approver = rows[0].get("approver")
    return str(approver) if approver else None


def _commits_ddl(catalog: str, schema: str) -> list[str]:
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{COMMITS_TABLE} (
            agent_name STRING,
            proposal_id STRING,
            target_workspace STRING,
            produced_change_ref STRING,
            pre_change_ref STRING,
            n_files INT,
            changed_paths STRING,
            summary STRING,
            approver STRING,
            committed_at STRING,
            recorded_at STRING
        ) USING DELTA
        COMMENT 'Append-only audit of committed open-ended AGENT_TASK changes (L7b-2 executor); carries the revert-point + approved change-set snapshot refs.'""",  # noqa: E501
    ]


def record_commit(
    record: Any,
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    recorded_at: str | None = None,
) -> None:
    """Append one :class:`~ail.executor.CommittedChangeRecord` to ``agent_executor_commits``."""
    for ddl in _commits_ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)
    stamp = recorded_at or datetime.now(UTC).isoformat()
    values = [
        record.agent_name,
        record.proposal_id,
        record.target_workspace,
        record.produced_change_ref,
        record.pre_change_ref,
        record.n_files,
        json.dumps(record.changed_paths),
        record.summary,
        record.approver,
        record.committed_at,
        stamp,
    ]
    fqn = f"`{catalog}`.`{schema}`.{COMMITS_TABLE}"
    cols = ", ".join(COMMIT_COLUMNS)
    literals = ", ".join(_lit(v) for v in values)
    _execute(client, warehouse_id, f"INSERT INTO {fqn} ({cols}) VALUES ({literals})")


# ---------------------------------------------------------------------------
# Live seam builders (monkeypatched in tests → no live client is ever built)
# ---------------------------------------------------------------------------


def _resolve_agent(agent_name: str, registry_path: str | None) -> Agent:
    path = registry_path if registry_path and os.path.exists(registry_path) else None
    return load_registry(path).get(agent_name)


def _build_volume_client(profile: str | None) -> Any:
    return new_volume_client(profile)


def _build_agent_runner(agent: Agent, trace: bool) -> Any:
    # Imported lazily via ail.executor so the module (and its default) stays offline.
    from ail.executor.executor import _default_agent_runner

    return _default_agent_runner(mlflow_experiment=agent.experiment_id if trace else None)


# ---------------------------------------------------------------------------
# One run: poll → preview pending → commit approved → surface
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute one executor run; return a process exit code.

    ``0`` on a completed run (previews produced/committed, or a clean dry-run).
    ``2`` fail-closed when the proposal table cannot be read (nothing is done — never a
    fabricated preview, never a commit on an unknown state).
    """
    try:
        agent = _resolve_agent(args.agent, args.registry)
    except KeyError as exc:
        print(f"{_TAG} ERROR: {exc}; nothing to do (fail-closed).")
        return 2

    print(
        f"{_TAG} agent={agent.agent_name} workspace={agent.target_workspace!r} "
        f"host={os.environ.get('DATABRICKS_HOST')} "
        f"table={args.catalog}.{args.schema}.{PROPOSALS_TABLE} volume_root={args.volume_root!r} "
        f"dry_run={args.dry_run}"
    )

    client = _build_workspace_client(None)  # static env token (resolve_static_auth pinned it)

    # Read both slices FIRST, fail-closed: an unreadable table prints an honest error
    # and does nothing (never previews on an unknown state).
    try:
        pending = list_agent_task_proposals(
            client,
            args.warehouse_id,
            status=ProposalStatus.PENDING,
            catalog=args.catalog,
            schema=args.schema,
        )
        approved = list_agent_task_proposals(
            client,
            args.warehouse_id,
            status=ProposalStatus.APPROVED,
            catalog=args.catalog,
            schema=args.schema,
        )
    except Exception as exc:  # noqa: BLE001 - surface honestly; do nothing on an unknown state
        print(
            f"{_TAG} ERROR: could not read AGENT_TASK proposals "
            f"({type(exc).__name__}: {exc}); doing nothing (fail-closed)."
        )
        return 2

    to_preview = [p for p in pending if not (p.change.produced_change_ref or "").strip()]
    already = len(pending) - len(to_preview)
    print(
        f"{_TAG} --- POLL --- pending AGENT_TASK={len(pending)} "
        f"(to preview={len(to_preview)}, already previewed={already}); approved={len(approved)}"
    )

    if args.dry_run:
        for p in to_preview:
            print(f"{_TAG}   WOULD PREVIEW {p.proposal_id} — plan: {(p.change.plan or '')[:120]!r}")
        for p in approved:
            print(f"{_TAG}   WOULD COMMIT {p.proposal_id} — ref: {p.change.produced_change_ref!r}")
        print(f"{_TAG} DRY-RUN: no agent run, no snapshot, no commit, no row write.")
        return 0

    volume_client = _build_volume_client(None)
    runner = _build_agent_runner(agent, args.trace)

    _run_previews(to_preview, agent, client, volume_client, runner, args)
    _run_commits(approved, agent, client, volume_client, args)
    return 0


def _run_previews(
    to_preview: list[ProposedAction],
    agent: Agent,
    client: Any,
    volume_client: Any,
    runner: Any,
    args: argparse.Namespace,
) -> None:
    print(f"{_TAG} --- PREVIEW ({len(to_preview)} pending without a preview) ---")
    for p in to_preview:

        def _writer(
            *, agent_name: str, proposal_id: str, preview_diff: str, produced_change_ref: str
        ) -> None:
            write_preview(
                client,
                args.warehouse_id,
                agent_name=agent_name,
                proposal_id=proposal_id,
                preview_diff=preview_diff,
                produced_change_ref=produced_change_ref,
                catalog=args.catalog,
                schema=args.schema,
            )

        try:
            res = produce_preview(
                p,
                agent,
                volume_client=volume_client,
                volume_root=args.volume_root,
                preview_writer=_writer,
                agent_runner=runner,
                timeout_seconds=args.timeout,
                model=args.model,
            )
            print(
                f"{_TAG}   PREVIEWED {p.proposal_id}: {res.n_files} file(s) changed -> "
                f"ref {res.produced_change_ref}"
            )
            for change in res.changes:
                print(f"{_TAG}     [{change.change_type}] {change.path}")
        except PreviewError as exc:
            print(f"{_TAG}   SKIPPED {p.proposal_id} (fail-closed): {exc}")


def _run_commits(
    approved: list[ProposedAction],
    agent: Agent,
    client: Any,
    volume_client: Any,
    args: argparse.Namespace,
) -> None:
    print(f"{_TAG} --- COMMIT ({len(approved)} approved) ---")
    for p in approved:
        approver = (
            latest_approver(
                client,
                args.warehouse_id,
                agent_name=p.agent_name,
                proposal_id=p.proposal_id,
                catalog=args.catalog,
                schema=args.schema,
            )
            or args.operator
        )
        committed_at = datetime.now(UTC).isoformat()

        def _recorder(record: Any) -> None:
            record_commit(
                record,
                client=client,
                warehouse_id=args.warehouse_id,
                catalog=args.catalog,
                schema=args.schema,
            )

        try:
            res = commit_approved(
                p,
                agent,
                volume_client=volume_client,
                volume_root=args.volume_root,
                commit_recorder=_recorder,
                approver=approver,
                committed_at=committed_at,
            )
            mark_committed(
                client,
                args.warehouse_id,
                agent_name=p.agent_name,
                proposal_id=p.proposal_id,
                catalog=args.catalog,
                schema=args.schema,
            )
            print(
                f"{_TAG}   COMMITTED {p.proposal_id}: {res.n_files} file(s) applied to "
                f"{res.target_workspace} (approver={approver}, revert_point={res.pre_change_ref})"
            )
        except CommitRefused as exc:
            print(f"{_TAG}   REFUSED {p.proposal_id} (fail-closed): {exc}")
        except CommitRecordError as exc:
            # The change is LIVE but its record failed — advance status and surface
            # committed-but-unrecorded for reconciliation (never a fake not-applied state).
            mark_committed(
                client,
                args.warehouse_id,
                agent_name=p.agent_name,
                proposal_id=p.proposal_id,
                catalog=args.catalog,
                schema=args.schema,
            )
            print(f"{_TAG}   COMMITTED-BUT-UNRECORDED {p.proposal_id} (reconcile): {exc}")
        except SnapshotError as exc:
            # The apply itself failed; L6 leaves the live tree restorable/untouched.
            print(
                f"{_TAG}   APPLY FAILED {p.proposal_id} "
                f"(live tree restorable, not committed): {exc}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local companion executor (L7b-2): preview PENDING AGENT_TASK proposals in a "
            "sandbox, and commit APPROVED ones to the live workspace via the L6 snapshot "
            "substrate. Never re-runs the agent at commit; applies the stored, approved change."
        )
    )
    parser.add_argument("--agent", default="claude_code", help="Agent name (proposal scope).")
    parser.add_argument(
        "--registry",
        default="config/agents.yaml",
        help="Agent registry YAML (must set target_workspace for --agent). Falls back to the "
        "in-code seed if absent (which has no target_workspace → the executor fails closed).",
    )
    parser.add_argument(
        "--volume-root",
        default=os.environ.get("AIL_SNAPSHOT_VOLUME"),
        help="UC Volume dir under /Volumes/... to snapshot change-sets into "
        "(or set AIL_SNAPSHOT_VOLUME).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("DATABRICKS_HOST"),
        help="Workspace host (sets DATABRICKS_HOST). A STATIC DATABRICKS_TOKEN pinned to this "
        "host is required; --profile OAuth is refused.",
    )
    parser.add_argument("--warehouse-id", default=os.environ.get("AIL_WAREHOUSE_ID"))
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--operator",
        default=os.environ.get("USER") or "ail-agent-executor",
        help="Fallback approver identity recorded for a commit when no approve decision is found "
        "in the audit (the human's authenticated approver is preferred when present).",
    )
    parser.add_argument("--model", default=None, help="Optional model override for the agent run.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-run agent timeout seconds (default: the executor default).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Trace each sandbox agent run to the agent's MLflow experiment (off by default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Poll and surface what WOULD be previewed/committed, but run no agent, take no "
        "snapshot, and commit/write nothing.",
    )
    parser.add_argument(
        "--token-secret-scope", default=os.environ.get("AIL_TOKEN_SECRET_SCOPE", "")
    )
    parser.add_argument("--token-secret-key", default=os.environ.get("AIL_TOKEN_SECRET_KEY", ""))
    args = parser.parse_args(argv)
    if not args.warehouse_id:
        parser.error("--warehouse-id is required (or set AIL_WAREHOUSE_ID)")
    if not args.volume_root:
        parser.error("--volume-root is required (or set AIL_SNAPSHOT_VOLUME)")
    if args.timeout is None:
        from ail.executor import DEFAULT_TIMEOUT_SECONDS

        args.timeout = DEFAULT_TIMEOUT_SECONDS
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth_path = resolve_static_auth(args)
    print(f"{_TAG} auth={auth_path} host={os.environ.get('DATABRICKS_HOST')}")
    if auth_path == "minted":
        print(
            f"{_TAG} WARNING: auth was MINTED from ambient identity, not a static token. For a "
            "long local run, export a static DATABRICKS_TOKEN pinned to --host instead — a minted "
            "OAuth bearer risks a mid-run refresh that cannot persist from a background process."
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
