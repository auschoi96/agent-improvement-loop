"""L9 — the opt-in **Tier-2** "Verify on my suite" request lifecycle (evidence only).

``docs/PRODUCT_ARCHITECTURE.md`` (§3/§7) makes frozen-suite proving *opt-in*: an
evidence-first proposal ships to the approval queue on its Tier-1 evidence + gate
alone (``proof=None``). A reviewer who wants **harder** evidence before deciding can
click **"Verify on my suite"** on a proposal — this module is the two halves of that
opt-in flow, both **fail-closed** and both **evidence only** (proving never approves):

1. **The request write-path** (:func:`run_verify_request` / :func:`main`). The app's
   authenticated verify route (``server/plugins/approvals``) bridges to this the same
   way the Approve/Reject route bridges to :mod:`ail.loop.apply_service`: a JSON
   request on stdin, a JSON :class:`VerifyRequestResult` on stdout. It loads the
   authoritative *pending* proposal (never trusting a client body), refuses a
   non-provable action kind (a metric-view / revert / agent-task the frozen suite
   cannot run) and an anonymous requester, and — only then — flips the proposal's
   ``verify_requested`` flag + ``verify_status='requested'`` in Unity Catalog. It
   applies nothing and proves nothing; it *requests*.

2. **The poll handler** (:func:`run_verify_tick`, driven by
   :func:`ail.companion.cli.run_poll`). On the deployer's companion it picks up the
   pending verify-requests, runs the **existing** frozen-suite prover
   (:func:`ail.optimize.run_phase2_comparison` — never reimplemented here), and writes
   the RESULT back keyed to the proposal. The result reuses the existing ``proof_*``
   columns (no parallel proof schema): a real proof populates them and sets a terminal
   ``verify_status`` of ``verified`` (PROMOTE: improvement proved *and* correctness
   held) or ``blocked`` (anything less).

**Fail-closed honesty (load-bearing).** A missing / unfrozen suite writes an honest
``no_suite`` state; a prove that raises writes an honest ``errored`` state — **never**
a fabricated ``verified``, and in both cases every ``proof_*`` column is written
``NULL`` (no invented numbers). The proof is **evidence only**: every write here is
scoped to the still-``pending`` row and **never touches ``status``** — verifying does
not, and cannot, approve a proposal. The human still decides.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ail.loop.apply_service import _query_rows, _resolve_agent, load_pending_proposal
from ail.loop.proposals import ActionKind, ProofSummary, ProposalStatus
from ail.loop.publish_proposals import PROPOSALS_TABLE
from ail.optimize.phase2 import Phase2Artifact
from ail.optimize.prompt_registry import DEFAULT_CATALOG, DEFAULT_SCHEMA
from ail.publish import _build_workspace_client, _execute, _lit
from ail.task_suite.schema import TaskSuite
from ail.workspace_config import resolve_catalog_schema

__all__ = [
    "PROVABLE_ACTION_KINDS",
    "VerifyStatus",
    "VerifyRequestOutcome",
    "VerifyRequestResult",
    "VerifyTickSummary",
    "is_provable",
    "mark_verify_requested",
    "select_pending_verify_requests",
    "write_verify_result",
    "run_verify_request",
    "run_verify_tick",
    "main",
]


#: The action kinds the frozen suite can actually run a baseline-vs-candidate proof
#: for — a skill / instruction / prompt-behaviour change. A metric-view (an additive
#: read-path asset), a revert, or an open-ended agent-task cannot be *proven* by the
#: suite, so "verify on my suite" is N/A for them: the app greys the button and this
#: engine refuses a request for one (defence in depth — the browser is never trusted).
PROVABLE_ACTION_KINDS: frozenset[ActionKind] = frozenset(
    {ActionKind.SKILL_UPDATE, ActionKind.INSTRUCTION_UPDATE, ActionKind.GEPA_PROMPT}
)


def is_provable(action_kind: ActionKind) -> bool:
    """True iff the frozen suite can run a proof for ``action_kind`` (see above)."""
    return action_kind in PROVABLE_ACTION_KINDS


class VerifyStatus(StrEnum):
    """The proposal's verify lifecycle state, stored in ``verify_status`` (UC).

    ``requested`` is set by the write-path; the poll handler advances it to exactly
    one honest terminal state. There is no ``running`` intermediate — a tick either
    completes the prove (``verified`` / ``blocked``) or records the honest failure
    (``errored`` / ``no_suite``) in the same write.
    """

    REQUESTED = "requested"
    #: PROMOTE — the frozen suite proved an improvement AND correctness held.
    VERIFIED = "verified"
    #: BLOCK — the prove completed but did not clear the bar (shown honestly as a block).
    BLOCKED = "blocked"
    #: The prove raised — an honest error state, never a fabricated verified.
    ERRORED = "errored"
    #: No frozen suite is configured / it is unfrozen — fail-closed, honest.
    NO_SUITE = "no_suite"


class VerifyRequestOutcome(StrEnum):
    """The outcome the app surfaces for a *verify request* (not the proof itself)."""

    REQUESTED = "requested"
    #: A fail-closed refusal — nothing was requested (surface :attr:`refused_reason`).
    REFUSED = "refused"
    #: An infrastructure error around the request write — never a fake "requested".
    ERROR = "error"


class VerifyRequestResult(BaseModel):
    """The flat, JSON-round-trippable outcome of a verify *request* the app renders."""

    model_config = ConfigDict(extra="forbid")

    outcome: VerifyRequestOutcome
    proposal_id: str
    agent_name: str
    requested_by: str
    requested_at: str
    action_kind: str | None = None
    #: The verify_status now stored on the proposal (``requested``) on success.
    verify_status: str | None = None
    refused_reason: str | None = None
    error: str | None = None


class VerifyTickSummary(BaseModel):
    """What one poll tick did — counts per terminal state (for the run log)."""

    model_config = ConfigDict(extra="forbid")

    n_requested: int = 0
    n_verified: int = 0
    n_blocked: int = 0
    n_errored: int = 0
    n_no_suite: int = 0


# ---------------------------------------------------------------------------
# UC reads / writes (reuse ail.publish rendering + apply_service's SELECT reader)
# ---------------------------------------------------------------------------

#: The ten ``proof_*`` columns :func:`ail.loop.publish_proposals._proposal_row` writes;
#: the verify RESULT reuses them (no parallel proof schema). Declared once so the two
#: write branches (real proof / all-NULL) can never drift.
_PROOF_COLUMNS: tuple[str, ...] = (
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
)


def _proof_assignments(proof: ProofSummary | None) -> str:
    """Render the ``SET proof_* = …`` clause for :func:`write_verify_result`.

    ``proof is None`` (an errored / no-suite tick) writes every ``proof_*`` column
    ``NULL`` — a failed prove NEVER leaves a fabricated proof behind. A real proof
    writes its verbatim values; the booleans/ints stay typed via :func:`_lit`.
    """
    if proof is None:
        return ", ".join(f"{c} = NULL" for c in _PROOF_COLUMNS)
    values: dict[str, Any] = {
        "proof_objective_metric": proof.objective_metric,
        "proof_proved_improvement": proof.proved_improvement,
        "proof_correctness_held": proof.correctness_held,
        "proof_realized_savings_absolute": proof.realized_savings_absolute,
        "proof_realized_savings_pct": proof.realized_savings_pct,
        "proof_n_promote": proof.n_promote,
        "proof_n_block": proof.n_block,
        "proof_n_errored": proof.n_errored,
        "proof_suite_content_hash": proof.suite_content_hash,
        "proof_suite_version": proof.suite_version,
    }
    return ", ".join(f"{c} = {_lit(values[c])}" for c in _PROOF_COLUMNS)


def mark_verify_requested(
    *,
    client: Any,
    warehouse_id: str,
    agent_name: str,
    experiment_id: str,
    proposal_id: str,
    requested_by: str,
    requested_at: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    """Flip a pending proposal to ``verify_requested`` (in place, scoped to the row).

    Clears any prior ``verify_completed_at`` / ``verify_error`` so a *re-request* of a
    previously errored/blocked proof starts clean. Scoped to ``status='pending'`` —
    a decided/superseded proposal is never (re-)verified. The proof itself is left
    untouched here; the poll handler overwrites ``proof_*`` when it runs.
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    _execute(
        client,
        warehouse_id,
        f"UPDATE {fqn} SET verify_requested = TRUE, "
        f"verify_status = {_lit(VerifyStatus.REQUESTED.value)}, "
        f"verify_requested_by = {_lit(requested_by)}, "
        f"verify_requested_at = {_lit(requested_at)}, "
        f"verify_completed_at = NULL, verify_error = NULL "
        f"WHERE agent_name = {_lit(agent_name)} AND experiment_id = {_lit(experiment_id)} "
        f"AND proposal_id = {_lit(proposal_id)} "
        f"AND status = {_lit(ProposalStatus.PENDING.value)}",
    )


def select_pending_verify_requests(
    *,
    client: Any,
    warehouse_id: str,
    agent_name: str,
    experiment_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> list[dict[str, Any]]:
    """Return ``[{proposal_id, action_kind}]`` for this agent's pending verify requests.

    A row qualifies only while it is still ``pending`` AND ``verify_status`` is
    ``requested`` — so a decided proposal, or one whose proof already ran, is never
    picked up again. Reuses :func:`ail.loop.apply_service._query_rows` (the canonical
    SELECT reader) rather than reimplementing the statement wait loop.
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    sql = (
        f"SELECT proposal_id, action_kind FROM {fqn} "
        f"WHERE agent_name = {_lit(agent_name)} "
        f"AND experiment_id = {_lit(experiment_id)} "
        f"AND status = {_lit(ProposalStatus.PENDING.value)} "
        f"AND verify_status = {_lit(VerifyStatus.REQUESTED.value)}"
    )
    return _query_rows(client, warehouse_id, sql)


def write_verify_result(
    *,
    client: Any,
    warehouse_id: str,
    agent_name: str,
    experiment_id: str,
    proposal_id: str,
    proof: ProofSummary | None,
    verify_status: VerifyStatus,
    verify_error: str | None,
    completed_at: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    """Write the frozen-suite proof RESULT (or an honest failure) to the proposal.

    **Evidence-only / fail-closed (load-bearing):** the write reuses the ``proof_*``
    columns for the result and is scoped to the still-``pending`` row, and it **never
    touches ``status``** — a proof, however strong, does not approve a proposal.
    ``proof is None`` (errored / no-suite) writes every ``proof_*`` column ``NULL`` so
    a failed prove can never masquerade as a verified one.
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    _execute(
        client,
        warehouse_id,
        f"UPDATE {fqn} SET {_proof_assignments(proof)}, "
        f"verify_status = {_lit(verify_status.value)}, "
        f"verify_completed_at = {_lit(completed_at)}, "
        f"verify_error = {_lit(verify_error)} "
        f"WHERE agent_name = {_lit(agent_name)} AND experiment_id = {_lit(experiment_id)} "
        f"AND proposal_id = {_lit(proposal_id)} "
        f"AND status = {_lit(ProposalStatus.PENDING.value)}",
    )


# ---------------------------------------------------------------------------
# 1) The request write-path (app -> companion bridge, mirrors apply_service)
# ---------------------------------------------------------------------------


def _refused(
    proposal_id: str,
    agent_name: str,
    requested_by: str,
    requested_at: str,
    reason: str,
    *,
    action_kind: str | None = None,
) -> VerifyRequestResult:
    return VerifyRequestResult(
        outcome=VerifyRequestOutcome.REFUSED,
        proposal_id=proposal_id,
        agent_name=agent_name,
        requested_by=requested_by,
        requested_at=requested_at,
        action_kind=action_kind,
        refused_reason=reason,
    )


def _errored(
    proposal_id: str, agent_name: str, requested_by: str, requested_at: str, message: str
) -> VerifyRequestResult:
    return VerifyRequestResult(
        outcome=VerifyRequestOutcome.ERROR,
        proposal_id=proposal_id,
        agent_name=agent_name,
        requested_by=requested_by,
        requested_at=requested_at,
        error=message,
    )


def run_verify_request(
    *,
    proposal_id: str,
    agent_name: str,
    requested_by: str,
    requested_at: str,
    profile: str | None = None,
    warehouse_id: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    registry_path: str | None = None,
) -> VerifyRequestResult:
    """Load the authoritative pending proposal and (if provable) request a verify.

    Fail-closed at every step: an anonymous requester, an unresolvable warehouse, a
    missing/non-pending proposal, or a non-provable action kind all REFUSE — nothing
    is flagged. Only a provable, pending proposal requested by an authenticated user
    is flipped to ``verify_requested``. Any infra failure is an honest ERROR, never a
    fake "requested".
    """
    if not requested_by.strip():
        return _refused(
            proposal_id,
            agent_name,
            requested_by,
            requested_at,
            "refusing an anonymous verify request — no authenticated requester identity",
        )
    wh = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not wh:
        return _errored(
            proposal_id,
            agent_name,
            requested_by,
            requested_at,
            "no SQL warehouse id (set DATABRICKS_WAREHOUSE_ID or pass warehouse_id)",
        )

    try:
        resolved_catalog, resolved_schema = resolve_catalog_schema(catalog, schema)
        client = _build_workspace_client(profile)
        agent = _resolve_agent(
            agent_name,
            registry_path,
            client=client,
            warehouse_id=wh,
            catalog=resolved_catalog,
            schema=resolved_schema,
        )
        proposal = load_pending_proposal(
            client=client,
            warehouse_id=wh,
            agent_name=agent_name,
            proposal_id=proposal_id,
            experiment_id=agent.experiment_id,
            catalog=resolved_catalog,
            schema=resolved_schema,
        )
        if proposal is None:
            return _refused(
                proposal_id,
                agent_name,
                requested_by,
                requested_at,
                "no pending proposal with that id for this agent — already decided, "
                "superseded, or unknown (fail-closed)",
            )
        if not is_provable(proposal.action_kind):
            return _refused(
                proposal_id,
                agent_name,
                requested_by,
                requested_at,
                f"action kind {proposal.action_kind.value!r} cannot be proven on the frozen "
                "suite — 'verify on my suite' is evidence for skill / instruction / prompt "
                "changes only (fail-closed)",
                action_kind=proposal.action_kind.value,
            )
        mark_verify_requested(
            client=client,
            warehouse_id=wh,
            agent_name=agent_name,
            experiment_id=agent.experiment_id,
            proposal_id=proposal_id,
            requested_by=requested_by,
            requested_at=requested_at,
            catalog=resolved_catalog,
            schema=resolved_schema,
        )
        return VerifyRequestResult(
            outcome=VerifyRequestOutcome.REQUESTED,
            proposal_id=proposal_id,
            agent_name=agent_name,
            requested_by=requested_by,
            requested_at=requested_at,
            action_kind=proposal.action_kind.value,
            verify_status=VerifyStatus.REQUESTED.value,
        )
    except Exception as exc:  # noqa: BLE001 - any infra failure is an honest ERROR, never a fake request
        return _errored(
            proposal_id,
            agent_name,
            requested_by,
            requested_at,
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# 2) The poll handler (companion) — run the existing prover, write the result
# ---------------------------------------------------------------------------


def run_verify_tick(
    *,
    agent_name: str,
    select_requested: Callable[[], list[dict[str, Any]]],
    load_suite: Callable[[], TaskSuite],
    run_prover: Callable[[TaskSuite], Phase2Artifact],
    write_result: Callable[..., None],
    now: Callable[[], str] | None = None,
) -> VerifyTickSummary:
    """Process this agent's pending verify requests once — fail-closed, evidence only.

    Every seam is injected (:func:`ail.companion.cli.run_poll` wires the live ones) so
    the whole fail-closed matrix is testable with no live workspace. The flow:

    * **No requests** → clean no-op.
    * **Suite missing / unfrozen** (``load_suite`` raises) → write an honest
      ``no_suite`` state to every request; no proof numbers.
    * **Prove raises** → write an honest ``errored`` state to every request; no proof.
    * **Prove completes** → attach the resulting :class:`ProofSummary` to every
      request and set ``verified`` (improvement proved AND correctness held) or
      ``blocked`` (anything less — shown honestly as a block, not dressed up).

    The frozen-suite prove runs **once** per tick: in the current lever model it is a
    baseline-vs-candidate comparison (proposal-independent, same as the ``prove``
    subcommand), so the tick's proof is the evidence attached to each requesting
    proposal. No write here touches ``status``.
    """
    stamp = now or (lambda: datetime.now(UTC).isoformat())
    requested = select_requested()
    summary = VerifyTickSummary(n_requested=len(requested))
    if not requested:
        return summary

    proposal_ids = [str(row["proposal_id"]) for row in requested]

    try:
        suite = load_suite()
    except Exception as exc:  # noqa: BLE001 - a missing/unfrozen suite fails closed to no_suite
        message = f"no frozen suite configured: {type(exc).__name__}: {exc}"
        for pid in proposal_ids:
            write_result(
                proposal_id=pid,
                proof=None,
                verify_status=VerifyStatus.NO_SUITE,
                verify_error=message,
                completed_at=stamp(),
            )
        summary.n_no_suite = len(proposal_ids)
        return summary

    try:
        artifact = run_prover(suite)
    except Exception as exc:  # noqa: BLE001 - a prove failure is an HONEST errored state, never verified
        message = f"prove failed: {type(exc).__name__}: {exc}"
        for pid in proposal_ids:
            write_result(
                proposal_id=pid,
                proof=None,
                verify_status=VerifyStatus.ERRORED,
                verify_error=message,
                completed_at=stamp(),
            )
        summary.n_errored = len(proposal_ids)
        return summary

    proof = ProofSummary.from_phase2_artifact(artifact)
    verified = proof.proved_improvement and proof.correctness_held
    status = VerifyStatus.VERIFIED if verified else VerifyStatus.BLOCKED
    for pid in proposal_ids:
        write_result(
            proposal_id=pid,
            proof=proof,
            verify_status=status,
            verify_error=None,
            completed_at=stamp(),
        )
    if verified:
        summary.n_verified = len(proposal_ids)
    else:
        summary.n_blocked = len(proposal_ids)
    return summary


# ---------------------------------------------------------------------------
# CLI bridge — read a JSON verify request on stdin, print a JSON result on stdout
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI bridge for the app's authenticated verify route (subprocess transport).

    Mirrors :func:`ail.loop.apply_service.main`: the Node/AppKit route (which
    authenticates the requester) invokes ``python -m ail.loop.verify_service`` as a
    subprocess, passing ``{proposal_id, agent_name, requested_by, requested_at}`` —
    ``requested_by`` is the *authenticated* app user injected by the route (never
    trusted from the browser). Always prints a parseable :class:`VerifyRequestResult`
    and returns ``0`` for a request-level outcome (including a fail-closed REFUSED);
    returns non-zero only when stdin itself is unparseable.
    """
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        print(json.dumps({"outcome": "error", "error": f"unparseable stdin: {exc}"}))
        return 2
    if not isinstance(payload, dict):
        print(json.dumps({"outcome": "error", "error": "stdin must be a JSON object"}))
        return 2

    result = run_verify_request(
        proposal_id=str(payload.get("proposal_id", "")),
        agent_name=str(payload.get("agent_name", "")),
        requested_by=str(payload.get("requested_by", "")),
        requested_at=str(payload.get("requested_at") or datetime.now(UTC).isoformat()),
        profile=payload.get("profile"),
        warehouse_id=payload.get("warehouse_id"),
        catalog=payload.get("catalog"),
        schema=payload.get("schema"),
        registry_path=payload.get("registry_path"),
    )
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
