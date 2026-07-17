"""Lane 3b — the **server side** of the app's authenticated Approve/Reject write-path.

``docs/LOOP_CONTROLLER.md`` (Option A) closes the loop with a human approval gate:
the reviewer reads a proposal's *why + proof + gate status* in the app and clicks
**Approve** / **Reject**. The app is a thin AppKit (Node/React) app; this module is
the **Python** the app's authenticated server action invokes — the bridge between
the AppKit route (which authenticates the reviewer) and the lane-3a apply *engine*
(:func:`ail.loop.apply.apply_approved_proposal`, which turns an approved proposal
into a live change behind the fail-closed gates).

It does exactly three things, and **reuses** the framework's existing capabilities
rather than reimplementing any of them:

1. **Loads the authoritative proposal** from the unified ``agent_proposed_actions``
   UC table by ``(agent_name, proposal_id)`` — never trusting a client-supplied
   proposal body (the human could tamper). Only a still-``pending`` row is acted on.
2. **Wires the REAL seams** the engine needs and calls
   :func:`~ail.loop.apply.apply_approved_proposal` — the *only* place a proposal
   becomes a live change. The seams are wired to the MLflow prompt registry
   (:mod:`ail.optimize.prompt_registry`), the framework SQL warehouse
   (:mod:`ail.publish`), the lineage timeline (:mod:`ail.publish_lineage`), and the
   readiness gate (:mod:`ail.readiness` + :func:`ail.loop.controller.evaluate_gate`).
   The button does **not** bypass the gates: the engine re-verifies the proof and
   re-runs the readiness/judge gate at apply time; the UI only triggers.
3. **Records the human decision** — the authenticated approver, timestamp, decision,
   and (required for a reject) reason — into an append-only ``agent_action_decisions``
   audit table, and advances the proposal's ``status`` so the queue drops it on the
   next refresh. A **reject** records the decision and calls no capability.

**Fail-closed everywhere.** An empty approver, a missing/non-pending proposal, a
failed apply-time gate re-check, or an unproven proposal all refuse — nothing is
applied. A refusal is surfaced with its reason (never a silent no-op); an
``ApplyRecordError`` (the change is live but the lineage record failed) is surfaced
as *applied-but-unrecorded — reconcile*, distinct from both success and refusal.

**Unit-testable with no live write.** Every seam and both persistence writers are
injectable; :func:`decide_on_proposal` (the orchestration the tests exercise) runs
against fakes, and the live wiring (:func:`run_decision`, :func:`main`) is a thin
composition on top. The one CLI entry (``python -m ail.loop.apply_service``) reads a
JSON decision on stdin and prints a JSON :class:`ApplyServiceResult` on stdout, so
the Node route invokes it as a subprocess.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from ail.loop.apply import (
    CHAMPION_ALIAS,
    ApplyRecordError,
    ApplyRefused,
    ApplyRegistryClient,
    ApplyResult,
    ApprovalDecision,
    BodyResolver,
    DecisionKind,
    GateRecheck,
    GateRecheckResult,
    LineageRecorder,
    RegisterableBody,
    WarehouseExecutor,
    apply_approved_proposal,
)
from ail.loop.controller import evaluate_gate
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    LocalApplySpec,
    ProofSummary,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    RiskClass,
    TriggerKind,
    TriggerSignal,
)
from ail.loop.publish_proposals import PROPOSALS_TABLE
from ail.optimize.prompt_registry import (
    DEFAULT_CATALOG,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SCHEMA,
    PromptProvenance,
    PromptSource,
    resolve_prompt_name,
)
from ail.publish import _build_workspace_client, _execute, _lit
from ail.publish_lineage import new_lineage_client, publish_agent_lineage
from ail.workspace_config import resolve_catalog_schema

__all__ = [
    "DECISIONS_TABLE",
    "DECISION_COLUMNS",
    "ApplyServiceOutcome",
    "ApplyServiceResult",
    "DecisionWriter",
    "StatusWriter",
    "decide_on_proposal",
    "record_decision",
    "mark_proposal_status",
    "load_pending_proposal",
    "build_registry_client",
    "build_warehouse_executor",
    "build_lineage_recorder",
    "build_gate_recheck",
    "build_body_resolver",
    "run_decision",
    "main",
]

#: Append-only audit of every human decision (approve *and* reject). Distinct from
#: the controller-owned ``agent_proposed_actions`` (which it never writes) and from
#: the ``agent_prompt_lineage`` timeline (the applied *change* record). Rejections
#: write nothing to lineage, so this is the only durable record that they happened.
DECISIONS_TABLE = "agent_action_decisions"

#: Column order — declared once, reused by the DDL and the INSERT (the
#: :mod:`ail.publish` convention) so the two can never drift.
DECISION_COLUMNS: list[str] = [
    "agent_name",
    "proposal_id",
    "decision",
    "outcome",
    "approver",
    "reason",
    "decided_at",
    "action_kind",
    "risk_class",
    "summary",
    "prompt_name",
    "new_version",
    "new_uri",
    "created_view",
    "champion_alias",
    "reverted_to_version",
    "lineage_recorded",
    "refused_reason",
    "recorded_at",
]


# ---------------------------------------------------------------------------
# The service result (typed, JSON-serializable — the app renders it)
# ---------------------------------------------------------------------------


class ApplyServiceOutcome(StrEnum):
    """The outcome the app surfaces to the reviewer."""

    APPLIED = "applied"
    #: Human-authorized, but intentionally not applied by hosted compute. The local
    #: companion must fetch and commit the exact reviewed artifact.
    APPROVED = "approved"
    REJECTED = "rejected"
    #: A fail-closed refusal — nothing was applied (surface :attr:`refused_reason`).
    REFUSED = "refused"
    #: The change IS live but its lineage record failed — reconcile the audit.
    APPLIED_UNRECORDED = "applied_unrecorded"
    #: An infrastructure error before/around the apply — never a fake success.
    ERROR = "error"


class ApplyServiceResult(BaseModel):
    """The flat, JSON-round-trippable outcome the app renders and the CLI prints."""

    model_config = ConfigDict(extra="forbid")

    outcome: ApplyServiceOutcome
    proposal_id: str
    agent_name: str
    decision: DecisionKind
    approver: str
    decided_at: str
    action_kind: str | None = None
    risk_class: str | None = None
    #: The proposal's effective status after this decision (applied/rejected/pending).
    status: str = ProposalStatus.PENDING.value
    reason: str | None = None
    # what was applied (populated only when outcome is APPLIED / APPLIED_UNRECORDED)
    summary: str = ""
    prompt_name: str | None = None
    new_version: int | None = None
    new_uri: str | None = None
    created_view: str | None = None
    champion_alias: str | None = None
    reverted_to_version: int | None = None
    lineage_recorded: bool = False
    # bookkeeping the app / an operator uses to reconcile
    decision_recorded: bool = False
    refused_reason: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Injectable persistence writers (faked in tests → no live UC write)
# ---------------------------------------------------------------------------


class DecisionWriter(Protocol):
    """Persists one decision to the append-only ``agent_action_decisions`` audit."""

    def __call__(self, result: ApplyServiceResult) -> None: ...


class StatusWriter(Protocol):
    """Advances a proposal's ``status`` in ``agent_proposed_actions`` (in place)."""

    def __call__(self, *, agent_name: str, proposal_id: str, status: ProposalStatus) -> None: ...


# ---------------------------------------------------------------------------
# The orchestration — engine call + decision record + status advance (fail-closed)
# ---------------------------------------------------------------------------


def decide_on_proposal(
    proposal: ProposedAction,
    decision: ApprovalDecision,
    *,
    registry_client: ApplyRegistryClient,
    warehouse_executor: WarehouseExecutor,
    lineage_recorder: LineageRecorder,
    gate_recheck: GateRecheck,
    body_resolver: BodyResolver | None,
    decision_writer: DecisionWriter,
    status_writer: StatusWriter,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> ApplyServiceResult:
    """Apply (or reject) a decided proposal, then record the decision — fail-closed.

    Calls the lane-3a engine (:func:`ail.loop.apply.apply_approved_proposal`) — the
    engine re-verifies the proof and re-runs the gate; this function never bypasses
    it. Then it persists the human decision to the append-only audit and advances
    the proposal's status so the queue drops it. Every branch returns a typed
    :class:`ApplyServiceResult`; nothing here raises for a *decision-level* outcome
    (a refusal is a result, not an exception) so the caller/CLI always has something
    to surface.

    * **APPROVE, applied** → status ``applied``, decision recorded, artifact ids set.
    * **REJECT** → status ``rejected``, decision (with reason) recorded, no capability.
    * **REFUSED** (:class:`~ail.loop.apply.ApplyRefused`) → nothing applied, status
      stays ``pending`` (the proposal still needs attention / may re-surface); the
      attempted decision is recorded with its refusal reason.
    * **APPLIED_UNRECORDED** (:class:`~ail.loop.apply.ApplyRecordError`) → the change
      is LIVE; status is advanced to ``applied`` and the decision recorded, but the
      lineage timeline record failed — surfaced for reconciliation.
    """
    # GEPA local rewrites have a hard trust boundary: hosted compute records the
    # approval but never resolves or writes a laptop path. The approved row is then
    # consumed by the local companion. A legacy GEPA proposal without the immutable
    # local-apply spec is refused rather than falling through to prompt-registry apply.
    if proposal.action_kind is ActionKind.GEPA_PROMPT and decision.decision is DecisionKind.APPROVE:
        try:
            approved = _approve_gepa_for_local_companion(
                proposal, decision, gate_recheck=gate_recheck
            )
        except ApplyRefused as exc:
            refused = _refused_result(proposal, decision, reason=exc.reason)
            _persist(
                refused,
                proposal=proposal,
                decision_writer=decision_writer,
                status_writer=None,
            )
            return refused
        _persist(
            approved,
            proposal=proposal,
            decision_writer=decision_writer,
            status_writer=status_writer,
        )
        return approved

    try:
        result = apply_approved_proposal(
            proposal,
            decision,
            registry_client=registry_client,
            warehouse_executor=warehouse_executor,
            lineage_recorder=lineage_recorder,
            gate_recheck=gate_recheck,
            body_resolver=body_resolver,
            prompt_name=prompt_name,
            catalog=catalog,
            schema=schema,
        )
    except ApplyRefused as exc:
        refused = _refused_result(proposal, decision, reason=exc.reason)
        _persist(refused, proposal=proposal, decision_writer=decision_writer, status_writer=None)
        return refused
    except ApplyRecordError as exc:
        # The capability applied (the change is LIVE) but the engine's lineage record
        # failed. Advance status to applied and record the decision, then surface
        # applied-but-unrecorded so the operator reconciles the lineage entry.
        unrecorded = _applied_result(
            exc.result,
            outcome=ApplyServiceOutcome.APPLIED_UNRECORDED,
            error=str(exc),
        )
        _persist(
            unrecorded,
            proposal=proposal,
            decision_writer=decision_writer,
            status_writer=status_writer,
        )
        return unrecorded

    service = _service_result_from_apply(result)
    _persist(
        service, proposal=proposal, decision_writer=decision_writer, status_writer=status_writer
    )
    return service


def _persist(
    service: ApplyServiceResult,
    *,
    proposal: ProposedAction,
    decision_writer: DecisionWriter,
    status_writer: StatusWriter | None,
) -> None:
    """Advance the proposal status and append the decision audit row.

    The live change (if any) already happened inside the engine, so a persistence
    failure must never masquerade as *not applied* — it is folded into the result
    rather than raised. **Audit-integrity invariant (fail-loud):** when the engine
    outcome was a LIVE apply (``APPLIED``/``APPLIED_UNRECORDED``) and *either* the
    status advancement *or* the decision-audit append then fails, the surfaced
    outcome is downgraded to :attr:`~ApplyServiceOutcome.APPLIED_UNRECORDED` — the
    change is live but its record did not fully land, so the caller/UI treats it as
    needs-reconcile, never a clean ``applied``. A genuine ``REJECTED``/``REFUSED``
    outcome (no live change) is left as-is; only its ``error`` is annotated.

    The two writes are attempted independently so a live apply plus *any* failed
    persistence deterministically returns ``APPLIED_UNRECORDED``.
    """
    if service.outcome is ApplyServiceOutcome.APPROVED:
        # Approval is an authorization token for a later local mutation. Record the
        # authenticated decision FIRST; only then expose status=approved to the
        # companion. If either write fails, no hosted/local apply is reported here.
        try:
            decision_writer(service)
            service.decision_recorded = True
        except Exception as exc:  # noqa: BLE001 - fail closed before status becomes approved
            service.decision_recorded = False
            service.outcome = ApplyServiceOutcome.ERROR
            service.status = ProposalStatus.PENDING.value
            service.error = f"decision audit not recorded ({type(exc).__name__}: {exc})"
            return
        try:
            if status_writer is None:
                raise RuntimeError("no status writer configured")
            status_writer(
                agent_name=proposal.agent_name,
                proposal_id=proposal.proposal_id,
                status=ProposalStatus.APPROVED,
            )
        except Exception as exc:  # noqa: BLE001 - decision exists, but companion must not see approval
            service.outcome = ApplyServiceOutcome.ERROR
            service.status = ProposalStatus.PENDING.value
            service.error = f"approval status not advanced ({type(exc).__name__}: {exc})"
        return

    live_apply = service.outcome in (
        ApplyServiceOutcome.APPLIED,
        ApplyServiceOutcome.APPLIED_UNRECORDED,
    )
    errors: list[str] = []

    target_status = _status_for_outcome(service.outcome)
    if status_writer is not None and target_status is not None:
        try:
            status_writer(
                agent_name=proposal.agent_name,
                proposal_id=proposal.proposal_id,
                status=target_status,
            )
        except Exception as exc:  # noqa: BLE001 - a status-write failure must not fake a rollback
            errors.append(f"status not advanced ({type(exc).__name__}: {exc})")

    if errors and live_apply:
        service.outcome = ApplyServiceOutcome.APPLIED_UNRECORDED

    try:
        decision_writer(service)
        service.decision_recorded = True
    except Exception as exc:  # noqa: BLE001 - an audit-write failure must not fake a rollback
        service.decision_recorded = False
        errors.append(f"decision audit not recorded ({type(exc).__name__}: {exc})")

    if errors:
        prior = f"{service.error}; " if service.error else ""
        service.error = prior + "; ".join(errors)
        # A LIVE apply whose status/audit record did not fully land is applied-but-
        # unrecorded (reconcile) — never a clean success. Do not touch REJECTED/REFUSED.
        if live_apply:
            service.outcome = ApplyServiceOutcome.APPLIED_UNRECORDED
        elif service.outcome is ApplyServiceOutcome.APPROVED:
            # No local file changed, so a failed approval/status audit is a plain
            # error and must never be shown as "waiting for companion".
            service.outcome = ApplyServiceOutcome.ERROR


def _status_for_outcome(outcome: ApplyServiceOutcome) -> ProposalStatus | None:
    if outcome in (ApplyServiceOutcome.APPLIED, ApplyServiceOutcome.APPLIED_UNRECORDED):
        return ProposalStatus.APPLIED
    if outcome is ApplyServiceOutcome.REJECTED:
        return ProposalStatus.REJECTED
    if outcome is ApplyServiceOutcome.APPROVED:
        return ProposalStatus.APPROVED
    return None  # REFUSED / ERROR leave the proposal pending


def _service_result_from_apply(result: ApplyResult) -> ApplyServiceResult:
    if result.outcome.value == "rejected":
        return ApplyServiceResult(
            outcome=ApplyServiceOutcome.REJECTED,
            proposal_id=result.proposal_id,
            agent_name=result.agent_name,
            decision=DecisionKind.REJECT,
            approver=result.approver,
            decided_at=result.decided_at,
            action_kind=result.action_kind.value,
            risk_class=result.proposal.risk_class.value,
            status=ProposalStatus.REJECTED.value,
            reason=result.reason,
        )
    return _applied_result(result, outcome=ApplyServiceOutcome.APPLIED)


def _applied_result(
    result: ApplyResult,
    *,
    outcome: ApplyServiceOutcome,
    error: str | None = None,
) -> ApplyServiceResult:
    return ApplyServiceResult(
        outcome=outcome,
        proposal_id=result.proposal_id,
        agent_name=result.agent_name,
        decision=DecisionKind.APPROVE,
        approver=result.approver,
        decided_at=result.decided_at,
        action_kind=result.action_kind.value,
        risk_class=result.proposal.risk_class.value,
        status=ProposalStatus.APPLIED.value,
        reason=result.reason,
        summary=result.summary,
        prompt_name=result.prompt_name,
        new_version=result.new_version,
        new_uri=result.new_uri,
        created_view=result.created_view,
        champion_alias=result.champion_alias,
        reverted_to_version=result.reverted_to_version,
        lineage_recorded=result.lineage_recorded,
        error=error,
    )


def _refused_result(
    proposal: ProposedAction, decision: ApprovalDecision, *, reason: str
) -> ApplyServiceResult:
    return ApplyServiceResult(
        outcome=ApplyServiceOutcome.REFUSED,
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        decision=decision.decision,
        approver=decision.approver,
        decided_at=decision.decided_at,
        action_kind=proposal.action_kind.value,
        risk_class=proposal.risk_class.value,
        status=proposal.status.value,
        reason=decision.reason,
        refused_reason=reason,
    )


def _approve_gepa_for_local_companion(
    proposal: ProposedAction,
    decision: ApprovalDecision,
    *,
    gate_recheck: GateRecheck,
) -> ApplyServiceResult:
    """Authorize an exact GEPA local rewrite without applying it on hosted compute."""
    if decision.proposal_id != proposal.proposal_id:
        raise ApplyRefused(
            f"decision targets {decision.proposal_id!r}, not proposal {proposal.proposal_id!r}"
        )
    if proposal.status is not ProposalStatus.PENDING:
        raise ApplyRefused(
            f"proposal {proposal.proposal_id!r} is {proposal.status.value!r}, not pending"
        )
    spec: LocalApplySpec | None = proposal.change.local_apply_spec
    if spec is None:
        raise ApplyRefused(
            "GEPA proposal has no immutable local_apply_spec; hosted compute cannot write "
            "the user's machine and an unbound target must not be approved (fail-closed)"
        )
    if not proposal.change.diff or not proposal.change.diff.strip():
        raise ApplyRefused("GEPA proposal has no exact reviewed diff (fail-closed)")
    proof = proposal.proof
    if proof is None or not (proof.proved_improvement and proof.correctness_held):
        raise ApplyRefused(
            "GEPA proposal lacks a held-out improvement with correctness held (fail-closed)"
        )
    if not proposal.gate_status.gated:
        raise ApplyRefused("GEPA proposal's recorded gate is not cleared (fail-closed)")
    current_gate = gate_recheck(proposal)
    if not current_gate.ok:
        reasons = " | ".join(current_gate.reasons) or "current gate did not clear"
        raise ApplyRefused(f"apply-time gate re-check failed: {reasons}")
    return ApplyServiceResult(
        outcome=ApplyServiceOutcome.APPROVED,
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        decision=DecisionKind.APPROVE,
        approver=decision.approver,
        decided_at=decision.decided_at,
        action_kind=proposal.action_kind.value,
        risk_class=proposal.risk_class.value,
        status=ProposalStatus.APPROVED.value,
        reason=decision.reason,
        summary=(f"approved local rewrite of {spec.target_path}; waiting for the local companion"),
    )


# ---------------------------------------------------------------------------
# The REAL seams (wired to the framework's existing capabilities)
# ---------------------------------------------------------------------------


class _CompositeRegistryClient:
    """One object satisfying :class:`~ail.loop.apply.ApplyRegistryClient`.

    The engine needs the union of the prompt-registry write API and the
    lineage/alias read API, but no single live client exposes both. This composes
    the two live clients and delegates each method — no registry logic is
    reimplemented here.
    """

    def __init__(self, prompt_client: Any, lineage_client: Any) -> None:
        self._prompt = prompt_client
        self._lineage = lineage_client

    # -- PromptRegistryClient --
    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None,
        tags: dict[str, str] | None,
    ) -> Any:
        return self._prompt.register_prompt(name, template, commit_message, tags)

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        self._prompt.set_prompt_alias(name, alias, version)

    def search_prompts(self, filter_string: str | None) -> list[Any]:
        return list(self._prompt.search_prompts(filter_string))

    def load_prompt(self, name_or_uri: str) -> Any:
        return self._prompt.load_prompt(name_or_uri)

    # -- LineageRegistryClient --
    def search_prompt_versions(self, name: str) -> list[Any]:
        return list(self._lineage.search_prompt_versions(name))

    def get_prompt_version_by_alias(self, name: str, alias: str) -> Any | None:
        return self._lineage.get_prompt_version_by_alias(name, alias)


def build_registry_client(
    profile: str | None = None,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> ApplyRegistryClient:
    """The live registry seam: the prompt-registry client + the lineage client.

    Reuses :func:`ail.optimize.prompt_registry._new_prompt_client` (via
    :func:`~ail.optimize.prompt_registry.register_prompt_body`'s own default when
    ``client`` is ``None`` — but the engine requires an explicit union client, so we
    build both here) and :func:`ail.publish_lineage.new_lineage_client`.
    """
    resolve_catalog_schema(catalog, schema)

    from ail.optimize.prompt_registry import _new_prompt_client

    return _CompositeRegistryClient(_new_prompt_client(profile), new_lineage_client(profile))


def build_warehouse_executor(client: Any, warehouse_id: str) -> WarehouseExecutor:
    """The live warehouse seam: run one statement on the framework SQL warehouse."""

    def _run(sql: str) -> None:
        _execute(client, warehouse_id, sql)

    return _run


def build_lineage_recorder(
    *,
    agent: Any,
    prompt_name: str,
    registry_client: ApplyRegistryClient,
    warehouse_client: Any,
    warehouse_id: str,
    catalog: str | None = None,
    schema: str | None = None,
) -> LineageRecorder:
    """The live lineage seam: refresh the agent's ``agent_prompt_lineage`` slice.

    An applied prompt/skill/instruction change (or a revert) has just registered a
    version / moved the champion alias; re-publishing this agent's lineage slice via
    :func:`ail.publish_lineage.publish_agent_lineage` picks that up — the audit
    timeline the app already renders. Reuses the publish, never reimplements it. The
    *who/why* of the decision is recorded separately in ``agent_action_decisions``.
    """

    resolved_catalog, resolved_schema = resolve_catalog_schema(catalog, schema)

    def _record(record: Any) -> None:
        publish_agent_lineage(
            agent,
            prompt_name=prompt_name,
            registry_client=registry_client,
            warehouse_client=warehouse_client,
            warehouse_id=warehouse_id,
            catalog=resolved_catalog,
            schema=resolved_schema,
        )

    return _record


def build_gate_recheck(
    *,
    experiment_id: str,
    cohort: Any,
    profile: str | None = None,
    warehouse_id: str | None = None,
) -> GateRecheck:
    """The live gate seam: re-run readiness + judge-trust at apply time.

    Reuses the readiness driver (:func:`ail.jobs.readiness_preflight.gather_facts`)
    and the controller's verdict (:func:`ail.loop.controller.evaluate_gate`) — the
    same two fail-closed dimensions the controller gated on originally
    (``docs/LOOP_CONTROLLER.md`` §"Gate status"): the readiness wall must still be
    cleared and the trigger's certifying judge (if any) must still be trusted. This
    is the wall that stops a proposal proven hours ago from shipping after the world
    changed. Never reimplements a gate.
    """

    def _recheck(proposal: ProposedAction) -> GateRecheckResult:
        from ail.jobs.readiness_preflight import build_goal, gather_facts

        try:
            facts = gather_facts(experiment_id, cohort, profile=profile, warehouse_id=warehouse_id)
        except Exception as exc:  # noqa: BLE001 - a facts failure fails the gate closed
            return GateRecheckResult(ok=False, reasons=[f"could not gather readiness facts: {exc}"])
        goal = build_goal(proposal.objective_metric)
        from ail.readiness import compute_readiness

        readiness = compute_readiness(cohort, goal, facts)
        gated, reasons = evaluate_gate(readiness, judge_name=proposal.trigger.judge_name)
        return GateRecheckResult(ok=gated, reasons=reasons)

    return _recheck


def build_body_resolver(
    *,
    registry_client: ApplyRegistryClient,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    catalog: str | None = None,
    schema: str | None = None,
    champion_alias: str = CHAMPION_ALIAS,
) -> BodyResolver:
    """The live body seam for a skill/instruction update (its change is a *diff*).

    The reviewed full body is reconstructed **server-side** from the authoritative
    current champion body (loaded from the registry) + the proposal's stored unified
    diff — never a client-supplied body. Fail-closed: if the diff does not apply
    cleanly to the current champion (e.g. the champion moved since the proposal was
    minted), :func:`_apply_unified_diff` raises and the engine refuses — a stale diff
    is never force-applied.
    """
    resolved_catalog, resolved_schema = resolve_catalog_schema(catalog, schema)
    full_name = resolve_prompt_name(prompt_name, catalog=resolved_catalog, schema=resolved_schema)

    def _resolve(proposal: ProposedAction) -> RegisterableBody:
        diff = proposal.change.diff
        if not diff or not diff.strip():
            raise ValueError(
                f"proposal {proposal.proposal_id!r} carries no diff to resolve into a body"
            )
        uri = f"prompts:/{full_name}@{champion_alias}"
        loaded = registry_client.load_prompt(uri)
        current = getattr(loaded, "template", None) or getattr(loaded, "text", None)
        if not current or not str(current).strip():
            raise ValueError(
                f"current champion {uri!r} has no body to apply the reviewed diff onto"
            )
        current_str = str(current)
        new_body = _apply_unified_diff(current_str, diff)
        # Fail-closed on a no-op / diff-equal patch: a body identical to the current
        # champion is not a change and must NOT be registered as a new version (it
        # would create a fake "change" in the lineage). Compared against the applier's
        # own splitlines-normalized source (same normalization the patched body uses).
        if new_body == "\n".join(current_str.splitlines()):
            raise ValueError(
                f"diff for proposal {proposal.proposal_id!r} is a no-op — the patched body is "
                "identical to the current champion; refusing to register a fake change "
                "(fail-closed)"
            )
        provenance = PromptProvenance(
            source=PromptSource.SEED,
            changed=True,
            registration_reason=(
                f"human-approved {proposal.action_kind.value} (proposal {proposal.proposal_id})"
            ),
        )
        return RegisterableBody(
            body=new_body,
            provenance=provenance,
            commit_message=f"Apply approved {proposal.action_kind.value}",
        )

    return _resolve


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def _apply_unified_diff(original: str, diff: str) -> str:
    """Apply a unified ``diff`` to ``original``; raise on any context mismatch.

    A deliberately strict applier: every context (` `) and removed (`-`) line must
    match the source exactly, hunks must be in order, and at least one hunk must be
    present. Any deviation raises :class:`ValueError` — a diff that does not apply
    cleanly is refused, never approximated (fail-closed). Returns the patched text.
    """
    src = original.splitlines()
    lines = diff.splitlines()
    out: list[str] = []
    src_idx = 0
    i = 0
    saw_hunk = False
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("--- ") or line.startswith("+++ "):
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if m is None:
            raise ValueError(f"unexpected line outside a hunk in diff: {line!r}")
        saw_hunk = True
        hunk_start = int(m.group(1)) - 1  # unified diff line numbers are 1-based
        if hunk_start < src_idx:
            raise ValueError("overlapping or out-of-order hunk in diff")
        if hunk_start > len(src):
            raise ValueError("hunk starts past the end of the source")
        out.extend(src[src_idx:hunk_start])
        src_idx = hunk_start
        i += 1
        while i < n and not lines[i].startswith("@@"):
            h = lines[i]
            if h.startswith("--- ") or h.startswith("+++ "):
                break
            if h.startswith("\\"):  # "\ No newline at end of file"
                i += 1
                continue
            tag, body = (h[0], h[1:]) if h else (" ", "")
            if tag == " ":
                if src_idx >= len(src) or src[src_idx] != body:
                    raise ValueError("context line does not match the source (stale diff)")
                out.append(src[src_idx])
                src_idx += 1
            elif tag == "-":
                if src_idx >= len(src) or src[src_idx] != body:
                    raise ValueError("removed line does not match the source (stale diff)")
                src_idx += 1
            elif tag == "+":
                out.append(body)
            else:
                raise ValueError(f"unexpected diff line: {h!r}")
            i += 1
    if not saw_hunk:
        raise ValueError("no hunks found — not an applicable unified diff")
    out.extend(src[src_idx:])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Persistence: load the proposal, record the decision, advance the status
# ---------------------------------------------------------------------------


def _query_rows(client: Any, warehouse_id: str, statement: str) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as ``{column: value}`` dicts (values as strings).

    Mirrors :func:`ail.publish._execute`'s wait loop but reads the result set. All
    values come back as strings (or ``None``); callers cast as needed.
    """
    from databricks.sdk.service.sql import StatementState

    resp = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
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

    manifest = resp.manifest
    columns = [c.name for c in manifest.schema.columns] if manifest and manifest.schema else []
    data = resp.result.data_array if resp.result and resp.result.data_array else []
    return [dict(zip(columns, row, strict=False)) for row in data]


def load_pending_proposal(
    *,
    client: Any,
    warehouse_id: str,
    agent_name: str,
    proposal_id: str,
    experiment_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> ProposedAction | None:
    """Load the authoritative still-``pending`` proposal from ``agent_proposed_actions``.

    Returns ``None`` when no *pending* row matches (already decided, superseded, or
    unknown) — the caller refuses fail-closed. Never trusts a client-supplied body.
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    sql = (
        f"SELECT * FROM {fqn} WHERE agent_name = {_lit(agent_name)} "
        f"AND experiment_id = {_lit(experiment_id)} "
        f"AND proposal_id = {_lit(proposal_id)} "
        f"AND status = {_lit(ProposalStatus.PENDING.value)} "
        "LIMIT 1"
    )
    rows = _query_rows(client, warehouse_id, sql)
    if not rows:
        return None
    return _row_to_proposal(rows[0])


def _s(row: dict[str, Any], key: str) -> str | None:
    v = row.get(key)
    return None if v is None else str(v)


def _i(row: dict[str, Any], key: str) -> int | None:
    v = row.get(key)
    return None if v is None or str(v) == "" else int(float(str(v)))


def _f(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    return None if v is None or str(v) == "" else float(str(v))


def _b(row: dict[str, Any], key: str) -> bool:
    v = row.get(key)
    return str(v).strip().lower() == "true" if v is not None else False


#: The ten ``proof_*`` columns :func:`ail.loop.publish_proposals._proposal_row` writes.
#: They are written atomically — all populated for a *prove-before-propose* proposal, or
#: all ``NULL`` for an **evidence-first** proposal (``proof=None``).
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


def _has_proof(row: dict[str, Any]) -> bool:
    """True iff the row carries a frozen-suite proof (any ``proof_*`` column populated).

    An evidence-first proposal (:func:`ail.loop.evidence_cycle.run_evidence_cycle`)
    stores ``proof=None`` as all-``NULL`` proof_* columns, so a row with every proof_*
    column NULL/empty reconstructs as ``proof=None`` — never a fabricated zero-value
    :class:`~ail.loop.proposals.ProofSummary` that would misread as an unproven proof
    and be refused at apply time (lane L7a). A real proof always populates the boolean
    and integer columns, so it is always detected.
    """
    return any(row.get(k) not in (None, "") for k in _PROOF_COLUMNS)


def _json_list(row: dict[str, Any], key: str) -> list[str]:
    v = row.get(key)
    if not v:
        return []
    try:
        parsed = json.loads(str(v))
    except (ValueError, TypeError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def _row_to_proposal(row: dict[str, Any]) -> ProposedAction:
    """Reconstruct a :class:`ProposedAction` from a flat ``agent_proposed_actions`` row.

    The inverse of :func:`ail.loop.publish_proposals._proposal_row`; all SQL values
    arrive as strings, so scalars are cast back to their typed form here.
    """
    trigger = TriggerSignal(
        kind=TriggerKind(str(row["trigger_kind"])),
        summary=str(row.get("trigger_summary") or ""),
        metric=_s(row, "trigger_metric"),
        observed_value=_f(row, "trigger_observed_value"),
        threshold=_f(row, "trigger_threshold"),
        n_traces=_i(row, "trigger_n_traces") or 0,
        judge_name=_s(row, "trigger_judge_name"),
        asset_type=_s(row, "trigger_asset_type"),
        source_rank=_i(row, "trigger_source_rank"),
        trace_refs=_json_list(row, "trigger_trace_refs"),
    )
    local_spec_raw = row.get("change_local_apply_spec_json")
    local_spec = (
        LocalApplySpec.model_validate_json(str(local_spec_raw))
        if local_spec_raw not in (None, "")
        else None
    )
    change = ProposedChange(
        kind=ChangeKind(str(row["change_kind"])),
        summary=str(row.get("change_summary") or ""),
        sql=_s(row, "change_sql"),
        diff=_s(row, "change_diff"),
        evolved_body_ref=_s(row, "change_evolved_body_ref"),
        revert_target=_s(row, "change_revert_target"),
        # AGENT_TASK payload (nullable): the NL plan is required for an AGENT_TASK_PLAN
        # change; preview_diff / produced_change_ref stay None until the executor (L7b-2)
        # fills them. All None for a non-AGENT_TASK proposal — round-trips losslessly.
        plan=_s(row, "change_plan"),
        preview_diff=_s(row, "change_preview_diff"),
        produced_change_ref=_s(row, "change_produced_change_ref"),
        local_apply_spec=local_spec,
    )
    # Reconstruct proof=None for an evidence-first proposal (all proof_* columns NULL),
    # so it round-trips as evidence-only and the apply engine applies it on evidence +
    # gate alone (lane L7a) rather than refusing it as an unproven zero-value proof.
    proof = (
        ProofSummary(
            objective_metric=str(row.get("proof_objective_metric") or ""),
            proved_improvement=_b(row, "proof_proved_improvement"),
            correctness_held=_b(row, "proof_correctness_held"),
            realized_savings_absolute=_f(row, "proof_realized_savings_absolute") or 0.0,
            realized_savings_pct=_f(row, "proof_realized_savings_pct"),
            n_promote=_i(row, "proof_n_promote") or 0,
            n_block=_i(row, "proof_n_block") or 0,
            n_errored=_i(row, "proof_n_errored") or 0,
            suite_content_hash=str(row.get("proof_suite_content_hash") or ""),
            suite_version=str(row.get("proof_suite_version") or ""),
        )
        if _has_proof(row)
        else None
    )
    gate = GateStatus(
        readiness_tier=str(row.get("gate_readiness_tier") or ""),
        can_prove_improvement=_b(row, "gate_can_prove_improvement"),
        judge_agreement=_f(row, "gate_judge_agreement"),
        scored_coverage=_f(row, "gate_scored_coverage") or 0.0,
        n_distrusted_judges=_i(row, "gate_n_distrusted_judges") or 0,
        gated=_b(row, "gate_gated"),
        reasons=_json_list(row, "gate_reasons"),
    )
    return ProposedAction(
        proposal_id=str(row["proposal_id"]),
        agent_name=str(row["agent_name"]),
        experiment_id=str(row.get("experiment_id") or ""),
        action_kind=ActionKind(str(row["action_kind"])),
        risk_class=RiskClass(str(row["risk_class"])),
        status=ProposalStatus(str(row.get("status") or ProposalStatus.PENDING.value)),
        objective_metric=str(row.get("objective_metric") or ""),
        goal_cohort=str(row.get("goal_cohort") or ""),
        trigger=trigger,
        change=change,
        proof=proof,
        gate_status=gate,
        created_at=_s(row, "created_at"),
    )


def _decisions_ddl(catalog: str, schema: str) -> list[str]:
    fqn = f"`{catalog}`.`{schema}`"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {fqn} "
        "COMMENT 'Agent self-optimization loop: L0 deterministic metrics (Tier A).'",
        f"""CREATE TABLE IF NOT EXISTS {fqn}.{DECISIONS_TABLE} (
            agent_name STRING,
            proposal_id STRING,
            decision STRING,
            outcome STRING,
            approver STRING,
            reason STRING,
            decided_at STRING,
            action_kind STRING,
            risk_class STRING,
            summary STRING,
            prompt_name STRING,
            new_version INT,
            new_uri STRING,
            created_view STRING,
            champion_alias STRING,
            reverted_to_version INT,
            lineage_recorded BOOLEAN,
            refused_reason STRING,
            recorded_at STRING
        ) USING DELTA
        COMMENT 'Append-only audit of human approve/reject decisions (lane 3b write-path).'""",
    ]


def record_decision(
    result: ApplyServiceResult,
    *,
    client: Any,
    warehouse_id: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    recorded_at: str | None = None,
) -> None:
    """Append one decision to ``agent_action_decisions`` (create the table if needed)."""
    for ddl in _decisions_ddl(catalog, schema):
        _execute(client, warehouse_id, ddl)
    stamp = recorded_at or datetime.now(UTC).isoformat()
    values = [
        result.agent_name,
        result.proposal_id,
        result.decision.value,
        result.outcome.value,
        result.approver,
        result.reason,
        result.decided_at,
        result.action_kind,
        result.risk_class,
        result.summary,
        result.prompt_name,
        result.new_version,
        result.new_uri,
        result.created_view,
        result.champion_alias,
        result.reverted_to_version,
        result.lineage_recorded,
        result.refused_reason,
        stamp,
    ]
    fqn = f"`{catalog}`.`{schema}`.{DECISIONS_TABLE}"
    cols = ", ".join(DECISION_COLUMNS)
    literals = ", ".join(_lit(v) for v in values)
    _execute(client, warehouse_id, f"INSERT INTO {fqn} ({cols}) VALUES ({literals})")


def mark_proposal_status(
    *,
    client: Any,
    warehouse_id: str,
    agent_name: str,
    experiment_id: str,
    proposal_id: str,
    status: ProposalStatus,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    """Advance a proposal's ``status`` in ``agent_proposed_actions`` (only if pending).

    An in-place ``UPDATE`` scoped to the single
    ``(agent_name, experiment_id, proposal_id)`` row so the queue's SELECT (which
    orders pending first) drops the decided proposal on refresh. The controller
    *owns* this table via an atomic experiment-scoped REPLACE, so a subsequent
    controller cycle re-materializes the pending set — the durable decision record
    lives in :data:`DECISIONS_TABLE` (and, for an apply, the lineage timeline), never
    solely in this ephemeral status.
    """
    fqn = f"`{catalog}`.`{schema}`.{PROPOSALS_TABLE}"
    local_state = (
        f", local_apply_status = {_lit('waiting_for_companion')}, local_apply_error = NULL"
        if status is ProposalStatus.APPROVED
        else ""
    )
    _execute(
        client,
        warehouse_id,
        f"UPDATE {fqn} SET status = {_lit(status.value)}{local_state} "
        f"WHERE agent_name = {_lit(agent_name)} AND experiment_id = {_lit(experiment_id)} "
        f"AND proposal_id = {_lit(proposal_id)} "
        f"AND status = {_lit(ProposalStatus.PENDING.value)}",
    )


# ---------------------------------------------------------------------------
# Live composition: load → wire seams → decide → record
# ---------------------------------------------------------------------------


def run_decision(
    *,
    proposal_id: str,
    agent_name: str,
    decision: str,
    approver: str,
    reason: str | None,
    decided_at: str,
    profile: str | None = None,
    warehouse_id: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    registry_path: str | None = None,
) -> ApplyServiceResult:
    """Load the proposal, wire the live seams, and decide — the CLI's body.

    Fail-closed at every step: an empty approver, an unresolvable warehouse, a
    missing/non-pending proposal, or an unknown agent all yield a
    ``REFUSED``/``ERROR`` result (never a fake success). This is a thin composition
    over :func:`decide_on_proposal`; all the branching logic is tested there against
    fakes.
    """
    try:
        decided = DecisionKind(decision)
    except ValueError:
        return _error_result(
            proposal_id,
            agent_name,
            DecisionKind.APPROVE,
            approver,
            decided_at,
            f"unknown decision {decision!r} — must be 'approve' or 'reject'",
            outcome=ApplyServiceOutcome.REFUSED,
        )
    wh = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not approver.strip():
        return _error_result(
            proposal_id,
            agent_name,
            decided,
            approver,
            decided_at,
            "refusing an anonymous decision — no authenticated approver identity",
            outcome=ApplyServiceOutcome.REFUSED,
        )
    if not wh:
        return _error_result(
            proposal_id,
            agent_name,
            decided,
            approver,
            decided_at,
            "no SQL warehouse id (set DATABRICKS_WAREHOUSE_ID or pass --warehouse-id)",
        )

    try:
        # Validate the decision shape early (approver/reason/decided_at) — a reject
        # without a reason is refused at construction (fail-closed).
        approval = ApprovalDecision(
            proposal_id=proposal_id,
            decision=decided,
            approver=approver,
            reason=reason,
            decided_at=decided_at,
        )
    except ValueError as exc:
        return _error_result(
            proposal_id,
            agent_name,
            decided,
            approver,
            decided_at,
            f"invalid decision: {exc}",
            outcome=ApplyServiceOutcome.REFUSED,
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
            return _error_result(
                proposal_id,
                agent_name,
                decided,
                approver,
                decided_at,
                "no pending proposal with that id for this agent — already decided, "
                "superseded, or unknown (fail-closed)",
                outcome=ApplyServiceOutcome.REFUSED,
            )

        registry_client = build_registry_client(
            profile, catalog=resolved_catalog, schema=resolved_schema
        )
        lineage_recorder = build_lineage_recorder(
            agent=agent,
            prompt_name=prompt_name,
            registry_client=registry_client,
            warehouse_client=client,
            warehouse_id=wh,
            catalog=resolved_catalog,
            schema=resolved_schema,
        )
        gate_recheck = build_gate_recheck(
            experiment_id=agent.experiment_id,
            cohort=agent.cohort(),
            profile=profile,
            warehouse_id=wh,
        )

        def _decision_writer(result: ApplyServiceResult) -> None:
            record_decision(
                result,
                client=client,
                warehouse_id=wh,
                catalog=resolved_catalog,
                schema=resolved_schema,
            )

        def _status_writer(*, agent_name: str, proposal_id: str, status: ProposalStatus) -> None:
            mark_proposal_status(
                client=client,
                warehouse_id=wh,
                agent_name=agent_name,
                experiment_id=agent.experiment_id,
                proposal_id=proposal_id,
                status=status,
                catalog=resolved_catalog,
                schema=resolved_schema,
            )

        return decide_on_proposal(
            proposal,
            approval,
            registry_client=registry_client,
            warehouse_executor=build_warehouse_executor(client, wh),
            lineage_recorder=lineage_recorder,
            gate_recheck=gate_recheck,
            body_resolver=build_body_resolver(
                registry_client=registry_client,
                prompt_name=prompt_name,
                catalog=resolved_catalog,
                schema=resolved_schema,
            ),
            decision_writer=_decision_writer,
            status_writer=_status_writer,
            prompt_name=prompt_name,
            catalog=resolved_catalog,
            schema=resolved_schema,
        )
    except Exception as exc:  # noqa: BLE001 - any infra failure is an honest ERROR, never a fake apply
        return _error_result(
            proposal_id,
            agent_name,
            decided,
            approver,
            decided_at,
            f"{type(exc).__name__}: {exc}",
        )


def _resolve_agent(
    agent_name: str,
    registry_path: str | None,
    *,
    client: Any,
    warehouse_id: str,
    catalog: str,
    schema: str,
) -> Any:
    """Resolve the live UC registry by default, with an explicit local-file override."""
    from ail.jobs.multi_agent import resolve_registered_agent
    from ail.registry import load_registry

    path = registry_path or os.environ.get("AIL_AGENT_REGISTRY")
    if path:
        return load_registry(path).get(agent_name)
    return resolve_registered_agent(
        agent_name,
        client=client,
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
    )


def _error_result(
    proposal_id: str,
    agent_name: str,
    decision: DecisionKind,
    approver: str,
    decided_at: str,
    message: str,
    *,
    outcome: ApplyServiceOutcome = ApplyServiceOutcome.ERROR,
) -> ApplyServiceResult:
    return ApplyServiceResult(
        outcome=outcome,
        proposal_id=proposal_id,
        agent_name=agent_name,
        decision=decision,
        approver=approver,
        decided_at=decided_at,
        refused_reason=message if outcome is ApplyServiceOutcome.REFUSED else None,
        error=message if outcome is ApplyServiceOutcome.ERROR else None,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI bridge: read a JSON decision on stdin, print a JSON result on stdout.

    The Node/AppKit route (which authenticates the reviewer) invokes this as a
    subprocess, passing ``{proposal_id, agent_name, decision, approver, reason,
    decided_at}`` — the ``approver`` is the *authenticated* app user, injected by the
    route (never trusted from the browser). Always prints a parseable
    :class:`ApplyServiceResult` and returns ``0`` for a *decision-level* outcome
    (including a fail-closed REFUSED); returns non-zero only when stdin itself is
    unparseable.
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

    result = run_decision(
        proposal_id=str(payload.get("proposal_id", "")),
        agent_name=str(payload.get("agent_name", "")),
        decision=str(payload.get("decision", "")),
        approver=str(payload.get("approver", "")),
        reason=payload.get("reason"),
        decided_at=str(payload.get("decided_at") or datetime.now(UTC).isoformat()),
        profile=payload.get("profile"),
        warehouse_id=payload.get("warehouse_id"),
        catalog=payload.get("catalog"),
        schema=payload.get("schema"),
        prompt_name=str(payload.get("prompt_name") or DEFAULT_PROMPT_NAME),
        registry_path=payload.get("registry_path"),
    )
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
