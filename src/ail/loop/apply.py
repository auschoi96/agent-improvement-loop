"""The apply-on-approval **engine** — lane 3a of ``docs/LOOP_CONTROLLER.md``.

The autonomous controller (:mod:`ail.loop.controller`) *proposes* human-gated
changes and applies nothing (there is no apply seam to call). Lane 3 closes the
loop: a human reviews a proposal's *why + proof + gate status* in the app and
clicks **Approve** / **Reject**. The app's approval button (built in lane 3b)
calls the one function here — :func:`apply_approved_proposal` — which is the *only*
place in the framework that turns an approved proposal into a live change.

Everything is **fail-closed** and driven through **injectable seams**, so the
engine is unit-testable with no live MLflow/warehouse write:

* it **refuses** (never applies) a stale, non-pending, unproven, or ungated
  proposal, or a decision that does not reference the proposal it is applied to;
* on **reject** it records the decision (approver + timestamp + reason) and calls
  **no** capability;
* on **approve** it re-verifies the proof *and* re-runs the gate at apply-time
  (:paramref:`apply_approved_proposal.gate_recheck`) — a proposal whose gate no
  longer holds (judge went distrusted, readiness dropped) is refused, not applied;
* it **reuses** the existing capabilities rather than reimplementing them — the
  metric-view ``CREATE`` DDL the asset generator already produced
  (:meth:`~ail.optimize.assets.asset_contract.MetricViewSpec.to_create_sql`, carried
  verbatim on the proposal so the human approves exactly what ships), the prompt
  registry's :func:`~ail.optimize.prompt_registry.register_prompt_body` /
  :func:`~ail.optimize.prompt_registry.register_gepa_candidate` + champion alias,
  and the guarded revert logic (:func:`ail.jobs.revert_champion.revert_champion`);
* it records every applied change to the **lineage / audit timeline** (through the
  :class:`LineageRecorder` seam) so the trail reads *what changed, why, approved by
  whom*.

Why a ``body_resolver`` seam (beyond the registry/warehouse/lineage/gate seams):
the lane-2 proposal record deliberately stores a **diff** for a skill/instruction
update (``ChangeKind.SKILL_DIFF`` / ``INSTRUCTION_DIFF``), not the full body — "the
body itself lives in the prompt registry / candidate artifact, not inlined here"
(:mod:`ail.loop.proposals`). Registering a *diff* as a prompt body would be a
correctness bug, so the engine refuses to fabricate a body from a diff and instead
resolves the reviewed full body through an injected :class:`BodyResolver` (lane 3b
supplies one; a GEPA-evolved body needs no resolver — its body lives in the
candidate artifact ``register_gepa_candidate`` reads). The resolver is optional so
the four other action kinds apply with the seams above alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from ail.jobs.revert_champion import revert_champion
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProposalStatus,
    ProposedAction,
    RiskClass,
)
from ail.optimize.prompt_registry import (
    CHAMPION_ALIASES,
    DEFAULT_CATALOG,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SCHEMA,
    PromptProvenance,
    PromptRegistryClient,
    RegisteredPrompt,
    register_gepa_candidate,
    register_prompt_body,
)
from ail.publish_lineage import LineageRegistryClient

__all__ = [
    "CHAMPION_ALIAS",
    "DecisionKind",
    "ApprovalDecision",
    "ApplyOutcome",
    "ApplyResult",
    "AppliedChangeRecord",
    "GateRecheckResult",
    "RegisterableBody",
    "ApplyRefused",
    "ApplyRecordError",
    "ApplyRegistryClient",
    "WarehouseExecutor",
    "LineageRecorder",
    "GateRecheck",
    "BodyResolver",
    "apply_approved_proposal",
]

#: The single production alias re-pointed on every applied prompt/skill/instruction
#: change (and on a revert). ``champion`` is canonical; the synonym set lives in
#: :data:`~ail.optimize.prompt_registry.CHAMPION_ALIASES` (single source of truth).
CHAMPION_ALIAS = CHAMPION_ALIASES[0]


# ---------------------------------------------------------------------------
# The human decision (typed; recorded on every approve *and* reject)
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    """Base for the apply-engine models: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class DecisionKind(StrEnum):
    """The two human decisions on a proposal."""

    APPROVE = "approve"
    REJECT = "reject"


class ApprovalDecision(_Model):
    """One human decision on a proposal — the audited input to the apply engine.

    Both an approve and a reject are recorded with the **approver identity** and the
    **decision timestamp** (``docs/LOOP_CONTROLLER.md`` — "both are recorded with the
    approver identity + timestamp; rejections are signal too"). ``approver`` is
    supplied by the caller: lane 3b passes the *authenticated* app user, so the
    engine never fabricates or trusts a client-supplied identity of its own.

    Args:
        proposal_id: The proposal this decision is about; must match the proposal
            passed to :func:`apply_approved_proposal` (fail-closed cross-check).
        decision: Approve or reject.
        approver: The identity of the human who decided (authenticated by the caller).
        reason: Required for a reject (why it was rejected — feedback the controller
            can learn a rule mis-fired from); optional context on an approve.
        decided_at: ISO-8601 timestamp of the decision (supplied by the caller; not
            defaulted, so the recorded time is the real decision time, not import time).
    """

    proposal_id: str
    decision: DecisionKind
    approver: str
    reason: str | None = None
    decided_at: str

    @model_validator(mode="after")
    def _reason_required_for_reject(self) -> ApprovalDecision:
        if self.decision is DecisionKind.REJECT and not (self.reason and self.reason.strip()):
            raise ValueError("a reject decision must carry a non-empty reason (fail-closed)")
        if not self.approver.strip():
            raise ValueError("a decision must carry a non-empty approver identity")
        if not self.decided_at.strip():
            # The audit trail records *when* a change was decided; an empty/whitespace
            # timestamp is not a real decision time — refuse it at construction.
            raise ValueError("a decision must carry a non-empty decided_at timestamp")
        return self


# ---------------------------------------------------------------------------
# The apply outcome (typed result the caller persists / shows) + audit record
# ---------------------------------------------------------------------------


class ApplyOutcome(StrEnum):
    """The outcome of a *decided* proposal (a refusal raises :class:`ApplyRefused`)."""

    APPLIED = "applied"
    REJECTED = "rejected"


class AppliedChangeRecord(_Model):
    """The audit record of one applied change — *what changed, why, approved by whom*.

    Handed to the :class:`LineageRecorder` seam so the applied change lands on the
    lineage / audit timeline. Nothing here is recomputed: *what* comes from the
    applied capability's result, *why* from the proposal's trigger + proof, and
    *who/when* from the human :class:`ApprovalDecision`.
    """

    proposal_id: str
    agent_name: str
    action_kind: ActionKind
    risk_class: RiskClass
    # what changed
    summary: str
    prompt_name: str | None = None
    new_version: int | None = None
    new_uri: str | None = None
    created_view: str | None = None
    champion_alias: str | None = None
    reverted_to_version: int | None = None
    # why
    trigger_summary: str
    objective_metric: str
    proved_improvement: bool
    realized_savings_pct: float | None = None
    # who / when
    approver: str
    decided_at: str
    approval_reason: str | None = None


class ApplyResult(_Model):
    """The typed outcome of :func:`apply_approved_proposal` (what was applied).

    Carries the proposal with its status advanced (:attr:`proposal`) so the caller
    persists the new state, and — on an apply — the concrete artifact identity (the
    new prompt version + uri, the created view name, or the reverted-to version).
    """

    proposal_id: str
    agent_name: str
    action_kind: ActionKind
    outcome: ApplyOutcome
    proposal: ProposedAction
    approver: str
    decided_at: str
    reason: str | None = None
    # what was applied (populated only when outcome is APPLIED)
    summary: str = ""
    prompt_name: str | None = None
    new_version: int | None = None
    new_uri: str | None = None
    created_view: str | None = None
    champion_alias: str | None = None
    reverted_to_version: int | None = None
    lineage_recorded: bool = False


@dataclass(frozen=True, slots=True)
class GateRecheckResult:
    """The verdict of the apply-time gate re-check (readiness + judge trust)."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RegisterableBody:
    """A full, registerable prompt body + its provenance — the resolver's output.

    Bridges a skill/instruction proposal (which carries a *diff*, not a body) to
    :func:`~ail.optimize.prompt_registry.register_prompt_body`. The
    :class:`BodyResolver` (lane 3b) decides the exact body and provenance from the
    reviewed change; the engine registers exactly what the resolver returns.
    """

    body: str
    provenance: PromptProvenance
    commit_message: str | None = None


class ApplyRefused(RuntimeError):
    """A fail-closed refusal: a precondition was unmet, so **nothing was applied**.

    Raised (never a silent no-op) for a stale/non-pending proposal, a mismatched
    decision, an unproven proposal, a failed apply-time gate re-check, or a change
    the engine cannot apply safely. ``reason`` explains the refusal for the audit log
    / the app to surface.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ApplyRecordError(RuntimeError):
    """The change WAS applied (live) but recording it to the lineage failed.

    Cross-system atomicity between the capability apply and the audit write is
    impossible, so the invariant is **fail-loud, never silently inconsistent**: once
    the capability apply *succeeds* the change **is applied**. If the subsequent
    ``lineage_recorder`` then fails, this distinct error is raised — it is neither a
    clean success nor a not-applied refusal. It carries:

    * :attr:`result` — the APPLIED :class:`ApplyResult` (the proposal's status is
      already advanced to ``applied``; :attr:`ApplyResult.lineage_recorded` is
      ``False``), so the caller/UI surfaces *applied-but-unrecorded, reconcile* rather
      than rolling a live change back into a fake not-applied state;
    * :attr:`record` — the :class:`AppliedChangeRecord` that failed to land, so the
      audit entry can be reconciled;
    * :attr:`cause` — the underlying recorder exception.
    """

    def __init__(
        self,
        *,
        result: ApplyResult,
        record: AppliedChangeRecord,
        cause: BaseException,
    ) -> None:
        self.result = result
        self.record = record
        self.cause = cause
        super().__init__(
            f"proposal {result.proposal_id!r} was APPLIED ({result.summary}) but recording it to "
            f"the lineage failed ({type(cause).__name__}: {cause}) — the change is LIVE and the "
            "audit record must be reconciled (applied-but-unrecorded)."
        )


# ---------------------------------------------------------------------------
# Injectable seams (unit-testable: no live MLflow / warehouse write on any path)
# ---------------------------------------------------------------------------


class ApplyRegistryClient(PromptRegistryClient, LineageRegistryClient, Protocol):
    """The registry surface the apply engine needs — the union of two existing seams.

    :class:`~ail.optimize.prompt_registry.PromptRegistryClient` covers the *write*
    (register / set-alias) path used to register a new prompt version;
    :class:`~ail.publish_lineage.LineageRegistryClient` covers the version-listing +
    alias-resolution the guarded revert needs. A single client implementing both
    (structurally) is passed straight through to
    :func:`~ail.optimize.prompt_registry.register_prompt_body` and
    :func:`ail.jobs.revert_champion.revert_champion` — no registry logic is
    reimplemented here.
    """


class WarehouseExecutor(Protocol):
    """Executes one SQL statement against the framework's SQL warehouse.

    The only warehouse write the engine makes: the metric-view ``CREATE OR REPLACE
    VIEW`` DDL. Injected so tests capture the SQL instead of running it; lane 3b
    wires it to the framework service principal's warehouse (``docs/DEPLOY.md``).
    """

    def __call__(self, sql: str) -> None: ...


class LineageRecorder(Protocol):
    """Records one applied change onto the lineage / audit timeline.

    The engine builds the typed :class:`AppliedChangeRecord` (what/why/who) and hands
    it here; lane 3b implements the persist (append the audit entry and refresh the
    ``agent_prompt_lineage`` timeline via :mod:`ail.publish_lineage`). Kept a seam so
    the engine does not reimplement the lineage publish and stays unit-testable.
    """

    def __call__(self, record: AppliedChangeRecord) -> None: ...


class GateRecheck(Protocol):
    """Re-runs the readiness + judge-trust gate for a proposal **at apply time**.

    Wraps :func:`ail.readiness.compute_readiness` + :func:`ail.loop.controller.evaluate_gate`
    for the proposal's goal/cohort (and its certifying judge). The apply-time re-check
    is the wall that stops a proposal proven/gated hours ago from shipping after the
    world changed (a judge lost trust, readiness dropped). Injected so tests drive the
    ``ok=False`` path with no live readiness computation.
    """

    def __call__(self, proposal: ProposedAction) -> GateRecheckResult: ...


class BodyResolver(Protocol):
    """Resolves a skill/instruction proposal's reviewed change into a full body.

    Only invoked for ``SKILL_UPDATE`` / ``INSTRUCTION_UPDATE`` (whose change is a
    diff, not a body). Returns the full :class:`RegisterableBody` to register as a new
    version; the engine refuses to apply such a proposal if no resolver is provided
    (fail-closed — it never registers a diff as a body).
    """

    def __call__(self, proposal: ProposedAction) -> RegisterableBody: ...


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def apply_approved_proposal(
    proposal: ProposedAction,
    decision: ApprovalDecision,
    *,
    registry_client: ApplyRegistryClient,
    warehouse_executor: WarehouseExecutor,
    lineage_recorder: LineageRecorder,
    gate_recheck: GateRecheck,
    body_resolver: BodyResolver | None = None,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> ApplyResult:
    """Apply (or reject) a human-decided proposal — the app's Approve/Reject write-path.

    Fail-closed preconditions (any unmet ⇒ :class:`ApplyRefused`, nothing applied):

    * the ``decision`` must reference *this* ``proposal``;
    * the proposal must still be :attr:`~ail.loop.proposals.ProposalStatus.PENDING`
      (never re-apply an already applied/rejected/superseded proposal);
    * for an **approve** only: the proposal must still carry a proven improvement
      (``proof.proved_improvement and proof.correctness_held``) *and* pass the
      apply-time ``gate_recheck``.

    On **reject**: the decision (approver + timestamp + reason) is recorded on the
    returned result and the proposal is marked rejected; **no** capability is called.

    On **approve** (after preconditions pass) the change is applied by kind:

    * ``METRIC_VIEW`` → execute the proposal's ``CREATE`` DDL (the asset generator's
      :meth:`~ail.optimize.assets.asset_contract.MetricViewSpec.to_create_sql` output,
      carried verbatim on the proposal) via ``warehouse_executor``;
    * ``SKILL_UPDATE`` / ``INSTRUCTION_UPDATE`` → resolve the full body via
      ``body_resolver`` and
      :func:`~ail.optimize.prompt_registry.register_prompt_body`, pointing the
      champion alias at the new version;
    * ``GEPA_PROMPT`` → :func:`~ail.optimize.prompt_registry.register_gepa_candidate`
      from the proposal's ``evolved_body_ref`` artifact (which re-checks the held-out
      improvement — a second fail-closed proof), pointing champion at the new version;
    * ``REVERT`` → re-point the champion alias to the target prior version via the
      guarded :func:`ail.jobs.revert_champion.revert_champion`.

    After a successful apply the change is recorded to the lineage (``lineage_recorder``),
    the proposal is marked applied, and a typed :class:`ApplyResult` (with the new
    version/uri or created view name) is returned.

    Raises:
        ApplyRefused: if any fail-closed precondition is unmet, or the change cannot
            be applied safely (e.g. a skill/instruction update with no ``body_resolver``,
            a GEPA proposal whose candidate artifact is missing, an unparseable revert
            target). The capability's own fail-closed errors (e.g.
            :class:`~ail.optimize.prompt_registry.NonImprovingCandidateError` from the
            apply-time GEPA re-check) propagate unchanged.
    """
    # --- universal preconditions (apply to approve *and* reject) --------------
    if decision.proposal_id != proposal.proposal_id:
        raise ApplyRefused(
            f"decision references proposal {decision.proposal_id!r} but was applied to "
            f"{proposal.proposal_id!r} — refusing (fail-closed)"
        )
    if proposal.status is not ProposalStatus.PENDING:
        raise ApplyRefused(
            f"proposal {proposal.proposal_id!r} is {proposal.status.value!r}, not pending — "
            "refusing to act on an already-decided/superseded proposal (fail-closed)"
        )

    # --- reject: record the decision, call no capability, return --------------
    if decision.decision is DecisionKind.REJECT:
        rejected = proposal.model_copy(update={"status": ProposalStatus.REJECTED})
        return ApplyResult(
            proposal_id=proposal.proposal_id,
            agent_name=proposal.agent_name,
            action_kind=proposal.action_kind,
            outcome=ApplyOutcome.REJECTED,
            proposal=rejected,
            approver=decision.approver,
            decided_at=decision.decided_at,
            reason=decision.reason,
        )

    # --- approve preconditions (re-verify proof, then re-run the gate) --------
    proof = proposal.proof
    if proof is None:
        # An evidence-first proposal (ail.loop.evidence_cycle) carries no frozen-suite
        # proof (proof=None). This prove-requiring apply path refuses it — identical to
        # the unproven case below — until an opt-in Tier-2 verification attaches a
        # measured delta (docs/PRODUCT_ARCHITECTURE.md §3/§7). Fail-closed.
        raise ApplyRefused(
            f"proposal {proposal.proposal_id!r} carries no frozen-suite proof "
            "(evidence-first, proof=None) — refusing to apply without an opt-in Tier-2 "
            "verification (fail-closed); run 'verify on my suite' to attach a measured "
            "delta, then approve"
        )
    if not (proof.proved_improvement and proof.correctness_held):
        raise ApplyRefused(
            f"proposal {proposal.proposal_id!r} no longer carries a proven improvement "
            f"(proved_improvement={proof.proved_improvement}, "
            f"correctness_held={proof.correctness_held}) — refusing to apply an "
            "unproven/stale proposal (fail-closed)"
        )
    recheck = gate_recheck(proposal)
    if not recheck.ok:
        detail = "; ".join(recheck.reasons) if recheck.reasons else "gate no longer holds"
        raise ApplyRefused(
            f"apply-time gate re-check failed for proposal {proposal.proposal_id!r}: {detail} "
            "— refusing to apply (fail-closed)"
        )

    # --- apply by action kind (reusing existing capabilities) -----------------
    # If this raises (a capability failure or a routing/precondition guard) NOTHING
    # was applied — correct fail-closed: the refusal propagates, status unchanged.
    applied = _apply_change(
        proposal,
        registry_client=registry_client,
        warehouse_executor=warehouse_executor,
        body_resolver=body_resolver,
        prompt_name=prompt_name,
        catalog=catalog,
        schema=schema,
    )

    # The capability has now made a LIVE change (the CREATE ran / the champion alias
    # moved). From here the change IS applied: advance the status and build the
    # APPLIED result BEFORE recording, so a lineage-record failure below is reported
    # fail-loud as applied-but-unrecorded (:class:`ApplyRecordError`) — never as a
    # clean success and never rolled back into a fake not-applied refusal.
    applied_proposal = proposal.model_copy(update={"status": ProposalStatus.APPLIED})
    result = ApplyResult(
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        action_kind=proposal.action_kind,
        outcome=ApplyOutcome.APPLIED,
        proposal=applied_proposal,
        approver=decision.approver,
        decided_at=decision.decided_at,
        reason=decision.reason,
        summary=applied.summary,
        prompt_name=applied.prompt_name,
        new_version=applied.new_version,
        new_uri=applied.new_uri,
        created_view=applied.created_view,
        champion_alias=applied.champion_alias,
        reverted_to_version=applied.reverted_to_version,
        lineage_recorded=False,
    )

    # --- record the applied change to the lineage / audit timeline ------------
    record = AppliedChangeRecord(
        proposal_id=proposal.proposal_id,
        agent_name=proposal.agent_name,
        action_kind=proposal.action_kind,
        risk_class=proposal.risk_class,
        summary=applied.summary,
        prompt_name=applied.prompt_name,
        new_version=applied.new_version,
        new_uri=applied.new_uri,
        created_view=applied.created_view,
        champion_alias=applied.champion_alias,
        reverted_to_version=applied.reverted_to_version,
        trigger_summary=proposal.trigger.summary,
        objective_metric=proposal.objective_metric,
        proved_improvement=proof.proved_improvement,
        realized_savings_pct=proof.realized_savings_pct,
        approver=decision.approver,
        decided_at=decision.decided_at,
        approval_reason=decision.reason,
    )
    try:
        lineage_recorder(record)
    except Exception as exc:
        # Fail loud, do not swallow: the change is live but its audit record did not
        # land. Surface applied-but-unrecorded (carrying the APPLIED result) so the
        # caller reconciles the record — not a not-applied refusal.
        raise ApplyRecordError(result=result, record=record, cause=exc) from exc

    return result.model_copy(update={"lineage_recorded": True})


# ---------------------------------------------------------------------------
# Internals: route one approved proposal to its capability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Applied:
    """What a single apply produced — folded into the result + the lineage record."""

    summary: str
    prompt_name: str | None = None
    new_version: int | None = None
    new_uri: str | None = None
    created_view: str | None = None
    champion_alias: str | None = None
    reverted_to_version: int | None = None


#: Defense-in-depth: the change form each action kind must carry.
#: :class:`~ail.loop.proposals.ProposedAction` already enforces this at construction;
#: re-checking it at the routing boundary makes a proposal that was somehow built
#: malformed (e.g. via ``model_construct``, bypassing validation) fail closed here
#: rather than deeper inside a capability against the wrong payload field.
_EXPECTED_CHANGE_KIND: dict[ActionKind, ChangeKind] = {
    ActionKind.METRIC_VIEW: ChangeKind.METRIC_VIEW_SQL,
    ActionKind.SKILL_UPDATE: ChangeKind.SKILL_DIFF,
    ActionKind.INSTRUCTION_UPDATE: ChangeKind.INSTRUCTION_DIFF,
    ActionKind.GEPA_PROMPT: ChangeKind.EVOLVED_BODY_REF,
    ActionKind.REVERT: ChangeKind.REVERT_REF,
}


def _apply_change(
    proposal: ProposedAction,
    *,
    registry_client: ApplyRegistryClient,
    warehouse_executor: WarehouseExecutor,
    body_resolver: BodyResolver | None,
    prompt_name: str,
    catalog: str,
    schema: str,
) -> _Applied:
    kind = proposal.action_kind
    expected = _EXPECTED_CHANGE_KIND.get(kind)
    if expected is not None and proposal.change.kind is not expected:
        raise ApplyRefused(
            f"proposal {proposal.proposal_id!r} action_kind {kind.value!r} requires change kind "
            f"{expected.value!r} but carries {proposal.change.kind.value!r} — refusing a "
            "malformed proposal at the routing boundary (fail-closed)"
        )
    if kind is ActionKind.METRIC_VIEW:
        return _apply_metric_view(proposal, warehouse_executor)
    if kind in (ActionKind.SKILL_UPDATE, ActionKind.INSTRUCTION_UPDATE):
        return _apply_prompt_body(
            proposal,
            registry_client=registry_client,
            body_resolver=body_resolver,
            prompt_name=prompt_name,
            catalog=catalog,
            schema=schema,
        )
    if kind is ActionKind.GEPA_PROMPT:
        return _apply_gepa_prompt(
            proposal,
            registry_client=registry_client,
            prompt_name=prompt_name,
            catalog=catalog,
            schema=schema,
        )
    if kind is ActionKind.REVERT:
        return _apply_revert(
            proposal,
            registry_client=registry_client,
            prompt_name=prompt_name,
            catalog=catalog,
            schema=schema,
        )
    raise ApplyRefused(f"unknown action_kind {kind!r} — refusing to apply (fail-closed)")


_VIEW_NAME_RE = re.compile(r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(?P<name>[^\s(]+)", re.IGNORECASE)


def _view_name(sql: str) -> str | None:
    """Best-effort UC name of the view a ``CREATE ... VIEW`` DDL defines (for audit)."""
    m = _VIEW_NAME_RE.search(sql)
    if m is None:
        return None
    return m.group("name").replace("`", "")


def _apply_metric_view(proposal: ProposedAction, warehouse_executor: WarehouseExecutor) -> _Applied:
    sql = proposal.change.sql
    if not sql or not sql.strip():
        # The proposal model requires a non-empty ``sql`` for a metric_view change,
        # so this is a belt-and-braces fail-closed guard, never the normal path.
        raise ApplyRefused(
            f"metric_view proposal {proposal.proposal_id!r} carries no CREATE SQL — refusing"
        )
    warehouse_executor(sql)
    view = _view_name(sql)
    return _Applied(
        summary=(f"created metric view {view}" if view else "executed metric-view CREATE DDL"),
        created_view=view,
    )


def _apply_prompt_body(
    proposal: ProposedAction,
    *,
    registry_client: ApplyRegistryClient,
    body_resolver: BodyResolver | None,
    prompt_name: str,
    catalog: str,
    schema: str,
) -> _Applied:
    if body_resolver is None:
        raise ApplyRefused(
            f"{proposal.action_kind.value} proposal {proposal.proposal_id!r} needs a "
            "body_resolver to obtain the full registerable body (the proposal carries only a "
            "diff); none provided — refusing to register a diff as a body (fail-closed)"
        )
    resolved = body_resolver(proposal)
    # Defensively guard the resolved body: refuse an empty one, and refuse one that is
    # exactly the proposal's diff — that would be a resolver wiring bug registering a
    # diff as the skill body (the very thing this seam exists to prevent). Fail-closed.
    if not resolved.body or not resolved.body.strip():
        raise ApplyRefused(
            f"body_resolver returned an empty body for proposal {proposal.proposal_id!r} — "
            "refusing to register an empty prompt body (fail-closed)"
        )
    if proposal.change.diff is not None and resolved.body == proposal.change.diff:
        raise ApplyRefused(
            f"body_resolver returned the proposal's diff as the body for "
            f"{proposal.proposal_id!r} — refusing to register a diff as a body (fail-closed)"
        )
    registered = register_prompt_body(
        body=resolved.body,
        provenance=resolved.provenance,
        name=prompt_name,
        catalog=catalog,
        schema=schema,
        commit_message=resolved.commit_message,
        alias=CHAMPION_ALIAS,
        client=registry_client,
    )
    return _registered_applied(proposal.action_kind, registered)


def _apply_gepa_prompt(
    proposal: ProposedAction,
    *,
    registry_client: ApplyRegistryClient,
    prompt_name: str,
    catalog: str,
    schema: str,
) -> _Applied:
    ref = proposal.change.evolved_body_ref
    if not ref or not ref.strip():
        raise ApplyRefused(
            f"gepa_prompt proposal {proposal.proposal_id!r} carries no evolved_body_ref — refusing"
        )
    if not Path(ref).is_file():
        raise ApplyRefused(
            f"gepa_prompt proposal {proposal.proposal_id!r} evolved_body_ref {ref!r} is not a "
            "readable candidate artifact — refusing to apply (fail-closed)"
        )
    # register_gepa_candidate re-runs the held-out improvement check at apply time; a
    # NonImprovingCandidateError propagates as the fail-closed signal it is.
    registered = register_gepa_candidate(
        ref,
        name=prompt_name,
        catalog=catalog,
        schema=schema,
        alias=CHAMPION_ALIAS,
        client=registry_client,
    )
    return _registered_applied(proposal.action_kind, registered)


def _registered_applied(action_kind: ActionKind, registered: RegisteredPrompt) -> _Applied:
    return _Applied(
        summary=(
            f"registered {action_kind.value} as {registered.name} v{registered.version} "
            f"and pointed {CHAMPION_ALIAS!r} at it"
        ),
        prompt_name=registered.name,
        new_version=registered.version,
        new_uri=registered.uri,
        champion_alias=CHAMPION_ALIAS,
    )


_TRAILING_INT_RE = re.compile(r"(\d+)\s*$")


def _parse_target_version(target: str | None) -> int | None:
    """Parse a revert target (``"v1"`` / ``"1"`` / ``"prompts:/c.s.p/3"``) to a version."""
    if not target:
        return None
    stripped = target.strip()
    exact = re.fullmatch(r"[vV]?(\d+)", stripped)
    if exact is not None:
        return int(exact.group(1))
    trailing = _TRAILING_INT_RE.search(stripped)
    if trailing is not None:
        return int(trailing.group(1))
    return None


def _apply_revert(
    proposal: ProposedAction,
    *,
    registry_client: ApplyRegistryClient,
    prompt_name: str,
    catalog: str,
    schema: str,
) -> _Applied:
    to_version = _parse_target_version(proposal.change.revert_target)
    if to_version is None:
        raise ApplyRefused(
            f"revert proposal {proposal.proposal_id!r} has an unparseable revert_target "
            f"{proposal.change.revert_target!r} — refusing (fail-closed)"
        )
    # Reuse the guarded revert (fail-closed on an unknown version; no-op when already
    # champion) rather than re-implementing alias logic. It writes the alias only
    # because apply=True; its audit lines are captured for the refusal message.
    lines: list[str] = []
    code = revert_champion(
        agent_name=proposal.agent_name,
        to_version=to_version,
        client=registry_client,
        prompt_name=prompt_name,
        alias=CHAMPION_ALIAS,
        catalog=catalog,
        schema=schema,
        apply=True,
        out=lines.append,
    )
    if code != 0:
        raise ApplyRefused(
            f"revert of proposal {proposal.proposal_id!r} to v{to_version} refused: "
            + " | ".join(lines)
        )
    return _Applied(
        summary=f"re-pointed {CHAMPION_ALIAS!r} champion alias to v{to_version}",
        prompt_name=None,
        reverted_to_version=to_version,
        champion_alias=CHAMPION_ALIAS,
    )
