"""Apply-on-approval engine tests (:mod:`ail.loop.apply`) — fail-closed, seams faked.

Every test is **offline**: the registry client, warehouse executor, lineage
recorder, gate re-check, and body resolver are injected fakes, so no live
MLflow/warehouse write is ever made (no ``live`` marker). Covers the lane-3a plan
items (a)-(h):

* (a) approve a proven+gated PENDING metric_view → the CREATE SQL reaches the
  warehouse executor, lineage recorded, status=applied;
* (b) approve a proven+gated skill_update / gepa_prompt → register + champion alias
  set, lineage recorded, status=applied;
* (c) approve a revert → champion alias re-pointed to the target version;
* (d) reject → status=rejected + reason stored, NO capability called;
* (e) refuse a non-pending (already applied/rejected) proposal;
* (f) refuse when the apply-time gate re-check fails (no capability called);
* (g) refuse when the proof no longer shows proved_improvement / correctness_held;
* (h) the approver identity is recorded on the decision (and flows to the audit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ail.loop.apply import (
    CHAMPION_ALIAS,
    ApplyOutcome,
    ApplyRecordError,
    ApplyRefused,
    ApprovalDecision,
    DecisionKind,
    GateRecheckResult,
    RegisterableBody,
    apply_approved_proposal,
)
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    ProofSummary,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
)
from ail.optimize.gepa_runner import GepaOptimizationResult
from ail.optimize.phase2 import Phase2Artifact
from ail.optimize.prompt_registry import (
    DEFAULT_PROMPT_NAME,
    PromptProvenance,
    PromptSource,
)

TEST_CATALOG = "test_catalog"
TEST_SCHEMA = "test_schema"
FULL_PROMPT_NAME = f"{TEST_CATALOG}.{TEST_SCHEMA}.{DEFAULT_PROMPT_NAME}"
METRIC_VIEW_SQL = (
    "CREATE OR REPLACE VIEW `cat`.`sch`.`mv_token_waste`\n"
    "WITH METRICS\nLANGUAGE YAML\nAS $$\nversion: '1.1'\n$$"
)


# ---------------------------------------------------------------------------
# Fakes for every seam (no live MLflow / warehouse write on any path)
# ---------------------------------------------------------------------------


@dataclass
class FakeVersion:
    version: int
    uri: str = ""


@dataclass
class FakeRegistryClient:
    """Combined stand-in for the prompt-registry + lineage registry seams.

    Implements every method of :class:`ail.loop.apply.ApplyRegistryClient` (the union
    of ``PromptRegistryClient`` and ``LineageRegistryClient``) and records calls.
    """

    next_version: int = 1
    versions: list[FakeVersion] = field(default_factory=list)
    alias_to_version: dict[str, int] = field(default_factory=dict)
    register_calls: list[dict[str, Any]] = field(default_factory=list)
    alias_calls: list[tuple[str, str, int]] = field(default_factory=list)

    # -- PromptRegistryClient --
    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None,
        tags: dict[str, str] | None,
    ) -> FakeVersion:
        self.register_calls.append(
            {"name": name, "template": template, "commit_message": commit_message, "tags": tags}
        )
        version = self.next_version
        self.next_version += 1
        fv = FakeVersion(version=version, uri=f"prompts:/{name}/{version}")
        self.versions.append(fv)
        return fv

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        self.alias_calls.append((name, alias, version))
        self.alias_to_version[alias] = version

    def search_prompts(self, filter_string: str | None) -> list[Any]:
        return []

    def load_prompt(self, name_or_uri: str) -> Any:
        return None

    # -- LineageRegistryClient --
    def search_prompt_versions(self, name: str) -> list[FakeVersion]:
        return list(self.versions)

    def get_prompt_version_by_alias(self, name: str, alias: str) -> FakeVersion | None:
        v = self.alias_to_version.get(alias)
        if v is None:
            return None
        return next((fv for fv in self.versions if fv.version == v), None)

    @property
    def any_write(self) -> bool:
        """True iff any registry write (register or alias) happened."""
        return bool(self.register_calls or self.alias_calls)


@dataclass
class Seams:
    """The four+one injected seams plus the spies that prove what was (not) called."""

    registry: FakeRegistryClient = field(default_factory=FakeRegistryClient)
    warehouse_sql: list[str] = field(default_factory=list)
    lineage_records: list[Any] = field(default_factory=list)
    gate_calls: list[ProposedAction] = field(default_factory=list)
    gate_result: GateRecheckResult = field(default_factory=lambda: GateRecheckResult(ok=True))

    def warehouse(self, sql: str) -> None:
        self.warehouse_sql.append(sql)

    def lineage(self, record: Any) -> None:
        self.lineage_records.append(record)

    def gate(self, proposal: ProposedAction) -> GateRecheckResult:
        self.gate_calls.append(proposal)
        return self.gate_result

    @property
    def any_capability_called(self) -> bool:
        """True iff any live-effecting capability ran (warehouse / registry write)."""
        return bool(self.warehouse_sql) or self.registry.any_write


def _resolver(proposal: ProposedAction) -> RegisterableBody:
    """A body resolver that returns a full skill body (the reviewed change, in lane 3b)."""
    return RegisterableBody(
        body="# Read-cache skill\n\nReuse prior reads; do not re-read unchanged files.",
        # Source is the resolver's (lane 3b's) call; the engine registers verbatim.
        provenance=PromptProvenance(source=PromptSource.SEED, registration_reason="skill_update"),
        commit_message="Apply approved read-cache skill update",
    )


def _forbidden_resolver(proposal: ProposedAction) -> RegisterableBody:
    raise AssertionError("body_resolver must not be called on this path")


# ---------------------------------------------------------------------------
# Proposal builders
# ---------------------------------------------------------------------------


def _proposal(
    action_kind: ActionKind,
    change: ProposedChange,
    *,
    proposal_id: str = "prop-1",
    status: ProposalStatus = ProposalStatus.PENDING,
    proved_improvement: bool = True,
    correctness_held: bool = True,
    proof_present: bool = True,
) -> ProposedAction:
    # proof_present=False mints an evidence-first proposal (proof=None): it rests on its
    # evidence + gate status alone (ail.loop.evidence_cycle), no frozen-suite proof.
    proof = (
        ProofSummary(
            objective_metric="total_tokens",
            proved_improvement=proved_improvement,
            correctness_held=correctness_held,
            realized_savings_pct=35.4,
            n_promote=3,
        )
        if proof_present
        else None
    )
    return ProposedAction(
        proposal_id=proposal_id,
        agent_name="claude_code",
        action_kind=action_kind,
        risk_class=default_risk_class(action_kind),
        status=status,
        objective_metric="total_tokens",
        goal_cohort="claude_code",
        trigger=TriggerSignal(
            kind=TriggerKind.RLM_RECOMMENDED_ASSET,
            summary="RLM recommended a token-waste metric view",
            trace_refs=["t1", "t2"],
        ),
        change=change,
        proof=proof,
        gate_status=GateStatus(readiness_tier="ready_to_prove", gated=True),
    )


def _metric_view_proposal(**kw: Any) -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.METRIC_VIEW_SQL, summary="token waste view", sql=METRIC_VIEW_SQL
    )
    return _proposal(ActionKind.METRIC_VIEW, change, **kw)


def _skill_update_proposal(**kw: Any) -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.SKILL_DIFF, summary="read-cache skill", diff="--- a/skill\n+++ b/skill"
    )
    return _proposal(ActionKind.SKILL_UPDATE, change, **kw)


def _gepa_proposal(candidate_path: Path, **kw: Any) -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.EVOLVED_BODY_REF,
        summary="gepa-evolved skill",
        evolved_body_ref=str(candidate_path),
    )
    return _proposal(ActionKind.GEPA_PROMPT, change, **kw)


def _revert_proposal(target: str = "v1", **kw: Any) -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.REVERT_REF, summary="revert regression", revert_target=target
    )
    return _proposal(ActionKind.REVERT, change, **kw)


def _agent_task_proposal(**kw: Any) -> ProposedAction:
    change = ProposedChange(
        kind=ChangeKind.AGENT_TASK_PLAN,
        summary="agent-produced read-cache tool",
        plan="Add a read-cache tool; the agent re-reads unchanged files across 5 traces.",
    )
    return _proposal(ActionKind.AGENT_TASK, change, **kw)


def _approve(
    proposal: ProposedAction, *, approver: str = "austin@databricks.com"
) -> ApprovalDecision:
    return ApprovalDecision(
        proposal_id=proposal.proposal_id,
        decision=DecisionKind.APPROVE,
        approver=approver,
        decided_at="2026-06-30T12:00:00+00:00",
    )


def _reject(
    proposal: ProposedAction, *, approver: str = "austin@databricks.com", reason: str = "mis-fired"
) -> ApprovalDecision:
    return ApprovalDecision(
        proposal_id=proposal.proposal_id,
        decision=DecisionKind.REJECT,
        approver=approver,
        reason=reason,
        decided_at="2026-06-30T12:00:00+00:00",
    )


def _improving_candidate_json(tmp_path: Path) -> Path:
    """Write a GEPA candidate that beats seed on the held-out split (so apply proceeds)."""
    result = GepaOptimizationResult(
        component_name="token-efficient-execution",
        seed_skill_body="# Seed skill\n\nAvoid re-reading.",
        evolved_skill_body="# Evolved skill\n\nReuse context; skip redundant reads.",
        changed=True,
        reflection_lm="databricks:/databricks-claude-sonnet-4-6",
        gepa_num_candidates=4,
        gepa_best_val_score=0.82,
        suite_version="phase2-mini",
        suite_content_hash="deadbeefcafe0001",
        holdout_task_ids=["ts-04", "ts-05"],
        train_task_ids=["ts-01", "ts-02", "ts-03"],
        holdout_evolved=Phase2Artifact(n_tasks=3, n_promote=2, realized_token_savings_pct=45.0),
        holdout_seed_baseline=Phase2Artifact(
            n_tasks=3, n_promote=1, realized_token_savings_pct=30.0
        ),
    )
    path = tmp_path / "gepa_candidate.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def _run(
    proposal: ProposedAction,
    decision: ApprovalDecision,
    *,
    seams: Seams,
    body_resolver: Any = _forbidden_resolver,
) -> Any:
    return apply_approved_proposal(
        proposal,
        decision,
        registry_client=seams.registry,
        warehouse_executor=seams.warehouse,
        lineage_recorder=seams.lineage,
        gate_recheck=seams.gate,
        body_resolver=body_resolver,
        catalog=TEST_CATALOG,
        schema=TEST_SCHEMA,
    )


# ---------------------------------------------------------------------------
# (a) approve a proven+gated PENDING metric_view -> CREATE SQL to the warehouse
# ---------------------------------------------------------------------------


def test_approve_metric_view_executes_create_sql_and_records_lineage() -> None:
    seams = Seams()
    proposal = _metric_view_proposal()
    result = _run(proposal, _approve(proposal), seams=seams)

    # the CREATE DDL reached the warehouse executor verbatim (no regeneration)
    assert seams.warehouse_sql == [METRIC_VIEW_SQL]
    # no prompt registry write for an additive asset
    assert not seams.registry.any_write
    # gate was re-checked at apply time
    assert len(seams.gate_calls) == 1
    # lineage recorded once, carrying what/why/who
    assert len(seams.lineage_records) == 1
    rec = seams.lineage_records[0]
    assert rec.action_kind is ActionKind.METRIC_VIEW
    assert rec.created_view == "cat.sch.mv_token_waste"
    assert rec.approver == "austin@databricks.com"
    assert rec.trigger_summary.startswith("RLM recommended")
    # typed result
    assert result.outcome is ApplyOutcome.APPLIED
    assert result.proposal.status is ProposalStatus.APPLIED
    assert result.created_view == "cat.sch.mv_token_waste"
    assert result.lineage_recorded is True


# ---------------------------------------------------------------------------
# (b) approve a proven+gated skill_update / gepa_prompt -> register + champion
# ---------------------------------------------------------------------------


def test_approve_skill_update_registers_body_and_sets_champion() -> None:
    seams = Seams()
    proposal = _skill_update_proposal()
    result = _run(proposal, _approve(proposal), seams=seams, body_resolver=_resolver)

    # the resolved body (not the diff) was registered
    assert len(seams.registry.register_calls) == 1
    assert seams.registry.register_calls[0]["template"].startswith("# Read-cache skill")
    assert seams.registry.register_calls[0]["name"] == FULL_PROMPT_NAME
    # champion alias pointed at the new version
    assert seams.registry.alias_calls == [(FULL_PROMPT_NAME, CHAMPION_ALIAS, 1)]
    # no warehouse write for a prompt change
    assert seams.warehouse_sql == []
    # lineage + status
    assert len(seams.lineage_records) == 1
    assert seams.lineage_records[0].new_version == 1
    assert seams.lineage_records[0].champion_alias == CHAMPION_ALIAS
    assert result.outcome is ApplyOutcome.APPLIED
    assert result.proposal.status is ProposalStatus.APPLIED
    assert result.new_version == 1
    assert result.champion_alias == CHAMPION_ALIAS


def test_approve_skill_update_without_resolver_is_refused() -> None:
    seams = Seams()
    proposal = _skill_update_proposal()
    with pytest.raises(ApplyRefused, match="body_resolver"):
        apply_approved_proposal(
            proposal,
            _approve(proposal),
            registry_client=seams.registry,
            warehouse_executor=seams.warehouse,
            lineage_recorder=seams.lineage,
            gate_recheck=seams.gate,
            body_resolver=None,
        )
    # fail-closed: never register a diff as a body
    assert not seams.any_capability_called
    assert seams.lineage_records == []


def test_approve_gepa_prompt_registers_evolved_body_and_sets_champion(tmp_path: Path) -> None:
    seams = Seams()
    candidate = _improving_candidate_json(tmp_path)
    proposal = _gepa_proposal(candidate)
    result = _run(proposal, _approve(proposal), seams=seams)

    # the evolved body from the candidate artifact was registered + champion set
    assert len(seams.registry.register_calls) == 1
    assert seams.registry.register_calls[0]["template"].startswith("# Evolved skill")
    assert seams.registry.alias_calls == [(FULL_PROMPT_NAME, CHAMPION_ALIAS, 1)]
    assert result.outcome is ApplyOutcome.APPLIED
    assert result.new_version == 1
    assert seams.lineage_records[0].action_kind is ActionKind.GEPA_PROMPT


def test_approve_gepa_prompt_missing_artifact_is_refused(tmp_path: Path) -> None:
    seams = Seams()
    proposal = _gepa_proposal(tmp_path / "does_not_exist.json")
    with pytest.raises(ApplyRefused, match="not a readable candidate artifact"):
        _run(proposal, _approve(proposal), seams=seams)
    assert not seams.any_capability_called


# ---------------------------------------------------------------------------
# (c) approve a revert -> champion alias re-pointed to the target version
# ---------------------------------------------------------------------------


def test_approve_revert_repoints_champion_alias() -> None:
    seams = Seams(
        registry=FakeRegistryClient(
            versions=[FakeVersion(1, "prompts:/p/1"), FakeVersion(2, "prompts:/p/2")],
            alias_to_version={CHAMPION_ALIAS: 2},  # current champion is v2
            next_version=3,
        )
    )
    proposal = _revert_proposal(target="v1")
    result = _run(proposal, _approve(proposal), seams=seams)

    # champion re-pointed from v2 to the target v1 (reusing the guarded revert logic)
    assert (FULL_PROMPT_NAME, CHAMPION_ALIAS, 1) in seams.registry.alias_calls
    assert seams.registry.alias_to_version[CHAMPION_ALIAS] == 1
    # a revert registers no new version
    assert seams.registry.register_calls == []
    assert result.outcome is ApplyOutcome.APPLIED
    assert result.reverted_to_version == 1
    assert seams.lineage_records[0].reverted_to_version == 1


def test_approve_revert_unknown_version_is_refused() -> None:
    seams = Seams(
        registry=FakeRegistryClient(
            versions=[FakeVersion(1), FakeVersion(2)],
            alias_to_version={CHAMPION_ALIAS: 2},
        )
    )
    proposal = _revert_proposal(target="v99")  # no such version
    with pytest.raises(ApplyRefused, match="refused"):
        _run(proposal, _approve(proposal), seams=seams)
    # guarded revert never points champion at a missing version
    assert seams.registry.alias_calls == []
    assert seams.lineage_records == []


# ---------------------------------------------------------------------------
# (d) reject -> status=rejected + reason stored, NO capability called
# ---------------------------------------------------------------------------


def test_reject_records_reason_and_calls_no_capability() -> None:
    seams = Seams()
    proposal = _metric_view_proposal()
    result = _run(
        proposal, _reject(proposal, reason="rule mis-fired, view not useful"), seams=seams
    )

    assert result.outcome is ApplyOutcome.REJECTED
    assert result.proposal.status is ProposalStatus.REJECTED
    assert result.reason == "rule mis-fired, view not useful"
    assert result.approver == "austin@databricks.com"
    # no capability, no gate re-check, no lineage on reject
    assert not seams.any_capability_called
    assert seams.gate_calls == []
    assert seams.lineage_records == []


def test_reject_without_reason_is_rejected_at_construction() -> None:
    proposal = _metric_view_proposal()
    with pytest.raises(ValueError, match="non-empty reason"):
        ApprovalDecision(
            proposal_id=proposal.proposal_id,
            decision=DecisionKind.REJECT,
            approver="austin@databricks.com",
            decided_at="2026-06-30T12:00:00+00:00",
        )


# ---------------------------------------------------------------------------
# (e) refuse a non-pending proposal (already applied / rejected / superseded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [ProposalStatus.APPLIED, ProposalStatus.REJECTED, ProposalStatus.SUPERSEDED]
)
def test_refuse_non_pending_proposal(status: ProposalStatus) -> None:
    seams = Seams()
    proposal = _metric_view_proposal(status=status)
    with pytest.raises(ApplyRefused, match="not pending"):
        _run(proposal, _approve(proposal), seams=seams)
    assert not seams.any_capability_called
    assert seams.gate_calls == []  # short-circuits before the gate re-check
    assert seams.lineage_records == []


def test_refuse_decision_referencing_a_different_proposal() -> None:
    seams = Seams()
    proposal = _metric_view_proposal(proposal_id="prop-A")
    other_decision = _approve(_metric_view_proposal(proposal_id="prop-B"))
    with pytest.raises(ApplyRefused, match="references proposal"):
        _run(proposal, other_decision, seams=seams)
    assert not seams.any_capability_called


# ---------------------------------------------------------------------------
# (f) refuse when the apply-time gate re-check fails (fail-closed, no capability)
# ---------------------------------------------------------------------------


def test_refuse_when_gate_recheck_fails() -> None:
    seams = Seams(
        gate_result=GateRecheckResult(ok=False, reasons=["judge modularity went distrusted"])
    )
    proposal = _metric_view_proposal()
    with pytest.raises(ApplyRefused, match="gate re-check failed"):
        _run(proposal, _approve(proposal), seams=seams)
    # the gate WAS consulted, but nothing was applied
    assert len(seams.gate_calls) == 1
    assert not seams.any_capability_called
    assert seams.lineage_records == []


# ---------------------------------------------------------------------------
# (g) refuse when the proof no longer shows proved_improvement / correctness_held
# ---------------------------------------------------------------------------


def test_refuse_when_not_proved_improvement() -> None:
    seams = Seams()
    proposal = _metric_view_proposal(proved_improvement=False)
    with pytest.raises(ApplyRefused, match="no longer carries a proven improvement"):
        _run(proposal, _approve(proposal), seams=seams)
    assert not seams.any_capability_called
    # proof is checked before the gate re-check
    assert seams.gate_calls == []
    assert seams.lineage_records == []


def test_refuse_when_correctness_not_held() -> None:
    seams = Seams()
    proposal = _metric_view_proposal(correctness_held=False)
    with pytest.raises(ApplyRefused, match="no longer carries a proven improvement"):
        _run(proposal, _approve(proposal), seams=seams)
    assert not seams.any_capability_called
    assert seams.lineage_records == []


# ---------------------------------------------------------------------------
# (h) the approver identity is recorded on the decision (and flows to the audit)
# ---------------------------------------------------------------------------


def test_approver_identity_recorded_on_decision_and_audit() -> None:
    seams = Seams()
    proposal = _metric_view_proposal()
    decision = _approve(proposal, approver="reviewer@databricks.com")
    result = _run(proposal, decision, seams=seams)

    assert decision.approver == "reviewer@databricks.com"
    assert decision.decided_at == "2026-06-30T12:00:00+00:00"
    assert result.approver == "reviewer@databricks.com"
    assert result.decided_at == "2026-06-30T12:00:00+00:00"
    rec = seams.lineage_records[0]
    assert rec.approver == "reviewer@databricks.com"
    assert rec.decided_at == "2026-06-30T12:00:00+00:00"


def test_decision_requires_non_empty_approver() -> None:
    proposal = _metric_view_proposal()
    with pytest.raises(ValueError, match="approver identity"):
        ApprovalDecision(
            proposal_id=proposal.proposal_id,
            decision=DecisionKind.APPROVE,
            approver="   ",
            decided_at="2026-06-30T12:00:00+00:00",
        )


# ---------------------------------------------------------------------------
# BLOCKING 1 — a lineage-record failure AFTER a successful apply must fail LOUD:
# the change is live (APPLIED), the record must be reconciled, and it is neither a
# clean success nor a not-applied refusal.
# ---------------------------------------------------------------------------


def test_lineage_record_failure_after_apply_raises_apply_record_error() -> None:
    seams = Seams()
    proposal = _metric_view_proposal()
    boom = RuntimeError("lineage table write failed")

    def failing_lineage(record: Any) -> None:
        raise boom

    with pytest.raises(ApplyRecordError) as excinfo:
        apply_approved_proposal(
            proposal,
            _approve(proposal),
            registry_client=seams.registry,
            warehouse_executor=seams.warehouse,
            lineage_recorder=failing_lineage,
            gate_recheck=seams.gate,
        )
    err = excinfo.value
    # the capability DID apply — the CREATE ran and is now live (not rolled back)
    assert seams.warehouse_sql == [METRIC_VIEW_SQL]
    # reported applied-but-unrecorded: APPLIED result, status advanced, not recorded
    assert err.result.outcome is ApplyOutcome.APPLIED
    assert err.result.proposal.status is ProposalStatus.APPLIED
    assert err.result.lineage_recorded is False
    assert err.record.proposal_id == proposal.proposal_id
    assert err.cause is boom
    # DISTINCT from a fail-closed refusal — must never read as not-applied
    assert not isinstance(err, ApplyRefused)


# ---------------------------------------------------------------------------
# BLOCKING 2 — decided_at must be a real timestamp (non-empty / non-whitespace)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decided_at", ["", "   ", "\t\n"])
def test_decision_requires_non_empty_decided_at(decided_at: str) -> None:
    proposal = _metric_view_proposal()
    with pytest.raises(ValueError, match="decided_at"):
        ApprovalDecision(
            proposal_id=proposal.proposal_id,
            decision=DecisionKind.APPROVE,
            approver="austin@databricks.com",
            decided_at=decided_at,
        )


# ---------------------------------------------------------------------------
# FOLD IN — defensive resolver-body guards (never register an empty/diff-as body)
# ---------------------------------------------------------------------------


def test_resolver_returning_empty_body_is_refused() -> None:
    seams = Seams()
    proposal = _skill_update_proposal()

    def empty_resolver(p: ProposedAction) -> RegisterableBody:
        return RegisterableBody(body="   ", provenance=PromptProvenance(source=PromptSource.SEED))

    with pytest.raises(ApplyRefused, match="empty prompt body"):
        _run(proposal, _approve(proposal), seams=seams, body_resolver=empty_resolver)
    assert not seams.registry.any_write
    assert seams.lineage_records == []


def test_resolver_returning_diff_as_body_is_refused() -> None:
    seams = Seams()
    proposal = _skill_update_proposal()
    assert proposal.change.diff is not None  # precondition: a skill_update carries a diff
    diff_text: str = proposal.change.diff

    def diff_as_body(p: ProposedAction) -> RegisterableBody:
        return RegisterableBody(
            body=diff_text, provenance=PromptProvenance(source=PromptSource.SEED)
        )

    with pytest.raises(ApplyRefused, match="diff as a body"):
        _run(proposal, _approve(proposal), seams=seams, body_resolver=diff_as_body)
    assert not seams.registry.any_write
    assert seams.lineage_records == []


# ---------------------------------------------------------------------------
# FOLD IN — routing boundary refuses a change_kind / action_kind mismatch
# ---------------------------------------------------------------------------


def test_routing_refuses_change_kind_action_kind_mismatch() -> None:
    seams = Seams()
    # A METRIC_VIEW proposal carrying a SKILL_DIFF change. model_copy does NOT re-run
    # the ProposedAction cross-field validator, so this simulates a proposal that
    # reached the engine with validation bypassed — it must fail closed at routing.
    malformed = _metric_view_proposal().model_copy(
        update={"change": ProposedChange(kind=ChangeKind.SKILL_DIFF, summary="x", diff="d")}
    )
    with pytest.raises(ApplyRefused, match="requires change kind"):
        _run(malformed, _approve(malformed), seams=seams)
    assert not seams.any_capability_called
    assert seams.lineage_records == []


# ---------------------------------------------------------------------------
# LANE L7a — an approved evidence-only (proof=None) proposal of a DETERMINISTIC
# kind APPLIES on its evidence + gate status alone (docs/PRODUCT_ARCHITECTURE.md
# §3/§7). The gate re-check is still the wall; a proof-dependent kind (GEPA) and a
# stale/non-pending/mismatched decision are still refused; proven applies unchanged.
# ---------------------------------------------------------------------------


def test_approve_evidence_only_metric_view_applies() -> None:
    seams = Seams()
    proposal = _metric_view_proposal(proof_present=False)
    assert proposal.proof is None  # evidence-first: no frozen-suite proof
    result = _run(proposal, _approve(proposal), seams=seams)

    # the CREATE DDL still reached the warehouse verbatim; the gate was re-checked
    assert seams.warehouse_sql == [METRIC_VIEW_SQL]
    assert len(seams.gate_calls) == 1
    # applied, status advanced, lineage recorded
    assert result.outcome is ApplyOutcome.APPLIED
    assert result.proposal.status is ProposalStatus.APPLIED
    assert result.created_view == "cat.sch.mv_token_waste"
    assert result.lineage_recorded is True
    # the audit record honestly reflects an evidence-only apply (no proven delta)
    rec = seams.lineage_records[0]
    assert rec.proved_improvement is False
    assert rec.realized_savings_pct is None


def test_approve_evidence_only_skill_update_applies() -> None:
    seams = Seams()
    proposal = _skill_update_proposal(proof_present=False)
    assert proposal.proof is None
    result = _run(proposal, _approve(proposal), seams=seams, body_resolver=_resolver)

    # a full body (not the diff) registered + champion re-pointed, on evidence alone
    assert len(seams.registry.register_calls) == 1
    assert seams.registry.alias_calls == [(FULL_PROMPT_NAME, CHAMPION_ALIAS, 1)]
    assert len(seams.gate_calls) == 1
    assert result.outcome is ApplyOutcome.APPLIED
    assert result.proposal.status is ProposalStatus.APPLIED
    assert result.new_version == 1
    assert result.champion_alias == CHAMPION_ALIAS


def test_evidence_only_still_refused_when_gate_recheck_fails() -> None:
    # The wall holds: an evidence-only proposal whose apply-time gate no longer holds
    # is refused, not applied — exactly as a proven proposal would be.
    seams = Seams(
        gate_result=GateRecheckResult(ok=False, reasons=["readiness dropped below the wall"])
    )
    proposal = _metric_view_proposal(proof_present=False)
    with pytest.raises(ApplyRefused, match="gate re-check failed"):
        _run(proposal, _approve(proposal), seams=seams)
    # the gate WAS consulted, but nothing was applied
    assert len(seams.gate_calls) == 1
    assert not seams.any_capability_called
    assert seams.lineage_records == []


@pytest.mark.parametrize(
    "status", [ProposalStatus.APPLIED, ProposalStatus.REJECTED, ProposalStatus.SUPERSEDED]
)
def test_evidence_only_non_pending_is_still_refused(status: ProposalStatus) -> None:
    seams = Seams()
    proposal = _metric_view_proposal(proof_present=False, status=status)
    with pytest.raises(ApplyRefused, match="not pending"):
        _run(proposal, _approve(proposal), seams=seams)
    assert not seams.any_capability_called
    assert seams.gate_calls == []  # short-circuits before the gate re-check


def test_evidence_only_decision_mismatch_is_still_refused() -> None:
    seams = Seams()
    proposal = _metric_view_proposal(proof_present=False, proposal_id="prop-A")
    other_decision = _approve(_metric_view_proposal(proof_present=False, proposal_id="prop-B"))
    with pytest.raises(ApplyRefused, match="references proposal"):
        _run(proposal, other_decision, seams=seams)
    assert not seams.any_capability_called
    assert seams.gate_calls == []


def test_evidence_only_gepa_prompt_is_still_refused(tmp_path: Path) -> None:
    # GEPA's apply re-runs the held-out improvement check — it is proof-DEPENDENT, so an
    # evidence-only (proof=None) GEPA proposal must STILL be refused (no blanket accept),
    # even though its candidate artifact exists and would otherwise register.
    seams = Seams()
    candidate = _improving_candidate_json(tmp_path)
    proposal = _gepa_proposal(candidate, proof_present=False)
    assert proposal.proof is None
    with pytest.raises(ApplyRefused, match="not an evidence-only-applyable"):
        _run(proposal, _approve(proposal), seams=seams)
    # refused before any capability ran and before the gate re-check
    assert not seams.any_capability_called
    assert seams.gate_calls == []
    assert seams.lineage_records == []


# ---------------------------------------------------------------------------
# LANE L7b-1 — AGENT_TASK (the open-ended executor's change) is RECOGNIZED at the
# apply routing boundary but FAIL-CLOSED REFUSED until the executor lane (L7b-2)
# exists, and it is NEVER on the deterministic evidence-only apply allowlist.
# ---------------------------------------------------------------------------


def test_agent_task_excluded_from_evidence_only_allowlist() -> None:
    # An open-ended agent-produced change must never apply via L7a's deterministic
    # evidence-only path — it needs the executor + a human diff-preview. Assert the
    # allowlist itself excludes it (the module also asserts this at import).
    from ail.loop.apply import _EVIDENCE_ONLY_APPLYABLE_KINDS

    assert ActionKind.AGENT_TASK not in _EVIDENCE_ONLY_APPLYABLE_KINDS


def test_apply_agent_task_is_refused_not_yet_wired() -> None:
    # A proof present + gate ok gets the proposal past the proof/gate walls, so the
    # ROUTING refusal (the executor lane does not exist) is what surfaces. Nothing
    # half-applies: no capability ran, no lineage recorded.
    seams = Seams()
    proposal = _agent_task_proposal()  # proof present, gate ok
    with pytest.raises(ApplyRefused, match="not yet wired"):
        _run(proposal, _approve(proposal), seams=seams)
    assert len(seams.gate_calls) == 1  # reached routing only after the gate wall
    assert not seams.any_capability_called
    assert seams.lineage_records == []


def test_apply_agent_task_evidence_only_is_refused_before_gate() -> None:
    # The realistic shape: an AGENT_TASK carries no frozen-suite proof (proof=None). It is
    # NOT an evidence-only-applyable kind, so it is refused at the proof/evidence check —
    # before the gate re-check and before any capability.
    seams = Seams()
    proposal = _agent_task_proposal(proof_present=False)
    assert proposal.proof is None
    with pytest.raises(ApplyRefused, match="not an evidence-only-applyable"):
        _run(proposal, _approve(proposal), seams=seams)
    assert not seams.any_capability_called
    assert seams.gate_calls == []  # short-circuits before the gate re-check
    assert seams.lineage_records == []


def test_evidence_only_revert_is_still_refused() -> None:
    # REVERT is excluded from _EVIDENCE_ONLY_APPLYABLE_KINDS (its evidence-only apply is
    # out of L7a scope), so a proof=None revert is still refused fail-closed — no proof,
    # no allowlisted deterministic kind, nothing applied.
    seams = Seams()
    proposal = _revert_proposal(target="v1", proof_present=False)
    assert proposal.proof is None
    with pytest.raises(ApplyRefused, match="not an evidence-only-applyable"):
        _run(proposal, _approve(proposal), seams=seams)
    assert not seams.any_capability_called
    assert seams.gate_calls == []
    assert seams.lineage_records == []


# ---------------------------------------------------------------------------
# LANE L7a (blocking fix) — the resolved-body guard is robust: a body that IS the
# diff up to normalization, or that still carries unified-diff MARKERS, is refused
# on the evidence-only lane (no proof backstop); a legitimate body that merely
# starts a line with '+'/'-' or mentions "diff" still applies (no over-correction).
# ---------------------------------------------------------------------------


def test_evidence_only_resolver_near_diff_body_is_refused() -> None:
    # A near-diff (diff + trailing newline) is non-empty and NOT byte-identical to the
    # diff, but normalizes to it — refuse it (else an essentially-unresolved diff ships
    # as the skill body). The diff here carries NO structure markers, so ONLY the
    # normalized-equality check can catch it.
    seams = Seams()
    change = ProposedChange(
        kind=ChangeKind.SKILL_DIFF, summary="skill", diff="line one\n-remove me\n+add me"
    )
    proposal = _proposal(ActionKind.SKILL_UPDATE, change, proof_present=False)
    assert proposal.proof is None

    def near_diff_resolver(p: ProposedAction) -> RegisterableBody:
        assert p.change.diff is not None
        return RegisterableBody(
            body=p.change.diff + "\n", provenance=PromptProvenance(source=PromptSource.SEED)
        )

    with pytest.raises(ApplyRefused, match="diff as a body"):
        _run(proposal, _approve(proposal), seams=seams, body_resolver=near_diff_resolver)
    assert not seams.registry.any_write
    assert seams.lineage_records == []


def test_evidence_only_resolver_diff_structure_body_is_refused() -> None:
    # A body that still carries a unified-diff hunk header (@@ ... @@) is diff-shaped, not
    # a resolved body — refuse it even though it is NOT equal to the proposal's diff (so
    # only the diff-structure detection can catch it).
    seams = Seams()
    proposal = _skill_update_proposal(proof_present=False)

    def diff_structure_resolver(p: ProposedAction) -> RegisterableBody:
        return RegisterableBody(
            body="# Skill\n\n@@ -1,2 +1,3 @@\n context\n+added\n",
            provenance=PromptProvenance(source=PromptSource.SEED),
        )

    with pytest.raises(ApplyRefused, match="diff-shaped body"):
        _run(proposal, _approve(proposal), seams=seams, body_resolver=diff_structure_resolver)
    assert not seams.registry.any_write
    assert seams.lineage_records == []


def test_evidence_only_resolver_legit_body_with_plus_line_still_applies() -> None:
    # No over-correction: a real multi-line body that merely starts lines with '+' / '-'
    # and mentions the word "diff" has NO unified-diff markers — it must still apply.
    seams = Seams()
    proposal = _skill_update_proposal(proof_present=False)

    def legit_resolver(p: ProposedAction) -> RegisterableBody:
        return RegisterableBody(
            body=(
                "# Read-cache skill\n\n"
                "+ Reuse prior reads; do not re-read unchanged files.\n"
                "- Never re-open a file already loaded.\n"
                "Run a quick diff review before editing.\n"
            ),
            provenance=PromptProvenance(source=PromptSource.SEED, registration_reason="skill"),
        )

    result = _run(proposal, _approve(proposal), seams=seams, body_resolver=legit_resolver)
    assert result.outcome is ApplyOutcome.APPLIED
    assert seams.registry.register_calls[0]["template"].startswith("# Read-cache skill")
    assert seams.registry.alias_calls == [(FULL_PROMPT_NAME, CHAMPION_ALIAS, 1)]


def test_evidence_only_resolver_generic_header_diff_body_is_refused() -> None:
    # A GENERIC unified-diff body with plain headers (--- old / +++ new — NO a//b/
    # prefixes, NO @@ hunk header, NO `diff --git` line) is still diff-shaped and must be
    # refused. Guards the broadened, prefix-agnostic header-pair detection.
    seams = Seams()
    proposal = _skill_update_proposal(proof_present=False)

    def generic_header_resolver(p: ProposedAction) -> RegisterableBody:
        return RegisterableBody(
            body="--- old_skill\n+++ new_skill\n context\n+added\n",
            provenance=PromptProvenance(source=PromptSource.SEED),
        )

    with pytest.raises(ApplyRefused, match="diff-shaped body"):
        _run(proposal, _approve(proposal), seams=seams, body_resolver=generic_header_resolver)
    assert not seams.registry.any_write
    assert seams.lineage_records == []


def test_evidence_only_resolver_lone_horizontal_rule_body_still_applies() -> None:
    # No over-correction: a legitimate body with a LONE `---` markdown horizontal rule
    # (not immediately followed by a `+++ ` line) is not a diff header pair — it must
    # still apply, even alongside a leading '-' line and the word "diff".
    seams = Seams()
    proposal = _skill_update_proposal(proof_present=False)

    def lone_rule_resolver(p: ProposedAction) -> RegisterableBody:
        return RegisterableBody(
            body=(
                "# Read-cache skill\n\n"
                "Reuse prior reads; do not re-read unchanged files.\n\n"
                "---\n\n"
                "- Run a quick diff review before editing.\n"
            ),
            provenance=PromptProvenance(source=PromptSource.SEED, registration_reason="skill"),
        )

    result = _run(proposal, _approve(proposal), seams=seams, body_resolver=lone_rule_resolver)
    assert result.outcome is ApplyOutcome.APPLIED
    assert seams.registry.register_calls[0]["template"].startswith("# Read-cache skill")
    assert seams.registry.alias_calls == [(FULL_PROMPT_NAME, CHAMPION_ALIAS, 1)]
