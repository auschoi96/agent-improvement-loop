"""Lane 3b apply-service tests (:mod:`ail.loop.apply_service`) — offline, seams faked.

Every test is offline: the registry client, warehouse executor, lineage recorder,
gate re-check, body resolver, and both persistence writers are fakes/spies, so no
live MLflow/warehouse write is ever made (no ``live`` marker). Covers the write-path
contract lane 3b owns on the server side:

* approve a proven+gated proposal → the engine applies, the authenticated approver
  flows through, the status advances to ``applied``, and the decision is recorded;
* reject → status ``rejected`` + reason recorded, NO capability called;
* a fail-closed refusal (gate re-check fails) → REFUSED surfaced with its reason,
  nothing applied, status stays pending, the attempted decision is still audited;
* an apply that goes live but whose lineage record fails → APPLIED_UNRECORDED;
* the real body resolver reconstructs the reviewed body from champion + diff, and
  fails closed on a stale diff;
* the gate re-check reuses the real readiness path (fake facts, no live call);
* the flat-row ↔ proposal mapping round-trips against the publish shape;
* the decision/status persistence emit the expected SQL; and
* the live entry refuses an anonymous approver / a missing warehouse fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ail.cohorts import Cohort
from ail.loop import apply_service
from ail.loop.apply import (
    CHAMPION_ALIAS,
    ApprovalDecision,
    DecisionKind,
    GateRecheckResult,
    RegisterableBody,
)
from ail.loop.apply_service import (
    ApplyServiceOutcome,
    ApplyServiceResult,
    _apply_unified_diff,
    _row_to_proposal,
    build_body_resolver,
    build_gate_recheck,
    decide_on_proposal,
    load_pending_proposal,
    mark_proposal_status,
    record_decision,
    run_decision,
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
from ail.loop.publish_proposals import PROPOSAL_COLUMNS, _proposal_row
from ail.optimize.prompt_registry import DEFAULT_CATALOG, DEFAULT_PROMPT_NAME, DEFAULT_SCHEMA
from ail.readiness import ReadinessFacts

FULL_PROMPT_NAME = f"{DEFAULT_CATALOG}.{DEFAULT_SCHEMA}.{DEFAULT_PROMPT_NAME}"
METRIC_VIEW_SQL = (
    "CREATE OR REPLACE VIEW `cat`.`sch`.`mv_token_waste`\nWITH METRICS\nLANGUAGE YAML\nAS $$x$$"
)


# ---------------------------------------------------------------------------
# Fakes / spies
# ---------------------------------------------------------------------------


@dataclass
class FakeVersion:
    version: int
    uri: str = ""
    template: str | None = None


@dataclass
class FakeRegistryClient:
    """Composite prompt + lineage registry seam; records writes, serves a champion."""

    next_version: int = 1
    champion_body: str = "# Champion skill\n\nline one\nline two\n"
    register_calls: list[dict[str, Any]] = field(default_factory=list)
    alias_calls: list[tuple[str, str, int]] = field(default_factory=list)
    versions: list[FakeVersion] = field(default_factory=list)
    alias_to_version: dict[str, int] = field(default_factory=dict)

    def register_prompt(
        self, name: str, template: str, commit_message: str | None, tags: dict[str, str] | None
    ) -> FakeVersion:
        self.register_calls.append({"name": name, "template": template, "tags": tags})
        v = self.next_version
        self.next_version += 1
        fv = FakeVersion(version=v, uri=f"prompts:/{name}/{v}")
        self.versions.append(fv)
        return fv

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        self.alias_calls.append((name, alias, version))
        self.alias_to_version[alias] = version

    def search_prompts(self, filter_string: str | None) -> list[Any]:
        return []

    def load_prompt(self, name_or_uri: str) -> FakeVersion:
        return FakeVersion(version=0, uri=name_or_uri, template=self.champion_body)

    def search_prompt_versions(self, name: str) -> list[FakeVersion]:
        return list(self.versions)

    def get_prompt_version_by_alias(self, name: str, alias: str) -> FakeVersion | None:
        v = self.alias_to_version.get(alias)
        return next((fv for fv in self.versions if fv.version == v), None) if v else None

    @property
    def any_write(self) -> bool:
        return bool(self.register_calls or self.alias_calls)


@dataclass
class Harness:
    """The seams + persistence spies for :func:`decide_on_proposal`."""

    registry: FakeRegistryClient = field(default_factory=FakeRegistryClient)
    warehouse_sql: list[str] = field(default_factory=list)
    lineage_records: list[Any] = field(default_factory=list)
    gate_calls: list[ProposedAction] = field(default_factory=list)
    gate_result: GateRecheckResult = field(default_factory=lambda: GateRecheckResult(ok=True))
    decisions: list[ApplyServiceResult] = field(default_factory=list)
    status_calls: list[tuple[str, str, ProposalStatus]] = field(default_factory=list)
    decision_writer_boom: Exception | None = None
    status_writer_boom: Exception | None = None

    def warehouse(self, sql: str) -> None:
        self.warehouse_sql.append(sql)

    def lineage(self, record: Any) -> None:
        self.lineage_records.append(record)

    def gate(self, proposal: ProposedAction) -> GateRecheckResult:
        self.gate_calls.append(proposal)
        return self.gate_result

    def decision_writer(self, result: ApplyServiceResult) -> None:
        if self.decision_writer_boom is not None:
            raise self.decision_writer_boom
        self.decisions.append(result.model_copy(deep=True))

    def status_writer(self, *, agent_name: str, proposal_id: str, status: ProposalStatus) -> None:
        if self.status_writer_boom is not None:
            raise self.status_writer_boom
        self.status_calls.append((agent_name, proposal_id, status))

    @property
    def any_capability_called(self) -> bool:
        return bool(self.warehouse_sql) or self.registry.any_write


def _full_body_resolver(proposal: ProposedAction) -> RegisterableBody:
    from ail.optimize.prompt_registry import PromptProvenance, PromptSource

    return RegisterableBody(
        body="# Reviewed skill\n\nReuse prior reads.",
        provenance=PromptProvenance(source=PromptSource.SEED, changed=True),
    )


def _proposal(
    action_kind: ActionKind,
    change: ProposedChange,
    *,
    proposal_id: str = "prop-1",
    proved_improvement: bool = True,
    correctness_held: bool = True,
) -> ProposedAction:
    return ProposedAction(
        proposal_id=proposal_id,
        agent_name="claude_code",
        action_kind=action_kind,
        risk_class=default_risk_class(action_kind),
        status=ProposalStatus.PENDING,
        objective_metric="total_tokens",
        goal_cohort="claude_code",
        trigger=TriggerSignal(
            kind=TriggerKind.RLM_RECOMMENDED_ASSET,
            summary="RLM recommended a token-waste metric view",
            trace_refs=["t1", "t2"],
        ),
        change=change,
        proof=ProofSummary(
            objective_metric="total_tokens",
            proved_improvement=proved_improvement,
            correctness_held=correctness_held,
            realized_savings_pct=35.4,
            n_promote=3,
        ),
        gate_status=GateStatus(readiness_tier="ready_to_prove", gated=True),
    )


def _metric_view_proposal(**kw: Any) -> ProposedAction:
    change = ProposedChange(kind=ChangeKind.METRIC_VIEW_SQL, summary="view", sql=METRIC_VIEW_SQL)
    return _proposal(ActionKind.METRIC_VIEW, change, **kw)


def _skill_update_proposal(
    diff: str = "--- a\n+++ b\n@@ -1,1 +1,2 @@\n line one\n+inserted\n",
) -> ProposedAction:
    change = ProposedChange(kind=ChangeKind.SKILL_DIFF, summary="skill", diff=diff)
    return _proposal(ActionKind.SKILL_UPDATE, change)


def _approve(p: ProposedAction, *, approver: str = "reviewer@databricks.com") -> ApprovalDecision:
    return ApprovalDecision(
        proposal_id=p.proposal_id,
        decision=DecisionKind.APPROVE,
        approver=approver,
        decided_at="2026-06-30T12:00:00+00:00",
    )


def _reject(p: ProposedAction, *, reason: str = "rule mis-fired") -> ApprovalDecision:
    return ApprovalDecision(
        proposal_id=p.proposal_id,
        decision=DecisionKind.REJECT,
        approver="reviewer@databricks.com",
        reason=reason,
        decided_at="2026-06-30T12:00:00+00:00",
    )


def _decide(
    proposal: ProposedAction,
    decision: ApprovalDecision,
    *,
    h: Harness,
    body_resolver: Any = _full_body_resolver,
) -> ApplyServiceResult:
    return decide_on_proposal(
        proposal,
        decision,
        registry_client=h.registry,
        warehouse_executor=h.warehouse,
        lineage_recorder=h.lineage,
        gate_recheck=h.gate,
        body_resolver=body_resolver,
        decision_writer=h.decision_writer,
        status_writer=h.status_writer,
    )


# ---------------------------------------------------------------------------
# decide_on_proposal — approve / reject / refuse / applied-unrecorded
# ---------------------------------------------------------------------------


def test_approve_applies_records_decision_and_advances_status() -> None:
    h = Harness()
    proposal = _metric_view_proposal()
    result = _decide(proposal, _approve(proposal), h=h)

    assert result.outcome is ApplyServiceOutcome.APPLIED
    assert result.status == ProposalStatus.APPLIED.value
    assert result.approver == "reviewer@databricks.com"  # authenticated approver flows through
    assert result.created_view == "cat.sch.mv_token_waste"
    # the engine actually applied (CREATE reached the warehouse) + gate re-checked
    assert h.warehouse_sql == [METRIC_VIEW_SQL]
    assert len(h.gate_calls) == 1
    # status advanced to applied, decision recorded once with the approver
    assert h.status_calls == [("claude_code", "prop-1", ProposalStatus.APPLIED)]
    assert len(h.decisions) == 1
    assert h.decisions[0].approver == "reviewer@databricks.com"
    assert result.decision_recorded is True


def test_reject_records_reason_and_calls_no_capability() -> None:
    h = Harness()
    proposal = _metric_view_proposal()
    result = _decide(proposal, _reject(proposal, reason="view not useful"), h=h)

    assert result.outcome is ApplyServiceOutcome.REJECTED
    assert result.status == ProposalStatus.REJECTED.value
    assert result.reason == "view not useful"
    # no capability, no gate re-check, no lineage on reject
    assert not h.any_capability_called
    assert h.gate_calls == []
    assert h.lineage_records == []
    # status set rejected + decision (with reason) recorded
    assert h.status_calls == [("claude_code", "prop-1", ProposalStatus.REJECTED)]
    assert h.decisions[0].reason == "view not useful"


def test_refused_surfaces_reason_applies_nothing_and_stays_pending() -> None:
    h = Harness(gate_result=GateRecheckResult(ok=False, reasons=["judge went distrusted"]))
    proposal = _metric_view_proposal()
    result = _decide(proposal, _approve(proposal), h=h)

    assert result.outcome is ApplyServiceOutcome.REFUSED
    assert "judge went distrusted" in (result.refused_reason or "")
    assert result.status == ProposalStatus.PENDING.value  # still needs attention
    # nothing applied; status NOT advanced; the refusal IS audited
    assert not h.any_capability_called
    assert h.status_calls == []
    assert len(h.decisions) == 1
    assert h.decisions[0].outcome is ApplyServiceOutcome.REFUSED


def test_applied_but_lineage_record_failure_surfaces_applied_unrecorded() -> None:
    h = Harness()

    def failing_lineage(record: Any) -> None:
        raise RuntimeError("lineage write failed")

    proposal = _metric_view_proposal()
    result = decide_on_proposal(
        proposal,
        _approve(proposal),
        registry_client=h.registry,
        warehouse_executor=h.warehouse,
        lineage_recorder=failing_lineage,
        gate_recheck=h.gate,
        body_resolver=None,
        decision_writer=h.decision_writer,
        status_writer=h.status_writer,
    )
    assert result.outcome is ApplyServiceOutcome.APPLIED_UNRECORDED
    assert h.warehouse_sql == [METRIC_VIEW_SQL]  # the change IS live
    assert result.status == ProposalStatus.APPLIED.value
    assert result.lineage_recorded is False
    # status still advanced + decision recorded (so the queue drops it; operator reconciles)
    assert h.status_calls == [("claude_code", "prop-1", ProposalStatus.APPLIED)]
    assert len(h.decisions) == 1


def test_live_apply_with_audit_write_failure_surfaces_applied_unrecorded() -> None:
    # BLOCKING-1 regression: a LIVE apply whose decision-audit append then fails must
    # surface as APPLIED_UNRECORDED (needs-reconcile), never a clean `applied`.
    h = Harness(decision_writer_boom=RuntimeError("decisions table write failed"))
    proposal = _metric_view_proposal()
    result = _decide(proposal, _approve(proposal), h=h)
    assert h.warehouse_sql == [METRIC_VIEW_SQL]  # the change IS live
    assert result.outcome is ApplyServiceOutcome.APPLIED_UNRECORDED
    assert result.outcome is not ApplyServiceOutcome.APPLIED  # must not read as clean success
    assert result.decision_recorded is False
    assert "decision audit not recorded" in (result.error or "")


def test_live_apply_with_status_write_failure_surfaces_applied_unrecorded() -> None:
    # BLOCKING-1 regression: the two persistence writes are independent — a LIVE apply
    # whose STATUS advancement fails is also applied-but-unrecorded (the queue would
    # not drop it; the operator must reconcile).
    h = Harness(status_writer_boom=RuntimeError("proposals status UPDATE failed"))
    proposal = _metric_view_proposal()
    result = _decide(proposal, _approve(proposal), h=h)
    assert h.warehouse_sql == [METRIC_VIEW_SQL]  # the change IS live
    assert result.outcome is ApplyServiceOutcome.APPLIED_UNRECORDED
    assert "status not advanced" in (result.error or "")
    # the decision audit itself still landed (only the status write failed)
    assert result.decision_recorded is True
    assert len(h.decisions) == 1
    assert h.decisions[0].outcome is ApplyServiceOutcome.APPLIED_UNRECORDED

    apply_h = Harness()
    apply_proposal = _metric_view_proposal(proposal_id="prop-apply")
    apply_result = _decide(apply_proposal, _approve(apply_proposal), h=apply_h)

    assert apply_result.outcome is ApplyServiceOutcome.APPLIED
    assert apply_h.decisions[0].outcome is ApplyServiceOutcome.APPLIED

    reject_h = Harness()
    reject_proposal = _metric_view_proposal(proposal_id="prop-reject")
    reject_result = _decide(
        reject_proposal, _reject(reject_proposal, reason="not useful"), h=reject_h
    )

    assert reject_result.outcome is ApplyServiceOutcome.REJECTED
    assert reject_h.decisions[0].outcome is ApplyServiceOutcome.REJECTED


def test_reject_audit_failure_stays_rejected_not_unrecorded() -> None:
    # A REJECTED outcome has NO live change, so a failed audit is NOT applied-unrecorded
    # — it stays rejected (only annotated). Guards against over-downgrading.
    h = Harness(decision_writer_boom=RuntimeError("decisions table write failed"))
    proposal = _metric_view_proposal()
    result = _decide(proposal, _reject(proposal, reason="not useful"), h=h)
    assert result.outcome is ApplyServiceOutcome.REJECTED
    assert result.decision_recorded is False
    assert "decision audit not recorded" in (result.error or "")


def test_approve_skill_update_registers_resolved_body_and_sets_champion() -> None:
    h = Harness()
    proposal = _skill_update_proposal()
    result = _decide(proposal, _approve(proposal), h=h, body_resolver=_full_body_resolver)
    assert result.outcome is ApplyServiceOutcome.APPLIED
    assert h.registry.register_calls[0]["template"].startswith("# Reviewed skill")
    assert h.registry.alias_calls == [(FULL_PROMPT_NAME, CHAMPION_ALIAS, 1)]
    assert result.new_version == 1


# ---------------------------------------------------------------------------
# build_body_resolver — reconstruct reviewed body from champion + diff (fail-closed)
# ---------------------------------------------------------------------------


def test_body_resolver_applies_diff_to_current_champion() -> None:
    registry = FakeRegistryClient(champion_body="line one\nline two\n")
    resolver = build_body_resolver(registry_client=registry)
    proposal = _skill_update_proposal(diff="@@ -1,1 +1,2 @@\n line one\n+inserted\n")
    resolved = resolver(proposal)
    assert resolved.body == "line one\ninserted\nline two"
    # the resolved body is a full body, never the diff (the engine also guards this)
    assert resolved.body != proposal.change.diff


def test_body_resolver_fails_closed_on_stale_diff() -> None:
    registry = FakeRegistryClient(champion_body="totally different\ncontent\n")
    resolver = build_body_resolver(registry_client=registry)
    proposal = _skill_update_proposal(diff="@@ -1,1 +1,2 @@\n line one\n+inserted\n")
    with pytest.raises(ValueError, match="does not match the source"):
        resolver(proposal)


def test_body_resolver_refuses_a_no_op_diff_equal_body() -> None:
    # BLOCKING-2 regression: a no-op patch (only context lines) yields a body identical
    # to the champion — it must be refused (fail-closed), never registered as a fake
    # new version. Nothing is registered.
    registry = FakeRegistryClient(champion_body="line one\nline two\n")
    resolver = build_body_resolver(registry_client=registry)
    proposal = _skill_update_proposal(diff="@@ -1,2 +1,2 @@\n line one\n line two\n")
    with pytest.raises(ValueError, match="no-op"):
        resolver(proposal)
    assert registry.register_calls == []  # nothing registered


def test_body_resolver_refuses_diff_that_nets_to_current() -> None:
    # A diff that removes a line then re-adds the identical line nets to the champion
    # body — also a fake change; refuse it.
    registry = FakeRegistryClient(champion_body="alpha\nbeta\n")
    resolver = build_body_resolver(registry_client=registry)
    proposal = _skill_update_proposal(diff="@@ -1,2 +1,2 @@\n alpha\n-beta\n+beta\n")
    with pytest.raises(ValueError, match="no-op"):
        resolver(proposal)


def test_apply_unified_diff_requires_a_hunk() -> None:
    with pytest.raises(ValueError, match="no hunks"):
        _apply_unified_diff("a\nb\n", "--- a\n+++ b\n")


def test_apply_unified_diff_deletes_and_inserts() -> None:
    original = "one\ntwo\nthree\n"
    diff = "@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
    assert _apply_unified_diff(original, diff) == "one\nTWO\nthree"


# ---------------------------------------------------------------------------
# build_gate_recheck — reuse the real readiness path (fake facts, no live call)
# ---------------------------------------------------------------------------


def test_gate_recheck_ok_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = ReadinessFacts(
        trace_count=10_000, label_count=0, frozen_suite_present=True, n_scored_traces=0, judges=[]
    )
    monkeypatch.setattr("ail.jobs.readiness_preflight.gather_facts", lambda *a, **k: facts)
    recheck = build_gate_recheck(experiment_id="exp1", cohort=Cohort.by_agent("claude_code"))
    result = recheck(_metric_view_proposal())
    assert result.ok is True


def test_gate_recheck_fails_when_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = ReadinessFacts(
        trace_count=1, label_count=0, frozen_suite_present=False, n_scored_traces=0, judges=[]
    )
    monkeypatch.setattr("ail.jobs.readiness_preflight.gather_facts", lambda *a, **k: facts)
    recheck = build_gate_recheck(experiment_id="exp1", cohort=Cohort.by_agent("claude_code"))
    result = recheck(_metric_view_proposal())
    assert result.ok is False
    assert result.reasons


def test_gate_recheck_fails_closed_when_facts_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("trace store unreachable")

    monkeypatch.setattr("ail.jobs.readiness_preflight.gather_facts", boom)
    recheck = build_gate_recheck(experiment_id="exp1", cohort=Cohort.by_agent("claude_code"))
    result = recheck(_metric_view_proposal())
    assert result.ok is False
    assert "could not gather readiness facts" in result.reasons[0]


# ---------------------------------------------------------------------------
# flat-row ↔ proposal mapping round-trip (against the publish shape)
# ---------------------------------------------------------------------------


def test_row_to_proposal_round_trips_the_publish_shape() -> None:
    original = _skill_update_proposal(diff="@@ -1,1 +1,2 @@\n line one\n+x\n")
    flat = _proposal_row(original, generated_at="2026-06-30T00:00:00+00:00")
    row = dict(zip(PROPOSAL_COLUMNS, [None if v is None else str(v) for v in flat], strict=True))
    restored = _row_to_proposal(row)
    assert restored.proposal_id == original.proposal_id
    assert restored.action_kind is ActionKind.SKILL_UPDATE
    assert restored.change.diff == original.change.diff
    assert restored.proof.proved_improvement is True
    assert restored.proof.n_promote == 3
    assert restored.trigger.trace_refs == ["t1", "t2"]
    assert restored.gate_status.gated is True


# ---------------------------------------------------------------------------
# load_pending_proposal / persistence SQL (offline via monkeypatch)
# ---------------------------------------------------------------------------


def test_load_pending_proposal_returns_none_when_no_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_service, "_query_rows", lambda *a, **k: [])
    got = load_pending_proposal(
        client=object(), warehouse_id="w", agent_name="claude_code", proposal_id="nope"
    )
    assert got is None


def test_load_pending_proposal_maps_row(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _metric_view_proposal()
    flat = _proposal_row(original, generated_at="2026-06-30T00:00:00+00:00")
    row = dict(zip(PROPOSAL_COLUMNS, [None if v is None else str(v) for v in flat], strict=True))
    monkeypatch.setattr(apply_service, "_query_rows", lambda *a, **k: [row])
    got = load_pending_proposal(
        client=object(), warehouse_id="w", agent_name="claude_code", proposal_id="prop-1"
    )
    assert got is not None
    assert got.action_kind is ActionKind.METRIC_VIEW
    assert got.change.sql == METRIC_VIEW_SQL


def test_record_decision_emits_ddl_and_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(apply_service, "_execute", lambda c, w, s: executed.append(s))
    result = ApplyServiceResult(
        outcome=ApplyServiceOutcome.APPLIED,
        proposal_id="prop-1",
        agent_name="claude_code",
        decision=DecisionKind.APPROVE,
        approver="reviewer@databricks.com",
        decided_at="2026-06-30T12:00:00+00:00",
        action_kind="metric_view",
        summary="created view",
    )
    record_decision(result, client=object(), warehouse_id="w")
    assert any(
        "CREATE TABLE IF NOT EXISTS" in s and "agent_action_decisions" in s for s in executed
    )
    insert = [s for s in executed if s.startswith("INSERT INTO")][0]
    assert "'reviewer@databricks.com'" in insert
    assert "'applied'" in insert


def test_mark_proposal_status_updates_only_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(apply_service, "_execute", lambda c, w, s: executed.append(s))
    mark_proposal_status(
        client=object(),
        warehouse_id="w",
        agent_name="claude_code",
        proposal_id="prop-1",
        status=ProposalStatus.APPLIED,
    )
    sql = executed[0]
    assert sql.startswith("UPDATE")
    assert "SET status = 'applied'" in sql
    assert "status = 'pending'" in sql  # only advances a still-pending row


# ---------------------------------------------------------------------------
# run_decision — live entry fail-closed guards (no client is ever built)
# ---------------------------------------------------------------------------


def test_run_decision_refuses_anonymous_approver() -> None:
    result = run_decision(
        proposal_id="prop-1",
        agent_name="claude_code",
        decision="approve",
        approver="   ",
        reason=None,
        decided_at="2026-06-30T12:00:00+00:00",
        warehouse_id="w",
    )
    assert result.outcome is ApplyServiceOutcome.REFUSED
    assert "anonymous" in (result.refused_reason or "")


def test_run_decision_refuses_unknown_decision() -> None:
    result = run_decision(
        proposal_id="prop-1",
        agent_name="claude_code",
        decision="delete",
        approver="reviewer@databricks.com",
        reason=None,
        decided_at="2026-06-30T12:00:00+00:00",
        warehouse_id="w",
    )
    assert result.outcome is ApplyServiceOutcome.REFUSED
    assert "unknown decision" in (result.refused_reason or "")


def test_run_decision_errors_without_warehouse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    result = run_decision(
        proposal_id="prop-1",
        agent_name="claude_code",
        decision="approve",
        approver="reviewer@databricks.com",
        reason=None,
        decided_at="2026-06-30T12:00:00+00:00",
        warehouse_id=None,
    )
    assert result.outcome is ApplyServiceOutcome.ERROR
    assert "warehouse" in (result.error or "")
