"""The autonomous loop controller: detect → decide → prove → gate → **propose**.

``docs/LOOP_CONTROLLER.md`` lane 2. The framework autonomously detects a problem
from feedback, decides which action addresses it, generates and **proves** a
candidate on the frozen Task Suite, gates on readiness + judge trust, and emits a
human-gated :class:`~ail.loop.proposals.ProposedAction`. It **never applies a
change** — a human approves the apply in the app (lane 3).

* :mod:`ail.loop.proposals` — the typed, inert proposed-action record (why + what
  + proof + gate status).
* :mod:`ail.loop.decision_rules` — goal-parameterized pure rules mapping a feedback
  signal to a candidate action kind.
* :mod:`ail.loop.controller` — :func:`~ail.loop.controller.run_cycle`, sequencing
  the loop over injectable seams (fail-closed, unit-testable without live runs).
* :mod:`ail.loop.publish_proposals` — write pending proposals to the unified
  ``agent_proposed_actions`` UC table lane 3 reads.
* :mod:`ail.loop.apply` — lane 3a: the fail-closed apply-on-approval engine the
  app's Approve button calls (register a version + set champion, ``CREATE`` a view,
  or revert), recorded to the lineage / audit timeline.
"""

from __future__ import annotations

from ail.loop.apply import (
    CHAMPION_ALIAS,
    AppliedChangeRecord,
    ApplyOutcome,
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
from ail.loop.apply_service import (
    DECISIONS_TABLE,
    ApplyServiceOutcome,
    ApplyServiceResult,
    DecisionWriter,
    StatusWriter,
    build_body_resolver,
    build_gate_recheck,
    build_lineage_recorder,
    build_registry_client,
    build_warehouse_executor,
    decide_on_proposal,
    load_pending_proposal,
    mark_proposal_status,
    record_decision,
    run_decision,
)
from ail.loop.controller import (
    Candidate,
    CandidateBuilder,
    CycleResult,
    FeedbackSource,
    Gate,
    Prover,
    SkippedDecision,
    evaluate_gate,
    run_cycle,
)
from ail.loop.decision_rules import (
    Decision,
    DecisionThresholds,
    FeedbackBundle,
    JudgeDimensionSignal,
    PostApplyRegressionSignal,
    RedundantReadSignal,
    RlmAssetSignal,
    decide,
    objective_target_met,
)
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    GateStatus,
    ProofSummary,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    RiskClass,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
    derive_proposal_id,
)

__all__ = [
    # apply-on-approval engine (lane 3a)
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
    # apply-service write-path (lane 3b, server side)
    "DECISIONS_TABLE",
    "ApplyServiceOutcome",
    "ApplyServiceResult",
    "DecisionWriter",
    "StatusWriter",
    "build_body_resolver",
    "build_gate_recheck",
    "build_lineage_recorder",
    "build_registry_client",
    "build_warehouse_executor",
    "decide_on_proposal",
    "load_pending_proposal",
    "mark_proposal_status",
    "record_decision",
    "run_decision",
    # proposals
    "ActionKind",
    "RiskClass",
    "ProposalStatus",
    "TriggerKind",
    "ChangeKind",
    "TriggerSignal",
    "ProposedChange",
    "ProofSummary",
    "GateStatus",
    "ProposedAction",
    "default_risk_class",
    "derive_proposal_id",
    # decision rules
    "DecisionThresholds",
    "RlmAssetSignal",
    "RedundantReadSignal",
    "JudgeDimensionSignal",
    "PostApplyRegressionSignal",
    "FeedbackBundle",
    "Decision",
    "objective_target_met",
    "decide",
    # controller
    "Candidate",
    "FeedbackSource",
    "CandidateBuilder",
    "Prover",
    "Gate",
    "SkippedDecision",
    "CycleResult",
    "evaluate_gate",
    "run_cycle",
]
