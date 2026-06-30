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
"""

from __future__ import annotations

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
