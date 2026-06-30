"""The autonomous loop controller — detect → decide → prove → gate → **propose**.

This is the sequencer of ``docs/LOOP_CONTROLLER.md`` (Option A). On each cycle it:

1. **gathers feedback** for an agent/goal (the injected ``feedback_source``);
2. **decides** which actions address it (:func:`ail.loop.decision_rules.decide` over
   the **compiled goal** — no magic thresholds);
3. **builds a candidate** change for each decision (the injected
   ``candidate_builder``, wrapping :mod:`ail.optimize.assets` / GEPA);
4. **proves** each candidate on the frozen Task Suite (the injected ``prover``,
   wrapping :func:`ail.compare.compare_candidate` /
   :func:`ail.optimize.phase2.run_phase2_comparison` → a
   :class:`~ail.optimize.phase2.Phase2Artifact`);
5. **gates** on readiness + judge trust (the injected ``gate``, wrapping
   :func:`ail.readiness.compute_readiness`); and
6. **proposes** — emits a :class:`~ail.loop.proposals.ProposedAction` with
   ``status=pending`` **only if** the candidate proved an improvement (correctness
   held) *and* the state is gated.

**The controller never applies a change.** There is no apply seam, no champion
alias, no ``CREATE`` — it returns the proposals it *would* emit and nothing else.
A human approves the apply later (lane 3).

**Fail-closed, everywhere.** A decision yields no proposal when the candidate
builder produces nothing, when the proof shows no PROMOTE / a correctness
regression, or when the gate is unmet (readiness wall not cleared, or the
certifying judge is distrusted). A non-improving or crashed candidate is *never*
surfaced — there is no speculative or fabricated proposal.

**Injectable seams = unit-testable.** Every expensive/live step (feedback,
candidate generation, proving, gating) is a small Protocol the tests supply a fake
for, so a cycle runs with **no** live MLflow, agent, or warehouse call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from ail.goals.compiler import CompiledGoal
from ail.loop.decision_rules import (
    Decision,
    DecisionThresholds,
    FeedbackBundle,
    decide,
)
from ail.loop.proposals import (
    GateStatus,
    ProofSummary,
    ProposalStatus,
    ProposedAction,
    ProposedChange,
    derive_proposal_id,
)
from ail.optimize.phase2 import Phase2Artifact
from ail.readiness.contract import ReadinessStatus
from ail.registry import Agent

__all__ = [
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


@dataclass(frozen=True, slots=True)
class Candidate:
    """A built candidate change: the proposal-facing :class:`ProposedChange` plus an
    opaque payload the prover consumes.

    ``change`` is what the proposal records (the diff / SQL / evolved-body ref).
    ``prover_input`` is whatever the (injected) prover needs to run the frozen-suite
    comparison for this change — e.g. an :class:`ail.compare.Intervention` or a
    lever config. The controller **never introspects** ``prover_input``; it only
    hands it back to the prover, keeping the controller agnostic to how each action
    kind is proved.
    """

    change: ProposedChange
    prover_input: object | None = None


class FeedbackSource(Protocol):
    """Gathers the typed feedback for one cycle (no live call from the controller)."""

    def __call__(self) -> FeedbackBundle: ...


class CandidateBuilder(Protocol):
    """Builds a :class:`Candidate` for a decided action, or ``None`` if it cannot.

    Wraps the existing generation capabilities (``ail.optimize.assets`` for a
    metric view, GEPA for a prompt). Returning ``None`` is a first-class
    fail-closed outcome (no candidate ⇒ no proposal), never an error.
    """

    def __call__(
        self, decision: Decision, *, goal: CompiledGoal, agent: Agent
    ) -> Candidate | None: ...


class Prover(Protocol):
    """Proves a candidate on the frozen Task Suite, returning the comparison aggregate.

    Wraps :func:`ail.compare.compare_candidate` /
    :func:`ail.optimize.phase2.run_phase2_comparison`; the returned
    :class:`~ail.optimize.phase2.Phase2Artifact` carries the WITH/WITHOUT,
    correctness-held proof the controller reads (it does not recompute any metric).
    """

    def __call__(
        self, candidate: Candidate, *, goal: CompiledGoal, agent: Agent
    ) -> Phase2Artifact: ...


class Gate(Protocol):
    """Computes readiness for the goal/cohort (wraps :func:`ail.readiness.compute_readiness`)."""

    def __call__(self, *, goal: CompiledGoal, agent: Agent) -> ReadinessStatus: ...


@dataclass(frozen=True, slots=True)
class SkippedDecision:
    """A decision that did *not* become a proposal, with the fail-closed reason.

    Surfaced so a cycle is auditable (and testable): a reviewer can see that a
    candidate was considered but blocked by the proof or the gate, distinct from a
    signal that never fired a rule.
    """

    action_kind: str
    trigger_kind: str
    reason: str


@dataclass(frozen=True, slots=True)
class CycleResult:
    """The outcome of one controller cycle.

    ``proposals`` are the pending :class:`ProposedAction`\\ s the controller would
    emit (it applies none of them); ``skipped`` records the decisions that were
    considered but blocked, fail-closed.
    """

    agent_name: str
    proposals: tuple[ProposedAction, ...] = ()
    skipped: tuple[SkippedDecision, ...] = field(default_factory=tuple)


def evaluate_gate(
    readiness: ReadinessStatus, *, judge_name: str | None = None
) -> tuple[bool, list[str]]:
    """The controller's gate verdict: readiness wall cleared **and** judge trusted.

    Two fail-closed dimensions (``docs/LOOP_CONTROLLER.md`` §"Gate status"):

    * **Readiness** — the data must be sufficient to prove an improvement
      (:attr:`ReadinessStatus.can_prove_improvement`, i.e. ``READY_TO_PROVE``). For
      a quality goal this *already* requires the goal's guardrail judges to be
      trusted (readiness's ``judge_trusted`` gate), so a distrusted required judge
      drops the tier and fails here.
    * **Certifying judge** — when the trigger rests on a specific judge
      (``judge_name``), that judge must be present in the cohort's eval-health and
      not distrusted. This is the explicit "an uncalibrated/distrusted judge cannot
      certify a proposal" wall, independent of whether the goal is quality-typed.

    Returns ``(gated, reasons)``; ``reasons`` is empty iff gated.
    """
    reasons: list[str] = []
    if not readiness.can_prove_improvement:
        detail = "; ".join(readiness.reasons) if readiness.reasons else "no unmet-gate detail"
        reasons.append(f"readiness not met (tier={readiness.tier.value}): {detail}")
    if judge_name is not None:
        jh = next((j for j in readiness.eval_health.judges if j.judge_name == judge_name), None)
        if jh is None:
            reasons.append(
                f"certifying judge {judge_name!r} has no measurement in this cohort — "
                "cannot certify (fail closed)"
            )
        elif jh.distrusted:
            reasons.append(
                f"certifying judge {judge_name!r} is distrusted ({jh.reason}) — cannot certify"
            )
    return (not reasons, reasons)


def run_cycle(
    agent: Agent,
    goal: CompiledGoal,
    *,
    feedback_source: FeedbackSource,
    candidate_builder: CandidateBuilder,
    prover: Prover,
    gate: Gate,
    decision_thresholds: DecisionThresholds | None = None,
    now: str | None = None,
) -> CycleResult:
    """Run one detect→decide→prove→gate→propose cycle; return the proposals to emit.

    Emits a pending :class:`ProposedAction` for a decision **only when** its
    candidate proved an improvement with correctness held *and* the state is gated;
    every other decision is recorded as a fail-closed :class:`SkippedDecision`. The
    controller applies nothing and sets no alias — there is no apply seam to call.

    Args:
        agent: The registered agent the proposals are scoped to (its name keys the
            proposals and the publish table).
        goal: The compiled, **confirmed** optimization goal. The cycle refuses to
            run on an unconfirmed goal (the human-in-the-loop gate the goals
            compiler defers to the controller).
        feedback_source: Produces the cycle's :class:`FeedbackBundle`.
        candidate_builder: Builds a :class:`Candidate` per decision (or ``None``).
        prover: Proves a candidate on the frozen suite (→ ``Phase2Artifact``).
        gate: Computes the goal/cohort readiness.
        decision_thresholds: Adjustable decision-rule bars (defaults applied).
        now: ISO-8601 stamp for ``created_at`` (defaults to now; supplied in tests
            for reproducibility).

    Raises:
        ValueError: if ``goal`` is not human-confirmed.
    """
    if not goal.human_confirmed:
        raise ValueError(
            "refusing to run the loop on an unconfirmed goal; confirm it after human review "
            "(CompiledGoal.confirm()) before driving optimization."
        )

    stamp = now or datetime.now(UTC).isoformat()
    feedback = feedback_source()
    decisions = decide(feedback, goal, thresholds=decision_thresholds)

    # Readiness is per goal/cohort, identical for every decision this cycle, so
    # compute it once. (The certifying-judge check below is per-decision.)
    readiness = gate(goal=goal, agent=agent)

    proposals: list[ProposedAction] = []
    skipped: list[SkippedDecision] = []

    for decision in decisions:
        ak = decision.action_kind.value
        tk = decision.trigger.kind.value

        candidate = candidate_builder(decision, goal=goal, agent=agent)
        if candidate is None:
            skipped.append(
                SkippedDecision(ak, tk, "candidate builder produced no candidate (fail-closed)")
            )
            continue

        artifact = prover(candidate, goal=goal, agent=agent)
        proof = ProofSummary.from_phase2_artifact(artifact)
        if not (proof.proved_improvement and proof.correctness_held):
            skipped.append(
                SkippedDecision(
                    ak,
                    tk,
                    f"not proven on the frozen suite (promote={proof.n_promote}, "
                    f"block={proof.n_block}, errored={proof.n_errored}, "
                    f"correctness_held={proof.correctness_held}) — fail-closed, no proposal",
                )
            )
            continue

        judge_name = decision.trigger.judge_name
        gated, gate_reasons = evaluate_gate(readiness, judge_name=judge_name)
        gate_status = GateStatus.from_readiness(
            readiness, gated=gated, reasons=gate_reasons, judge_name=judge_name
        )
        if not gated:
            skipped.append(SkippedDecision(ak, tk, "not gated: " + "; ".join(gate_reasons)))
            continue

        proposal = ProposedAction(
            proposal_id=derive_proposal_id(
                agent_name=agent.agent_name,
                action_kind=decision.action_kind,
                change=candidate.change,
            ),
            agent_name=agent.agent_name,
            action_kind=decision.action_kind,
            risk_class=decision.risk_class,
            status=ProposalStatus.PENDING,
            objective_metric=goal.objective_metric,
            goal_cohort=goal.cohort_name,
            trigger=decision.trigger,
            change=candidate.change,
            proof=proof,
            gate_status=gate_status,
            created_at=stamp,
        )
        proposals.append(proposal)

    return CycleResult(
        agent_name=agent.agent_name,
        proposals=tuple(proposals),
        skipped=tuple(skipped),
    )
