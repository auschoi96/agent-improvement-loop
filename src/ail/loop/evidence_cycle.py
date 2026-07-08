"""Evidence-first cycle: detect → decide → build → **gate on evidence** → propose.

The **local companion planner** lane (``docs/PRODUCT_ARCHITECTURE.md`` §3/§7). It is
the *evidence-first* sibling of :func:`ail.loop.controller.run_cycle`: same detect →
decide (Lane A + Lane B) → build-a-candidate pipeline, but it **decouples proving
from proposing**. Where :func:`~ail.loop.controller.run_cycle` hardwires a frozen-suite
``prover(candidate)`` into the emit condition — a proposal only exists after it
*proved* an improvement — this cycle emits a PENDING proposal on **evidence + gate
alone**, calling **no prover** and (for every ordinary builder) carrying no
:class:`~ail.loop.proposals.ProofSummary`. The lone exception is a builder that is
*itself* a frozen-suite verification — the GEPA candidate builder, whose
:func:`ail.optimize.gepa_runner.run_gepa_optimization` already held-out-validated the
evolved body; it hands that proof through on the :class:`~ail.loop.controller.Candidate`
(``candidate.proof``) so the GEPA_PROMPT proposal it produces is applyable at all (the
apply engine refuses a proof-less GEPA_PROMPT). The cycle still runs no prover — it
merely records the proof the builder computed.

Why this is the product's default path (not a weakening):

    | Tier | When | What it gives |
    |------|------|---------------|
    | **1 — Predict** | Always, cheap | Judge + RLM + L0 evidence: *why* a change is recommended |
    | **2 — Verify** | Opt-in, on demand | Candidate vs. baseline on the frozen suite → a delta |
    | **3 — Confirm** | After ship | Organic before/after by agent version + one-click revert |

Proving moved from a *mandatory gate on every change* to *Tier-2 verification the
human can run when undecided* (``docs/PRODUCT_ARCHITECTURE.md`` §11). **The human is
the gate; evidence informs the call.** So the planner reads the judge/RLM/L0 evidence,
builds a concrete candidate (so a later executor / opt-in prover has a target), gates
on **readiness + judge trust only** (the same :func:`ail.loop.controller.evaluate_gate`,
*unweakened* — a distrusted certifying judge or an unready cohort still blocks), and
proposes. It never proves and never applies.

Hard lines (identical in spirit to the controller; the reviewer checks these):

* **Propose-only.** No prover call, no apply seam, no ``CREATE`` / alias / register.
  It emits PENDING :class:`~ail.loop.proposals.ProposedAction`\\ s and nothing else.
* **Fail-closed everywhere.** No candidate ⇒ no proposal; not gated ⇒ no proposal; a
  per-decision error ⇒ a recorded :class:`~ail.loop.controller.SkippedDecision`, never
  a crashed cycle or a dropped-but-silent decision.
* **No fabricated evidence.** The feedback read is the contract: if
  ``feedback_source`` raises (the trace store is unreachable, auth failed), the error
  **propagates** — the cycle never invents a "why" to propose on. Lane B (the LLM
  planner) failing is the *one* soft failure: :func:`combined_decisions` catches it,
  records it, and Lane A still runs and emits (preserved exactly from
  :mod:`ail.loop.planner` — a Lane B failure never suppresses Lane A).

Maximal reuse: Lane A + Lane B combination and its fail-closed planner handling are
:func:`ail.loop.planner.combined_decisions` verbatim; the gate is
:func:`ail.loop.controller.evaluate_gate` verbatim; the candidate seam, result, and
skip records are the controller's own types. Only the *emit condition* differs
(evidence + gate, no proof).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ail.goals.compiler import CompiledGoal
from ail.loop.controller import (
    CandidateBuilder,
    CycleResult,
    FeedbackSource,
    Gate,
    SkippedDecision,
    evaluate_gate,
)
from ail.loop.decision_rules import DecisionThresholds
from ail.loop.planner import CombinedDecisions, Planner, agent_planner, combined_decisions
from ail.loop.proposals import (
    ActionKind,
    GateStatus,
    ProposalStatus,
    ProposedAction,
    derive_proposal_id,
)
from ail.registry import Agent

__all__ = [
    "EvidenceCycleResult",
    "run_evidence_cycle",
]

#: Action kinds whose apply RE-VERIFIES a frozen-suite proof and so are **absent** from
#: the apply engine's ``ail.loop.apply._EVIDENCE_ONLY_APPLYABLE_KINDS`` allowlist: they
#: cannot be applied on evidence + gate alone. This evidence-first cycle calls **no**
#: prover, so such a kind can only legitimately reach PENDING carrying a *pre-computed*
#: genuine :class:`~ail.loop.proposals.ProofSummary` from a builder that is itself a
#: frozen-suite verification (the GEPA candidate builder). The cycle therefore refuses to
#: **emit** one without a genuine proof — the invariant "a proof-required proposal cannot
#: reach PENDING without a real proof" is enforced here at the emit boundary, not merely
#: relied upon as a downstream apply refusal. Kept narrow to the kind this cycle's
#: builders can produce (``GEPA_PROMPT``); ``REVERT`` / ``AGENT_TASK`` are never emitted
#: by the companion's builders and the apply engine remains the backstop for them.
_PROOF_REQUIRED_ACTION_KINDS: frozenset[ActionKind] = frozenset({ActionKind.GEPA_PROMPT})


@dataclass(frozen=True, slots=True)
class EvidenceCycleResult:
    """The outcome of one evidence-first cycle: the controller-shaped result + the plan.

    Mirrors :class:`ail.loop.planner.PlannerCycleResult` so a caller reads a layered
    A+B evidence cycle the same way it reads a layered A+B *proven* cycle:

    * ``result`` — the :class:`~ail.loop.controller.CycleResult`: the PENDING
      :class:`~ail.loop.proposals.ProposedAction`\\ s to publish (each with
      ``proof=None``) and the fail-closed :class:`~ail.loop.controller.SkippedDecision`\\ s.
    * ``plan`` — the :class:`~ail.loop.planner.CombinedDecisions`: how many decisions
      each lane contributed, how many Lane-B duplicates were dropped, and — if Lane B
      failed closed — the recorded reason (never a fabricated decision).
    """

    result: CycleResult
    plan: CombinedDecisions


def run_evidence_cycle(
    agent: Agent,
    goal: CompiledGoal,
    *,
    feedback_source: FeedbackSource,
    candidate_builder: CandidateBuilder,
    gate: Gate,
    planner: Planner = agent_planner,
    decision_thresholds: DecisionThresholds | None = None,
    now: str | None = None,
) -> EvidenceCycleResult:
    """Run one evidence-first cycle; return the PENDING proposals to publish.

    Reads the feedback once, forms the de-duped Lane A ∪ Lane B decision union
    (:func:`ail.loop.planner.combined_decisions` — Lane A survives a Lane B failure),
    then for each decision: builds a concrete candidate, and — if the state is gated
    on **readiness + judge trust** (:func:`ail.loop.controller.evaluate_gate`,
    unweakened) — emits a PENDING :class:`~ail.loop.proposals.ProposedAction` carrying
    the trigger **evidence** and the gate status but **no**
    :class:`~ail.loop.proposals.ProofSummary` (``proof=None``). It calls **no** prover
    and applies nothing.

    Every non-emit is a fail-closed :class:`~ail.loop.controller.SkippedDecision`: no
    candidate, not gated, or a per-decision error (isolated so one failure cannot drop
    proposals already built this pass). The ``feedback_source`` read is **not** caught
    — if the evidence cannot be read the error propagates, so a proposal is never built
    on a fabricated "why" (contrast Lane B, whose failure :func:`combined_decisions`
    catches so Lane A still emits).

    Args:
        agent: The registered agent the proposals are scoped to.
        goal: The compiled, **human-confirmed** optimization goal (refuses an
            unconfirmed goal, exactly as :func:`ail.loop.controller.run_cycle` does).
        feedback_source: Produces the cycle's :class:`~ail.loop.decision_rules.FeedbackBundle`.
        candidate_builder: Builds a :class:`~ail.loop.controller.Candidate` per decision
            (or ``None``) — the concrete change the proposal records and a later opt-in
            Tier-2 prover / executor targets.
        gate: Computes the goal/cohort readiness (wraps
            :func:`ail.readiness.compute_readiness`).
        planner: The Lane-B decision source (defaults to
            :func:`ail.loop.planner.agent_planner`).
        decision_thresholds: Adjustable Lane-A decision-rule bars (defaults applied).
        now: ISO-8601 stamp for ``created_at`` (defaults to now; supplied in tests).

    Raises:
        ValueError: if ``goal`` is not human-confirmed.
    """
    if not goal.human_confirmed:
        raise ValueError(
            "refusing to run the evidence-first cycle on an unconfirmed goal; confirm it after "
            "human review (CompiledGoal.confirm()) before driving planning."
        )

    stamp = now or datetime.now(UTC).isoformat()
    feedback = feedback_source()
    plan = combined_decisions(
        feedback, goal, agent, planner=planner, thresholds=decision_thresholds
    )

    # Readiness is per goal/cohort, identical for every decision this cycle, so compute
    # it once. (The certifying-judge check inside evaluate_gate is per-decision.)
    readiness = gate(goal=goal, agent=agent)

    proposals: list[ProposedAction] = []
    skipped: list[SkippedDecision] = []

    for decision in plan.decisions:
        ak = decision.action_kind.value
        tk = decision.trigger.kind.value

        # Per-decision fault isolation (mirrors ail.loop.controller.run_cycle): one
        # decision's failure (a candidate builder that raised, say) must not crash the
        # cycle or drop proposals already built earlier this pass. Any exception is
        # caught, recorded as a fail-closed skip *with the error*, and the loop
        # continues — fail-closed is preserved exactly (an errored decision yields NO
        # proposal, only a SkippedDecision that keeps a genuine bug visible).
        try:
            candidate = candidate_builder(decision, goal=goal, agent=agent)
            if candidate is None:
                skipped.append(
                    SkippedDecision(ak, tk, "candidate builder produced no candidate (fail-closed)")
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

            # Emit-boundary proof invariant (fail-closed): a proof-REQUIRED action kind
            # (GEPA_PROMPT — its apply re-verifies a held-out proof) must carry a GENUINE
            # ProofSummary or it is NOT emitted. "Genuine" is all four of: present, both
            # flags true (proved_improvement + correctness_held), objective matches the
            # goal, AND held-out provenance from a real run (n_promote >= 1 + a frozen-suite
            # identity) — the last blocks a hand-built flags-only proof that merely sets the
            # booleans. This guards the EMIT boundary itself, so a builder that hands back a
            # proof-required candidate with proof=None, a fake/unrelated proof, or a
            # provenance-less flags-only proof cannot create a broken PENDING row the apply
            # engine would later have to refuse. "Cannot reach PENDING without a real proof"
            # is an invariant of THIS cycle, not merely a downstream apply refusal.
            if decision.action_kind in _PROOF_REQUIRED_ACTION_KINDS:
                proof = candidate.proof
                if proof is None or not (proof.proved_improvement and proof.correctness_held):
                    skipped.append(
                        SkippedDecision(
                            ak,
                            tk,
                            "proof-required action kind carries no genuine frozen-suite proof "
                            "(missing, or does not prove an improvement with correctness held) "
                            "— fail-closed, cannot reach PENDING",
                        )
                    )
                    continue
                # The proof must speak to the objective this proposal claims to optimize;
                # an unrelated-objective proof (e.g. a token-savings proof on a quality
                # goal) is mislabeled and rejected. (The GEPA builder already binds to the
                # routed target; this is the independent emit-boundary backstop.)
                if proof.objective_metric != goal.objective_metric:
                    skipped.append(
                        SkippedDecision(
                            ak,
                            tk,
                            f"proof objective {proof.objective_metric!r} does not match the goal "
                            f"objective {goal.objective_metric!r} — mislabeled/unrelated proof, "
                            "fail-closed",
                        )
                    )
                    continue
                # Genuine held-out PROVENANCE (fail-closed): the boolean flags + objective
                # are hand-settable, so a matching-objective proof with both flags True is
                # NOT enough — a proof-required kind must additionally carry the provenance a
                # REAL held-out run populates. A genuine proof is built by
                # ``ProofSummary.from_phase2_artifact`` from a Phase2Artifact
                # ``run_gepa_optimization``/``run_phase2_comparison`` produced: it PROMOTEd at
                # least one held-out task (``n_promote >= 1`` — ``proved_improvement`` is
                # literally ``artifact.n_promote > 0``) and records the frozen suite it
                # validated against (a non-empty ``suite_content_hash`` — always a sha256 for
                # a frozen suite). A hand-constructed flags-only / provenance-less proof
                # (``n_promote == 0``, empty suite identity) evidences NO real run, so it is
                # REJECTED here: it cannot reach PENDING for a kind whose apply re-verifies a
                # held-out proof.
                if proof.n_promote < 1 or not proof.suite_content_hash.strip():
                    skipped.append(
                        SkippedDecision(
                            ak,
                            tk,
                            "proof-required action kind carries a flags-only / provenance-less "
                            f"proof (no genuine held-out run: n_promote={proof.n_promote}, "
                            f"suite_content_hash={proof.suite_content_hash!r}) — fail-closed, "
                            "cannot reach PENDING",
                        )
                    )
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
                # Evidence-first: this cycle runs NO prover, so ordinary builders carry
                # no frozen-suite proof (``candidate.proof is None`` -> ``proof=None``,
                # opt-in Tier-2). The one exception is a builder that is *itself* a
                # frozen-suite verification — the GEPA candidate builder, whose
                # run_gepa_optimization already held-out-validated the evolved body
                # (a real run_phase2_comparison). It hands that ProofSummary through on
                # the Candidate so the resulting GEPA_PROMPT proposal carries a proof
                # (the apply engine refuses a proof-less GEPA_PROMPT — its apply
                # re-verifies the held-out check). No prover is called here either way.
                proof=candidate.proof,
                gate_status=gate_status,
                created_at=stamp,
            )
            proposals.append(proposal)
        except Exception as exc:  # noqa: BLE001 - one decision's failure must not torpedo the rest
            skipped.append(
                SkippedDecision(
                    ak,
                    tk,
                    f"errored ({type(exc).__name__}: {exc}) — fail-closed, no proposal",
                )
            )
            continue

    result = CycleResult(
        agent_name=agent.agent_name,
        proposals=tuple(proposals),
        skipped=tuple(skipped),
    )
    return EvidenceCycleResult(result=result, plan=plan)
