"""Candidate builders: map a decided action to a frozen-suite-provable :class:`Candidate`.

The controller's ``candidate_builder`` seam (:class:`ail.loop.controller.CandidateBuilder`)
turns a :class:`~ail.loop.decision_rules.Decision` into a
:class:`~ail.loop.controller.Candidate` — the proposal-facing
:class:`~ail.loop.proposals.ProposedChange` plus the opaque ``prover_input`` the
(injected) prover consumes — or ``None`` (the first-class fail-closed "no candidate
⇒ no proposal" outcome). This module wires the **first real** such path:

* **The token-efficiency skill install.** A ``SKILL_UPDATE`` decision for a
  token-reduction goal becomes a candidate whose ``prover_input`` is the *proven*
  :func:`ail.optimize.lever.token_efficiency_intervention` (the 35.4%-token-reduction
  skill), so it flows through the unchanged frozen-suite prover
  (:func:`ail.jobs.optimization_cycle._default_prover` →
  :func:`ail.optimize.phase2.run_phase2_comparison`) and, if it proves + gates,
  becomes a PENDING proposal a human approves in lane 3.

Every other action kind returns ``None``: an additive ``metric_view`` has no
frozen-suite intervention (an additive read-path leaves the agent's suite behaviour
unchanged), ``gepa_prompt`` bodies come from the separate heavy GEPA run, and
``instruction_update`` / ``revert`` have no wired intervention. Returning ``None``
keeps the cycle honest — it proposes nothing it cannot prove and fabricates no
proof.

**Which decision is the "token-efficiency skill decision"?** Lane A already emits
one: :func:`ail.loop.decision_rules.decide_redundant_read` fires on the dominant,
recurring L0 redundant-read / boilerplate waste pattern and returns a
``SKILL_UPDATE`` decision — its own docstring names it "the read-cache /
context-compaction skill target", which is exactly what the token-efficiency skill
does (avoid re-reading files already read in-session, batch related shell commands,
drop repeated ``cd``/setup boilerplate). So this builder consumes that existing,
evidence-grounded, deterministic decision rather than duplicating the rule. It keys
on the decision's **trigger**: only a ``SKILL_UPDATE`` triggered by that redundant-read
waste signal (:attr:`~ail.loop.proposals.TriggerKind.REDUNDANT_READ_PATTERN`) is built
into the token-efficiency candidate. A ``SKILL_UPDATE`` with any *other* trigger — e.g.
a Lane B ``AGENT_PLANNER`` proposal intending some other skill, or a judge-dimension
trigger — returns ``None`` (fail-closed): the token-efficiency intervention cannot
faithfully prove a change the triggering evidence did not call for, so it declines
rather than misattributing this skill onto that decision. The builder additionally
gates on the goal being **token-reduction** (:func:`is_token_reduction_goal`) so the
frozen-suite ``total_tokens`` proof actually speaks to the goal's objective; the
proof gate downstream is the ultimate guard that the skill helps *this* suite.

**Cost guard (essential).** Building a candidate triggers an expensive real-agent
frozen-suite comparison (many agent sessions, minutes). The builder closes over the
agent's currently-PENDING proposal ids and returns ``None`` when the would-be
proposal id is already pending (from a prior firing) or was already built this cycle
— so the proof runs at most once per open proposal, never re-proving the same known
result every hourly firing. When the pending set is unavailable (the entrypoint's
read failed — surfaced as ``None``), the builder skips everything: fail-closed
toward **not** spending, never re-proving blindly.
"""

from __future__ import annotations

import difflib
from collections.abc import Collection

from ail.goals.compiler import CompiledGoal
from ail.loop.controller import Candidate, CandidateBuilder
from ail.loop.decision_rules import Decision
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProposedChange,
    TriggerKind,
    derive_proposal_id,
)
from ail.optimize.lever import token_efficiency_intervention, token_efficiency_skill
from ail.registry import Agent

__all__ = [
    "TOKEN_REDUCTION_METRIC",
    "is_token_reduction_goal",
    "token_efficiency_skill_change",
    "token_efficiency_candidate_builder",
    "evidence_candidate_builder",
]

#: The deterministic headline objective the token-efficiency skill is proven against
#: (the frozen-suite Phase-2 metric, :data:`ail.optimize.phase2._TOKEN_METRIC`). The
#: proven skill reduces this metric; proposing it for any *other* objective would
#: attach a ``total_tokens`` proof to a goal it does not speak to — so the builder
#: gates on this.
TOKEN_REDUCTION_METRIC = "total_tokens"


def is_token_reduction_goal(goal: CompiledGoal) -> bool:
    """Whether ``goal`` is the token-reduction objective the skill is proven against.

    ``minimize total_tokens`` — the frozen-suite Phase-2 objective the token-efficiency
    skill's WITH/WITHOUT proof measures. Any other objective/direction returns
    ``False`` so the builder does not attach a ``total_tokens`` proof to a goal it
    cannot speak to.
    """
    return goal.objective_metric == TOKEN_REDUCTION_METRIC and goal.direction == "minimize"


def token_efficiency_skill_change() -> ProposedChange:
    """The concrete ``SKILL_UPDATE`` change: install the token-efficiency skill.

    Installing the skill means appending its instruction section (the exact text the
    proven :class:`~ail.optimize.lever.SkillInjectionIntervention` injects into a
    candidate run) to the agent's champion prompt. The change is a
    :class:`~ail.loop.proposals.ChangeKind.SKILL_DIFF` carrying the canonical
    unified-diff *addition* of that section, so a human reviewer sees precisely what
    would be installed and lane 3b reconstructs the full body server-side from the
    current champion + this diff (fail-closing on a stale champion — the controller
    here only proposes).

    Deterministic: the skill is a static package asset, so the diff — and therefore
    :func:`~ail.loop.proposals.derive_proposal_id` over this change — is stable across
    cycles/processes. That stability is what the cost guard keys on: a re-run that
    re-decides the same install yields the *same* proposal id, so an already-pending
    proposal is recognised and not re-proven.
    """
    skill = token_efficiency_skill()
    section_lines = skill.as_system_prompt_section().splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            [],
            section_lines,
            fromfile="champion_prompt",
            tofile=f"champion_prompt+{skill.slug}",
            lineterm="",
        )
    )
    return ProposedChange(
        kind=ChangeKind.SKILL_DIFF,
        summary=(
            f"Install the token-efficiency skill {skill.name!r} (avoid re-reading files "
            "already read in-session, batch related shell commands, drop repeated cd/setup "
            "boilerplate) by appending its instruction section to the agent's champion prompt."
        ),
        diff=diff,
    )


def token_efficiency_candidate_builder(
    *,
    pending_proposal_ids: Collection[str] | None,
) -> CandidateBuilder:
    """Build the cost-guarded token-efficiency candidate builder.

    Returns a :class:`ail.loop.controller.CandidateBuilder` that maps a
    ``SKILL_UPDATE`` decision for a token-reduction goal to a :class:`Candidate`
    carrying the proven :func:`ail.optimize.lever.token_efficiency_intervention`, and
    ``None`` for every other action kind / goal (fail-closed). ``pending_proposal_ids``
    is the agent's currently-PENDING proposal ids (fetched once by the entrypoint);
    ``None`` means the pending check could not run, in which case the builder skips
    everything (fail-closed toward **not** spending on an expensive proof).
    """
    # Fail-closed toward NOT spending: the pending-proposal check could not run, so
    # build no candidate rather than re-prove an expensive frozen-suite comparison
    # blindly. (A distinct empty set — table present, no pending rows — falls through
    # to the real builder so the first proposal can be built.)
    if pending_proposal_ids is None:

        def _skip(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
            return None

        return _skip

    # Copy into a mutable set so we also guard against building the SAME candidate
    # twice within one cycle: two SKILL_UPDATE decisions (e.g. Lane A's redundant-read
    # rule and a Lane B planner proposal that survived de-dup) collapse to one proof.
    seen: set[str] = set(pending_proposal_ids)

    def _build(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        # The only frozen-suite-provable intervention wired today is the token-efficiency
        # skill install. metric_view (additive, no intervention), gepa_prompt (heavy GEPA
        # run), instruction_update and revert have no wired prover input -> None.
        if decision.action_kind is not ActionKind.SKILL_UPDATE:
            return None
        # Build the token-efficiency candidate ONLY for the genuine redundant-read waste
        # signal that skill actually addresses. A SKILL_UPDATE with any other trigger
        # (e.g. a Lane B AGENT_PLANNER proposal intending some *other* skill, or a
        # judge-dimension trigger) is NOT something the token-efficiency intervention
        # can faithfully prove — mapping it here would prove/propose this skill in place
        # of the one the evidence called for (a misattribution). Decline, fail-closed.
        if decision.trigger.kind is not TriggerKind.REDUNDANT_READ_PATTERN:
            return None
        # Only propose it for the objective it is proven against, so the frozen-suite
        # total_tokens proof actually speaks to the goal.
        if not is_token_reduction_goal(goal):
            return None

        change = token_efficiency_skill_change()
        proposal_id = derive_proposal_id(
            agent_name=agent.agent_name,
            action_kind=ActionKind.SKILL_UPDATE,
            change=change,
        )
        # Cost guard: the same install is already pending (a prior firing) or was
        # already built this cycle -> skip, so the expensive proof runs at most once
        # per open proposal.
        if proposal_id in seen:
            return None
        seen.add(proposal_id)
        return Candidate(change=change, prover_input=token_efficiency_intervention())

    return _build


def evidence_candidate_builder() -> CandidateBuilder:
    """Build the **evidence-first** token-efficiency candidate builder (no prove).

    The counterpart of :func:`token_efficiency_candidate_builder` for the
    evidence-first lane (:func:`ail.loop.evidence_cycle.run_evidence_cycle`, per
    ``docs/PRODUCT_ARCHITECTURE.md`` §3/§7: the planner does **not** prove — proving
    is opt-in Tier-2). It maps the *same* decision — a ``SKILL_UPDATE`` triggered by
    the dominant L0 redundant-read waste pattern
    (:attr:`~ail.loop.proposals.TriggerKind.REDUNDANT_READ_PATTERN`) on a
    token-reduction goal — to the *same* concrete change
    (:func:`token_efficiency_skill_change`), and declines every other action kind /
    trigger / goal exactly as the proving builder does (fail-closed: no fabricated
    candidate for a change the triggering evidence did not call for).

    Two deliberate differences from :func:`token_efficiency_candidate_builder`:

    * **No frozen-suite cost guard.** That builder skips (re-)building when the
      proposal is already pending because building triggers an expensive real-agent
      proof; the evidence lane never proves, so there is nothing costly to guard and
      no ``pending_proposal_ids`` to thread. Idempotency is preserved downstream by
      the agent-scoped atomic ``REPLACE`` in
      :func:`ail.loop.publish_proposals.publish_agent_proposals` (a re-decided install
      hashes to the same :func:`~ail.loop.proposals.derive_proposal_id`, so it
      replaces its own row rather than duplicating).
    * The carried ``prover_input`` (the proven
      :func:`ail.optimize.lever.token_efficiency_intervention`) is **not** consumed by
      this lane; it travels so a *later*, opt-in Tier-2 "verify on my suite" run has a
      concrete target to prove against without re-deriving it.
    """

    def _build(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        if decision.action_kind is not ActionKind.SKILL_UPDATE:
            return None
        if decision.trigger.kind is not TriggerKind.REDUNDANT_READ_PATTERN:
            return None
        if not is_token_reduction_goal(goal):
            return None
        return Candidate(
            change=token_efficiency_skill_change(),
            prover_input=token_efficiency_intervention(),
        )

    return _build
