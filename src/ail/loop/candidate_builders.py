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
import hashlib
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ail.goals.compiler import CompiledGoal
from ail.loop.controller import Candidate, CandidateBuilder
from ail.loop.decision_rules import Decision
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProofSummary,
    ProposedChange,
    TriggerKind,
    derive_proposal_id,
)
from ail.optimize.lever import token_efficiency_intervention, token_efficiency_skill
from ail.optimize.prompt_registry import candidate_improvement
from ail.registry import Agent

if TYPE_CHECKING:
    from ail.optimize.gepa_runner import GepaOptimizationResult

__all__ = [
    "TOKEN_REDUCTION_METRIC",
    "is_token_reduction_goal",
    "token_efficiency_skill_change",
    "token_efficiency_candidate_builder",
    "evidence_candidate_builder",
    # Pluggable dispatch (piece 1)
    "registry_candidate_builder",
    "chain_candidate_builders",
    # Generic agent-authored quick-edit (piece 3)
    "ChampionBodyResolver",
    "SkillEditor",
    "agent_skill_edit_builder",
    # Generic GEPA optimization builder (piece 2)
    "GepaSeed",
    "GepaSeedResolver",
    "GepaRunFn",
    "gepa_target_key",
    "gepa_target_objective",
    "gepa_candidate_builder",
    # Cost-aware routing (piece 4)
    "GepaCostPolicy",
    "gepa_cost_gate",
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


# ===========================================================================
# Piece 1 — the pluggable candidate-builder registry (dispatch by ActionKind)
# ===========================================================================


def registry_candidate_builder(
    builders: Mapping[ActionKind, CandidateBuilder],
) -> CandidateBuilder:
    """A :class:`~ail.loop.controller.CandidateBuilder` that dispatches by action kind.

    The single seam the loop hands to :func:`ail.loop.controller.run_cycle` /
    :func:`ail.loop.evidence_cycle.run_evidence_cycle`, so *which* builder handles a
    decision is data, not a hardwired ``if action_kind is …`` ladder. It looks up
    ``decision.action_kind`` in ``builders`` and delegates to the registered builder
    (which may itself decline and return ``None``, e.g. on a trigger it does not
    serve); an action kind with **no** registered builder returns ``None`` — the
    first-class fail-closed "no candidate ⇒ no proposal" outcome the controller
    records as a :class:`~ail.loop.controller.SkippedDecision`. The controller
    contract is preserved exactly: this *is* a ``CandidateBuilder``, so nothing about
    proving, gating, or propose-only emission changes.

    A production wiring registers the existing, unregressed token-efficiency /
    evidence builders under :attr:`~ail.loop.proposals.ActionKind.SKILL_UPDATE`
    (typically :func:`chain_candidate_builders`-ed ahead of the generic quick-edit
    builder) and the GEPA builder under
    :attr:`~ail.loop.proposals.ActionKind.GEPA_PROMPT`; every unregistered kind
    (``METRIC_VIEW``, ``INSTRUCTION_UPDATE``, ``REVERT``, ``AGENT_TASK``) falls
    through to the fail-closed ``None``.
    """
    table: dict[ActionKind, CandidateBuilder] = dict(builders)

    def _dispatch(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        builder = table.get(decision.action_kind)
        if builder is None:
            return None
        return builder(decision, goal=goal, agent=agent)

    return _dispatch


def chain_candidate_builders(*builders: CandidateBuilder) -> CandidateBuilder:
    """Compose builders that serve the *same* action kind: first non-``None`` wins.

    Several builders can legitimately serve one action kind by *trigger*: under
    :attr:`~ail.loop.proposals.ActionKind.SKILL_UPDATE` the proven token-efficiency
    builder handles the dominant
    :attr:`~ail.loop.proposals.TriggerKind.REDUNDANT_READ_PATTERN` waste signal, while
    the generic agent-authored quick-edit builder (:func:`agent_skill_edit_builder`)
    handles every *other* SKILL_UPDATE trigger/dimension. This tries each builder in
    order and returns the first :class:`~ail.loop.controller.Candidate` produced —
    so the specific, cost-guarded/proven builder is consulted first and the generic
    one only fills the gap it declined — or ``None`` if all decline (fail-closed).
    """
    chain = tuple(builders)

    def _chain(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        for builder in chain:
            candidate = builder(decision, goal=goal, agent=agent)
            if candidate is not None:
                return candidate
        return None

    return _chain


# ===========================================================================
# Piece 3 — the generic agent-authored quick-edit builder (SKILL_UPDATE)
# ===========================================================================


class ChampionBodyResolver(Protocol):
    """Resolve the **current champion body** a decision would change, or ``None``.

    The injectable read seam behind the generic builders: given a decision (its
    trigger names the dimension/target), return the agent's current champion skill /
    prompt body — the text a quick edit revises, or a GEPA run seeds from. Production
    reads it from the prompt registry (the champion alias) or the agent's skill file;
    tests inject a canned body. Returning ``None`` (or empty) is the first-class
    fail-closed outcome: there is no champion body to change, so the builder produces
    no candidate rather than editing a fabricated one.
    """

    def __call__(self, decision: Decision, *, goal: CompiledGoal, agent: Agent) -> str | None: ...


class SkillEditor(Protocol):
    """Author a **small skill edit** of ``current_body`` toward the goal, or decline.

    The injectable authoring seam: the local companion agent (Claude Agent SDK)
    reads the current champion body plus the decision's evidence (which
    dimension/target fell short) and returns a *revised* body — a lightweight,
    targeted edit, **generic across any skill/judge dimension** (never the fixed
    token-efficiency install). It returns ``None`` to **decline** (it saw no
    worthwhile edit), and the builder fails closed on a decline or on an edit that
    does not actually change the body. Tests inject a deterministic editor; nothing
    here calls a live model.
    """

    def __call__(
        self, *, current_body: str, decision: Decision, goal: CompiledGoal, agent: Agent
    ) -> str | None: ...


def _unified_body_diff(before: str, after: str, *, fromfile: str, tofile: str) -> str:
    """A stable unified diff between two bodies (same shape as the token-eff install).

    Splits on lines (no keepends) with ``lineterm=""`` so the rendered diff is
    deterministic and matches :func:`token_efficiency_skill_change`'s form — a human
    reviewer sees exactly what changes, and lane-3b reconstructs the full body
    server-side from the current champion + this diff.
    """
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )


def agent_skill_edit_builder(
    *,
    editor: SkillEditor,
    body_resolver: ChampionBodyResolver,
) -> CandidateBuilder:
    """Build a ``SKILL_UPDATE`` candidate from an **agent-authored** small skill edit.

    The generic, dimension-agnostic counterpart to the token-efficiency install: for
    a ``SKILL_UPDATE`` decision (whatever its trigger/dimension), it resolves the
    current champion body (``body_resolver``), asks the agent to author a targeted
    edit toward the goal (``editor``), and packages the *real* change as a
    :class:`~ail.loop.proposals.ChangeKind.SKILL_DIFF` :class:`Candidate` (unified
    diff of current → edited body). Its proof is ``None``: a skill update is an
    evidence-only-applyable kind (the apply engine reconstructs + registers the body
    on human approval, re-running the gate — no frozen-suite proof required).

    Fail-closed (no fabricated candidate) at every step: not a ``SKILL_UPDATE`` → the
    builder is not for this decision; no champion body to edit → ``None``; the editor
    declines (``None``) → ``None``; the edit does not change the body (identical after
    strip, or an empty diff) → ``None``. Chaining (:func:`chain_candidate_builders`)
    places this **after** the proven token-efficiency builder so the specific waste
    signal keeps its proven skill and only *other* SKILL_UPDATE dimensions reach here.
    """

    def _build(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        if decision.action_kind is not ActionKind.SKILL_UPDATE:
            return None
        current = body_resolver(decision, goal=goal, agent=agent)
        if current is None or not current.strip():
            return None
        edited = editor(current_body=current, decision=decision, goal=goal, agent=agent)
        if edited is None or not edited.strip():
            return None
        # A real edit, not a no-op: reject an unchanged body (identical after strip).
        if edited.strip() == current.strip():
            return None
        diff = _unified_body_diff(
            current, edited, fromfile="champion_skill", tofile="champion_skill+agent_edit"
        )
        # Defensive: an empty/whitespace-only diff is no change to review — fail closed.
        if not diff.strip():
            return None
        change = ProposedChange(
            kind=ChangeKind.SKILL_DIFF,
            summary=(
                f"Agent-authored skill edit toward goal {goal.objective_metric!r} "
                f"(direction {goal.direction}): {decision.trigger.summary}"
            ),
            diff=diff,
        )
        return Candidate(change=change, prover_input=None, proof=None)

    return _build


# ===========================================================================
# Piece 2 — the generic GEPA optimization candidate builder (GEPA_PROMPT)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class GepaSeed:
    """The resolved GEPA target: *which* champion artifact to evolve + its seed body.

    Generic, **not hardcoded to token_efficiency**: ``target_key``
    (:func:`gepa_target_key`) is the artifact's stable identity — derived from the
    decision's evidence (the judged dimension / metric it fired on), used for
    provenance, the candidate-artifact filename, and the GEPA component name.
    ``seed_body`` is the current champion skill/prompt body GEPA starts evolving from
    (resolved from the prompt registry / champion alias by the injected
    :class:`GepaSeedResolver`). What *fitness* GEPA climbs is owned by the injected
    :class:`GepaRunFn` (it configures :func:`ail.optimize.gepa_runner.run_gepa_optimization`'s
    frozen-suite objective), so this seed carries only the target, keeping the builder
    agnostic to how the run is scored.
    """

    target_key: str
    seed_body: str


class GepaSeedResolver(Protocol):
    """Resolve the champion artifact a GEPA_PROMPT decision should evolve, or ``None``.

    The generic-target seam: maps a decision (its trigger names the judged dimension /
    metric) + goal + agent to the :class:`GepaSeed` GEPA seeds from — reading the
    relevant champion body from the prompt registry / skill file. Returns ``None``
    (fail-closed) when no champion target is resolvable, so the builder proposes
    nothing rather than evolving a fabricated seed. Tests inject a canned seed; the
    production resolver reads the registry (and is the single place that decides
    *which* artifact to evolve — never a hardcoded token-efficiency default).
    """

    def __call__(
        self, decision: Decision, *, goal: CompiledGoal, agent: Agent
    ) -> GepaSeed | None: ...


class GepaRunFn(Protocol):
    """Run a **local** GEPA optimization for a resolved seed, or ``None`` fail-closed.

    The single expensive seam, wrapping
    :func:`ail.optimize.gepa_runner.run_gepa_optimization` bound to the deployer's
    frozen Task Suite + live local agent adapter (many real agent sessions, minutes).
    It returns the :class:`~ail.optimize.gepa_runner.GepaOptimizationResult` — the
    evolved body **already held-out-validated** against the seed on a disjoint split
    (the anti-overfit wall) — or ``None`` when GEPA **cannot run locally**: no frozen
    suite present, no local Claude/agent available, or the run otherwise could not
    produce a result. ``None`` is the fail-closed signal (no fabricated candidate);
    tests inject a fake that returns a canned result (or ``None``) so nothing runs
    live GEPA, a live agent arm, or a live model.
    """

    def __call__(
        self, seed: GepaSeed, *, decision: Decision, goal: CompiledGoal, agent: Agent
    ) -> GepaOptimizationResult | None: ...


def gepa_target_key(decision: Decision, *, goal: CompiledGoal) -> str:
    """The stable, **generic** identity of the artifact a GEPA_PROMPT decision evolves.

    Resolved from the decision's own evidence, never hardcoded: the judged dimension's
    judge (``judge:<judge_name>``) when the trigger rests on a judge — the dominant
    GEPA case (a trusted judge dimension below goal, Lane A
    :func:`ail.loop.decision_rules.decide_judge_dimension`, or a Lane B planner
    GEPA proposal for a judge) — else the targeted ``metric:<metric>``, else the goal's
    own ``goal:<objective_metric>``. Used for provenance, the candidate-artifact
    filename, and the GEPA component name, so two decisions about the same target
    resolve to the same key.
    """
    t = decision.trigger
    if t.judge_name:
        return f"judge:{t.judge_name}"
    if t.metric:
        return f"metric:{t.metric}"
    return f"goal:{goal.objective_metric}"


def gepa_target_objective(decision: Decision, *, goal: CompiledGoal) -> str:
    """The objective metric a GEPA proof MUST speak to for the routed target.

    The companion of :func:`gepa_target_key`: where that returns the routed target's
    stable *identity* (``judge:<name>`` / ``metric:<metric>`` / ``goal:<obj>``), this
    returns — in the **same** resolution order — the objective metric a genuine proof
    for that target must carry: the judged dimension (``judge_name``) when the trigger
    rests on a judge, else the targeted ``metric``, else the goal's own objective.

    A GEPA candidate is honest only when the objective it *actually proved*
    (:attr:`~ail.loop.proposals.ProofSummary.objective_metric`, sourced from the
    held-out :class:`~ail.optimize.phase2.Phase2Artifact`) equals this. Otherwise the
    proposal would attach a proof for one dimension to a target routed for another —
    e.g. a ``total_tokens`` savings proof onto a ``judge:modularity`` target — which is
    exactly the mislabeled-proof path :func:`gepa_candidate_builder` fails closed on.
    """
    t = decision.trigger
    if t.judge_name:
        return t.judge_name
    if t.metric:
        return t.metric
    return goal.objective_metric


def _write_gepa_candidate_artifact(
    result: GepaOptimizationResult, *, root: Path, target_key: str
) -> str:
    """Persist ``result`` as the ``gepa_candidate*.json`` the apply engine consumes.

    Writes the full :class:`~ail.optimize.gepa_runner.GepaOptimizationResult` JSON to a
    real file under ``root`` and returns its path — the ``evolved_body_ref`` the
    proposal carries. :func:`ail.loop.apply._apply_gepa_prompt` reads this file back
    (:func:`ail.optimize.prompt_registry.register_gepa_candidate` →
    ``GepaOptimizationResult.model_validate_json``) and **re-runs** the held-out
    improvement check at apply time, so the on-disk artifact is the source of truth.
    The filename embeds a sanitized ``target_key`` (provenance) and a content digest
    (a re-decided identical run overwrites its own file rather than proliferating).
    """
    root.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump_json()
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", target_key).strip("-") or "gepa"
    path = root / f"gepa_candidate_{slug}_{digest}.json"
    path.write_text(payload, encoding="utf-8")
    return str(path)


def gepa_candidate_builder(
    *,
    gepa_run: GepaRunFn,
    seed_resolver: GepaSeedResolver,
    artifacts_root: str | Path,
) -> CandidateBuilder:
    """Build a ``GEPA_PROMPT`` candidate from a **local, self-proving** GEPA run.

    For a ``GEPA_PROMPT`` decision it resolves the champion artifact to evolve
    (``seed_resolver`` — generic, not token-efficiency-bound), runs GEPA **locally**
    on it (``gepa_run``, wrapping
    :func:`ail.optimize.gepa_runner.run_gepa_optimization`), and packages the result
    as an :class:`~ail.loop.proposals.ChangeKind.EVOLVED_BODY_REF` :class:`Candidate`.
    A GEPA candidate is **self-proving**: ``run_gepa_optimization`` already
    held-out-validated the evolved body against the seed on a disjoint split, so the
    candidate carries a **pre-computed** :class:`~ail.loop.proposals.ProofSummary`
    (:attr:`Candidate.proof`). That proof is what lets a GEPA_PROMPT proposal flow
    through the evidence-first companion (which runs no prover) *and* clear the apply
    engine, which refuses a proof-less GEPA_PROMPT (its apply re-verifies the held-out
    check). The ``evolved_body_ref`` points at a written ``gepa_candidate*.json`` that
    :func:`ail.optimize.prompt_registry.register_gepa_candidate` consumes on approval.

    Fail-closed everywhere (no fabricated **or mislabeled** candidate): not a
    ``GEPA_PROMPT`` decision, no resolvable seed, the resolved seed does not match the
    decision's routed target (:func:`gepa_target_key`), ``gepa_run`` returned ``None``
    (no frozen suite / no local Claude), the run did not change the body
    (``changed=False``), the run evolved a *different* seed than the one resolved, the
    held-out result did not beat the seed
    (:func:`ail.optimize.prompt_registry.candidate_improvement`), the held-out proof
    does not both prove an improvement and hold correctness, **or the proof's objective
    does not match the routed target dimension** — each returns ``None``.

    **Honest genericity (the routed-objective gate).** The builder routes *any* target
    (:func:`gepa_target_key`), but it refuses to emit a proposal whose proof does not
    genuinely correspond to that routed dimension: it requires
    ``proof.objective_metric == gepa_target_objective(decision, goal)``. Because
    :func:`ail.optimize.gepa_runner.run_gepa_optimization`'s evaluator today proves
    ONLY the token/efficiency objective, a routed dimension GEPA cannot yet prove (a
    ``judge:<quality>`` like modularity/correctness) FAILS CLOSED here rather than
    shipping a token-savings proof mislabeled as a quality improvement. The token/
    efficiency-objective path — where the proof genuinely matches — is emitted as
    before. **FOLLOW-ON** for full "optimize anything": generalize GEPA's evaluator to
    score a candidate against an arbitrary routed judge on the held-out split; that
    body of work is what lets a quality dimension pass this gate honestly.
    """
    root = Path(artifacts_root)

    def _build(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        if decision.action_kind is not ActionKind.GEPA_PROMPT:
            return None
        seed = seed_resolver(decision, goal=goal, agent=agent)
        if seed is None or not seed.seed_body.strip():
            return None
        # The expensive, self-proving run. None => GEPA could not run locally
        # (no frozen suite / no local Claude) => fail-closed, no candidate.
        result = gepa_run(seed, decision=decision, goal=goal, agent=agent)
        if result is None:
            return None
        # A no-op evolution proves nothing to ship.
        if not result.changed:
            return None
        # The honest anti-overfit gate the apply path re-runs: the evolved body must
        # have beaten the seed on the held-out split GEPA never trained on.
        improving, _reason = candidate_improvement(result)
        if not improving:
            return None
        artifact = result.holdout_evolved
        if artifact is None:  # defensive: candidate_improvement already required it
            return None
        # The pre-computed proof carried onto the proposal. Only surface a GEPA_PROMPT
        # the apply engine will accept: a real proved improvement with correctness held.
        proof = ProofSummary.from_phase2_artifact(artifact)
        if not (proof.proved_improvement and proof.correctness_held):
            return None
        # BLOCKING: the proof must genuinely correspond to the ROUTED target dimension.
        # run_gepa_optimization's evaluator today proves ONLY the token/efficiency
        # objective (GepaConfig.objective_metric -> the held-out Phase2Artifact's
        # objective_metric), so proof.objective_metric is whatever GEPA actually
        # optimized. If the routed target is a dimension GEPA cannot yet prove — e.g. a
        # judge:<quality> like modularity/correctness, whose routed objective is that
        # judge name, not total_tokens — the proof would NOT match, so we FAIL CLOSED
        # here rather than emit a mislabeled token-savings proof onto a quality target.
        # FOLLOW-ON: generalizing GEPA's evaluator to score candidates against an
        # arbitrary routed judge on the held-out split (full "optimize anything") is what
        # lets those dimensions pass this gate honestly; until then only a target whose
        # objective GEPA genuinely proved (the token/efficiency path) is emitted.
        if proof.objective_metric != gepa_target_objective(decision, goal=goal):
            return None
        ref = _write_gepa_candidate_artifact(result, root=root, target_key=seed.target_key)
        change = ProposedChange(
            kind=ChangeKind.EVOLVED_BODY_REF,
            summary=(
                f"GEPA-evolved champion body for {seed.target_key} (toward goal "
                f"{goal.objective_metric!r}); held-out realized-savings delta "
                f"{result.holdout_savings_delta_pct} pct-pts beats seed."
            ),
            evolved_body_ref=ref,
        )
        return Candidate(change=change, prover_input=None, proof=proof)

    return _build


# ===========================================================================
# Piece 4 — cost-aware routing: keep the expensive GEPA run the exception
# ===========================================================================


@dataclass(frozen=True, slots=True)
class GepaCostPolicy:
    """The deterministic guard that keeps a GEPA run the **exception**, not the default.

    A GEPA run is expensive — real agent arms, minutes, the whole frozen suite — so it
    must never fire on a trivial or every-cycle signal. The routing has two layers,
    and this is the *floor* under the second:

    * **Prefer the LLM planner's judgment.** Lane B (:mod:`ail.loop.planner`) already
      decides *what to try* and can propose a ``GEPA_PROMPT`` when a judged dimension
      warrants a full optimization; Lane A's :func:`ail.loop.decision_rules.decide_judge_dimension`
      likewise emits ``GEPA_PROMPT`` only for a **trusted** judge dimension below the
      goal's own threshold. Routing GEPA is thus primarily their call.
    * **A deterministic backstop.** This policy is a cheap, explainable gate that runs
      *before* the expensive builder, so GEPA cannot fire on a trivial signal even if
      one reaches the GEPA slot. It permits a ``GEPA_PROMPT`` decision only when **all**
      hold: its trigger is a GEPA-warranting kind (:attr:`allowed_triggers` — the
      trusted-judge Lane-A rule or a Lane-B planner proposal, never a redundant-read /
      asset signal); it rests on a **judged dimension** (``trigger.judge_name`` is set)
      when :attr:`require_judge_dimension`; and the signal is **persistent** —
      recurring across at least :attr:`min_trace_recurrence` traces (a one-off is not
      worth a GEPA run). Anything else is declined (the expensive builder is never
      even consulted).

    Every bar is a visible, adjustable field (mirroring
    :class:`ail.loop.decision_rules.DecisionThresholds`), never a constant buried in a
    function body.
    """

    min_trace_recurrence: int = 3
    require_judge_dimension: bool = True
    allowed_triggers: frozenset[TriggerKind] = frozenset(
        {TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD, TriggerKind.AGENT_PLANNER}
    )

    def permits(self, decision: Decision) -> tuple[bool, str]:
        """Whether ``decision`` warrants an (expensive) GEPA run, with a reason.

        Returns ``(permitted, reason)``; ``permitted`` is ``True`` only when the
        decision clears every bar above. The ``reason`` explains a decline so a
        reviewer sees *why* GEPA was (or was not) warranted — the same auditable-skip
        discipline the controller applies.
        """
        t = decision.trigger
        if t.kind not in self.allowed_triggers:
            allowed = ", ".join(sorted(k.value for k in self.allowed_triggers))
            return (
                False,
                f"trigger {t.kind.value!r} is not a GEPA-warranting signal (allowed: {allowed}); "
                "GEPA is the exception, not fired on every signal",
            )
        if self.require_judge_dimension and not t.judge_name:
            return (
                False,
                "GEPA only fires on a judged-dimension signal, but the trigger names no judge "
                "(judge_name is unset) — declining the expensive run",
            )
        if t.n_traces < self.min_trace_recurrence:
            return (
                False,
                f"signal recurs on only {t.n_traces} trace(s) (< {self.min_trace_recurrence}); "
                "not persistent enough to warrant an expensive GEPA run",
            )
        return (
            True,
            f"trusted judged-dimension signal persistent across {t.n_traces} trace(s) — "
            "GEPA run warranted",
        )


def gepa_cost_gate(
    builder: CandidateBuilder, *, policy: GepaCostPolicy | None = None
) -> CandidateBuilder:
    """Wrap the (expensive) GEPA builder in the deterministic :class:`GepaCostPolicy`.

    Returns a :class:`~ail.loop.controller.CandidateBuilder` that, for a
    ``GEPA_PROMPT`` decision, consults ``policy`` (default :class:`GepaCostPolicy`)
    **before** the wrapped ``builder`` runs: if the policy declines, it returns
    ``None`` (a fail-closed :class:`~ail.loop.controller.SkippedDecision` upstream) and
    the expensive GEPA run is **never invoked**; only a permitted decision reaches
    ``builder``. Every non-``GEPA_PROMPT`` decision passes straight through to
    ``builder`` untouched (the gate guards GEPA cost only). Registered under
    :attr:`~ail.loop.proposals.ActionKind.GEPA_PROMPT` in
    :func:`registry_candidate_builder`, this is what makes GEPA the exception rather
    than firing on every judged-dimension blip.
    """
    pol = policy or GepaCostPolicy()

    def _gated(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        # The gate guards the expensive GEPA path only; anything else is not its concern.
        if decision.action_kind is not ActionKind.GEPA_PROMPT:
            return builder(decision, goal=goal, agent=agent)
        permitted, _reason = pol.permits(decision)
        if not permitted:
            # Deterministic cost guard: GEPA not warranted — decline BEFORE the expensive
            # run (the wrapped builder is never consulted). Fail-closed, no candidate.
            return None
        return builder(decision, goal=goal, agent=agent)

    return _gated
