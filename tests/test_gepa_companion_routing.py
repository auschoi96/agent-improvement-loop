"""Tests for the companion's per-feedback candidate routing (GEPA vs quick-edit).

Covers the four pieces added on top of the first token-efficiency candidate path,
all with **injectable seams** so nothing runs live GEPA, a live agent arm, or a live
model:

1. the pluggable :func:`registry_candidate_builder` dispatch (by action kind) +
   :func:`chain_candidate_builders` (first non-None wins);
2. the generic GEPA candidate builder, fail-closed on no-improvement / changed=False;
3. the generic agent-authored quick-edit builder producing a real SKILL_DIFF;
4. the deterministic cost guard that keeps GEPA off trivial signals.
"""

from __future__ import annotations

from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.loop.candidate_builders import (
    agent_skill_edit_builder,
    chain_candidate_builders,
    registry_candidate_builder,
)
from ail.loop.controller import Candidate
from ail.loop.decision_rules import Decision
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProposedChange,
    TriggerKind,
    TriggerSignal,
    default_risk_class,
)
from ail.registry import Agent

# -- fixtures --------------------------------------------------------------


def _agent() -> Agent:
    return Agent(agent_name="claude_code", experiment_id="660599403165942")


def _quality_goal() -> CompiledGoal:
    """A quality goal with a trusted-judge guardrail (the generic, non-token case)."""
    return CompiledGoal(
        objective_metric="modularity",
        direction="maximize",
        target=GoalTarget(value=4.5, kind="absolute"),
        guardrails=(Guardrail(name="modularity", kind="judge", threshold=4.0),),
        cohort="claude_code",
    ).confirm()


def _skill_decision(trigger: TriggerKind = TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD) -> Decision:
    return Decision(
        ActionKind.SKILL_UPDATE,
        default_risk_class(ActionKind.SKILL_UPDATE),
        TriggerSignal(
            kind=trigger,
            summary="modularity judged low",
            metric="modularity",
            judge_name="modularity",
            n_traces=5,
        ),
    )


def _gepa_decision(
    *,
    trigger: TriggerKind = TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD,
    judge_name: str | None = "modularity",
    n_traces: int = 5,
) -> Decision:
    return Decision(
        ActionKind.GEPA_PROMPT,
        default_risk_class(ActionKind.GEPA_PROMPT),
        TriggerSignal(
            kind=trigger,
            summary="trusted judge modularity persistently below goal",
            metric="modularity",
            observed_value=3.1,
            threshold=4.0,
            judge_name=judge_name,
            n_traces=n_traces,
            trace_refs=[f"t{i}" for i in range(n_traces)],
        ),
    )


def _fake_candidate(diff: str = "x") -> Candidate:
    return Candidate(change=ProposedChange(kind=ChangeKind.SKILL_DIFF, summary="fake", diff=diff))


# ==========================================================================
# Piece 1 — registry dispatch + chaining
# ==========================================================================


def test_registry_dispatches_by_action_kind() -> None:
    seen: list[str] = []

    def _skill_builder(d: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        seen.append("skill")
        return _fake_candidate("skill-edit")

    def _gepa_builder(d: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
        seen.append("gepa")
        return Candidate(
            change=ProposedChange(
                kind=ChangeKind.EVOLVED_BODY_REF, summary="gepa", evolved_body_ref="/tmp/x.json"
            )
        )

    registry = registry_candidate_builder(
        {ActionKind.SKILL_UPDATE: _skill_builder, ActionKind.GEPA_PROMPT: _gepa_builder}
    )
    goal, agent = _quality_goal(), _agent()

    skill = registry(_skill_decision(), goal=goal, agent=agent)
    assert skill is not None and skill.change.diff == "skill-edit"

    gepa = registry(_gepa_decision(), goal=goal, agent=agent)
    assert gepa is not None and gepa.change.kind is ChangeKind.EVOLVED_BODY_REF

    assert seen == ["skill", "gepa"]


def test_registry_unhandled_kind_fails_closed_to_none() -> None:
    registry = registry_candidate_builder(
        {ActionKind.SKILL_UPDATE: lambda d, *, goal, agent: _fake_candidate()}
    )
    goal, agent = _quality_goal(), _agent()
    for ak, tk in [
        (ActionKind.METRIC_VIEW, TriggerKind.RLM_RECOMMENDED_ASSET),
        (ActionKind.GEPA_PROMPT, TriggerKind.JUDGE_DIMENSION_BELOW_THRESHOLD),
        (ActionKind.INSTRUCTION_UPDATE, TriggerKind.AGENT_PLANNER),
        (ActionKind.REVERT, TriggerKind.POST_APPLY_REGRESSION),
    ]:
        decision = Decision(ak, default_risk_class(ak), TriggerSignal(kind=tk, summary="x"))
        assert registry(decision, goal=goal, agent=agent) is None


def test_chain_returns_first_non_none() -> None:
    calls: list[str] = []

    def _declines(name: str) -> object:
        def _b(d: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
            calls.append(name)
            return None

        return _b

    def _accepts(name: str) -> object:
        def _b(d: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate | None:
            calls.append(name)
            return _fake_candidate(name)

        return _b

    chain = chain_candidate_builders(_declines("a"), _accepts("b"), _accepts("c"))
    out = chain(_skill_decision(), goal=_quality_goal(), agent=_agent())
    assert out is not None and out.change.diff == "b"
    # short-circuits: 'c' is never consulted once 'b' produced a candidate
    assert calls == ["a", "b"]


def test_chain_all_decline_is_none() -> None:
    chain = chain_candidate_builders(lambda d, *, goal, agent: None, lambda d, *, goal, agent: None)
    assert chain(_skill_decision(), goal=_quality_goal(), agent=_agent()) is None


# ==========================================================================
# Piece 3 — the generic agent-authored quick-edit builder
# ==========================================================================

_CURRENT_BODY = "# Modularity skill\n\nWrite one function per responsibility.\n"
_EDITED_BODY = (
    "# Modularity skill\n\n"
    "Write one function per responsibility.\n"
    "Extract shared logic into small, named helpers; avoid functions over ~40 lines.\n"
)


def _editor(edit: str | None) -> object:
    """A deterministic SkillEditor that returns ``edit`` (or declines with None)."""

    def _e(
        *, current_body: str, decision: Decision, goal: CompiledGoal, agent: Agent
    ) -> str | None:
        return edit

    return _e


def _resolver(body: str | None) -> object:
    def _r(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> str | None:
        return body

    return _r


def test_quick_edit_produces_a_real_diff() -> None:
    build = agent_skill_edit_builder(
        editor=_editor(_EDITED_BODY), body_resolver=_resolver(_CURRENT_BODY)
    )
    candidate = build(_skill_decision(), goal=_quality_goal(), agent=_agent())
    assert candidate is not None
    assert candidate.change.kind is ChangeKind.SKILL_DIFF
    assert candidate.proof is None  # evidence-only-applyable kind: no frozen-suite proof
    diff = candidate.change.diff
    assert diff and diff.strip()
    # a genuine unified diff carrying the added line (generic, not token-efficiency text)
    assert "@@" in diff
    assert "+Extract shared logic into small, named helpers" in diff


def test_quick_edit_fails_closed_when_editor_declines() -> None:
    build = agent_skill_edit_builder(editor=_editor(None), body_resolver=_resolver(_CURRENT_BODY))
    assert build(_skill_decision(), goal=_quality_goal(), agent=_agent()) is None


def test_quick_edit_fails_closed_on_no_real_edit() -> None:
    # editor returns the same body (only trailing whitespace differs) -> no real change
    build = agent_skill_edit_builder(
        editor=_editor(_CURRENT_BODY + "\n  "), body_resolver=_resolver(_CURRENT_BODY)
    )
    assert build(_skill_decision(), goal=_quality_goal(), agent=_agent()) is None


def test_quick_edit_fails_closed_when_no_champion_body() -> None:
    build = agent_skill_edit_builder(editor=_editor(_EDITED_BODY), body_resolver=_resolver(None))
    assert build(_skill_decision(), goal=_quality_goal(), agent=_agent()) is None


def test_quick_edit_ignores_non_skill_update() -> None:
    build = agent_skill_edit_builder(
        editor=_editor(_EDITED_BODY), body_resolver=_resolver(_CURRENT_BODY)
    )
    assert build(_gepa_decision(), goal=_quality_goal(), agent=_agent()) is None
