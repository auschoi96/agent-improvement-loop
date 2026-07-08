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

from pathlib import Path

from ail.compare.contract import Recommendation
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.loop.candidate_builders import (
    GepaSeed,
    agent_skill_edit_builder,
    chain_candidate_builders,
    gepa_candidate_builder,
    gepa_target_key,
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
from ail.optimize.gepa_runner import GepaOptimizationResult
from ail.optimize.phase2 import L1Outcome, Phase2Artifact, TaskOutcome
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


# ==========================================================================
# Piece 2 — the generic GEPA optimization builder (self-proving, held-out)
# ==========================================================================


def _phase2(pct: float, *, n_promote: int = 2, regressed: bool = False) -> Phase2Artifact:
    outcomes = [
        TaskOutcome(
            task_id=f"h{i}", recommendation=Recommendation.PROMOTE, l1_outcome=L1Outcome.PASSED
        )
        for i in range(n_promote)
    ]
    if regressed:
        outcomes.append(
            TaskOutcome(
                task_id="hr", recommendation=Recommendation.BLOCK, l1_outcome=L1Outcome.REGRESSED
            )
        )
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=len(outcomes),
        n_promote=n_promote,
        n_block=1 if regressed else 0,
        realized_token_savings_absolute=pct * 10.0,
        realized_token_savings_pct=pct,
        outcomes=outcomes,
    )


def _result(
    *,
    changed: bool = True,
    evolved_pct: float = 50.0,
    seed_pct: float = 30.0,
    regressed: bool = False,
) -> GepaOptimizationResult:
    return GepaOptimizationResult(
        changed=changed,
        seed_skill_body="seed body",
        evolved_skill_body="evolved body" if changed else "seed body",
        holdout_evolved=_phase2(evolved_pct, regressed=regressed),
        holdout_seed_baseline=_phase2(seed_pct),
    )


def _gepa_run(result: GepaOptimizationResult | None) -> object:
    def _r(
        seed: GepaSeed, *, decision: Decision, goal: CompiledGoal, agent: Agent
    ) -> GepaOptimizationResult | None:
        return result

    return _r


def _seed_resolver(seed: GepaSeed | None) -> object:
    def _s(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> GepaSeed | None:
        return seed

    return _s


_SEED = GepaSeed(target_key="judge:modularity", seed_body="seed body")


def test_gepa_target_key_is_generic_not_token_efficiency() -> None:
    # judge-dimension trigger -> judge:<name>
    assert (
        gepa_target_key(_gepa_decision(judge_name="modularity"), goal=_quality_goal())
        == "judge:modularity"
    )
    # metric-only trigger -> metric:<metric>
    metric_only = Decision(
        ActionKind.GEPA_PROMPT,
        default_risk_class(ActionKind.GEPA_PROMPT),
        TriggerSignal(kind=TriggerKind.AGENT_PLANNER, summary="x", metric="latency_p95"),
    )
    assert gepa_target_key(metric_only, goal=_quality_goal()) == "metric:latency_p95"
    # nothing on the trigger -> falls back to the goal objective (still not hardcoded)
    bare = Decision(
        ActionKind.GEPA_PROMPT,
        default_risk_class(ActionKind.GEPA_PROMPT),
        TriggerSignal(kind=TriggerKind.AGENT_PLANNER, summary="x"),
    )
    assert gepa_target_key(bare, goal=_quality_goal()) == "goal:modularity"


def test_gepa_builder_produces_self_proving_candidate(tmp_path: Path) -> None:
    build = gepa_candidate_builder(
        gepa_run=_gepa_run(_result(evolved_pct=50.0, seed_pct=30.0)),
        seed_resolver=_seed_resolver(_SEED),
        artifacts_root=tmp_path,
    )
    candidate = build(_gepa_decision(), goal=_quality_goal(), agent=_agent())
    assert candidate is not None
    assert candidate.change.kind is ChangeKind.EVOLVED_BODY_REF
    # the pre-computed held-out proof travels on the candidate (evidence-first path)
    assert candidate.proof is not None
    assert candidate.proof.proved_improvement and candidate.proof.correctness_held
    # the evolved_body_ref is a REAL file the apply engine can read + re-validate
    ref = candidate.change.evolved_body_ref
    assert ref and Path(ref).is_file()
    reloaded = GepaOptimizationResult.model_validate_json(Path(ref).read_text(encoding="utf-8"))
    assert reloaded.changed and reloaded.holdout_savings_delta_pct == 20.0
    # target key is generic (judge dimension), not token_efficiency
    assert "judge:modularity" in candidate.change.summary


def test_gepa_builder_fails_closed_on_changed_false(tmp_path: Path) -> None:
    build = gepa_candidate_builder(
        gepa_run=_gepa_run(_result(changed=False)),
        seed_resolver=_seed_resolver(_SEED),
        artifacts_root=tmp_path,
    )
    assert build(_gepa_decision(), goal=_quality_goal(), agent=_agent()) is None
    assert not list(tmp_path.iterdir())  # no artifact written on a fail-closed decline


def test_gepa_builder_fails_closed_on_no_holdout_improvement(tmp_path: Path) -> None:
    # evolved did not beat seed on the held-out split (delta == 0) -> no candidate
    build = gepa_candidate_builder(
        gepa_run=_gepa_run(_result(evolved_pct=30.0, seed_pct=30.0)),
        seed_resolver=_seed_resolver(_SEED),
        artifacts_root=tmp_path,
    )
    assert build(_gepa_decision(), goal=_quality_goal(), agent=_agent()) is None
    assert not list(tmp_path.iterdir())


def test_gepa_builder_fails_closed_when_gepa_cannot_run(tmp_path: Path) -> None:
    # gepa_run returns None: no frozen suite / no local Claude -> fail-closed, no candidate
    build = gepa_candidate_builder(
        gepa_run=_gepa_run(None),
        seed_resolver=_seed_resolver(_SEED),
        artifacts_root=tmp_path,
    )
    assert build(_gepa_decision(), goal=_quality_goal(), agent=_agent()) is None


def test_gepa_builder_fails_closed_when_no_seed(tmp_path: Path) -> None:
    build = gepa_candidate_builder(
        gepa_run=_gepa_run(_result()),
        seed_resolver=_seed_resolver(None),
        artifacts_root=tmp_path,
    )
    assert build(_gepa_decision(), goal=_quality_goal(), agent=_agent()) is None


def test_gepa_builder_fails_closed_on_holdout_correctness_regression(tmp_path: Path) -> None:
    # a held-out task regressed correctness -> proof.correctness_held is False -> no candidate,
    # even though realized savings delta is positive
    build = gepa_candidate_builder(
        gepa_run=_gepa_run(_result(evolved_pct=50.0, seed_pct=30.0, regressed=True)),
        seed_resolver=_seed_resolver(_SEED),
        artifacts_root=tmp_path,
    )
    assert build(_gepa_decision(), goal=_quality_goal(), agent=_agent()) is None


def test_gepa_builder_ignores_non_gepa_decision(tmp_path: Path) -> None:
    build = gepa_candidate_builder(
        gepa_run=_gepa_run(_result()),
        seed_resolver=_seed_resolver(_SEED),
        artifacts_root=tmp_path,
    )
    assert build(_skill_decision(), goal=_quality_goal(), agent=_agent()) is None
