"""Lane B (LLM-agent planner) tests — fail-closed, propose-only, A∪B layered.

No live model call: the planner's LLM seam is always an injected canned-string
callable. Covers the hard constraints the reviewer hammers:

* **Fail-closed parsing** (mirror the HALO parser): garbage / empty / all-low-
  confidence agent output → ZERO decisions via a typed :class:`PlanParseError`,
  never a fabricated decision; individual bad entries degrade (dropped) but a
  wholly-unusable response fails loud.
* **B never applies**: the planner produces :class:`Decision` objects only; a B
  decision flows through the *same* controller pipeline and comes out merely
  ``pending`` (proven + gated) or a fail-closed skip — never applied, and never a
  proposal when unproven / ungated.
* **A ∪ B union + de-dup**: :func:`combined_decisions` concatenates both lanes and
  de-dups by (action_kind, target identity), Lane A winning a collision.
* **Layered cycle**: :func:`run_cycle_with_planner` feeds the union through
  :func:`ail.loop.controller.run_cycle` unchanged; a Lane-B parse failure is
  recorded and Lane A is unaffected.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from ail.compare.contract import Recommendation
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.loop.controller import Candidate
from ail.loop.decision_rules import (
    Decision,
    FeedbackBundle,
    RlmAssetSignal,
)
from ail.loop.planner import (
    CombinedDecisions,
    PlanParseError,
    agent_planner,
    combined_decisions,
    parse_plan,
    run_cycle_with_planner,
)
from ail.loop.proposals import (
    ActionKind,
    ChangeKind,
    ProposalStatus,
    ProposedChange,
    RiskClass,
    TriggerKind,
)
from ail.optimize.phase2 import L1Outcome, Phase2Artifact, TaskOutcome
from ail.readiness.contract import (
    EvalHealth,
    Gate,
    GateName,
    JudgeHealth,
    ReadinessStatus,
    ReadinessTier,
)
from ail.registry import Agent

# -- fixtures / builders ---------------------------------------------------


def _agent() -> Agent:
    return Agent(agent_name="claude_code", experiment_id="660599403165942")


def _goal() -> CompiledGoal:
    return CompiledGoal(
        objective_metric="total_tokens",
        direction="minimize",
        target=GoalTarget(value=-0.30, kind="relative"),
        guardrails=(Guardrail(name="modularity", kind="judge", threshold=4.0),),
        cohort="claude_code",
    ).confirm()


def _llm(payload: object) -> object:
    """A canned PlannerLLM returning ``payload`` (dict → JSON, str → verbatim)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)

    def _call(*, system: str, user: str) -> str:
        return text

    return _call


def _proven_artifact(*, regressed: bool = False, n_promote: int = 3) -> Phase2Artifact:
    outcomes = [
        TaskOutcome(
            task_id=f"p{i}", recommendation=Recommendation.PROMOTE, l1_outcome=L1Outcome.PASSED
        )
        for i in range(n_promote)
    ]
    if regressed:
        outcomes.append(
            TaskOutcome(
                task_id="reg", recommendation=Recommendation.BLOCK, l1_outcome=L1Outcome.REGRESSED
            )
        )
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=n_promote + (1 if regressed else 0),
        n_promote=n_promote,
        realized_token_savings_absolute=1200.0,
        realized_token_savings_pct=35.4,
        outcomes=outcomes,
    )


def _blocked_artifact() -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=4,
        n_promote=0,
        n_block=4,
        outcomes=[
            TaskOutcome(task_id=f"b{i}", recommendation=Recommendation.BLOCK) for i in range(4)
        ],
    )


def _ready(*, modularity_distrusted: bool = False) -> ReadinessStatus:
    judges = [
        JudgeHealth(
            judge_name="modularity",
            measured=not modularity_distrusted,
            agreement_rate=None if modularity_distrusted else 0.82,
            distrusted=modularity_distrusted,
            reason="unmeasured" if modularity_distrusted else "trusted",
        )
    ]
    return ReadinessStatus(
        cohort_name="claude_code",
        objective_metric="total_tokens",
        requires_quality=True,
        guardrail_names=["modularity"],
        trace_count=80,
        tier=ReadinessTier.READY_TO_PROVE,
        gates=[Gate(name=GateName.TRACE_PROVE, passed=True, reason="enough traces")],
        reasons=[],
        eval_health=EvalHealth(
            cohort_name="claude_code",
            scored_coverage=0.9,
            n_distrusted_judges=1 if modularity_distrusted else 0,
            distrusted_judges=["modularity"] if modularity_distrusted else [],
            judges=judges,
        ),
    )


def _build_candidate(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate:
    ak = decision.action_kind
    if ak is ActionKind.METRIC_VIEW:
        change = ProposedChange(
            kind=ChangeKind.METRIC_VIEW_SQL,
            summary="token waste view",
            sql="CREATE OR REPLACE VIEW c.s.v WITH METRICS LANGUAGE YAML AS $$ ... $$",
        )
    elif ak is ActionKind.SKILL_UPDATE:
        change = ProposedChange(
            kind=ChangeKind.SKILL_DIFF, summary="read-cache skill", diff="--- a\n+++ b"
        )
    elif ak is ActionKind.GEPA_PROMPT:
        change = ProposedChange(
            kind=ChangeKind.EVOLVED_BODY_REF,
            summary="gepa-evolved skill",
            evolved_body_ref="prompts:/c.s.p/4",
        )
    elif ak is ActionKind.INSTRUCTION_UPDATE:
        change = ProposedChange(kind=ChangeKind.INSTRUCTION_DIFF, summary="instr", diff="--- a")
    else:  # REVERT
        change = ProposedChange(kind=ChangeKind.REVERT_REF, summary="revert", revert_target="v1")
    return Candidate(change=change, prover_input=ak.value)


# ==========================================================================
# Fail-closed parsing (mirror the HALO parser)
# ==========================================================================


def test_non_json_response_raises_parse_error() -> None:
    with pytest.raises(PlanParseError):
        parse_plan("I could not analyze the traces. No JSON here.")


def test_missing_plan_array_raises_parse_error() -> None:
    with pytest.raises(PlanParseError):
        parse_plan(json.dumps({"notplan": []}))


def test_empty_plan_array_raises_parse_error() -> None:
    # An empty plan yields zero decisions -> fail loud (never a silent []).
    with pytest.raises(PlanParseError):
        parse_plan(json.dumps({"plan": []}))


def test_all_low_confidence_entries_raise_parse_error() -> None:
    payload = {
        "plan": [
            {"action_kind": "gepa_prompt", "rationale": "maybe", "confidence": 0.1},
            {"action_kind": "metric_view", "rationale": "maybe", "confidence": 0.2},
        ]
    }
    with pytest.raises(PlanParseError):
        parse_plan(json.dumps(payload), confidence_floor=0.5)


def test_boolean_confidence_is_dropped_not_coerced() -> None:
    # bool is an int subclass; float(True)==1.0 must NOT sneak past as high confidence.
    payload = {"plan": [{"action_kind": "gepa_prompt", "rationale": "x", "confidence": True}]}
    with pytest.raises(PlanParseError):
        parse_plan(json.dumps(payload))


def test_bad_entries_dropped_but_valid_entry_kept() -> None:
    # An unknown action_kind and a no-rationale entry are dropped; the one well-formed,
    # confident metric_view survives — never fabricated, just the good one.
    payload = {
        "plan": [
            {"action_kind": "teleport", "rationale": "nonsense", "confidence": 0.9},
            {"action_kind": "gepa_prompt", "rationale": "", "confidence": 0.9},
            {
                "action_kind": "metric_view",
                "rationale": "recurring token waste by tool",
                "confidence": 0.8,
                "asset_type": "metric_view",
                "metric": "total_tokens",
                "trace_refs": ["t1", "t2"],
            },
        ]
    }
    decisions = parse_plan(json.dumps(payload))
    assert len(decisions) == 1
    d = decisions[0]
    assert d.action_kind is ActionKind.METRIC_VIEW
    assert d.trigger.kind is TriggerKind.AGENT_PLANNER  # A-vs-B attribution marker
    assert d.trigger.asset_type == "metric_view"
    assert d.trigger.trace_refs == ["t1", "t2"]
    assert d.trigger.observed_value == 0.8


def test_response_wrapped_in_prose_and_fences_still_parses() -> None:
    body = (
        "Here is my plan:\n```json\n"
        + json.dumps(
            {
                "plan": [
                    {"action_kind": "gepa_prompt", "rationale": "low modularity", "confidence": 0.9}
                ]
            }
        )
        + "\n```\nThanks!"
    )
    decisions = parse_plan(body)
    assert [d.action_kind for d in decisions] == [ActionKind.GEPA_PROMPT]


def test_agent_planner_fails_closed_on_garbage_llm() -> None:
    # The planner NEVER fabricates a decision from garbage; it raises PlanParseError.
    with pytest.raises(PlanParseError):
        agent_planner(FeedbackBundle(), _goal(), _agent(), llm=_llm("garbage, not a plan"))


# ==========================================================================
# A ∪ B union + de-dup
# ==========================================================================


def _feedback_metric_view() -> FeedbackBundle:
    return FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,  # bar 700; 900 > 700 -> objective not met
        rlm_assets=(
            RlmAssetSignal(
                asset_type="metric_view",
                title="token waste by tool",
                n_traces=5,
                rank=1,
                trace_ids=("t1", "t2", "t3"),
            ),
        ),
    )


def test_combined_union_keeps_distinct_b_decision_after_a() -> None:
    # A fires a metric_view (from the RLM asset). B proposes a distinct gepa_prompt.
    feedback = _feedback_metric_view()
    planner_payload = {
        "plan": [
            {
                "action_kind": "gepa_prompt",
                "rationale": "modularity looks low across traces",
                "confidence": 0.9,
                "judge_name": "modularity",
                "metric": "modularity",
            }
        ]
    }
    combined = combined_decisions(
        feedback, _goal(), _agent(), planner=lambda f, g, a: parse_plan(json.dumps(planner_payload))
    )
    assert isinstance(combined, CombinedDecisions)
    assert combined.n_from_a == 1
    assert combined.n_from_b == 1
    assert combined.n_deduped == 0
    assert combined.planner_error is None
    # A first, then B
    kinds = [d.action_kind for d in combined.decisions]
    assert kinds == [ActionKind.METRIC_VIEW, ActionKind.GEPA_PROMPT]
    assert combined.decisions[0].trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET
    assert combined.decisions[1].trigger.kind is TriggerKind.AGENT_PLANNER


def test_combined_union_dedups_b_that_collides_with_a() -> None:
    # B proposes a metric_view for the SAME asset_type+metric A already decided ->
    # de-duped; A's (deterministic, evidence-grounded) decision wins.
    feedback = _feedback_metric_view()
    planner_payload = {
        "plan": [
            {
                "action_kind": "metric_view",
                "rationale": "same idea as the RLM asset",
                "confidence": 0.95,
                "asset_type": "metric_view",
                "metric": "total_tokens",
            }
        ]
    }
    combined = combined_decisions(
        feedback, _goal(), _agent(), planner=lambda f, g, a: parse_plan(json.dumps(planner_payload))
    )
    assert combined.n_from_a == 1
    assert combined.n_from_b == 1
    assert combined.n_deduped == 1
    assert len(combined.decisions) == 1
    # the surviving decision is A's, not B's
    assert combined.decisions[0].trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET


def test_combined_records_planner_error_and_keeps_lane_a() -> None:
    # A garbage planner fails closed: Lane A is unaffected, the error is recorded,
    # and NO fabricated B decision is added.
    feedback = _feedback_metric_view()
    combined = combined_decisions(
        feedback,
        _goal(),
        _agent(),
        planner=lambda f, g, a: agent_planner(f, g, a, llm=_llm("junk")),
    )
    assert combined.n_from_a == 1
    assert combined.n_from_b == 0
    assert combined.planner_error is not None
    assert [d.action_kind for d in combined.decisions] == [ActionKind.METRIC_VIEW]


def test_combined_survives_planner_llm_call_failure_and_keeps_lane_a() -> None:
    # Resilience: a planner whose underlying LLM CALL fails for a NON-parse reason
    # (a network timeout, auth error, or MlflowException — simulated here as a
    # generic RuntimeError) must NOT crash the cycle. Lane A's already-computed
    # decisions are preserved unchanged, the error is recorded, ZERO Lane-B
    # decisions are added (never a fabricated one), and no exception propagates.
    feedback = _feedback_metric_view()

    def _exploding_planner(f: FeedbackBundle, g: CompiledGoal, a: Agent) -> list[Decision]:
        raise RuntimeError("simulated MlflowException: serving endpoint unreachable")

    combined = combined_decisions(feedback, _goal(), _agent(), planner=_exploding_planner)

    assert combined.n_from_a == 1
    assert combined.n_from_b == 0
    assert combined.n_deduped == 0
    assert combined.planner_error is not None
    assert "serving endpoint unreachable" in combined.planner_error
    # Lane A's metric_view decision survived verbatim, provenance intact.
    assert [d.action_kind for d in combined.decisions] == [ActionKind.METRIC_VIEW]
    assert combined.decisions[0].trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET


# ==========================================================================
# B never applies: it flows through the SAME pipeline, comes out pending
# ==========================================================================


def _empty_feedback() -> FeedbackBundle:
    """Feedback that fires NO Lane-A rule, so any proposal is Lane-B's alone."""
    return FeedbackBundle(objective_metric_value=900.0, objective_baseline_value=1000.0)


def _b_only_planner(action_kind: str, **fields: object):  # type: ignore[no-untyped-def]
    entry = {"action_kind": action_kind, "rationale": "b proposes", "confidence": 0.9, **fields}

    def _planner(f: FeedbackBundle, g: CompiledGoal, a: Agent) -> list[Decision]:
        return parse_plan(json.dumps({"plan": [entry]}))

    return _planner


def test_b_decision_proven_and_gated_emits_pending_proposal_not_applied() -> None:
    # A pure Lane-B decision, proven + gated, becomes exactly one PENDING proposal.
    # PENDING (not APPLIED) is the whole point: B proposes, it never applies.
    pc = run_cycle_with_planner(
        _agent(),
        _goal(),
        feedback_source=_empty_feedback,
        candidate_builder=_build_candidate,
        prover=lambda candidate, *, goal, agent: _proven_artifact(),
        gate=lambda *, goal, agent: _ready(),
        planner=_b_only_planner("metric_view", asset_type="metric_view", metric="total_tokens"),
        now="2026-06-30T00:00:00+00:00",
    )
    assert len(pc.result.proposals) == 1
    p = pc.result.proposals[0]
    assert p.status is ProposalStatus.PENDING
    assert p.action_kind is ActionKind.METRIC_VIEW
    assert p.trigger.kind is TriggerKind.AGENT_PLANNER
    assert pc.plan.n_from_a == 0
    assert pc.plan.n_from_b == 1


def test_b_decision_unproven_yields_zero_proposals() -> None:
    # A blocked prove -> fail-closed skip, no proposal. B cannot ship an unproven change.
    pc = run_cycle_with_planner(
        _agent(),
        _goal(),
        feedback_source=_empty_feedback,
        candidate_builder=_build_candidate,
        prover=lambda candidate, *, goal, agent: _blocked_artifact(),
        gate=lambda *, goal, agent: _ready(),
        planner=_b_only_planner("metric_view", asset_type="metric_view", metric="total_tokens"),
    )
    assert pc.result.proposals == ()
    assert len(pc.result.skipped) == 1
    assert "not proven" in pc.result.skipped[0].reason


def test_b_judge_decision_gated_by_distrusted_certifying_judge() -> None:
    # B proposes a gepa_prompt naming judge 'modularity'; the certifying-judge gate
    # applies to a B decision exactly as to an A one -> distrusted judge blocks it.
    pc = run_cycle_with_planner(
        _agent(),
        _goal(),
        feedback_source=_empty_feedback,
        candidate_builder=_build_candidate,
        prover=lambda candidate, *, goal, agent: _proven_artifact(),
        gate=lambda *, goal, agent: _ready(modularity_distrusted=True),
        planner=_b_only_planner("gepa_prompt", judge_name="modularity", metric="modularity"),
    )
    assert pc.result.proposals == ()
    assert "distrusted" in pc.result.skipped[0].reason


def test_layered_cycle_emits_both_a_and_b_proposals() -> None:
    # A metric_view (from A) + a distinct gepa_prompt (from B), both proven + gated,
    # come out as two PENDING proposals — the union flowed through run_cycle intact.
    feedback = _feedback_metric_view()
    pc = run_cycle_with_planner(
        _agent(),
        _goal(),
        feedback_source=lambda: feedback,
        candidate_builder=_build_candidate,
        prover=lambda candidate, *, goal, agent: _proven_artifact(),
        gate=lambda *, goal, agent: _ready(),
        planner=_b_only_planner("gepa_prompt", judge_name="modularity", metric="modularity"),
        now="2026-06-30T00:00:00+00:00",
    )
    kinds = sorted(p.action_kind.value for p in pc.result.proposals)
    assert kinds == ["gepa_prompt", "metric_view"]
    assert all(p.status is ProposalStatus.PENDING for p in pc.result.proposals)
    # exactly one proposal is Lane B's (the gepa_prompt), attributable by trigger kind
    b_props = [p for p in pc.result.proposals if p.trigger.kind is TriggerKind.AGENT_PLANNER]
    assert [p.action_kind for p in b_props] == [ActionKind.GEPA_PROMPT]
    assert b_props[0].risk_class is RiskClass.AGENT_CHANGE


# ==========================================================================
# Default LLM seam: Databricks Claude temperature-parameter fallback
# ==========================================================================
#
# Databricks Claude serving endpoints (e.g. ``databricks-claude-opus-4-7``, backed
# by ``us.anthropic.claude-opus-4-7``) reject the ``temperature`` sampling parameter
# with ``400 BAD_REQUEST: ... does not support the temperature parameter``. The
# default planner LLM must retry the same request without ``temperature`` on that
# specific 400 (so Lane B works against Databricks Claude) while leaving
# temperature-accepting endpoints untouched — and must NOT swallow any other failure
# into a fabricated decision (the fail-closed contract). These tests inject a fake
# deploy client via ``get_deploy_client``, so no live model call is made.


_GEPA_PLAN = {
    "plan": [
        {
            "action_kind": "gepa_prompt",
            "rationale": "modularity looks low across traces",
            "confidence": 0.9,
            "judge_name": "modularity",
        }
    ]
}


def _chat_response(payload: object) -> dict[str, object]:
    """Shape a deploy-client chat completion whose content is ``payload`` as JSON."""
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class _RecordingDeployClient:
    """A fake MLflow deploy client: records each ``predict`` input and delegates the
    response (or exception) to an injected behavior. No live model call is made."""

    def __init__(self, behavior: Callable[[dict[str, object]], object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._behavior = behavior

    def predict(self, *, endpoint: str, inputs: dict[str, object]) -> object:
        self.calls.append(inputs)
        return self._behavior(inputs)


def test_default_planner_retries_without_temperature_on_databricks_claude_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Databricks Claude 400 rejecting `temperature` must trigger a transparent
    # retry of the SAME request without it, which then succeeds -> Lane B produces
    # decisions instead of failing every run.
    response = _chat_response(_GEPA_PLAN)

    def _reject_temperature(inputs: dict[str, object]) -> object:
        if "temperature" in inputs:
            raise RuntimeError(
                "400 BAD_REQUEST: Model us.anthropic.claude-opus-4-7 "
                "does not support the temperature parameter"
            )
        return response

    client = _RecordingDeployClient(_reject_temperature)
    import mlflow.deployments

    monkeypatch.setattr(mlflow.deployments, "get_deploy_client", lambda target: client)

    decisions = agent_planner(
        FeedbackBundle(), _goal(), _agent(), model="databricks:/databricks-claude-opus-4-7"
    )

    assert [d.action_kind for d in decisions] == [ActionKind.GEPA_PROMPT]
    assert decisions[0].trigger.kind is TriggerKind.AGENT_PLANNER
    # First call sent temperature (got the 400); retry dropped it and succeeded.
    assert len(client.calls) == 2
    assert client.calls[0].get("temperature") == 0
    assert "temperature" not in client.calls[1]


def test_default_planner_keeps_temperature_when_endpoint_accepts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An endpoint that accepts `temperature` is unaffected: exactly one call with
    # temperature intact, no retry. The fallback is scoped to the temperature 400.
    response = _chat_response(_GEPA_PLAN)
    client = _RecordingDeployClient(lambda inputs: response)
    import mlflow.deployments

    monkeypatch.setattr(mlflow.deployments, "get_deploy_client", lambda target: client)

    decisions = agent_planner(
        FeedbackBundle(), _goal(), _agent(), model="databricks:/some-temperature-ok-endpoint"
    )

    assert [d.action_kind for d in decisions] == [ActionKind.GEPA_PROMPT]
    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0


def test_default_planner_non_temperature_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A genuine LLM failure that is NOT the temperature 400 (network / auth / 5xx)
    # must NOT be retried-without-temperature and must NOT be swallowed into a
    # fabricated decision: it propagates, `combined_decisions` records it, Lane B
    # contributes ZERO decisions, and Lane A survives unchanged (fail-closed).
    def _boom(inputs: dict[str, object]) -> object:
        raise RuntimeError("500 INTERNAL_ERROR: serving endpoint unreachable")

    client = _RecordingDeployClient(_boom)
    import mlflow.deployments

    monkeypatch.setattr(mlflow.deployments, "get_deploy_client", lambda target: client)

    feedback = _feedback_metric_view()
    combined = combined_decisions(
        feedback,
        _goal(),
        _agent(),
        planner=lambda f, g, a: agent_planner(
            f, g, a, model="databricks:/databricks-claude-opus-4-7"
        ),
    )

    assert combined.n_from_b == 0
    assert combined.planner_error is not None
    assert "serving endpoint unreachable" in combined.planner_error
    # No retry on a non-temperature error: the endpoint saw exactly one call.
    assert len(client.calls) == 1
    # Lane A is unaffected: its metric_view decision survived, provenance intact.
    assert [d.action_kind for d in combined.decisions] == [ActionKind.METRIC_VIEW]
    assert combined.decisions[0].trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET


def test_default_planner_retries_on_mlflow_bad_request_without_status_in_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real production shape: the deploy client raises an MlflowException whose
    # str() carries the message but NOT the HTTP status. The retry must still fire —
    # driven by the structured 400 (get_http_status_code), not by "400" in the text —
    # proving the fallback keys on status, not on a coincidental substring.
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import BAD_REQUEST

    response = _chat_response(_GEPA_PLAN)

    def _reject_temperature(inputs: dict[str, object]) -> object:
        if "temperature" in inputs:
            raise MlflowException(
                "Model us.anthropic.claude-opus-4-7 does not support the temperature parameter",
                error_code=BAD_REQUEST,
            )
        return response

    client = _RecordingDeployClient(_reject_temperature)
    import mlflow.deployments

    monkeypatch.setattr(mlflow.deployments, "get_deploy_client", lambda target: client)

    decisions = agent_planner(
        FeedbackBundle(), _goal(), _agent(), model="databricks:/databricks-claude-opus-4-7"
    )

    assert [d.action_kind for d in decisions] == [ActionKind.GEPA_PROMPT]
    assert len(client.calls) == 2
    assert client.calls[0].get("temperature") == 0
    assert "temperature" not in client.calls[1]


def test_default_planner_non_400_mentioning_temperature_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Guard against matching on message text alone: a NON-400 failure (proxy / auth /
    # 5xx) whose text merely happens to mention 'temperature ... unsupported' must NOT
    # take the retry-without-temperature path. It must propagate so Lane B stays
    # fail-closed (zero decisions, Lane A intact) — never emitting decisions for an
    # error that was not the specific unsupported-temperature 400.
    def _boom(inputs: dict[str, object]) -> object:
        raise RuntimeError("502 Bad Gateway from proxy: upstream said temperature is unsupported")

    client = _RecordingDeployClient(_boom)
    import mlflow.deployments

    monkeypatch.setattr(mlflow.deployments, "get_deploy_client", lambda target: client)

    feedback = _feedback_metric_view()
    combined = combined_decisions(
        feedback,
        _goal(),
        _agent(),
        planner=lambda f, g, a: agent_planner(
            f, g, a, model="databricks:/databricks-claude-opus-4-7"
        ),
    )

    assert combined.n_from_b == 0
    assert combined.planner_error is not None
    assert "502" in combined.planner_error  # the ORIGINAL non-400 error propagated
    # No retry: the fallback keyed on status, not the word 'temperature' — one call.
    assert len(client.calls) == 1
    assert [d.action_kind for d in combined.decisions] == [ActionKind.METRIC_VIEW]
    assert combined.decisions[0].trigger.kind is TriggerKind.RLM_RECOMMENDED_ASSET
