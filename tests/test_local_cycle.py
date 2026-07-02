"""Local-runner tests — injectable seams + reporting, no live Databricks/Claude.

The local runner (:mod:`ail.jobs.local_cycle`) reuses the serverless cycle spine
(:func:`ail.jobs.optimization_cycle.run_optimization_cycle`) verbatim and adds only
(1) env-based static-token auth, (2) a reporting layer, and (3) an LLM-gateway thread.
These tests cover exactly that net-new surface with fakes for the prover / Claude Agent
SDK and the UC writes — never a live workspace:

* **Auth is fail-loud + OAuth-proof**: missing host/token raises; a profile is dropped.
* **Gateway resolution**: defaults to ``<host>/serving-endpoints`` + the static token;
  honours ``AIL_LLM_*`` overrides; is threaded into the reused RLM reviewer.
* **Fail-closed is preserved**: a prover that raises → no proposal, the honest error
  surfaced, publish still called with the empty set (superseded slice cleared).
* **Every stage is surfaced**: RLM findings, feedback, readiness/gate, the Lane A/B
  decision + why, the candidate, the baseline-vs-candidate proof (token + tool-call
  deltas, correctness held), and the written proposal all appear in the output.

A single live smoke test is gated behind ``AIL_LIVE_LOCAL_CYCLE=1`` (the repo's
``AIL_LIVE_*`` convention) and is deselected by default.
"""

from __future__ import annotations

import io
import os
from typing import Any

import pytest

from ail.compare.contract import ComparisonResult, MetricDelta, Recommendation
from ail.goals.compiler import CompiledGoal, GoalTarget, Guardrail
from ail.jobs import local_cycle as lc
from ail.l3.continuous import ContinuousRlmRunReport
from ail.l3.contract import TraceReviewOutcome
from ail.loop.controller import Candidate
from ail.loop.decision_rules import Decision, FeedbackBundle, RlmAssetSignal
from ail.loop.planner import PlanParseError
from ail.loop.proposals import ChangeKind, ProposalStatus, ProposedChange
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

# -- builders (mirror tests/test_optimization_cycle.py) --------------------


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


def _outcome_with_deltas() -> TaskOutcome:
    """A PROMOTE task carrying both a token delta and a tool-call delta to render."""
    comparison = ComparisonResult(
        task_id="p1",
        recommendation=Recommendation.PROMOTE,
        objective_met=True,
        guardrails_passed=True,
        deltas=[
            MetricDelta(
                metric="total_tokens",
                baseline=1000.0,
                candidate=650.0,
                delta_absolute=-350.0,
                delta_pct=-35.0,
                improved=True,
            ),
            MetricDelta(
                metric="total_tool_calls",
                unit="calls",
                baseline=20.0,
                candidate=12.0,
                delta_absolute=-8.0,
                delta_pct=-40.0,
                improved=True,
            ),
        ],
    )
    return TaskOutcome(
        task_id="p1",
        category="coding",
        difficulty="medium",
        recommendation=Recommendation.PROMOTE,
        objective_met=True,
        guardrails_passed=True,
        baseline_total_tokens=1000.0,
        candidate_total_tokens=650.0,
        token_delta_absolute=-350.0,
        token_delta_pct=-35.0,
        token_improved=True,
        l1_outcome=L1Outcome.PASSED,
        l1_verification_configured=True,
        baseline_succeeded=True,
        candidate_succeeded=True,
        comparison=comparison,
    )


def _proven_artifact() -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=1,
        n_promote=1,
        realized_baseline_tokens=1000.0,
        realized_candidate_tokens=650.0,
        realized_token_savings_absolute=350.0,
        realized_token_savings_pct=35.0,
        outcomes=[_outcome_with_deltas()],
    )


def _blocked_artifact() -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="v1-seed",
        suite_content_hash="deadbeef",
        objective_metric="total_tokens",
        n_tasks=2,
        n_promote=0,
        n_block=2,
        outcomes=[TaskOutcome(task_id="b", recommendation=Recommendation.BLOCK)],
    )


def _ready() -> ReadinessStatus:
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
            judges=[
                JudgeHealth(
                    judge_name="modularity",
                    measured=True,
                    agreement_rate=0.82,
                    distrusted=False,
                    reason="trusted",
                )
            ],
        ),
    )


def _metric_view_candidate(decision: Decision, *, goal: CompiledGoal, agent: Agent) -> Candidate:
    return Candidate(
        change=ProposedChange(
            kind=ChangeKind.METRIC_VIEW_SQL,
            summary="token waste view",
            sql="CREATE OR REPLACE VIEW c.s.v WITH METRICS LANGUAGE YAML AS $$ ... $$",
        ),
        prover_input="metric_view",
    )


def _feedback_metric_view() -> FeedbackBundle:
    return FeedbackBundle(
        objective_metric_value=900.0,
        objective_baseline_value=1000.0,
        rlm_assets=(
            RlmAssetSignal(
                asset_type="metric_view", title="token waste by tool", n_traces=5, rank=1
            ),
        ),
    )


def _rlm_report(*, outcomes: list[TraceReviewOutcome] | None = None) -> ContinuousRlmRunReport:
    return ContinuousRlmRunReport(
        experiment_id="660599403165942",
        judge_model="databricks-claude-sonnet-4-6",
        n_scanned=10,
        n_already_reviewed=3,
        n_reviewer_traces_skipped=0,
        n_sampled_out=5,
        n_selected=2,
        n_reviewed=1,
        n_failed=1,
        sample_rate=0.10,
        max_reviews=2,
        outcomes=outcomes if outcomes is not None else [],
    )


def _no_planner(feedback: FeedbackBundle, goal: CompiledGoal, agent: Agent) -> list[Decision]:
    # Keep these tests about the local runner, not the LLM planner (it has its own suite).
    raise PlanParseError("no plan (test)")


def _run(**overrides: Any) -> tuple[Any, str]:
    """Drive run_local_cycle with fakes + a captured reporter; return (report, output)."""
    buf = io.StringIO()
    kwargs: dict[str, Any] = {
        "rlm_step": lambda: _rlm_report(),
        "feedback_source": _feedback_metric_view,
        "candidate_builder": _metric_view_candidate,
        "prover": lambda c, *, goal, agent: _proven_artifact(),
        "gate": lambda *, goal, agent: _ready(),
        "publish_fn": lambda proposals: len(proposals),
        "planner": _no_planner,
        "reporter": lc.LocalCycleReporter(stream=buf),
        "now": "2026-07-02T00:00:00+00:00",
    }
    kwargs.update(overrides)
    report = lc.run_local_cycle(_agent(), _goal(), **kwargs)
    return report, buf.getvalue()


# ==========================================================================
# Auth (net-new): static env token, fail-loud, OAuth-proof
# ==========================================================================


def test_resolve_local_auth_requires_host_and_token() -> None:
    with pytest.raises(SystemExit) as ei:
        lc.resolve_local_auth({"DATABRICKS_HOST": "https://x"})  # token missing
    assert "DATABRICKS_TOKEN" in str(ei.value)

    with pytest.raises(SystemExit) as ei2:
        lc.resolve_local_auth({})  # both missing
    assert "DATABRICKS_HOST" in str(ei2.value)
    assert "DATABRICKS_TOKEN" in str(ei2.value)


def test_resolve_local_auth_drops_profile_and_returns_static() -> None:
    env = {
        "DATABRICKS_HOST": "https://ws ",
        "DATABRICKS_TOKEN": " dapi123",
        "DATABRICKS_CONFIG_PROFILE": "dais-demo",
    }
    host, token = lc.resolve_local_auth(env)
    assert host == "https://ws"
    assert token == "dapi123"
    # the profile is dropped so no span can fall back to OAuth mid prover run
    assert "DATABRICKS_CONFIG_PROFILE" not in env
    assert env["DATABRICKS_HOST"] == "https://ws"
    assert env["DATABRICKS_TOKEN"] == "dapi123"


def test_resolve_llm_gateway_default_and_override() -> None:
    base, key = lc.resolve_llm_gateway({}, "https://ws/", "dapiTOK")
    assert base == "https://ws/serving-endpoints"  # trailing slash normalized
    assert key == "dapiTOK"

    base2, key2 = lc.resolve_llm_gateway(
        {"AIL_LLM_BASE_URL": "https://gw/openai", "AIL_LLM_API_KEY": "sk-9"},
        "https://ws",
        "dapiTOK",
    )
    assert base2 == "https://gw/openai"
    assert key2 == "sk-9"


def test_local_rlm_step_threads_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(experiment: str, **kwargs: Any) -> ContinuousRlmRunReport:
        captured["experiment"] = experiment
        captured.update(kwargs)
        return _rlm_report()

    monkeypatch.setattr(lc, "run_continuous_rlm", _fake_run)
    args = _ns(experiment="660599403165942", judge_model="m", warehouse_id="wh1")
    step = lc._local_rlm_step(args, base_url="https://gw", api_key="sk-1")
    step()

    assert captured["experiment"] == "660599403165942"
    assert captured["base_url"] == "https://gw"  # the gateway is threaded in
    assert captured["api_key"] == "sk-1"
    assert captured["judge_model"] == "m"


def _ns(**kw: Any) -> Any:
    """A minimal args namespace with the fields _local_rlm_step reads."""
    from types import SimpleNamespace

    defaults: dict[str, Any] = {
        "experiment": "e",
        "judge_model": "m",
        "warehouse_id": "wh",
        "max_results": 100,
        "max_reviews": 2,
        "sample_rate": 0.10,
        "min_tokens": 50_000,
        "reviewer_experiment": "",
        "max_turns": 40,
        "temperature": None,
    }
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ==========================================================================
# Preflight
# ==========================================================================


def test_preflight_claude_sdk_warns_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc.importlib.util, "find_spec", lambda name: None)
    buf = io.StringIO()
    ok = lc._preflight_claude_sdk(lc.LocalCycleReporter(stream=buf))
    assert ok is False
    out = buf.getvalue()
    assert "claude-agent-sdk is not importable" in out
    assert "fail closed" in out


def test_preflight_claude_sdk_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc.importlib.util, "find_spec", lambda name: object())
    assert lc._preflight_claude_sdk(lc.LocalCycleReporter(stream=io.StringIO())) is True


# ==========================================================================
# End-to-end run_local_cycle with fakes: surface every stage, fail-closed
# ==========================================================================


def test_run_local_cycle_surfaces_all_stages_and_publishes() -> None:
    rlm = _rlm_report(
        outcomes=[
            TraceReviewOutcome(
                trace_id="tr-abc",
                status="reviewed",
                total_tokens=60000,
                token_efficiency="fair",
                token_waste_score=40,
                n_recommended_assets=1,
            ),
            TraceReviewOutcome(trace_id="tr-bad", status="review_failed", error="boom"),
        ]
    )
    published: list[Any] = []

    report, out = _run(
        rlm_step=lambda: rlm,
        publish_fn=lambda proposals: published.append(proposals) or len(proposals),
    )

    # one real proposal published
    assert len(report.cycle.proposals) == 1
    assert report.cycle.proposals[0].status is ProposalStatus.PENDING
    assert report.n_published == 1
    assert len(published) == 1 and len(published[0]) == 1

    # (a) RLM findings per trace
    assert "IN-CYCLE RLM REVIEW" in out
    assert "tr-abc" in out and "efficiency=fair" in out
    assert "tr-bad" in out and "review_failed" in out
    # (b) decision + why + lane
    assert "PLAN → PROVE" in out
    assert "metric_view" in out
    assert "Lane A" in out
    assert "RLM recommended" in out  # the trigger's "why" summary
    # (c) candidate
    assert "candidate:" in out
    assert "token waste view" in out
    # (d) proof: token + tool-call deltas + correctness
    assert "tokens 1,000→650" in out
    assert "tools 20→12 (Δ-8)" in out
    assert "correctness=passed" in out
    assert "correctness_held=True" in out
    # (e) gate
    assert "READINESS GATE" in out
    assert "ready_to_prove" in out
    assert "modularity" in out and "trusted" in out
    # (f) proposal written
    assert "PROPOSALS WRITTEN TO agent_proposed_actions" in out
    assert "proposal(s) written for approval in the app" in out


def test_run_local_cycle_fail_closed_when_prover_raises() -> None:
    published: list[Any] = []

    def _boom(candidate: Candidate, *, goal: CompiledGoal, agent: Agent) -> Phase2Artifact:
        raise TimeoutError("prover session exceeded hard timeout")

    report, out = _run(
        prover=_boom,
        publish_fn=lambda proposals: published.append(proposals) or len(proposals),
    )

    # no proposal, honest error surfaced, publish still called with the empty set
    assert report.cycle.proposals == ()
    assert report.n_published == 0
    assert published == [[]]
    assert "PROOF FAILED (fail-closed)" in out
    assert "TimeoutError" in out
    assert "prover session exceeded hard timeout" in out
    # the controller recorded it as a fail-closed skip (surfaced in the summary)
    assert any("errored" in s.reason for s in report.cycle.skipped)
    assert "fail-closed skips" in out


def test_run_local_cycle_blocked_proof_writes_no_proposal() -> None:
    published: list[Any] = []
    report, out = _run(
        prover=lambda c, *, goal, agent: _blocked_artifact(),
        publish_fn=lambda proposals: published.append(proposals) or len(proposals),
    )
    assert report.cycle.proposals == ()
    assert report.n_published == 0
    assert published == [[]]  # empty set published to clear a superseded slice
    assert "BLOCK" in out
    assert "0 proposals" in out
    assert "not proven on the frozen suite" in " ".join(s.reason for s in report.cycle.skipped)


def test_run_local_cycle_rlm_failure_is_non_blocking() -> None:
    def _rlm_boom() -> ContinuousRlmRunReport:
        raise RuntimeError("trace store unreachable")

    report, out = _run(rlm_step=_rlm_boom)

    # RLM failure recorded + surfaced, but the cycle still proved + published
    assert report.rlm_error is not None and "trace store unreachable" in report.rlm_error
    assert "RLM review failed (non-blocking)" in out
    assert len(report.cycle.proposals) == 1
    assert report.n_published == 1


# ==========================================================================
# Reporter unit: proof rendering carries token + tool-call deltas
# ==========================================================================


def test_reporter_proof_renders_token_and_tool_call_deltas() -> None:
    buf = io.StringIO()
    lc.LocalCycleReporter(stream=buf).proof(_proven_artifact())
    out = buf.getvalue()
    assert "1 PROMOTE / 0 BLOCK / 0 ERRORED" in out
    assert "realized token savings (PROMOTE only): 350" in out
    assert "tokens 1,000→650" in out
    assert "tools 20→12 (Δ-8)" in out
    assert "correctness=passed" in out


# ==========================================================================
# Live smoke (opt-in): AIL_LIVE_* convention, deselected by default
# ==========================================================================


@pytest.mark.live
def test_live_local_cycle_end_to_end() -> None:
    """Run the REAL local cycle against a live workspace + local Claude auth.

    Gated by ``AIL_LIVE_LOCAL_CYCLE=1`` and the same env the runner needs
    (``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` / ``AIL_WAREHOUSE_ID`` /
    ``AIL_LIVE_EXPERIMENT_ID``). Costly: it runs the frozen-suite prover via the
    Claude Agent SDK. Deselected by default so the suite is green offline.
    """
    if os.environ.get("AIL_LIVE_LOCAL_CYCLE") != "1":
        pytest.skip("set AIL_LIVE_LOCAL_CYCLE=1 to run the live local cycle")
    experiment_id = os.environ.get("AIL_LIVE_EXPERIMENT_ID")
    if not experiment_id:
        pytest.skip("set AIL_LIVE_EXPERIMENT_ID to the target experiment")
    if not (os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")):
        pytest.skip("set DATABRICKS_HOST + DATABRICKS_TOKEN (static) for the live run")

    rc = lc.main(
        [
            "--experiment",
            experiment_id,
            "--judge-model",
            os.environ.get("AIL_LIVE_MODEL", "databricks-claude-sonnet-4-6"),
            "--max-reviews",
            "1",
            "--confirm-goal",
        ]
    )
    assert rc == 0
