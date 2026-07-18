"""Offline tests for the scheduled RLM job wrapper (:mod:`ail.jobs.continuous_rlm`).

Auth (``resolve_job_auth``) and the whole review pass (``run_continuous_rlm``) are
stubbed, so the wrapper's own responsibilities are exercised without a workspace,
HALO, or a model: the ``databricks-claude-opus-4-8`` judge default, goal-derived-rubric
vs. default-rubric fallback, and the reasoning-effort passthrough.
"""

from __future__ import annotations

from typing import Any

import pytest

import ail.jobs.continuous_rlm as job
from ail.l3.rubric import DEFAULT_RUBRIC


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(job, "resolve_job_auth", lambda **kw: "minted")

    def fake_run(experiment_id: str, **kwargs: Any) -> Any:
        calls["experiment_id"] = experiment_id
        calls["kwargs"] = kwargs
        # a minimal report-shaped object for main()'s print()
        from types import SimpleNamespace

        return SimpleNamespace(
            n_scanned=0,
            n_already_reviewed=0,
            n_reviewer_traces_skipped=0,
            n_sampled_out=0,
            n_selected=0,
            n_reviewed=0,
            n_failed=0,
        )

    monkeypatch.setattr(job, "run_continuous_rlm", fake_run)
    return calls


class TestJudgeModelDefault:
    def test_defaults_to_databricks_claude_opus_4_8(self, captured: dict[str, Any]) -> None:
        rc = job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--objective-metric=",
            ]
        )
        assert rc == 0
        assert job.DEFAULT_JUDGE_MODEL == "databricks-claude-opus-4-8"
        assert captured["kwargs"]["judge_model"] == "databricks-claude-opus-4-8"
        assert captured["kwargs"]["enable_code_sandbox"] is False

    def test_fails_when_all_selected_reviews_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(job, "resolve_job_auth", lambda **kw: "minted")
        monkeypatch.setattr(
            job,
            "run_continuous_rlm",
            lambda *a, **kw: type(
                "Report",
                (),
                {
                    "n_scanned": 1,
                    "n_already_reviewed": 0,
                    "n_reviewer_traces_skipped": 0,
                    "n_sampled_out": 0,
                    "n_selected": 1,
                    "n_reviewed": 0,
                    "n_failed": 1,
                },
            )(),
        )
        assert (
            job.main(
                [
                    "--experiment=EXP1",
                    "--reviewer-experiment=REV1",
                    "--warehouse-id=wh-1",
                    "--objective-metric=",
                ]
            )
            == 1
        )

    def test_judge_model_is_overridable(self, captured: dict[str, Any]) -> None:
        job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--judge-model=databricks-gpt-5-5-pro-preview",
                "--objective-metric=",
            ]
        )
        assert captured["kwargs"]["judge_model"] == "databricks-gpt-5-5-pro-preview"

    def test_code_sandbox_can_be_enabled_for_manual_runs(self, captured: dict[str, Any]) -> None:
        job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--objective-metric=",
                "--code-sandbox=auto",
            ]
        )
        assert captured["kwargs"]["enable_code_sandbox"] is True


class TestGoalSteering:
    def test_goal_configured_builds_goal_derived_rubric(self, captured: dict[str, Any]) -> None:
        job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--objective-metric=total_tokens",
                "--goal-direction=minimize",
                "--goal-target=-0.30",
            ]
        )
        rubric = captured["kwargs"]["rubric"]
        assert rubric.rubric_id == "ail.l3.goal/total_tokens-minimize/v1"
        assert rubric.objective == "reduce the agent's total tokens by 30%"
        # Guidelines/scale reused from the default (goal only re-objectives).
        assert rubric.guideline_ids() == DEFAULT_RUBRIC.guideline_ids()

    def test_goal_with_judge_guardrail(self, captured: dict[str, Any]) -> None:
        job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--objective-metric=total_tokens",
                "--guardrail-judge=correctness:4",
            ]
        )
        assert "not regressing correctness" in captured["kwargs"]["rubric"].objective or (
            "correctness" in captured["kwargs"]["rubric"].objective
        )

    def test_empty_objective_metric_falls_back_to_default_rubric(
        self, captured: dict[str, Any]
    ) -> None:
        # The bundle passes an empty string when no goal is configured.
        job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--objective-metric=",
            ]
        )
        assert captured["kwargs"]["rubric"] is DEFAULT_RUBRIC

    def test_unmapped_objective_metric_fails_loud(self, captured: dict[str, Any]) -> None:
        from ail.goals.compiler import UnmappedMetricError

        with pytest.raises(UnmappedMetricError):
            job.main(
                [
                    "--experiment=EXP1",
                    "--reviewer-experiment=REV1",
                    "--warehouse-id=wh-1",
                    "--objective-metric=not_a_real_metric",
                ]
            )


class TestReasoningEffortPassthrough:
    @pytest.mark.parametrize("value", ["", "none", "NONE", "auto", "AUTO"])
    def test_auto_sentinels_pass_none_to_auto_resolve(
        self, captured: dict[str, Any], value: str
    ) -> None:
        # The footgun regression at the job boundary: empty / 'none' / 'auto' (any case)
        # must be normalized to None (auto-resolve), NOT forwarded as a literal effort.
        job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--objective-metric=",
                f"--reasoning-effort={value}",
            ]
        )
        assert captured["kwargs"]["reasoning_effort"] is None

    def test_explicit_effort_passes_through(self, captured: dict[str, Any]) -> None:
        job.main(
            [
                "--experiment=EXP1",
                "--reviewer-experiment=REV1",
                "--warehouse-id=wh-1",
                "--objective-metric=",
                "--reasoning-effort=high",
            ]
        )
        assert captured["kwargs"]["reasoning_effort"] == "high"

    def test_bogus_effort_rejected_at_boundary(self, captured: dict[str, Any]) -> None:
        # argparse choices reject a typo'd effort up front (fail loud, not silent).
        with pytest.raises(SystemExit):
            job.main(
                [
                    "--experiment=EXP1",
                    "--reviewer-experiment=REV1",
                    "--warehouse-id=wh-1",
                    "--objective-metric=",
                    "--reasoning-effort=banana",
                ]
            )


def test_requires_warehouse_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIL_WAREHOUSE_ID", raising=False)
    with pytest.raises(SystemExit):
        job.main(["--experiment=EXP1"])


def test_single_firing_drains_bounded_batches_only_to_per_run_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(job, "resolve_job_auth", lambda **kw: "minted")
    calls: list[tuple[set[str], int]] = []

    def report(trace_ids: list[str]) -> Any:
        return SimpleNamespace(
            n_scanned=3,
            n_already_reviewed=0,
            n_reviewer_traces_skipped=0,
            n_sampled_out=0,
            n_selected=len(trace_ids),
            n_reviewed=len(trace_ids),
            n_failed=0,
            outcomes=[SimpleNamespace(trace_id=trace_id) for trace_id in trace_ids],
        )

    batches = iter([report(["t1", "t2"]), report(["t3"])])

    def fake_run(_experiment_id: str, **kwargs: Any) -> Any:
        calls.append((set(kwargs["exclude_trace_ids"]), kwargs["max_reviews"]))
        return next(batches)

    monkeypatch.setattr(job, "run_continuous_rlm", fake_run)

    rc = job.main(
        [
            "--experiment=EXP1",
            "--reviewer-experiment=REV1",
            "--warehouse-id=wh-1",
            "--objective-metric=",
            "--max-reviews=3",
        ]
    )

    assert rc == 0
    assert calls == [(set(), 3), ({"t1", "t2"}, 1)]


# -- registry (multi-agent) mode -------------------------------------------


def _agent(name: str, exp: str, **over: Any) -> Any:
    from ail.registry import Agent

    over.setdefault("reviewer_experiment_id", f"{exp}_REVIEWER")
    return Agent(agent_name=name, experiment_id=exp, **over)


@pytest.fixture
def registry_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Neutralize auth + capture every per-agent run_continuous_rlm call."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(job, "resolve_job_auth", lambda **kw: "minted")

    def fake_run(experiment_id: str, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        calls.append({"experiment_id": experiment_id, **kwargs})
        return SimpleNamespace(
            n_scanned=0,
            n_already_reviewed=0,
            n_reviewer_traces_skipped=0,
            n_sampled_out=0,
            n_selected=0,
            n_reviewed=0,
            n_failed=0,
        )

    monkeypatch.setattr(job, "run_continuous_rlm", fake_run)
    return {"calls": calls}


def test_registry_mode_threads_each_agents_experiment_and_goal(
    registry_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        job,
        "load_registered_agents",
        lambda **kw: [
            _agent("a1", "EXP_A", goal_config={"objective_metric": "total_tokens"}),
            _agent("a2", "EXP_B"),  # no goal_config -> NEUTRAL default rubric (no leak)
        ],
    )

    rc = job.main(["--warehouse-id=wh", "--catalog=cat", "--schema=sch"])
    assert rc == 0

    calls = registry_capture["calls"]
    assert [c["experiment_id"] for c in calls] == ["EXP_A", "EXP_B"]
    # Agent a1 was reviewed against ITS objective (total_tokens), not a shared global.
    assert calls[0]["rubric"].rubric_id == "ail.l3.goal/total_tokens-minimize/v1"
    # Agent a2 has no goal_config => the neutral default rubric.
    assert calls[1]["rubric"] is DEFAULT_RUBRIC
    # The reviewer's own traces land in each agent's own experiment (per-agent skip).
    assert calls[0]["reviewer_experiment_id"] == "EXP_A_REVIEWER"
    assert calls[1]["reviewer_experiment_id"] == "EXP_B_REVIEWER"


def test_registry_mode_does_not_leak_global_goal_args_onto_agents(
    registry_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # BLOCKER-1 REGRESSION: even when a GLOBAL goal is passed on the CLI (as the
    # deployed bundle does via ${var.objective_metric} etc.), a registry agent with
    # NO goal_config must get the NEUTRAL default rubric — NOT a rubric built from the
    # leftover global objective. An agent WITH its own goal_config gets its own.
    monkeypatch.setattr(
        job,
        "load_registered_agents",
        lambda **kw: [
            _agent("a1", "EXP_A"),  # no goal_config
            _agent("a2", "EXP_B", goal_config={"objective_metric": "total_tokens"}),
        ],
    )

    rc = job.main(
        [
            "--warehouse-id=wh",
            "--catalog=cat",
            "--schema=sch",
            # A global goal that MUST NOT leak onto un-configured registry agents.
            "--objective-metric=total_tokens",
            "--goal-direction=minimize",
            "--goal-target=-0.30",
            "--guardrail-judge=correctness:4",
            "--reviewer-experiment=GLOBAL_REVIEWER",
        ]
    )
    assert rc == 0

    calls = registry_capture["calls"]
    # a1 has no goal_config -> NEUTRAL default rubric, NOT the global total_tokens goal.
    assert calls[0]["rubric"] is DEFAULT_RUBRIC
    # a2 keeps its own goal.
    assert calls[1]["rubric"].rubric_id == "ail.l3.goal/total_tokens-minimize/v1"
    # Defense-in-depth: the global --reviewer-experiment is ignored in registry mode;
    # each agent's reviewer traces go to its OWN experiment.
    assert calls[0]["reviewer_experiment_id"] == "EXP_A_REVIEWER"
    assert calls[1]["reviewer_experiment_id"] == "EXP_B_REVIEWER"


def test_registry_partial_goal_config_uses_neutral_defaults_not_global(
    registry_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A partial goal_config (objective set, direction/target unset) uses NEUTRAL
    # framework defaults for the unset knobs — never the global CLI args.
    monkeypatch.setattr(
        job,
        "load_registered_agents",
        lambda **kw: [_agent("a1", "EXP_A", goal_config={"objective_metric": "total_tokens"})],
    )
    rc = job.main(
        [
            "--warehouse-id=wh",
            "--catalog=cat",
            "--schema=sch",
            # Global direction=maximize must NOT override the agent's unset direction.
            "--objective-metric=total_cost",
            "--goal-direction=maximize",
        ]
    )
    assert rc == 0
    # neutral direction is 'minimize' (framework default), NOT the global 'maximize'.
    assert (
        registry_capture["calls"][0]["rubric"].rubric_id == "ail.l3.goal/total_tokens-minimize/v1"
    )


def test_single_agent_mode_still_honors_global_goal_args(
    registry_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The single-agent override path: the args ARE the legitimate goal source.
    monkeypatch.setattr(
        job, "load_registered_agents", lambda **kw: pytest.fail("must not read registry")
    )
    rc = job.main(
        [
            "--experiment=EXP_SOLO",
            "--reviewer-experiment=REV_SOLO",
            "--warehouse-id=wh",
            "--objective-metric=total_tokens",
            "--goal-direction=minimize",
        ]
    )
    assert rc == 0
    calls = registry_capture["calls"]
    assert calls[0]["experiment_id"] == "EXP_SOLO"
    assert calls[0]["rubric"].rubric_id == "ail.l3.goal/total_tokens-minimize/v1"


def test_registry_mode_isolation_one_agent_failure_continues(
    registry_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted: list[str] = []

    def fake_run(experiment_id: str, **kwargs: Any) -> Any:
        attempted.append(experiment_id)
        if experiment_id == "EXP_B":
            raise RuntimeError("HALO blew up for B")
        from types import SimpleNamespace

        return SimpleNamespace(
            n_scanned=0,
            n_already_reviewed=0,
            n_reviewer_traces_skipped=0,
            n_sampled_out=0,
            n_selected=0,
            n_reviewed=0,
            n_failed=0,
        )

    monkeypatch.setattr(job, "run_continuous_rlm", fake_run)
    monkeypatch.setattr(
        job,
        "load_registered_agents",
        lambda **kw: [_agent("a1", "EXP_A"), _agent("a2", "EXP_B"), _agent("a3", "EXP_C")],
    )

    rc = job.main(["--warehouse-id=wh", "--catalog=cat", "--schema=sch", "--objective-metric="])
    # B failed mid-loop, but A and C were STILL reviewed; worst_rc non-zero.
    assert attempted == ["EXP_A", "EXP_B", "EXP_C"]
    assert rc == 1


def test_registry_mode_empty_is_clean_no_op(
    registry_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job, "load_registered_agents", lambda **kw: [])
    rc = job.main(["--warehouse-id=wh", "--catalog=cat", "--schema=sch"])
    assert rc == 0
    assert registry_capture["calls"] == []  # no fabricated review work


def test_registry_read_error_propagates(
    registry_capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**kw: Any) -> Any:
        raise RuntimeError("permission denied on agent_registry")

    monkeypatch.setattr(job, "load_registered_agents", boom)
    with pytest.raises(RuntimeError, match="permission denied"):
        job.main(["--warehouse-id=wh", "--catalog=cat", "--schema=sch"])


def test_registry_mode_requires_catalog_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job, "resolve_job_auth", lambda **kw: "minted")
    monkeypatch.setattr(job, "load_registered_agents", lambda **kw: pytest.fail("should not read"))
    rc = job.main(["--warehouse-id=wh"])  # no catalog/schema
    assert rc == 2
