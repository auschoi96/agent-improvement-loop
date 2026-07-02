"""Offline tests for the scheduled RLM job wrapper (:mod:`ail.jobs.continuous_rlm`).

Auth (``resolve_job_auth``) and the whole review pass (``run_continuous_rlm``) are
stubbed, so the wrapper's own responsibilities are exercised without a workspace,
HALO, or a model: the ``databricks-gpt-5-5-pro`` judge default, goal-derived-rubric
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
    def test_defaults_to_gpt_5_5_pro(self, captured: dict[str, Any]) -> None:
        rc = job.main(["--experiment=EXP1", "--warehouse-id=wh-1", "--objective-metric="])
        assert rc == 0
        assert job.DEFAULT_JUDGE_MODEL == "databricks-gpt-5-5-pro"
        assert captured["kwargs"]["judge_model"] == "databricks-gpt-5-5-pro"

    def test_judge_model_is_overridable(self, captured: dict[str, Any]) -> None:
        job.main(
            [
                "--experiment=EXP1",
                "--warehouse-id=wh-1",
                "--judge-model=databricks-gpt-5-5-pro-preview",
                "--objective-metric=",
            ]
        )
        assert captured["kwargs"]["judge_model"] == "databricks-gpt-5-5-pro-preview"


class TestGoalSteering:
    def test_goal_configured_builds_goal_derived_rubric(self, captured: dict[str, Any]) -> None:
        job.main(
            [
                "--experiment=EXP1",
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
        job.main(["--experiment=EXP1", "--warehouse-id=wh-1", "--objective-metric="])
        assert captured["kwargs"]["rubric"] is DEFAULT_RUBRIC

    def test_unmapped_objective_metric_fails_loud(self, captured: dict[str, Any]) -> None:
        from ail.goals.compiler import UnmappedMetricError

        with pytest.raises(UnmappedMetricError):
            job.main(
                [
                    "--experiment=EXP1",
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
                    "--warehouse-id=wh-1",
                    "--objective-metric=",
                    "--reasoning-effort=banana",
                ]
            )


def test_requires_warehouse_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIL_WAREHOUSE_ID", raising=False)
    with pytest.raises(SystemExit):
        job.main(["--experiment=EXP1"])
