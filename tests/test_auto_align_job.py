"""Tests for the auto-align Job entrypoint (:mod:`ail.jobs.auto_align_job`).

The entrypoint is a thin driver, so the two runtime concerns it owns are exercised
offline: ``resolve_job_auth`` and ``auto_align_scorers`` are monkeypatched (no
workspace, no models), and the environment is fully restored after each test so a
direct ``os.environ`` write (``MLFLOW_TRACING_SQL_WAREHOUSE_ID``) cannot leak.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.jobs import auto_align_job as job
from ail.judges.auto_align import AutoAlignReport, AutoAlignStatus, JudgeAutoAlignResult


@pytest.fixture(autouse=True)
def _restore_env() -> Any:
    """Fully restore ``os.environ`` after each test (main() writes to it directly)."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def _result(name: str, status: AutoAlignStatus) -> JudgeAutoAlignResult:
    return JudgeAutoAlignResult(
        judge_name=name,
        status=status,
        label_count=25,
        watermark=0,
        prior_agreement=None,
        promoted=status is AutoAlignStatus.ALIGNED,
    )


@pytest.fixture
def stub_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Neutralize auth and capture the auto_align_scorers call; return knobs."""
    captured: dict[str, Any] = {
        "auth": [],
        "call": None,
        "results": [_result("correctness", AutoAlignStatus.ALIGNED)],
    }

    monkeypatch.setattr(job, "resolve_job_auth", lambda **kw: captured["auth"].append(kw) or "test")

    def fake_auto_align_scorers(experiment_id: str, **kwargs: Any) -> AutoAlignReport:
        captured["call"] = {"experiment_id": experiment_id, **kwargs}
        return AutoAlignReport(
            experiment_id=experiment_id,
            results=tuple(captured["results"]),
            generated_at="2026-07-02T00:00:00+00:00",
        )

    monkeypatch.setattr(job, "auto_align_scorers", fake_auto_align_scorers)
    return captured


class TestArgparse:
    def test_requires_a_warehouse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AIL_WAREHOUSE_ID", raising=False)
        monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
        with pytest.raises(SystemExit) as exc:
            job._parse_args([])
        assert exc.value.code == 2  # argparse usage error

    def test_warehouse_from_env_satisfies_the_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TRACING_WAREHOUSE_ENV, "wh-env")
        args = job._parse_args([])  # no --warehouse-id, but env is set
        assert args.warehouse_id is None


class TestConfigBuilders:
    def test_build_config_maps_all_knobs(self) -> None:
        args = job._parse_args(
            [
                "--warehouse-id",
                "wh1",
                "--label-floor",
                "30",
                "--agreement-floor",
                "0.8",
                "--min-anchor-samples",
                "5",
                "--numeric-tolerance",
                "1.0",
                "--anchor-fraction",
                "0.25",
                "--sampling-rate",
                "0.2",
            ]
        )
        config = job._build_config(args)
        assert config.label_floor == 30
        assert config.agreement.floor == 0.8
        assert config.agreement.min_samples == 5
        assert config.agreement.numeric_tolerance == 1.0
        assert config.anchor_fraction == 0.25
        assert config.sampling_rate == 0.2

    def test_build_optimizer_none_without_reflection_lm(self) -> None:
        args = job._parse_args(["--warehouse-id", "wh1"])
        assert job._build_optimizer(args) is None

    def test_resolve_scorers_all_by_default(self) -> None:
        assert set(job._resolve_scorers("")) == {
            "correctness",
            "modularity",
            "groundedness",
            "token_efficiency",
        }

    def test_resolve_scorers_filters(self) -> None:
        assert set(job._resolve_scorers("correctness, modularity")) == {"correctness", "modularity"}

    def test_resolve_scorers_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown judge"):
            job._resolve_scorers("correctness,not_a_judge")


class TestMain:
    def test_success_sets_warehouse_env_and_returns_zero(
        self, stub_backend: dict[str, Any]
    ) -> None:
        code = job.main(["--warehouse-id", "wh1", "--experiment", "exp1"])
        assert code == 0
        # The v4 trace-store read finds its warehouse via the env var.
        assert os.environ[TRACING_WAREHOUSE_ENV] == "wh1"
        # Auth was resolved before the cadence ran.
        assert stub_backend["auth"]
        assert stub_backend["call"]["experiment_id"] == "exp1"
        assert stub_backend["call"]["register"] is True

    def test_no_register_flag_is_forwarded(self, stub_backend: dict[str, Any]) -> None:
        # Single-agent override: an explicit --experiment runs just that one.
        code = job.main(["--warehouse-id", "wh1", "--experiment", "exp1", "--no-register"])
        assert code == 0
        assert stub_backend["call"]["register"] is False

    def test_returns_one_when_a_judge_failed(self, stub_backend: dict[str, Any]) -> None:
        stub_backend["results"] = [
            _result("correctness", AutoAlignStatus.ALIGNED),
            _result("modularity", AutoAlignStatus.FAILED),
        ]
        code = job.main(["--warehouse-id", "wh1", "--experiment", "exp1"])
        assert code == 1  # a failed cadence is a non-zero exit

    def test_held_and_rolled_back_are_still_success(self, stub_backend: dict[str, Any]) -> None:
        # A correct hold / rollback is NOT a failure — the run succeeded.
        stub_backend["results"] = [
            _result("correctness", AutoAlignStatus.HELD_DISTRUSTED),
            _result("modularity", AutoAlignStatus.ROLLED_BACK),
        ]
        code = job.main(["--warehouse-id", "wh1", "--experiment", "exp1"])
        assert code == 0

    def test_unknown_judge_exits_two(self, stub_backend: dict[str, Any]) -> None:
        code = job.main(
            ["--warehouse-id", "wh1", "--experiment", "exp1", "--judges", "not_a_judge"]
        )
        assert code == 2


# -- registry (multi-agent) mode -------------------------------------------


def _agent(name: str, exp: str) -> Any:
    from ail.registry import Agent

    return Agent(agent_name=name, experiment_id=exp)


@pytest.fixture
def registry_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture EVERY per-agent auto_align_scorers call; drive per-agent fail behavior."""
    state: dict[str, Any] = {"experiments": [], "fail_on": set()}
    monkeypatch.setattr(job, "resolve_job_auth", lambda **kw: "test")

    def fake_auto_align_scorers(experiment_id: str, **kwargs: Any) -> AutoAlignReport:
        state["experiments"].append(experiment_id)
        n_failed = 1 if experiment_id in state["fail_on"] else 0
        results = [
            JudgeAutoAlignResult(
                judge_name="correctness",
                status=(AutoAlignStatus.FAILED if n_failed else AutoAlignStatus.ALIGNED),
                label_count=25,
                watermark=0,
                prior_agreement=None,
                promoted=not n_failed,
            )
        ]
        return AutoAlignReport(
            experiment_id=experiment_id,
            results=tuple(results),
            generated_at="2026-07-02T00:00:00+00:00",
        )

    monkeypatch.setattr(job, "auto_align_scorers", fake_auto_align_scorers)
    return state


class TestRegistryMode:
    def test_threads_each_agents_own_experiment(
        self, registry_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            job,
            "load_registered_agents",
            lambda **kw: [_agent("a1", "EXP_A"), _agent("a2", "EXP_B")],
        )
        code = job.main(["--warehouse-id", "wh1", "--catalog", "cat", "--schema", "sch"])
        assert code == 0
        # Each agent aligned on ITS OWN experiment, not a shared/global one.
        assert registry_stub["experiments"] == ["EXP_A", "EXP_B"]

    def test_isolation_one_agent_failure_continues(
        self, registry_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempted: list[str] = []

        def boom_middle(experiment_id: str, **kwargs: Any) -> AutoAlignReport:
            attempted.append(experiment_id)
            if experiment_id == "EXP_B":
                raise RuntimeError("MemAlign blew up for B")
            return AutoAlignReport(experiment_id=experiment_id, results=(), generated_at="t")

        monkeypatch.setattr(job, "auto_align_scorers", boom_middle)
        monkeypatch.setattr(
            job,
            "load_registered_agents",
            lambda **kw: [_agent("a1", "EXP_A"), _agent("a2", "EXP_B"), _agent("a3", "EXP_C")],
        )
        code = job.main(["--warehouse-id", "wh1", "--catalog", "cat", "--schema", "sch"])
        assert attempted == ["EXP_A", "EXP_B", "EXP_C"]  # all three attempted
        assert code == 1  # worst_rc non-zero

    def test_judge_failure_bumps_worst_rc_without_raising(
        self, registry_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A judge cadence FAILED for one agent (rc 1, no raise) -> worst_rc non-zero,
        # but every agent still ran.
        registry_stub["fail_on"] = {"EXP_B"}
        monkeypatch.setattr(
            job,
            "load_registered_agents",
            lambda **kw: [_agent("a1", "EXP_A"), _agent("a2", "EXP_B")],
        )
        code = job.main(["--warehouse-id", "wh1", "--catalog", "cat", "--schema", "sch"])
        assert registry_stub["experiments"] == ["EXP_A", "EXP_B"]
        assert code == 1

    def test_empty_registry_is_clean_no_op(
        self, registry_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(job, "load_registered_agents", lambda **kw: [])
        code = job.main(["--warehouse-id", "wh1", "--catalog", "cat", "--schema", "sch"])
        assert code == 0
        assert registry_stub["experiments"] == []  # no fabricated work

    def test_registry_read_error_propagates(
        self, registry_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**kw: Any) -> Any:
            raise RuntimeError("warehouse cannot be found")

        monkeypatch.setattr(job, "load_registered_agents", boom)
        with pytest.raises(RuntimeError, match="warehouse cannot be found"):
            job.main(["--warehouse-id", "wh1", "--catalog", "cat", "--schema", "sch"])

    def test_requires_catalog_schema(
        self, registry_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            job, "load_registered_agents", lambda **kw: pytest.fail("should not read")
        )
        code = job.main(["--warehouse-id", "wh1"])  # no catalog/schema
        assert code == 2
