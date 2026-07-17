from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ail.compare.contract import Recommendation
from ail.jobs import gepa_job
from ail.optimize.gepa_runner import GepaOptimizationResult
from ail.optimize.phase2 import L1Outcome, Phase2Artifact, TaskOutcome
from ail.registry import Agent, OptimizationTarget


def _args(*extra: str) -> Any:
    return gepa_job._parse_args(
        [
            "--agent",
            "claude_code",
            "--experiment-id",
            "subject-1",
            "--warehouse-id",
            "wh-1",
            "--catalog",
            "cat",
            "--schema",
            "sch",
            "--confirmed-costly-run",
            "true",
            *extra,
        ]
    )


def _agent(**updates: Any) -> Agent:
    values: dict[str, Any] = {
        "agent_name": "claude_code",
        "experiment_id": "subject-1",
        "reviewer_experiment_id": "reviewer-1",
        "target_workspace": "/tmp/agent",
        "optimization_target": OptimizationTarget(
            path=".claude/skills/token/SKILL.md",
            validation_command=["python", "-m", "pytest", "-q"],
        ),
    }
    values.update(updates)
    return Agent(**values)


def test_request_requires_explicit_cost_confirmation() -> None:
    args = _args()
    args.confirmed_costly_run = "false"
    with pytest.raises(ValueError, match="live and costly"):
        gepa_job._validate_request(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_metric_calls", 0, "between 1 and 500"),
        ("holdout_fraction", 1.0, "between 0 and 1"),
        ("max_train_tasks", 21, "between 1 and 20"),
        ("reflection_lm", "openai:/gpt", "databricks:/"),
    ],
)
def test_request_bounds_cost_and_model(field: str, value: Any, message: str) -> None:
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        gepa_job._validate_request(args)


def test_resolve_agent_rejects_dispatcher_experiment_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gepa_job, "resolve_registered_agent", lambda *a, **kw: _agent())
    args = _args()
    args.experiment_id = "wrong"
    with pytest.raises(ValueError, match="experiment mismatch"):
        gepa_job._resolve_agent(args)


def test_resolve_agent_requires_supported_adapter_and_separate_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gepa_job,
        "resolve_registered_agent",
        lambda *a, **kw: _agent(agent_name="codex"),
    )
    args = _args()
    args.agent = "codex"
    with pytest.raises(ValueError, match="currently supports only"):
        gepa_job._resolve_agent(args)

    monkeypatch.setattr(
        gepa_job,
        "resolve_registered_agent",
        lambda *a, **kw: _agent(reviewer_experiment_id=None),
    )
    args.agent = "claude_code"
    with pytest.raises(ValueError, match="no reviewer_experiment_id"):
        gepa_job._resolve_agent(args)


def test_resolve_agent_requires_local_apply_target_before_costly_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gepa_job,
        "resolve_registered_agent",
        lambda *a, **kw: _agent(optimization_target=None),
    )
    with pytest.raises(ValueError, match="no optimization_target"):
        gepa_job._resolve_agent(_args())


def _phase2(pct: float) -> Phase2Artifact:
    return Phase2Artifact(
        suite_version="suite-v1",
        suite_content_hash="abc",
        objective_metric="total_tokens",
        n_tasks=1,
        n_promote=1,
        realized_token_savings_absolute=pct,
        realized_token_savings_pct=pct,
        outcomes=[
            TaskOutcome(
                task_id="heldout-1",
                recommendation=Recommendation.PROMOTE,
                l1_outcome=L1Outcome.PASSED,
            )
        ],
    )


def test_approval_proposal_binds_diff_hashes_artifact_and_heldout_evidence() -> None:
    result = GepaOptimizationResult(
        generated_at="2026-07-16T00:00:00+00:00",
        changed=True,
        seed_skill_body="# Seed\n",
        evolved_skill_body="# Better\n",
        suite_version="suite-v1",
        suite_content_hash="abc",
        holdout_task_ids=["heldout-1"],
        holdout_evolved=_phase2(40.0),
        holdout_seed_baseline=_phase2(20.0),
    )
    proposal, reason = gepa_job._approval_proposal(
        agent=_agent(),
        result=result,
        mlflow_run_id="run-1",
        artifact_uri="runs:/run-1/gepa/gepa_candidate.json",
    )
    assert proposal is not None
    assert proposal.experiment_id == "subject-1"
    assert proposal.change.diff and "# Better" in proposal.change.diff
    spec = proposal.change.local_apply_spec
    assert spec is not None
    assert spec.target_path == ".claude/skills/token/SKILL.md"
    assert spec.artifact_uri == "runs:/run-1/gepa/gepa_candidate.json"
    assert spec.holdout_savings_delta_pct == 20.0
    assert proposal.proof is not None and proposal.proof.correctness_held
    assert "beats seed" in reason


def test_load_verify_specs_reads_packaged_run_plan(tmp_path: Path) -> None:
    (tmp_path / "run_plan.yaml").write_text(
        "task-1:\n  command: [python, -m, pytest]\n  timeout_seconds: 12\n",
        encoding="utf-8",
    )
    specs = gepa_job._load_verify_specs(tmp_path)
    assert specs["task-1"].command == ["python", "-m", "pytest"]
    assert specs["task-1"].timeout_seconds == 12


def test_run_logs_candidate_and_prints_machine_readable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    suite_dir = tmp_path / "eval" / "task_suite" / "phase2-mini"
    fixtures_dir = tmp_path / "eval" / "phase2_fixtures"
    suite_dir.mkdir(parents=True)
    fixtures_dir.mkdir(parents=True)
    (suite_dir / "tasks.yaml").write_text("frozen: true\n", encoding="utf-8")
    (tmp_path / "run_plan.yaml").write_text(
        "task-1:\n  command: [python, -m, pytest]\n", encoding="utf-8"
    )

    calls: dict[str, Any] = {}

    def auth(**_kwargs: Any) -> str:
        monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
        monkeypatch.setenv("DATABRICKS_TOKEN", "secret-token")
        return "minted"

    monkeypatch.setattr(gepa_job, "resolve_job_auth", auth)
    monkeypatch.setattr(gepa_job, "_resolve_agent", lambda _args: _agent())
    monkeypatch.setattr(gepa_job, "configure_monitoring_warehouse", lambda *a, **kw: None)
    monkeypatch.setattr(gepa_job, "_assets_root", lambda: tmp_path)
    suite = SimpleNamespace(version="phase2-mini-v1")
    monkeypatch.setattr(gepa_job, "load_task_suite", lambda *a, **kw: suite)
    monkeypatch.setattr(gepa_job, "_experiment_name", lambda _id: "/Shared/reviewer")

    class Adapter:
        def __init__(self, *, mlflow_experiment: str) -> None:
            calls["adapter_experiment"] = mlflow_experiment

    monkeypatch.setattr(gepa_job, "ClaudeCodeAdapter", Adapter)
    result = GepaOptimizationResult(
        suite_version="phase2-mini-v1",
        suite_content_hash="abc",
        max_metric_calls=6,
        changed=True,
        gepa_total_metric_calls=8,
        gepa_num_candidates=2,
        gepa_best_val_score=0.75,
    )

    def optimize(**kwargs: Any) -> GepaOptimizationResult:
        calls["optimize"] = kwargs
        return result

    monkeypatch.setattr(gepa_job, "run_gepa_optimization", optimize)
    monkeypatch.setattr(
        gepa_job,
        "_log_candidate",
        lambda **kw: ("mlflow-run-1", "runs:/mlflow-run-1/gepa/gepa_candidate.json"),
    )

    assert gepa_job.run(_args()) == 0
    assert calls["adapter_experiment"] == "/Shared/reviewer"
    assert calls["optimize"]["suite"] is suite
    assert calls["optimize"]["fixtures_root"] == str(tmp_path)
    assert calls["optimize"]["config"].max_train_tasks == 2
    assert calls["optimize"]["config"].max_metric_calls == 6
    assert calls["optimize"]["verify_specs"]["task-1"].name == "verify-task-1"
    assert gepa_job.RESULT_ARTIFACT_PATH == "gepa/gepa_candidate.json"
    assert __import__("os").environ["ANTHROPIC_MODEL"] == gepa_job.DEFAULT_AGENT_MODEL

    marker_line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(gepa_job.RESULT_MARKER)
    )
    marker = json.loads(marker_line.removeprefix(gepa_job.RESULT_MARKER))
    assert marker["mlflow_run_id"] == "mlflow-run-1"
    assert marker["candidate_changed"] is True
    assert marker["candidate_promoted"] is False
    assert marker["human_gate_required"] is True


def test_main_returns_nonzero_and_names_failure(capsys: pytest.CaptureFixture[str]) -> None:
    rc = gepa_job.main(
        [
            "--agent",
            "claude_code",
            "--experiment-id",
            "subject-1",
            "--warehouse-id",
            "wh-1",
            "--catalog",
            "cat",
            "--schema",
            "sch",
        ]
    )
    assert rc == 1
    assert "confirmed_costly_run" in capsys.readouterr().err
