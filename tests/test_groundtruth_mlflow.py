"""Tests for the optional MLflow logging of candidate executions.

The offline test exercises the real MLflow logging path against a local
``file://`` tracking store (no Databricks, runs in CI). It asserts the audit
record carries the agent's own response and — critically — **no expectations**.
The live test hits Databricks-managed MLflow and is gated behind env vars so the
default suite stays green offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ail.groundtruth.capture import capture_candidates
from ail.groundtruth.execute import MLFLOW_STAGE_TAG, execute_candidate, log_candidate_run
from ail.ingest.base import (
    AgentAdapter,
    AgentRunResult,
    AgentTask,
    NormalizedTrace,
    TraceStatus,
)


class FakeAdapter(AgentAdapter):
    name = "fake"

    def run(self, task: AgentTask) -> AgentRunResult:
        trace = NormalizedTrace(trace_id="tr-exec-1", producer=self.name, model="claude-sonnet-4-6")
        return AgentRunResult(trace=trace, output_text="def add(a, b): return a + b", duration_ms=5)


def _trace() -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="tr-1",
        status=TraceStatus.OK,
        producer="claude_code",
        request_preview="Write an add(a, b) function",
    )


def _offline_experiment(tmp_path: Path) -> str:
    """Configure an offline MLflow backend and return a ready experiment name.

    MLflow 3 retired the bare filesystem store, so the offline backend is SQLite
    (no server, runs in CI). The experiment is pre-created with an artifact
    location under ``tmp_path`` so nothing is written into the working tree.
    """
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    name = "ail-groundtruth-test"
    mlflow.create_experiment(name, artifact_location=(tmp_path / "artifacts").as_uri())
    return name


def test_log_candidate_run_offline_file_store(tmp_path: Path) -> None:
    import mlflow

    experiment = _offline_experiment(tmp_path)

    [candidate] = capture_candidates([_trace()])
    executed = execute_candidate(candidate, FakeAdapter())
    run_id = log_candidate_run(executed, experiment=experiment)

    assert run_id is not None
    run = mlflow.get_run(run_id)

    # The agent's own output is recorded for audit…
    assert run.data.params["case_id"] == executed.case_id
    assert run.data.params["producer"] == "fake"
    assert run.data.tags["ail.stage"] == MLFLOW_STAGE_TAG
    assert run.data.tags["ail.groundtruth.review_status"] == "candidate"

    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
    assert "candidate_response.txt" in artifacts
    assert "task_prompt.txt" in artifacts

    # …and NOTHING expectation-shaped is ever logged (no synthesis to audit).
    all_keys = list(run.data.params) + list(run.data.tags) + list(artifacts)
    assert not any("expectation" in k.lower() for k in all_keys)


def test_log_candidate_run_without_execution_is_noop(tmp_path: Path) -> None:
    experiment = _offline_experiment(tmp_path)
    [candidate] = capture_candidates([_trace()])  # never executed -> no candidate response
    assert log_candidate_run(candidate, experiment=experiment) is None


@pytest.mark.live
def test_log_candidate_run_databricks_managed_mlflow() -> None:
    """Log a capture run to Databricks-managed MLflow.

    Gated by ``AIL_LIVE_MLFLOW=1`` and a writable ``AIL_MLFLOW_EXPERIMENT`` (an
    experiment id or path) so the default suite stays offline and no reference
    experiment is polluted by accident. Profile defaults to ``dais-demo``.
    """
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        pytest.skip("set AIL_LIVE_MLFLOW=1 to run the live MLflow logging test")
    experiment = os.environ.get("AIL_MLFLOW_EXPERIMENT")
    if not experiment:
        pytest.skip("set AIL_MLFLOW_EXPERIMENT to a writable experiment id/path")

    import mlflow

    profile = os.environ.get("AIL_DATABRICKS_PROFILE", "dais-demo")
    mlflow.set_tracking_uri(f"databricks://{profile}")

    class _LiveAdapter(AgentAdapter):
        name = "fake"

        def run(self, task: AgentTask) -> AgentRunResult:
            trace: NormalizedTrace = NormalizedTrace(trace_id="tr-live", producer=self.name)
            return AgentRunResult(trace=trace, output_text="ok", duration_ms=1)

    [candidate] = capture_candidates([_trace()])
    executed = execute_candidate(candidate, _LiveAdapter())
    run_id = log_candidate_run(executed, experiment=experiment)
    assert run_id

    run: Any = mlflow.get_run(run_id)
    assert run.data.tags["ail.stage"] == MLFLOW_STAGE_TAG
