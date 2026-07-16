"""Deterministic MLflow production scorers and their registration lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlflow
import pytest
from mlflow.genai.scorers import Scorer

from ail.metrics import mlflow_scorers as metrics


def test_deterministic_scorers_match_l0_facts(synthetic_trace: Any) -> None:
    assert metrics.duration_seconds_scorer(trace=synthetic_trace).value == 4.2
    assert metrics.total_tokens_scorer(trace=synthetic_trace).value == 1200
    assert metrics.total_tool_calls_scorer(trace=synthetic_trace).value == 2
    assert metrics.redundancy_rate_scorer(trace=synthetic_trace).value == 0.0
    # claude-opus-4-8: 1,000 input × $5/M + 200 output × $25/M.
    assert metrics.total_usd_scorer(trace=synthetic_trace).value == 0.01


def test_redundancy_rate_counts_exact_repeated_tool_inputs() -> None:
    spans = [
        SimpleNamespace(
            span_type="TOOL",
            name="tool_Read",
            attributes={"tool_name": "Read"},
            inputs={"file_path": "/a"},
        ),
        SimpleNamespace(
            span_type="TOOL",
            name="tool_Read",
            attributes={"tool_name": "Read"},
            inputs={"file_path": "/a"},
        ),
        SimpleNamespace(
            span_type="TOOL",
            name="tool_Read",
            attributes={"tool_name": "Read"},
            inputs={"file_path": "/b"},
        ),
    ]
    trace = SimpleNamespace(search_spans=lambda: spans)
    result = metrics.redundancy_rate_scorer(trace=trace)
    assert result.value == round(1 / 3, 6)


def test_every_scorer_round_trips_through_production_serialization(
    synthetic_trace: Any,
) -> None:
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri("databricks")
    try:
        for name, definition in metrics.DETERMINISTIC_MLFLOW_SCORERS.items():
            restored = Scorer.model_validate(definition.model_dump())
            assert restored.name == name
            assert restored(trace=synthetic_trace).value == definition(trace=synthetic_trace).value
    finally:
        mlflow.set_tracking_uri(original_uri)


class _Registered:
    def __init__(self, calls: list[tuple[str, str, Any]]) -> None:
        self.calls = calls

    def start(self, *, experiment_id: str, sampling_config: Any) -> _Registered:
        self.calls.append(("start", experiment_id, sampling_config))
        return self


class _Definition:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, str, Any]] = []

    def register(self, *, name: str, experiment_id: str) -> _Registered:
        self.calls.append(("register", experiment_id, name))
        return _Registered(self.calls)

    def update(self, *, name: str, experiment_id: str, sampling_config: Any) -> _Definition:
        self.calls.append(("update", experiment_id, sampling_config))
        return self


@pytest.fixture
def registration_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[_Definition, list[str]]:
    definition = _Definition("duration_seconds")
    activated: list[str] = []
    monkeypatch.setattr(metrics, "_require_databricks_agents", lambda: None)
    monkeypatch.setattr(metrics, "_configure_databricks", lambda **_: None)
    monkeypatch.setattr(metrics, "DETERMINISTIC_MLFLOW_SCORERS", {"duration_seconds": definition})
    monkeypatch.setattr(
        mlflow,
        "set_experiment",
        lambda *, experiment_id: activated.append(experiment_id),
    )
    return definition, activated


def test_registers_then_starts_new_code_scorer(
    monkeypatch: pytest.MonkeyPatch,
    registration_backend: tuple[_Definition, list[str]],
) -> None:
    definition, activated = registration_backend
    monkeypatch.setattr("mlflow.genai.scorers.list_scorers", lambda **_: [])

    names = metrics.register_deterministic_scorers("exp-1", ["duration_seconds"])

    assert names == ["duration_seconds"]
    assert activated == ["exp-1"]
    assert [call[0] for call in definition.calls] == ["register", "start"]
    assert definition.calls[1][2].sample_rate == 1.0


def test_updates_existing_code_scorer_idempotently(
    monkeypatch: pytest.MonkeyPatch,
    registration_backend: tuple[_Definition, list[str]],
) -> None:
    definition, _ = registration_backend
    monkeypatch.setattr(
        "mlflow.genai.scorers.list_scorers",
        lambda **_: [SimpleNamespace(name="duration_seconds")],
    )

    metrics.register_deterministic_scorers("exp-1", ["duration_seconds"], sampling_rate=0.5)

    assert [call[0] for call in definition.calls] == ["update"]
    assert definition.calls[0][2].sample_rate == 0.5


def test_registration_persists_monitoring_warehouse(
    monkeypatch: pytest.MonkeyPatch,
    registration_backend: tuple[_Definition, list[str]],
) -> None:
    configured: list[tuple[str, str]] = []
    monkeypatch.setattr("mlflow.genai.scorers.list_scorers", lambda **_: [])
    monkeypatch.setattr(
        "mlflow.tracing.set_databricks_monitoring_sql_warehouse_id",
        lambda *, sql_warehouse_id, experiment_id: configured.append(
            (sql_warehouse_id, experiment_id)
        ),
    )

    metrics.register_deterministic_scorers("exp-1", ["duration_seconds"], warehouse_id="wh-1")

    assert configured == [("wh-1", "exp-1")]


def test_registration_rejects_unknown_metric(
    registration_backend: tuple[_Definition, list[str]],
) -> None:
    with pytest.raises(ValueError, match="unknown deterministic"):
        metrics.register_deterministic_scorers("exp-1", ["invented_metric"])


def test_empty_registration_is_a_dependency_free_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metrics,
        "_require_databricks_agents",
        lambda: (_ for _ in ()).throw(AssertionError("must not load optional backend")),
    )

    assert metrics.register_deterministic_scorers("exp-1", []) == []
