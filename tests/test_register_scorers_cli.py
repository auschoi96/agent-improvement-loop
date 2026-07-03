from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ail.jobs import register_scorers_job as job
from ail.judges import registration


@dataclass(frozen=True)
class _FakeScorer:
    name: str


@dataclass(frozen=True)
class _FakeRegistration:
    scorer: _FakeScorer


def test_main_calls_register_scorers_with_parsed_kwargs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_register_scorers(experiment_id: str, **kwargs: Any) -> list[_FakeRegistration]:
        calls.append({"experiment_id": experiment_id, **kwargs})
        return [_FakeRegistration(_FakeScorer("correctness"))]

    monkeypatch.setattr(job, "register_scorers", fake_register_scorers)

    rc = job.main(
        [
            "--experiment-id",
            "exp-123",
            "--sampling-rate",
            "0.35",
            "--model",
            "databricks:/judge",
            "--filter-string",
            "status = 'OK'",
            "--profile",
            "prod",
            "--tracking-uri",
            "databricks://prod",
            "--registry-uri",
            "databricks-uc://prod",
        ]
    )

    assert rc == 0
    assert calls == [
        {
            "experiment_id": "exp-123",
            "sampling_rate": 0.35,
            "model": "databricks:/judge",
            "filter_string": "status = 'OK'",
            "profile": "prod",
            "tracking_uri": "databricks://prod",
            "registry_uri": "databricks-uc://prod",
        }
    ]
    out = capsys.readouterr().out
    assert "registered scorer count : 1" in out
    assert "correctness" in out


def test_missing_experiment_id_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        job.main([])

    assert exc.value.code == 2


def test_sampling_rate_value_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_register_scorers(experiment_id: str, **kwargs: Any) -> list[_FakeRegistration]:
        calls.append({"experiment_id": experiment_id, **kwargs})
        return [_FakeRegistration(_FakeScorer("groundedness"))]

    monkeypatch.setattr(job, "register_scorers", fake_register_scorers)

    rc = job.main(["--experiment-id", "exp-123", "--sampling-rate", "0.72"])

    assert rc == 0
    assert calls[0]["sampling_rate"] == 0.72


def test_default_sampling_rate_comes_from_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_register_scorers(experiment_id: str, **kwargs: Any) -> list[_FakeRegistration]:
        calls.append({"experiment_id": experiment_id, **kwargs})
        return [_FakeRegistration(_FakeScorer("token_efficiency"))]

    monkeypatch.setattr(job, "register_scorers", fake_register_scorers)

    rc = job.main(["--experiment-id", "exp-123"])

    assert rc == 0
    assert calls[0]["sampling_rate"] == registration.DEFAULT_SAMPLING_RATE
