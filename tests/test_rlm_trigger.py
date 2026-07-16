"""Load-bearing properties of the RLM trigger reconcile (:mod:`ail.jobs.rlm_trigger`).

These pin the invariants the two callers (onboarding + deploy heal) depend on:

* the spans table is derived from the annotations table by a literal suffix swap, and
  an underivable agent is reported, never fabricated;
* the reconcile is ADD-ONLY (never drops a hand-added or now-orphaned table) and
  issues a ``jobs.update`` ONLY when the watched set actually grew;
* a partial update preserves every other job setting (schedule, queue, debounce);
* a job with no ``table_update`` trigger is refused rather than reshaped at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ail.jobs.rlm_trigger import (
    reconcile_rlm_trigger_tables,
    spans_table_for_agent,
)
from ail.registry import Agent

# --- minimal fakes mirroring the databricks-sdk jobs shapes ----------------


@dataclass
class _FakeTableUpdate:
    table_names: list[str]
    min_time_between_triggers_seconds: int = 60
    wait_after_last_change_seconds: int = 61


@dataclass
class _FakeTrigger:
    table_update: _FakeTableUpdate | None = None
    pause_status: str = "UNPAUSED"


@dataclass
class _FakeSettings:
    name: str = "ail-continuous-rlm-trace-arrival"
    trigger: _FakeTrigger | None = None
    # A stand-in for the rest of the JobSettings we must preserve untouched.
    schedule: Any = None
    queue: Any = field(default_factory=lambda: {"enabled": False})
    max_concurrent_runs: int = 1


@dataclass
class _FakeJob:
    settings: _FakeSettings | None


class _FakeJobsApi:
    """Records get/update calls; returns the settings it was seeded with."""

    def __init__(self, settings: _FakeSettings | None) -> None:
        self._settings = settings
        self.get_calls: list[int] = []
        self.update_calls: list[tuple[int, _FakeSettings]] = []

    def get(self, job_id: int) -> _FakeJob:
        self.get_calls.append(job_id)
        return _FakeJob(settings=self._settings)

    def update(self, job_id: int, *, new_settings: _FakeSettings) -> None:
        self.update_calls.append((job_id, new_settings))


class _FakeClient:
    def __init__(self, settings: _FakeSettings | None) -> None:
        self.jobs = _FakeJobsApi(settings)


def _client_with(table_names: list[str]) -> _FakeClient:
    return _FakeClient(
        _FakeSettings(trigger=_FakeTrigger(table_update=_FakeTableUpdate(table_names=list(table_names))))
    )


def _agent(name: str, annotations_table: str | None) -> Agent:
    return Agent(agent_name=name, experiment_id=f"exp-{name}", annotations_table=annotations_table)


# --- spans_table_for_agent -------------------------------------------------


def test_spans_table_is_annotations_table_with_suffix_swapped() -> None:
    agent = _agent("claude_code", "cat.mlflow_traces.claude_code_otel_annotations")
    assert spans_table_for_agent(agent) == "cat.mlflow_traces.claude_code_otel_spans"


def test_spans_table_none_when_no_annotations_table() -> None:
    assert spans_table_for_agent(_agent("a", None)) is None
    assert spans_table_for_agent(_agent("a", "   ")) is None


def test_spans_table_none_on_unexpected_suffix() -> None:
    # A table that does not end in _otel_annotations is not swappable — fail closed
    # rather than emit a bogus watch target.
    assert spans_table_for_agent(_agent("a", "cat.sch.some_other_table")) is None


# --- reconcile: add-only, no-op-when-unchanged -----------------------------


def test_adds_new_spans_table_and_issues_single_update() -> None:
    client = _client_with(["cat.mlflow_traces.claude_code_otel_spans"])
    new = _agent("newbot", "cat.mlflow_traces.newbot_otel_annotations")
    result = reconcile_rlm_trigger_tables(client, rlm_job_id=42, agents=[new])

    assert result.updated is True
    assert result.added == ["cat.mlflow_traces.newbot_otel_spans"]
    assert len(client.jobs.update_calls) == 1
    job_id, settings = client.jobs.update_calls[0]
    assert job_id == 42
    # The watched list now contains BOTH the pre-existing and the new table (add-only).
    assert settings.trigger.table_update.table_names == [
        "cat.mlflow_traces.claude_code_otel_spans",
        "cat.mlflow_traces.newbot_otel_spans",
    ]


def test_no_update_when_table_already_watched() -> None:
    client = _client_with(["cat.mlflow_traces.claude_code_otel_spans"])
    existing = _agent("claude_code", "cat.mlflow_traces.claude_code_otel_annotations")
    result = reconcile_rlm_trigger_tables(client, rlm_job_id=42, agents=[existing])

    assert result.updated is False
    assert result.added == []
    assert result.already_watched == ["cat.mlflow_traces.claude_code_otel_spans"]
    assert client.jobs.update_calls == []  # a quiet no-op — never a redundant write


def test_add_only_never_drops_an_unattributed_table() -> None:
    # A hand-added table (or one for an agent since removed from the registry) must
    # survive a reconcile that only knows about a different agent.
    client = _client_with(["cat.mlflow_traces.handadded_otel_spans"])
    new = _agent("newbot", "cat.mlflow_traces.newbot_otel_annotations")
    result = reconcile_rlm_trigger_tables(client, rlm_job_id=1, agents=[new])

    assert result.updated is True
    _job_id, settings = client.jobs.update_calls[0]
    assert "cat.mlflow_traces.handadded_otel_spans" in settings.trigger.table_update.table_names
    assert "cat.mlflow_traces.newbot_otel_spans" in settings.trigger.table_update.table_names


def test_underivable_agent_reported_not_fabricated() -> None:
    client = _client_with(["cat.mlflow_traces.claude_code_otel_spans"])
    result = reconcile_rlm_trigger_tables(client, rlm_job_id=1, agents=[_agent("bare", None)])

    assert result.updated is False
    assert result.underivable == ["bare"]
    assert client.jobs.update_calls == []


def test_preserves_other_settings_on_update() -> None:
    client = _client_with(["cat.mlflow_traces.claude_code_otel_spans"])
    client.jobs._settings.max_concurrent_runs = 1
    client.jobs._settings.schedule = None
    new = _agent("newbot", "cat.mlflow_traces.newbot_otel_annotations")
    reconcile_rlm_trigger_tables(client, rlm_job_id=1, agents=[new])

    _job_id, settings = client.jobs.update_calls[0]
    # The partial update carries the SAME settings object with only table_names grown —
    # schedule stays None, queue/debounce untouched.
    assert settings.schedule is None
    assert settings.queue == {"enabled": False}
    assert settings.max_concurrent_runs == 1
    assert settings.trigger.table_update.wait_after_last_change_seconds == 61


def test_dedupes_multiple_agents_and_batches_one_update() -> None:
    client = _client_with([])
    agents = [
        _agent("a", "cat.mlflow_traces.a_otel_annotations"),
        _agent("b", "cat.mlflow_traces.b_otel_annotations"),
    ]
    result = reconcile_rlm_trigger_tables(client, rlm_job_id=7, agents=agents)

    assert result.updated is True
    assert result.added == [
        "cat.mlflow_traces.a_otel_spans",
        "cat.mlflow_traces.b_otel_spans",
    ]
    # All agents reconciled in ONE update, not one write per agent.
    assert len(client.jobs.update_calls) == 1


def test_refuses_job_without_table_update_trigger() -> None:
    # A job whose trigger kind is not table_update must not be reshaped at runtime.
    client = _FakeClient(_FakeSettings(trigger=_FakeTrigger(table_update=None)))
    with pytest.raises(ValueError, match="no table_update trigger"):
        reconcile_rlm_trigger_tables(
            client, rlm_job_id=1, agents=[_agent("a", "c.s.a_otel_annotations")]
        )


def test_refuses_job_without_settings() -> None:
    client = _FakeClient(None)
    with pytest.raises(ValueError, match="no settings"):
        reconcile_rlm_trigger_tables(
            client, rlm_job_id=1, agents=[_agent("a", "c.s.a_otel_annotations")]
        )
