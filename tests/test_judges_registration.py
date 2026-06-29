"""Tests for scheduled-scorer registration (:mod:`ail.judges.registration`).

The orchestration is exercised offline by replacing the model-touching pieces:
``make_scorer`` returns a scripted fake judge, and the ``mlflow.genai.scorers``
backend calls are monkeypatched, so no model is called and no workspace is hit.
The one genuinely-live path (reading scorers off a real experiment) is gated
behind ``@pytest.mark.live`` and self-skips without a workspace + databricks-agents.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest

from ail.judges import registration as reg


class _FakeScorer:
    """Duck-typed stand-in for a registered/scheduled MLflow ``Scorer``."""

    def __init__(self, name: str, *, sample_rate: float | None = None) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, Any, Any]] = []

    def register(self, *, experiment_id: str | None = None, name: str | None = None) -> _FakeScorer:
        self.calls.append(("register", experiment_id, None))
        return self

    def start(
        self,
        *,
        experiment_id: str | None = None,
        name: str | None = None,
        sampling_config: Any = None,
    ) -> _FakeScorer:
        self.calls.append(("start", experiment_id, sampling_config))
        return _FakeScorer(self.name, sample_rate=sampling_config.sample_rate)

    def stop(self, *, experiment_id: str | None = None, name: str | None = None) -> _FakeScorer:
        self.calls.append(("stop", experiment_id, None))
        return self


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeScorer]:
    """Neutralize the backend so registration runs offline; return built judges."""
    monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
    monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
    built: dict[str, _FakeScorer] = {}

    def fake_make_scorer(spec: Any, *, model: str | None = None, **kw: Any) -> _FakeScorer:
        built[spec.name] = _FakeScorer(spec.name)
        return built[spec.name]

    monkeypatch.setattr(reg, "make_scorer", fake_make_scorer)
    return built


class TestRegisterScorers:
    def test_registers_then_starts_each_scorer(self, offline: dict[str, _FakeScorer]) -> None:
        active = reg.register_scorers("exp1", sampling_rate=0.25)
        assert {s.name for s in active} == {"correctness", "modularity", "groundedness"}
        # Every active scorer carries the requested sampling rate.
        assert all(s.sample_rate == 0.25 for s in active)
        # Each built judge was registered *then* started, in that order.
        for judge in offline.values():
            assert [c[0] for c in judge.calls] == ["register", "start"]
            start_call = next(c for c in judge.calls if c[0] == "start")
            assert start_call[1] == "exp1"
            assert start_call[2].sample_rate == 0.25

    def test_default_sampling_rate_is_conservative(self, offline: dict[str, _FakeScorer]) -> None:
        # The default must be a fraction, never an implicit 100% (cost lever).
        assert 0.0 < reg.DEFAULT_SAMPLING_RATE < 1.0
        active = reg.register_scorers("exp1")
        assert all(s.sample_rate == reg.DEFAULT_SAMPLING_RATE for s in active)

    @pytest.mark.parametrize("bad_rate", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_sampling_rate(
        self, offline: dict[str, _FakeScorer], bad_rate: float
    ) -> None:
        with pytest.raises(ValueError, match="sampling_rate"):
            reg.register_scorers("exp1", sampling_rate=bad_rate)

    def test_rejects_empty_scorer_set(self, offline: dict[str, _FakeScorer]) -> None:
        with pytest.raises(ValueError, match="no scorers"):
            reg.register_scorers("exp1", scorers={})

    def test_passes_filter_string_through(self, offline: dict[str, _FakeScorer]) -> None:
        reg.register_scorers("exp1", sampling_rate=0.5, filter_string="status = 'OK'")
        judge = next(iter(offline.values()))
        start_call = next(c for c in judge.calls if c[0] == "start")
        assert start_call[2].filter_string == "status = 'OK'"


class TestRequireDatabricksAgents:
    def test_missing_package_raises_actionable_importerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ImportError, match="databricks-agents"):
            reg.register_scorers("exp1")


class TestListAndUnregister:
    def test_list_registered_scorers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
        monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
        import mlflow.genai.scorers as msc

        fakes = [_FakeScorer("correctness", sample_rate=0.1), _FakeScorer("modularity")]
        monkeypatch.setattr(msc, "list_scorers", lambda *, experiment_id=None: fakes)
        result = reg.list_registered_scorers("exp1")
        assert [s.name for s in result] == ["correctness", "modularity"]

    def test_unregister_stops_then_deletes_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
        monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
        import mlflow.genai.scorers as msc

        fakes = [_FakeScorer("correctness"), _FakeScorer("modularity")]
        deleted: list[str] = []
        monkeypatch.setattr(msc, "list_scorers", lambda *, experiment_id=None: fakes)
        monkeypatch.setattr(
            msc,
            "delete_scorer",
            lambda *, name, experiment_id=None, version=None: deleted.append(name),
        )
        removed = reg.unregister_scorers("exp1", names=["correctness"])
        assert removed == ["correctness"]
        assert deleted == ["correctness"]
        # The matching scorer was stopped before deletion; the other was untouched.
        assert ("stop", "exp1", None) in fakes[0].calls
        assert fakes[1].calls == []

    def test_unregister_all_when_names_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
        monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
        import mlflow.genai.scorers as msc

        fakes = [_FakeScorer("a"), _FakeScorer("b")]
        deleted: list[str] = []
        monkeypatch.setattr(msc, "list_scorers", lambda *, experiment_id=None: fakes)
        monkeypatch.setattr(
            msc,
            "delete_scorer",
            lambda *, name, experiment_id=None, version=None: deleted.append(name),
        )
        removed = reg.unregister_scorers("exp1")
        assert set(removed) == {"a", "b"}
        assert set(deleted) == {"a", "b"}


@pytest.mark.live
def test_live_list_registered_scorers() -> None:
    """Acceptance (live, read-only): list scorers off a real experiment.

    Gated by ``AIL_LIVE_MLFLOW=1`` + ``AIL_LIVE_EXPERIMENT_ID`` and the presence
    of ``databricks-agents`` + a workspace. Read-only: it never registers or
    deletes, so running it does not mutate the live experiment.
    """
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        pytest.skip("set AIL_LIVE_MLFLOW=1 to run the live scorer listing")
    pytest.importorskip("databricks.agents", reason="scheduled scorers need databricks-agents")
    experiment_id = os.environ.get("AIL_LIVE_EXPERIMENT_ID")
    if not experiment_id:
        pytest.skip("set AIL_LIVE_EXPERIMENT_ID to the target experiment")

    scorers = reg.list_registered_scorers(experiment_id)
    assert isinstance(scorers, list)
