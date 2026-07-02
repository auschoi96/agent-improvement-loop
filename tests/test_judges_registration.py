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
from ail.judges.contract import AlignmentReport
from ail.judges.scorers import TOKEN_EFFICIENCY
from ail.pools import AlignmentSet


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


class _FakeAlignableScorer(_FakeScorer):
    """A fake judge that also records ``align`` calls (for the align-then-register path)."""

    def __init__(self, name: str, *, sample_rate: float | None = None) -> None:
        super().__init__(name, sample_rate=sample_rate)
        self.align_calls: list[dict[str, Any]] = []

    def align(self, traces: list[Any], optimizer: Any = None) -> _FakeAlignableScorer:
        self.align_calls.append({"traces": list(traces), "optimizer": optimizer})
        return _FakeAlignableScorer(f"{self.name}+aligned")


class _Trace:
    """Minimal MLflow-trace shape (``info.trace_id``) for an AlignmentSet."""

    def __init__(self, trace_id: str) -> None:
        self.info = type("Info", (), {"trace_id": trace_id})()


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeScorer]:
    """Neutralize the backend so registration runs offline; return built judges."""
    monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
    monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
    # The aligned/unaligned experiment tag is a best-effort MLflow side effect;
    # neutralize it so offline registration never touches a tracking backend.
    monkeypatch.setattr(reg, "_tag_alignment", lambda *a, **kw: True)
    built: dict[str, _FakeScorer] = {}

    def fake_make_scorer(spec: Any, *, model: str | None = None, **kw: Any) -> _FakeScorer:
        built[spec.name] = _FakeScorer(spec.name)
        return built[spec.name]

    monkeypatch.setattr(reg, "make_scorer", fake_make_scorer)
    return built


class TestRegisterScorers:
    def test_registers_then_starts_each_scorer(self, offline: dict[str, _FakeScorer]) -> None:
        active = reg.register_scorers("exp1", sampling_rate=0.25)
        assert {r.scorer.name for r in active} == {
            "correctness",
            "modularity",
            "groundedness",
            "token_efficiency",
        }
        # Every active scorer carries the requested sampling rate.
        assert all(r.scorer.sample_rate == 0.25 for r in active)
        # With no alignment_set, every scorer is the base judge, flagged unaligned.
        assert all(r.aligned is False for r in active)
        assert all(r.report.aligned is False for r in active)
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
        assert all(r.scorer.sample_rate == reg.DEFAULT_SAMPLING_RATE for r in active)

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

    def test_alignment_set_aligns_every_scorer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # MemAlign by construction: a labeled set aligns ALL scorers, not just one.
        monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
        monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
        monkeypatch.setattr(reg, "_tag_alignment", lambda *a, **kw: True)
        monkeypatch.setattr(
            reg, "make_scorer", lambda spec, *, model=None, **kw: _FakeAlignableScorer(spec.name)
        )
        sentinel = object()
        monkeypatch.setattr(reg, "build_memalign_optimizer", lambda *a, **kw: sentinel)

        aset = AlignmentSet.of([_Trace("t1"), _Trace("t2")])
        active = reg.register_scorers("exp1", alignment_set=aset)
        assert active  # all four
        assert all(r.aligned is True for r in active)
        assert all(r.report.n_alignment_traces == 2 for r in active)
        assert all(r.scorer.name.endswith("+aligned") for r in active)


class TestCreateAlignedScorer:
    """The MemAlign-aware align-then-register flow (Part 2)."""

    @pytest.fixture
    def backend(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Neutralize the backend; capture tag calls and the built base judge."""
        monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
        monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
        captured: dict[str, Any] = {"tags": [], "base": None}

        def fake_tag(experiment_id: str, name: str, aligned: bool) -> bool:
            captured["tags"].append((experiment_id, name, aligned))
            return True

        def fake_make_scorer(spec: Any, *, model: str | None = None, **kw: Any) -> Any:
            captured["base"] = _FakeAlignableScorer(spec.name)
            return captured["base"]

        monkeypatch.setattr(reg, "_tag_alignment", fake_tag)
        monkeypatch.setattr(reg, "make_scorer", fake_make_scorer)
        return captured

    def test_aligns_and_registers_aligned_judge_when_labels_present(
        self, backend: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel_opt = object()
        monkeypatch.setattr(reg, "build_memalign_optimizer", lambda *a, **kw: sentinel_opt)
        aset = AlignmentSet.of([_Trace("t1"), _Trace("t2"), _Trace("t3")])

        result = reg.create_aligned_scorer(
            TOKEN_EFFICIENCY, experiment_id="exp1", alignment_set=aset
        )

        base = backend["base"]
        # The judge was aligned with the default MemAlign optimizer ...
        assert len(base.align_calls) == 1
        assert base.align_calls[0]["optimizer"] is sentinel_opt
        assert base.align_calls[0]["traces"] == list(aset.traces)
        # ... and the ALIGNED judge (not the base) was the one registered.
        assert result.aligned is True
        assert result.report.aligned is True
        assert result.report.n_alignment_traces == 3
        assert result.judge.name == "token_efficiency+aligned"
        assert backend["tags"][-1] == ("exp1", "token_efficiency", True)

    def test_explicit_optimizer_is_used_without_building_default(
        self, backend: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("build_memalign_optimizer must not be called when one is passed")

        monkeypatch.setattr(reg, "build_memalign_optimizer", boom)
        my_opt = object()
        aset = AlignmentSet.of([_Trace("t1")])
        reg.create_aligned_scorer(
            TOKEN_EFFICIENCY, experiment_id="exp1", alignment_set=aset, optimizer=my_opt
        )
        assert backend["base"].align_calls[0]["optimizer"] is my_opt

    def test_registers_base_unaligned_when_no_labels(self, backend: dict[str, Any]) -> None:
        result = reg.create_aligned_scorer(
            TOKEN_EFFICIENCY, experiment_id="exp1", alignment_set=None
        )
        base = backend["base"]
        assert base.align_calls == []  # never aligned
        assert result.aligned is False
        assert result.report.aligned is False
        assert result.report.n_alignment_traces == 0
        assert result.judge is base  # the base judge was registered as-is
        # Flagged not-yet-trusted (recorded on the report and best-effort tagged).
        assert backend["tags"][-1] == ("exp1", "token_efficiency", False)
        assert any("not-yet-trusted" in n for n in result.report.notes)

    def test_empty_alignment_set_is_treated_as_no_labels(self, backend: dict[str, Any]) -> None:
        result = reg.create_aligned_scorer(
            TOKEN_EFFICIENCY, experiment_id="exp1", alignment_set=AlignmentSet.of([])
        )
        assert backend["base"].align_calls == []
        assert result.aligned is False

    def test_rejects_out_of_range_sampling_rate(self, backend: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="sampling_rate"):
            reg.create_aligned_scorer(TOKEN_EFFICIENCY, experiment_id="exp1", sampling_rate=1.5)


class TestRegisterPrealignedScorer:
    """Register an already-aligned judge without re-running MemAlign (auto-align path)."""

    @pytest.fixture
    def offline(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, bool]]:
        monkeypatch.setattr(reg, "_require_databricks_agents", lambda: None)
        monkeypatch.setattr(reg, "_configure_databricks", lambda **kw: None)
        tags: list[tuple[str, str, bool]] = []
        monkeypatch.setattr(
            reg,
            "_tag_alignment",
            lambda experiment_id, name, aligned: (
                tags.append((experiment_id, name, aligned)) or True
            ),
        )
        return tags

    def test_registers_the_given_judge_without_aligning(
        self, offline: list[tuple[str, str, bool]]
    ) -> None:
        # The judge is ALREADY aligned; register must not call .align on it.
        aligned = _FakeAlignableScorer("token_efficiency+aligned")
        report = AlignmentReport(
            base_judge_name="token_efficiency", n_alignment_traces=14, aligned=True
        )
        result = reg.register_prealigned_scorer(
            aligned, report, experiment_id="exp1", sampling_rate=0.25
        )
        assert aligned.align_calls == []  # NOT re-aligned
        # Registered then started, and the SAME measured judge is carried through.
        assert [c[0] for c in aligned.calls] == ["register", "start"]
        assert result.aligned is True
        assert result.report is report
        assert result.scorer.sample_rate == 0.25
        assert offline[-1] == ("exp1", "token_efficiency", True)

    def test_rejects_out_of_range_sampling_rate(self, offline: list[tuple[str, str, bool]]) -> None:
        report = AlignmentReport(base_judge_name="token_efficiency", aligned=True)
        with pytest.raises(ValueError, match="sampling_rate"):
            reg.register_prealigned_scorer(
                _FakeAlignableScorer("j"), report, experiment_id="exp1", sampling_rate=0.0
            )


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
