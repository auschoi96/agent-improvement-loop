"""Tests for the in-app labeling page's server side (:mod:`ail.labeling`).

All offline: the pure orchestration (:func:`build_dimensions_state`,
:func:`apply_label`) runs against a fake trace source and an injected recorder — no
MLflow, no workspace. The headline invariants under test are L4's load-bearing
contract:

* the dimensions offered are **exactly** the registered judges (none invented);
* a written label is name-matched to a registered judge, and a name that is not a
  registered judge is **refused** (the single condition MemAlign aligns by);
* every failure is **fail-closed** — an empty labeler, a missing field, or a write
  failure never reports a fabricated ``labeled``;
* the label floor is the readiness floor, surfaced verbatim (never re-derived);
* the authenticated actor — never a client-supplied ``labeler`` — is the label source.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from ail.judges.labeling import TraceLabel
from ail.labeling import service
from ail.labeling.service import (
    DimensionsResult,
    ErrorResult,
    LabelingOutcome,
    LabelInput,
    LabelResult,
    apply_label,
    build_dimensions_state,
    run_action,
    run_dimensions,
    run_label,
)
from ail.readiness import ReadinessThresholds

# --- fakes (mirror tests/test_judges_auto_align.py) ------------------------


class _FakeAssessment:
    def __init__(
        self,
        name: str,
        value: Any,
        *,
        source_type: str = "HUMAN",
        source_id: str = "expert",
    ) -> None:
        self.name = name
        self.value = value
        self.source = SimpleNamespace(source_type=source_type, source_id=source_id)
        self.rationale = None


class _FakeTrace:
    def __init__(
        self,
        trace_id: str,
        assessments: list[_FakeAssessment],
        *,
        request_preview: str | None = None,
        request_time: Any = None,
    ) -> None:
        self.trace_id = trace_id
        self.request_preview = request_preview
        self.request_time = request_time
        self.raw = SimpleNamespace(info=SimpleNamespace(assessments=assessments))


class _FakeSource:
    def __init__(self, traces: list[_FakeTrace]) -> None:
        self._traces = traces
        self.calls: list[dict[str, Any]] = []

    def iter_traces(
        self,
        *,
        experiment_id: str,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
    ) -> Any:
        self.calls.append({"experiment_id": experiment_id, "max_results": max_results})
        return iter(self._traces)

    def get_trace(self, trace_id: str) -> Any:  # pragma: no cover - not exercised here
        return None


class _RecordingRecorder:
    """A fake :data:`ail.labeling.service.LabelRecorder` that captures the write."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[TraceLabel, str]] = []

    def __call__(self, label: TraceLabel, labeler: str) -> Any:
        self.calls.append((label, labeler))
        if self.fail is not None:
            raise self.fail
        return ["assessment"]


# --- build_dimensions_state ------------------------------------------------


class TestBuildDimensionsState:
    def test_offers_exactly_the_registered_judges_and_counts_human_labels(self) -> None:
        source = _FakeSource(
            [
                # correctness labeled by a human; modularity not.
                _FakeTrace("t1", [_FakeAssessment("correctness", "pass")]),
                # correctness present but from the LLM judge -> NOT a human label.
                _FakeTrace("t2", [_FakeAssessment("correctness", "fail", source_type="LLM_JUDGE")]),
                # a human label for a judge that is NOT registered -> ignored entirely.
                _FakeTrace("t3", [_FakeAssessment("latency", 3)]),
                # both registered judges labeled by a human -> fully labeled.
                _FakeTrace(
                    "t4",
                    [_FakeAssessment("correctness", "pass"), _FakeAssessment("modularity", 5)],
                ),
            ]
        )
        result = build_dimensions_state(
            experiment_id="exp1",
            judge_names=["correctness", "modularity"],
            source=source,
            label_floor=20,
        )
        assert isinstance(result, DimensionsResult)
        assert [d.name for d in result.dimensions] == ["correctness", "modularity"]
        by = {d.name: d for d in result.dimensions}
        # correctness: t1 + t4 human -> 2; t2 is LLM_JUDGE (ignored).
        assert by["correctness"].labels_so_far == 2
        assert by["modularity"].labels_so_far == 1
        assert by["correctness"].label_floor == 20
        assert by["correctness"].remaining == 18
        assert by["correctness"].complete is False
        assert result.scanned == 4

    def test_worklist_holds_only_traces_missing_a_dimension_with_the_labeled_map(self) -> None:
        source = _FakeSource(
            [
                _FakeTrace("t1", [_FakeAssessment("correctness", "pass")]),  # missing modularity
                _FakeTrace(
                    "t2",
                    [_FakeAssessment("correctness", "pass"), _FakeAssessment("modularity", 4)],
                ),  # fully labeled -> NOT on the worklist
                _FakeTrace("t3", []),  # missing both
            ]
        )
        result = build_dimensions_state(
            experiment_id="exp1",
            judge_names=["correctness", "modularity"],
            source=source,
            label_floor=20,
        )
        ids = [t.trace_id for t in result.traces]
        assert ids == ["t1", "t3"]  # t2 excluded (fully labeled)
        t1 = next(t for t in result.traces if t.trace_id == "t1")
        assert t1.labeled == {"correctness": True, "modularity": False}
        t3 = next(t for t in result.traces if t.trace_id == "t3")
        assert t3.labeled == {"correctness": False, "modularity": False}

    def test_summary_and_floor_are_python_composed_verbatim(self) -> None:
        # A distinctive floor (not the real default) proves nothing is hardcoded.
        source = _FakeSource([_FakeTrace("t1", [_FakeAssessment("correctness", "pass")])])
        result = build_dimensions_state(
            experiment_id="exp1",
            judge_names=["correctness"],
            source=source,
            label_floor=7,
        )
        assert result.label_floor == 7
        dim = result.dimensions[0]
        assert dim.label_floor == 7
        assert dim.remaining == 6
        assert "1 / 7 human labels" in dim.summary
        assert "6 more" in dim.summary

    def test_complete_when_floor_met(self) -> None:
        source = _FakeSource(
            [_FakeTrace(f"t{i}", [_FakeAssessment("correctness", "pass")]) for i in range(3)]
        )
        result = build_dimensions_state(
            experiment_id="exp1",
            judge_names=["correctness"],
            source=source,
            label_floor=3,
        )
        dim = result.dimensions[0]
        assert dim.labels_so_far == 3
        assert dim.remaining == 0
        assert dim.complete is True
        assert "floor met" in dim.summary
        # A fully-labeled corpus leaves an empty worklist.
        assert result.traces == []

    def test_worklist_is_capped_but_counts_span_the_full_scan(self) -> None:
        source = _FakeSource([_FakeTrace(f"t{i}", []) for i in range(10)])
        result = build_dimensions_state(
            experiment_id="exp1",
            judge_names=["correctness"],
            source=source,
            label_floor=20,
            worklist_limit=3,
        )
        assert len(result.traces) == 3  # capped
        assert result.scanned == 10  # full scan counted
        assert result.dimensions[0].labels_so_far == 0

    def test_scan_capped_flag(self) -> None:
        source = _FakeSource([_FakeTrace(f"t{i}", []) for i in range(5)])
        capped = build_dimensions_state(
            experiment_id="exp1", judge_names=["c"], source=source, label_floor=20, scan_limit=5
        )
        assert capped.scan_capped is True
        source2 = _FakeSource([_FakeTrace(f"t{i}", []) for i in range(3)])
        not_capped = build_dimensions_state(
            experiment_id="exp1", judge_names=["c"], source=source2, label_floor=20, scan_limit=5
        )
        assert not_capped.scan_capped is False

    def test_no_registered_judges_yields_no_dimensions_or_worklist(self) -> None:
        source = _FakeSource([_FakeTrace("t1", [_FakeAssessment("correctness", "pass")])])
        result = build_dimensions_state(
            experiment_id="exp1", judge_names=[], source=source, label_floor=20
        )
        assert result.dimensions == []
        assert result.traces == []  # no dimension can be "missing"
        assert result.summary == ""

    def test_carries_the_label_input_hint(self) -> None:
        source = _FakeSource([_FakeTrace("t1", [])])
        result = build_dimensions_state(
            experiment_id="exp1",
            judge_names=["correctness"],
            source=source,
            label_floor=20,
            label_inputs={
                "correctness": LabelInput(kind="pass_fail", positive="pass", negative="fail")
            },
        )
        assert result.dimensions[0].input == LabelInput(
            kind="pass_fail", positive="pass", negative="fail"
        )


# --- apply_label -----------------------------------------------------------

REGISTERED = ["correctness", "modularity"]


class TestApplyLabelFailClosed:
    def test_refuses_anonymous_labeler_and_never_writes(self) -> None:
        rec = _RecordingRecorder()
        result = apply_label(
            experiment_id="exp1",
            trace_id="t1",
            name="correctness",
            value="pass",
            labeler="   ",
            judge_names=REGISTERED,
            record=rec,
        )
        assert result.outcome == LabelingOutcome.REFUSED
        assert result.refused_reason is not None and "anonymous" in result.refused_reason
        assert rec.calls == []

    @pytest.mark.parametrize(
        ("trace_id", "name", "value"),
        [
            ("", "correctness", "pass"),  # missing trace
            ("t1", "", "pass"),  # missing name
            ("t1", "correctness", None),  # missing value
            ("t1", "correctness", "   "),  # blank string value
        ],
    )
    def test_refuses_missing_fields_and_never_writes(
        self, trace_id: str, name: str, value: Any
    ) -> None:
        rec = _RecordingRecorder()
        result = apply_label(
            experiment_id="exp1",
            trace_id=trace_id,
            name=name,
            value=value,
            labeler="me@x.com",
            judge_names=REGISTERED,
            record=rec,
        )
        assert result.outcome == LabelingOutcome.REFUSED
        assert rec.calls == []

    def test_refuses_a_name_that_is_not_a_registered_judge_the_sacred_guard(self) -> None:
        rec = _RecordingRecorder()
        result = apply_label(
            experiment_id="exp1",
            trace_id="t1",
            name="not_a_judge",
            value="pass",
            labeler="me@x.com",
            judge_names=REGISTERED,
            record=rec,
        )
        assert result.outcome == LabelingOutcome.REFUSED
        assert result.refused_reason is not None
        assert "not a registered judge" in result.refused_reason
        assert rec.calls == []  # a label that could never align is never written

    def test_a_write_failure_is_an_honest_error_never_labeled(self) -> None:
        rec = _RecordingRecorder(fail=PermissionError("trace not found / not authorized"))
        result = apply_label(
            experiment_id="exp1",
            trace_id="t1",
            name="correctness",
            value="pass",
            labeler="me@x.com",
            judge_names=REGISTERED,
            record=rec,
        )
        assert result.outcome == LabelingOutcome.ERROR
        assert result.outcome != LabelingOutcome.LABELED
        assert result.error is not None and "PermissionError" in result.error
        assert len(rec.calls) == 1  # it attempted the write, then reported the failure honestly


class TestApplyLabelHappyPath:
    def test_writes_a_name_matched_human_label_with_the_authenticated_labeler(self) -> None:
        rec = _RecordingRecorder()
        result = apply_label(
            experiment_id="exp1",
            trace_id="t1",
            name="correctness",
            value="pass",
            labeler="me@x.com",
            judge_names=REGISTERED,
            rationale="  clear evidence in the trace  ",
        )  # default recorder is record_label; but we pass ours via record for isolation
        # Re-run with the recording recorder to assert the written label shape.
        rec = _RecordingRecorder()
        result = apply_label(
            experiment_id="exp1",
            trace_id="t1",
            name="correctness",
            value="pass",
            labeler="me@x.com",
            judge_names=REGISTERED,
            rationale="  clear evidence  ",
            record=rec,
        )
        assert result.outcome == LabelingOutcome.LABELED
        assert result.labeler == "me@x.com"
        ((label, labeler),) = rec.calls
        assert isinstance(label, TraceLabel)
        assert label.name == "correctness"  # name == judge name (the alignment key)
        assert label.value == "pass"
        assert label.rationale == "clear evidence"  # trimmed
        assert labeler == "me@x.com"  # the authenticated identity

    def test_accepts_numeric_and_falsey_values(self) -> None:
        for value in (5, 0, 1.5, False):
            rec = _RecordingRecorder()
            result = apply_label(
                experiment_id="exp1",
                trace_id="t1",
                name="modularity",
                value=value,
                labeler="me@x.com",
                judge_names=REGISTERED,
                record=rec,
            )
            assert result.outcome == LabelingOutcome.LABELED, value
            assert rec.calls[0][0].value == value

    def test_success_attaches_recomputed_progress_from_the_source(self) -> None:
        # After a write, the count is re-read via read_human_labels off the source.
        source = _FakeSource(
            [
                _FakeTrace("t1", [_FakeAssessment("correctness", 5)]),
                _FakeTrace("t2", [_FakeAssessment("correctness", 4)]),
            ]
        )
        rec = _RecordingRecorder()
        result = apply_label(
            experiment_id="exp1",
            trace_id="t3",
            name="correctness",
            value=3,
            labeler="me@x.com",
            judge_names=REGISTERED,
            record=rec,
            source=source,
            label_floor=20,
        )
        assert result.outcome == LabelingOutcome.LABELED
        assert result.labels_so_far == 2  # from the (fake) re-read
        assert result.label_floor == 20
        assert result.remaining == 18
        assert result.complete is False


# --- dispatch + live-wiring fail-closed ------------------------------------


class TestRunActionDispatch:
    def test_label_uses_the_authenticated_actor_not_a_body_labeler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run_label(**kwargs: Any) -> LabelResult:
            captured.update(kwargs)
            return LabelResult(
                outcome=LabelingOutcome.LABELED,
                experiment_id=kwargs["experiment_id"],
                trace_id=kwargs["trace_id"],
                name=kwargs["name"],
                labeler=kwargs["labeler"],
            )

        monkeypatch.setattr(service, "run_label", fake_run_label)
        run_action(
            {
                "action": "label",
                "actor": "me@x.com",
                "labeler": "attacker@evil.com",  # a spoofed body identity — MUST be ignored
                "experiment_id": "exp1",
                "trace_id": "t1",
                "name": "correctness",
                "value": "pass",
                "rationale": "why",
            }
        )
        assert captured["labeler"] == "me@x.com"  # authenticated actor, not the body value
        assert captured["trace_id"] == "t1"
        assert captured["name"] == "correctness"
        assert captured["value"] == "pass"

    def test_dimensions_forwards_actor_and_experiment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run_dimensions(experiment_id: str, **kwargs: Any) -> DimensionsResult:
            captured["experiment_id"] = experiment_id
            captured.update(kwargs)
            return DimensionsResult(experiment_id=experiment_id, label_floor=20)

        monkeypatch.setattr(service, "run_dimensions", fake_run_dimensions)
        run_action({"action": "dimensions", "actor": "me@x.com", "experiment_id": "exp1"})
        assert captured["experiment_id"] == "exp1"
        assert captured["actor"] == "me@x.com"

    def test_unknown_action_is_an_error(self) -> None:
        result = run_action({"action": "nope"})
        assert isinstance(result, ErrorResult)
        assert result.outcome == LabelingOutcome.ERROR
        assert "nope" in result.error


class TestLiveWiringFailClosed:
    def test_run_dimensions_requires_an_experiment_id(self) -> None:
        result = run_dimensions("   ")
        assert isinstance(result, ErrorResult)
        assert "experiment id is required" in result.error

    def test_run_dimensions_refuses_to_invent_dimensions_when_judges_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(experiment_id: str, profile: str | None) -> list[str]:
            raise RuntimeError("databricks-agents not installed")

        monkeypatch.setattr(service, "_registered_judge_names", boom)
        result = run_dimensions("exp1")
        assert isinstance(result, ErrorResult)
        assert "cannot determine the registered judges" in result.error
        assert "refusing to invent" in result.error

    def test_run_label_refuses_anonymous_before_touching_the_workspace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If it tried to resolve judges/MLflow it would call these; assert it does not.
        def fail(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("workspace touched for an anonymous label")

        monkeypatch.setattr(service, "_registered_judge_names", fail)
        monkeypatch.setattr(service, "_configure_mlflow", fail)
        result = run_label(
            experiment_id="exp1", trace_id="t1", name="correctness", value="pass", labeler=""
        )
        assert isinstance(result, LabelResult)
        assert result.outcome == LabelingOutcome.REFUSED

    def test_run_label_fails_closed_when_the_name_cannot_be_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(experiment_id: str, profile: str | None) -> list[str]:
            raise RuntimeError("cannot list scorers")

        monkeypatch.setattr(service, "_registered_judge_names", boom)
        result = run_label(
            experiment_id="exp1",
            trace_id="t1",
            name="correctness",
            value="pass",
            labeler="me@x.com",
        )
        assert isinstance(result, ErrorResult)
        assert "cannot determine the registered judges" in result.error


class TestFloorIsTheReadinessFloor:
    def test_run_paths_use_readiness_quality_min_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # run_dimensions must surface exactly ReadinessThresholds().quality_min_labels.
        monkeypatch.setattr(service, "_registered_judge_names", lambda e, p: ["correctness"])
        monkeypatch.setattr(service, "_configure_mlflow", lambda p: None)
        monkeypatch.setattr(service, "_read_label_inputs", lambda n, e, p: {})
        monkeypatch.setattr(service, "_build_source", lambda p: _FakeSource([_FakeTrace("t1", [])]))
        result = run_dimensions("exp1")
        assert isinstance(result, DimensionsResult)
        assert result.label_floor == ReadinessThresholds().quality_min_labels


# --- coerce_label_input ----------------------------------------------------


class TestCoerceLabelInput:
    def test_numeric(self) -> None:
        raw = SimpleNamespace(min_value=1, max_value=5)
        got = service._coerce_label_input(raw)
        assert got == LabelInput(kind="numeric", min=1.0, max=5.0)

    def test_pass_fail(self) -> None:
        raw = SimpleNamespace(positive_label="pass", negative_label="fail")
        got = service._coerce_label_input(raw)
        assert got == LabelInput(kind="pass_fail", positive="pass", negative="fail")

    def test_other_is_free(self) -> None:
        got = service._coerce_label_input(SimpleNamespace(something="else"))
        assert got == LabelInput(kind="free")

    def test_none(self) -> None:
        assert service._coerce_label_input(None) is None


# --- the CLI bridge --------------------------------------------------------


class TestMainCli:
    def test_unparseable_stdin_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("not json {"))
        code = service.main()
        assert code == 2
        assert "unparseable stdin" in capsys.readouterr().out

    def test_non_object_stdin_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("[1, 2, 3]"))
        code = service.main()
        assert code == 2
        assert "must be a JSON object" in capsys.readouterr().out

    def test_valid_action_prints_json_and_returns_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            service,
            "run_action",
            lambda payload: ErrorResult(action="dimensions", error="determined error"),
        )
        monkeypatch.setattr("sys.stdin", io.StringIO('{"action": "dimensions"}'))
        code = service.main()
        assert code == 0
        assert "determined error" in capsys.readouterr().out
