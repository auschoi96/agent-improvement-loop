"""Tests for the auto-align trigger (:mod:`ail.judges.auto_align`).

The trigger is orchestration, so the model/MLflow-touching seams it *reuses* are
replaced offline: ``read_human_labels`` / ``assemble_pools`` / ``make_scorer`` /
``align_judge`` / ``score_anchor`` / ``register_prealigned_scorer`` are
monkeypatched at the module boundary, and the watermark store is an in-memory
fake. No model is called and no workspace is hit. The one genuinely-live path
(running the cadence against a real experiment) is gated behind
``@pytest.mark.live`` and self-skips without a workspace.

What is asserted (the trigger's contract):

* **floor + watermark gating** — aligns only when ``>= label_floor`` labels AND
  the count grew past the watermark;
* **idempotency** — no re-align when the count has not grown;
* **distrusted-when-unmeasured** — an unmeasured / below-floor aligned judge is
  held, never promoted (fail closed);
* **rollback-on-regression** — a re-alignment below the prior aligned version
  keeps the prior version live (fail closed toward last-known-good).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from ail.judges import auto_align as aa
from ail.judges.agreement import AgreementConfig
from ail.judges.alignment import AlignmentOutcome
from ail.judges.auto_align import (
    AutoAlignConfig,
    AutoAlignReport,
    AutoAlignState,
    AutoAlignStatus,
    ExperimentTagWatermarkStore,
    JudgeAutoAlignResult,
    WatermarkReadError,
    auto_align_judge,
    auto_align_scorers,
    read_human_labels,
)
from ail.judges.contract import AgreementReport, AlignmentReport
from ail.judges.labeling import TraceLabel
from ail.judges.scorers import CORRECTNESS, DEFAULT_SCORERS
from ail.pools import AlignmentSet, AnchorItem, HumanAnchor

# --- fakes -----------------------------------------------------------------


class _MemStore:
    """An in-memory :class:`ail.judges.auto_align.WatermarkStore`."""

    def __init__(self, seed: dict[str, AutoAlignState] | None = None) -> None:
        self.data: dict[str, AutoAlignState] = dict(seed or {})
        self.writes: list[tuple[str, AutoAlignState]] = []

    def read(self, judge_name: str) -> AutoAlignState:
        return self.data.get(judge_name, AutoAlignState())

    def write(self, judge_name: str, state: AutoAlignState) -> None:
        self.data[judge_name] = state
        self.writes.append((judge_name, state))


def _agreement(
    rate: float,
    *,
    distrusted: bool = False,
    insufficient: bool = False,
    floor: float = 0.7,
    name: str = "correctness",
) -> AgreementReport:
    return AgreementReport(
        judge_name=name,
        agreement_rate=rate,
        floor=floor,
        distrusted=distrusted,
        insufficient_data=insufficient,
        n_items=8,
        n_scored=0 if insufficient else 8,
        n_agreements=round(rate * 8),
    )


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the reused L2 seams; expose knobs and recorded calls.

    Knobs: ``labels`` (how many human labels ``read_human_labels`` returns),
    ``score`` (the :class:`AgreementReport` ``score_anchor`` returns), and
    ``pools_empty`` (force the empty-pool guard). Records: ``align_calls`` and
    ``register_calls``.
    """
    state: dict[str, Any] = {
        "labels": 25,
        "score": _agreement(0.85),
        "pools_empty": False,
        "align_calls": [],
        "register_calls": [],
    }

    def fake_read(
        source: Any,
        *,
        experiment_id: str,
        judge_name: str,
        max_results: int | None = None,
        labeler_id: str | None = None,
    ) -> list[TraceLabel]:
        return [
            TraceLabel(trace_id=f"t{i}", name=judge_name, value="yes")
            for i in range(state["labels"])
        ]

    def fake_assemble(
        source: Any,
        labels: Any,
        *,
        judge_name: str | None = None,
        anchor_fraction: float = 0.3,
        seed: int = 0,
        labeler_id: str = "expert",
    ) -> tuple[AlignmentSet, HumanAnchor]:
        labels = list(labels)
        if state["pools_empty"]:
            return AlignmentSet.of([]), HumanAnchor.of([])
        n_anchor = max(1, round(len(labels) * anchor_fraction))
        anchor = HumanAnchor.of(
            AnchorItem(item_id=lab.trace_id, human_label=lab.value) for lab in labels[:n_anchor]
        )
        alignment = AlignmentSet.of([lab.trace_id for lab in labels[n_anchor:]])
        return alignment, anchor

    def fake_make_scorer(spec: Any, *, model: str | None = None, **kw: Any) -> Any:
        return SimpleNamespace(name=spec.name)

    def fake_align(
        judge: Any,
        alignment_set: AlignmentSet,
        *,
        optimizer: Any = None,
        generated_at: str | None = None,
    ) -> AlignmentOutcome:
        state["align_calls"].append({"n": len(alignment_set), "optimizer": optimizer})
        name = getattr(judge, "name", "judge")
        return AlignmentOutcome(
            judge=SimpleNamespace(name=f"{name}+aligned"),
            report=AlignmentReport(
                base_judge_name=name,
                n_alignment_traces=len(alignment_set),
                aligned=True,
                generated_at=generated_at,
            ),
        )

    def fake_score(
        judge: Any, anchor: Any, *, config: Any = None, generated_at: str | None = None
    ) -> AgreementReport:
        return state["score"]

    def fake_register(
        judge: Any,
        report: AlignmentReport,
        *,
        experiment_id: str,
        sampling_rate: float = 0.1,
        filter_string: str | None = None,
        profile: str | None = None,
    ) -> Any:
        reg = SimpleNamespace(
            judge=judge,
            report=report,
            aligned=report.aligned,
            scorer=SimpleNamespace(name=getattr(judge, "name", "judge")),
        )
        state["register_calls"].append(reg)
        return reg

    monkeypatch.setattr(aa, "read_human_labels", fake_read)
    monkeypatch.setattr(aa, "assemble_pools", fake_assemble)
    monkeypatch.setattr(aa, "make_scorer", fake_make_scorer)
    monkeypatch.setattr(aa, "align_judge", fake_align)
    monkeypatch.setattr(aa, "score_anchor", fake_score)
    monkeypatch.setattr(aa, "register_prealigned_scorer", fake_register)
    return state


def _run(
    store: _MemStore,
    seams: dict[str, Any],
    *,
    register: bool = True,
    config: AutoAlignConfig | None = None,
) -> JudgeAutoAlignResult:
    return auto_align_judge(
        CORRECTNESS,
        experiment_id="exp1",
        source=object(),
        store=store,
        config=config or AutoAlignConfig(),
        register=register,
        now="2026-07-02T00:00:00+00:00",
    )


# --- floor + watermark gating ---------------------------------------------


class TestFloorAndWatermarkGating:
    def test_below_floor_does_not_align(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 10  # < default floor 20
        store = _MemStore()
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.SKIPPED_BELOW_FLOOR
        assert result.promoted is False
        assert seams["align_calls"] == []  # never aligned
        assert seams["register_calls"] == []
        assert store.writes == []  # watermark untouched below the floor

    def test_aligns_when_floor_met_and_new_labels(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 25
        seams["score"] = _agreement(0.85)
        store = _MemStore()
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.ALIGNED
        assert result.promoted is True
        assert len(seams["align_calls"]) == 1  # aligned exactly once
        assert len(seams["register_calls"]) == 1  # promoted -> registered
        # Watermark advanced to the label count; agreement bar set to the measured rate.
        assert store.data["correctness"].label_count == 25
        assert store.data["correctness"].agreement == 0.85
        assert store.data["correctness"].aligned_at == "2026-07-02T00:00:00+00:00"

    def test_exactly_at_floor_aligns(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 20  # == floor: floor is inclusive (>=)
        store = _MemStore()
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.ALIGNED

    def test_configurable_floor(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 30
        store = _MemStore()
        result = _run(store, seams, config=AutoAlignConfig(label_floor=50))
        assert result.status is AutoAlignStatus.SKIPPED_BELOW_FLOOR
        assert seams["align_calls"] == []


class TestIdempotency:
    def test_no_realign_without_new_labels(self, seams: dict[str, Any]) -> None:
        # Watermark already at the current label count -> idempotent no-op.
        seams["labels"] = 25
        store = _MemStore({"correctness": AutoAlignState(label_count=25, agreement=0.8)})
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.SKIPPED_NO_NEW_LABELS
        assert seams["align_calls"] == []  # did NOT re-align on the same labels
        assert seams["register_calls"] == []

    def test_realigns_when_labels_accrue_past_watermark(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 40  # grew past the watermark of 25
        seams["score"] = _agreement(0.85)
        store = _MemStore({"correctness": AutoAlignState(label_count=25, agreement=0.8)})
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.ALIGNED
        assert len(seams["align_calls"]) == 1  # re-aligned on the grown set
        assert store.data["correctness"].label_count == 40
        assert store.data["correctness"].agreement == 0.85


# --- distrusted-when-unmeasured (fail closed) ------------------------------


class TestAgreementFloorGuard:
    def test_unmeasured_judge_is_held_distrusted(self, seams: dict[str, Any]) -> None:
        # An empty/under-sampled anchor -> insufficient_data -> distrusted.
        seams["labels"] = 25
        seams["score"] = _agreement(0.0, distrusted=True, insufficient=True)
        store = _MemStore()
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.HELD_DISTRUSTED
        assert result.promoted is False
        assert len(seams["align_calls"]) == 1  # we aligned then measured...
        assert seams["register_calls"] == []  # ...but did NOT promote
        # Watermark advanced (don't retry same labels); agreement bar stays unset.
        assert store.data["correctness"].label_count == 25
        assert store.data["correctness"].agreement is None

    def test_below_floor_agreement_is_held_distrusted(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 25
        seams["score"] = _agreement(0.5, distrusted=True, floor=0.7)
        store = _MemStore()
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.HELD_DISTRUSTED
        assert seams["register_calls"] == []

    def test_empty_pools_held_without_aligning(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 25
        seams["pools_empty"] = True
        store = _MemStore()
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.HELD_DISTRUSTED
        assert seams["align_calls"] == []  # guarded BEFORE any alignment
        assert seams["register_calls"] == []
        assert store.data["correctness"].label_count == 25


# --- rollback-on-regression (fail closed toward last-known-good) -----------


class TestRollback:
    def test_regression_keeps_prior_aligned_version(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 40  # new labels -> re-align
        seams["score"] = _agreement(0.75)  # trusted, but < prior 0.9
        store = _MemStore({"correctness": AutoAlignState(label_count=25, agreement=0.9)})
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.ROLLED_BACK
        assert result.promoted is False
        assert len(seams["align_calls"]) == 1  # aligned + measured the candidate...
        assert seams["register_calls"] == []  # ...but kept the prior version live
        # Watermark advances; the last-known-good agreement bar is PRESERVED (0.9).
        assert store.data["correctness"].label_count == 40
        assert store.data["correctness"].agreement == 0.9

    def test_tie_promotes(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 40
        seams["score"] = _agreement(0.9)  # == prior: not a regression
        store = _MemStore({"correctness": AutoAlignState(label_count=25, agreement=0.9)})
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.ALIGNED
        assert len(seams["register_calls"]) == 1

    def test_improvement_promotes(self, seams: dict[str, Any]) -> None:
        seams["labels"] = 40
        seams["score"] = _agreement(0.95)
        store = _MemStore({"correctness": AutoAlignState(label_count=25, agreement=0.9)})
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.ALIGNED
        assert store.data["correctness"].agreement == 0.95


class TestRegisterFalse:
    def test_dry_run_computes_but_does_not_persist_state(self, seams: dict[str, Any]) -> None:
        # A dry run reports the decision (ALIGNED) but must NOT persist the
        # watermark or the last-known-good agreement bar, so a later REAL run with
        # the same labels still promotes.
        seams["labels"] = 25
        seams["score"] = _agreement(0.85)
        store = _MemStore()
        result = _run(store, seams, register=False)
        assert result.status is AutoAlignStatus.ALIGNED
        assert result.promoted is True
        assert result.registration is None
        assert seams["register_calls"] == []  # no scheduled scorer created
        # Reported in the result for the operator...
        assert result.state is not None
        assert result.state.agreement == 0.85
        # ...but NOT written to the store.
        assert store.writes == []
        assert "correctness" not in store.data

    def test_dry_run_does_not_advance_watermark_on_rollback(self, seams: dict[str, Any]) -> None:
        # A held/rolled-back dry run must not advance the watermark either.
        seams["labels"] = 40
        seams["score"] = _agreement(0.75)  # trusted but < prior 0.9 -> rollback
        store = _MemStore({"correctness": AutoAlignState(label_count=25, agreement=0.9)})
        result = _run(store, seams, register=False)
        assert result.status is AutoAlignStatus.ROLLED_BACK
        assert store.writes == []  # prior state untouched
        assert store.data["correctness"] == AutoAlignState(label_count=25, agreement=0.9)


# --- watermark read fails closed (never downgrade to 'never aligned') ------


class _RaisingStore:
    """A store whose read RAISES (an existing-but-unreadable watermark)."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, AutoAlignState]] = []

    def read(self, judge_name: str) -> AutoAlignState:
        raise WatermarkReadError(f"existing watermark for {judge_name!r} is unreadable")

    def write(self, judge_name: str, state: AutoAlignState) -> None:  # pragma: no cover
        self.writes.append((judge_name, state))


class TestWatermarkReadFailsClosed:
    def test_unreadable_watermark_skips_without_aligning_or_promoting(
        self, seams: dict[str, Any]
    ) -> None:
        # Plenty of labels + a would-be-trusting score: if the read fell back to
        # 'never aligned' this judge would (wrongly) align and promote. It must not.
        seams["labels"] = 40
        seams["score"] = _agreement(0.9)
        store = _RaisingStore()
        result = _run(store, seams)
        assert result.status is AutoAlignStatus.FAILED
        assert result.error is not None and "unreadable" in result.error
        assert result.promoted is False
        assert seams["align_calls"] == []  # NEVER aligned (fail closed)
        assert seams["register_calls"] == []  # NEVER promoted
        assert store.writes == []  # prior state left intact (no overwrite)

    def test_orchestrator_isolates_unreadable_watermark_as_failed(
        self, seams: dict[str, Any]
    ) -> None:
        seams["labels"] = 40
        seams["score"] = _agreement(0.9)
        store = _RaisingStore()
        report = auto_align_scorers(
            "exp1",
            source=object(),
            store=store,
            scorers={"correctness": CORRECTNESS},
            now="2026-07-02T00:00:00+00:00",
        )
        assert report.n_failed == 1
        assert report.n_aligned == 0
        assert seams["align_calls"] == []
        assert store.writes == []


# --- reading human labels (the L1 name-matching read side) -----------------


class _FakeAssessment:
    def __init__(
        self,
        name: str,
        value: Any,
        *,
        source_type: str = "HUMAN",
        source_id: str = "expert",
        rationale: str | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.source = SimpleNamespace(source_type=source_type, source_id=source_id)
        self.rationale = rationale


class _FakeTrace:
    def __init__(self, trace_id: str, assessments: list[_FakeAssessment]) -> None:
        self.trace_id = trace_id
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


class TestReadHumanLabels:
    def test_reads_only_human_labels_named_for_the_judge(self) -> None:
        source = _FakeSource(
            [
                _FakeTrace("t1", [_FakeAssessment("correctness", "yes", rationale="clean")]),
                # Wrong judge name -> ignored.
                _FakeTrace("t2", [_FakeAssessment("modularity", 4)]),
                # LLM_JUDGE source, right name -> ignored (not HUMAN).
                _FakeTrace("t3", [_FakeAssessment("correctness", "no", source_type="LLM_JUDGE")]),
                # No assessments -> skipped, never fabricated.
                _FakeTrace("t4", []),
                _FakeTrace("t5", [_FakeAssessment("correctness", "no", rationale="wrong file")]),
            ]
        )
        labels = read_human_labels(source, experiment_id="exp1", judge_name="correctness")
        assert [lab.trace_id for lab in labels] == ["t1", "t5"]
        assert [lab.value for lab in labels] == ["yes", "no"]
        assert labels[0].rationale == "clean"
        assert all(lab.name == "correctness" for lab in labels)

    def test_skips_traces_without_resolvable_id(self) -> None:
        source = _FakeSource(
            [
                _FakeTrace("", [_FakeAssessment("correctness", "yes")]),  # empty id -> skipped
                _FakeTrace("t2", [_FakeAssessment("correctness", "yes")]),
            ]
        )
        labels = read_human_labels(source, experiment_id="exp1", judge_name="correctness")
        assert [lab.trace_id for lab in labels] == ["t2"]

    def test_prefers_requested_labeler(self) -> None:
        source = _FakeSource(
            [
                _FakeTrace(
                    "t1",
                    [
                        _FakeAssessment("correctness", "no", source_id="other"),
                        _FakeAssessment("correctness", "yes", source_id="expert"),
                    ],
                ),
            ]
        )
        labels = read_human_labels(
            source, experiment_id="exp1", judge_name="correctness", labeler_id="expert"
        )
        assert labels[0].value == "yes"


# --- the multi-judge orchestrator ------------------------------------------


class TestAutoAlignScorers:
    def test_isolates_one_judge_failure_and_reports_counts(
        self, monkeypatch: pytest.MonkeyPatch, seams: dict[str, Any]
    ) -> None:
        # correctness has enough labels; the others have none (skip). Make the
        # 'modularity' read raise to prove one judge's failure is isolated.
        def fake_read(
            source: Any,
            *,
            experiment_id: str,
            judge_name: str,
            max_results: int | None = None,
            labeler_id: str | None = None,
        ) -> list[TraceLabel]:
            if judge_name == "modularity":
                raise RuntimeError("boom reading labels")
            if judge_name == "correctness":
                return [
                    TraceLabel(trace_id=f"t{i}", name=judge_name, value="yes") for i in range(25)
                ]
            return []  # groundedness / token_efficiency: unlabeled -> skip

        monkeypatch.setattr(aa, "read_human_labels", fake_read)
        store = _MemStore()
        report = auto_align_scorers(
            "exp1",
            source=object(),
            store=store,
            scorers=DEFAULT_SCORERS,
            register=True,
            now="2026-07-02T00:00:00+00:00",
        )
        by_name = {r.judge_name: r for r in report.results}
        assert by_name["correctness"].status is AutoAlignStatus.ALIGNED
        assert by_name["modularity"].status is AutoAlignStatus.FAILED
        assert "boom" in (by_name["modularity"].error or "")
        assert by_name["groundedness"].status is AutoAlignStatus.SKIPPED_BELOW_FLOOR
        assert report.n_aligned == 1
        assert report.n_failed == 1
        assert report.experiment_id == "exp1"


class TestAutoAlignReportCounts:
    def test_counts_by_status(self) -> None:
        def r(name: str, status: AutoAlignStatus) -> JudgeAutoAlignResult:
            return JudgeAutoAlignResult(
                judge_name=name,
                status=status,
                label_count=0,
                watermark=0,
                prior_agreement=None,
                promoted=status is AutoAlignStatus.ALIGNED,
            )

        report = AutoAlignReport(
            experiment_id="exp1",
            results=(
                r("a", AutoAlignStatus.ALIGNED),
                r("b", AutoAlignStatus.ROLLED_BACK),
                r("c", AutoAlignStatus.HELD_DISTRUSTED),
                r("d", AutoAlignStatus.SKIPPED_BELOW_FLOOR),
                r("e", AutoAlignStatus.SKIPPED_NO_NEW_LABELS),
                r("f", AutoAlignStatus.FAILED),
            ),
            generated_at="2026-07-02T00:00:00+00:00",
        )
        assert report.n_aligned == 1
        assert report.n_rolled_back == 1
        assert report.n_held_distrusted == 1
        assert report.n_skipped == 2
        assert report.n_failed == 1


# --- watermark store (experiment-tag persistence) --------------------------


class _FakeMlflowClient:
    def __init__(self, tags: dict[str, str] | None = None, *, raise_on_read: bool = False) -> None:
        self.tags = dict(tags or {})
        self.raise_on_read = raise_on_read
        self.set_calls: list[tuple[str, str, str]] = []

    def get_experiment(self, experiment_id: str) -> Any:
        if self.raise_on_read:
            raise RuntimeError("no access")
        return SimpleNamespace(tags=dict(self.tags))

    def set_experiment_tag(self, experiment_id: str, key: str, value: str) -> None:
        self.set_calls.append((experiment_id, key, value))
        self.tags[key] = value


class TestExperimentTagWatermarkStore:
    def test_write_then_read_round_trips(self) -> None:
        client = _FakeMlflowClient()
        store = ExperimentTagWatermarkStore(experiment_id="exp1", client=client)
        store.write("correctness", AutoAlignState(label_count=42, agreement=0.83, aligned_at="t0"))
        # The three tags are namespaced per judge under the auto-align prefix.
        assert ("exp1", "ail.autoalign.correctness.label_count", "42") in client.set_calls
        got = store.read("correctness")
        assert got.label_count == 42
        assert got.agreement == 0.83
        assert got.aligned_at == "t0"

    def test_missing_tags_read_as_never_aligned(self) -> None:
        # A genuine first run (no watermark tags at all) -> fresh state, proceed.
        store = ExperimentTagWatermarkStore(experiment_id="exp1", client=_FakeMlflowClient())
        got = store.read("correctness")
        assert got == AutoAlignState(label_count=0, agreement=None, aligned_at=None)

    def test_unreadable_experiment_raises_fail_closed(self) -> None:
        # A backend read failure -> state unknown -> raise (NOT a zeroed state).
        store = ExperimentTagWatermarkStore(
            experiment_id="exp1", client=_FakeMlflowClient(raise_on_read=True)
        )
        with pytest.raises(WatermarkReadError):
            store.read("correctness")

    def test_malformed_label_count_raises_fail_closed(self) -> None:
        # An EXISTING watermark whose label_count is malformed must NOT read as
        # 'never aligned' (which would re-align + drop the rollback bar).
        client = _FakeMlflowClient(
            tags={
                "ail.autoalign.correctness.label_count": "not-an-int",
                "ail.autoalign.correctness.agreement": "0.9",
                "ail.autoalign.correctness.aligned_at": "t0",
            }
        )
        store = ExperimentTagWatermarkStore(experiment_id="exp1", client=client)
        with pytest.raises(WatermarkReadError):
            store.read("correctness")

    def test_malformed_agreement_raises_fail_closed(self) -> None:
        client = _FakeMlflowClient(
            tags={
                "ail.autoalign.correctness.label_count": "25",
                "ail.autoalign.correctness.agreement": "high",  # present but unparseable
                "ail.autoalign.correctness.aligned_at": "t0",
            }
        )
        store = ExperimentTagWatermarkStore(experiment_id="exp1", client=client)
        with pytest.raises(WatermarkReadError):
            store.read("correctness")

    def test_partial_watermark_missing_label_count_raises(self) -> None:
        # A watermark EXISTS (agreement present) but label_count is absent: partial
        # / truncated write -> fail closed rather than assume count 0.
        client = _FakeMlflowClient(tags={"ail.autoalign.correctness.agreement": "0.9"})
        store = ExperimentTagWatermarkStore(experiment_id="exp1", client=client)
        with pytest.raises(WatermarkReadError):
            store.read("correctness")

    def test_none_agreement_serializes_to_empty(self) -> None:
        client = _FakeMlflowClient()
        store = ExperimentTagWatermarkStore(experiment_id="exp1", client=client)
        store.write("correctness", AutoAlignState(label_count=25, agreement=None))
        assert client.tags["ail.autoalign.correctness.agreement"] == ""
        assert store.read("correctness").agreement is None


# --- live (self-skips without a workspace) ---------------------------------


@pytest.mark.live
def test_live_auto_align_dry_run() -> None:
    """Run the cadence (register=False) against a real experiment.

    Guarded by ``AIL_LIVE_MLFLOW=1`` + ``AIL_LIVE_EXPERIMENT_ID`` and the presence
    of a workspace + the ``align`` extra (``dspy``). Skips otherwise so the default
    suite stays offline. ``register=False`` keeps it read-only (no scheduled scorer
    is created), but it still calls the reflection/judge models, so it is billable.
    """
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        pytest.skip("set AIL_LIVE_MLFLOW=1 to run the live auto-align cadence")
    experiment_id = os.environ.get("AIL_LIVE_EXPERIMENT_ID")
    if not experiment_id:
        pytest.skip("set AIL_LIVE_EXPERIMENT_ID to the target experiment")
    profile = os.environ.get("AIL_LIVE_PROFILE")

    report = auto_align_scorers(
        experiment_id,
        config=AutoAlignConfig(agreement=AgreementConfig(numeric_tolerance=1.0)),
        register=False,
        profile=profile,
    )
    # We assert the shape, never a fabricated trust outcome: every dimension gets a
    # recorded result and nothing crashed the cadence.
    assert len(report.results) == len(DEFAULT_SCORERS)
    assert report.n_failed == 0
