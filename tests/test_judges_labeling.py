"""Tests for the human-labeling helper (:mod:`ail.judges.labeling`).

Offline throughout: ``mlflow.log_feedback`` / ``mlflow.log_expectation`` are
monkeypatched to record their calls (no tracking backend), and pool assembly
runs against a fake :class:`~ail.ingest.base.TraceSource`. The point of the
module — that the two MemAlign pools are assembled **disjoint** — is asserted
directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from ail.ingest.base import NormalizedTrace, TraceSource
from ail.judges.labeling import (
    TraceLabel,
    assemble_pools,
    record_label,
    record_labels,
    split_labels,
    to_alignment_set,
    to_human_anchor,
)
from ail.pools import PoolOverlapError


class _RawTrace:
    """Minimal raw-MLflow-trace shape: a resolvable id plus a settable assessments list.

    Mirrors the surface MemAlign reads off an MLflow ``Trace``: ``info.trace_id``
    and ``info.assessments`` (the human feedback the optimizer learns from).
    """

    def __init__(self, trace_id: str) -> None:
        self.info = type("Info", (), {"trace_id": trace_id, "assessments": []})()


class _FakeSource(TraceSource):
    """A trace source whose ``get_trace`` returns a NormalizedTrace with ``.raw`` set."""

    def __init__(self, *, missing: frozenset[str] = frozenset()) -> None:
        self.missing = missing
        self.fetched: list[str] = []

    def iter_traces(self, **kwargs: Any) -> Any:
        return iter([])

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        self.fetched.append(trace_id)
        if trace_id in self.missing:
            return None
        return NormalizedTrace(trace_id=trace_id, raw=_RawTrace(trace_id))


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    """Capture mlflow.log_feedback / log_expectation calls without a backend."""
    import mlflow

    calls: dict[str, list[dict[str, Any]]] = {"feedback": [], "expectation": []}

    def fake_feedback(**kw: Any) -> str:
        calls["feedback"].append(kw)
        return "assessment-id"

    def fake_expectation(**kw: Any) -> str:
        calls["expectation"].append(kw)
        return "assessment-id"

    monkeypatch.setattr(mlflow, "log_feedback", fake_feedback)
    monkeypatch.setattr(mlflow, "log_expectation", fake_expectation)
    return calls


# ---------------------------------------------------------------------------
# record_label / record_labels — write HUMAN assessments
# ---------------------------------------------------------------------------


class TestRecordLabel:
    def test_logs_human_feedback(self, logged: dict[str, list[dict[str, Any]]]) -> None:
        label = TraceLabel(
            trace_id="tr-1", name="token_efficiency", value=2, rationale="re-read foo 34x"
        )
        record_label(label, labeler_id="austin")
        assert len(logged["feedback"]) == 1
        fb = logged["feedback"][0]
        assert fb["trace_id"] == "tr-1"
        assert fb["name"] == "token_efficiency"
        assert fb["value"] == 2
        assert fb["rationale"] == "re-read foo 34x"
        # The source is a HUMAN assessment, attributed to the labeler.
        assert str(fb["source"].source_type) == "HUMAN"
        assert fb["source"].source_id == "austin"

    def test_logs_expectations_as_ground_truth(
        self, logged: dict[str, list[dict[str, Any]]]
    ) -> None:
        label = TraceLabel(
            trace_id="tr-2",
            name="correctness",
            value="yes",
            expectations={"expected_answer": "Paris", "expected_files": ["a.py"]},
        )
        record_label(label)
        assert len(logged["feedback"]) == 1
        assert len(logged["expectation"]) == 2
        names = {e["name"] for e in logged["expectation"]}
        assert names == {"expected_answer", "expected_files"}
        assert all(str(e["source"].source_type) == "HUMAN" for e in logged["expectation"])

    def test_record_labels_counts_assessments(
        self, logged: dict[str, list[dict[str, Any]]]
    ) -> None:
        labels = [
            TraceLabel(trace_id="t1", name="token_efficiency", value=4),
            TraceLabel(trace_id="t2", name="correctness", value="no", expectations={"x": 1}),
        ]
        n = record_labels(labels)
        assert n == 3  # 2 feedback + 1 expectation


# ---------------------------------------------------------------------------
# split_labels — trace-level disjoint partition
# ---------------------------------------------------------------------------


class TestSplitLabels:
    def test_partition_is_disjoint_by_trace(self) -> None:
        labels = [TraceLabel(trace_id=f"t{i}", name="token_efficiency", value=3) for i in range(10)]
        align, anchor = split_labels(labels, anchor_fraction=0.3, seed=0)
        align_ids = {label.trace_id for label in align}
        anchor_ids = {label.trace_id for label in anchor}
        assert align_ids.isdisjoint(anchor_ids)
        assert align_ids | anchor_ids == {f"t{i}" for i in range(10)}
        assert len(anchor_ids) == 3  # round(10 * 0.3)

    def test_multiple_labels_per_trace_stay_together(self) -> None:
        # A trace labeled for two judges must not be split across pools — that
        # would put the same trace id in both the Alignment Set and the anchor.
        labels = []
        for i in range(6):
            labels.append(TraceLabel(trace_id=f"t{i}", name="token_efficiency", value=3))
            labels.append(TraceLabel(trace_id=f"t{i}", name="correctness", value="yes"))
        align, anchor = split_labels(labels, anchor_fraction=0.5, seed=1)
        align_ids = {label.trace_id for label in align}
        anchor_ids = {label.trace_id for label in anchor}
        assert align_ids.isdisjoint(anchor_ids)
        # Each trace contributes BOTH its labels to exactly one pool.
        for tid in align_ids:
            assert sum(1 for label in align if label.trace_id == tid) == 2

    def test_deterministic_for_a_seed(self) -> None:
        labels = [TraceLabel(trace_id=f"t{i}", name="x", value=1) for i in range(8)]
        assert split_labels(labels, seed=7) == split_labels(labels, seed=7)

    def test_single_trace_goes_to_alignment(self) -> None:
        align, anchor = split_labels([TraceLabel(trace_id="solo", name="x", value=1)])
        assert len(align) == 1
        assert anchor == []

    def test_empty_is_empty(self) -> None:
        assert split_labels([]) == ([], [])

    def test_rejects_bad_fraction(self) -> None:
        with pytest.raises(ValueError, match="anchor_fraction"):
            split_labels([], anchor_fraction=1.5)


# ---------------------------------------------------------------------------
# pool assembly — AlignmentSet (raw traces) + HumanAnchor, never mixed
# ---------------------------------------------------------------------------


def _te(trace_id: str, value: int = 3, **kw: Any) -> TraceLabel:
    """A token_efficiency TraceLabel (the demo's graded judge) for ``trace_id``."""
    return TraceLabel(trace_id=trace_id, name="token_efficiency", value=value, **kw)


def _human_assessments(raw: Any, name: str) -> list[Any]:
    """The HUMAN assessments of ``name`` on a raw trace — exactly what MemAlign reads.

    Mirrors ``mlflow.genai.judges.optimizers.dspy_utils.trace_to_dspy_example``'s
    filter: ``trace.info.assessments`` where the source is HUMAN and the name
    matches the judge.
    """
    return [
        a
        for a in (raw.info.assessments or [])
        if a.name == name and str(a.source.source_type) == "HUMAN"
    ]


class TestPoolAssembly:
    def test_to_alignment_set_uses_raw_traces(self) -> None:
        source = _FakeSource()
        # duplicate trace de-duped to one fetch
        aset = to_alignment_set(source, [_te("t1"), _te("t2"), _te("t1", value=4)])
        assert aset.ids == frozenset({"t1", "t2"})
        assert source.fetched == ["t1", "t2"]  # de-duped before fetch

    def test_to_alignment_set_skips_unfetchable(self) -> None:
        source = _FakeSource(missing=frozenset({"gone"}))
        aset = to_alignment_set(source, [_te("t1"), _te("gone")])
        assert aset.ids == frozenset({"t1"})

    def test_alignment_set_traces_carry_human_assessments(self) -> None:
        # The MemAlign bug: the raw traces dropped their human assessments, so
        # alignment failed with "No valid feedback records found". The fix attaches
        # each label's value as a HUMAN feedback the optimizer can read back.
        source = _FakeSource()
        labels = [
            _te("t1", value=2, rationale="re-read foo.py 34x for no gain"),
            _te("t2", value=5),
        ]
        aset = to_alignment_set(source, labels, labeler_id="austin")
        by_id = {raw.info.trace_id: raw for raw in aset.traces}

        human = _human_assessments(by_id["t1"], "token_efficiency")
        assert len(human) == 1
        # The value MemAlign learns from is the human's grade...
        assert human[0].feedback.value == 2
        # ...with its rationale and HUMAN attribution intact.
        assert human[0].rationale == "re-read foo.py 34x for no gain"
        assert human[0].source.source_id == "austin"
        assert _human_assessments(by_id["t2"], "token_efficiency")[0].feedback.value == 5

    def test_alignment_set_replaces_only_same_named_human_assessment(self) -> None:
        # Re-assembly is idempotent for the written name and leaves other
        # assessments (a different judge, an LLM-sourced one) untouched.
        class _Src(_FakeSource):
            def get_trace(self, trace_id: str) -> NormalizedTrace | None:
                from mlflow.entities import AssessmentSource, Feedback
                from mlflow.entities.assessment_source import AssessmentSourceType

                raw = _RawTrace(trace_id)
                raw.info.assessments = [
                    Feedback(
                        name="token_efficiency",
                        value=1,
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.HUMAN, source_id="stale"
                        ),
                    ),
                    Feedback(
                        name="correctness",
                        value="yes",
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.HUMAN, source_id="austin"
                        ),
                    ),
                ]
                return NormalizedTrace(trace_id=trace_id, raw=raw)

        aset = to_alignment_set(_Src(), [_te("t1", value=4)])
        raw = aset.traces[0]
        te = _human_assessments(raw, "token_efficiency")
        assert len(te) == 1  # the stale value-1 assessment was replaced, not doubled
        assert te[0].feedback.value == 4
        # The unrelated correctness label is preserved.
        assert _human_assessments(raw, "correctness")[0].feedback.value == "yes"

    def test_to_human_anchor_filters_by_judge_name(self) -> None:
        labels = [
            TraceLabel(trace_id="t1", name="token_efficiency", value=4, outputs="r1"),
            TraceLabel(trace_id="t1", name="correctness", value="yes"),
            TraceLabel(trace_id="t2", name="token_efficiency", value=2, outputs="r2"),
        ]
        anchor = to_human_anchor(labels, name="token_efficiency")
        assert len(anchor) == 2  # the correctness label is filtered out
        assert anchor.ids == frozenset({"t1", "t2"})

    def test_anchor_trace_is_blinded_of_human_gold(self) -> None:
        # Anti-co-adaptation: the live anchor trace still carries the human gold
        # label (e.g. added in the MLflow UI). If score_anchor handed that trace to
        # a {{ trace }} judge, the judge could read its own answer off
        # trace.info.assessments → circular agreement. The anchor item's trace must
        # be blinded; the gold lives only on AnchorItem.human_label.
        from mlflow.entities import AssessmentSource, Feedback
        from mlflow.entities.assessment_source import AssessmentSourceType

        class _Src(_FakeSource):
            def __init__(self) -> None:
                super().__init__()
                self.returned: dict[str, Any] = {}

            def get_trace(self, trace_id: str) -> NormalizedTrace | None:
                raw = _RawTrace(trace_id)
                raw.info.assessments = [
                    Feedback(
                        name="token_efficiency",
                        value=2,  # the HUMAN gold the judge must NOT see
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.HUMAN, source_id="austin"
                        ),
                    ),
                    Feedback(
                        name="token_efficiency",
                        value=4,  # a prior LLM-judge score: not gold, kept
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.LLM_JUDGE, source_id="judge"
                        ),
                    ),
                ]
                self.returned[trace_id] = raw
                return NormalizedTrace(trace_id=trace_id, raw=raw)

        src = _Src()
        anchor = to_human_anchor([_te("t1", value=2)], name="token_efficiency", source=src)
        item = anchor.items[0]

        # The gold survives on the item (this is what agreement compares against)...
        assert item.human_label == 2
        # ...but the trace the judge sees carries NO human assessment.
        seen = item.trace.info.assessments
        assert [a for a in seen if str(a.source.source_type) == "HUMAN"] == []
        # Non-human assessments are preserved (we blind only the human gold).
        assert any(str(a.source.source_type) == "LLM_JUDGE" for a in seen)
        # Blinding is a copy: the source's own trace object is left untouched.
        assert len(src.returned["t1"].info.assessments) == 2

    def test_assemble_pools_returns_disjoint_pools(self) -> None:
        source = _FakeSource()
        labels = [
            TraceLabel(
                trace_id=f"t{i}", name="token_efficiency", value=(i % 5) + 1, outputs=f"r{i}"
            )
            for i in range(10)
        ]
        aset, anchor = assemble_pools(
            source, labels, judge_name="token_efficiency", anchor_fraction=0.3, seed=0
        )
        # Disjoint by construction (assemble_pools also asserts this internally).
        assert aset.ids.isdisjoint(anchor.ids)
        assert len(aset) + len(anchor) == 10

    def test_assemble_pools_proves_disjointness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If a buggy split leaked a trace into both pools, assemble_pools must
        # raise via assert_pools_disjoint rather than return a mixed result.
        import ail.judges.labeling as labeling

        def leaky_split(labels: Any, **kw: Any) -> tuple[list[Any], list[Any]]:
            shared = list(labels)
            return shared, shared  # same trace ids in both pools

        monkeypatch.setattr(labeling, "split_labels", leaky_split)
        source = _FakeSource()
        labels = [TraceLabel(trace_id="t1", name="x", value=1)]
        with pytest.raises(PoolOverlapError):
            assemble_pools(source, labels)
