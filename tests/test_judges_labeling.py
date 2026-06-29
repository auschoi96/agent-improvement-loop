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
    """Minimal raw-MLflow-trace shape carrying a resolvable id (``info.trace_id``)."""

    def __init__(self, trace_id: str) -> None:
        self.info = type("Info", (), {"trace_id": trace_id})()


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


class TestPoolAssembly:
    def test_to_alignment_set_uses_raw_traces(self) -> None:
        source = _FakeSource()
        aset = to_alignment_set(source, ["t1", "t2", "t1"])  # duplicate de-duped
        assert aset.ids == frozenset({"t1", "t2"})
        assert source.fetched == ["t1", "t2"]  # de-duped before fetch

    def test_to_alignment_set_skips_unfetchable(self) -> None:
        source = _FakeSource(missing=frozenset({"gone"}))
        aset = to_alignment_set(source, ["t1", "gone"])
        assert aset.ids == frozenset({"t1"})

    def test_to_human_anchor_filters_by_judge_name(self) -> None:
        labels = [
            TraceLabel(trace_id="t1", name="token_efficiency", value=4, outputs="r1"),
            TraceLabel(trace_id="t1", name="correctness", value="yes"),
            TraceLabel(trace_id="t2", name="token_efficiency", value=2, outputs="r2"),
        ]
        anchor = to_human_anchor(labels, name="token_efficiency")
        assert len(anchor) == 2  # the correctness label is filtered out
        assert anchor.ids == frozenset({"t1", "t2"})

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
