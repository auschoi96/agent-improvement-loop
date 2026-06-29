"""Tests for the L3 OTLP adapter (:mod:`ail.l3.adapter`).

The adapter projects a :class:`~ail.ingest.base.NormalizedTrace` onto the flat
OpenInference/OTLP ``SpanRecord`` JSONL HALO indexes. These tests run fully
offline against the recorded ``synthetic_trace`` fixture (normalized through the
real MLflow path) and assert the SpanRecord shape and OpenInference attribute
conventions directly. One extra test additionally validates each record against
HALO's *real* ``SpanRecord`` model when ``halo-engine`` happens to be installed
(skipped in CI, which does not install the ``l3`` extra).
"""

from __future__ import annotations

import json
from typing import Any

from ail.ingest.base import NormalizedTrace, TraceSource
from ail.ingest.mlflow_source import normalize_trace
from ail.l3.adapter import (
    mlflow_trace_to_otlp_jsonl,
    normalized_trace_to_span_records,
    write_span_records_jsonl,
)

# Fields HALO's SpanRecord requires on every line.
_REQUIRED_KEYS = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "trace_state",
    "name",
    "kind",
    "start_time",
    "end_time",
    "status",
    "resource",
    "scope",
    "attributes",
}


class _FakeSource(TraceSource):
    """A trace source that serves one pre-normalized trace by id."""

    def __init__(self, trace: NormalizedTrace | None) -> None:
        self._trace = trace

    def iter_traces(self, **_: Any) -> Any:
        yield from ([self._trace] if self._trace is not None else [])

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        return self._trace


def _records(synthetic_trace: Any) -> list[dict[str, Any]]:
    return normalized_trace_to_span_records(normalize_trace(synthetic_trace))


def _by_name(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(r for r in records if r["name"] == name)


class TestSpanRecordShape:
    def test_one_record_per_span(self, synthetic_trace: Any) -> None:
        records = _records(synthetic_trace)
        # The fixture has 4 spans: AGENT + LLM + two TOOL.
        assert len(records) == 4

    def test_every_record_has_required_keys(self, synthetic_trace: Any) -> None:
        for record in _records(synthetic_trace):
            assert _REQUIRED_KEYS <= set(record), f"missing keys in {record['name']}"

    def test_nested_blocks_well_formed(self, synthetic_trace: Any) -> None:
        for record in _records(synthetic_trace):
            assert set(record["status"]) == {"code", "message"}
            assert isinstance(record["resource"]["attributes"], dict)
            assert record["scope"]["name"] == "ail.l3.otlp_adapter"
            assert isinstance(record["attributes"], dict)

    def test_trace_id_is_constant_across_spans(self, synthetic_trace: Any) -> None:
        trace = normalize_trace(synthetic_trace)
        records = normalized_trace_to_span_records(trace)
        assert {r["trace_id"] for r in records} == {trace.trace_id}

    def test_parent_link_preserved(self, synthetic_trace: Any) -> None:
        records = _records(synthetic_trace)
        agent = _by_name(records, "claude_code_conversation")
        llm = _by_name(records, "llm")
        # The root AGENT span has no parent; the LLM span hangs off it.
        assert agent["parent_span_id"] == ""
        assert llm["parent_span_id"] == agent["span_id"]


class TestOpenInferenceAttributes:
    def test_span_kinds_mapped(self, synthetic_trace: Any) -> None:
        records = _records(synthetic_trace)
        assert (
            _by_name(records, "claude_code_conversation")["attributes"]["openinference.span.kind"]
            == "AGENT"
        )
        assert _by_name(records, "llm")["attributes"]["openinference.span.kind"] == "LLM"
        assert _by_name(records, "tool_Read")["attributes"]["openinference.span.kind"] == "TOOL"

    def test_llm_span_carries_model_and_int_tokens(self, synthetic_trace: Any) -> None:
        attrs = _by_name(_records(synthetic_trace), "llm")["attributes"]
        assert attrs["llm.model_name"] == "claude-opus-4-8"
        assert attrs["inference.llm.model_name"] == "claude-opus-4-8"
        # HALO's index only counts integer token attributes.
        assert attrs["inference.llm.input_tokens"] == 1000
        assert attrs["inference.llm.output_tokens"] == 200
        assert isinstance(attrs["inference.llm.input_tokens"], int)

    def test_tool_span_carries_bare_name_and_params(self, synthetic_trace: Any) -> None:
        attrs = _by_name(_records(synthetic_trace), "tool_Read")["attributes"]
        # Bare tool name (not the ``tool_`` span-name prefix).
        assert attrs["tool.name"] == "Read"
        assert "file_path" in attrs["tool.parameters"]
        assert "input.value" in attrs

    def test_agent_span_carries_agent_identity(self, synthetic_trace: Any) -> None:
        attrs = _by_name(_records(synthetic_trace), "claude_code_conversation")["attributes"]
        assert attrs["inference.agent_name"]
        assert attrs["inference.agent_id"]

    def test_resource_carries_service_name(self, synthetic_trace: Any) -> None:
        record = _by_name(_records(synthetic_trace), "llm")
        # The fixture is detected as Claude Code.
        assert record["resource"]["attributes"]["service.name"] == "claude_code"


class TestStatusAndTimes:
    def test_error_status_mapped_to_otel_code(self, synthetic_trace: Any) -> None:
        bash = _by_name(_records(synthetic_trace), "tool_Bash")
        assert bash["status"]["code"] == "STATUS_CODE_ERROR"

    def test_ok_status_mapped(self, synthetic_trace: Any) -> None:
        read = _by_name(_records(synthetic_trace), "tool_Read")
        assert read["status"]["code"] == "STATUS_CODE_OK"

    def test_timestamps_are_iso8601_and_sortable(self, synthetic_trace: Any) -> None:
        from datetime import datetime

        records = _records(synthetic_trace)
        starts = [r["start_time"] for r in records]
        assert all(starts), "every span carries a start_time"
        # All parseable as ISO-8601.
        for s in starts:
            datetime.fromisoformat(s)
        # The property HALO relies on: lexicographic order == chronological order.
        assert sorted(starts) == [
            r["start_time"]
            for r in sorted(records, key=lambda r: datetime.fromisoformat(r["start_time"]))
        ]


class TestJsonlWriting:
    def test_writes_one_json_object_per_line(self, synthetic_trace: Any, tmp_path: Any) -> None:
        records = _records(synthetic_trace)
        out = write_span_records_jsonl(records, tmp_path / "trace.jsonl")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(records)
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_mlflow_trace_to_otlp_jsonl_pulls_from_source(
        self, synthetic_trace: Any, tmp_path: Any
    ) -> None:
        trace = normalize_trace(synthetic_trace)
        export = mlflow_trace_to_otlp_jsonl(
            trace.trace_id,
            "660599403165942",
            path=tmp_path / "out.jsonl",
            source=_FakeSource(trace),
        )
        assert export.trace_id == trace.trace_id
        assert export.n_spans == len(trace.spans)
        assert export.path.exists()
        assert len(export.path.read_text().splitlines()) == export.n_spans

    def test_missing_trace_raises_lookuperror(self, tmp_path: Any) -> None:
        import pytest

        with pytest.raises(LookupError):
            mlflow_trace_to_otlp_jsonl("nope", path=tmp_path / "x.jsonl", source=_FakeSource(None))


def test_records_validate_against_real_halo_spanrecord(synthetic_trace: Any) -> None:
    """When ``halo-engine`` is installed, every record validates as a real SpanRecord.

    Skipped in CI (the ``l3`` extra is not installed there); this is the contract
    check that the offline shape assertions above stay faithful to HALO's model.
    """
    import pytest

    canonical_span = pytest.importorskip("engine.traces.models.canonical_span")
    for record in _records(synthetic_trace):
        canonical_span.SpanRecord.model_validate(record)
