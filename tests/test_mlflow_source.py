"""Tests for the producer-agnostic MLflow trace source.

The offline tests reconstruct a real-shaped ``Trace`` from a recorded fixture
(no network) and assert the normalization. The ``live`` test pulls from the
reference experiment and is guarded so the default suite stays green offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ail.ingest.base import SpanKind, TokenUsage, TraceStatus
from ail.ingest.mlflow_source import (
    MLflowTraceSource,
    _as_dict,
    _maybe_json,
    _token_usage_from_dict,
    normalize_trace,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
REFERENCE_EXPERIMENT = "660599403165942"


class TestNormalizeFixture:
    def test_core_fields(self, synthetic_trace: Any) -> None:
        nt = normalize_trace(synthetic_trace)
        assert nt.trace_id == "tr-synthetic0000000000000000000001"
        assert nt.status is TraceStatus.OK
        assert nt.experiment_id == REFERENCE_EXPERIMENT
        assert nt.session_id == "11111111-2222-3333-4444-555555555555"

    def test_producer_detected_best_effort(self, synthetic_trace: Any) -> None:
        assert normalize_trace(synthetic_trace).producer == "claude_code"

    def test_model_resolved_from_llm_span(self, synthetic_trace: Any) -> None:
        assert normalize_trace(synthetic_trace).model == "claude-opus-4-8"

    def test_token_usage(self, synthetic_trace: Any) -> None:
        usage = normalize_trace(synthetic_trace).token_usage
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 200
        assert usage.total_tokens == 1200

    def test_timing(self, synthetic_trace: Any) -> None:
        nt = normalize_trace(synthetic_trace)
        assert nt.execution_duration_ms == 4200
        assert nt.duration_seconds == 4.2
        assert nt.request_time is not None
        assert nt.end_time is not None
        assert nt.end_time > nt.request_time

    def test_spans(self, synthetic_trace: Any) -> None:
        nt = normalize_trace(synthetic_trace)
        assert [s.kind for s in nt.spans] == [
            SpanKind.AGENT,
            SpanKind.LLM,
            SpanKind.TOOL,
            SpanKind.TOOL,
        ]
        llm = next(s for s in nt.spans if s.kind is SpanKind.LLM)
        assert llm.model == "claude-opus-4-8"
        assert llm.token_usage is not None
        assert llm.token_usage.total_tokens == 1200

    def test_tool_calls_extracted(self, synthetic_trace: Any) -> None:
        nt = normalize_trace(synthetic_trace)
        assert nt.tool_counts == {"Read": 1, "Bash": 1}
        read = next(tc for tc in nt.tool_calls if tc.name == "Read")
        assert read.id == "toolu_1"
        assert read.arguments == {"file_path": "/a"}
        assert read.status is TraceStatus.OK
        bash = next(tc for tc in nt.tool_calls if tc.name == "Bash")
        assert bash.status is TraceStatus.ERROR  # the errored span maps through


class TestTokenUsageFallbacks:
    """Token usage must resolve from metadata or span sums, not just the
    trace-level field."""

    def _trace_from_modified(self, mutate: Any) -> Any:
        from mlflow.entities import Trace

        data = json.loads((FIXTURE_DIR / "synthetic_trace.json").read_text())
        mutate(data)
        return Trace.from_dict(data)

    def test_falls_back_to_span_usage(self) -> None:
        def drop_trace_level(data: dict[str, Any]) -> None:
            data["info"]["trace_metadata"].pop("mlflow.trace.tokenUsage", None)

        trace = self._trace_from_modified(drop_trace_level)
        # Even without the trace-level total, the LLM span's usage is summed.
        assert normalize_trace(trace).token_usage.total_tokens == 1200


class TestHelpers:
    def test_token_usage_from_dict_none(self) -> None:
        assert _token_usage_from_dict(None) is None
        assert _token_usage_from_dict("not a dict") is None

    def test_token_usage_from_dict(self) -> None:
        usage = _token_usage_from_dict({"input_tokens": 5, "output_tokens": 7, "total_tokens": 12})
        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens == 12

    def test_maybe_json(self) -> None:
        assert _maybe_json('{"a": 1}') == {"a": 1}
        assert _maybe_json("not json") == "not json"
        assert _maybe_json({"a": 1}) == {"a": 1}

    def test_as_dict(self) -> None:
        assert _as_dict({"a": 1}) == {"a": 1}
        assert _as_dict(None) == {}
        assert _as_dict("x") == {"input": "x"}


@pytest.mark.live
def test_live_pull_reference_experiment() -> None:
    """Acceptance: connect to the reference experiment and normalize traces.

    Guarded by ``AIL_LIVE_MLFLOW=1`` so the default suite is green offline.
    Profile defaults to ``dais-demo`` (override with ``AIL_DATABRICKS_PROFILE``).
    """
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        pytest.skip("set AIL_LIVE_MLFLOW=1 to run the live MLflow pull")

    source = MLflowTraceSource(profile=os.environ.get("AIL_DATABRICKS_PROFILE", "dais-demo"))
    traces = source.fetch_traces(experiment_id=REFERENCE_EXPERIMENT, max_results=10)

    assert traces, "expected at least one trace in the reference experiment"
    sample = traces[0]
    assert sample.trace_id
    assert sample.experiment_id == REFERENCE_EXPERIMENT
    assert isinstance(sample.token_usage, TokenUsage)
    assert sample.status in set(TraceStatus)
