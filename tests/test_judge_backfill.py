"""Offline tests for idempotent full-corpus judge coverage repair."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ail.judges.backfill as backfill
from ail.ingest.base import NormalizedTrace, TraceSource
from ail.ingest.mlflow_source import normalize_trace


class _Source(TraceSource):
    def __init__(self, traces: list[NormalizedTrace]) -> None:
        self.traces = traces
        self.calls: list[dict[str, Any]] = []

    def iter_traces(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        yield from self.traces

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        return next((trace for trace in self.traces if trace.trace_id == trace_id), None)


def _assessment(name: str, *, value: Any = "yes", source: str = "LLM_JUDGE") -> Any:
    return SimpleNamespace(
        name=name,
        source=SimpleNamespace(source_type=source),
        feedback=SimpleNamespace(value=value),
        error=None,
    )


def _trace(
    trace_id: str,
    *,
    assessments: list[Any] | None = None,
    internal: bool = False,
) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=trace_id,
        tags={"ail.internal": "true"} if internal else {},
        raw=SimpleNamespace(info=SimpleNamespace(assessments=assessments or [])),
    )


def test_successful_assessment_requires_judge_source_and_value() -> None:
    assert backfill.has_successful_judge_assessment(
        _trace("covered", assessments=[_assessment("correctness")]), "correctness"
    )
    assert not backfill.has_successful_judge_assessment(
        _trace("human", assessments=[_assessment("correctness", source="HUMAN")]),
        "correctness",
    )
    assert not backfill.has_successful_judge_assessment(
        _trace("empty", assessments=[_assessment("correctness", value=None)]), "correctness"
    )


def test_feedback_parts_recovers_valid_json_wrapped_by_the_judge() -> None:
    error = SimpleNamespace(
        error_message=(
            "Invalid JSON response\n\nLLM output: ```json\n"
            '{"result": "yes", "rationale": "supported by the trace"}\n```'
        )
    )
    feedback = SimpleNamespace(name="correctness", value=None, error=error)
    assert backfill._feedback_parts(feedback, "correctness") == (
        "yes",
        "supported by the trace",
    )


def test_trace_digest_is_bounded_and_preserves_observed_totals() -> None:
    trace = _trace("large")
    trace.request_preview = "request"
    trace.response_preview = "response"
    trace.token_usage.input_tokens = 40
    trace.token_usage.output_tokens = 2
    digest = backfill._trace_digest(trace)
    assert '"trace_id": "large"' in digest
    assert '"total_tokens": 42' in digest
    assert "Bounded digest" in digest


def test_bounded_trace_is_a_single_span_trace() -> None:
    from mlflow.entities import Trace

    raw = json.loads((Path(__file__).parent / "fixtures" / "synthetic_trace.json").read_text())
    trace_id = "tr-00000000000000000000000000000001"
    raw["info"]["trace_id"] = trace_id
    encoded_trace_id = base64.b64encode(bytes.fromhex(trace_id.removeprefix("tr-"))).decode()
    for span in raw["data"]["spans"]:
        span["trace_id"] = encoded_trace_id
        span["attributes"]["mlflow.traceRequestId"] = json.dumps(trace_id)
    trace = normalize_trace(Trace.from_dict(raw))
    bounded = backfill._bounded_trace(trace)
    assert len(bounded.data.spans) == 1
    assert "trace_digest" in bounded.data.spans[0].inputs
    assert bounded.info.assessments == []
    assert bounded.info.trace_metadata == {"ail.judge.input_mode": "bounded_digest"}


def test_backfill_skips_internal_and_covered_pairs_then_evaluates_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Source(
        [
            _trace("internal", internal=True),
            _trace("one", assessments=[_assessment("correctness")]),
            _trace("two"),
        ]
    )
    scorers = [SimpleNamespace(name="correctness"), SimpleNamespace(name="groundedness")]
    evaluated: list[tuple[str, str]] = []
    monkeypatch.setattr(backfill, "_configure_mlflow", lambda **_: None)

    def evaluate(trace: NormalizedTrace, scorer: Any) -> backfill.JudgeBackfillOutcome:
        evaluated.append((trace.trace_id, scorer.name))
        return backfill.JudgeBackfillOutcome(trace.trace_id, scorer.name, "evaluated")

    monkeypatch.setattr(backfill, "_evaluate_one", evaluate)
    report = backfill.run_judge_backfill(
        "subject-exp",
        reviewer_experiment_id="reviewer-exp",
        source=source,
        scorers=scorers,
        max_evaluations=10,
        max_workers=1,
    )

    assert source.calls == [
        {
            "experiment_id": "subject-exp",
            "max_results": None,
            "order_by": ["timestamp_ms ASC"],
        }
    ]
    assert evaluated == [
        ("one", "groundedness"),
        ("two", "correctness"),
        ("two", "groundedness"),
    ]
    assert report.n_scanned == 3
    assert report.n_internal_skipped == 1
    assert report.n_already_covered == 1
    assert report.n_selected == 3
    assert report.n_evaluated == 3
    assert report.n_failed == 0


def test_discovered_backfill_scorers_exclude_custom_code_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ail.judges import registration

    source = _Source([_trace("one")])
    monkeypatch.setattr(backfill, "_configure_mlflow", lambda **_: None)
    monkeypatch.setattr(
        registration,
        "list_registered_scorers",
        lambda *_args, **_kwargs: [
            SimpleNamespace(name="correctness", kind="instructions"),
            SimpleNamespace(name="duration_seconds", kind="decorator"),
        ],
    )
    evaluated: list[str] = []

    def evaluate(trace: NormalizedTrace, scorer: Any) -> backfill.JudgeBackfillOutcome:
        evaluated.append(scorer.name)
        return backfill.JudgeBackfillOutcome(trace.trace_id, scorer.name, "evaluated")

    monkeypatch.setattr(backfill, "_evaluate_one", evaluate)
    backfill.run_judge_backfill(
        "subject-exp",
        reviewer_experiment_id="reviewer-exp",
        source=source,
        max_evaluations=10,
        max_workers=1,
    )

    assert evaluated == ["correctness"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reviewer_experiment_id": ""}, "reviewer_experiment_id"),
        ({"reviewer_experiment_id": "reviewer", "max_evaluations": 0}, "max_evaluations"),
        ({"reviewer_experiment_id": "reviewer", "max_workers": 0}, "max_workers"),
    ],
)
def test_backfill_rejects_unsafe_configuration(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        backfill.run_judge_backfill("subject", source=_Source([]), scorers=[], **kwargs)
