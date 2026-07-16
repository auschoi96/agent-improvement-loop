"""Idempotent judge coverage repair for subject traces."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.trace_policy import is_internal_trace

if TYPE_CHECKING:
    from ail.ingest.base import NormalizedTrace, TraceSource

__all__ = [
    "JudgeBackfillOutcome",
    "JudgeBackfillReport",
    "has_successful_judge_assessment",
    "run_judge_backfill",
]


@dataclass(frozen=True, slots=True)
class JudgeBackfillOutcome:
    trace_id: str
    judge_name: str
    status: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class JudgeBackfillReport:
    experiment_id: str
    reviewer_experiment_id: str
    n_scanned: int
    n_internal_skipped: int
    n_already_covered: int
    n_selected: int
    n_evaluated: int
    n_failed: int
    outcomes: tuple[JudgeBackfillOutcome, ...]


def _source_type(assessment: Any) -> str:
    source = getattr(assessment, "source", None)
    raw = getattr(source, "source_type", "")
    return str(getattr(raw, "value", raw) or "")


def _assessment_value(assessment: Any) -> Any:
    feedback = getattr(assessment, "feedback", None)
    return getattr(feedback, "value", None) if feedback is not None else None


def has_successful_judge_assessment(trace: NormalizedTrace, judge_name: str) -> bool:
    """Return whether ``trace`` has a successful LLM-judge assessment by this name."""
    info = getattr(getattr(trace, "raw", None), "info", None)
    for assessment in list(getattr(info, "assessments", None) or []):
        if str(getattr(assessment, "name", "") or "") != judge_name:
            continue
        if _source_type(assessment) not in {"LLM_JUDGE", "AI_JUDGE"}:
            continue
        if getattr(assessment, "error", None) is None and _assessment_value(assessment) is not None:
            return True
    return False


def _configure_mlflow(
    *, profile: str | None, reviewer_experiment_id: str, sql_warehouse_id: str | None
) -> None:
    import mlflow

    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    if sql_warehouse_id:
        os.environ[TRACING_WAREHOUSE_ENV] = sql_warehouse_id
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_id=reviewer_experiment_id)


def _feedback_parts(result: Any, judge_name: str) -> tuple[Any, str | None]:
    values = result if isinstance(result, list) else [result]
    selected = next(
        (item for item in values if str(getattr(item, "name", "") or "") == judge_name),
        values[0] if values else None,
    )
    if selected is None:
        raise ValueError(f"judge {judge_name!r} returned no feedback")
    if hasattr(selected, "value"):
        error = getattr(selected, "error", None)
        if error is not None:
            recovered = _recover_judge_json(error)
            if recovered is not None:
                return recovered
            raise RuntimeError(str(error))
        return selected.value, getattr(selected, "rationale", None)
    return selected, None


def _recover_judge_json(error: Any) -> tuple[Any, str | None] | None:
    message = str(getattr(error, "error_message", None) or error)
    if "LLM output:" not in message:
        return None
    output = message.split("LLM output:", 1)[1].strip()
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", output, flags=re.DOTALL)
    if not candidates:
        start = output.find("{")
        end = output.rfind("}")
        if start >= 0 and end > start:
            candidates = [output[start : end + 1]]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "result" in parsed:
            rationale = parsed.get("rationale")
            return parsed["result"], str(rationale) if rationale is not None else None
    return None


def _trace_digest(trace: NormalizedTrace) -> str:
    tool_calls = []
    for call in trace.tool_calls[:100]:
        tool_calls.append(
            {
                "name": call.name,
                "status": call.status.value,
                "arguments": json.dumps(call.arguments, default=str)[:4_000],
                "result": call.result[:1_500] if call.result else None,
            }
        )
    errors = []
    for span in trace.spans:
        if span.status.value == "ERROR":
            errors.append(
                {
                    "name": span.name,
                    "kind": span.kind.value,
                    "attributes": json.dumps(span.attributes, default=str)[:2_000],
                }
            )
        if len(errors) >= 25:
            break
    return json.dumps(
        {
            "notice": "Bounded digest of an oversized MLflow trace; all values are observed.",
            "trace_id": trace.trace_id,
            "status": trace.status.value,
            "request": (trace.request_preview or "")[:12_000],
            "response": (trace.response_preview or "")[:16_000],
            "execution_duration_ms": trace.execution_duration_ms,
            "token_usage": {
                "input_tokens": trace.token_usage.input_tokens,
                "output_tokens": trace.token_usage.output_tokens,
                "total_tokens": trace.total_tokens,
                "cache_tokens": trace.token_usage.cache_tokens,
            },
            "tool_calls": tool_calls,
            "error_spans": errors,
        },
        default=str,
    )


def _bounded_trace(trace: NormalizedTrace) -> Any:
    from mlflow.entities import Trace

    raw = trace.raw.to_dict()
    spans = list(raw.get("data", {}).get("spans", []))
    root = next((span for span in spans if span.get("parent_span_id") is None), None)
    if root is None:
        raise ValueError(f"trace {trace.trace_id} has no root span")
    attributes = root.get("attributes", {})
    retained = {
        key: value
        for key, value in attributes.items()
        if key
        in {
            "mlflow.traceRequestId",
            "mlflow.spanType",
            "mlflow.spanStartTimeNs",
        }
    }
    retained["mlflow.spanInputs"] = json.dumps({"trace_digest": json.loads(_trace_digest(trace))})
    retained["mlflow.spanOutputs"] = json.dumps(
        {"response": (trace.response_preview or "")[:16_000]}
    )
    root["attributes"] = retained
    raw["data"]["spans"] = [root]
    raw["info"]["request_preview"] = (trace.request_preview or "")[:12_000]
    raw["info"]["response_preview"] = (trace.response_preview or "")[:16_000]
    raw["info"]["trace_metadata"] = {"ail.judge.input_mode": "bounded_digest"}
    raw["info"]["assessments"] = []
    return Trace.from_dict(raw)


def _score_trace(
    trace: NormalizedTrace, scorer: Any, judge_name: str
) -> tuple[Any, str | None, str]:
    try:
        value, rationale = _feedback_parts(scorer(trace=trace.raw), judge_name)
        return value, rationale, "full_trace"
    except RuntimeError as exc:
        message = str(exc).lower()
        limit_error = any(
            marker in message
            for marker in (
                "context limit exceeded",
                "exceeds maximum allowed content length",
                "requestsize(bytes)",
            )
        )
        if not limit_error:
            raise
    value, rationale = _feedback_parts(scorer(trace=_bounded_trace(trace)), judge_name)
    return value, rationale, "bounded_digest"


def _evaluate_one(trace: NormalizedTrace, scorer: Any) -> JudgeBackfillOutcome:
    import mlflow
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    judge_name = str(getattr(scorer, "name", "") or "")
    try:
        with mlflow.start_span(
            name="ail_judge_backfill",
            span_type="AGENT",
            attributes={
                "ail.judge.subject_trace_id": trace.trace_id,
                "ail.judge.name": judge_name,
            },
        ) as span:
            mlflow.update_current_trace(
                tags={"ail.internal": "true", "mlflow.traceName": "ail_judge_backfill"}
            )
            value, rationale, input_mode = _score_trace(trace, scorer, judge_name)
            reviewer_trace_id = span.trace_id
        mlflow.log_feedback(
            trace_id=trace.trace_id,
            name=judge_name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=f"ail.judge.backfill:{judge_name}",
            ),
            rationale=rationale,
            metadata={
                "ail.judge.backfill": "true",
                "ail.judge.input_mode": input_mode,
                "reviewer_trace_id": reviewer_trace_id,
            },
        )
        return JudgeBackfillOutcome(trace.trace_id, judge_name, "evaluated")
    except Exception as exc:  # noqa: BLE001 - one judge/trace must not stop the sweep
        return JudgeBackfillOutcome(
            trace.trace_id, judge_name, "failed", f"{type(exc).__name__}: {exc}"
        )


def run_judge_backfill(
    experiment_id: str,
    *,
    reviewer_experiment_id: str,
    sql_warehouse_id: str | None = None,
    source: TraceSource | None = None,
    scorers: list[Any] | None = None,
    profile: str | None = None,
    max_results: int | None = None,
    max_evaluations: int = 32,
    max_workers: int = 4,
) -> JudgeBackfillReport:
    """Evaluate missing registered-judge assessments across the full subject corpus."""
    if not reviewer_experiment_id.strip():
        raise ValueError("reviewer_experiment_id is required for isolated judge traces")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    _configure_mlflow(
        profile=profile,
        reviewer_experiment_id=reviewer_experiment_id,
        sql_warehouse_id=sql_warehouse_id,
    )
    if source is None:
        from ail.ingest.mlflow_source import MLflowTraceSource

        source = MLflowTraceSource(profile=profile)
    if scorers is None:
        from ail.judges.registration import is_code_scorer, list_registered_scorers

        scorers = [
            scorer
            for scorer in list_registered_scorers(experiment_id, profile=profile)
            if not is_code_scorer(scorer)
        ]

    traces = list(
        source.iter_traces(
            experiment_id=experiment_id,
            max_results=max_results,
            order_by=["timestamp_ms ASC"],
        )
    )
    internal = [trace for trace in traces if is_internal_trace(trace)]
    subjects = [trace for trace in traces if not is_internal_trace(trace)]
    tasks: list[tuple[NormalizedTrace, Any]] = []
    already_covered = 0
    for trace in subjects:
        for scorer in scorers:
            name = str(getattr(scorer, "name", "") or "")
            if not name:
                continue
            if has_successful_judge_assessment(trace, name):
                already_covered += 1
                continue
            tasks.append((trace, scorer))
            if len(tasks) >= max_evaluations:
                break
        if len(tasks) >= max_evaluations:
            break

    outcomes: list[JudgeBackfillOutcome] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks) or 1)) as pool:
        futures = [pool.submit(_evaluate_one, trace, scorer) for trace, scorer in tasks]
        for future in as_completed(futures):
            outcomes.append(future.result())
    outcomes.sort(key=lambda item: (item.trace_id, item.judge_name))
    n_evaluated = sum(item.status == "evaluated" for item in outcomes)
    return JudgeBackfillReport(
        experiment_id=experiment_id,
        reviewer_experiment_id=reviewer_experiment_id,
        n_scanned=len(traces),
        n_internal_skipped=len(internal),
        n_already_covered=already_covered,
        n_selected=len(tasks),
        n_evaluated=n_evaluated,
        n_failed=len(outcomes) - n_evaluated,
        outcomes=tuple(outcomes),
    )
