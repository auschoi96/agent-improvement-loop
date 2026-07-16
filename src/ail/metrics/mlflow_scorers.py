"""MLflow production-monitoring scorers for AIL's deterministic L0 metrics.

AIL keeps its governed L0 tables for cohort analysis, readiness, comparison, and
version evidence.  These scorers add the complementary MLflow-native surface:
each exact measurement is registered as a custom ``@scorer`` so Databricks
production monitoring evaluates incoming traces asynchronously and attaches the
numeric result to the trace itself.

The decorated functions are deliberately self-contained.  Databricks serializes
their source for remote execution, so every runtime import and every constant the
function needs lives inside its body.  Do not refactor their calculation into a
module helper unless the production-scorer serializer gains dependency capture.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mlflow.genai.scorers import scorer

from ail.judges.registration import _configure_databricks, _require_databricks_agents

__all__ = [
    "DETERMINISTIC_MLFLOW_SCORERS",
    "duration_seconds_scorer",
    "redundancy_rate_scorer",
    "total_tokens_scorer",
    "total_tool_calls_scorer",
    "total_usd_scorer",
    "register_deterministic_scorers",
]


@scorer(
    name="duration_seconds",
    description="Exact wall-clock duration of the complete trace in seconds.",
    aggregations=["mean", "min", "max", "median", "p90"],
)
def duration_seconds_scorer(trace: Any) -> Any:
    from mlflow.entities import Feedback

    duration_ms = getattr(getattr(trace, "info", None), "execution_duration", None)
    if duration_ms is not None:
        value = float(duration_ms) / 1000.0
    else:
        spans = list(trace.search_spans())
        roots = [span for span in spans if getattr(span, "parent_id", None) is None]
        candidates = roots or spans
        starts = [
            int(raw_start)
            for span in candidates
            if (raw_start := getattr(span, "start_time_ns", None)) is not None
        ]
        ends = [
            int(raw_end)
            for span in candidates
            if (raw_end := getattr(span, "end_time_ns", None)) is not None
        ]
        if not starts or not ends:
            return Feedback(
                name="duration_seconds",
                value=None,
                error="trace has no execution duration or timed spans",
                rationale="Duration could not be measured from this trace.",
                metadata={"unit": "seconds", "ail.metric_kind": "deterministic"},
                valid=False,
            )
        value = (max(ends) - min(starts)) / 1_000_000_000.0
    return Feedback(
        name="duration_seconds",
        value=round(value, 6),
        rationale=f"Trace wall-clock duration is {value:.6f} seconds.",
        metadata={"unit": "seconds", "ail.metric_kind": "deterministic"},
    )


@scorer(
    name="total_tokens",
    description="Exact input plus output token usage recorded on the trace.",
    aggregations=["mean", "min", "max", "median", "p90"],
)
def total_tokens_scorer(trace: Any) -> Any:
    import json

    from mlflow.entities import Feedback

    usage = getattr(getattr(trace, "info", None), "token_usage", None)
    if usage:
        value = usage.get("total_tokens")
        if value is None:
            value = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    else:
        value = 0
        found = False
        for span in trace.search_spans():
            raw = (getattr(span, "attributes", None) or {}).get("mlflow.chat.tokenUsage")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (TypeError, ValueError):
                    raw = None
            if not isinstance(raw, dict):
                continue
            found = True
            span_total = raw.get("total_tokens")
            if span_total is None:
                span_total = int(raw.get("input_tokens") or 0) + int(raw.get("output_tokens") or 0)
            value += int(span_total)
        if not found:
            return Feedback(
                name="total_tokens",
                value=None,
                error="trace has no recorded token usage",
                rationale="Token usage was not emitted by the traced model provider.",
                metadata={"unit": "tokens", "ail.metric_kind": "deterministic"},
                valid=False,
            )
    return Feedback(
        name="total_tokens",
        value=int(value),
        rationale=f"Trace metadata records {int(value)} total input and output tokens.",
        metadata={"unit": "tokens", "ail.metric_kind": "deterministic"},
    )


@scorer(
    name="total_tool_calls",
    description="Exact number of TOOL spans recorded on the trace.",
    aggregations=["mean", "min", "max", "median", "p90"],
)
def total_tool_calls_scorer(trace: Any) -> Any:
    from mlflow.entities import Feedback

    count = sum(
        1 for span in trace.search_spans() if str(getattr(span, "span_type", "")).upper() == "TOOL"
    )
    return Feedback(
        name="total_tool_calls",
        value=count,
        rationale=f"The trace contains {count} TOOL spans.",
        metadata={"unit": "calls", "ail.metric_kind": "deterministic"},
    )


@scorer(
    name="redundancy_rate",
    description="Fraction of TOOL calls that repeat an identical tool name and input.",
    aggregations=["mean", "min", "max", "median", "p90"],
)
def redundancy_rate_scorer(trace: Any) -> Any:
    import json

    from mlflow.entities import Feedback

    signatures = []
    for span in trace.search_spans():
        if str(getattr(span, "span_type", "")).upper() != "TOOL":
            continue
        attributes = getattr(span, "attributes", None) or {}
        tool_name = attributes.get("tool_name") or getattr(span, "name", "")
        tool_name = str(tool_name)
        if tool_name.startswith("tool_"):
            tool_name = tool_name[len("tool_") :]
        inputs = getattr(span, "inputs", None)
        try:
            canonical_inputs = json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            canonical_inputs = str(inputs)
        signatures.append(f"{tool_name}\x1f{canonical_inputs}")
    total = len(signatures)
    redundant = total - len(set(signatures))
    value = round(redundant / total, 6) if total else 0.0
    return Feedback(
        name="redundancy_rate",
        value=value,
        rationale=f"{redundant} of {total} TOOL calls repeat an identical tool/input signature.",
        metadata={"unit": "fraction", "ail.metric_kind": "deterministic"},
    )


@scorer(
    name="total_usd",
    description="Estimated trace cost in USD from recorded model and token usage.",
    aggregations=["mean", "min", "max", "median", "p90"],
)
def total_usd_scorer(trace: Any) -> Any:
    import json

    from mlflow.entities import Feedback

    # Prefer cost emitted directly by MLflow/provider instrumentation when it is
    # available. Claude Code OTEL traces currently omit it, so the fallback below
    # uses the same explicit, fail-closed AIL pricebook as the governed L0 table.
    reported = getattr(getattr(trace, "info", None), "cost", None)
    if isinstance(reported, dict):
        direct = reported.get("total_cost")
        if direct is None:
            direct = reported.get("total_usd")
        if direct is not None:
            value = round(float(direct), 6)
            return Feedback(
                name="total_usd",
                value=value,
                rationale=f"Trace/provider metadata reports an estimated cost of ${value:.6f}.",
                metadata={
                    "unit": "usd",
                    "ail.metric_kind": "deterministic_estimate",
                    "ail.price_source": "trace_metadata",
                },
            )

    # USD per million tokens. This table is intentionally embedded because
    # production custom scorers are serialized without module dependencies.
    prices = {
        "claude-opus-4-8": (5.0, 25.0),
        "claude-opus-4-7": (5.0, 25.0),
        "claude-opus-4-6": (5.0, 25.0),
        "claude-opus-4-5": (5.0, 25.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-fable-5": (10.0, 50.0),
    }
    aliases = {
        "claude-haiku-4-5-20251001": "claude-haiku-4-5",
        "claude-opus-4-5-20251101": "claude-opus-4-5",
        "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
    }
    total = 0.0
    found_usage = False
    unpriced = set()
    for span in trace.search_spans():
        attributes = getattr(span, "attributes", None) or {}
        raw_usage = attributes.get("mlflow.chat.tokenUsage")
        if isinstance(raw_usage, str):
            try:
                raw_usage = json.loads(raw_usage)
            except (TypeError, ValueError):
                raw_usage = None
        if not isinstance(raw_usage, dict):
            continue
        found_usage = True
        raw_model = attributes.get("model") or getattr(span, "model_name", None)
        model = str(raw_model or "").strip().strip('"').lower()
        if model.startswith("anthropic."):
            model = model[len("anthropic.") :]
        model = aliases.get(model, model)
        rates = prices.get(model)
        if rates is None:
            unpriced.add(model or "<missing model>")
            continue
        input_rate, output_rate = rates
        input_tokens = int(raw_usage.get("input_tokens") or 0)
        output_tokens = int(raw_usage.get("output_tokens") or 0)
        cache_write = int(raw_usage.get("cache_creation_input_tokens") or 0)
        cache_read = int(raw_usage.get("cache_read_input_tokens") or 0)
        total += input_tokens / 1_000_000 * input_rate
        total += output_tokens / 1_000_000 * output_rate
        total += cache_write / 1_000_000 * (input_rate * 1.25)
        total += cache_read / 1_000_000 * (input_rate * 0.10)
    if not found_usage:
        return Feedback(
            name="total_usd",
            value=None,
            error="trace has no per-model token usage; cost was not estimated",
            rationale="AIL refuses to fabricate a dollar value without recorded usage.",
            metadata={"unit": "usd", "ail.metric_kind": "deterministic_estimate"},
            valid=False,
        )
    if unpriced:
        models = ", ".join(sorted(unpriced))
        return Feedback(
            name="total_usd",
            value=None,
            error=f"model(s) absent from the AIL pricebook: {models}",
            rationale="AIL refuses partial or guessed cost estimates for unpriced models.",
            metadata={"unit": "usd", "ail.metric_kind": "deterministic_estimate"},
            valid=False,
        )
    value = round(total, 6)
    return Feedback(
        name="total_usd",
        value=value,
        rationale=f"Recorded per-model token usage maps to an estimated cost of ${value:.6f}.",
        metadata={
            "unit": "usd",
            "ail.metric_kind": "deterministic_estimate",
            "ail.price_source": "AIL pricebook 2026-06-04",
        },
    )


DETERMINISTIC_MLFLOW_SCORERS = {
    "total_tokens": total_tokens_scorer,
    "total_usd": total_usd_scorer,
    "redundancy_rate": redundancy_rate_scorer,
    "total_tool_calls": total_tool_calls_scorer,
    "duration_seconds": duration_seconds_scorer,
}


def register_deterministic_scorers(
    experiment_id: str,
    metric_names: Iterable[str],
    *,
    sampling_rate: float = 1.0,
    filter_string: str | None = None,
    profile: str | None = None,
    warehouse_id: str | None = None,
) -> list[str]:
    """Register/update selected deterministic metrics for continuous monitoring.

    The operation is idempotent. New metrics use ``register`` then ``start``;
    existing metrics are updated with the current serialized implementation and
    sampling configuration. Every metric defaults to 100% sampling because these
    code scorers do not incur an LLM call.
    """
    if not experiment_id.strip():
        raise ValueError("experiment_id is required")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate!r}")
    selected = list(dict.fromkeys(str(name).strip() for name in metric_names if str(name).strip()))
    unknown = sorted(set(selected) - set(DETERMINISTIC_MLFLOW_SCORERS))
    if unknown:
        raise ValueError(f"unknown deterministic MLflow scorer(s): {', '.join(unknown)}")
    if not selected:
        return []

    _require_databricks_agents()
    _configure_databricks(profile=profile, tracking_uri="databricks", registry_uri="databricks-uc")
    import mlflow
    from mlflow.genai.scorers import ScorerSamplingConfig, list_scorers

    mlflow.set_experiment(experiment_id=experiment_id)
    if warehouse_id:
        from mlflow.tracing import set_databricks_monitoring_sql_warehouse_id

        set_databricks_monitoring_sql_warehouse_id(
            sql_warehouse_id=warehouse_id,
            experiment_id=experiment_id,
        )
    existing = {item.name: item for item in list_scorers(experiment_id=experiment_id)}
    config = ScorerSamplingConfig(sample_rate=sampling_rate, filter_string=filter_string)
    for name in selected:
        definition = DETERMINISTIC_MLFLOW_SCORERS[name]
        if name in existing:
            definition.update(name=name, experiment_id=experiment_id, sampling_config=config)
        else:
            registered = definition.register(name=name, experiment_id=experiment_id)
            registered.start(experiment_id=experiment_id, sampling_config=config)
    return selected
