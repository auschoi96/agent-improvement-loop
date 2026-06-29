"""L3 recursive trace reviewer: run HALO over a trace and attach a verdict.

This is the thin MLflow adapter around the **adopted** HALO engine
(``halo-engine``, see ``PROVENANCE.md`` / ``NOTICE``). It does four things, in
order:

1. **Export** the subject trace to the flat OpenInference/OTLP JSONL HALO
   indexes (:func:`ail.l3.adapter.mlflow_trace_to_otlp_jsonl`).
2. **Review** it by running HALO (``engine.main.run_engine``) against a
   Databricks FMAPI (OpenAI-compatible) serving endpoint, with a prompt that
   asks for a structured verdict — HALO navigates the arbitrarily-large trace
   with its own bounded tools, so a 943K-token trace never lands in a single
   model call.
3. **Isolate** that review as HALO's *own* MLflow trace and parse HALO's
   free-text ``<final/>`` report into a :class:`~ail.l3.contract.HaloReviewVerdict`.
4. **Attach** the verdict to the *subject* trace as an ``LLM_JUDGE`` feedback
   assessment (``mlflow.log_feedback``, name ``l3_halo_review``) linked back to
   the reviewer's trace via ``metadata.reviewer_trace_id``. (``AI_JUDGE`` is a
   deprecated alias of ``LLM_JUDGE`` in ``mlflow>=3``.)

**Why the own-trace isolation matters (``docs/ARCHITECTURE.md`` §11).** The L0
cost metric reads ``mlflow.trace.tokenUsage``, which sums a trace's child LLM
spans. If HALO's review ran *inside* the subject trace, the subject's measured
cost would silently absorb the *reviewer's* tokens — breaking the one metric the
architecture promises to keep deterministic and un-gameable. So HALO's work
runs under a separate trace and only a verdict *assessment* is attached to the
subject; HALO spans are never nested in the agent's trace.

HALO is imported lazily inside :func:`run_halo_review` so the rest of the module
(and ``import ail.l3``) stays importable without the optional ``l3`` extra.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ail.ingest.base import TraceSource
from ail.l3.adapter import OtlpExport, mlflow_trace_to_otlp_jsonl
from ail.l3.contract import HaloReviewVerdict
from ail.l3.parser import parse_halo_report

if TYPE_CHECKING:
    from engine.engine_config import EngineConfig

__all__ = [
    "FEEDBACK_NAME",
    "REVIEW_PROMPT_TEMPLATE",
    "build_engine_config",
    "run_halo_review",
    "review_trace",
]

#: Assessment name the verdict is attached under. Together with the
#: ``LLM_JUDGE`` source type this keys the assessment on the subject trace,
#: alongside (not colliding with) the L2 judge and ``HUMAN`` assessments, which
#: carry different names.
FEEDBACK_NAME = "l3_halo_review"

#: Name of the reviewer's own MLflow trace/span.
REVIEW_SPAN_NAME = "l3_halo_review"

# Conservative HALO bounds. Depth 2 (root + one subagent layer) and a modest
# turn budget keep an L3 review from running away on a huge trace; raise them
# deliberately for a deeper review.
_DEFAULT_MAX_TURNS = 40
_DEFAULT_MAX_DEPTH = 2
_DEFAULT_MAX_PARALLEL = 4

#: The review prompt. HALO returns free text, so the structure is imposed here:
#: the prompt fixes exactly what to look for and demands a single JSON object
#: (matching :class:`~ail.l3.contract.HaloReviewVerdict`) before the ``<final/>``
#: marker. :mod:`ail.l3.parser` extracts and validates that object.
REVIEW_PROMPT_TEMPLATE = """\
You are an expert reviewer auditing a single coding-agent execution trace for \
TOKEN EFFICIENCY and QUALITY. The trace id is `{trace_id}`. It may be very large \
(hundreds of thousands of tokens) — use your trace navigation tools to inspect \
it incrementally; never assume you must read it all at once.

Work AUTONOMOUSLY. Begin immediately and call your trace-navigation tools \
yourself. Do NOT ask for confirmation, permission, or clarification, and do NOT \
stop to check in — there is no human available to answer, and any turn that ends \
without a tool call or the final verdict ends the whole review. Keep \
investigating across as many turns as you need to gather evidence, then write \
the final verdict.

Focus your review on:
1. TOKEN WASTE / AVOIDABLE REDUNDANCY — repeated tool calls against the same \
target (e.g. the same file read many times, the same shell setup re-run), \
re-fetching context the agent already had, and other spend that produced no new \
information. Quantify it where you can (which tool, which target, how many times, \
roughly how many tokens wasted).
2. QUALITY-PER-TOKEN — did the spend buy progress? Grade how efficiently the \
trace converted tokens into useful work.
3. NOTABLE FAILURE MODES — looping, abandoned plans, ignored errors, \
hallucinated paths/APIs, or other quality problems a fixed scorer would miss. \
Cite span ids as evidence.

Only after you have actually inspected the trace with your tools, end your \
report with a SINGLE JSON object on its own, inside a ```json fenced block, with \
exactly these fields, then the marker <final/>:

```json
{{
  "token_efficiency": "poor | fair | good | excellent",
  "token_waste_score": <integer 0-100, share of spend that was avoidable>,
  "estimated_wasted_tokens": <integer or null>,
  "summary": "<2-4 sentence overall assessment>",
  "redundancy_findings": [
    {{
      "description": "<what was repeated>",
      "tool": "<tool name or null>",
      "repeated_target": "<file path / command / target or null>",
      "occurrences": <integer or null>,
      "estimated_wasted_tokens": <integer or null>,
      "evidence_span_ids": ["<span id>", "..."]
    }}
  ],
  "failure_modes": [
    {{
      "title": "<short title>",
      "severity": "low | medium | high",
      "description": "<what went wrong>",
      "evidence_span_ids": ["<span id>", "..."]
    }}
  ],
  "recommendations": ["<concrete, actionable fix>", "..."]
}}
```
"""


def build_engine_config(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
    temperature: float | None = None,
) -> EngineConfig:
    """Build a HALO :class:`EngineConfig` from a model + OpenAI-compatible endpoint.

    A thin assembly over HALO's *public* config classes (the same shape the HALO
    CLI builds), using one :class:`ModelConfig` for the root agent, subagents,
    synthesis, and compaction. ``base_url`` / ``api_key`` are threaded onto the
    provider; when ``None`` HALO's underlying OpenAI client falls back to the
    ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` environment variables.

    Imports HALO lazily — calling this without the ``l3`` extra installed raises
    :class:`ImportError` with install guidance.
    """
    try:
        from engine.agents.agent_config import AgentConfig
        from engine.engine_config import EngineConfig
        from engine.model_config import ModelConfig
        from engine.model_provider_config import ModelProviderConfig
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the L3 HALO reviewer requires the 'halo-engine' package. "
            "Install it with: pip install 'ail[l3]'"
        ) from exc

    model_cfg = ModelConfig(name=model, temperature=temperature)
    return EngineConfig(
        root_agent=AgentConfig(name="l3-reviewer", model=model_cfg, maximum_turns=max_turns),
        subagent=AgentConfig(name="l3-reviewer-subagent", model=model_cfg, maximum_turns=max_turns),
        synthesis_model=model_cfg,
        compaction_model=model_cfg,
        model_provider=ModelProviderConfig(base_url=base_url, api_key=api_key),
        maximum_depth=max_depth,
        maximum_parallel_subagents=max_parallel,
    )


def _extract_report_text(output_items: list[Any]) -> str:
    """Concatenate the text of HALO's terminating (root, ``final``) messages.

    HALO marks the root agent's terminating assistant message with
    ``AgentOutputItem.final``. We join those messages' string content (the
    free-text report); if none is marked final, fall back to the last
    assistant message so a verdict can still be attempted.
    """

    def _text(item: Any) -> str:
        msg = getattr(item, "item", None)
        content = getattr(msg, "content", None)
        return content if isinstance(content, str) else ""

    finals = [_text(it) for it in output_items if getattr(it, "final", False)]
    finals = [t for t in finals if t]
    if finals:
        return "\n\n".join(finals)

    assistant_texts = [
        _text(it)
        for it in output_items
        if getattr(getattr(it, "item", None), "role", None) == "assistant"
    ]
    assistant_texts = [t for t in assistant_texts if t]
    return assistant_texts[-1] if assistant_texts else ""


def run_halo_review(
    prompt: str,
    trace_path: str | Path,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
    temperature: float | None = None,
    telemetry: bool = False,
    use_responses_api: bool = False,
    disable_agent_tracing: bool = True,
) -> str:
    """Run HALO over ``trace_path`` and return its free-text report.

    This is the single seam that touches the HALO engine and a live model;
    tests replace it with a function returning a recorded report so neither
    ``halo-engine`` nor a model is needed in CI.

    HALO drives the OpenAI Agents SDK, which defaults to the **Responses API**.
    Databricks FMAPI serving endpoints are **chat-completions** (``llm/v1/chat``)
    and do not implement ``/responses``, so ``use_responses_api`` defaults to
    ``False`` and we flip the SDK's global default to ``chat_completions`` (the
    supported ``set_default_openai_api`` hook — HALO constructs its
    ``OpenAIProvider`` with no explicit ``use_responses``, so it honours this).
    ``disable_agent_tracing`` turns off the Agents SDK's own trace exporter
    (which would otherwise try to POST to OpenAI with the wrong credentials);
    our reviewer trace is the MLflow one opened by :func:`review_trace`, not the
    SDK's.
    """
    import agents
    from engine.main import run_engine
    from engine.models.messages import AgentMessage

    if disable_agent_tracing:
        agents.set_tracing_disabled(True)
    agents.set_default_openai_api("responses" if use_responses_api else "chat_completions")

    config = build_engine_config(
        model,
        base_url=base_url,
        api_key=api_key,
        max_turns=max_turns,
        max_depth=max_depth,
        max_parallel=max_parallel,
        temperature=temperature,
    )
    messages = [AgentMessage(role="user", content=prompt)]
    output_items = run_engine(messages, config, Path(trace_path), telemetry=telemetry)
    return _extract_report_text(output_items)


def _resolve_databricks_openai(profile: str | None) -> tuple[str | None, str | None]:
    """Resolve a Databricks FMAPI (OpenAI-compatible) base URL + bearer token.

    Builds ``<workspace host>/serving-endpoints`` and mints a token from the
    active Databricks CLI profile via the SDK. Best-effort: if the SDK is absent
    or the profile is unusable, returns ``(None, None)`` so the caller can fall
    back to ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` from the environment.
    """
    try:
        from databricks.sdk.core import Config
    except ImportError:  # pragma: no cover - databricks-sdk is a core dep
        return (None, None)
    try:
        cfg = Config(profile=profile) if profile else Config()
        host = cfg.host
        headers = cfg.authenticate() or {}
    except Exception:  # noqa: BLE001 - unusable profile: defer to ambient env auth
        return (None, None)
    auth = headers.get("Authorization", "")
    token = auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else None
    base_url = f"{host.rstrip('/')}/serving-endpoints" if host else None
    return (base_url, token)


def _configure_databricks(
    *,
    profile: str | None,
    tracking_uri: str,
    registry_uri: str,
    experiment_id: str | None,
) -> None:
    """Point MLflow at Databricks-managed MLflow + UC and set the reviewer experiment.

    Mirrors :mod:`ail.ingest.mlflow_source` / :mod:`ail.judges.registration`: an
    optional CLI profile selects the workspace, host is resolved best-effort, and
    the reviewer's own trace is logged to ``experiment_id`` when given.
    """
    import mlflow

    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
        if not os.environ.get("DATABRICKS_HOST"):
            try:
                from databricks.sdk import WorkspaceClient

                host = WorkspaceClient(profile=profile).config.host
            except Exception:  # noqa: BLE001 - unusable profile: defer to ambient auth
                host = None
            if host:
                os.environ["DATABRICKS_HOST"] = host

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)


@contextmanager
def _review_trace_context(attributes: dict[str, Any]) -> Iterator[str]:
    """Open HALO's *own* MLflow trace and yield its trace id.

    A fresh root span (the subject trace is a historical record, never the active
    one), so HALO's review is a separate, linkable trace — never nested in the
    agent's. Yields the reviewer trace id for back-linking.
    """
    import mlflow

    with mlflow.start_span(name=REVIEW_SPAN_NAME, span_type="AGENT", attributes=attributes) as span:
        yield span.trace_id


def _feedback_value(verdict: HaloReviewVerdict) -> int:
    """The feedback's headline ``value``: the 0–100 token-waste score.

    A single scalar by necessity — the Databricks v4 trace store only accepts a
    number / bool / string / list-of-strings as a feedback value (not a struct).
    The numeric waste score is the sortable headline; the categorical grade,
    counts, and the full structured verdict ride in :func:`_feedback_metadata`.
    """
    return verdict.token_waste_score


def _feedback_metadata(verdict: HaloReviewVerdict) -> dict[str, str]:
    """Assessment metadata: the back-link, provenance, headline grade, full verdict.

    All values are strings (the metadata map is ``dict[str, str]``). The full
    verdict round-trips as ``verdict_json``; the headline grade and counts are
    promoted to top-level keys for quick filtering without parsing that JSON.
    """
    metadata: dict[str, str] = {
        "schema_version": verdict.schema_version,
        "verdict_json": verdict.model_dump_json(),
        "token_efficiency": verdict.token_efficiency,
        "token_waste_score": str(verdict.token_waste_score),
        "n_redundancy_findings": str(len(verdict.redundancy_findings)),
        "n_failure_modes": str(len(verdict.failure_modes)),
    }
    if verdict.estimated_wasted_tokens is not None:
        metadata["estimated_wasted_tokens"] = str(verdict.estimated_wasted_tokens)
    if verdict.reviewer_trace_id:
        metadata["reviewer_trace_id"] = verdict.reviewer_trace_id
    if verdict.model:
        metadata["judge_model"] = verdict.model
    if verdict.parse_warnings:
        metadata["parse_warnings"] = "; ".join(verdict.parse_warnings)
    return metadata


def _attach_verdict(verdict: HaloReviewVerdict, *, source_id: str) -> None:
    """Attach ``verdict`` to the subject trace as an LLM-judge feedback assessment.

    The source type is ``LLM_JUDGE``: in ``mlflow>=3`` the older ``AI_JUDGE``
    spelling is a **deprecated alias** that is coerced to ``LLM_JUDGE`` (and emits
    a ``FutureWarning``), so we use the canonical name directly. This is the same
    source type the L2 judge assessments use (``docs/ARCHITECTURE.md`` §11); the
    distinct assessment *name* (``l3_halo_review``) keeps it from colliding with
    them on a shared subject trace.
    """
    import mlflow
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    mlflow.log_feedback(
        trace_id=verdict.subject_trace_id,
        name=FEEDBACK_NAME,
        value=_feedback_value(verdict),
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=source_id),
        rationale=verdict.summary or None,
        metadata=_feedback_metadata(verdict),
    )


def review_trace(
    trace_id: str,
    *,
    experiment_id: str | None = None,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    profile: str | None = None,
    reviewer_experiment_id: str | None = None,
    attach: bool = True,
    source: TraceSource | None = None,
    jsonl_path: str | Path | None = None,
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
    temperature: float | None = None,
    use_responses_api: bool = False,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> HaloReviewVerdict:
    """Review one trace with HALO and attach a structured verdict to it.

    Runs the full L3 flow: export → HALO review under its own trace → parse →
    attach. The HALO review is isolated as its own MLflow trace so its tokens are
    never summed into the subject trace's L0 cost (``docs/ARCHITECTURE.md`` §11).

    Args:
        trace_id: The subject trace to review.
        experiment_id: The experiment the subject trace lives in (recorded for
            context; not required to fetch the trace by id).
        model: Databricks serving-endpoint / FMAPI chat model name the HALO judge
            runs on (e.g. ``"databricks-claude-sonnet-4-6"``).
        base_url / api_key: OpenAI-compatible endpoint for HALO. When both are
            ``None`` they are resolved from the Databricks ``profile`` (workspace
            ``/serving-endpoints`` + a minted token); failing that, HALO falls
            back to the ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` env vars.
        profile: Databricks CLI profile selecting the workspace (auth + FMAPI).
        reviewer_experiment_id: Where to log HALO's own review trace. Defaults to
            ``experiment_id``.
        attach: When ``True`` (default), attach the verdict to the subject trace.
            ``False`` returns the verdict without writing — useful for dry runs.
        source: Trace source for the export. Defaults to a Databricks
            :class:`~ail.ingest.mlflow_source.MLflowTraceSource`; inject a fake in
            tests.
        jsonl_path: Where to write the HALO input JSONL. ``None`` uses a temp file.
        max_turns / max_depth / max_parallel / temperature: HALO bounds + sampling.
        use_responses_api: Leave ``False`` for Databricks FMAPI / any
            chat-completions endpoint (the default); set ``True`` only for an
            endpoint that implements the OpenAI Responses API.
        tracking_uri / registry_uri: MLflow backends (Databricks-managed + UC).

    Returns:
        The parsed :class:`~ail.l3.contract.HaloReviewVerdict`, with
        ``reviewer_trace_id`` set when a reviewer trace was opened.
    """
    if base_url is None and api_key is None:
        base_url, api_key = _resolve_databricks_openai(profile)

    _configure_databricks(
        profile=profile,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        experiment_id=reviewer_experiment_id or experiment_id,
    )

    export: OtlpExport = mlflow_trace_to_otlp_jsonl(
        trace_id,
        experiment_id,
        path=jsonl_path,
        source=source,
        profile=profile,
    )

    prompt = REVIEW_PROMPT_TEMPLATE.format(trace_id=export.trace_id)
    span_attributes: dict[str, Any] = {
        "ail.l3.subject_trace_id": export.trace_id,
        "ail.l3.subject_experiment_id": experiment_id or "",
        "ail.l3.judge_model": model,
        "ail.l3.subject_n_spans": export.n_spans,
    }

    with _review_trace_context(span_attributes) as reviewer_trace_id:
        report = run_halo_review(
            prompt,
            export.path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_turns=max_turns,
            max_depth=max_depth,
            max_parallel=max_parallel,
            temperature=temperature,
            use_responses_api=use_responses_api,
        )

    verdict = parse_halo_report(
        report,
        subject_trace_id=export.trace_id,
        reviewer_trace_id=reviewer_trace_id,
        model=model,
        generated_at=datetime.now(UTC).isoformat(),
    )

    if attach:
        _attach_verdict(verdict, source_id=model or "halo")

    return verdict
