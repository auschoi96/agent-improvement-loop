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
   free-text ``<final/>`` report into a :class:`~ail.l3.contract.HaloReviewVerdict`
   against a configurable rubric (:mod:`ail.l3.rubric`).
4. **Attach** the verdict to the *subject* trace as a set of ``LLM_JUDGE`` feedback
   assessments (``mlflow.log_feedback``): one ``rlm_<guideline_id>`` per scored
   guideline, ``rlm_recommended_assets`` (assets JSON in metadata), and an overall
   ``rlm_review`` — each linked back to the reviewer's trace via
   ``metadata.reviewer_trace_id``. (``AI_JUDGE`` is a deprecated alias of
   ``LLM_JUDGE`` in ``mlflow>=3``.)

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

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.ingest.base import TraceSource
from ail.l3.adapter import OtlpExport, mlflow_trace_to_otlp_jsonl
from ail.l3.contract import AssetType, GuidelineAssessment, HaloReviewVerdict
from ail.l3.parser import parse_halo_report
from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric

if TYPE_CHECKING:
    from engine.engine_config import EngineConfig
    from engine.model_config import ReasoningEffort

__all__ = [
    "OVERALL_FEEDBACK_NAME",
    "ASSETS_FEEDBACK_NAME",
    "GUIDELINE_FEEDBACK_PREFIX",
    "guideline_feedback_name",
    "resolve_reasoning_effort",
    "normalize_reasoning_effort",
    "build_engine_config",
    "build_review_prompt",
    "run_halo_review",
    "review_trace",
]

#: Assessment name the **overall** verdict is attached under. Together with the
#: per-guideline ``rlm_<id>`` and the ``rlm_recommended_assets`` names — and the
#: ``LLM_JUDGE`` source type — this keys the L3 assessments on the subject trace
#: alongside (not colliding with) the L2 judge and ``HUMAN`` assessments.
OVERALL_FEEDBACK_NAME = "rlm_review"

#: Assessment name the recommended-assets list is attached under (assets JSON in
#: metadata; the headline value is the asset count).
ASSETS_FEEDBACK_NAME = "rlm_recommended_assets"

#: Prefix for the per-guideline assessment names, e.g. ``rlm_token_efficiency``.
GUIDELINE_FEEDBACK_PREFIX = "rlm_"


def guideline_feedback_name(guideline_id: str) -> str:
    """The assessment name a guideline's score is attached under (``rlm_<id>``)."""
    return f"{GUIDELINE_FEEDBACK_PREFIX}{guideline_id}"


#: Name of the reviewer's own MLflow trace/span.
REVIEW_SPAN_NAME = "rlm_review"

# Conservative HALO bounds. Depth 2 (root + one subagent layer) and a modest
# turn budget keep an L3 review from running away on a huge trace; raise them
# deliberately for a deeper review.
_DEFAULT_MAX_TURNS = 40
_DEFAULT_MAX_DEPTH = 2
_DEFAULT_MAX_PARALLEL = 4


# The review prompt is built from the rubric (HALO returns free text, so the
# structure is imposed here). Sentinel tokens — never f-string / ``.format`` —
# are substituted so the literal JSON braces below need no escaping.
_PROMPT_TEMPLATE = """\
You are an expert reviewer auditing a single long coding-agent execution trace. \
The subject trace id is `<<TRACE_ID>>`. It may be very large (hundreds of \
thousands of tokens) — use your trace-navigation tools to inspect it \
incrementally; never assume you must read it all at once.

OPERATING RULES — follow them exactly:
- Work AUTONOMOUSLY from your very first turn: call your trace-navigation tools \
yourself and keep going across as many turns as the evidence needs.
- NEVER ask for confirmation, permission, or clarification, and NEVER stop to \
check in. There is no human available to answer. A turn that ends WITHOUT a tool \
call and WITHOUT the final verdict ENDS the entire review and produces an \
unparseable report — so never end a turn idle.
- Ground EVERY score and EVERY recommendation in concrete evidence you actually \
observed in the trace, citing span ids. Give no generic advice: if the trace \
does not show it, do not claim it.
- The objective of this review is to <<OBJECTIVE>>. Judge each guideline, and \
make every recommendation, in service of that objective.

Evaluate the trace against the following guidelines. For each SCORED guideline, \
give an integer score from <<LO>> (worst) to <<HI>> (best) and a rationale that \
cites span-id evidence:

<<GUIDELINES_BLOCK>>

Only after you have actually inspected the trace with your tools, end your \
report with a SINGLE JSON object, on its own inside a ```json fenced block, with \
exactly these fields, followed by the marker <final/>:

```json
{
  "token_efficiency": "poor | fair | good | excellent",
  "token_waste_score": <integer 0-100, share of total spend that was avoidable>,
  "estimated_wasted_tokens": <integer or null>,
  "summary": "<2-4 sentence overall assessment>",
  "guideline_assessments": [
    {
      "guideline_id": "<one of: <<IDS>>>",
      "score": <integer <<LO>>-<<HI>>>,
      "rationale": "<why this score, citing span ids>",
      "evidence_span_ids": ["<span id>", "..."]
    }
  ],
<<ASSETS_SCHEMA>>  "redundancy_findings": [
    {
      "description": "<what was repeated>",
      "tool": "<tool name or null>",
      "repeated_target": "<file path / command / target or null>",
      "occurrences": <integer or null>,
      "estimated_wasted_tokens": <integer or null>,
      "evidence_span_ids": ["<span id>", "..."]
    }
  ],
  "failure_modes": [
    {
      "title": "<short title>",
      "severity": "low | medium | high",
      "description": "<what went wrong>",
      "evidence_span_ids": ["<span id>", "..."]
    }
  ],
  "recommendations": ["<concrete, actionable fix>", "..."]
}
```

Include exactly ONE entry in "guideline_assessments" for every guideline id \
listed above (<<IDS>>).\
"""

# The recommended-assets array, spliced into the schema only when the rubric asks
# for assets (guideline 5). Kept as its own fragment so a rubric with
# ``recommend_assets=False`` produces a prompt that neither asks for nor expects them.
_ASSETS_SCHEMA_FRAGMENT = """\
  "recommended_assets": [
    {
      "asset_type": "<one of: <<ASSET_TYPES>>>",
      "title": "<short asset name>",
      "rationale": "<what trace behaviour justifies it>",
      "expected_benefit": "<expected token / latency benefit>",
      "evidence_span_ids": ["<span id>", "..."],
      "trace_pattern": "<recurring pattern or null>"
    }
  ],
"""


def build_review_prompt(trace_id: str, rubric: ReviewRubric = DEFAULT_RUBRIC) -> str:
    """Render HALO's review prompt for ``trace_id`` from ``rubric``.

    The prompt fixes exactly what to look for and demands a single JSON object
    (matching :class:`~ail.l3.contract.HaloReviewVerdict`) before the ``<final/>``
    marker, which :mod:`ail.l3.parser` extracts and validates. The numbered
    guideline list and the JSON schema's guideline-id / score-range / asset-type
    hints are all derived from the rubric, so swapping the rubric swaps the
    review without touching this function.

    The prompt is deliberately crisp and unambiguous about autonomy: HALO must
    never pause to ask (a no-tool-call turn ends the run and yields an unparseable
    report — a prior bug), and every score and asset must be grounded in observed
    trace evidence (no generic advice).
    """
    lo, hi = str(rubric.score_min), str(rubric.score_max)
    lines = [
        f"{i}. {g.title} (`{g.id}`) — {g.description}"
        for i, g in enumerate(rubric.guidelines, start=1)
    ]
    asset_types = ", ".join(t for t in get_args(AssetType) if t != "other")
    if rubric.recommend_assets:
        n = len(rubric.guidelines) + 1
        lines.append(
            f"{n}. Recommended assets (`recommended_assets`) — Recommend concrete, "
            f"specific assets to build that would let the agent {rubric.objective}. "
            f"Allowed asset types: {asset_types}. Each asset must be justified by "
            "behaviour you observed in THIS trace (cite span ids, or describe the "
            "recurring pattern) and state its expected token / latency benefit. Do "
            "not invent assets the trace does not motivate."
        )
    guidelines_block = "\n".join(lines)
    # Resolve the assets fragment's own structural <<ASSET_TYPES>> here (framework
    # content only) so no sentinel survives inside any substituted *value*.
    assets_schema = (
        _ASSETS_SCHEMA_FRAGMENT.replace("<<ASSET_TYPES>>", asset_types)
        if rubric.recommend_assets
        else ""
    )

    # Single-pass substitution: one regex sweep replaces each sentinel exactly once
    # and never re-scans the text it inserts. So user-supplied content (guideline
    # ids, titles, objective) that happens to contain a sentinel literal is emitted
    # verbatim — it can't be re-scanned and corrupted by a later replacement,
    # regardless of substitution order. The rendering is hermetic to arbitrary
    # rubric content.
    substitutions = {
        "<<TRACE_ID>>": trace_id,
        "<<OBJECTIVE>>": rubric.objective,
        "<<LO>>": lo,
        "<<HI>>": hi,
        "<<GUIDELINES_BLOCK>>": guidelines_block,
        "<<IDS>>": " | ".join(rubric.guideline_ids()),
        "<<ASSETS_SCHEMA>>": assets_schema,
    }
    sentinel_re = re.compile(
        "|".join(re.escape(s) for s in sorted(substitutions, key=len, reverse=True))
    )
    return sentinel_re.sub(lambda m: substitutions[m.group(0)], _PROMPT_TEMPLATE)


def _normalize_model_for_reasoning(model: str) -> str:
    """Normalize a Databricks FMAPI model alias to the form HALO's effort check reads.

    HALO's :func:`engine.model_config.max_reasoning_effort_for_model` keys reasoning
    effort off *dotted* OpenAI family prefixes — ``gpt-5.4`` / ``gpt-5.5`` /
    ``gpt-5.1-codex-max`` map to ``xhigh``, other ``gpt-5`` / o-series map to
    ``high``. Databricks serving endpoints expose the same families under a
    provider-prefixed, *hyphenated* alias (``databricks-gpt-5-5-pro``) that matches
    none of those prefixes — it does not even start with ``gpt-5`` — so the strongest
    effort would silently never auto-apply. We (a) drop a leading ``databricks-``
    provider segment and (b) restore the dotted minor version on the gpt-5 family
    (``gpt-5-5`` → ``gpt-5.5``) so HALO's own table, not a hardcoded guess here,
    decides the effort.
    """
    n = model.strip().lower()
    if n.startswith("databricks-"):
        n = n[len("databricks-") :]
    # Restore the dotted minor version so a hyphenated gpt-5 alias matches HALO's
    # dotted family prefixes: gpt-5-5-pro → gpt-5.5-pro, gpt-5-1-codex-max →
    # gpt-5.1-codex-max. Only the first hyphen after ``gpt-5`` is a version dot.
    match = re.match(r"^(gpt-5)-(\d+)(.*)$", n)
    if match:
        n = f"{match.group(1)}.{match.group(2)}{match.group(3)}"
    return n


def resolve_reasoning_effort(model: str) -> str | None:
    """Reasoning effort HALO should send for ``model`` (honoring Databricks aliases).

    Delegates to HALO's authoritative
    :func:`~engine.model_config.max_reasoning_effort_for_model` after normalizing the
    Databricks alias (:func:`_normalize_model_for_reasoning`), so this stays in
    lockstep with HALO's family→effort table and never hardcodes an effort. Returns
    ``None`` for non-reasoning families (e.g. Claude), so the parameter is simply
    omitted for them.

    HALO's :class:`~engine.model_config.ModelConfig` is an EXTERNAL library we do not
    edit, and its prefix check misses the hyphenated Databricks alias — so the caller
    sets the resolved effort as an EXPLICIT ``ModelConfig.reasoning_effort`` override,
    which HALO honors ahead of its own auto-detection (see
    ``ModelConfig.effective_reasoning_effort``). Imports HALO lazily so importing this
    module needs no ``l3`` extra.
    """
    from engine.model_config import max_reasoning_effort_for_model

    return max_reasoning_effort_for_model(_normalize_model_for_reasoning(model))


#: Case-insensitive reasoning-effort inputs that mean "no explicit override — let the
#: resolver auto-detect from the model", NOT a literal HALO effort. ``none`` is included
#: deliberately: as an operator input it reads as "auto", so we normalize it to
#: auto-detect rather than injecting HALO's literal ``effort=none`` (which would DISABLE
#: reasoning — the opposite of what someone typing "none" for "no override" expects). A
#: caller that genuinely wants a level passes it explicitly (e.g. ``xhigh``).
_AUTO_EFFORT_SENTINELS = frozenset({"", "none", "auto"})


def normalize_reasoning_effort(value: str | None) -> str | None:
    """Map an operator-supplied reasoning-effort input to an explicit override or ``None``.

    Returns ``None`` — meaning "omit the override; auto-resolve from the model" — for
    ``None``, empty/whitespace-only, and the case-insensitive sentinels ``none`` /
    ``auto`` (:data:`_AUTO_EFFORT_SENTINELS`). Any other value is returned stripped as a
    genuine explicit override, so an unrecognized effort still reaches — and fails loud
    at — HALO's ``ReasoningEffort`` validation rather than silently passing.

    This closes a footgun: a user setting ``--reasoning-effort none`` (or the bundle var
    to ``none`` / ``auto`` / a quoted empty string) reads as "no override, auto-detect",
    and must NOT inject a literal effort into HALO.
    """
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in _AUTO_EFFORT_SENTINELS:
        return None
    return stripped


def build_engine_config(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> EngineConfig:
    """Build a HALO :class:`EngineConfig` from a model + OpenAI-compatible endpoint.

    A thin assembly over HALO's *public* config classes (the same shape the HALO
    CLI builds), using one :class:`ModelConfig` for the root agent, subagents,
    synthesis, and compaction. ``base_url`` / ``api_key`` are threaded onto the
    provider; when ``None`` HALO's underlying OpenAI client falls back to the
    ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` environment variables.

    ``reasoning_effort`` is set as an EXPLICIT override on the ``ModelConfig``. It is
    first passed through :func:`normalize_reasoning_effort`, so ``None`` / empty /
    ``none`` / ``auto`` all mean "no override" and fall through to
    :func:`resolve_reasoning_effort` — which normalizes Databricks FMAPI aliases so a
    reasoning family gets its documented max even when HALO's own hyphen-blind prefix
    check would not match (e.g. ``databricks-gpt-5-5-pro`` → ``xhigh``). A non-reasoning
    model resolves to ``None`` and receives no effort parameter, exactly as before. This
    defensive normalization means no caller can accidentally inject the ``none`` / ``auto``
    / empty literal as an effort; a genuine unrecognized value still fails loud at HALO's
    ``ReasoningEffort`` validation.

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

    normalized = normalize_reasoning_effort(reasoning_effort)
    effort = normalized if normalized is not None else resolve_reasoning_effort(model)
    model_cfg = ModelConfig(
        name=model,
        temperature=temperature,
        reasoning_effort=cast("ReasoningEffort | None", effort),
    )
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
    reasoning_effort: str | None = None,
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
        reasoning_effort=reasoning_effort,
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
    sql_warehouse_id: str | None = None,
) -> None:
    """Point MLflow at Databricks-managed MLflow + UC and set the reviewer experiment.

    Mirrors :mod:`ail.ingest.mlflow_source` / :mod:`ail.judges.registration`: an
    optional CLI profile selects the workspace, host is resolved best-effort, and
    the reviewer's own trace is logged to ``experiment_id`` when given.

    ``sql_warehouse_id`` is surfaced as :data:`~ail.compare.monitoring.TRACING_WAREHOUSE_ENV`
    (``MLFLOW_TRACING_SQL_WAREHOUSE_ID``) so an in-process trace read against the
    MLflow v4 (UC-backed) store has the warehouse it needs — the same plumbing
    :mod:`ail.publish` / :mod:`ail.compare.monitoring` use. The calling identity
    still needs ``CAN_USE`` on that warehouse.
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

    if sql_warehouse_id:
        os.environ[TRACING_WAREHOUSE_ENV] = sql_warehouse_id

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
    """The overall feedback's headline ``value``: the 0–100 token-waste score.

    A single scalar by necessity — the Databricks v4 trace store only accepts a
    number / bool / string / list-of-strings as a feedback value (not a struct).
    The numeric waste score is the sortable headline; the categorical grade,
    counts, and the full structured verdict ride in :func:`_overall_metadata`.
    """
    return verdict.token_waste_score


def _common_metadata(verdict: HaloReviewVerdict) -> dict[str, str]:
    """Provenance every L3 assessment shares: schema, rubric, back-link, judge model.

    ``reviewer_trace_id`` is the token-isolation back-link — it ties the
    assessment on the subject trace to HALO's *own* trace, where the reviewer's
    tokens are accounted, so the subject's L0 cost never absorbs the review's
    spend (``docs/ARCHITECTURE.md`` §11).
    """
    metadata: dict[str, str] = {
        "schema_version": verdict.schema_version,
        "rubric_id": verdict.rubric_id,
    }
    if verdict.reviewer_trace_id:
        metadata["reviewer_trace_id"] = verdict.reviewer_trace_id
    if verdict.model:
        metadata["judge_model"] = verdict.model
    return metadata


def _overall_metadata(verdict: HaloReviewVerdict) -> dict[str, str]:
    """Overall-assessment metadata: provenance, headline grade, counts, full verdict.

    All values are strings (the metadata map is ``dict[str, str]``). The full
    verdict round-trips as ``verdict_json``; the headline grade and counts are
    promoted to top-level keys for quick filtering without parsing that JSON.
    """
    metadata = _common_metadata(verdict)
    metadata.update(
        {
            "verdict_json": verdict.model_dump_json(),
            "token_efficiency": verdict.token_efficiency,
            "token_waste_score": str(verdict.token_waste_score),
            "n_guideline_assessments": str(len(verdict.guideline_assessments)),
            "n_recommended_assets": str(len(verdict.recommended_assets)),
            "n_redundancy_findings": str(len(verdict.redundancy_findings)),
            "n_failure_modes": str(len(verdict.failure_modes)),
        }
    )
    if verdict.estimated_wasted_tokens is not None:
        metadata["estimated_wasted_tokens"] = str(verdict.estimated_wasted_tokens)
    if verdict.parse_warnings:
        metadata["parse_warnings"] = "; ".join(verdict.parse_warnings)
    return metadata


def _guideline_metadata(
    verdict: HaloReviewVerdict, assessment: GuidelineAssessment
) -> dict[str, str]:
    """Per-guideline metadata: provenance + the guideline id, score, and evidence."""
    metadata = _common_metadata(verdict)
    metadata["guideline_id"] = assessment.guideline_id
    metadata["score"] = str(assessment.score)
    metadata["n_evidence_spans"] = str(len(assessment.evidence_span_ids))
    if assessment.evidence_span_ids:
        metadata["evidence_span_ids"] = ", ".join(assessment.evidence_span_ids)
    return metadata


def _assets_metadata(verdict: HaloReviewVerdict) -> dict[str, str]:
    """Recommended-assets metadata: provenance, count, type breakdown, the assets JSON."""
    metadata = _common_metadata(verdict)
    metadata["n_recommended_assets"] = str(len(verdict.recommended_assets))
    if verdict.recommended_assets:
        metadata["asset_types"] = ", ".join(a.asset_type for a in verdict.recommended_assets)
    metadata["recommended_assets_json"] = json.dumps(
        [a.model_dump() for a in verdict.recommended_assets]
    )
    return metadata


def _attach_verdict(
    verdict: HaloReviewVerdict, *, source_id: str, recommend_assets: bool = True
) -> None:
    """Attach ``verdict`` to the subject trace as a set of LLM-judge assessments.

    Writes one ``rlm_<guideline_id>`` feedback per scored guideline (value = the
    bounded score), one overall ``rlm_review`` feedback (value = the headline
    token-waste score; the full verdict in metadata), and — when
    ``recommend_assets`` (the rubric asked for assets) — one
    ``rlm_recommended_assets`` feedback (value = the asset count, ``0`` meaning
    "asked, found none"; the assets ride as JSON in metadata since the v4 store
    rejects struct values). A rubric that opted out of assets attaches no
    ``rlm_recommended_assets`` assessment at all.

    The source type is ``LLM_JUDGE``: in ``mlflow>=3`` the older ``AI_JUDGE``
    spelling is a **deprecated alias** coerced to ``LLM_JUDGE`` (and emits a
    ``FutureWarning``), so we use the canonical name directly. It is the same
    source type the L2 judge assessments use (``docs/ARCHITECTURE.md`` §11); the
    distinct assessment *names* keep these from colliding with the L2/``HUMAN``
    assessments on a shared subject trace.
    """
    import mlflow
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    source = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=source_id)

    def _log(name: str, value: int, rationale: str | None, metadata: dict[str, str]) -> None:
        mlflow.log_feedback(
            trace_id=verdict.subject_trace_id,
            name=name,
            value=value,
            source=source,
            rationale=rationale or None,
            metadata=metadata,
        )

    for assessment in verdict.guideline_assessments:
        _log(
            guideline_feedback_name(assessment.guideline_id),
            assessment.score,
            assessment.rationale,
            _guideline_metadata(verdict, assessment),
        )

    if recommend_assets:
        _log(
            ASSETS_FEEDBACK_NAME,
            len(verdict.recommended_assets),
            "; ".join(f"[{a.asset_type}] {a.title}" for a in verdict.recommended_assets) or None,
            _assets_metadata(verdict),
        )

    _log(
        OVERALL_FEEDBACK_NAME,
        _feedback_value(verdict),
        verdict.summary,
        _overall_metadata(verdict),
    )


def review_trace(
    trace_id: str,
    *,
    experiment_id: str | None = None,
    model: str,
    rubric: ReviewRubric = DEFAULT_RUBRIC,
    base_url: str | None = None,
    api_key: str | None = None,
    profile: str | None = None,
    sql_warehouse_id: str | None = None,
    reviewer_experiment_id: str | None = None,
    attach: bool = True,
    source: TraceSource | None = None,
    jsonl_path: str | Path | None = None,
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    use_responses_api: bool = False,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> HaloReviewVerdict:
    """Review one trace with HALO against ``rubric`` and attach the verdict to it.

    Runs the full L3 flow: export → HALO review under its own trace → parse →
    attach. The HALO review is isolated as its own MLflow trace so its tokens are
    never summed into the subject trace's L0 cost (``docs/ARCHITECTURE.md`` §11).
    On a degenerate report the parse raises :class:`~ail.l3.parser.HaloReportParseError`
    *before* the attach — so a broken review is never recorded as a fake-good
    assessment.

    Args:
        trace_id: The subject trace to review.
        experiment_id: The experiment the subject trace lives in (recorded for
            context; not required to fetch the trace by id).
        model: Databricks serving-endpoint / FMAPI chat model name the HALO judge
            runs on (e.g. ``"databricks-claude-sonnet-4-6"``).
        rubric: The review rubric (guidelines + score scale + asset directive).
            Defaults to :data:`ail.l3.rubric.DEFAULT_RUBRIC` (the user's five
            guidelines). Drives both the prompt and the parser's validation.
        base_url / api_key: OpenAI-compatible endpoint for HALO. When both are
            ``None`` they are resolved from the Databricks ``profile`` (workspace
            ``/serving-endpoints`` + a minted token); failing that, HALO falls
            back to the ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` env vars.
        profile: Databricks CLI profile selecting the workspace (auth + FMAPI).
        sql_warehouse_id: SQL warehouse the MLflow v4 (UC-backed) trace read uses,
            surfaced as ``MLFLOW_TRACING_SQL_WAREHOUSE_ID``. Needed when the
            backing store can only fetch traces through a warehouse.
        reviewer_experiment_id: Where to log HALO's own review trace. Defaults to
            ``experiment_id``.
        attach: When ``True`` (default), attach the assessments to the subject
            trace. ``False`` returns the verdict without writing — useful for dry
            runs.
        source: Trace source for the export. Defaults to a Databricks
            :class:`~ail.ingest.mlflow_source.MLflowTraceSource`; inject a fake in
            tests.
        jsonl_path: Where to write the HALO input JSONL. ``None`` uses a temp file.
        max_turns / max_depth / max_parallel / temperature: HALO bounds + sampling.
        reasoning_effort: Explicit HALO reasoning-effort override (one of HALO's
            ``ReasoningEffort`` levels). Leave ``None`` (default) to auto-resolve from
            ``model`` via :func:`resolve_reasoning_effort` — which lets a Databricks
            reasoning alias like ``databricks-gpt-5-5-pro`` still get ``xhigh`` despite
            HALO's hyphen-blind prefix check; non-reasoning models resolve to ``None``.
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
        sql_warehouse_id=sql_warehouse_id,
    )

    export: OtlpExport = mlflow_trace_to_otlp_jsonl(
        trace_id,
        experiment_id,
        path=jsonl_path,
        source=source,
        profile=profile,
    )

    prompt = build_review_prompt(export.trace_id, rubric)
    span_attributes: dict[str, Any] = {
        "ail.l3.subject_trace_id": export.trace_id,
        "ail.l3.subject_experiment_id": experiment_id or "",
        "ail.l3.judge_model": model,
        "ail.l3.rubric_id": rubric.rubric_id,
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
            reasoning_effort=reasoning_effort,
            use_responses_api=use_responses_api,
        )
        # Parse INSIDE the reviewer-trace context: a degenerate HALO report
        # raises HaloReportParseError, which (a) marks the reviewer's own trace
        # as errored — so the failed review is visible — and (b) propagates out
        # of review_trace before the attach below, so a broken review is NEVER
        # recorded as a (fake-good) assessment on the subject trace.
        verdict = parse_halo_report(
            report,
            subject_trace_id=export.trace_id,
            rubric=rubric,
            reviewer_trace_id=reviewer_trace_id,
            model=model,
            generated_at=datetime.now(UTC).isoformat(),
        )

    if attach:
        _attach_verdict(
            verdict, source_id=model or "halo", recommend_assets=rubric.recommend_assets
        )

    return verdict
