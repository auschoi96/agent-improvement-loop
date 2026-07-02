"""L3 — recursive trace deep-review (the adopted HALO path).

This package is the **L3 — RLM deep review** tier of the layered metrics design
(``docs/ARCHITECTURE.md`` §3, §11). It reads *whole* long traces — this corpus
reaches 943K tokens, far past any single judge call's context — to **discover**
token waste and failure modes a fixed scorer would miss, used to decide *what to
fix*, never to score the leaderboard.

It does this by **adopting HALO** (``halo-engine``, MIT — see ``PROVENANCE.md`` /
``NOTICE``), a trace-specialized Recursive LM that indexes a flat trace file by
byte offset and navigates it with bounded tools, recursive subagents, and
compaction. AIL contributes a *thin* layer around the engine, not a
reimplementation:

* :mod:`ail.l3.adapter` — MLflow trace → OpenInference/OTLP ``SpanRecord`` JSONL.
* :mod:`ail.l3.contract` — the structured verdict (:class:`HaloReviewVerdict`) and
  the cohort aggregate (:class:`CohortReviewReport`).
* :mod:`ail.l3.rubric` — the configurable review rubric (the user's five
  guidelines by default).
* :mod:`ail.l3.parser` — HALO's free-text ``<final/>`` report → verdict.
* :mod:`ail.l3.reviewer` — run HALO under its **own** trace (token isolation),
  parse against the rubric, and attach per-guideline / assets / overall
  ``LLM_JUDGE`` assessments to the subject trace, linked by ``reviewer_trace_id``.
* :mod:`ail.l3.selection` — choose which (biggest/most-interesting) traces to
  review, because L3 is expensive.
* :mod:`ail.l3.cohort_review` — review a tag-defined cohort and roll the
  recommended assets up into a deduped, recurrence-ranked report.

The HALO engine itself is the optional ``l3`` extra (``pip install 'ail[l3]'``)
and is lazy-imported, so importing this package never requires it.
"""

from __future__ import annotations

from ail.l3.adapter import (
    OtlpExport,
    mlflow_trace_to_otlp_jsonl,
    normalized_trace_to_span_records,
    write_span_records_jsonl,
)
from ail.l3.cohort_review import aggregate_assets, review_cohort
from ail.l3.contract import (
    SCHEMA_VERSION,
    AssetRecommendation,
    AssetType,
    CohortReviewReport,
    FailureMode,
    GuidelineAssessment,
    HaloReviewVerdict,
    RankedAsset,
    RedundancyFinding,
    TraceReviewOutcome,
    TraceReviewStatus,
)
from ail.l3.parser import HaloReportParseError, parse_halo_report, strip_final_marker
from ail.l3.reviewer import (
    ASSETS_FEEDBACK_NAME,
    GUIDELINE_FEEDBACK_PREFIX,
    OVERALL_FEEDBACK_NAME,
    build_engine_config,
    build_review_prompt,
    guideline_feedback_name,
    resolve_reasoning_effort,
    review_trace,
    run_halo_review,
)
from ail.l3.rubric import (
    DEFAULT_GUIDELINES,
    DEFAULT_OBJECTIVE,
    DEFAULT_RUBRIC,
    ReviewRubric,
    ScoredGuideline,
)
from ail.l3.selection import (
    TraceSelection,
    select_from_experiment,
    select_traces_to_review,
)

__all__ = [
    # contract
    "SCHEMA_VERSION",
    "HaloReviewVerdict",
    "GuidelineAssessment",
    "AssetRecommendation",
    "AssetType",
    "RedundancyFinding",
    "FailureMode",
    "TraceReviewOutcome",
    "TraceReviewStatus",
    "RankedAsset",
    "CohortReviewReport",
    # rubric
    "ReviewRubric",
    "ScoredGuideline",
    "DEFAULT_RUBRIC",
    "DEFAULT_GUIDELINES",
    "DEFAULT_OBJECTIVE",
    # adapter
    "OtlpExport",
    "normalized_trace_to_span_records",
    "write_span_records_jsonl",
    "mlflow_trace_to_otlp_jsonl",
    # parser
    "HaloReportParseError",
    "parse_halo_report",
    "strip_final_marker",
    # reviewer
    "OVERALL_FEEDBACK_NAME",
    "ASSETS_FEEDBACK_NAME",
    "GUIDELINE_FEEDBACK_PREFIX",
    "guideline_feedback_name",
    "resolve_reasoning_effort",
    "build_engine_config",
    "build_review_prompt",
    "run_halo_review",
    "review_trace",
    # selection
    "TraceSelection",
    "select_traces_to_review",
    "select_from_experiment",
    # cohort review
    "review_cohort",
    "aggregate_assets",
]
