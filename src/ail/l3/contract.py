"""The L3 reviewer **output contract** — a typed, JSON-shaped structured verdict.

HALO returns a **free-text** report (terminated by ``<final/>``), not structured
data. The reviewer (:mod:`ail.l3.reviewer`) imposes structure by asking HALO to
end its report with a JSON object matching this schema, then
:mod:`ail.l3.parser` validates that object into a :class:`HaloReviewVerdict`.
This is the shape attached to the subject trace as ``LLM_JUDGE`` feedback
assessments and read by a reviewer/leaderboard.

Like the L0 (:mod:`ail.metrics.contract`) and L2 (:mod:`ail.judges.contract`)
contracts, these are pydantic v2 models that forbid unknown fields so drift is
loud, and they round-trip through JSON without custom serialization.

The verdict is scoped to what an L3 *deep review* is for (``docs/ARCHITECTURE.md``
§3): discovering **token waste / avoidable redundancy**, rating
**quality-per-token**, and naming **failure modes** a fixed scorer would miss —
to decide *what to fix*, never to score the leaderboard. The v2 schema scores the
review against an explicit, configurable **rubric** (:mod:`ail.l3.rubric`): a
per-guideline {score, rationale, evidence} for each scored guideline, plus
concrete **recommended assets** (a metric view, tool, skill, semantic layer, data
pipeline, or prompt change) — each grounded in observed trace behaviour and aimed
at the rubric's standing objective: **the same task quality with fewer tokens and
lower latency**.

The cohort-level shapes (:class:`TraceReviewOutcome`, :class:`RankedAsset`,
:class:`CohortReviewReport`) are the aggregate :mod:`ail.l3.cohort_review`
produces over a tag-defined cohort of subject traces.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Version of the L3 reviewer output contract. Bump the minor for additive,
#: backward-compatible fields; the major for breaking shape changes. v2 adds the
#: rubric-driven ``guideline_assessments`` and ``recommended_assets`` (v1 carried
#: only the token-waste score, redundancy findings, and failure modes, which v2
#: keeps).
SCHEMA_VERSION = "l3.halo/v2"

#: Headline quality-per-token grade. Ordered worst→best; a consumer can map it
#: to an ordinal without re-encoding the verdict.
TokenEfficiency = Literal["poor", "fair", "good", "excellent"]

#: Severity of a discovered failure mode.
Severity = Literal["low", "medium", "high"]

#: The kinds of asset the reviewer may recommend (guideline 5). These mirror the
#: user's rubric vocabulary — a governed **metric view**, an agent **tool**, a
#: **skill**, a **semantic layer**, a **data pipeline**, or a **prompt/instruction
#: change** — plus ``other`` as the fail-soft bucket the parser coerces an
#: unrecognized type into (so a novel suggestion is never dropped, only labelled).
AssetType = Literal[
    "metric_view",
    "tool",
    "skill",
    "semantic_layer",
    "data_pipeline",
    "prompt_change",
    "other",
]

#: Outcome of reviewing one subject trace in a cohort batch. ``reviewed`` carries
#: a parsed verdict; ``review_failed`` is the **skipped** state — a degenerate or
#: unparseable HALO report (see :class:`ail.l3.parser.HaloReportParseError`) — and
#: is recorded as a skip, **never** as a fabricated (fake-good) pass.
TraceReviewStatus = Literal["reviewed", "review_failed"]


class _Contract(BaseModel):
    """Base for every contract model: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


class GuidelineAssessment(_Contract):
    """One scored rubric guideline's verdict: a bounded score with cited evidence.

    Covers rubric guidelines 1–4 (tool-calling efficiency, token efficiency,
    tooling purpose, instruction clarity). ``score`` is an integer on the
    rubric's scale (:attr:`ail.l3.rubric.ReviewRubric.score_min` ..
    :attr:`~ail.l3.rubric.ReviewRubric.score_max`, default 1=worst .. 5=best); the
    parser enforces that range, so the raw int here is always in-scale.
    ``evidence_span_ids`` points back into the subject trace so a human can verify
    the rationale rather than trust it — the same evidence discipline as
    :class:`RedundancyFinding` / :class:`FailureMode`.
    """

    guideline_id: str
    score: int
    rationale: str = ""
    evidence_span_ids: list[str] = Field(default_factory=list)


class AssetRecommendation(_Contract):
    """One concrete asset the reviewer recommends building (rubric guideline 5).

    Every recommendation must be **grounded in observed trace behaviour** (not
    generic advice) and aimed at the rubric's standing objective — the same task
    quality with fewer tokens / lower latency. ``expected_benefit`` states that
    benefit in token/latency terms; ``evidence_span_ids`` and/or ``trace_pattern``
    say *what in the trace* justifies it (span ids when the evidence is local, a
    described recurring pattern when it spans the whole trace).
    """

    asset_type: AssetType
    title: str
    rationale: str = ""
    expected_benefit: str = ""
    evidence_span_ids: list[str] = Field(default_factory=list)
    trace_pattern: str | None = None


class RedundancyFinding(_Contract):
    """One pattern of avoidable, repeated work the reviewer found in the trace.

    The canonical L3 example is "the same tool target hit N times" (e.g. a file
    read 34×, the same shell prologue re-run repeatedly). ``evidence_span_ids``
    points back into the subject trace so a human can verify the finding rather
    than trust it.
    """

    description: str
    tool: str | None = None
    repeated_target: str | None = None
    occurrences: int | None = None
    estimated_wasted_tokens: int | None = None
    evidence_span_ids: list[str] = Field(default_factory=list)


class FailureMode(_Contract):
    """A notable failure / quality problem the reviewer found.

    Distinct from redundancy: this is about the agent doing the *wrong* or
    *low-quality* thing (looping, abandoning a plan, ignoring an error,
    hallucinating a path), not merely repeating work.
    """

    title: str
    severity: Severity
    description: str
    evidence_span_ids: list[str] = Field(default_factory=list)


class HaloReviewVerdict(_Contract):
    """Structured verdict parsed from a HALO recursive trace review.

    ``token_waste_score`` is a 0–100 estimate of the share of the trace's spend
    that was *avoidable* (0 = nothing wasted, 100 = almost all wasted);
    ``token_efficiency`` is the headline quality-per-token grade. The two views
    coexist on purpose: a coarse grade for slicing and a finer score for ranking.
    These (with :attr:`redundancy_findings` and :attr:`failure_modes`) are the v1
    fields, kept verbatim — ``token_waste_score`` remains the required, sortable,
    un-gameable headline that the L0 optimization loop keys off.

    The v2 rubric fields layer on top: :attr:`guideline_assessments` carries one
    scored verdict per rubric guideline (1–4), and :attr:`recommended_assets`
    carries the concrete assets to build (guideline 5). ``rubric_id`` records
    which rubric produced the verdict so a consumer knows the scale and dimensions
    the scores are on.

    ``raw_report`` keeps HALO's full free-text report so nothing the reviewer
    said is lost to parsing, and ``parse_warnings`` records any degradation (a
    missing JSON block, an out-of-range value coerced) so a partial parse is
    never silently mistaken for a clean one.
    """

    schema_version: str = SCHEMA_VERSION
    rubric_id: str = ""
    subject_trace_id: str
    reviewer_trace_id: str | None = None
    model: str | None = None

    token_efficiency: TokenEfficiency
    token_waste_score: int = Field(ge=0, le=100)
    estimated_wasted_tokens: int | None = None

    summary: str
    guideline_assessments: list[GuidelineAssessment] = Field(default_factory=list)
    recommended_assets: list[AssetRecommendation] = Field(default_factory=list)
    redundancy_findings: list[RedundancyFinding] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    raw_report: str = ""
    parse_warnings: list[str] = Field(default_factory=list)
    generated_at: str | None = None  # ISO-8601

    def score_for(self, guideline_id: str) -> int | None:
        """The score recorded for ``guideline_id``, or ``None`` if not assessed."""
        return next(
            (g.score for g in self.guideline_assessments if g.guideline_id == guideline_id),
            None,
        )


# ---------------------------------------------------------------------------
# Cohort-level aggregate (``ail.l3.cohort_review``). A batch reviews each subject
# trace in a tag-defined cohort, attaches its per-guideline assessments, and rolls
# the recommended assets up into a deduped, recurrence-ranked report.
# ---------------------------------------------------------------------------


class TraceReviewOutcome(_Contract):
    """What happened when one cohort subject trace was reviewed.

    A ``reviewed`` outcome carries the parsed headline figures (and the
    reviewer's own trace id, for the token-isolation back-link); a
    ``review_failed`` outcome carries the ``error`` and **no** scores — a failed
    review is a skip, never a fabricated pass.
    """

    trace_id: str
    status: TraceReviewStatus
    reviewer_trace_id: str | None = None
    total_tokens: int | None = None
    token_efficiency: TokenEfficiency | None = None
    token_waste_score: int | None = None
    n_recommended_assets: int = 0
    error: str | None = None


class RankedAsset(_Contract):
    """One asset recommendation aggregated across a cohort, ranked by recurrence.

    An asset that recurs across **many** traces is the highest-value Phase-2
    target (the whole point of the cohort roll-up), so :attr:`n_traces` — the
    count of *distinct* subject traces that recommended this (type, title) — is
    the ranking key. :attr:`occurrences` is the total recommendation count (≥
    :attr:`n_traces`; a single trace could name it more than once).
    ``rationales`` / ``expected_benefits`` keep a sample of the per-trace
    justifications so the aggregate stays auditable.
    """

    asset_type: AssetType
    title: str
    rank: int
    n_traces: int
    occurrences: int
    trace_ids: list[str] = Field(default_factory=list)
    expected_benefits: list[str] = Field(default_factory=list)
    rationales: list[str] = Field(default_factory=list)
    evidence_span_ids: list[str] = Field(default_factory=list)


class CohortReviewReport(_Contract):
    """The aggregate of an L3 review run over a tag-defined cohort of traces.

    Carries per-trace :attr:`outcomes` (each ``reviewed`` or ``review_failed`` —
    fail-closed: a degenerate review is a recorded skip, never a fake pass) and
    the deduped, recurrence-ranked :attr:`ranked_assets` roll-up that names the
    highest-value Phase-2 targets. Provenance (cohort, location, tag filter, judge
    model, rubric) travels with it so no figure is opaque.
    """

    schema_version: str = SCHEMA_VERSION
    rubric_id: str = ""
    cohort_name: str
    location: str
    tag_filter: str = ""
    judge_model: str
    guideline_ids: list[str] = Field(default_factory=list)

    n_selected: int = 0
    n_reviewed: int = 0
    n_failed: int = 0

    outcomes: list[TraceReviewOutcome] = Field(default_factory=list)
    ranked_assets: list[RankedAsset] = Field(default_factory=list)

    generated_at: str | None = None  # ISO-8601
    notes: list[str] = Field(default_factory=list)
