"""The L3 reviewer **output contract** — a typed, JSON-shaped structured verdict.

HALO returns a **free-text** report (terminated by ``<final/>``), not structured
data. The reviewer (:mod:`ail.l3.reviewer`) imposes structure by asking HALO to
end its report with a JSON object matching this schema, then
:mod:`ail.l3.parser` validates that object into a :class:`HaloReviewVerdict`.
This is the shape attached to the subject trace as an ``LLM_JUDGE`` feedback
assessment and read by a reviewer/leaderboard.

Like the L0 (:mod:`ail.metrics.contract`) and L2 (:mod:`ail.judges.contract`)
contracts, these are pydantic v2 models that forbid unknown fields so drift is
loud, and they round-trip through JSON without custom serialization.

The verdict is scoped to what an L3 *deep review* is for (``docs/ARCHITECTURE.md``
§3): discovering **token waste / avoidable redundancy**, rating
**quality-per-token**, and naming **failure modes** a fixed scorer would miss —
to decide *what to fix*, never to score the leaderboard.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Version of the L3 reviewer output contract. Bump the minor for additive,
#: backward-compatible fields; the major for breaking shape changes.
SCHEMA_VERSION = "l3.halo/v1"

#: Headline quality-per-token grade. Ordered worst→best; a consumer can map it
#: to an ordinal without re-encoding the verdict.
TokenEfficiency = Literal["poor", "fair", "good", "excellent"]

#: Severity of a discovered failure mode.
Severity = Literal["low", "medium", "high"]


class _Contract(BaseModel):
    """Base for every contract model: forbid unknown fields so drift is loud."""

    model_config = ConfigDict(extra="forbid")


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

    ``raw_report`` keeps HALO's full free-text report so nothing the reviewer
    said is lost to parsing, and ``parse_warnings`` records any degradation (a
    missing JSON block, an out-of-range value coerced) so a partial parse is
    never silently mistaken for a clean one.
    """

    schema_version: str = SCHEMA_VERSION
    subject_trace_id: str
    reviewer_trace_id: str | None = None
    model: str | None = None

    token_efficiency: TokenEfficiency
    token_waste_score: int = Field(ge=0, le=100)
    estimated_wasted_tokens: int | None = None

    summary: str
    redundancy_findings: list[RedundancyFinding] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    raw_report: str = ""
    parse_warnings: list[str] = Field(default_factory=list)
    generated_at: str | None = None  # ISO-8601
