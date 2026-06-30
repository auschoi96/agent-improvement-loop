"""Parse HALO's free-text ``<final/>`` report into a :class:`HaloReviewVerdict`.

HALO has no structured output schema: it returns a free-text report terminated
by a ``<final/>`` marker. The reviewer prompt (:mod:`ail.l3.reviewer`) asks HALO
to end that report with a single JSON object carrying the verdict's *content*
fields — the headline token-waste figures, the per-guideline scores, and the
recommended assets. This module extracts that object and validates it against the
rubric (:mod:`ail.l3.rubric`), filling the parser-owned fields (subject/reviewer
trace ids, model, timestamp, the full raw report) itself.

Parsing is **defensive and loud**: the optional content (guideline assessments,
recommended assets, redundancy findings, failure modes) degrades — warn + clamp
an out-of-range guideline score, drop an unscorable one, label an unknown asset
type ``other`` — while keeping the full ``raw_report`` and recording every
degradation in ``parse_warnings``, so a partial parse is never mistaken for a
clean one. Only the **required headline** is fail-closed: a report with no
parseable JSON block, or a missing / out-of-range ``token_waste_score``, raises
:class:`HaloReportParseError` rather than returning a fabricated default.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast, get_args

from ail.l3.contract import (
    AssetRecommendation,
    AssetType,
    FailureMode,
    GuidelineAssessment,
    HaloReviewVerdict,
    RedundancyFinding,
    Severity,
)
from ail.l3.rubric import DEFAULT_RUBRIC, ReviewRubric

__all__ = ["HaloReportParseError", "parse_halo_report", "strip_final_marker"]


class HaloReportParseError(ValueError):
    """Raised when a HALO report has no usable structured verdict.

    A degenerate report — HALO terminating on a no-tool-call turn without
    emitting the JSON verdict, an unparseable token-waste score, or a score
    outside 0–100 — **must fail loudly**. Silently returning a default verdict
    would record a broken review as a real one, and because
    ``token_waste_score=0`` is the *best* possible score, a swallowed failure
    would read as "this trace is perfectly efficient" — a fake-good signal that
    would poison the optimization loop. Fail closed instead.
    """


# HALO terminates its report with this marker; tolerate ``<final/>``,
# ``<final />`` and a stray closing ``</final>``.
_FINAL_RE = re.compile(r"</?\s*final\s*/?\s*>", re.IGNORECASE)

# Fenced code block (```json ... ``` or bare ``` ... ```), non-greedy.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.IGNORECASE | re.DOTALL)

_ALLOWED_EFFICIENCY: set[str] = {"poor", "fair", "good", "excellent"}
# Common synonyms a model emits → the contract's vocabulary.
_EFFICIENCY_SYNONYMS: dict[str, str] = {
    "low": "poor",
    "bad": "poor",
    "very poor": "poor",
    "medium": "fair",
    "moderate": "fair",
    "average": "fair",
    "ok": "fair",
    "high": "good",
    "very good": "excellent",
    "great": "excellent",
}
_ALLOWED_SEVERITY: set[str] = {"low", "medium", "high"}

# The contract's asset vocabulary, plus synonyms a model tends to emit for it.
# An unrecognized type is coerced to ``"other"`` (with a warning) so a novel
# suggestion is labelled, never dropped.
_ALLOWED_ASSET_TYPES: set[str] = set(get_args(AssetType))
_ASSET_TYPE_SYNONYMS: dict[str, str] = {
    "metric": "metric_view",
    "metricview": "metric_view",
    "view": "metric_view",
    "semanticlayer": "semantic_layer",
    "semantic": "semantic_layer",
    "pipeline": "data_pipeline",
    "datapipeline": "data_pipeline",
    "etl": "data_pipeline",
    "prompt": "prompt_change",
    "promptchange": "prompt_change",
    "instruction": "prompt_change",
    "instructions": "prompt_change",
    "instructionchange": "prompt_change",
    "systemprompt": "prompt_change",
}


def strip_final_marker(report: str) -> str:
    """Remove any ``<final/>`` / ``</final>`` markers and surrounding whitespace."""
    return _FINAL_RE.sub("", report).strip()


def _iter_json_candidates(text: str) -> list[str]:
    """Yield candidate JSON strings: fenced blocks first, then a trailing object.

    Fenced blocks are tried in reverse (a report tends to end with its verdict),
    then the last balanced ``{...}`` object in the text as a fallback for a model
    that emitted the JSON without fences.
    """
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    candidates.reverse()
    trailing = _last_brace_object(text)
    if trailing is not None:
        candidates.append(trailing)
    return candidates


def _last_brace_object(text: str) -> str | None:
    """Return the last balanced ``{...}`` substring, or ``None`` if there is none."""
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i : end + 1]
    return None


def _extract_payload(text: str) -> dict[str, Any] | None:
    """Find and decode the first candidate that parses to a JSON object, or ``None``."""
    for candidate in _iter_json_candidates(text):
        try:
            decoded = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def _coerce_efficiency(value: Any, warnings: list[str]) -> str:
    text = str(value).strip().lower()
    if text in _ALLOWED_EFFICIENCY:
        return text
    if text in _EFFICIENCY_SYNONYMS:
        mapped = _EFFICIENCY_SYNONYMS[text]
        warnings.append(f"mapped token_efficiency {value!r} -> {mapped!r}")
        return mapped
    warnings.append(f"unrecognized token_efficiency {value!r}; defaulted to 'fair'")
    return "fair"


def _coerce_score(value: Any) -> int:
    """Coerce the headline token-waste score to an int in ``0..100``, or fail loud.

    The score is the verdict's required headline signal, so — unlike the optional
    fields — it never degrades to a default. An unparseable value, or one outside
    ``0..100``, raises :class:`HaloReportParseError` rather than being silently
    coerced to ``0`` or clamped (clamping a wildly out-of-range score, e.g.
    ``150``, would mask a structural hallucination and the clamped value would
    still read as a real verdict). No tolerance: a well-behaved judge emits an
    integer in range, so anything else is a broken review.
    """
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise HaloReportParseError(
            f"token_waste_score is required and must be a number in 0-100; got {value!r}"
        ) from exc
    if score < 0 or score > 100:
        raise HaloReportParseError(f"token_waste_score {score} is outside the valid range 0-100")
    return score


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _opt_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _coerce_guideline_score(
    value: Any, rubric: ReviewRubric, guideline_id: str, warnings: list[str]
) -> int | None:
    """Coerce one guideline score to an int in the rubric range, or ``None`` if unparseable.

    Unlike the headline ``token_waste_score`` (the un-gameable signal the loop
    keys off, which fails loud), a per-guideline score is diagnostic detail: an
    out-of-range value is **clamped** into the rubric's scale with a warning, and
    only a non-numeric value degrades to ``None`` (the caller then drops that one
    assessment with a warning, rather than failing the whole verdict).
    """
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    clamped = rubric.clamp_score(score)
    if clamped != score:
        warnings.append(
            f"guideline {guideline_id!r} score {score} out of range "
            f"[{rubric.score_min}, {rubric.score_max}]; clamped to {clamped}"
        )
    return clamped


def _coerce_guideline_assessments(
    items: Any, rubric: ReviewRubric, warnings: list[str]
) -> list[GuidelineAssessment]:
    """Parse ``guideline_assessments`` against ``rubric``, warning on every gap.

    Keeps only assessments whose ``guideline_id`` is one the rubric asked for
    (unknown ids are dropped with a warning), de-duplicates repeats, and records a
    warning for any rubric guideline the report left unscored — so a partial set
    is never mistaken for a complete one.
    """
    out: list[GuidelineAssessment] = []
    valid_ids = set(rubric.guideline_ids())
    seen: set[str] = set()
    if not isinstance(items, list):
        if items not in (None, ""):
            warnings.append("guideline_assessments was not a list; ignored")
    else:
        for raw in items:
            if not isinstance(raw, dict):
                warnings.append("dropped a non-object guideline assessment")
                continue
            gid = str(raw.get("guideline_id", "")).strip()
            if gid not in valid_ids:
                warnings.append(f"dropped guideline assessment for unknown id {gid!r}")
                continue
            if gid in seen:
                warnings.append(f"dropped duplicate guideline assessment for {gid!r}")
                continue
            score = _coerce_guideline_score(raw.get("score"), rubric, gid, warnings)
            if score is None:
                warnings.append(
                    f"dropped guideline {gid!r}: missing or unparseable score {raw.get('score')!r}"
                )
                continue
            seen.add(gid)
            out.append(
                GuidelineAssessment(
                    guideline_id=gid,
                    score=score,
                    rationale=str(raw.get("rationale", "")).strip(),
                    evidence_span_ids=_str_list(raw.get("evidence_span_ids")),
                )
            )
    missing = [gid for gid in rubric.guideline_ids() if gid not in seen]
    if missing:
        warnings.append("no score for guideline(s): " + ", ".join(missing))
    return out


def _coerce_asset_type(value: Any, warnings: list[str]) -> AssetType:
    """Map a model's asset-type string onto the contract vocabulary (``other`` fallback)."""
    text = str(value).strip().lower()
    norm = text.replace("-", "_").replace(" ", "_")
    if norm in _ALLOWED_ASSET_TYPES:
        return cast(AssetType, norm)
    collapsed = norm.replace("_", "")
    if collapsed in _ASSET_TYPE_SYNONYMS:
        mapped = _ASSET_TYPE_SYNONYMS[collapsed]
        warnings.append(f"mapped asset_type {value!r} -> {mapped!r}")
        return cast(AssetType, mapped)
    warnings.append(f"unrecognized asset_type {value!r}; recorded as 'other'")
    return "other"


def _coerce_assets(items: Any, warnings: list[str]) -> list[AssetRecommendation]:
    """Parse ``recommended_assets`` defensively (drop non-objects, label unknown types)."""
    out: list[AssetRecommendation] = []
    if not isinstance(items, list):
        if items not in (None, ""):
            warnings.append("recommended_assets was not a list; ignored")
        return out
    for raw in items:
        if not isinstance(raw, dict):
            warnings.append("dropped a non-object recommended asset")
            continue
        out.append(
            AssetRecommendation(
                asset_type=_coerce_asset_type(raw.get("asset_type"), warnings),
                title=str(raw.get("title", "")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
                expected_benefit=str(raw.get("expected_benefit", "")).strip(),
                evidence_span_ids=_str_list(raw.get("evidence_span_ids")),
                trace_pattern=_opt_str(raw.get("trace_pattern")),
            )
        )
    return out


def _coerce_redundancy(items: Any, warnings: list[str]) -> list[RedundancyFinding]:
    out: list[RedundancyFinding] = []
    if not isinstance(items, list):
        if items not in (None, ""):
            warnings.append("redundancy_findings was not a list; ignored")
        return out
    for raw in items:
        if not isinstance(raw, dict):
            warnings.append("dropped a non-object redundancy finding")
            continue
        out.append(
            RedundancyFinding(
                description=str(raw.get("description", "")),
                tool=_opt_str(raw.get("tool")),
                repeated_target=_opt_str(raw.get("repeated_target")),
                occurrences=_opt_int(raw.get("occurrences")),
                estimated_wasted_tokens=_opt_int(raw.get("estimated_wasted_tokens")),
                evidence_span_ids=_str_list(raw.get("evidence_span_ids")),
            )
        )
    return out


def _coerce_failures(items: Any, warnings: list[str]) -> list[FailureMode]:
    out: list[FailureMode] = []
    if not isinstance(items, list):
        if items not in (None, ""):
            warnings.append("failure_modes was not a list; ignored")
        return out
    for raw in items:
        if not isinstance(raw, dict):
            warnings.append("dropped a non-object failure mode")
            continue
        sev = str(raw.get("severity", "medium")).strip().lower()
        if sev not in _ALLOWED_SEVERITY:
            warnings.append(f"unrecognized severity {raw.get('severity')!r}; defaulted to 'medium'")
            sev = "medium"
        out.append(
            FailureMode(
                title=str(raw.get("title", "")),
                severity=cast(Severity, sev),
                description=str(raw.get("description", "")),
                evidence_span_ids=_str_list(raw.get("evidence_span_ids")),
            )
        )
    return out


def parse_halo_report(
    report: str,
    *,
    subject_trace_id: str,
    rubric: ReviewRubric = DEFAULT_RUBRIC,
    reviewer_trace_id: str | None = None,
    model: str | None = None,
    generated_at: str | None = None,
) -> HaloReviewVerdict:
    """Parse a HALO free-text report into a structured :class:`HaloReviewVerdict`.

    Args:
        report: HALO's full report text (with or without the ``<final/>`` marker).
        subject_trace_id: The trace HALO reviewed (parser-owned, never trusted to
            the model's JSON).
        rubric: The rubric the review was run against — sets which guideline ids
            to expect and the valid score range. Defaults to
            :data:`ail.l3.rubric.DEFAULT_RUBRIC`.
        reviewer_trace_id: HALO's own review trace id, for back-linking.
        model: The judge model HALO ran on.
        generated_at: ISO-8601 timestamp to stamp on the verdict.

    Returns:
        A :class:`HaloReviewVerdict` built from the JSON block HALO emitted.

    Raises:
        HaloReportParseError: If the report has no parseable JSON verdict block,
            or its required ``token_waste_score`` is missing, unparseable, or
            outside ``0..100``. A degenerate review must fail loudly, never
            return a fabricated default (see :class:`HaloReportParseError`). The
            per-guideline scores and recommended assets are diagnostic detail and
            degrade defensively (warn + clamp/drop) rather than raising — only the
            un-gameable headline fails the whole parse.
    """
    warnings: list[str] = []
    body = strip_final_marker(report)
    payload = _extract_payload(body)

    if payload is None:
        raise HaloReportParseError(
            "HALO report contained no parseable JSON verdict block "
            "(the review likely terminated before producing a verdict)"
        )

    # The required headline score is coerced first so a malformed score fails the
    # whole parse loudly before any partial verdict is constructed.
    token_waste_score = _coerce_score(payload.get("token_waste_score"))

    return HaloReviewVerdict(
        rubric_id=rubric.rubric_id,
        subject_trace_id=subject_trace_id,
        reviewer_trace_id=reviewer_trace_id,
        model=model,
        token_efficiency=_coerce_efficiency(  # type: ignore[arg-type]
            payload.get("token_efficiency", "fair"), warnings
        ),
        token_waste_score=token_waste_score,
        estimated_wasted_tokens=_opt_int(payload.get("estimated_wasted_tokens")),
        summary=str(payload.get("summary", "")).strip(),
        guideline_assessments=_coerce_guideline_assessments(
            payload.get("guideline_assessments"), rubric, warnings
        ),
        recommended_assets=_coerce_assets(payload.get("recommended_assets"), warnings),
        redundancy_findings=_coerce_redundancy(payload.get("redundancy_findings"), warnings),
        failure_modes=_coerce_failures(payload.get("failure_modes"), warnings),
        recommendations=_str_list(payload.get("recommendations")),
        raw_report=report,
        parse_warnings=warnings,
        generated_at=generated_at,
    )
