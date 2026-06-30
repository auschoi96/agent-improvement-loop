"""Tests for the HALO report → structured verdict parser (:mod:`ail.l3.parser`).

The parser must turn HALO's free-text ``<final/>`` report into a
:class:`~ail.l3.contract.HaloReviewVerdict`, imposing structure the engine never
guarantees. These tests cover the happy path (fenced and unfenced JSON), the
defensive degradations (missing/garbled JSON, out-of-range and synonym values),
and the invariant that the parser — not the model's JSON — owns the trace ids.
"""

from __future__ import annotations

import pytest

from ail.l3.parser import HaloReportParseError, parse_halo_report, strip_final_marker
from ail.l3.rubric import DEFAULT_RUBRIC

_GOOD_JSON = """\
Here is my analysis of the trace. The agent re-read the same file repeatedly.

```json
{
  "token_efficiency": "poor",
  "token_waste_score": 65,
  "estimated_wasted_tokens": 120000,
  "summary": "Large trace dominated by repeated reads of the same file.",
  "guideline_assessments": [
    {"guideline_id": "tool_calling_efficiency", "score": 2,
     "rationale": "Read /repo/main.py 34x", "evidence_span_ids": ["span-1", "span-2"]},
    {"guideline_id": "token_efficiency", "score": 2, "rationale": "Re-loaded context"},
    {"guideline_id": "tooling_purpose", "score": 4, "rationale": "Mostly purposeful"},
    {"guideline_id": "instruction_clarity", "score": 3,
     "rationale": "Task prompt was ambiguous about scope", "evidence_span_ids": ["span-3"]}
  ],
  "recommended_assets": [
    {"asset_type": "skill", "title": "Cache file reads",
     "rationale": "Same file read 34x", "expected_benefit": "~90k tokens saved",
     "evidence_span_ids": ["span-1"], "trace_pattern": "repeated identical Read"}
  ],
  "redundancy_findings": [
    {
      "description": "Same file read many times",
      "tool": "Read",
      "repeated_target": "/repo/main.py",
      "occurrences": 34,
      "estimated_wasted_tokens": 90000,
      "evidence_span_ids": ["span-1", "span-2"]
    }
  ],
  "failure_modes": [
    {
      "title": "Re-read loop",
      "severity": "high",
      "description": "The agent kept re-reading instead of caching context.",
      "evidence_span_ids": ["span-9"]
    }
  ],
  "recommendations": ["Cache file contents", "Batch edits"]
}
```
<final/>
"""


class TestStripFinalMarker:
    def test_removes_final_variants(self) -> None:
        assert strip_final_marker("report <final/>") == "report"
        assert strip_final_marker("report <final />") == "report"
        assert strip_final_marker("report </final>") == "report"

    def test_no_marker_is_noop(self) -> None:
        assert strip_final_marker("just a report") == "just a report"


class TestHappyPath:
    def test_parses_fenced_json(self) -> None:
        v = parse_halo_report(_GOOD_JSON, subject_trace_id="t1")
        assert v.token_efficiency == "poor"
        assert v.token_waste_score == 65
        assert v.estimated_wasted_tokens == 120000
        assert v.summary.startswith("Large trace")
        assert v.parse_warnings == []

    def test_parses_findings_and_failures(self) -> None:
        v = parse_halo_report(_GOOD_JSON, subject_trace_id="t1")
        assert len(v.redundancy_findings) == 1
        finding = v.redundancy_findings[0]
        assert finding.tool == "Read"
        assert finding.occurrences == 34
        assert finding.evidence_span_ids == ["span-1", "span-2"]
        assert len(v.failure_modes) == 1
        assert v.failure_modes[0].severity == "high"
        assert v.recommendations == ["Cache file contents", "Batch edits"]

    def test_parses_trailing_unfenced_object(self) -> None:
        report = (
            'Analysis done.\n{"token_efficiency": "good", "token_waste_score": 10, '
            '"summary": "Efficient."}\n<final/>'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.token_efficiency == "good"
        assert v.token_waste_score == 10
        # The unfenced object is found and parsed; the only warnings are the
        # rubric guidelines this minimal report left unscored (not a parse failure).
        assert all("guideline" in w for w in v.parse_warnings)

    def test_keeps_full_raw_report(self) -> None:
        v = parse_halo_report(_GOOD_JSON, subject_trace_id="t1")
        assert v.raw_report == _GOOD_JSON

    def test_records_rubric_id(self) -> None:
        v = parse_halo_report(_GOOD_JSON, subject_trace_id="t1")
        assert v.rubric_id == DEFAULT_RUBRIC.rubric_id


class TestGuidelinesAndAssets:
    """The v2 rubric fields: per-guideline scores (1–4) and recommended assets (5)."""

    def test_parses_all_four_scored_guidelines(self) -> None:
        v = parse_halo_report(_GOOD_JSON, subject_trace_id="t1")
        scored = {g.guideline_id: g for g in v.guideline_assessments}
        assert set(scored) == set(DEFAULT_RUBRIC.guideline_ids())
        assert scored["tool_calling_efficiency"].score == 2
        assert scored["tool_calling_efficiency"].evidence_span_ids == ["span-1", "span-2"]
        assert scored["instruction_clarity"].rationale.startswith("Task prompt")
        # All five guidelines accounted for: 4 scored + the recommended assets.
        assert v.score_for("tooling_purpose") == 4

    def test_parses_recommended_assets(self) -> None:
        v = parse_halo_report(_GOOD_JSON, subject_trace_id="t1")
        assert len(v.recommended_assets) == 1
        asset = v.recommended_assets[0]
        assert asset.asset_type == "skill"
        assert asset.title == "Cache file reads"
        assert asset.expected_benefit == "~90k tokens saved"
        assert asset.evidence_span_ids == ["span-1"]
        assert asset.trace_pattern == "repeated identical Read"

    def test_out_of_range_guideline_score_clamped_with_warning(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"guideline_assessments": [{"guideline_id": "token_efficiency", "score": 9, '
            '"rationale": "r"}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.score_for("token_efficiency") == 5  # clamped to score_max
        assert any("clamped" in w for w in v.parse_warnings)

    def test_unknown_guideline_id_dropped_with_warning(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"guideline_assessments": [{"guideline_id": "made_up", "score": 3}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.guideline_assessments == []
        assert any("unknown id" in w for w in v.parse_warnings)

    def test_unscorable_guideline_dropped_with_warning(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"guideline_assessments": [{"guideline_id": "tooling_purpose", "score": "n/a"}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.score_for("tooling_purpose") is None
        assert any("unparseable score" in w for w in v.parse_warnings)

    def test_missing_guideline_recorded_in_warnings(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"guideline_assessments": [{"guideline_id": "tooling_purpose", "score": 3}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        warning = next(w for w in v.parse_warnings if "no score for guideline" in w)
        assert "tool_calling_efficiency" in warning
        assert "tooling_purpose" not in warning  # it was scored

    def test_unknown_asset_type_recorded_as_other(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"recommended_assets": [{"asset_type": "spaceship", "title": "warp drive"}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.recommended_assets[0].asset_type == "other"
        assert any("unrecognized asset_type" in w for w in v.parse_warnings)

    def test_asset_type_synonym_mapped(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"recommended_assets": [{"asset_type": "metric view", "title": "m"}, '
            '{"asset_type": "instruction", "title": "p"}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert [a.asset_type for a in v.recommended_assets] == ["metric_view", "prompt_change"]

    def test_custom_rubric_ids_and_scale(self) -> None:
        from ail.l3.rubric import ReviewRubric, ScoredGuideline

        rubric = ReviewRubric(
            rubric_id="custom/v1",
            guidelines=(ScoredGuideline("clarity", "Clarity", "d"),),
            score_min=0,
            score_max=10,
        )
        report = (
            '```json\n{"token_efficiency": "good", "token_waste_score": 5, "summary": "x", '
            '"guideline_assessments": [{"guideline_id": "clarity", "score": 8}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1", rubric=rubric)
        assert v.rubric_id == "custom/v1"
        assert v.score_for("clarity") == 8  # in-range for the 0..10 scale, not clamped
        assert not any("clamped" in w for w in v.parse_warnings)


class TestParserOwnedFields:
    def test_parser_sets_trace_ids_and_model(self) -> None:
        v = parse_halo_report(
            _GOOD_JSON,
            subject_trace_id="subj-1",
            reviewer_trace_id="rev-1",
            model="databricks-claude-sonnet-4-6",
            generated_at="2026-06-29T00:00:00+00:00",
        )
        assert v.subject_trace_id == "subj-1"
        assert v.reviewer_trace_id == "rev-1"
        assert v.model == "databricks-claude-sonnet-4-6"
        assert v.generated_at == "2026-06-29T00:00:00+00:00"

    def test_model_json_in_payload_cannot_override_subject_id(self) -> None:
        # A model that injects subject_trace_id into its JSON must not win.
        report = (
            '```json\n{"subject_trace_id": "ATTACKER", "token_efficiency": "fair", '
            '"token_waste_score": 0, "summary": "x"}\n```<final/>'
        )
        v = parse_halo_report(report, subject_trace_id="real-subject")
        assert v.subject_trace_id == "real-subject"


class TestDefensiveDegradation:
    """Non-fatal normalizations: a real verdict with a real score, plus warnings."""

    def test_efficiency_synonyms_coerced(self) -> None:
        report = (
            '```json\n{"token_efficiency": "medium", "token_waste_score": 40, "summary": "x"}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.token_efficiency == "fair"
        assert any("token_efficiency" in w for w in v.parse_warnings)

    def test_unknown_efficiency_defaults_to_fair(self) -> None:
        report = (
            '```json\n{"token_efficiency": "stellar", "token_waste_score": 5, "summary": "x"}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.token_efficiency == "fair"
        assert v.parse_warnings

    def test_bad_severity_defaults_to_medium(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"failure_modes": [{"title": "t", "severity": "catastrophic", '
            '"description": "d"}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.failure_modes[0].severity == "medium"
        assert any("severity" in w for w in v.parse_warnings)

    def test_non_object_findings_dropped(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": 20, "summary": "x", '
            '"redundancy_findings": ["not an object", {"description": "real"}]}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert len(v.redundancy_findings) == 1
        assert v.redundancy_findings[0].description == "real"
        assert any("non-object" in w for w in v.parse_warnings)


class TestFailLoud:
    """A degenerate review must raise, never fabricate a (fake-good) verdict.

    ``token_waste_score=0`` is the *best* score, so silently defaulting a broken
    review to 0 would read as 'perfectly efficient' and poison the loop.
    """

    def test_missing_json_raises(self) -> None:
        # HALO terminating without a verdict (the no-tool-call-turn failure mode).
        with pytest.raises(HaloReportParseError):
            parse_halo_report("Just prose, no JSON here.<final/>", subject_trace_id="t1")

    def test_unparseable_score_raises(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": "lots", '
            '"summary": "x"}\n```'
        )
        with pytest.raises(HaloReportParseError):
            parse_halo_report(report, subject_trace_id="t1")

    def test_missing_score_key_raises(self) -> None:
        report = '```json\n{"token_efficiency": "fair", "summary": "x"}\n```'
        with pytest.raises(HaloReportParseError):
            parse_halo_report(report, subject_trace_id="t1")

    @pytest.mark.parametrize("bad", [150, 101, -1, -50])
    def test_out_of_range_score_raises(self, bad: int) -> None:
        report = (
            f'```json\n{{"token_efficiency": "poor", "token_waste_score": {bad}, '
            '"summary": "x"}\n```'
        )
        with pytest.raises(HaloReportParseError):
            parse_halo_report(report, subject_trace_id="t1")

    @pytest.mark.parametrize("ok", [0, 100, 42])
    def test_boundary_scores_accepted(self, ok: int) -> None:
        report = (
            f'```json\n{{"token_efficiency": "poor", "token_waste_score": {ok}, '
            '"summary": "x"}\n```'
        )
        assert parse_halo_report(report, subject_trace_id="t1").token_waste_score == ok
