"""Tests for the HALO report → structured verdict parser (:mod:`ail.l3.parser`).

The parser must turn HALO's free-text ``<final/>`` report into a
:class:`~ail.l3.contract.HaloReviewVerdict`, imposing structure the engine never
guarantees. These tests cover the happy path (fenced and unfenced JSON), the
defensive degradations (missing/garbled JSON, out-of-range and synonym values),
and the invariant that the parser — not the model's JSON — owns the trace ids.
"""

from __future__ import annotations

from ail.l3.parser import parse_halo_report, strip_final_marker

_GOOD_JSON = """\
Here is my analysis of the trace. The agent re-read the same file repeatedly.

```json
{
  "token_efficiency": "poor",
  "token_waste_score": 65,
  "estimated_wasted_tokens": 120000,
  "summary": "Large trace dominated by repeated reads of the same file.",
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
        assert v.parse_warnings == []

    def test_keeps_full_raw_report(self) -> None:
        v = parse_halo_report(_GOOD_JSON, subject_trace_id="t1")
        assert v.raw_report == _GOOD_JSON


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
    def test_missing_json_degrades_with_warning(self) -> None:
        v = parse_halo_report("Just prose, no JSON here.<final/>", subject_trace_id="t1")
        assert v.parse_warnings
        assert v.raw_report == "Just prose, no JSON here.<final/>"
        # Defaults are neutral, and the warning makes the degradation explicit.
        assert v.token_efficiency == "fair"
        assert v.token_waste_score == 0

    def test_out_of_range_score_clamped(self) -> None:
        report = (
            '```json\n{"token_efficiency": "poor", "token_waste_score": 150, "summary": "x"}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.token_waste_score == 100
        assert any("clamped" in w for w in v.parse_warnings)

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

    def test_unparseable_score_defaults_to_zero(self) -> None:
        report = (
            '```json\n{"token_efficiency": "fair", "token_waste_score": "lots", '
            '"summary": "x"}\n```'
        )
        v = parse_halo_report(report, subject_trace_id="t1")
        assert v.token_waste_score == 0
        assert any("token_waste_score" in w for w in v.parse_warnings)
