"""Tests for the L3 reviewer (:mod:`ail.l3.reviewer`) and selection.

The reviewer's two external dependencies — the HALO engine and a live model, and
the MLflow assessment/trace APIs — are mocked so the orchestration is exercised
fully offline: ``run_halo_review`` returns a recorded report, the own-trace
context yields a fixed reviewer trace id, and ``mlflow.log_feedback`` is captured.
The pieces that genuinely need ``halo-engine`` (config build, the engine seam)
are ``importorskip``-guarded, and the end-to-end live review is
``@pytest.mark.live`` + env-gated.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from ail.ingest.base import NormalizedTrace, TraceSource
from ail.ingest.mlflow_source import normalize_trace
from ail.l3 import reviewer as rv
from ail.l3.contract import AssetRecommendation, GuidelineAssessment, HaloReviewVerdict
from ail.l3.rubric import DEFAULT_RUBRIC
from ail.l3.selection import select_traces_to_review

_REPORT = """\
The agent re-read the same file many times and worked from a vague task prompt.

```json
{
  "token_efficiency": "poor",
  "token_waste_score": 60,
  "estimated_wasted_tokens": 90000,
  "summary": "Repeated reads dominate the spend.",
  "guideline_assessments": [
    {"guideline_id": "tool_calling_efficiency", "score": 2,
     "rationale": "same file read 34x", "evidence_span_ids": ["s1", "s2"]},
    {"guideline_id": "token_efficiency", "score": 2, "rationale": "re-loaded context"},
    {"guideline_id": "tooling_purpose", "score": 4, "rationale": "mostly purposeful"},
    {"guideline_id": "instruction_clarity", "score": 3, "rationale": "ambiguous scope"}
  ],
  "recommended_assets": [
    {"asset_type": "skill", "title": "Cache file reads", "rationale": "repeated reads",
     "expected_benefit": "~90k tokens", "evidence_span_ids": ["s1"]},
    {"asset_type": "prompt_change", "title": "Clarify task scope",
     "rationale": "ambiguous prompt", "expected_benefit": "less rework"}
  ],
  "redundancy_findings": [
    {"description": "same file read 34x", "tool": "Read", "repeated_target": "/a",
     "occurrences": 34, "evidence_span_ids": ["s1"]}
  ],
  "failure_modes": [
    {"title": "re-read loop", "severity": "high", "description": "no caching"}
  ],
  "recommendations": ["cache reads"]
}
```
<final/>
"""


class _FakeSource(TraceSource):
    def __init__(self, trace: NormalizedTrace) -> None:
        self._trace = trace

    def iter_traces(self, **_: Any) -> Any:
        yield self._trace

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        return self._trace


@pytest.fixture
def captured_feedback(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Neutralize the workspace + HALO; capture every ``log_feedback`` call."""
    import mlflow

    monkeypatch.setattr(rv, "_configure_databricks", lambda **kw: None)
    monkeypatch.setattr(rv, "_resolve_databricks_openai", lambda profile: ("http://fmapi", "tok"))
    monkeypatch.setattr(rv, "run_halo_review", lambda *a, **k: _REPORT)

    @contextmanager
    def fake_trace_context(attributes: dict[str, Any]) -> Any:
        yield "rev-trace-xyz"

    monkeypatch.setattr(rv, "_review_trace_context", fake_trace_context)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mlflow,
        "log_feedback",
        lambda **kw: calls.append(kw) or SimpleNamespace(assessment_id="a1"),
    )
    return calls


class TestReviewTrace:
    def test_returns_parsed_verdict_with_reviewer_trace_id(
        self, synthetic_trace: Any, captured_feedback: list[dict[str, Any]]
    ) -> None:
        trace = normalize_trace(synthetic_trace)
        verdict = rv.review_trace(
            trace.trace_id,
            experiment_id="660599403165942",
            model="databricks-claude-sonnet-4-6",
            source=_FakeSource(trace),
        )
        assert isinstance(verdict, HaloReviewVerdict)
        assert verdict.subject_trace_id == trace.trace_id
        assert verdict.reviewer_trace_id == "rev-trace-xyz"
        assert verdict.token_efficiency == "poor"
        assert verdict.token_waste_score == 60
        assert verdict.model == "databricks-claude-sonnet-4-6"
        assert verdict.rubric_id == DEFAULT_RUBRIC.rubric_id
        # All four scored guidelines + the recommended assets parsed.
        assert {g.guideline_id for g in verdict.guideline_assessments} == set(
            DEFAULT_RUBRIC.guideline_ids()
        )
        assert verdict.score_for("tool_calling_efficiency") == 2
        assert [a.asset_type for a in verdict.recommended_assets] == ["skill", "prompt_change"]

    def test_attaches_per_guideline_assets_and_overall_feedback(
        self, synthetic_trace: Any, captured_feedback: list[dict[str, Any]]
    ) -> None:
        trace = normalize_trace(synthetic_trace)
        rv.review_trace(
            trace.trace_id,
            experiment_id="660599403165942",
            model="m",
            source=_FakeSource(trace),
        )
        by_name = {c["name"]: c for c in captured_feedback}
        # One assessment per scored guideline, plus assets, plus the overall.
        assert set(by_name) == {
            "rlm_tool_calling_efficiency",
            "rlm_token_efficiency",
            "rlm_tooling_purpose",
            "rlm_instruction_clarity",
            "rlm_recommended_assets",
            "rlm_review",
        }
        # Every assessment is an LLM_JUDGE on the subject trace, back-linked to the
        # reviewer's own trace (token isolation) — none nested in the subject.
        for call in captured_feedback:
            assert call["trace_id"] == trace.trace_id
            assert call["source"].source_type == "LLM_JUDGE"
            assert call["metadata"]["reviewer_trace_id"] == "rev-trace-xyz"
            assert call["metadata"]["rubric_id"] == DEFAULT_RUBRIC.rubric_id

        # Per-guideline value is the bounded score; evidence rides in metadata.
        tool_eff = by_name["rlm_tool_calling_efficiency"]
        assert tool_eff["value"] == 2
        assert tool_eff["metadata"]["guideline_id"] == "tool_calling_efficiency"
        assert tool_eff["metadata"]["evidence_span_ids"] == "s1, s2"

        # Assets: scalar count headline, assets JSON in metadata (v4 store needs scalars).
        assets = by_name["rlm_recommended_assets"]
        assert assets["value"] == 2
        parsed = json.loads(assets["metadata"]["recommended_assets_json"])
        assert [a["asset_type"] for a in parsed] == ["skill", "prompt_change"]
        assert assets["metadata"]["asset_types"] == "skill, prompt_change"

        # Overall: token-waste headline, full verdict + grade in metadata, summary rationale.
        overall = by_name["rlm_review"]
        assert overall["value"] == 60
        assert overall["rationale"] == "Repeated reads dominate the spend."
        assert overall["metadata"]["token_efficiency"] == "poor"
        assert overall["metadata"]["n_recommended_assets"] == "2"
        assert overall["metadata"]["n_guideline_assessments"] == "4"
        assert "verdict_json" in overall["metadata"]

    def test_recommend_assets_false_skips_assets_feedback(
        self, synthetic_trace: Any, captured_feedback: list[dict[str, Any]]
    ) -> None:
        from ail.l3.rubric import DEFAULT_GUIDELINES, ReviewRubric

        # A rubric that opts out of assets must not attach a spurious
        # rlm_recommended_assets=0 assessment.
        rubric = ReviewRubric(
            rubric_id="no-assets/v1", guidelines=DEFAULT_GUIDELINES, recommend_assets=False
        )
        trace = normalize_trace(synthetic_trace)
        rv.review_trace(trace.trace_id, model="m", source=_FakeSource(trace), rubric=rubric)
        names = {c["name"] for c in captured_feedback}
        assert "rlm_recommended_assets" not in names
        # The scored guidelines and the overall verdict are still attached.
        assert "rlm_review" in names
        assert "rlm_tool_calling_efficiency" in names

    def test_attach_false_skips_feedback(
        self, synthetic_trace: Any, captured_feedback: list[dict[str, Any]]
    ) -> None:
        trace = normalize_trace(synthetic_trace)
        verdict = rv.review_trace(
            trace.trace_id, model="m", source=_FakeSource(trace), attach=False
        )
        assert verdict.reviewer_trace_id == "rev-trace-xyz"
        assert captured_feedback == []

    def test_parse_failure_raises_and_does_not_attach(
        self,
        synthetic_trace: Any,
        captured_feedback: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A degenerate HALO report (e.g. terminated before emitting a verdict)
        # must surface as an error and never be recorded as a fake-good assessment.
        from ail.l3.parser import HaloReportParseError

        trace = normalize_trace(synthetic_trace)
        monkeypatch.setattr(rv, "run_halo_review", lambda *a, **k: "no JSON verdict here <final/>")
        with pytest.raises(HaloReportParseError):
            rv.review_trace(trace.trace_id, model="m", source=_FakeSource(trace))
        assert captured_feedback == []

    def test_explicit_endpoint_skips_databricks_resolution(
        self, synthetic_trace: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlflow

        trace = normalize_trace(synthetic_trace)
        monkeypatch.setattr(rv, "_configure_databricks", lambda **kw: None)
        monkeypatch.setattr(rv, "run_halo_review", lambda *a, **k: _REPORT)

        @contextmanager
        def ctx(attributes: dict[str, Any]) -> Any:
            yield "rev-1"

        monkeypatch.setattr(rv, "_review_trace_context", ctx)
        monkeypatch.setattr(mlflow, "log_feedback", lambda **kw: None)

        def boom(profile: str | None) -> Any:  # pragma: no cover - must not run
            raise AssertionError("should not resolve Databricks when endpoint is explicit")

        monkeypatch.setattr(rv, "_resolve_databricks_openai", boom)

        verdict = rv.review_trace(
            trace.trace_id,
            model="m",
            base_url="http://explicit",
            api_key="key",
            source=_FakeSource(trace),
            attach=False,
        )
        assert verdict.subject_trace_id == trace.trace_id

    def test_halo_seam_receives_prompt_and_path(
        self, synthetic_trace: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlflow

        trace = normalize_trace(synthetic_trace)
        seen: dict[str, Any] = {}

        def fake_runner(prompt: str, trace_path: Any, **kw: Any) -> str:
            seen["prompt"] = prompt
            seen["trace_path"] = trace_path
            seen["model"] = kw.get("model")
            seen["base_url"] = kw.get("base_url")
            return _REPORT

        monkeypatch.setattr(rv, "_configure_databricks", lambda **kw: None)
        monkeypatch.setattr(rv, "_resolve_databricks_openai", lambda p: ("http://fmapi", "tok"))
        monkeypatch.setattr(rv, "run_halo_review", fake_runner)

        @contextmanager
        def ctx(attributes: dict[str, Any]) -> Any:
            yield "rev-1"

        monkeypatch.setattr(rv, "_review_trace_context", ctx)
        monkeypatch.setattr(mlflow, "log_feedback", lambda **kw: None)

        rv.review_trace(trace.trace_id, model="m", source=_FakeSource(trace), attach=False)
        # The rubric-driven prompt carries the trace id and the guideline ids.
        assert trace.trace_id in seen["prompt"]
        assert "tool_calling_efficiency" in seen["prompt"]
        assert str(seen["trace_path"]).endswith(".jsonl")
        assert seen["model"] == "m"
        assert seen["base_url"] == "http://fmapi"


class TestBuildReviewPrompt:
    def test_sentinel_in_guideline_id_survives_substitution(self) -> None:
        from ail.l3.rubric import ReviewRubric, ScoredGuideline

        # A user-supplied guideline id that contains a framework sentinel literal
        # must render verbatim. The old chained order substituted <<IDS>> (user
        # content) BEFORE <<LO>>/<<HI>>, so a later replace re-scanned and
        # corrupted the id; the single-pass render never re-scans inserted text.
        rubric = ReviewRubric(
            rubric_id="hermetic/v1",
            guidelines=(
                ScoredGuideline(
                    id="sentinel_<<LO>>_<<HI>>_guard",
                    title="Sentinel-laden",
                    description="d",
                ),
            ),
            score_min=1,
            score_max=5,
        )
        prompt = rv.build_review_prompt("trace-1", rubric=rubric)

        # The id is emitted intact wherever it appears (the <<IDS>> schema slots
        # AND the numbered guideline block) ...
        assert "sentinel_<<LO>>_<<HI>>_guard" in prompt
        # ... and is NEVER corrupted by the later <<LO>>/<<HI>> score-scale
        # substitution (the bug: <<IDS>>-injected sentinels rewritten to 1/5).
        assert "sentinel_1_5_guard" not in prompt
        # The genuine score-scale sentinels still resolved, and the trace id too.
        assert "from 1 (worst) to 5 (best)" in prompt
        assert "trace-1" in prompt


class TestFeedbackProjection:
    def _verdict(self, **kw: Any) -> HaloReviewVerdict:
        base: dict[str, Any] = {
            "rubric_id": DEFAULT_RUBRIC.rubric_id,
            "subject_trace_id": "t1",
            "reviewer_trace_id": "rev-1",
            "model": "m",
            "token_efficiency": "poor",
            "token_waste_score": 60,
            "estimated_wasted_tokens": 90000,
            "summary": "s",
            "guideline_assessments": [
                GuidelineAssessment(
                    guideline_id="tool_calling_efficiency", score=2, evidence_span_ids=["s1", "s2"]
                )
            ],
            "recommended_assets": [
                AssetRecommendation(asset_type="skill", title="Cache reads"),
                AssetRecommendation(asset_type="metric_view", title="Dedup metric"),
            ],
        }
        base.update(kw)
        return HaloReviewVerdict(**base)

    def test_value_is_scalar_score(self) -> None:
        # The v4 trace store rejects struct values, so the headline value is the
        # single numeric waste score.
        assert rv._feedback_value(self._verdict()) == 60
        assert isinstance(rv._feedback_value(self._verdict()), int)

    def test_overall_metadata_carries_grade_counts_and_full_verdict(self) -> None:
        md = rv._overall_metadata(self._verdict())
        assert md["token_efficiency"] == "poor"
        assert md["token_waste_score"] == "60"
        assert md["estimated_wasted_tokens"] == "90000"
        assert md["n_guideline_assessments"] == "1"
        assert md["n_recommended_assets"] == "2"
        assert md["rubric_id"] == DEFAULT_RUBRIC.rubric_id
        assert md["reviewer_trace_id"] == "rev-1"
        assert md["judge_model"] == "m"
        assert '"subject_trace_id":"t1"' in md["verdict_json"].replace(" ", "")
        assert all(isinstance(v, str) for v in md.values())

    def test_overall_metadata_omits_none_estimate(self) -> None:
        md = rv._overall_metadata(self._verdict(estimated_wasted_tokens=None))
        assert "estimated_wasted_tokens" not in md

    def test_overall_metadata_records_parse_warnings(self) -> None:
        md = rv._overall_metadata(self._verdict(parse_warnings=["w1", "w2"]))
        assert md["parse_warnings"] == "w1; w2"

    def test_guideline_metadata(self) -> None:
        verdict = self._verdict()
        md = rv._guideline_metadata(verdict, verdict.guideline_assessments[0])
        assert md["guideline_id"] == "tool_calling_efficiency"
        assert md["score"] == "2"
        assert md["evidence_span_ids"] == "s1, s2"
        assert md["n_evidence_spans"] == "2"
        assert md["reviewer_trace_id"] == "rev-1"
        assert all(isinstance(v, str) for v in md.values())

    def test_assets_metadata(self) -> None:
        md = rv._assets_metadata(self._verdict())
        assert md["n_recommended_assets"] == "2"
        assert md["asset_types"] == "skill, metric_view"
        parsed = json.loads(md["recommended_assets_json"])
        assert [a["title"] for a in parsed] == ["Cache reads", "Dedup metric"]
        assert all(isinstance(v, str) for v in md.values())


class TestExtractReportText:
    def _item(self, role: str, content: str, final: bool) -> Any:
        return SimpleNamespace(final=final, item=SimpleNamespace(role=role, content=content))

    def test_prefers_final_messages(self) -> None:
        items = [
            self._item("assistant", "interim", False),
            self._item("assistant", "FINAL REPORT", True),
        ]
        assert rv._extract_report_text(items) == "FINAL REPORT"

    def test_falls_back_to_last_assistant(self) -> None:
        items = [
            self._item("assistant", "first", False),
            self._item("tool", "tool stuff", False),
            self._item("assistant", "last", False),
        ]
        assert rv._extract_report_text(items) == "last"

    def test_empty_when_no_text(self) -> None:
        assert rv._extract_report_text([]) == ""


class TestSelection:
    def _trace(self, tid: str, tokens: int, status: Any = None) -> NormalizedTrace:
        from ail.ingest.base import TokenUsage, TraceStatus

        return NormalizedTrace(
            trace_id=tid,
            status=status or TraceStatus.OK,
            token_usage=TokenUsage(input_tokens=tokens, output_tokens=0),
        )

    def test_ranks_by_tokens_desc_and_caps(self) -> None:
        traces = [self._trace("a", 10), self._trace("b", 100), self._trace("c", 50)]
        picked = select_traces_to_review(traces, top_n=2)
        assert [s.trace_id for s in picked] == ["b", "c"]

    def test_min_tokens_floor(self) -> None:
        traces = [self._trace("a", 10), self._trace("b", 100)]
        picked = select_traces_to_review(traces, min_tokens=50)
        assert [s.trace_id for s in picked] == ["b"]

    def test_filters_non_ok_by_default(self) -> None:
        from ail.ingest.base import TraceStatus

        traces = [self._trace("ok", 100), self._trace("err", 200, status=TraceStatus.ERROR)]
        picked = select_traces_to_review(traces)
        assert [s.trace_id for s in picked] == ["ok"]


# --- pieces that genuinely need halo-engine (skipped in CI) -----------------


class TestEngineSeam:
    def test_build_engine_config_uses_model_and_endpoint(self) -> None:
        pytest.importorskip("engine", reason="needs the l3 extra (halo-engine)")
        cfg = rv.build_engine_config("my-model", base_url="http://x", api_key="k", max_turns=7)
        assert cfg.root_agent.model.name == "my-model"
        assert cfg.synthesis_model.name == "my-model"
        assert cfg.model_provider.base_url == "http://x"
        assert cfg.root_agent.maximum_turns == 7

    def test_run_halo_review_wires_messages_to_engine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        pytest.importorskip("engine", reason="needs the l3 extra (halo-engine)")
        import engine.main as engine_main

        captured: dict[str, Any] = {}

        def fake_run_engine(messages: Any, config: Any, trace_path: Any, **kw: Any) -> list[Any]:
            captured["messages"] = messages
            captured["trace_path"] = trace_path
            return [
                SimpleNamespace(final=True, item=SimpleNamespace(role="assistant", content=_REPORT))
            ]

        monkeypatch.setattr(engine_main, "run_engine", fake_run_engine)
        jsonl = tmp_path / "t.jsonl"
        jsonl.write_text("{}\n")
        report = rv.run_halo_review("PROMPT", jsonl, model="m", base_url="http://x", api_key="k")
        assert "token_efficiency" in report
        assert captured["messages"][0].content == "PROMPT"


@pytest.mark.live
def test_live_review_trace() -> None:
    """End-to-end live review of one real trace (read-only: does not attach).

    Gated by ``AIL_LIVE_HALO=1`` plus ``AIL_LIVE_EXPERIMENT_ID`` /
    ``AIL_LIVE_TRACE_ID`` / ``AIL_LIVE_MODEL``, and requires ``halo-engine`` +
    a reachable FMAPI endpoint. ``attach=False`` so the live run never mutates
    the experiment.
    """
    if os.environ.get("AIL_LIVE_HALO") != "1":
        pytest.skip("set AIL_LIVE_HALO=1 to run the live HALO review")
    pytest.importorskip("engine", reason="live review needs halo-engine")
    experiment_id = os.environ.get("AIL_LIVE_EXPERIMENT_ID")
    trace_id = os.environ.get("AIL_LIVE_TRACE_ID")
    model = os.environ.get("AIL_LIVE_MODEL")
    if not (experiment_id and trace_id and model):
        pytest.skip("set AIL_LIVE_EXPERIMENT_ID, AIL_LIVE_TRACE_ID, AIL_LIVE_MODEL")

    verdict = rv.review_trace(
        trace_id,
        experiment_id=experiment_id,
        model=model,
        profile=os.environ.get("AIL_LIVE_PROFILE"),
        attach=False,
    )
    assert verdict.subject_trace_id
    assert verdict.token_efficiency in {"poor", "fair", "good", "excellent"}
