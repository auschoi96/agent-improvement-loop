"""Tests for the cohort batch-runner (:mod:`ail.l3.cohort_review`).

Two layers, both fully offline:

* :func:`aggregate_assets` — a pure function over verdicts; tested directly for the
  dedupe + recurrence-ranking that names the highest-value Phase-2 assets.
* :func:`review_cohort` — the batch orchestration; tested against a fake trace
  source with HALO and the MLflow assessment APIs mocked, so it exercises cohort
  selection, per-trace review, fail-closed skips, and the asset roll-up without a
  live workspace, model, or ``halo-engine``.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from ail.cohorts import Cohort
from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.ingest.base import NormalizedTrace, TokenUsage, TraceSource, TraceStatus
from ail.l3 import reviewer as rv
from ail.l3.cohort_review import aggregate_assets, review_cohort
from ail.l3.contract import AssetRecommendation, HaloReviewVerdict
from ail.l3.rubric import DEFAULT_RUBRIC


def _verdict(trace_id: str, assets: list[AssetRecommendation]) -> HaloReviewVerdict:
    return HaloReviewVerdict(
        subject_trace_id=trace_id,
        token_efficiency="fair",
        token_waste_score=20,
        summary="s",
        recommended_assets=assets,
    )


def _asset(asset_type: str, title: str, **kw: Any) -> AssetRecommendation:
    return AssetRecommendation(asset_type=asset_type, title=title, **kw)  # type: ignore[arg-type]


class TestAggregateAssets:
    def test_dedupes_and_ranks_by_recurrence(self) -> None:
        verdicts = [
            _verdict(
                "tr-1",
                [
                    _asset("skill", "Cache reads", rationale="r1", evidence_span_ids=["s1"]),
                    _asset("tool", "Grep helper", rationale="g1"),
                ],
            ),
            # Same asset, different case/whitespace — must dedupe onto "Cache reads".
            _verdict("tr-2", [_asset("skill", "  cache   reads ", rationale="r2")]),
            _verdict("tr-3", [_asset("skill", "Cache Reads", rationale="r3")]),
        ]
        ranked = aggregate_assets(verdicts)

        top = ranked[0]
        assert top.rank == 1
        assert (top.asset_type, top.title) == ("skill", "Cache reads")  # first-seen casing
        assert top.n_traces == 3
        assert top.occurrences == 3
        assert set(top.trace_ids) == {"tr-1", "tr-2", "tr-3"}
        # Sample rationales are kept (deduped) for auditability.
        assert {"r1", "r2", "r3"} <= set(top.rationales)
        assert top.evidence_span_ids == ["s1"]

        # The single-trace asset ranks below the recurring one.
        rest = ranked[1:]
        assert all(r.n_traces == 1 for r in rest)
        assert {(r.asset_type, r.title) for r in rest} == {("tool", "Grep helper")}

    def test_occurrences_can_exceed_distinct_traces(self) -> None:
        # One trace naming the same asset twice: occurrences=2 but n_traces=1.
        verdicts = [_verdict("tr-1", [_asset("tool", "X"), _asset("tool", "x")])]
        ranked = aggregate_assets(verdicts)
        assert len(ranked) == 1
        assert ranked[0].n_traces == 1
        assert ranked[0].occurrences == 2

    def test_empty_when_no_assets(self) -> None:
        assert aggregate_assets([_verdict("tr-1", [])]) == []


# --- review_cohort: orchestration over a fake source -----------------------


class _FakeCohortSource(TraceSource):
    """Yields a fixed set of traces; ``filter_string`` is ignored (post-filter is truth)."""

    def __init__(self, traces: list[NormalizedTrace]) -> None:
        self._traces = traces
        self._by_id = {t.trace_id: t for t in traces}

    def iter_traces(self, **_: Any) -> Any:
        yield from self._traces

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        return self._by_id.get(trace_id)


def _tagged_trace(trace_id: str, tokens: int, *, agent: str) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=trace_id,
        status=TraceStatus.OK,
        token_usage=TokenUsage(input_tokens=tokens, output_tokens=0),
        tags={"ail.agent": agent},
    )


def _assets_report(*asset_objs: str, score: int = 30) -> str:
    body = ", ".join(asset_objs)
    return (
        "```json\n"
        f'{{"token_efficiency": "fair", "token_waste_score": {score}, "summary": "s", '
        f'"guideline_assessments": [], "recommended_assets": [{body}]}}\n'
        "```<final/>"
    )


# Per-trace HALO reports. tr-1 and tr-2 both recommend "Cache reads" (recurs);
# tr-3 is degenerate (no JSON verdict) and must be skipped, not faked.
_COHORT_REPORTS = {
    "tr-1": _assets_report(
        '{"asset_type": "skill", "title": "Cache reads", "expected_benefit": "~50k tokens"}',
        '{"asset_type": "tool", "title": "Grep helper"}',
    ),
    "tr-2": _assets_report(
        '{"asset_type": "skill", "title": "cache reads"}',
        '{"asset_type": "metric_view", "title": "Usage view"}',
    ),
    "tr-3": "HALO stopped to ask a question; no verdict. <final/>",
}


@pytest.fixture
def cohort_env(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Neutralize workspace/HALO; route per-trace reports by trace id; capture feedback."""
    import mlflow

    monkeypatch.setattr(rv, "_configure_databricks", lambda **kw: None)
    monkeypatch.setattr(rv, "_resolve_databricks_openai", lambda profile: ("http://fmapi", "tok"))

    def fake_runner(prompt: str, trace_path: Any, **kw: Any) -> str:
        for tid, report in _COHORT_REPORTS.items():
            if f"`{tid}`" in prompt:
                return report
        raise AssertionError(f"reviewed an unexpected trace; prompt head: {prompt[:80]!r}")

    monkeypatch.setattr(rv, "run_halo_review", fake_runner)

    @contextmanager
    def ctx(attributes: dict[str, Any]) -> Any:
        yield f"rev-{attributes.get('ail.l3.subject_trace_id', '?')}"

    monkeypatch.setattr(rv, "_review_trace_context", ctx)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mlflow,
        "log_feedback",
        lambda **kw: calls.append(kw) or SimpleNamespace(assessment_id="a"),
    )
    return calls


def _source() -> _FakeCohortSource:
    return _FakeCohortSource(
        [
            _tagged_trace("tr-1", 500_000, agent="claude_code"),
            _tagged_trace("tr-2", 600_000, agent="claude_code"),
            _tagged_trace("tr-3", 400_000, agent="claude_code"),
            _tagged_trace("tr-other", 900_000, agent="codex"),  # excluded by the cohort filter
        ]
    )


class TestReviewCohort:
    def test_tag_filter_selects_the_cohort(self, cohort_env: list[dict[str, Any]]) -> None:
        report = review_cohort(
            "exp-1",
            {"ail.agent": "claude_code"},
            judge_model="m",
            source=_source(),
        )
        reviewed_ids = {o.trace_id for o in report.outcomes}
        # The codex trace is excluded even though it is the largest by tokens.
        assert "tr-other" not in reviewed_ids
        assert reviewed_ids == {"tr-1", "tr-2", "tr-3"}
        assert report.n_selected == 3
        assert "ail.agent=claude_code" in report.tag_filter
        assert report.judge_model == "m"
        assert report.guideline_ids == list(DEFAULT_RUBRIC.guideline_ids())

    def test_failed_review_is_skipped_not_faked(self, cohort_env: list[dict[str, Any]]) -> None:
        report = review_cohort(
            "exp-1", {"ail.agent": "claude_code"}, judge_model="m", source=_source()
        )
        by_id = {o.trace_id: o for o in report.outcomes}
        assert by_id["tr-1"].status == "reviewed"
        assert by_id["tr-2"].status == "reviewed"
        # The degenerate report is recorded as review_failed — never a fake pass.
        assert by_id["tr-3"].status == "review_failed"
        assert by_id["tr-3"].token_waste_score is None
        assert "HaloReportParseError" in (by_id["tr-3"].error or "")
        assert report.n_reviewed == 2
        assert report.n_failed == 1
        # No assessment was attached for the failed trace.
        assert all(c["trace_id"] != "tr-3" for c in cohort_env)

    def test_aggregates_and_ranks_assets_across_traces(
        self, cohort_env: list[dict[str, Any]]
    ) -> None:
        report = review_cohort(
            "exp-1", {"ail.agent": "claude_code"}, judge_model="m", source=_source()
        )
        top = report.ranked_assets[0]
        # "Cache reads" recurs across tr-1 and tr-2 -> highest-value Phase-2 target.
        # (Display casing is whichever was reviewed first; recurrence is what ranks.)
        assert top.asset_type == "skill"
        assert top.title.lower() == "cache reads"
        assert top.n_traces == 2
        assert set(top.trace_ids) == {"tr-1", "tr-2"}
        assert top.rank == 1
        # Single-trace assets rank below; the failed trace contributes none.
        single = {(a.asset_type, a.title): a for a in report.ranked_assets[1:]}
        assert single.keys() == {("metric_view", "Usage view"), ("tool", "Grep helper")}
        assert all(a.n_traces == 1 for a in single.values())

    def test_accepts_cohort_object_and_sets_warehouse_env(
        self, cohort_env: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
        cohort = Cohort.by_agent("claude_code")
        report = review_cohort(
            "exp-1",
            cohort,
            judge_model="m",
            sql_warehouse_id="wh-123",
            source=_source(),
        )
        # The cohort's own name is preserved when a Cohort is passed directly.
        assert report.cohort_name == "claude_code"
        # The v4 trace-read warehouse is surfaced for the in-process read.
        import os

        assert os.environ[TRACING_WAREHOUSE_ENV] == "wh-123"

    def test_top_n_caps_the_review(self, cohort_env: list[dict[str, Any]]) -> None:
        # Only the single largest cohort trace (tr-2, 600k) is reviewed.
        report = review_cohort(
            "exp-1",
            {"ail.agent": "claude_code"},
            judge_model="m",
            source=_source(),
            top_n=1,
        )
        assert report.n_selected == 1
        assert [o.trace_id for o in report.outcomes] == ["tr-2"]

    def test_empty_cohort_is_a_clean_report_not_an_error(
        self, cohort_env: list[dict[str, Any]]
    ) -> None:
        report = review_cohort(
            "exp-1", {"ail.agent": "nonexistent"}, judge_model="m", source=_source()
        )
        assert report.n_selected == 0
        assert report.outcomes == []
        assert report.ranked_assets == []
        assert any("collecting" in n for n in report.notes)
