"""Offline tests for the arrival-triggered continuous RLM runner."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

import ail.l3.continuous as cr
from ail.compare.monitoring import TRACING_WAREHOUSE_ENV
from ail.ingest.base import NormalizedTrace, TokenUsage, TraceSource, TraceStatus
from ail.l3.contract import HaloReviewVerdict


class _FakeSource(TraceSource):
    def __init__(self, traces: list[NormalizedTrace]) -> None:
        self._traces = traces
        self._by_id = {t.trace_id: t for t in traces}
        self.calls: list[dict[str, Any]] = []

    def iter_traces(
        self,
        *,
        experiment_id: str,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
    ) -> Any:
        self.calls.append(
            {
                "experiment_id": experiment_id,
                "filter_string": filter_string,
                "max_results": max_results,
                "order_by": order_by,
            }
        )
        yield from self._traces[:max_results]

    def get_trace(self, trace_id: str) -> NormalizedTrace | None:
        return self._by_id.get(trace_id)


def _assessment(name: str) -> Any:
    return SimpleNamespace(name=name)


def _trace(
    trace_id: str,
    tokens: int,
    *,
    status: TraceStatus = TraceStatus.OK,
    assessments: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> NormalizedTrace:
    raw = SimpleNamespace(info=SimpleNamespace(assessments=list(assessments or [])))
    return NormalizedTrace(
        trace_id=trace_id,
        status=status,
        token_usage=TokenUsage(input_tokens=tokens),
        metadata=dict(metadata or {}),
        raw=raw,
    )


def _reviewer_trace(trace_id: str, subject_trace_id: str, tokens: int = 900) -> NormalizedTrace:
    """A HALO reviewer's own token-isolated trace, tagged with its subject id.

    ``ail.l3.reviewer`` writes ``ail.l3.subject_trace_id`` metadata on the trace it
    emits for each review; those reviewer traces share the subject experiment, so the
    continuous runner must never re-review them.
    """
    return _trace(trace_id, tokens, metadata={"ail.l3.subject_trace_id": subject_trace_id})


def _verdict(trace_id: str) -> HaloReviewVerdict:
    return HaloReviewVerdict(
        subject_trace_id=trace_id,
        reviewer_trace_id=f"review-{trace_id}",
        model="judge",
        token_efficiency="good",
        token_waste_score=10,
        summary="reviewed",
    )


class TestContinuousSelection:
    def test_has_rlm_assessment_detects_existing_feedback(self) -> None:
        assert cr.has_rlm_assessment(_trace("t1", 1, assessments=[_assessment("rlm_review")]))
        assert cr.has_rlm_assessment(
            _trace("t2", 1, assessments=[_assessment("rlm_token_efficiency")])
        )
        assert not cr.has_rlm_assessment(_trace("t3", 1, assessments=[_assessment("correctness")]))

    def test_failure_marker_is_not_a_successful_rlm_assessment(self) -> None:
        assert cr.REVIEW_FAILED_FEEDBACK_NAME == "rlm_review_failed"
        trace = _trace("t4", 1, assessments=[_assessment(cr.REVIEW_FAILED_FEEDBACK_NAME)])
        assert cr.has_rlm_assessment(trace) is False
        assert cr.has_rlm_failure_marker(trace) is True

    def test_select_skips_reviewed_reviewer_traces_and_ranks_by_tokens(self) -> None:
        traces = [
            _trace("already", 900, assessments=[_assessment("rlm_review")]),
            _reviewer_trace("halo-run", subject_trace_id="already", tokens=950),
            _trace("small", 100),
            _trace("big", 800),
            _trace("medium", 500),
            _trace("errored", 1_000, status=TraceStatus.ERROR),
        ]
        selections, n_already, n_reviewer_skipped, n_sampled_out = cr.select_unreviewed_traces(
            traces,
            max_reviews=2,
            sample_rate=1.0,
            min_tokens=200,
        )

        assert n_already == 1
        assert n_reviewer_skipped == 1
        assert n_sampled_out == 0
        assert [s.trace_id for s in selections] == ["errored", "big"]

    def test_sampling_is_deterministic_and_can_sample_out_all(self) -> None:
        assert cr.sample_trace_id("trace-a", 0.0) is False
        assert cr.sample_trace_id("trace-a", 1.0) is True
        assert cr.sample_trace_id("trace-a", 0.37) is cr.sample_trace_id("trace-a", 0.37)

        selections, n_already, n_reviewer_skipped, n_sampled_out = cr.select_unreviewed_traces(
            [_trace("a", 100), _trace("b", 200)],
            max_reviews=2,
            sample_rate=0.0,
        )
        assert selections == []
        assert n_already == 0
        assert n_reviewer_skipped == 0
        assert n_sampled_out == 2

    def test_drain_exclusion_prevents_same_run_reselection(self) -> None:
        selections, *_ = cr.select_unreviewed_traces(
            [_trace("attempted", 1_000), _trace("next", 500)],
            max_reviews=2,
            sample_rate=1.0,
            min_tokens=0,
            retry_failed=True,
            exclude_trace_ids={"attempted"},
        )

        assert [selection.trace_id for selection in selections] == ["next"]


class TestContinuousRun:
    def test_prepares_optional_sandbox_once_before_parallel_reviews(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        class UnavailableSandbox:
            @classmethod
            def get(cls) -> None:
                nonlocal calls
                calls += 1
                return None

        monkeypatch.setattr(cr, "_halo_sandbox_class", lambda: UnavailableSandbox)

        cr._prepare_halo_sandbox()
        cr._prepare_halo_sandbox()

        assert calls == 1

    def test_can_disable_optional_sandbox_without_probing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        class Sandbox:
            @classmethod
            def get(cls) -> object:
                nonlocal calls
                calls += 1
                return object()

        monkeypatch.setattr(cr, "_halo_sandbox_class", lambda: Sandbox)

        cr._disable_halo_sandbox()

        assert Sandbox.get() is None
        assert calls == 0

    def test_run_reviews_only_selected_unreviewed_traces(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _FakeSource(
            [
                _trace("already", 900, assessments=[_assessment("rlm_review")]),
                _reviewer_trace("halo-run", subject_trace_id="already", tokens=950),
                _trace("big", 800),
                _trace("medium", 500),
                _trace("small", 100),
            ]
        )
        reviewed: list[str] = []

        def fake_review_trace(trace_id: str, **_: Any) -> HaloReviewVerdict:
            reviewed.append(trace_id)
            return _verdict(trace_id)

        monkeypatch.setattr(cr, "review_trace", fake_review_trace)
        disabled: list[bool] = []
        monkeypatch.setattr(cr, "_disable_halo_sandbox", lambda: disabled.append(True))
        report = cr.run_continuous_rlm(
            "exp-1",
            judge_model="judge",
            sql_warehouse_id="wh",
            source=source,
            max_results=10,
            max_reviews=2,
            sample_rate=1.0,
            min_tokens=200,
            enable_code_sandbox=False,
        )

        assert source.calls == [
            {
                "experiment_id": "exp-1",
                "filter_string": None,
                "max_results": 10,
                "order_by": ["timestamp_ms DESC"],
            }
        ]
        assert reviewed == ["big", "medium"]
        assert disabled == [True]
        assert report.n_scanned == 5
        assert report.n_already_reviewed == 1
        assert report.n_reviewer_traces_skipped == 1
        assert report.n_selected == 2
        assert report.n_reviewed == 2
        assert report.n_failed == 0
        assert [o.trace_id for o in report.outcomes] == ["big", "medium"]
        assert report.outcomes[0].reviewer_trace_id == "review-big"

    def test_run_records_failed_review_without_fabricating_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _FakeSource([_trace("bad", 1_000)])

        def fail_review_trace(trace_id: str, **_: Any) -> HaloReviewVerdict:
            raise RuntimeError("degenerate HALO output")

        marked: list[dict[str, Any]] = []

        def fake_mark_failed(trace_id: str, *, error: str, judge_model: str) -> None:
            marked.append({"trace_id": trace_id, "error": error, "judge_model": judge_model})

        monkeypatch.setattr(cr, "review_trace", fail_review_trace)
        monkeypatch.setattr(cr, "_mark_review_failed", fake_mark_failed)
        monkeypatch.setattr(cr, "_prepare_halo_sandbox", lambda: None)
        report = cr.run_continuous_rlm(
            "exp-1",
            judge_model="judge",
            source=source,
            max_reviews=1,
            sample_rate=1.0,
            min_tokens=0,
        )

        assert report.n_reviewed == 0
        assert report.n_failed == 1
        outcome = report.outcomes[0]
        assert outcome.status == "review_failed"
        assert outcome.token_waste_score is None
        # The verdict fields stay empty (fail-closed: no fabricated score) and the
        # error text is unchanged — the honest failure marker was attached out of band.
        assert "RuntimeError: degenerate HALO output" == outcome.error
        assert marked == [
            {
                "trace_id": "bad",
                "error": "RuntimeError: degenerate HALO output",
                "judge_model": "judge",
            }
        ]

    def test_failed_trace_is_not_reviewed_again_on_next_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A review that failed once must never be re-selected (cost runaway guard).

        Proves fail-without / pass-with: run 1 selects and fails the trace; the
        honest ``rlm_review_failed`` marker lands on it; run 2 skips it as
        already-handled instead of burning another HALO review.
        """
        failing = _trace("bad", 1_000)
        source = _FakeSource([failing])

        reviewed: list[str] = []

        def fail_review_trace(trace_id: str, **_: Any) -> HaloReviewVerdict:
            reviewed.append(trace_id)
            raise RuntimeError("degenerate HALO output")

        def fake_mark_failed(trace_id: str, *, error: str, judge_model: str) -> None:
            # Simulate the real attach landing on the subject trace's assessments so
            # the next read sees it exactly as MLflow would surface it.
            failing.raw.info.assessments.append(_assessment(cr.REVIEW_FAILED_FEEDBACK_NAME))

        monkeypatch.setattr(cr, "review_trace", fail_review_trace)
        monkeypatch.setattr(cr, "_mark_review_failed", fake_mark_failed)
        monkeypatch.setattr(cr, "_prepare_halo_sandbox", lambda: None)

        run1 = cr.run_continuous_rlm(
            "exp-1",
            judge_model="judge",
            source=source,
            max_reviews=1,
            sample_rate=1.0,
            min_tokens=0,
        )
        assert reviewed == ["bad"]
        assert run1.n_failed == 1
        assert cr.has_rlm_assessment(failing) is False
        assert cr.has_rlm_failure_marker(failing) is True

        run2 = cr.run_continuous_rlm(
            "exp-1",
            judge_model="judge",
            source=source,
            max_reviews=1,
            sample_rate=1.0,
            min_tokens=0,
        )
        assert reviewed == ["bad"]  # NOT reviewed a second time
        assert run2.n_reviewed == 0
        assert run2.n_failed == 0
        assert run2.n_selected == 0
        assert run2.n_already_reviewed == 1

    def test_recovery_mode_reselects_failed_trace(self) -> None:
        failing = _trace(
            "bad-retry", 1_000, assessments=[_assessment(cr.REVIEW_FAILED_FEEDBACK_NAME)]
        )
        selections, n_already, *_ = cr.select_unreviewed_traces(
            [failing], max_reviews=1, sample_rate=1.0, min_tokens=0, retry_failed=True
        )
        assert [selection.trace_id for selection in selections] == ["bad-retry"]
        assert n_already == 0

    def test_recovery_mode_prioritizes_never_attempted_traces(self) -> None:
        traces = [
            _trace(
                "failed-high-token",
                10_000,
                assessments=[_assessment(cr.REVIEW_FAILED_FEEDBACK_NAME)],
            ),
            _trace("fresh-1", 500),
            _trace("fresh-2", 400),
        ]
        selections, n_already, *_ = cr.select_unreviewed_traces(
            traces,
            max_reviews=2,
            sample_rate=1.0,
            min_tokens=0,
            retry_failed=True,
        )

        assert [selection.trace_id for selection in selections] == ["fresh-1", "fresh-2"]
        assert n_already == 0

    def test_recovery_mode_uses_spare_capacity_for_failed_retries(self) -> None:
        traces = [
            _trace("fresh", 100),
            _trace(
                "failed-high",
                1_000,
                assessments=[_assessment(cr.REVIEW_FAILED_FEEDBACK_NAME)],
            ),
            _trace(
                "failed-low",
                500,
                assessments=[_assessment(cr.REVIEW_FAILED_FEEDBACK_NAME)],
            ),
        ]
        selections, *_ = cr.select_unreviewed_traces(
            traces,
            max_reviews=2,
            sample_rate=1.0,
            min_tokens=0,
            retry_failed=True,
        )

        assert [selection.trace_id for selection in selections] == ["fresh", "failed-high"]

    def test_sets_trace_store_warehouse_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _FakeSource([])
        monkeypatch.delenv(TRACING_WAREHOUSE_ENV, raising=False)
        cr.run_continuous_rlm(
            "exp-1",
            judge_model="judge",
            sql_warehouse_id="warehouse-123",
            source=source,
        )
        assert os.environ[TRACING_WAREHOUSE_ENV] == "warehouse-123"

    def test_rejects_unbounded_sampling_inputs(self) -> None:
        with pytest.raises(ValueError, match="sample-rate"):
            cr.run_continuous_rlm(
                "exp-1",
                judge_model="judge",
                source=_FakeSource([]),
                sample_rate=2,
            )
        with pytest.raises(ValueError, match="max-reviews"):
            cr.run_continuous_rlm(
                "exp-1",
                judge_model="judge",
                source=_FakeSource([]),
                max_reviews=0,
            )
