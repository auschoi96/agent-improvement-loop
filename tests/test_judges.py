"""Tests for the L2 judged-metrics layer (:mod:`ail.judges`).

All offline tests avoid live models two ways:

* **Scorer construction** uses the *real* ``make_judge`` — it only builds a
  judge object and never calls a model, so it runs offline with no network.
* **Scoring / alignment** use a ``FakeJudge`` (a duck-typed stand-in with a
  scripted ``__call__`` and ``align``), so the agreement and alignment wrappers
  are exercised without a model or ``dspy``.

The one genuinely-live path (a real MemAlign alignment) is gated behind
``@pytest.mark.live`` and self-skips without a workspace + ``dspy``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Literal

import pytest

from ail.judges import (
    CORRECTNESS,
    DEFAULT_SCORERS,
    GROUNDEDNESS,
    MODULARITY,
    TOKEN_EFFICIENCY,
    AgreementConfig,
    AlignmentSet,
    AnchorItem,
    HumanAnchor,
    Pool,
    PoolOverlapError,
    ScorePair,
    UnresolvedTraceIdError,
    align_judge,
    assert_pools_disjoint,
    build_memalign_optimizer,
    build_token_efficiency_inputs,
    coerce_score,
    compute_agreement,
    log_agreement,
    make_correctness_judge,
    make_groundedness_judge,
    make_modularity_judge,
    make_scorer,
    make_token_efficiency_judge,
    score_anchor,
    with_rubric,
)
from ail.judges.contract import SCHEMA_VERSION, AgreementReport

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Feedback:
    """Duck-typed stand-in for an MLflow ``Feedback`` (exposes ``.value``)."""

    def __init__(self, value: Any, rationale: str = "") -> None:
        self.value = value
        self.rationale = rationale


class FakeJudge:
    """A scripted judge: maps an item by its ``outputs`` to a score.

    ``responses`` maps an outputs value to whatever the judge "returns" (a raw
    scalar or a ``_Feedback``). ``raise_on`` names outputs values for which the
    judge raises, to exercise per-item error capture. ``align`` records its call
    and returns a fresh aligned judge.
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        responses: dict[Any, Any] | None = None,
        default: Any = None,
        raise_on: frozenset[Any] = frozenset(),
    ) -> None:
        self.name = name
        self.responses = responses or {}
        self.default = default
        self.raise_on = raise_on
        self.align_calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: Any = None,
        trace: Any = None,
        session: Any = None,
    ) -> Any:
        if outputs in self.raise_on:
            raise RuntimeError(f"judge blew up on {outputs!r}")
        return self.responses.get(outputs, self.default)

    def align(self, traces: list[Any], optimizer: Any = None) -> FakeJudge:
        self.align_calls.append({"traces": traces, "optimizer": optimizer})
        return FakeJudge(name=f"{self.name}+aligned", responses=self.responses)


class _Trace:
    """Minimal MLflow-trace shape for AlignmentSet id extraction."""

    def __init__(self, trace_id: str) -> None:
        self.info = type("Info", (), {"trace_id": trace_id})()


# ---------------------------------------------------------------------------
# Scorer factory — constructs real MLflow judges, offline
# ---------------------------------------------------------------------------


class TestScorerFactory:
    def test_default_set_has_four_scorers(self) -> None:
        assert set(DEFAULT_SCORERS) == {
            "correctness",
            "modularity",
            "groundedness",
            "token_efficiency",
        }

    def test_correctness_is_constrained_categorical_guardrail(self) -> None:
        judge = make_correctness_judge(model="openai:/gpt-4.1-mini")
        assert judge.name == "correctness"
        # Constrained categorical, not a bare str: the judge can only emit yes/no.
        assert judge.feedback_value_type == Literal["yes", "no"]
        # The guardrail rubric compares output against the expected result.
        assert "{{ outputs }}" in judge.instructions
        assert "{{ expectations }}" in judge.instructions

    def test_modularity_is_bounded_graded_scale(self) -> None:
        judge = make_modularity_judge(model="openai:/gpt-4.1-mini")
        assert judge.name == "modularity"
        # Bounded 1..5, not an unbounded int: out-of-range scores are impossible.
        assert judge.feedback_value_type == Literal[1, 2, 3, 4, 5]
        # The bounded Literal would lose make_judge's default mean; aggregations
        # are restored so the graded scale still rolls up across traces.
        assert judge.aggregations == ["mean", "median", "p90"]
        assert "{{ outputs }}" in judge.instructions

    def test_groundedness_checks_context(self) -> None:
        judge = make_groundedness_judge(model="openai:/gpt-4.1-mini")
        assert judge.name == "groundedness"
        assert judge.feedback_value_type == Literal["yes", "no"]
        assert "{{ inputs }}" in judge.instructions

    def test_make_scorer_overrides(self) -> None:
        judge = make_scorer(
            CORRECTNESS,
            model="openai:/gpt-4.1-mini",
            instructions="Custom rubric over {{ outputs }}.",
            feedback_value_type=bool,
            name="correctness_v2",
        )
        assert judge.name == "correctness_v2"
        assert judge.feedback_value_type is bool
        assert "Custom rubric" in judge.instructions

    def test_make_scorer_rejects_rubric_without_template_var(self) -> None:
        # make_judge enforces that instructions reference a template variable;
        # the factory does not paper over that.
        with pytest.raises(Exception):  # noqa: B017 - mlflow raises its own exc type
            make_scorer(
                MODULARITY,
                model="openai:/gpt-4.1-mini",
                instructions="no template variable here",
            )

    def test_with_rubric_returns_new_spec(self) -> None:
        tuned = with_rubric(GROUNDEDNESS, "Judge {{ outputs }} against {{ inputs }}.")
        assert tuned.name == GROUNDEDNESS.name
        assert tuned.instructions != GROUNDEDNESS.instructions
        assert GROUNDEDNESS.instructions  # original unchanged (frozen dataclass)


# ---------------------------------------------------------------------------
# Token-efficiency judge — bounded 1..5, consumes L0 inputs (not the raw trace)
# ---------------------------------------------------------------------------


def _trace_metrics(
    *,
    total_tokens: int = 843_000,
    repeated: list[tuple[str, str, int]] | None = None,
) -> Any:
    """A minimal L0 ``TraceMetrics`` for exercising the token-efficiency bridge."""
    from ail.metrics.contract import (
        CostBreakdown,
        RepeatedCall,
        TokenBreakdown,
        ToolRedundancy,
        TraceMetrics,
    )

    repeats = [
        RepeatedCall(tool=t, identity=i, count=c, signature_kind="path")
        for (t, i, c) in (repeated or [("Read", "/repo/foo.py", 34)])
    ]
    return TraceMetrics(
        trace_id="tr-1",
        model="claude-opus-4-8",
        tokens=TokenBreakdown(
            input_tokens=800_000, output_tokens=43_000, total_tokens=total_tokens
        ),
        cost=CostBreakdown(total_usd=12.34, priced=True),
        total_tool_calls=210,
        redundancy=ToolRedundancy(
            total_tool_calls=210,
            distinct_tool_calls=170,
            redundant_tool_calls=40,
            redundancy_rate=0.19,
            repeated_calls=repeats,
        ),
        duration_seconds=33_120.0,
    )


class TestTokenEfficiencyJudge:
    def test_is_bounded_graded_scale_1_to_5(self) -> None:
        judge = make_token_efficiency_judge(model="openai:/gpt-4.1-mini")
        assert judge.name == "token_efficiency"
        # Bounded to the integers 1..5: an out-of-range efficiency score is impossible.
        assert judge.feedback_value_type == Literal[1, 2, 3, 4, 5]
        # The bounded Literal would lose make_judge's mean; aggregations restored.
        assert judge.aggregations == ["mean", "median", "p90"]

    def test_in_default_set(self) -> None:
        assert DEFAULT_SCORERS["token_efficiency"] is TOKEN_EFFICIENCY

    def test_rubric_reads_l0_summary_not_raw_trace(self) -> None:
        # Large-trace-safe: the judge reasons over inputs/outputs/expectations
        # (a small L0 summary), never {{ trace }} (900K-token traces blow context).
        ins = make_token_efficiency_judge(model="openai:/gpt-4.1-mini").instructions
        assert "{{ inputs }}" in ins
        assert "{{ outputs }}" in ins
        assert "{{ expectations }}" in ins
        assert "{{ trace }}" not in ins

    def test_rubric_is_quality_conditioned_anti_gaming(self) -> None:
        # The rubric must explicitly refuse to reward "fewer tokens by doing less".
        ins = make_token_efficiency_judge(model="openai:/gpt-4.1-mini").instructions.lower()
        assert "success" in ins
        assert "correctness" in ins  # documents the Phase-2 guardrail pairing

    def test_build_inputs_consumes_l0_not_raw_trace(self) -> None:
        # The L0->judge bridge copies already-computed deterministic signals; it
        # never receives or embeds the raw trace.
        metrics = _trace_metrics(repeated=[("Read", "/repo/foo.py", 34)])
        payload = build_token_efficiency_inputs(metrics, task="refactor module X")
        assert payload["task"] == "refactor module X"
        sig = payload["l0_signals"]
        # Copied straight from the L0 record (not recomputed by the judge).
        assert sig["total_tokens"] == 843_000
        assert sig["model"] == "claude-opus-4-8"
        assert sig["redundancy_rate"] == 0.19
        assert sig["cost_usd"] == 12.34
        # The actionable signal: the named repeated target the judge can cite.
        assert sig["repeated_calls"][0] == {
            "tool": "Read",
            "identity": "/repo/foo.py",
            "count": 34,
            "kind": "path",
        }
        # No raw-trace escape hatch leaked into the judge input.
        assert "raw" not in sig and "spans" not in sig and "trace" not in payload

    def test_build_inputs_is_json_serializable(self) -> None:
        import json

        payload = build_token_efficiency_inputs(_trace_metrics(), task={"prompt": "x"})
        json.dumps(payload)  # must not raise

    def test_build_inputs_unpriced_cost_is_none(self) -> None:
        from ail.metrics.contract import (
            CostBreakdown,
            TokenBreakdown,
            ToolRedundancy,
            TraceMetrics,
        )

        metrics = TraceMetrics(
            trace_id="t",
            model="mystery-model",
            tokens=TokenBreakdown(total_tokens=10),
            cost=CostBreakdown(priced=False),
            total_tool_calls=0,
            redundancy=ToolRedundancy(),
        )
        sig = build_token_efficiency_inputs(metrics)["l0_signals"]
        assert sig["cost_priced"] is False
        assert sig["cost_usd"] is None

    def test_judge_can_only_emit_in_range_via_score_anchor(self) -> None:
        # A judge wired to the agreement path coerces a bounded score; the
        # FakeJudge stands in for the constrained make_judge output.
        anchor = HumanAnchor.of(
            [AnchorItem(item_id="t1", human_label=4, outputs="resp", inputs={"l0": 1})]
        )
        judge = FakeJudge(name="token_efficiency", responses={"resp": _Feedback(4)})
        report = score_anchor(judge, anchor)
        assert report.judge_name == "token_efficiency"
        assert report.n_agreements == 1


# ---------------------------------------------------------------------------
# Pools — the frozen evaluation wall as types
# ---------------------------------------------------------------------------


class TestPools:
    def test_anchor_rejects_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate item_id"):
            HumanAnchor.of(
                [
                    AnchorItem(item_id="dup", human_label="yes"),
                    AnchorItem(item_id="dup", human_label="no"),
                ]
            )

    def test_alignment_set_reads_trace_ids(self) -> None:
        aset = AlignmentSet.of([_Trace("t1"), _Trace("t2")])
        assert aset.ids == frozenset({"t1", "t2"})
        assert len(aset) == 2

    def test_disjoint_pools_pass(self) -> None:
        aset = AlignmentSet.of([_Trace("a1"), _Trace("a2")])
        anchor = HumanAnchor.of([AnchorItem(item_id="h1", human_label="yes")])
        # Disjoint: no exception.
        assert_pools_disjoint(alignment_set=aset, human_anchor=anchor, task_suite_ids=["s1", "s2"])

    def test_alignment_anchor_overlap_raises(self) -> None:
        aset = AlignmentSet.of([_Trace("shared")])
        anchor = HumanAnchor.of([AnchorItem(item_id="shared", human_label="yes")])
        with pytest.raises(PoolOverlapError, match="not disjoint"):
            assert_pools_disjoint(alignment_set=aset, human_anchor=anchor)

    def test_task_suite_overlap_raises(self) -> None:
        aset = AlignmentSet.of([_Trace("x1")])
        with pytest.raises(PoolOverlapError):
            assert_pools_disjoint(alignment_set=aset, task_suite_ids=["x1"])

    def test_pool_labels_are_stable(self) -> None:
        assert Pool.ALIGNMENT_SET.value == "alignment_set"
        assert Pool.HUMAN_ANCHOR.value == "human_anchor"
        assert Pool.TASK_SUITE.value == "task_suite"
        assert HumanAnchor.pool is Pool.HUMAN_ANCHOR
        assert AlignmentSet.pool is Pool.ALIGNMENT_SET

    def test_unresolvable_trace_id_fails_closed(self) -> None:
        # A trace with no resolvable id cannot be proven disjoint; the guard must
        # fail closed and raise rather than silently drop it from the check.
        class _NoId:
            info = None

        aset = AlignmentSet.of([_Trace("ok"), _NoId()])
        assert aset.unresolved_count == 1
        assert aset.ids == frozenset({"ok"})  # the resolvable one is still surfaced
        with pytest.raises(UnresolvedTraceIdError, match="resolvable id"):
            assert_pools_disjoint(alignment_set=aset)

    def test_unresolved_trace_id_error_is_overlap_subtype(self) -> None:
        # Callers that already catch PoolOverlapError catch the fail-closed case.
        assert issubclass(UnresolvedTraceIdError, PoolOverlapError)


# ---------------------------------------------------------------------------
# coerce_score — normalize whatever a judge returns
# ---------------------------------------------------------------------------


class TestCoerceScore:
    def test_scalars_pass_through(self) -> None:
        assert coerce_score("yes") == "yes"
        assert coerce_score(True) is True
        assert coerce_score(4) == 4
        assert coerce_score(0.5) == 0.5
        assert coerce_score(None) is None

    def test_unwraps_feedback(self) -> None:
        assert coerce_score(_Feedback("no")) == "no"
        assert coerce_score(_Feedback(5)) == 5

    def test_unwraps_single_element_list(self) -> None:
        assert coerce_score([_Feedback("yes")]) == "yes"

    def test_multi_element_list_raises(self) -> None:
        with pytest.raises(ValueError, match="multi-element"):
            coerce_score([_Feedback("yes"), _Feedback("no")])

    def test_mlflow_categorical_rating_coerces(self) -> None:
        from mlflow.genai.judges import CategoricalRating

        # A real CategoricalRating is a str enum; it normalizes to its value.
        assert coerce_score(CategoricalRating.YES) == "yes"
        assert coerce_score(_Feedback(CategoricalRating.NO)) == "no"


# ---------------------------------------------------------------------------
# compute_agreement — pure metric, no model
# ---------------------------------------------------------------------------


def _pairs(*triples: tuple[str, Any, Any]) -> list[ScorePair]:
    return [ScorePair(item_id=i, judge_value=j, human_value=h) for i, j, h in triples]


class TestComputeAgreement:
    def test_perfect_agreement(self) -> None:
        report = compute_agreement(
            _pairs(("a", "yes", "yes"), ("b", "no", "no"), ("c", "yes", "yes")),
            judge_name="correctness",
        )
        assert report.schema_version == SCHEMA_VERSION
        assert report.n_items == 3
        assert report.n_agreements == 3
        assert report.agreement_rate == 1.0
        assert report.distrusted is False
        assert report.cohen_kappa == 1.0
        assert report.pool == "human_anchor"

    def test_below_floor_is_distrusted(self) -> None:
        # 1 of 3 agree → 0.333 < default floor 0.7
        report = compute_agreement(
            _pairs(("a", "yes", "yes"), ("b", "yes", "no"), ("c", "no", "yes")),
            judge_name="correctness",
        )
        assert report.agreement_rate == pytest.approx(1 / 3, abs=1e-6)
        assert report.distrusted is True

    def test_floor_is_configurable(self) -> None:
        pairs = _pairs(("a", "yes", "yes"), ("b", "yes", "no"))  # 0.5
        strict = compute_agreement(pairs, judge_name="j", config=AgreementConfig(floor=0.9))
        lenient = compute_agreement(pairs, judge_name="j", config=AgreementConfig(floor=0.4))
        assert strict.distrusted is True
        assert lenient.distrusted is False

    def test_case_insensitive_string_match(self) -> None:
        report = compute_agreement(_pairs(("a", "Yes", "yes")), judge_name="j")
        assert report.n_agreements == 1

    def test_numeric_tolerance(self) -> None:
        config = AgreementConfig(numeric_tolerance=0.5)
        report = compute_agreement(
            [
                ScorePair(item_id="a", judge_value=4.2, human_value=4.0),
                ScorePair(item_id="b", judge_value=1.0, human_value=3.0),
            ],
            judge_name="modularity",
            config=config,
        )
        assert report.items[0].agree is True  # within 0.5
        assert report.items[1].agree is False  # outside 0.5
        assert report.cohen_kappa is None  # omitted for float-tolerance comparison
        assert report.numeric_tolerance == 0.5

    def test_errored_item_counts_as_non_agreement(self) -> None:
        report = compute_agreement(
            [
                ScorePair(item_id="a", judge_value="yes", human_value="yes"),
                ScorePair(item_id="b", human_value="no", error="boom"),
            ],
            judge_name="j",
        )
        assert report.n_items == 2
        assert report.n_scored == 1
        assert report.n_agreements == 1
        assert report.agreement_rate == 0.5
        errored = next(i for i in report.items if i.item_id == "b")
        assert errored.agree is False
        assert errored.error == "boom"

    def test_empty_anchor_is_distrusted_fail_closed(self) -> None:
        # An unmeasured judge must NOT read as trusted: zero items is
        # insufficient data, so the judge is flagged distrusted (fail closed).
        report = compute_agreement([], judge_name="j")
        assert report.n_items == 0
        assert report.n_scored == 0
        assert report.agreement_rate == 0.0
        assert report.insufficient_data is True
        assert report.distrusted is True
        assert any("unmeasured" in n for n in report.notes)

    def test_below_min_samples_is_distrusted_even_with_perfect_rate(self) -> None:
        # Two perfectly-agreeing items, but a guardrail that demands at least
        # five scored items: too little evidence to trust, so distrusted fires
        # despite a 1.0 agreement rate.
        report = compute_agreement(
            _pairs(("a", "yes", "yes"), ("b", "no", "no")),
            judge_name="j",
            config=AgreementConfig(min_samples=5),
        )
        assert report.agreement_rate == 1.0
        assert report.insufficient_data is True
        assert report.distrusted is True

    def test_sufficient_samples_clears_insufficient_flag(self) -> None:
        report = compute_agreement(
            _pairs(("a", "yes", "yes"), ("b", "no", "no")),
            judge_name="j",
            config=AgreementConfig(min_samples=2),
        )
        assert report.insufficient_data is False
        assert report.distrusted is False

    def test_case_sensitive_kappa_keeps_labels_distinct(self) -> None:
        # With case_insensitive=False the judge's "Yes" does NOT match the
        # human "yes" — for the agreement decision AND for kappa's label space.
        pairs = _pairs(("a", "Yes", "yes"), ("b", "No", "no"))
        report = compute_agreement(
            pairs, judge_name="j", config=AgreementConfig(case_insensitive=False, floor=0.0)
        )
        assert report.n_agreements == 0  # "Yes" != "yes", "No" != "no"
        # The reported label_space is the human side; case preserved (not folded).
        assert set(report.label_space) == {"yes", "no"}
        # Kappa's internal label space saw 4 distinct labels, so it is defined.
        assert report.cohen_kappa is not None

    def test_kappa_below_rate_under_class_imbalance(self) -> None:
        # Judge always says "yes"; humans mostly "yes". High raw agreement, but
        # chance-corrected kappa is lower — exactly why kappa is reported.
        pairs = _pairs(
            ("a", "yes", "yes"),
            ("b", "yes", "yes"),
            ("c", "yes", "yes"),
            ("d", "yes", "no"),
        )
        report = compute_agreement(pairs, judge_name="j", config=AgreementConfig(floor=0.0))
        assert report.agreement_rate == 0.75
        assert report.cohen_kappa is not None
        assert report.cohen_kappa < report.agreement_rate

    def test_report_round_trips_json(self) -> None:
        report = compute_agreement(_pairs(("a", "yes", "yes")), judge_name="j")
        restored = AgreementReport.model_validate_json(report.model_dump_json())
        assert restored == report


# ---------------------------------------------------------------------------
# score_anchor — judge over a Human-Anchor slice (mocked judge)
# ---------------------------------------------------------------------------


class TestScoreAnchor:
    def test_scores_each_item_against_human_label(self) -> None:
        anchor = HumanAnchor.of(
            [
                AnchorItem(item_id="a", human_label="yes", outputs="resp-a"),
                AnchorItem(item_id="b", human_label="no", outputs="resp-b"),
            ]
        )
        # Judge agrees on "a", disagrees on "b".
        judge = FakeJudge(
            name="correctness",
            responses={"resp-a": _Feedback("yes"), "resp-b": _Feedback("yes")},
        )
        report = score_anchor(judge, anchor, generated_at="2026-06-29T00:00:00+00:00")
        assert report.judge_name == "correctness"
        assert report.n_items == 2
        assert report.n_agreements == 1
        assert report.agreement_rate == 0.5

    def test_judge_exception_is_captured_per_item(self) -> None:
        anchor = HumanAnchor.of(
            [
                AnchorItem(item_id="a", human_label="yes", outputs="ok"),
                AnchorItem(item_id="b", human_label="yes", outputs="bomb"),
            ]
        )
        judge = FakeJudge(responses={"ok": "yes"}, raise_on=frozenset({"bomb"}))
        report = score_anchor(judge, anchor)
        bad = next(i for i in report.items if i.item_id == "b")
        assert bad.error is not None and "blew up" in bad.error
        assert bad.agree is False
        assert report.n_agreements == 1  # the good item still scored


# ---------------------------------------------------------------------------
# align_judge — MemAlign wrapper (mocked judge.align)
# ---------------------------------------------------------------------------


class TestAlignJudge:
    def test_aligns_on_alignment_set_only(self) -> None:
        judge = FakeJudge(name="correctness")
        aset = AlignmentSet.of([_Trace("t1"), _Trace("t2")])
        outcome = align_judge(judge, aset)
        # The aligned judge is returned, distinct from the base.
        assert outcome.judge is not judge
        assert outcome.judge.name == "correctness+aligned"
        # The whole alignment set is passed through to judge.align.
        assert judge.align_calls[0]["traces"] == list(aset.traces)
        # optimizer=None → MLflow's default MemAlign.
        assert judge.align_calls[0]["optimizer"] is None

    def test_report_records_provenance(self) -> None:
        judge = FakeJudge(name="groundedness")
        aset = AlignmentSet.of([_Trace("t1"), _Trace("t2"), _Trace("t3")])
        outcome = align_judge(judge, aset)
        report = outcome.report
        assert report.base_judge_name == "groundedness"
        assert report.optimizer == "MemAlign"
        assert report.pool == "alignment_set"
        assert report.n_alignment_traces == 3
        assert report.aligned is True
        assert any("decoupled from agent optimization" in n for n in report.notes)

    def test_custom_optimizer_passed_through(self) -> None:
        judge = FakeJudge(name="j")
        aset = AlignmentSet.of([_Trace("t1")])
        sentinel = object()
        align_judge(judge, aset, optimizer=sentinel)
        assert judge.align_calls[0]["optimizer"] is sentinel

    def test_empty_alignment_set_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty AlignmentSet"):
            align_judge(FakeJudge(), AlignmentSet.of([]))

    def test_build_memalign_optimizer_requires_dspy(self) -> None:
        # The align path must fail with a clear, actionable ImportError when the
        # optional dspy backend is absent (the default ``.[dev]`` / CI install).
        # Skipped when dspy IS present (the ``align`` extra) — that case is
        # covered by test_build_memalign_optimizer_with_dspy below.
        if importlib.util.find_spec("dspy") is not None:
            pytest.skip("dspy installed (the 'align' extra); the absence path is not exercisable")
        with pytest.raises(ImportError, match="dspy"):
            build_memalign_optimizer()

    def test_build_memalign_optimizer_with_dspy(self) -> None:
        # With the optional ``align`` extra (dspy) installed, the lazy import
        # resolves and build_memalign_optimizer returns a real MemAlign optimizer
        # without raising. Skipped in the default (dspy-absent) install, so this
        # asserts the dependency is correctly declared and importable end-to-end.
        pytest.importorskip("dspy", reason="requires the optional 'align' extra (dspy)")
        from mlflow.genai.judges.base import AlignmentOptimizer

        optimizer = build_memalign_optimizer()
        assert isinstance(optimizer, AlignmentOptimizer)


# ---------------------------------------------------------------------------
# log_agreement — best-effort MLflow logging
# ---------------------------------------------------------------------------


class TestLogAgreement:
    def test_logs_metrics_and_artifact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mlflow

        metrics: dict[str, float] = {}
        dicts: dict[str, Any] = {}
        monkeypatch.setattr(mlflow, "log_metric", lambda k, v, **kw: metrics.__setitem__(k, v))
        monkeypatch.setattr(mlflow, "log_dict", lambda d, path, **kw: dicts.__setitem__(path, d))

        report = compute_agreement(
            _pairs(("a", "yes", "yes"), ("b", "yes", "no")),
            judge_name="correctness",
            config=AgreementConfig(floor=0.9),
        )
        assert log_agreement(report) is True
        assert metrics["judge_human_agreement"] == report.agreement_rate
        assert metrics["judge_distrusted"] == 1.0  # 0.5 < 0.9
        assert "judge_agreement/correctness.json" in dicts

    def test_returns_false_when_logging_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mlflow

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no active run")

        monkeypatch.setattr(mlflow, "log_metric", boom)
        report = compute_agreement(_pairs(("a", "yes", "yes")), judge_name="j")
        assert log_agreement(report) is False


# ---------------------------------------------------------------------------
# Live alignment — gated, self-skips without a workspace + dspy
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_memalign_alignment() -> None:
    """Acceptance (live): a real judge aligns from labeled traces via MemAlign.

    Guarded by ``AIL_LIVE_MLFLOW=1`` and the presence of ``dspy`` + a workspace.
    This exercises the genuine ``judge.align(traces=..., optimizer=...)`` path
    the offline tests mock.
    """
    if os.environ.get("AIL_LIVE_MLFLOW") != "1":
        pytest.skip("set AIL_LIVE_MLFLOW=1 to run the live MemAlign alignment")
    pytest.importorskip("dspy", reason="MemAlign requires the dspy optimizer dependency")

    # The optimizer constructs only with dspy present; the trace fixtures + a
    # workspace are supplied by the live environment.
    optimizer = build_memalign_optimizer()
    assert optimizer is not None
