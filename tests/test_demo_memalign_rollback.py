"""Offline tests for the MemAlign rollback demo (``scripts/demo_memalign_rollback.py``).

The demo itself is **operational** (live model + trace calls, run by hand). These
tests cover only its pure, offline pieces — never the live path:

* :func:`classify_rollback_dynamics` — the down/recover self-check logic, on
  synthetic agreement numbers (the honest FIRED formula, which must stay
  ``overfit < aligned`` AND recovered).
* :func:`human_grade` — reading a *real* human label off a trace's assessments
  (mocked MLflow ``Feedback`` objects; no tracking backend).
* :func:`invert_grade` and the bias step — that the biased subset's grades are
  *inverted* (``g -> 6 - g``), the maximally-wrong manipulation, while real labels
  elsewhere are untouched.
* anchor blinding — that the held-out anchor trace the judge scores carries no
  human gold (the gold lives only on ``AnchorItem.human_label``), via the same
  ``to_human_anchor(..., source=...)`` call the demo makes (mocked source).

The demo lives in ``scripts/`` (not an importable package), so it is loaded by
path. It must be registered in ``sys.modules`` before exec for its
``@dataclass(slots=True)`` definitions to resolve their module namespace.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ail.judges.labeling import TraceLabel

_DEMO_PATH = Path(__file__).resolve().parent.parent / "scripts" / "demo_memalign_rollback.py"


def _load_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_memalign_rollback", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass(slots=True) looks the module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    return _load_demo()


# ---------------------------------------------------------------------------
# classify_rollback_dynamics — the honest down/recover self-check
# ---------------------------------------------------------------------------


class TestClassifyRollbackDynamics:
    def test_down_then_recover_fires(self, demo: ModuleType) -> None:
        d = demo.classify_rollback_dynamics(aligned=0.95, overfit=0.55, rolled_back=0.93)
        assert d.manipulation_moved_down
        assert d.rollback_recovered
        assert d.fired

    def test_overfit_not_below_aligned_does_not_fire(self, demo: ModuleType) -> None:
        # The original failure mode: manipulation moved agreement UP, not down.
        d = demo.classify_rollback_dynamics(aligned=0.95, overfit=1.00, rolled_back=0.97)
        assert not d.manipulation_moved_down
        assert not d.fired

    def test_partial_recovery_below_aligned_is_not_recovered(self, demo: ModuleType) -> None:
        # Rolled back UP from overfit, but nowhere near ALIGNED → not recovered.
        d = demo.classify_rollback_dynamics(aligned=0.95, overfit=0.55, rolled_back=0.65)
        assert d.manipulation_moved_down
        assert not d.rollback_recovered
        assert not d.fired

    def test_recovery_within_tolerance_counts(self, demo: ModuleType) -> None:
        # Episodic memory need not reconstruct byte-identically; back to within the
        # tolerance of ALIGNED counts as recovered.
        d = demo.classify_rollback_dynamics(
            aligned=0.90, overfit=0.50, rolled_back=0.86, recovery_tolerance=0.05
        )
        assert d.rollback_recovered
        d2 = demo.classify_rollback_dynamics(
            aligned=0.90, overfit=0.50, rolled_back=0.84, recovery_tolerance=0.05
        )
        assert not d2.rollback_recovered

    def test_recovery_requires_rising_above_overfit(self, demo: ModuleType) -> None:
        # Even if rolled_back is within tolerance of aligned, it must exceed overfit
        # (otherwise nothing was retracted/recovered).
        d = demo.classify_rollback_dynamics(aligned=0.90, overfit=0.90, rolled_back=0.90)
        assert not d.manipulation_moved_down
        assert not d.rollback_recovered

    def test_formula_is_exactly_down_and_recovered(self, demo: ModuleType) -> None:
        # Pin the honest self-check formula so a future edit cannot quietly change
        # it: moved_down is purely OVERFIT < ALIGNED, and recovered is ROLLED-BACK
        # above OVERFIT AND within tolerance of ALIGNED. Never hardcoded/faked.
        for aligned, overfit, rolled_back in [
            (0.95, 0.55, 0.93),
            (0.95, 1.00, 0.97),
            (0.90, 0.50, 0.84),
            (0.80, 0.80, 0.80),
            (0.70, 0.40, 0.71),
        ]:
            tol = demo.DEFAULT_RECOVERY_TOLERANCE
            d = demo.classify_rollback_dynamics(
                aligned=aligned, overfit=overfit, rolled_back=rolled_back
            )
            assert d.manipulation_moved_down == (overfit < aligned)
            assert d.rollback_recovered == (rolled_back > overfit and rolled_back >= aligned - tol)
            assert d.fired == (d.manipulation_moved_down and d.rollback_recovered)


# ---------------------------------------------------------------------------
# human_grade — read a REAL human label off a trace (never fabricated)
# ---------------------------------------------------------------------------


class _Info:
    def __init__(self, trace_id: str, assessments: list[Any]) -> None:
        self.trace_id = trace_id
        self.assessments = assessments


class _Raw:
    def __init__(self, trace_id: str, assessments: list[Any]) -> None:
        self.info = _Info(trace_id, assessments)


class _Trace:
    """Minimal NormalizedTrace surface the demo reader touches: ``.raw.info.assessments``."""

    def __init__(self, trace_id: str, assessments: list[Any]) -> None:
        self.trace_id = trace_id
        self.raw = _Raw(trace_id, assessments)


def _feedback(name: str, value: Any, *, human: bool = True, source_id: str = "austin") -> Any:
    from mlflow.entities import AssessmentSource, Feedback
    from mlflow.entities.assessment_source import AssessmentSourceType

    source_type = AssessmentSourceType.HUMAN if human else AssessmentSourceType.LLM_JUDGE
    return Feedback(
        name=name,
        value=value,
        rationale="re-read foo.py 34x" if human else None,
        source=AssessmentSource(source_type=source_type, source_id=source_id),
    )


class TestHumanGrade:
    def test_reads_human_label_and_rationale(self, demo: ModuleType) -> None:
        trace = _Trace("t1", [_feedback("token_efficiency", 2)])
        result = demo.human_grade(trace, name="token_efficiency")
        assert result is not None
        value, rationale = result
        assert value == 2.0
        assert isinstance(value, float)  # numeric for tolerance-based agreement
        assert rationale == "re-read foo.py 34x"

    def test_ignores_non_human_assessment(self, demo: ModuleType) -> None:
        # A prior LLM-judge score on the trace is not a human label and is skipped.
        trace = _Trace("t1", [_feedback("token_efficiency", 5, human=False)])
        assert demo.human_grade(trace, name="token_efficiency") is None

    def test_ignores_other_judge_name(self, demo: ModuleType) -> None:
        trace = _Trace("t1", [_feedback("correctness", 1)])
        assert demo.human_grade(trace, name="token_efficiency") is None

    def test_none_when_unlabeled(self, demo: ModuleType) -> None:
        assert demo.human_grade(_Trace("t1", []), name="token_efficiency") is None

    def test_coerces_numeric_string_value(self, demo: ModuleType) -> None:
        trace = _Trace("t1", [_feedback("token_efficiency", "3")])
        result = demo.human_grade(trace, name="token_efficiency")
        assert result is not None and result[0] == 3.0

    def test_prefers_named_labeler(self, demo: ModuleType) -> None:
        trace = _Trace(
            "t1",
            [
                _feedback("token_efficiency", 5, source_id="other"),
                _feedback("token_efficiency", 1, source_id="austin"),
            ],
        )
        result = demo.human_grade(trace, name="token_efficiency", labeler_id="austin")
        assert result is not None and result[0] == 1.0


# ---------------------------------------------------------------------------
# the label-inversion bias — maximally-wrong feedback (g -> 6 - g)
# ---------------------------------------------------------------------------


class TestInvertGrade:
    def test_reflects_across_scale_midpoint(self, demo: ModuleType) -> None:
        # 5<->1, 4<->2, 3->3 on the 1-5 scale.
        assert demo.invert_grade(5.0) == 1.0
        assert demo.invert_grade(4.0) == 2.0
        assert demo.invert_grade(3.0) == 3.0
        assert demo.invert_grade(2.0) == 4.0
        assert demo.invert_grade(1.0) == 5.0

    def test_uses_the_declared_scale_endpoints(self, demo: ModuleType) -> None:
        # The reflection point is (MIN + MAX), not a hardcoded 6.
        mid_sum = demo.GRADE_SCALE_MIN + demo.GRADE_SCALE_MAX
        for g in (1.0, 2.0, 3.0, 4.0, 5.0):
            assert demo.invert_grade(g) == mid_sum - g

    def test_coerces_numeric_strings_and_ints(self, demo: ModuleType) -> None:
        assert demo.invert_grade("2") == 4.0
        assert demo.invert_grade(4) == 2.0

    def test_raises_on_non_numeric_grade(self, demo: ModuleType) -> None:
        # Fail loudly rather than silently mislabel — the demo only ever feeds it
        # real numeric human grades.
        with pytest.raises(ValueError, match="cannot invert"):
            demo.invert_grade("efficient")


class TestBiasInversionStep:
    def test_bias_step_inverts_each_grade(self, demo: ModuleType) -> None:
        # The manipulation as the demo applies it: invert every biased trace's real
        # grade (g -> 6 - g). This conflicts with the held-out truth regardless of
        # skew — unlike the old constant-high relabel a high anchor could not detect.
        src = [
            TraceLabel(trace_id="a", name="token_efficiency", value=1.0),
            TraceLabel(trace_id="b", name="token_efficiency", value=3.0),
            TraceLabel(trace_id="c", name="token_efficiency", value=5.0),
        ]
        biased = [replace(lab, value=demo.invert_grade(lab.value)) for lab in src]
        assert [lab.value for lab in biased] == [5.0, 3.0, 1.0]
        # Trace ids are untouched (the disjoint wall still partitions by trace).
        assert [lab.trace_id for lab in biased] == ["a", "b", "c"]
        # Names and the rest of the label are preserved (only the value flips).
        assert {lab.name for lab in biased} == {"token_efficiency"}

    def test_inversion_is_not_a_no_op_on_skewed_grades(self, demo: ModuleType) -> None:
        # The whole point: on a high-skewed set (4s and 5s — what the real corpus
        # has), inversion still moves every grade, so it conflicts with a high anchor.
        high = [
            TraceLabel(trace_id="x", name="token_efficiency", value=5.0),
            TraceLabel(trace_id="y", name="token_efficiency", value=4.0),
        ]
        biased = [replace(lab, value=demo.invert_grade(lab.value)) for lab in high]
        assert [lab.value for lab in biased] == [1.0, 2.0]
        assert all(b.value != s.value for b, s in zip(biased, high, strict=True))


# ---------------------------------------------------------------------------
# anchor blinding — the held-out trace the judge scores carries no human gold
# (preserved from PR #28/#30; must not regress)
# ---------------------------------------------------------------------------


class TestAnchorBlinding:
    def test_demo_anchor_construction_blinds_the_human_gold(self, demo: ModuleType) -> None:
        # The demo builds its held-out anchor with to_human_anchor(..., source=...),
        # which blinds each anchor trace of its HUMAN gold so a {{ trace }} judge
        # cannot read its own answer off trace.info.assessments. The gold survives
        # only on AnchorItem.human_label, for comparison. Guard that invariant here
        # against the exact call the demo makes (mocked source; no live calls).
        from collections.abc import Iterator

        from mlflow.entities import AssessmentSource, Feedback
        from mlflow.entities.assessment_source import AssessmentSourceType

        from ail.ingest.base import NormalizedTrace, TraceSource
        from ail.judges import to_human_anchor

        class _RawInfo:
            def __init__(self, trace_id: str) -> None:
                self.trace_id = trace_id
                self.assessments: list[Any] = []

        class _Raw:
            def __init__(self, trace_id: str) -> None:
                self.info = _RawInfo(trace_id)

        class _Src(TraceSource):
            def iter_traces(self, **_: Any) -> Iterator[NormalizedTrace]:  # pragma: no cover
                return iter([])

            def get_trace(self, trace_id: str) -> NormalizedTrace | None:
                raw = _Raw(trace_id)
                raw.info.assessments = [
                    Feedback(
                        name="token_efficiency",
                        value=2,  # HUMAN gold the judge must NOT see
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.HUMAN, source_id="austin"
                        ),
                    ),
                    Feedback(
                        name="token_efficiency",
                        value=4,  # prior LLM-judge score: not gold, kept
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.LLM_JUDGE, source_id="judge"
                        ),
                    ),
                ]
                return NormalizedTrace(trace_id=trace_id, raw=raw)

        anchor_labels = [TraceLabel(trace_id="t1", name="token_efficiency", value=2.0)]
        anchor = to_human_anchor(anchor_labels, name="token_efficiency", source=_Src())
        item = anchor.items[0]

        # Gold survives on the item (agreement compares against this)...
        assert item.human_label == 2.0
        # ...but the trace handed to the judge carries NO human assessment.
        seen = item.trace.info.assessments
        assert [a for a in seen if str(a.source.source_type) == "HUMAN"] == []
        # Non-human assessments are preserved (we blind only the human gold).
        assert any(str(a.source.source_type) == "LLM_JUDGE" for a in seen)
