"""Offline tests for the MemAlign rollback demo (``scripts/demo_memalign_rollback.py``).

The demo itself is **operational** (live model + trace calls, run by hand). These
tests cover only its pure, offline pieces — never the live path:

* :func:`classify_rollback_dynamics` — the down/recover self-check logic, on
  synthetic agreement numbers.
* :func:`human_grade` — reading a *real* human label off a trace's assessments
  (mocked MLflow ``Feedback`` objects; no tracking backend).
* the constant-high bias relabel — that the bias subset is pushed to one wrong
  direction.

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
# the constant-high bias — one known-wrong direction
# ---------------------------------------------------------------------------


class TestBiasRelabel:
    def test_bias_pushes_every_grade_to_the_constant_high_target(self, demo: ModuleType) -> None:
        # The manipulation: relabel the bias subset to one constant high grade,
        # regardless of the true grade. Against an anchor that includes low
        # examples this is a known-wrong direction (it over-scores them).
        src = [
            TraceLabel(trace_id="a", name="token_efficiency", value=1.0),
            TraceLabel(trace_id="b", name="token_efficiency", value=3.0),
            TraceLabel(trace_id="c", name="token_efficiency", value=5.0),
        ]
        biased = [replace(lab, value=demo.BIAS_TARGET_GRADE) for lab in src]
        assert {lab.value for lab in biased} == {demo.BIAS_TARGET_GRADE}
        assert demo.BIAS_TARGET_GRADE >= 5.0  # the high end of the 1-5 scale
        # Trace ids are untouched (the wall still partitions by trace).
        assert [lab.trace_id for lab in biased] == ["a", "b", "c"]
