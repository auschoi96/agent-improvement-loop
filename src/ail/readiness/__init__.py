"""Readiness gating + eval-health: the "refuse to claim improvement until ready" wall.

This package is the product spine of ``docs/READINESS_AND_TRUST.md``: it computes,
**per cohort and per goal**, whether the system is allowed to claim an improvement,
and **refuses to lie** when it is not. Without it, scores climb while real quality
stalls — the exact failure the doc exists to prevent.

Three surfaces, all pure functions of a :class:`~ail.cohorts.Cohort` (identity) and
the measured :class:`ReadinessFacts`:

* **Readiness** (:func:`compute_readiness`) — a fail-closed
  :class:`ReadinessStatus` with a :class:`ReadinessTier`, one :class:`Gate` per
  data gate (trace counts, frozen suite, human labels, judge trust, scored
  coverage), and human-readable reasons for every unmet gate. Never green on
  missing data.
* **Eval-health** (:func:`compute_eval_health`) — an :class:`EvalHealth` surface:
  scored-coverage %, judge-run success rate, and the count of distrusted judges.
  A judge unmeasured against humans is **distrusted by default**.
* **Goal decoupling** (:class:`GoalView`) — a structural Protocol the goals lane's
  ``CompiledGoal`` satisfies, so readiness depends on the *shape* of a goal, not on
  :mod:`ail.goals`.

It imports the shared modules (:mod:`ail.cohorts`, :mod:`ail.judges`) but never
modifies them, and ties trust to :mod:`ail.judges.agreement`'s ``distrusted``
concept (:meth:`JudgeFact.from_agreement_report`).
"""

from __future__ import annotations

from ail.readiness.compute import (
    ReadinessThresholds,
    compute_eval_health,
    compute_readiness,
)
from ail.readiness.contract import (
    SCHEMA_VERSION,
    EvalHealth,
    Gate,
    GateName,
    JudgeHealth,
    ReadinessStatus,
    ReadinessTier,
)
from ail.readiness.facts import JudgeFact, ReadinessFacts
from ail.readiness.goal import GoalView

__all__ = [
    # contract
    "SCHEMA_VERSION",
    "ReadinessTier",
    "GateName",
    "Gate",
    "JudgeHealth",
    "EvalHealth",
    "ReadinessStatus",
    # goal decoupling
    "GoalView",
    # facts (inputs)
    "JudgeFact",
    "ReadinessFacts",
    # thresholds + compute
    "ReadinessThresholds",
    "compute_eval_health",
    "compute_readiness",
]
