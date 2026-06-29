"""The :class:`GoalView` Protocol — readiness's view of a goal, decoupled.

Readiness gates a *goal* (``docs/READINESS_AND_TRUST.md`` §2: "compute readiness
**per goal**"), but it must not depend on the goals lane. A parallel lane compiles
a natural-language goal into a ``CompiledGoal`` (``docs/ARCHITECTURE.md`` §4,
``goals/compiler.py``); this module needs only three facts about whatever that
produces, so it asks for them **structurally** rather than importing the concrete
type.

Anything exposing :attr:`~GoalView.objective_metric`,
:attr:`~GoalView.guardrail_names`, and :attr:`~GoalView.requires_quality` satisfies
:class:`GoalView` — a ``CompiledGoal``, a test stub, or a one-off namedtuple. The
:func:`typing.runtime_checkable` decorator makes ``isinstance(x, GoalView)`` a
structural attribute-presence check, so a caller can assert conformance without a
nominal base class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class GoalView(Protocol):
    """The minimal, read-only view of a goal that readiness consumes.

    Args:
        objective_metric: The metric whose movement is the goal's objective (e.g.
            ``"total_tokens"``, ``"total_usd"``). Carried as provenance on the
            :class:`~ail.readiness.contract.ReadinessStatus`.
        guardrail_names: The guardrail checks the goal ships behind (e.g. a
            correctness judge). For a quality goal these name the judges whose
            trust the ``judge_trusted`` gate requires.
        requires_quality: Whether proving this goal needs a *judged* quality
            signal. When ``True``, the quality gates (frozen suite, human labels,
            a trusted judge, scored-coverage) are evaluated and must pass; a
            deterministic token/cost goal leaves them out.
    """

    @property
    def objective_metric(self) -> str: ...

    @property
    def guardrail_names(self) -> Sequence[str]: ...

    @property
    def requires_quality(self) -> bool: ...
