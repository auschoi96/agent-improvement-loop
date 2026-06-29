"""Phase 2 core: the candidate-vs-baseline comparison harness.

This package is the reusable "evaluate candidate vs baseline on the frozen Task
Suite" step of the loop (``docs/ARCHITECTURE.md`` §4). Given one frozen
Task-Suite case, an :class:`~ail.ingest.base.AgentAdapter`, and an optional
:class:`Intervention`, :func:`compare_candidate` runs the agent with and without
the intervention, computes deterministic L0 deltas (tokens / cost / redundancy),
applies the anti-co-adaptation **correctness guardrail**, and emits a structured
:class:`ComparisonResult` with a ``PROMOTE`` / ``BLOCK`` recommendation.

* **Objective** — the L0 token/cost reduction, read straight off
  :mod:`ail.metrics`: deterministic and un-gameable.
* **Guardrail** — correctness must not regress. Uses the **BASE** correctness
  judge from :mod:`ail.judges` (interim; switches to the MemAlign-aligned judge
  once human labels exist — see :data:`~ail.compare.contract.INTERIM_JUDGE_NOTE`),
  plus any caller-supplied L1 programmatic signal.

:func:`~ail.compare.monitoring.configure_monitoring_warehouse` is the operational
helper that wires an experiment's monitoring SQL warehouse so the scheduled
scorers and future live scoring can fetch traces.
"""

from ail.compare.contract import (
    INTERIM_JUDGE_NOTE,
    SCHEMA_VERSION,
    ComparisonResult,
    GuardrailCheck,
    MetricDelta,
    Recommendation,
)
from ail.compare.harness import (
    CORRECTNESS_GUARDRAIL,
    EXECUTION_GUARDRAIL,
    PROGRAMMATIC_GUARDRAIL,
    CallableIntervention,
    ComparisonConfig,
    Intervention,
    ProgrammaticCheck,
    ProgrammaticSignal,
    compare_candidate,
)
from ail.compare.monitoring import (
    MONITORING_WAREHOUSE_TAG,
    TRACING_WAREHOUSE_ENV,
    MonitoringWarehouseConfig,
    configure_monitoring_warehouse,
)

__all__ = [
    # contract
    "SCHEMA_VERSION",
    "INTERIM_JUDGE_NOTE",
    "ComparisonResult",
    "MetricDelta",
    "GuardrailCheck",
    "Recommendation",
    # harness
    "compare_candidate",
    "ComparisonConfig",
    "Intervention",
    "CallableIntervention",
    "ProgrammaticSignal",
    "ProgrammaticCheck",
    "EXECUTION_GUARDRAIL",
    "CORRECTNESS_GUARDRAIL",
    "PROGRAMMATIC_GUARDRAIL",
    # monitoring warehouse
    "configure_monitoring_warehouse",
    "MonitoringWarehouseConfig",
    "MONITORING_WAREHOUSE_TAG",
    "TRACING_WAREHOUSE_ENV",
]
