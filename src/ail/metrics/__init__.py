"""L0 deterministic metrics: tokens, cost, latency, tool-call (re)use.

This package turns normalized traces (:class:`ail.ingest.base.NormalizedTrace`)
into a stable, typed, JSON-serializable metrics report — the **L0 output
contract** (:mod:`ail.metrics.contract`). L0 is the deterministic,
un-gameable tier of the layered metrics design: every number here is derived
mechanically from trace metadata (token counts, timestamps, tool spans) with no
model in the loop.

The computation engine lives in :mod:`ail.metrics.l0_deterministic`; the
diagnostic report (Example 1 reproduction) in :mod:`ail.metrics.report`.
"""

from __future__ import annotations

from ail.metrics.contract import (
    SCHEMA_VERSION,
    AggregateMetrics,
    CostAggregate,
    CostBreakdown,
    GroupMetrics,
    L0MetricsReport,
    PriceBookEntry,
    RepeatedCall,
    TokenBreakdown,
    TokenStats,
    ToolRedundancy,
    TraceMetrics,
)
from ail.metrics.l0_deterministic import (
    DEFAULT_PRICEBOOK,
    compute_cost,
    compute_l0,
    compute_redundancy,
    compute_trace_metrics,
)

__all__ = [
    "SCHEMA_VERSION",
    "AggregateMetrics",
    "CostAggregate",
    "CostBreakdown",
    "GroupMetrics",
    "L0MetricsReport",
    "PriceBookEntry",
    "RepeatedCall",
    "TokenBreakdown",
    "TokenStats",
    "ToolRedundancy",
    "TraceMetrics",
    "DEFAULT_PRICEBOOK",
    "compute_cost",
    "compute_l0",
    "compute_redundancy",
    "compute_trace_metrics",
]
