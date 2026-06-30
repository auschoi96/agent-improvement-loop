"""The goal **allowlist** — the only metric/judge names a goal may reference.

A compiled goal's objective and its guardrails may name **only** a metric the
framework actually produces. This module derives that allowlist from the two
real sources, so it can never drift into naming a metric that does not exist:

* **L0 deterministic metrics** (:data:`L0_OBJECTIVE_METRICS`) — the un-gameable
  token/cost/latency/reuse numbers from :mod:`ail.metrics`. Each name is a real
  field on an :mod:`ail.metrics.contract` model; the membership check below
  fails the **import** loud if the L0 output contract drifts, so the allowlist
  is anchored to the contract rather than to a hand-copied string list.
* **Registered judges** (:data:`JUDGE_METRICS`) — the L2 judged-metric scorer
  names, taken straight from :data:`ail.judges.scorers.DEFAULT_SCORERS` (the
  built-in scorer registry). Adding a scorer there extends this set for free.

An NL goal that maps to a name outside :data:`ALLOWLIST` must fail loud (see
:class:`ail.goals.compiler.UnmappedMetricError`) — the compiler never silently
invents or mis-maps a metric.
"""

from __future__ import annotations

from ail.judges.scorers import DEFAULT_SCORERS
from ail.metrics.contract import (
    AggregateMetrics,
    CostAggregate,
    TokenBreakdown,
    ToolRedundancy,
    TraceMetrics,
)

__all__ = [
    "L0_OBJECTIVE_METRICS",
    "JUDGE_METRICS",
    "ALLOWLIST",
    "is_l0_metric",
    "is_judge",
    "is_known_metric",
]

#: The L0 deterministic metric names a goal may optimize or guardrail. Each is a
#: real field on an L0 output-contract model (see the import-time check below);
#: this is the curated subset of L0 fields that make sense as an optimization
#: *objective* or a deterministic guardrail (not every contract field — e.g.
#: ``trace_id`` or ``status`` are descriptive, not optimizable).
L0_OBJECTIVE_METRICS: tuple[str, ...] = (
    "total_tokens",
    "total_usd",
    "redundancy_rate",
    "total_tool_calls",
    "duration_seconds",
)

# Anchor the L0 allowlist to the metrics output contract: every name above must
# be a field on one of these contract models. If ail.metrics.contract renames or
# drops a field, importing this module fails loud rather than letting a goal name
# a metric the metrics tier no longer produces.
_L0_CONTRACT_MODELS = (
    TokenBreakdown,
    CostAggregate,
    ToolRedundancy,
    TraceMetrics,
    AggregateMetrics,
)
_L0_CONTRACT_FIELDS = frozenset(
    name for model in _L0_CONTRACT_MODELS for name in model.model_fields
)
_unbacked = sorted(m for m in L0_OBJECTIVE_METRICS if m not in _L0_CONTRACT_FIELDS)
if _unbacked:  # pragma: no cover - guards against silent contract drift
    raise RuntimeError(
        f"goal allowlist names L0 metric(s) {_unbacked} with no backing field on "
        "ail.metrics.contract — the L0 output contract drifted. Update "
        "L0_OBJECTIVE_METRICS to match ail.metrics.contract."
    )

_L0_SET: frozenset[str] = frozenset(L0_OBJECTIVE_METRICS)

#: The registered L2 judge names, derived from the built-in scorer registry.
JUDGE_METRICS: frozenset[str] = frozenset(DEFAULT_SCORERS)

#: Every name a goal may reference — the union of the two real sources. L0 and
#: judge names are disjoint by construction; a name in neither is *unmapped* and
#: must fail loud.
ALLOWLIST: frozenset[str] = _L0_SET | JUDGE_METRICS


def is_l0_metric(name: str) -> bool:
    """Whether ``name`` is a known L0 deterministic metric."""
    return name in _L0_SET


def is_judge(name: str) -> bool:
    """Whether ``name`` is a registered L2 judge (a *quality* signal)."""
    return name in JUDGE_METRICS


def is_known_metric(name: str) -> bool:
    """Whether ``name`` is anywhere in the allowlist (L0 metric or judge)."""
    return name in ALLOWLIST
