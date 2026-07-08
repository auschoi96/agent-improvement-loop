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

An NL goal that maps to a name outside the allowlist must fail loud (see
:class:`ail.goals.compiler.UnmappedMetricError`) — the compiler never silently
invents or mis-maps a metric.

**Dynamic judges (the requirements-intake front door).** The built-in scorer
registry is not the only source of judges: the intake engine authors a fresh
``{{ trace }}`` judge per custom dimension (:func:`ail.judges.author_judge`), and
a goal that references such a judge must still validate. The built-in set
(:data:`JUDGE_METRICS`) is therefore the **static floor**, not the whole story:
:func:`is_judge` also admits any name registered in the ambient **dynamic judge
allowlist** (:func:`judge_allowlist` / :func:`dynamic_judge_names`), which a
caller populates from the authored/registered judges it knows about
(:func:`sourced_judge_names` reads them from the live scorer registry). This is
purely **additive** and **fail-closed**: with no dynamic context the behaviour is
identical to the static allowlist, an unreadable registry raises
:class:`AllowlistSourceError` (never "allow everything"), and the deterministic-L0
set is untouched.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from typing import Any

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
    "AllowlistSourceError",
    "ScorerLister",
    "is_l0_metric",
    "is_judge",
    "is_known_metric",
    "dynamic_judge_names",
    "judge_allowlist",
    "sourced_judge_names",
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

#: The built-in L2 judge names, derived from the built-in scorer registry. This is
#: the **static floor** of the judge allowlist — always admitted, needs no backend
#: to read, and is what makes this module importable offline. Authored/registered
#: judges beyond this set are admitted dynamically (see :func:`judge_allowlist`).
JUDGE_METRICS: frozenset[str] = frozenset(DEFAULT_SCORERS)

#: The **static** allowlist: L0 metrics ∪ the built-in judges. Used to ground the
#: compiler's system prompt (:func:`ail.goals.compiler._build_system_prompt`) and
#: as the always-available baseline. Dynamically-authored judges are *not* in here
#: (they have no static name to list); :func:`is_known_metric` consults the dynamic
#: set separately. L0 and judge names are disjoint by construction.
ALLOWLIST: frozenset[str] = _L0_SET | JUDGE_METRICS


# ---------------------------------------------------------------------------
# Dynamic judge allowlist — additive, fail-closed re-sourcing of authored judges
# ---------------------------------------------------------------------------
#
# A freshly-authored judge (ail.judges.author_judge) is not in the static
# JUDGE_METRICS, so a goal referencing it would fail UnmappedMetricError. Rather
# than widen the static set (which would let the compiler's system prompt offer
# judges that may not exist) the extra judge names are admitted through an ambient
# ContextVar a caller sets around goal construction. Default-empty means the
# behaviour is byte-for-byte the static allowlist; only inside an explicit
# ``with judge_allowlist(...)`` block is the set widened, and only with the exact
# authored/registered names the caller vouches for.


class AllowlistSourceError(RuntimeError):
    """The judge registry could not be read to re-source the dynamic allowlist.

    Raised by :func:`sourced_judge_names` when the injected lister fails. It is a
    **fail-closed** signal: the caller must refuse (or fall back to a bounded,
    already-trusted name set) rather than admit every possible judge name. The
    allowlist never downgrades an unreadable registry to "allow everything".
    """


#: A callable that lists the scorers registered on an experiment — the live
#: re-sourcing seam. ``ail.judges.registration.list_registered_scorers`` satisfies
#: it in production; tests inject a fake. Each returned object need only expose a
#: ``name`` attribute.
ScorerLister = Callable[[str], Iterable[Any]]

#: The ambient extra judge names admitted on top of :data:`JUDGE_METRICS`. Empty by
#: default (static behaviour); widened only within :func:`judge_allowlist`.
_DYNAMIC_JUDGES: ContextVar[frozenset[str]] = ContextVar(
    "ail_dynamic_judge_allowlist", default=frozenset()
)


def is_l0_metric(name: str) -> bool:
    """Whether ``name`` is a known L0 deterministic metric."""
    return name in _L0_SET


def is_judge(name: str) -> bool:
    """Whether ``name`` is an admitted L2 judge (a *quality* signal).

    A judge is admitted if it is a built-in scorer (:data:`JUDGE_METRICS`) **or**
    it is in the ambient dynamic allowlist set by :func:`judge_allowlist` (an
    authored/registered judge the caller vouches for). With no dynamic context this
    is exactly ``name in JUDGE_METRICS`` — the static behaviour.
    """
    return name in JUDGE_METRICS or name in _DYNAMIC_JUDGES.get()


def is_known_metric(name: str) -> bool:
    """Whether ``name`` is anywhere in the allowlist (L0 metric or judge).

    Consults the dynamic judge set via :func:`is_judge`, so an authored dimension
    validates inside a :func:`judge_allowlist` block; identical to
    ``name in ALLOWLIST`` when no dynamic context is active.
    """
    return is_l0_metric(name) or is_judge(name)


def dynamic_judge_names() -> frozenset[str]:
    """The judge names currently admitted *beyond* the built-in :data:`JUDGE_METRICS`."""
    return _DYNAMIC_JUDGES.get()


@contextlib.contextmanager
def judge_allowlist(extra_judge_names: Iterable[str]) -> Iterator[None]:
    """Temporarily admit ``extra_judge_names`` as valid judges within the block.

    Widens :func:`is_judge` / :func:`is_known_metric` for the duration of the
    ``with`` block only — used to construct or reconstruct a :class:`CompiledGoal`
    that references an authored/registered judge not in the static built-in set.
    The extra names union with (never replace) any already-active dynamic set, and
    are reset on exit, so the widening is scoped and cannot leak. Blank names are
    ignored.
    """
    extra = frozenset(n for n in extra_judge_names if n and n.strip())
    token = _DYNAMIC_JUDGES.set(_DYNAMIC_JUDGES.get() | extra)
    try:
        yield
    finally:
        _DYNAMIC_JUDGES.reset(token)


def sourced_judge_names(lister: ScorerLister, *, experiment_id: str) -> frozenset[str]:
    """Re-source the judge allowlist from the live scorer registry, fail-closed.

    Reads the scorers registered on ``experiment_id`` through the injected
    ``lister`` and returns the built-in judges (:data:`JUDGE_METRICS`) unioned with
    the registered names — the set to feed :func:`judge_allowlist` so an
    authored/registered dimension validates.

    Fail-closed: if the lister raises (registry unreadable — auth, network, or a
    backend error) this raises :class:`AllowlistSourceError` rather than returning a
    permissive set. It never "allows everything"; a read that cannot be certified is
    surfaced so the caller refuses or falls back to a bounded, already-trusted set.
    """
    try:
        scorers = list(lister(experiment_id))
    except Exception as exc:  # noqa: BLE001 - any read failure is fail-closed
        raise AllowlistSourceError(
            f"could not re-source the judge allowlist from the scorer registry for "
            f"experiment {experiment_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    registered = frozenset(str(s.name) for s in scorers if getattr(s, "name", None))
    return JUDGE_METRICS | registered
