"""Per-cohort L0 metrics: an additive wrapper over :func:`compute_l0`.

The Wave 0 L0 engine (:mod:`ail.metrics.l0_deterministic`) computes
token/cost/latency/redundancy metrics over a *set* of normalized traces. When
one experiment holds several agents or deployments (see :mod:`ail.cohorts`),
those numbers are only meaningful *per cohort* — the big-token tail of agent A
should not inflate agent B's median.

This module adds exactly that, and nothing more: it **selects** the traces in a
cohort and feeds them to the existing :func:`compute_l0`. The metrics engine and
its output contract (:class:`~ail.metrics.contract.L0MetricsReport`) are
unchanged — same shapes, same fields. There is no per-cohort metric type to
learn; a cohort report *is* an :class:`L0MetricsReport` over a filtered subset.

A cohort that matches **zero** traces yields a valid empty report
(``n_traces == 0``) rather than an error — the *collecting / not-ready* state. A
future per-cohort readiness module can read that directly; computing readiness
itself is out of scope here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ail.cohorts import Cohort
from ail.ingest.base import NormalizedTrace
from ail.metrics.contract import L0MetricsReport, PriceBookEntry
from ail.metrics.l0_deterministic import compute_l0

__all__ = ["compute_cohort_l0", "compute_l0_by_cohort"]


def compute_cohort_l0(
    traces: Iterable[NormalizedTrace],
    cohort: Cohort,
    *,
    pricebook: dict[str, PriceBookEntry] | None = None,
    experiment_id: str | None = None,
    generated_at: str | None = None,
    top_repeats: int = 20,
) -> L0MetricsReport:
    """Compute the L0 report over just the traces belonging to ``cohort``.

    A thin wrapper: ``cohort.select(traces)`` then :func:`compute_l0`. All
    keyword arguments pass straight through. When the cohort selects no traces,
    the returned report has ``n_traces == 0`` (the collecting / not-ready state).
    """
    selected = cohort.select(traces)
    return compute_l0(
        selected,
        pricebook=pricebook,
        experiment_id=experiment_id,
        generated_at=generated_at,
        top_repeats=top_repeats,
    )


def compute_l0_by_cohort(
    traces: Iterable[NormalizedTrace],
    cohorts: Sequence[Cohort],
    *,
    pricebook: dict[str, PriceBookEntry] | None = None,
    experiment_id: str | None = None,
    generated_at: str | None = None,
    top_repeats: int = 20,
) -> dict[str, L0MetricsReport]:
    """Compute one L0 report per cohort, keyed by :attr:`Cohort.name`.

    The traces are materialized once and each cohort is evaluated against the
    same set, so a trace may appear in more than one cohort (cohorts overlap
    freely — they are not the disjoint evaluation pools of :mod:`ail.pools`).
    Cohort names must be unique, since they are the result keys.
    """
    names = [cohort.name for cohort in cohorts]
    if len(set(names)) != len(names):
        raise ValueError(f"cohort names must be unique; got {names}")
    materialized = list(traces)
    return {
        cohort.name: compute_cohort_l0(
            materialized,
            cohort,
            pricebook=pricebook,
            experiment_id=experiment_id,
            generated_at=generated_at,
            top_repeats=top_repeats,
        )
        for cohort in cohorts
    }
