"""Reconcile the arrival-triggered RLM job's watched-table list with the registry.

The standalone continuous-RLM job (``resources/continuous_rlm.job.yml``) wakes on a
``table_update`` trigger over a *fixed* list of ``*_otel_spans`` UC Delta tables. The
job body itself is registry-driven (it reviews every agent in ``agent_registry``), but
the trigger's ``table_names`` is static — so a newly onboarded agent, whose traces land
in its OWN ``{prefix}_otel_spans`` table, never *wakes the job* on its own arrivals.
The job would only review that agent when some OTHER watched table updates.

This module closes that gap. It derives each registered agent's spans table from the
``annotations_table`` the registry already persists (the two share a prefix and differ
only in the ``_otel_annotations`` / ``_otel_spans`` suffix — the MLflow Traces-in-UC
naming convention, see :class:`ail.onboarding.experiment.UcTraceLocation`), then unions
those tables into the trigger's ``table_names`` **add-only** and issues a single partial
``jobs.update`` when (and only when) the set actually grew.

Two callers share this one helper:

* **Onboarding** (:func:`ail.onboarding.service.run_register`) reconciles the trigger
  for the just-registered agent, so arrival-triggering works the moment onboarding
  finishes — no redeploy required.
* **Deploy** (:func:`ail.jobs.bootstrap_grants.bootstrap`) reconciles for the WHOLE
  registry after each ``bundle deploy``, because the DAB-managed trigger reverts to its
  YAML (single-table) value on every deploy. The deploy heal re-adds every onboarded
  agent's table, so the fix is durable across redeploys rather than self-reverting.

Design invariants (do NOT weaken):

* **Add-only.** A table already in the trigger is never removed, and the reconcile
  never drops a table it cannot attribute to a current agent (a hand-added table, or
  an agent since removed from the registry, stays watched). Removing a table is an
  operator decision, not a side effect of onboarding.
* **No-op when unchanged.** If every derived spans table is already watched, NO
  ``jobs.update`` is issued — a redeploy that changed nothing stays quiet.
* **Fail-soft at the call sites, not here.** This function raises on a real Jobs API
  failure (so the deploy step can log it); the callers decide whether that is fatal.
  Onboarding treats a reconcile failure as non-fatal (the agent IS registered and the
  four cron jobs already cover it; only RLM arrival-latency is affected), so it never
  turns a successful registration into an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ail.registry import Agent

__all__ = [
    "ReconcileResult",
    "spans_table_for_agent",
    "reconcile_rlm_trigger_tables",
]

#: The registry persists each agent's annotations table; the spans table the trigger
#: watches is the same fully-qualified prefix with this suffix swapped.
_ANNOTATIONS_SUFFIX = "_otel_annotations"
_SPANS_SUFFIX = "_otel_spans"


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """What a reconcile pass resolved/did, for the caller's logs and tests.

    ``updated`` is ``True`` iff a ``jobs.update`` was actually issued (the watched set
    grew). ``added`` lists the spans tables newly added this pass; ``already_watched``
    lists the derived tables that were present already; ``underivable`` names the agents
    whose spans table could not be derived (no ``annotations_table`` on the registry
    row) — reported honestly rather than silently skipped.
    """

    rlm_job_id: int
    updated: bool
    added: list[str] = field(default_factory=list)
    already_watched: list[str] = field(default_factory=list)
    underivable: list[str] = field(default_factory=list)
    #: The full watched list after the reconcile (for logging / assertions).
    watched: list[str] = field(default_factory=list)


def spans_table_for_agent(agent: Agent) -> str | None:
    """Derive an agent's ``*_otel_spans`` table from its persisted ``annotations_table``.

    Returns the fully-qualified spans table (``catalog.schema.{prefix}_otel_spans``), or
    ``None`` when it cannot be derived without guessing:

    * the agent carries no ``annotations_table`` (a registered-but-not-fully-functional
      row — never fabricate a table name for it); or
    * the ``annotations_table`` does not end in the expected ``_otel_annotations``
      suffix (an unexpected shape — fail closed rather than emit a bogus watch target).

    The suffix swap is deliberately literal: both names come from the SAME
    :class:`ail.onboarding.experiment.UcTraceLocation` prefix, so a spans table is the
    annotations table with exactly this one suffix replaced.
    """
    table = (agent.annotations_table or "").strip()
    if not table or not table.endswith(_ANNOTATIONS_SUFFIX):
        return None
    return table[: -len(_ANNOTATIONS_SUFFIX)] + _SPANS_SUFFIX


def _trigger_table_names(settings: Any) -> list[str]:
    """The current ``trigger.table_update.table_names`` as a list (``[]`` if absent)."""
    trigger = getattr(settings, "trigger", None)
    table_update = getattr(trigger, "table_update", None) if trigger is not None else None
    names = getattr(table_update, "table_names", None) if table_update is not None else None
    return [str(n) for n in (names or [])]


def reconcile_rlm_trigger_tables(
    client: Any,
    *,
    rlm_job_id: int,
    agents: Sequence[Agent],
) -> ReconcileResult:
    """Union every agent's spans table into the RLM job's ``table_update`` trigger (add-only).

    Reads the job via ``client.jobs.get``, derives each agent's spans table
    (:func:`spans_table_for_agent`), and adds any not already watched. Issues exactly
    one partial ``client.jobs.update`` — mutating ONLY ``trigger.table_update.table_names``
    on the existing :class:`~databricks.sdk.service.jobs.JobSettings`, so every other
    setting (schedule absence, queue, tasks, debounce seconds) is preserved — and only
    when the set actually grew. When nothing new is derivable or everything is already
    watched, it returns ``updated=False`` and makes no write.

    Requires the job to already have a ``table_update`` trigger (the deployed RLM job
    does). If the job has no such trigger, this raises :class:`ValueError` rather than
    inventing one — reshaping a job's trigger kind is a bundle/deploy concern, not a
    runtime reconcile.

    Raises on a real Jobs API error (get/update). Call sites that must not fail their
    primary work (onboarding) wrap this call and log the failure instead.
    """
    job = client.jobs.get(rlm_job_id)
    settings = getattr(job, "settings", None)
    if settings is None:
        raise ValueError(f"RLM job {rlm_job_id} returned no settings; cannot reconcile trigger")

    trigger = getattr(settings, "trigger", None)
    table_update = getattr(trigger, "table_update", None) if trigger is not None else None
    if table_update is None:
        raise ValueError(
            f"RLM job {rlm_job_id} has no table_update trigger to reconcile; its trigger "
            "kind must be set by the bundle (resources/continuous_rlm.job.yml), not at runtime"
        )

    current = _trigger_table_names(settings)
    watched = list(current)
    watched_set = set(current)

    added: list[str] = []
    already_watched: list[str] = []
    underivable: list[str] = []
    for agent in agents:
        spans = spans_table_for_agent(agent)
        if spans is None:
            underivable.append(agent.agent_name)
            continue
        if spans in watched_set:
            already_watched.append(spans)
            continue
        watched.append(spans)
        watched_set.add(spans)
        added.append(spans)

    if not added:
        return ReconcileResult(
            rlm_job_id=rlm_job_id,
            updated=False,
            added=[],
            already_watched=already_watched,
            underivable=underivable,
            watched=watched,
        )

    # Partial update: mutate ONLY the watched-table list on the existing settings and
    # send them back. jobs.update merges new_settings, so preserving the rest of the
    # object (schedule=None, queue, tasks, debounce) keeps every other setting intact.
    table_update.table_names = watched
    client.jobs.update(rlm_job_id, new_settings=settings)

    return ReconcileResult(
        rlm_job_id=rlm_job_id,
        updated=True,
        added=added,
        already_watched=already_watched,
        underivable=underivable,
        watched=watched,
    )
