"""Provenance wall: the advisory memory must not be distilled from the eval suite.

The frozen Task Suite is a held-out benchmark (``docs/ARCHITECTURE.md`` §2,
:mod:`ail.pools`): if a "learning" the candidate is handed were distilled from
one of the very traces the eval reconstructs, the spike would be teaching to the
test — the memory would encode the answer rather than a transferable lesson.

The RLM findings come from **organic** production traces, so this holds by
construction; this module makes it an **explicit, tested** guard rather than an
assumption. It reuses the :mod:`ail.pools` disjointness vocabulary
(:class:`~ail.pools.Pool`, :class:`~ail.pools.PoolOverlapError`): the memory's
provenance ids (the traces each :class:`~ail.l3.contract.RankedAsset` was drawn
from) must be disjoint from the suite's ids (task ids **and** each task's
``source_trace_id``).
"""

from __future__ import annotations

from collections.abc import Iterable

from ail.l3.contract import RankedAsset
from ail.pools import Pool, PoolOverlapError
from ail.task_suite.schema import TaskSuite

__all__ = [
    "memory_provenance_ids",
    "task_suite_ids",
    "assert_memory_disjoint_from_suite",
]


def memory_provenance_ids(assets: Iterable[RankedAsset]) -> frozenset[str]:
    """The set of trace ids the ranked assets were distilled from."""
    ids: set[str] = set()
    for asset in assets:
        ids.update(asset.trace_ids)
    return frozenset(ids)


def task_suite_ids(suite: TaskSuite) -> frozenset[str]:
    """Every id the frozen suite occupies: task ids and their ``source_trace_id``s.

    Both are included because a leak could hide in either namespace — the memory
    could name a task id directly, or the organic trace a task was reconstructed
    from.
    """
    ids: set[str] = set()
    for task in suite.tasks:
        ids.add(task.task_id)
        ids.add(task.source_trace_id)
    return frozenset(ids)


def assert_memory_disjoint_from_suite(*, assets: Iterable[RankedAsset], suite: TaskSuite) -> None:
    """Prove the memory source shares no id with the frozen suite, or raise.

    Raises:
        PoolOverlapError: if any memory-provenance trace id coincides with a
            frozen-suite task id or ``source_trace_id`` — the teaching-to-the-test
            leak the frozen evaluation wall forbids.
    """
    overlap = memory_provenance_ids(assets) & task_suite_ids(suite)
    if overlap:
        shown = sorted(overlap)[:5]
        raise PoolOverlapError(
            f"advisory-memory provenance overlaps the {Pool.TASK_SUITE.value!r} pool: "
            f"{len(overlap)} shared id(s), e.g. {shown}. The frozen evaluation wall forbids "
            "distilling a learning from a trace the eval suite is built on (teaching to the test)."
        )
