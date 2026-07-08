"""Provenance wall: advisory memory must never be distilled from the eval suite.

The frozen Task Suite and Human-Anchor pools are held-out benchmarks
(docs/ARCHITECTURE.md section 2, ail.pools). If a "learning" the candidate is
handed were distilled from one of the very traces the eval reconstructs, the
spike would be teaching to the test and the memory would encode the answer
rather than a transferable lesson. Advisory memory is read back into the agent,
so a guideline distilled from a Task-Suite or Human-Anchor trace would leak the
eval set into the agent's context -- exactly the co-adaptation the frozen wall
exists to prevent. Feedback does land on those traces (the L2 judges score them
too), so without this wall the distiller would happily mint memory from them.

This module is that wall. The read/spike side (memory_provenance_ids,
task_suite_ids, assert_memory_disjoint_from_suite) proves a memory source shares
no id with the frozen suite, reusing the ail.pools disjointness vocabulary
(Pool, PoolOverlapError). The write/distiller side (partition_rows) splits
candidate memory rows into clean and dropped, and the clean set is proven
disjoint from the reserved pools with ail.pools.assert_pools_disjoint -- the same
guard the loop controller uses -- so a regression in the drop logic fails closed
(raises) rather than silently leaking. A dropped row is recorded with a reason
and never written.

Truncated ids: the frozen Task-Suite artifact stores some source_trace_id values
as short (12-char) prefixes, while assessment target_id values are full 32-char
ids. Exact-set intersection would miss those, so partition_rows drops on an exact
match or a shared prefix of at least _MIN_PREFIX_LEN chars -- the strictly safer
(fail-closed toward dropping) reading. The exact subset is what the
assert_pools_disjoint verification then re-checks.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from ail.l3.contract import RankedAsset
from ail.memory.schema import MemoryRow
from ail.pools import (
    AlignmentSet,
    AnchorItem,
    HumanAnchor,
    Pool,
    PoolOverlapError,
    assert_pools_disjoint,
)
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


#: The shortest id-overlap treated as a match. The frozen suite's truncated ids are
#: 12 chars; 12 keeps a full-id prefix collision astronomically unlikely while still
#: catching a truncated reserved id against a full candidate id.
_MIN_PREFIX_LEN = 12

#: Default artifact directory for the frozen Task Suite (see :mod:`ail.task_suite`).
DEFAULT_TASK_SUITE_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class ReservedPools:
    """The frozen trace ids that must never seed memory, by pool.

    ``task_suite_ids`` are the ``source_trace_id`` of the frozen Task Suite;
    ``human_anchor_ids`` are the Human-Anchor pool's case ids. Both are compared
    against a candidate row's ``source_trace_ids`` by :func:`partition_rows`.
    """

    task_suite_ids: frozenset[str] = field(default_factory=frozenset)
    human_anchor_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def all_ids(self) -> frozenset[str]:
        return self.task_suite_ids | self.human_anchor_ids

    def pool_of(self, reserved_id: str) -> Pool:
        """Which reserved pool ``reserved_id`` belongs to (Task Suite wins ties)."""
        if reserved_id in self.task_suite_ids:
            return Pool.TASK_SUITE
        return Pool.HUMAN_ANCHOR


@dataclass(frozen=True, slots=True)
class DroppedRow:
    """A candidate memory row the wall rejected, with the reason it was dropped."""

    row: MemoryRow
    reason: str


@dataclass(frozen=True, slots=True)
class Partition:
    """The result of :func:`partition_rows`: rows to write and rows dropped."""

    kept: tuple[MemoryRow, ...]
    dropped: tuple[DroppedRow, ...]


def _overlap(trace_id: str, reserved_ids: frozenset[str]) -> str | None:
    """The reserved id ``trace_id`` collides with (exact or ≥12-char prefix), or None."""
    if trace_id in reserved_ids:
        return trace_id
    for rid in reserved_ids:
        if len(rid) >= _MIN_PREFIX_LEN and trace_id.startswith(rid):
            return rid
        if len(trace_id) >= _MIN_PREFIX_LEN and rid.startswith(trace_id):
            return rid
    return None


def _trace_refs(ids: Iterable[str]) -> list[SimpleNamespace]:
    """Wrap bare ids as minimal trace-shaped objects for :class:`AlignmentSet`.

    :func:`ail.pools._trace_id` reads ``trace.info.trace_id``; a ``SimpleNamespace``
    with that attribute is the lightest object that resolves.
    """
    return [SimpleNamespace(info=SimpleNamespace(trace_id=tid)) for tid in ids]


def _shares_reserved_prefix(trace_id: str, reserved_ids: frozenset[str]) -> str | None:
    """The reserved id ``trace_id`` shares a ≥``_MIN_PREFIX_LEN``-char prefix with, or None.

    Deliberately re-implemented independently of :func:`_overlap` so the re-verification
    gate (:func:`_assert_kept_disjoint`) catches a regression IN the drop pass — a test
    (or bug) that neuters ``_overlap`` must not also neuter the guard that proves the
    drop worked.
    """
    for rid in reserved_ids:
        n = min(len(trace_id), len(rid))
        if n >= _MIN_PREFIX_LEN and trace_id[:n] == rid[:n]:
            return rid
    return None


def _assert_kept_disjoint(kept_ids: set[str], reserved: ReservedPools) -> None:
    """Prove the kept ids share no id with the reserved pools, or raise (fail-closed).

    Two independent checks, so a regression in the drop pass RAISES:

    * the canonical **exact** guard reused from :mod:`ail.pools`
      (:func:`assert_pools_disjoint`), and
    * a memory-specific **prefix** guard (:func:`_shares_reserved_prefix`) that catches
      a full candidate id sharing a ≥12-char prefix with a *truncated* reserved id —
      which the exact set-intersection guard cannot see.
    """
    assert_pools_disjoint(
        alignment_set=AlignmentSet.of(_trace_refs(kept_ids)),
        human_anchor=HumanAnchor.of(
            AnchorItem(item_id=hid, human_label="reserved") for hid in reserved.human_anchor_ids
        ),
        task_suite_ids=reserved.task_suite_ids,
    )
    for tid in kept_ids:
        rid = _shares_reserved_prefix(tid, reserved.all_ids)
        if rid is not None:
            raise PoolOverlapError(
                f"kept memory row trace id {tid!r} shares a >= {_MIN_PREFIX_LEN}-char prefix "
                f"with reserved pool id {rid!r} ({reserved.pool_of(rid).value}); the "
                "provenance drop logic regressed — refusing to write eval-derived memory."
            )


def partition_rows(rows: Iterable[MemoryRow], reserved: ReservedPools) -> Partition:
    """Split ``rows`` into clean (kept) and reserved-overlapping (dropped).

    A row is dropped if any of its ``source_trace_ids`` collides
    (:func:`_overlap`) with a reserved Task-Suite or Human-Anchor id; the reason
    names the offending trace id, the reserved id, and the pool. The kept set is
    then proven disjoint from the reserved pools with
    :func:`ail.pools.assert_pools_disjoint` (the canonical guard) — so if the drop
    logic ever regresses this raises rather than leaking eval-derived memory.
    """
    kept: list[MemoryRow] = []
    dropped: list[DroppedRow] = []
    reserved_all = reserved.all_ids

    for row in rows:
        hits: list[str] = []
        for tid in row.source_trace_ids:
            rid = _overlap(tid, reserved_all)
            if rid is not None:
                hits.append(f"{tid} ~ reserved {rid} ({reserved.pool_of(rid).value})")
        if hits:
            dropped.append(
                DroppedRow(
                    row=row,
                    reason="provenance wall: source_trace_ids overlap a frozen pool — "
                    + "; ".join(hits),
                )
            )
        else:
            kept.append(row)

    # Canonical re-verification (fail-closed): prove the kept set shares no id with
    # either reserved pool, by BOTH the shared exact guard and an independent prefix
    # guard, so a regression in the drop pass above RAISES rather than leaking.
    kept_ids = {tid for row in kept for tid in row.source_trace_ids}
    _assert_kept_disjoint(kept_ids, reserved)
    return Partition(kept=tuple(kept), dropped=tuple(dropped))


# ---------------------------------------------------------------------------
# Runtime resolution of the reserved pools (fail-closed on the Task Suite)
# ---------------------------------------------------------------------------


def _packaged_eval_root() -> str | None:
    """The wheel-bundled ``eval`` root (``ail/_eval_bundle``), or None if absent.

    The frozen Task Suite is force-included into the wheel under
    ``ail/_eval_bundle/eval/task_suite`` (see ``pyproject.toml``) so a serverless
    Job — where ``eval/`` is not otherwise on disk — can still load it.
    """
    import ail

    pkg_dir = Path(ail.__file__).resolve().parent
    root = pkg_dir / "_eval_bundle"
    return str(root) if (root / "eval" / "task_suite").is_dir() else None


def _task_suite_ids(version: str, root: str | None) -> set[str]:
    """The frozen Task Suite's ``source_trace_id`` set; raises if unavailable.

    Root precedence: explicit ``root`` > ``AIL_TASK_SUITE_ROOT`` env >
    wheel-bundled eval root > :func:`ail.task_suite.loader.load_task_suite`'s own
    upward search (editable installs). Fail-closed: any load/integrity failure
    propagates so the distiller writes nothing rather than distilling without a wall.
    """
    from ail.task_suite.loader import load_task_suite

    resolved_root = root or os.environ.get("AIL_TASK_SUITE_ROOT") or _packaged_eval_root()
    suite = load_task_suite(version, root=resolved_root)
    return {t.source_trace_id for t in suite.tasks if t.source_trace_id}


def _human_anchor_ids(groundtruth_root: str | None) -> set[str]:
    """Human-Anchor case ids from a ground-truth store, or ``set()`` if none configured.

    The JSON ground-truth store is not present on a serverless Job unless a root is
    provided (a synced volume/workspace path). When unset, the Human-Anchor pool is
    treated as empty — honest for a workspace with no promoted anchor cases — and
    the wall still fully enforces the Task Suite.
    """
    if not groundtruth_root:
        return set()
    from ail.groundtruth.store import JsonGroundTruthStore

    store = JsonGroundTruthStore(groundtruth_root)
    return set(store.load(Pool.HUMAN_ANCHOR).case_ids())


def resolve_reserved_pools(
    *,
    task_suite_version: str = DEFAULT_TASK_SUITE_VERSION,
    task_suite_root: str | None = None,
    groundtruth_root: str | None = None,
) -> ReservedPools:
    """Assemble :class:`ReservedPools` from the frozen artifacts, fail-closed.

    The Task Suite is required (raises if it cannot be loaded — no wall, no write);
    the Human Anchor is optional (empty when no store is configured). Called once by
    the distiller before any memory is written.
    """
    return ReservedPools(
        task_suite_ids=frozenset(_task_suite_ids(task_suite_version, task_suite_root)),
        human_anchor_ids=frozenset(_human_anchor_ids(groundtruth_root)),
    )
