"""The frozen evaluation wall, as one canonical module: the three disjoint pools.

``docs/ARCHITECTURE.md`` §2 names the single non-negotiable principle of this
system: if the agent is optimized against a judge *and* the judge is aligned
against feedback drawn from the same loop, the two co-adapt — scores climb while
real quality stalls. The defence is three **disjoint** data pools that are never
mixed:

* **Task Suite** — fixed tasks re-run to compare agent versions. *Never* fed to
  the optimizer or the judge-alignment step. (Frozen/curated in Wave 1b by
  :mod:`ail.task_suite`; this module only models its *identity* for disjointness
  checks.)
* **Alignment Set** — labeled traces used to align judges (MemAlign). The
  **only** pool :func:`ail.judges.alignment.align_judge` will consume.
* **Human Anchor** — a small human-labeled slice used to measure judge-vs-human
  agreement (:mod:`ail.judges.agreement`). Never fed to the optimizer or to
  alignment.

This is the *one* source of truth for "which pool". Both the ground-truth
storage layer (:mod:`ail.groundtruth`, which curates and stores cases per pool)
and the judge layer (:mod:`ail.judges`, which consumes the pools to align and
audit judges) import :class:`Pool` from here, so the vocabulary can never drift
into two structurally-equal copies.

The module also makes "never mixed" a property of the **types**, not a
convention. Alignment takes an :class:`AlignmentSet`; agreement takes a
:class:`HumanAnchor`; neither accepts the other. :func:`assert_pools_disjoint`
is the explicit guard the loop controller calls before a cadence to prove no
trace leaked across the wall.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

__all__ = [
    "Pool",
    "PoolOverlapError",
    "UnresolvedTraceIdError",
    "ScoreValue",
    "AnchorItem",
    "HumanAnchor",
    "AlignmentSet",
    "assert_pools_disjoint",
]


class Pool(StrEnum):
    """The three disjoint pools of the frozen evaluation wall.

    The string values are stable identifiers safe to persist on disk, record on
    a report, or tag a trace with. A case or trace lives in exactly one pool;
    :func:`assert_pools_disjoint` is the runtime guard that proves no id leaked
    across the wall.
    """

    TASK_SUITE = "task_suite"
    ALIGNMENT_SET = "alignment_set"
    HUMAN_ANCHOR = "human_anchor"


#: The value space of a judge score or a human label. Categorical labels
#: (``"yes"``/``"no"``, ``"pass"``/``"fail"``) are ``str``; pass/fail guardrails
#: are ``bool``; graded rubrics are ``int`` (e.g. 1–5); continuous scores are
#: ``float``. Agreement (:mod:`ail.judges.agreement`) compares values in this
#: space.
ScoreValue = bool | int | float | str


class PoolOverlapError(ValueError):
    """Raised when the disjointness guard cannot certify the wall is intact.

    The common case is a non-empty intersection between two pools (the Alignment
    Set and the Human Anchor, or either of those and the Task Suite) — exactly
    the co-adaptation leak the frozen wall exists to prevent, so it is an error,
    never a warning. :class:`UnresolvedTraceIdError` is the fail-closed subtype
    for the "cannot even check" case.
    """


class UnresolvedTraceIdError(PoolOverlapError):
    """Raised when a pool carries a trace whose id cannot be resolved.

    Disjointness is *proven* by comparing ids; a trace with no resolvable id
    cannot be proven disjoint from any other pool. Silently dropping it (the
    earlier behaviour) would let an unidentifiable trace sit in two pools
    undetected — the precise leak the wall guards against — so the guard **fails
    closed** and raises instead. A subtype of :class:`PoolOverlapError` so a
    caller that already catches the overlap error catches this too.
    """


@dataclass(frozen=True, slots=True)
class AnchorItem:
    """One human-labeled item on the Human Anchor: an evaluable plus its gold.

    ``item_id`` identifies the item across pools (use the trace id when the item
    is derived from a trace, so a disjointness check against trace-keyed pools is
    meaningful). ``inputs``/``outputs``/``expectations`` are passed to a judge's
    ``__call__`` verbatim; ``human_label`` is the gold value the judge's score is
    measured against — it is **never** shown to the judge.
    """

    item_id: str
    human_label: ScoreValue
    inputs: Any = None
    outputs: Any = None
    expectations: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class HumanAnchor:
    """A human-labeled slice for judge-vs-human agreement.

    This pool measures whether a judge still agrees with people; it is **never**
    fed to the optimizer or to MemAlign. Item ids must be unique within the
    slice (a duplicate id would double-count an item in the agreement rate).
    """

    pool: ClassVar[Pool] = Pool.HUMAN_ANCHOR

    items: tuple[AnchorItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ids = [item.item_id for item in self.items]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"HumanAnchor has duplicate item_id(s): {dupes}")

    @classmethod
    def of(cls, items: Iterable[AnchorItem]) -> HumanAnchor:
        """Build a :class:`HumanAnchor` from any iterable of items."""
        return cls(items=tuple(items))

    @property
    def ids(self) -> frozenset[str]:
        """The set of item ids in this slice."""
        return frozenset(item.item_id for item in self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class AlignmentSet:
    """Labeled traces used to align a judge with MemAlign.

    Wraps the raw MLflow ``Trace`` objects (each carrying the human assessments
    MemAlign learns from) and is the **only** pool
    :func:`ail.judges.alignment.align_judge` accepts. Disjoint from the Task
    Suite and the Human Anchor by construction of the wall.

    ``traces`` holds opaque MLflow trace objects (typed ``Any`` so this module
    stays import-light and producer-agnostic, like the rest of the ingest seam);
    :attr:`ids` reads each trace's id defensively.
    """

    pool: ClassVar[Pool] = Pool.ALIGNMENT_SET

    traces: tuple[Any, ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, traces: Iterable[Any]) -> AlignmentSet:
        """Build an :class:`AlignmentSet` from any iterable of MLflow traces."""
        return cls(traces=tuple(traces))

    @property
    def ids(self) -> frozenset[str]:
        """Resolvable trace ids in this set (via ``trace.info.trace_id``).

        A trace with no resolvable id is **not** silently included here; it is
        surfaced by :attr:`unresolved_count` so the disjointness guard can fail
        closed rather than under-count the wall.
        """
        return frozenset(tid for tid in (_trace_id(t) for t in self.traces) if tid is not None)

    @property
    def unresolved_count(self) -> int:
        """How many traces have no resolvable id (the fail-closed signal)."""
        return sum(1 for t in self.traces if _trace_id(t) is None)

    def __len__(self) -> int:
        return len(self.traces)


def _trace_id(trace: Any) -> str | None:
    """Best-effort id of an MLflow trace (``info.trace_id``/``request_id``).

    Returns ``None`` when no id can be resolved. Unlike the earlier behaviour,
    that ``None`` is **not** quietly dropped from the disjointness check:
    :func:`assert_pools_disjoint` treats an unresolvable id as a fail-closed
    error, because a trace that cannot be identified cannot be proven disjoint.
    """
    info = getattr(trace, "info", None)
    for attr in ("trace_id", "request_id"):
        value = getattr(info, attr, None) if info is not None else None
        if value:
            return str(value)
    return None


def assert_pools_disjoint(
    *,
    alignment_set: AlignmentSet | None = None,
    human_anchor: HumanAnchor | None = None,
    task_suite_ids: Sequence[str] | Iterable[str] | None = None,
) -> None:
    """Prove the supplied pools share no id, or raise :class:`PoolOverlapError`.

    The loop controller calls this before an alignment or agreement cadence so a
    trace that leaked across the wall is caught loudly rather than silently
    co-adapting the judge and the agent. Only the pools passed are checked, so it
    is usable with whichever subset a given cadence touches; the Task Suite is
    supplied as a bare id set because its storage is owned by :mod:`ail.task_suite`.

    Fails closed: if any Alignment-Set trace has no resolvable id, disjointness
    cannot be proven, so it raises :class:`UnresolvedTraceIdError` rather than
    dropping the trace from the comparison.

    Raises:
        UnresolvedTraceIdError: If a checked Alignment Set carries a trace whose
            id cannot be resolved.
        PoolOverlapError: If any two checked pools intersect.
    """
    named: list[tuple[Pool, frozenset[str]]] = []
    if alignment_set is not None:
        if alignment_set.unresolved_count:
            raise UnresolvedTraceIdError(
                f"the Alignment Set has {alignment_set.unresolved_count} trace(s) with no "
                "resolvable id; disjointness cannot be proven, so the wall fails closed. "
                "Give every trace an info.trace_id (or request_id) before checking pools."
            )
        named.append((Pool.ALIGNMENT_SET, alignment_set.ids))
    if human_anchor is not None:
        named.append((Pool.HUMAN_ANCHOR, human_anchor.ids))
    if task_suite_ids is not None:
        named.append((Pool.TASK_SUITE, frozenset(str(i) for i in task_suite_ids)))

    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            (pool_a, ids_a), (pool_b, ids_b) = named[i], named[j]
            overlap = ids_a & ids_b
            if overlap:
                shown = sorted(overlap)[:5]
                raise PoolOverlapError(
                    f"pools {pool_a.value!r} and {pool_b.value!r} are not disjoint: "
                    f"{len(overlap)} shared id(s), e.g. {shown}. The frozen evaluation "
                    "wall forbids a trace from appearing in more than one pool."
                )
