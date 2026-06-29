"""Persistence for the frozen pools, plus the review-queue round-trip.

A :class:`GroundTruthStore` keeps each :class:`~ail.groundtruth.schema.Pool` in
its own physically separate store, which is half of how "never mix pools" is
enforced (the other half is the promoter checking a case id is not already in a
*different* pool — see :func:`ail.groundtruth.promote.promote_approved`).

:class:`JsonGroundTruthStore` is the default: one JSON file per pool under a
root directory. It is dependency-free and offline, so the whole
capture -> approve -> promote round-trip is testable in CI without a workspace.

The review-queue helpers (:func:`dump_cases` / :func:`load_cases`) serialize the
*candidate* cases a human edits between :mod:`~ail.groundtruth.capture` /
:mod:`~ail.groundtruth.execute` and :mod:`~ail.groundtruth.approve`. They are a
plain list of cases on disk, deliberately **not** a pool (an unreviewed
candidate has no business in a frozen pool).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from ail.groundtruth.schema import GroundTruthCase, GroundTruthSet, Pool

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "GroundTruthStore",
    "JsonGroundTruthStore",
    "dump_cases",
    "load_cases",
    "write_review_queue",
    "read_review_queue",
]


class GroundTruthStore(ABC):
    """Read/write frozen pools, one disjoint store per pool.

    Implementations persist a :class:`GroundTruthSet` per :class:`Pool`. The
    base class provides :meth:`case_pool_index`, the cross-pool lookup the
    promoter uses to refuse putting one case id into two pools.
    """

    @abstractmethod
    def load(self, pool: Pool) -> GroundTruthSet:
        """Load a pool's set. Returns an empty set if the pool has none yet."""
        raise NotImplementedError

    @abstractmethod
    def save(self, gt_set: GroundTruthSet) -> None:
        """Persist a pool's set, replacing whatever was stored for that pool."""
        raise NotImplementedError

    def case_pool_index(self) -> dict[str, Pool]:
        """Map every stored case id to the pool it lives in, across all pools.

        Used to detect (and refuse) a case id that would otherwise end up in two
        pools. The same case id appearing twice in *different* pools is a
        wall-integrity bug, so this raises rather than silently picking one.
        """
        index: dict[str, Pool] = {}
        for pool in Pool:
            for case in self.load(pool).cases:
                existing = index.get(case.case_id)
                if existing is not None and existing is not pool:
                    raise ValueError(
                        f"case {case.case_id!r} already present in pool {existing.value!r} "
                        f"and {pool.value!r}: pools must stay disjoint"
                    )
                index[case.case_id] = pool
        return index


class JsonGroundTruthStore(GroundTruthStore):
    """File-backed store: ``<root>/<pool>.json`` holds one pool's set."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, pool: Pool) -> Path:
        return self.root / f"{pool.value}.json"

    def load(self, pool: Pool) -> GroundTruthSet:
        path = self._path(pool)
        if not path.exists():
            return GroundTruthSet(pool=pool, name=pool.value)
        return GroundTruthSet.model_validate_json(path.read_text())

    def save(self, gt_set: GroundTruthSet) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(gt_set.pool).write_text(gt_set.model_dump_json(indent=2) + "\n")


# ---------------------------------------------------------------------------
# Review-queue round-trip (candidate cases a human edits between stages)
# ---------------------------------------------------------------------------


def dump_cases(cases: Iterable[GroundTruthCase], *, indent: int | None = 2) -> str:
    """Serialize candidate cases to a JSON array string."""
    return json.dumps(
        [json.loads(c.model_dump_json()) for c in cases],
        indent=indent,
    )


def load_cases(payload: str) -> list[GroundTruthCase]:
    """Parse a JSON array (as produced by :func:`dump_cases`) back into cases."""
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("review queue payload must be a JSON array of cases")
    return [GroundTruthCase.model_validate(item) for item in raw]


def write_review_queue(cases: Iterable[GroundTruthCase], path: str | Path) -> Path:
    """Write candidate cases to ``path`` for a human to edit. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_cases(cases) + "\n")
    return out


def read_review_queue(path: str | Path) -> list[GroundTruthCase]:
    """Read a review queue written by :func:`write_review_queue`."""
    return load_cases(Path(path).read_text())
