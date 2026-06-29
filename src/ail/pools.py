"""Canonical pool vocabulary — the one source of truth for the three pools.

``docs/ARCHITECTURE.md`` §2 names the single non-negotiable principle of this
system: the agent must never be optimized against the same labels its judge is
aligned against, or the two co-adapt — scores climb while real quality stalls.
The defence is three **disjoint** data pools that are never mixed:

* **Task Suite** — fixed tasks re-run to compare agent versions. Never fed to
  the optimizer or to judge alignment.
* **Alignment Set** — labeled traces used to align judges (MemAlign).
* **Human Anchor** — a small human-labeled slice that audits judge-vs-human
  agreement.

This *identity* is shared across the codebase: :mod:`ail.groundtruth` curates and
stores cases per pool (its storage schemas, promotion, and Task-Suite
protection), while :mod:`ail.judges` consumes the pools to align and audit
judges (its ``AlignmentSet`` / ``HumanAnchor`` handles). Both import
:class:`Pool` from here so there is exactly **one** definition of "which pool" —
a single enum, never two that can silently drift apart.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Pool"]


class Pool(StrEnum):
    """The three disjoint pools of the frozen evaluation wall.

    The string values are stable identifiers safe to persist on disk, record on
    a report, or tag a trace with. A case or trace lives in exactly one pool;
    :func:`ail.judges.pools.assert_pools_disjoint` is the runtime guard that
    proves no id leaked across the wall.
    """

    TASK_SUITE = "task_suite"
    ALIGNMENT_SET = "alignment_set"
    HUMAN_ANCHOR = "human_anchor"
