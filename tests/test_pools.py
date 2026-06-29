"""The canonical Pool vocabulary (:mod:`ail.pools`) is the one source of truth.

Both the ground-truth storage layer and the judge layer must consume the *same*
``Pool`` enum object — not two structurally-equal copies that could drift apart.
These tests pin that invariant so a future edit cannot reintroduce a duplicate.
"""

from __future__ import annotations

from ail.pools import Pool


def test_pool_has_the_three_disjoint_pools() -> None:
    assert {p.value for p in Pool} == {"task_suite", "alignment_set", "human_anchor"}


def test_groundtruth_and_judges_share_one_pool_identity() -> None:
    # Importing through either layer must yield the *same* object, so the
    # vocabulary cannot diverge between storage and consumer code.
    from ail.groundtruth import Pool as groundtruth_pool
    from ail.groundtruth.schema import Pool as schema_pool
    from ail.judges import Pool as judges_pool
    from ail.judges.pools import Pool as judges_pools_pool

    assert schema_pool is Pool
    assert groundtruth_pool is Pool
    assert judges_pool is Pool
    assert judges_pools_pool is Pool


def test_pool_values_are_stable_identifiers() -> None:
    # These strings are persisted on disk and recorded on reports/trace tags;
    # changing them is a breaking schema change, so lock them down.
    assert Pool.TASK_SUITE.value == "task_suite"
    assert Pool.ALIGNMENT_SET.value == "alignment_set"
    assert Pool.HUMAN_ANCHOR.value == "human_anchor"
