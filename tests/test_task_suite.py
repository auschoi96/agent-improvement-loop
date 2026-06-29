"""The frozen Task Suite: schema, freeze/integrity, loader, and the
anti-co-adaptation wall.

The load-bearing guarantee (``docs/ARCHITECTURE.md`` §2) is that the optimizer /
ground-truth path can never write into — and therefore never train against — the
Task Suite pool. These tests pin both halves of that: the suite is read-only once
frozen, and the ground-truth promotion path structurally refuses the pool.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from ail.groundtruth.promote import TaskSuiteProtectedError, promote_approved
from ail.groundtruth.schema import (
    GroundTruthCase,
    GroundTruthError,
    Source,
    SourceKind,
    TaskInput,
    validate_pool_membership,
)
from ail.groundtruth.store import JsonGroundTruthStore
from ail.pools import Pool
from ail.task_suite import (
    Difficulty,
    Task,
    TaskCategory,
    TaskSuite,
    TaskSuiteError,
    TaskSuiteFrozenError,
    TaskSuiteIntegrityError,
    build_seed_suite,
    dump_task_suite_yaml,
    load_task_suite,
    save_task_suite,
    task_suite_path,
)


def _task(task_id: str = "t1") -> Task:
    return Task(
        task_id=task_id,
        prompt="do the thing",
        category=TaskCategory.TYPICAL_SHORT_SESSION,
        source_trace_id="abc123",
        difficulty=Difficulty.EASY,
    )


# --------------------------------------------------------------------------- #
# Schema contract
# --------------------------------------------------------------------------- #


def test_task_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Task(
            task_id="t1",
            prompt="p",
            category=TaskCategory.TYPICAL_SHORT_SESSION,
            source_trace_id="x",
            difficulty=Difficulty.EASY,
            bogus="nope",  # type: ignore[call-arg]
        )


def test_suite_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TaskSuite(version="v", surprise=1)  # type: ignore[call-arg]


def test_suite_rejects_duplicate_task_ids() -> None:
    with pytest.raises(TaskSuiteError) as ei:
        TaskSuite(version="v", tasks=(_task("dup"), _task("dup")))
    assert "duplicate" in str(ei.value).lower()


def test_suite_must_be_the_task_suite_pool() -> None:
    with pytest.raises(TaskSuiteError) as ei:
        TaskSuite(version="v", pool=Pool.ALIGNMENT_SET, tasks=(_task(),))
    assert "task_suite" in str(ei.value)


# --------------------------------------------------------------------------- #
# Freeze / immutability
# --------------------------------------------------------------------------- #


def test_unfrozen_suite_can_grow_then_freeze_seals_it() -> None:
    suite = TaskSuite(version="v", tasks=(_task("a"),))
    assert not suite.frozen and suite.content_hash == ""

    grown = suite.with_task(_task("b"))
    assert grown.task_ids() == {"a", "b"}
    assert not grown.frozen

    frozen = grown.freeze()
    assert frozen.frozen
    assert frozen.content_hash  # populated on freeze
    # freeze is idempotent
    assert frozen.freeze() is frozen


def test_frozen_suite_refuses_mutation_helpers() -> None:
    frozen = TaskSuite(version="v", tasks=(_task("a"),)).freeze()
    with pytest.raises(TaskSuiteFrozenError):
        frozen.with_task(_task("b"))
    with pytest.raises(TaskSuiteFrozenError):
        frozen.with_tasks([_task("b")])


def test_frozen_pydantic_model_blocks_attribute_assignment() -> None:
    frozen = TaskSuite(version="v", tasks=(_task("a"),)).freeze()
    with pytest.raises(ValidationError):
        frozen.version = "v2"  # type: ignore[misc]


def test_frozen_suite_with_tampered_hash_fails_integrity() -> None:
    frozen = TaskSuite(version="v", tasks=(_task("a"),)).freeze()
    # Reconstructing the frozen suite with edited tasks but the old hash must
    # fail closed — this is the on-disk tamper case modelled in memory.
    with pytest.raises(TaskSuiteIntegrityError):
        TaskSuite(
            version="v",
            frozen=True,
            content_hash=frozen.content_hash,
            tasks=(_task("a"), _task("b")),
        )


# --------------------------------------------------------------------------- #
# Loader + on-disk artifact
# --------------------------------------------------------------------------- #


def test_committed_v1_artifact_loads_frozen() -> None:
    suite = load_task_suite("v1")
    assert suite.frozen
    assert suite.pool is Pool.TASK_SUITE
    assert 15 <= len(suite) <= 30
    # every dominant-pattern category is represented
    assert {t.category for t in suite.tasks} == set(TaskCategory)
    # every task is keyed to a real source trace (provenance)
    assert all(t.source_trace_id for t in suite.tasks)


def test_committed_artifact_matches_curated_seed() -> None:
    # The on-disk artifact cannot drift from its reviewed source.
    on_disk = load_task_suite("v1")
    rebuilt = build_seed_suite().freeze()
    assert on_disk.content_hash == rebuilt.content_hash
    assert on_disk.version == rebuilt.version


def test_loading_tampered_on_disk_artifact_raises(tmp_path) -> None:
    suite = build_seed_suite().freeze()
    save_task_suite(suite, "v1", root=tmp_path)
    path = task_suite_path("v1", root=tmp_path)

    data = yaml.safe_load(path.read_text())
    data["tasks"][0]["prompt"] = "smuggled-in edit"  # tamper, keep the old hash
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(TaskSuiteIntegrityError):
        load_task_suite("v1", root=tmp_path)


def test_save_refuses_to_clobber_frozen_artifact(tmp_path) -> None:
    suite = build_seed_suite().freeze()
    save_task_suite(suite, "v1", root=tmp_path)
    with pytest.raises(TaskSuiteFrozenError):
        save_task_suite(suite, "v1", root=tmp_path)
    # explicit overwrite is allowed
    save_task_suite(suite, "v1", root=tmp_path, overwrite=True)


def test_yaml_round_trips(tmp_path) -> None:
    suite = build_seed_suite().freeze()
    text = dump_task_suite_yaml(suite)
    reloaded = TaskSuite.model_validate(yaml.safe_load(text))
    assert reloaded.content_hash == suite.content_hash
    assert reloaded.task_ids() == suite.task_ids()


def test_missing_artifact_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_task_suite("does-not-exist", root=tmp_path)


# --------------------------------------------------------------------------- #
# Anti-co-adaptation: the suite is read-only to the optimization / GT path
# --------------------------------------------------------------------------- #


def _minimal_case(case_id: str = "c1") -> GroundTruthCase:
    return GroundTruthCase(
        case_id=case_id,
        task_input=TaskInput(prompt="p"),
        sources=[Source(kind=SourceKind.TRACE, ref="trace-x")],
    )


def test_groundtruth_promotion_path_refuses_task_suite(tmp_path) -> None:
    # The bootstrap/optimization loop cannot promote ground truth into the held-
    # out benchmark — it raises before touching the store.
    store = JsonGroundTruthStore(tmp_path)
    with pytest.raises(TaskSuiteProtectedError):
        promote_approved([], pool=Pool.TASK_SUITE, store=store)


def test_task_suite_groundtruth_set_must_stay_empty() -> None:
    # The persistence boundary forbids any case from landing in the Task Suite
    # pool via the ground-truth path; an empty set is the only legal one.
    validate_pool_membership(Pool.TASK_SUITE, [])  # allowed
    with pytest.raises(GroundTruthError):
        validate_pool_membership(Pool.TASK_SUITE, [_minimal_case()])


def test_frozen_committed_suite_cannot_be_trained_against() -> None:
    # The complementary half: the curated benchmark itself is immutable, so the
    # optimizer cannot write back into it even if it held a reference.
    suite = load_task_suite("v1")
    assert suite.frozen
    with pytest.raises(TaskSuiteFrozenError):
        suite.with_task(_task("injected"))
