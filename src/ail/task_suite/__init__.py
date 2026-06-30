"""The frozen, versioned Task Suite — the held-out benchmark of the eval wall.

This package owns the *curation and storage* of the Task Suite pool
(:attr:`ail.pools.Pool.TASK_SUITE`): the fixed tasks an agent is re-run against
to compare versions. ``docs/ARCHITECTURE.md`` §2 makes it the load-bearing
anti-co-adaptation wall — the optimizer and judge-alignment path must **never**
train against it. That separation is enforced structurally elsewhere:
:mod:`ail.groundtruth.promote` refuses to promote ground truth into the Task
Suite, and :func:`ail.groundtruth.schema.validate_pool_membership` keeps the
Task-Suite ground-truth set empty. This package is the *separate* curation API
those modules' docs point to (Wave 1b), populated from task inputs — not from
labelled ground truth.

* **Schema** (:mod:`ail.task_suite.schema`) — :class:`Task` and the freezable,
  content-hashed :class:`TaskSuite`, Pydantic ``extra="forbid"`` / ``frozen``.
* **Loader** (:mod:`ail.task_suite.loader`) — read/write the versioned on-disk
  artifact (``eval/task_suite/<version>/tasks.yaml``) with integrity checks.
* **Seed** (:mod:`ail.task_suite.seed`) — the curated ``v1-seed`` content,
  abstracted from the real L0 diagnosis.
"""

from ail.task_suite.loader import (
    DEFAULT_ARTIFACT_VERSION,
    dump_task_suite_yaml,
    load_task_suite,
    save_task_suite,
    task_suite_path,
    task_suite_root,
)
from ail.task_suite.phase2_mini import PHASE2_MINI_VERSION, build_phase2_mini_suite
from ail.task_suite.schema import (
    SUITE_SCHEMA_VERSION,
    Difficulty,
    Task,
    TaskCategory,
    TaskSuite,
    TaskSuiteError,
    TaskSuiteFrozenError,
    TaskSuiteIntegrityError,
)
from ail.task_suite.seed import SEED_VERSION, build_seed_suite

__all__ = [
    # schema
    "SUITE_SCHEMA_VERSION",
    "Task",
    "TaskCategory",
    "Difficulty",
    "TaskSuite",
    "TaskSuiteError",
    "TaskSuiteFrozenError",
    "TaskSuiteIntegrityError",
    # loader
    "DEFAULT_ARTIFACT_VERSION",
    "task_suite_root",
    "task_suite_path",
    "load_task_suite",
    "dump_task_suite_yaml",
    "save_task_suite",
    # seed
    "SEED_VERSION",
    "build_seed_suite",
    # phase2-mini (runnable live-fixture suite)
    "PHASE2_MINI_VERSION",
    "build_phase2_mini_suite",
]
