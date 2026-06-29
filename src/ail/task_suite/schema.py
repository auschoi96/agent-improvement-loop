"""The frozen, versioned Task Suite contract — schema, content hash, freeze.

The Task Suite is the held-out benchmark of the frozen evaluation wall
(``docs/ARCHITECTURE.md`` §2): a fixed set of tasks re-run to compare agent
versions, **never** fed to the optimizer or to judge alignment. It is the
load-bearing anti-co-adaptation wall — if the optimizer could train against
these tasks, "improvement" would be measured against the very thing being
optimized and the number would lie.

Two guarantees live in this module:

* **No drift.** Every model is a Pydantic v2 model with ``extra="forbid"`` (an
  unknown field fails loudly instead of being silently dropped) and
  ``frozen=True`` (a suite is evolved by constructing a new validated instance,
  never by mutating one in place). This is the same contract discipline
  :mod:`ail.groundtruth.schema` uses.
* **Freeze + integrity.** A :class:`TaskSuite` carries a ``frozen`` flag and a
  ``content_hash`` over its tasks. :meth:`TaskSuite.freeze` seals the suite:
  once frozen, the mutation helpers raise :class:`TaskSuiteFrozenError`, and any
  later edit to the on-disk artifact makes the recomputed hash disagree with the
  stored one, so loading it raises :class:`TaskSuiteIntegrityError`. The hash is
  *tamper detection* for accidental drift and an integrity seal, not a
  cryptographic guarantee against a determined committer — the structural
  guarantee that the optimizer/ground-truth path cannot reach this pool lives in
  :mod:`ail.groundtruth` (see ``promote.py``'s ``TaskSuiteProtectedError``).

The pool *identity* comes from :class:`ail.pools.Pool` — the one canonical
vocabulary — so the suite and the disjointness guard can never disagree on
"which pool".
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from ail.pools import Pool

#: Version of the Task Suite *contract* (the schema shape). Distinct from a
#: suite's :attr:`TaskSuite.version`, which labels the *content* of a frozen
#: artifact (e.g. ``"v1-seed"``). Bump the minor for additive fields, the major
#: for a breaking shape change.
SUITE_SCHEMA_VERSION = "ail.task_suite/v1"

__all__ = [
    "SUITE_SCHEMA_VERSION",
    "TaskCategory",
    "Difficulty",
    "Task",
    "TaskSuite",
    "TaskSuiteError",
    "TaskSuiteFrozenError",
    "TaskSuiteIntegrityError",
]


class TaskSuiteError(Exception):
    """Base class for Task Suite contract/freeze errors."""


class TaskSuiteFrozenError(TaskSuiteError):
    """Raised when a mutation is attempted on a frozen suite.

    A frozen suite is immutable by contract: growing or editing it would change
    the benchmark the optimizer is measured against. The mutation helpers
    (:meth:`TaskSuite.with_task`, :meth:`TaskSuite.with_tasks`) raise this rather
    than silently returning an altered copy.
    """


class TaskSuiteIntegrityError(TaskSuiteError):
    """Raised when a frozen suite's ``content_hash`` disagrees with its tasks.

    This is the tamper-detection signal: a frozen on-disk artifact whose tasks
    were edited after sealing will not re-hash to its stored ``content_hash``,
    so loading it fails closed instead of serving a silently-mutated benchmark.
    """


class TaskCategory(StrEnum):
    """The task categories, derived from the dominant patterns in the real L0
    diagnosis (``artifacts/example1_diagnosis.{md,json}``).

    These are the failure shapes the token-reduction lever (Wave 2) must improve
    without regressing the typical case.
    """

    #: The heavy tail of the bimodal token distribution — sessions whose context
    #: balloons to hundreds of thousands of tokens. Where the spend lives.
    HEAVY_TAIL_HIGH_TOKEN = "heavy_tail_high_token"
    #: Sessions that issue an outsized number of tool calls (high action count),
    #: independent of raw token size.
    HIGH_TOOL_CALL_VOLUME = "high_tool_call_volume"
    #: Sessions dominated by re-run boilerplate — the same shell prologue
    #: (``cd``/env setup) or the same file edited over and over.
    REPEATED_TARGET_BOILERPLATE = "repeated_target_boilerplate"
    #: The low-median bulk: ordinary short sessions. Carried so the suite covers
    #: the normal case and the optimizer is held to "do not regress these", not
    #: only "shrink the pathologies".
    TYPICAL_SHORT_SESSION = "typical_short_session"


class Difficulty(StrEnum):
    """Coarse difficulty, set from the source session's magnitude."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class _Frozen(BaseModel):
    """Base for every model: forbid unknown fields and freeze instances.

    ``extra="forbid"`` makes schema drift fail loudly; ``frozen=True`` blocks
    in-place mutation so a suite is only ever evolved by building a new validated
    instance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Task(_Frozen):
    """One benchmark task in the frozen suite.

    ``prompt`` is the agent-facing input. ``source_trace_id`` ties the task back
    to the real session it was abstracted from (provenance, like a ground-truth
    :class:`~ail.groundtruth.schema.Source`). In the ``v1-seed`` artifact the
    prompt is a *reconstruction* derived from each trace's observable L0 profile
    (project, tool mix, repeated targets), not the verbatim session input — the
    raw trace content is not yet readable (see the suite README). ``notes``
    records what is derived vs. unknown so the reconstruction is auditable.
    """

    task_id: str
    prompt: str
    category: TaskCategory
    source_trace_id: str
    difficulty: Difficulty
    notes: str = ""


def _hash_tasks(version: str, tasks: tuple[Task, ...]) -> str:
    """A stable sha256 over the *content* of a suite (identity + ordered tasks).

    Covers the benchmark identity (pool + schema/content version) and the ordered
    tasks. Excludes the seal fields (``frozen``/``content_hash``/``created_at``)
    so the hash is a property of what the suite *contains*, and is canonical
    (sorted keys, no whitespace) so it is reproducible across processes.
    """
    payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "pool": Pool.TASK_SUITE.value,
        "version": version,
        "tasks": [task.model_dump(mode="json") for task in tasks],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class TaskSuite(_Frozen):
    """A versioned, content-hashed, freezable collection of benchmark tasks.

    The on-disk unit (``eval/task_suite/<dir>/tasks.yaml``). Built unfrozen,
    sealed with :meth:`freeze`, then loaded read-only thereafter. The
    ``model_validator`` enforces unique task ids, that the pool is exactly the
    held-out :attr:`Pool.TASK_SUITE`, and — for a frozen suite — that the stored
    ``content_hash`` matches the tasks (integrity).
    """

    schema_version: str = SUITE_SCHEMA_VERSION
    #: Content label of this frozen artifact (e.g. ``"v1-seed"``); see
    #: :data:`SUITE_SCHEMA_VERSION` for the distinct contract version.
    version: str
    pool: Pool = Pool.TASK_SUITE
    created_at: str | None = None
    frozen: bool = False
    content_hash: str = ""
    tasks: tuple[Task, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> TaskSuite:
        if self.pool is not Pool.TASK_SUITE:
            raise TaskSuiteError(
                f"a TaskSuite must live in the held-out {Pool.TASK_SUITE.value!r} pool, "
                f"not {self.pool.value!r}"
            )
        ids = [t.task_id for t in self.tasks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise TaskSuiteError(f"duplicate task_id(s) in suite: {dupes}")
        if self.frozen:
            expected = _hash_tasks(self.version, self.tasks)
            if self.content_hash != expected:
                raise TaskSuiteIntegrityError(
                    "frozen Task Suite failed its integrity check: stored content_hash "
                    f"{self.content_hash!r} != recomputed {expected!r}. The artifact was "
                    "edited after it was frozen; refusing to serve a mutated benchmark."
                )
        return self

    def _rebuild(self, *, tasks: tuple[Task, ...], frozen: bool, content_hash: str) -> TaskSuite:
        """Construct a fresh, **re-validated** suite from this one's identity.

        Goes through the constructor (not ``model_copy``) so the
        ``model_validator`` fires — ``model_copy(update=...)`` skips validators in
        Pydantic v2, which would let a mutation slip an invariant (e.g. a
        duplicate ``task_id``) past the checks.
        """
        return TaskSuite(
            schema_version=self.schema_version,
            version=self.version,
            pool=self.pool,
            created_at=self.created_at,
            frozen=frozen,
            content_hash=content_hash,
            tasks=tasks,
        )

    def freeze(self) -> TaskSuite:
        """Return a sealed copy: ``frozen=True`` with the content hash computed.

        Idempotent — freezing an already-frozen suite returns it unchanged.
        """
        if self.frozen:
            return self
        return self._rebuild(
            tasks=self.tasks,
            frozen=True,
            content_hash=_hash_tasks(self.version, self.tasks),
        )

    def with_task(self, task: Task) -> TaskSuite:
        """Return a new (unfrozen) suite with ``task`` appended.

        Raises :class:`TaskSuiteFrozenError` if this suite is frozen.
        """
        return self.with_tasks([task])

    def with_tasks(self, tasks: list[Task]) -> TaskSuite:
        """Return a new (unfrozen) suite with ``tasks`` appended.

        Raises :class:`TaskSuiteFrozenError` if this suite is frozen — a frozen
        benchmark does not grow. The result is re-validated, so appending a
        duplicate ``task_id`` raises :class:`TaskSuiteError`.
        """
        if self.frozen:
            raise TaskSuiteFrozenError(
                "the Task Suite is frozen and cannot be added to; the held-out benchmark "
                "is immutable once sealed. Build a new version instead."
            )
        return self._rebuild(tasks=(*self.tasks, *tasks), frozen=False, content_hash="")

    def task_ids(self) -> frozenset[str]:
        """The set of task ids in the suite (the Task-Suite side of the wall)."""
        return frozenset(t.task_id for t in self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)
