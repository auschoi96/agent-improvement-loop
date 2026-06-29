"""Load and persist the versioned Task Suite artifact (``tasks.yaml``).

The on-disk layout is ``<root>/eval/task_suite/<version>/tasks.yaml``, where
``<version>`` is the artifact directory (e.g. ``"v1"``) — distinct from a
suite's content label (:attr:`~ail.task_suite.schema.TaskSuite.version`, e.g.
``"v1-seed"``). ``<root>`` is discovered by walking up from this file to the
repository that ships ``eval/task_suite`` (this package is installed editable),
overridable via the ``AIL_TASK_SUITE_ROOT`` environment variable or the explicit
``root`` argument.

:func:`load_task_suite` validates on read — including the frozen-integrity check
in :class:`~ail.task_suite.schema.TaskSuite` — so a tampered artifact fails
closed. :func:`save_task_suite` refuses to overwrite an existing **frozen**
artifact unless explicitly forced, so the committed benchmark cannot be
clobbered by accident.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ail.task_suite.schema import TaskSuite, TaskSuiteFrozenError, TaskSuiteIntegrityError

__all__ = [
    "DEFAULT_ARTIFACT_VERSION",
    "task_suite_root",
    "task_suite_path",
    "load_task_suite",
    "dump_task_suite_yaml",
    "save_task_suite",
]

#: The artifact *directory* loaded by default (the current frozen suite).
DEFAULT_ARTIFACT_VERSION = "v1"

_ENV_ROOT = "AIL_TASK_SUITE_ROOT"
_REL = ("eval", "task_suite")


def task_suite_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the directory that contains ``eval/task_suite``.

    Precedence: explicit ``root`` argument > ``AIL_TASK_SUITE_ROOT`` env var >
    upward search from this module's location (editable install) > the current
    working directory as a last resort.
    """
    if root is not None:
        return Path(root)
    env = os.environ.get(_ENV_ROOT)
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent.joinpath(*_REL)).is_dir():
            return parent
    return Path.cwd()


def task_suite_path(
    version: str = DEFAULT_ARTIFACT_VERSION,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Path to the ``tasks.yaml`` for an artifact ``version`` directory."""
    return task_suite_root(root).joinpath(*_REL, version, "tasks.yaml")


def load_task_suite(
    version: str = DEFAULT_ARTIFACT_VERSION,
    *,
    root: str | os.PathLike[str] | None = None,
) -> TaskSuite:
    """Load, validate, and return the Task Suite at artifact ``version``.

    A *persisted* suite must always be sealed: there is **no** load path that
    returns a suite whose ``content_hash`` was not verified. The schema's
    frozen-integrity check verifies the hash when ``frozen`` is set, and this
    loader additionally rejects any artifact that is **not** frozen — an
    unfrozen on-disk artifact is itself a tamper/corruption signal (otherwise an
    editor could mutate the tasks and flip ``frozen: true -> false`` to skip the
    hash check entirely). Combined, the only artifact that loads is one that is
    frozen *and* hashes to its stored ``content_hash``.

    Raises:
        FileNotFoundError: if the artifact does not exist.
        TaskSuiteIntegrityError: if the artifact is not frozen, or a frozen
            artifact's hash no longer matches its tasks.
    """
    path = task_suite_path(version, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"no Task Suite artifact at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    suite = TaskSuite.model_validate(data)  # frozen path verifies the content hash
    if not suite.frozen:
        raise TaskSuiteIntegrityError(
            f"the Task Suite artifact at {path} is not frozen; a persisted suite must be "
            "sealed (frozen=true with a matching content_hash). An unfrozen on-disk "
            "artifact is a tamper/corruption signal — refusing to serve an unverified "
            "benchmark."
        )
    return suite


def dump_task_suite_yaml(suite: TaskSuite) -> str:
    """Serialize a suite to canonical YAML (the on-disk artifact form)."""
    return yaml.safe_dump(
        suite.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )


def save_task_suite(
    suite: TaskSuite,
    version: str = DEFAULT_ARTIFACT_VERSION,
    *,
    root: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``suite`` to the artifact ``version`` directory; return the path.

    Refuses to clobber an existing **frozen** artifact unless ``overwrite`` is
    ``True`` — the committed benchmark is not casually replaceable.

    Raises:
        TaskSuiteFrozenError: if a frozen artifact already exists and
            ``overwrite`` is ``False``.
    """
    path = task_suite_path(version, root=root)
    if path.is_file() and not overwrite:
        existing = load_task_suite(version, root=root)
        if existing.frozen:
            raise TaskSuiteFrozenError(
                f"refusing to overwrite the frozen Task Suite at {path}; pass "
                "overwrite=True only if you intend to replace a sealed benchmark."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_task_suite_yaml(suite), encoding="utf-8")
    return path
