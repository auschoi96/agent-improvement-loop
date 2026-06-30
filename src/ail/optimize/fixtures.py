"""Live Phase-2 fixtures: per-task ``seed``/``verify`` dirs + per-arm workspaces.

A live fixture lives at ``eval/phase2_fixtures/<task_id>/`` with two
subdirectories:

* ``seed/`` — the starting repo state the agent edits. Copied into a **fresh,
  per-arm** workspace before each run, so the agent mutates an isolated copy and
  the baseline and candidate arms never share a directory.
* ``verify/`` — the pristine, deterministic L1 check (e.g. a pytest test). It is
  **not** part of what the agent edits: it is *restored* into each arm's
  workspace **after** that arm's run (overwriting any agent edits — see
  :func:`restore_verify`), so the agent cannot game the check by editing or
  deleting the test. The verify command (from the run plan's
  :class:`~ail.optimize.phase2.VerifySpec`) then runs with ``cwd`` set to that
  arm's workspace.

This module is the **loader + workspace lifecycle** only. It depends on stdlib
plus the comparison contract via the runner — never on
:mod:`ail.optimize.phase2` — so the runner imports *it* without a cycle. The
runner owns the verify-command semantics: it builds the arm-aware verifier and
wires these workspaces into :func:`ail.compare.compare_candidate` through
:class:`~ail.compare.ArmWorkspaces`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PHASE2_FIXTURES_REL",
    "SEED_DIRNAME",
    "VERIFY_DIRNAME",
    "TaskFixture",
    "ArmWorkspacePaths",
    "phase2_fixtures_root",
    "fixture_dir",
    "load_fixture",
    "restore_verify",
    "isolated_arm_workspaces",
]

#: Env override for the fixtures root (mirrors ``AIL_TASK_SUITE_ROOT``).
_ENV_ROOT = "AIL_PHASE2_FIXTURES_ROOT"
#: Location of the fixtures tree, relative to the discovered root.
PHASE2_FIXTURES_REL = ("eval", "phase2_fixtures")
#: Subdirectory holding the agent's starting state.
SEED_DIRNAME = "seed"
#: Subdirectory holding the pristine L1 check, restored post-run.
VERIFY_DIRNAME = "verify"

_WORKSPACE_PREFIX = "ail-phase2-"
_BASELINE_ARM = "baseline"
_CANDIDATE_ARM = "candidate"


def phase2_fixtures_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the directory that contains ``eval/phase2_fixtures``.

    Precedence mirrors :func:`ail.task_suite.loader.task_suite_root`: explicit
    ``root`` argument > ``AIL_PHASE2_FIXTURES_ROOT`` env var > upward search from
    this module's location (editable install) > the current working directory as
    a last resort.
    """
    if root is not None:
        return Path(root)
    env = os.environ.get(_ENV_ROOT)
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.joinpath(*PHASE2_FIXTURES_REL).is_dir():
            return parent
    return Path.cwd()


def fixture_dir(task_id: str, *, root: str | os.PathLike[str] | None = None) -> Path:
    """Path to one task's fixture directory (``eval/phase2_fixtures/<task_id>``)."""
    return phase2_fixtures_root(root).joinpath(*PHASE2_FIXTURES_REL, task_id)


@dataclass(frozen=True, slots=True)
class TaskFixture:
    """A located live fixture: its ``seed/`` and ``verify/`` directory paths.

    ``verify_dir`` is recorded even when it does not exist on disk so the caller
    has a stable path; :func:`restore_verify` no-ops (returns ``False``) when the
    fixture carries no ``verify/`` directory.
    """

    task_id: str
    seed_dir: Path
    verify_dir: Path

    @property
    def has_verify(self) -> bool:
        """Whether the fixture ships a ``verify/`` directory to restore."""
        return self.verify_dir.is_dir()


def load_fixture(task_id: str, *, root: str | os.PathLike[str] | None = None) -> TaskFixture | None:
    """Locate the live fixture for ``task_id``; ``None`` if it has no ``seed/`` dir.

    A fixture is identified by the presence of a ``seed/`` directory (the agent's
    starting state — the thing the arms isolate). The ``verify/`` directory is
    optional: a fixture with seed but no verify still gives the arms isolation,
    but with no verify command the harness has no L1 correctness signal and fails
    closed (``BLOCK``). A task with no fixture at all returns ``None`` and the
    runner falls back to its legacy (arm-blind, non-isolated) path.
    """
    base = fixture_dir(task_id, root=root)
    seed = base / SEED_DIRNAME
    if not seed.is_dir():
        return None
    return TaskFixture(task_id=task_id, seed_dir=seed, verify_dir=base / VERIFY_DIRNAME)


def restore_verify(verify_dir: Path, workspace: Path) -> bool:
    """Restore the pristine ``verify/`` tree into ``<workspace>/verify``, overwriting.

    **Tamper-proofing.** The agent's run may have edited, added to, or deleted the
    check files (or the whole ``verify/`` directory). This replaces
    ``<workspace>/verify`` *wholesale* with the pristine fixture copy — removing
    any agent-added files first — so a verdict is always measured against the
    original test, never one the agent could rewrite to pass. Returns ``False``
    when the fixture has no ``verify/`` directory (nothing to restore).
    """
    if not verify_dir.is_dir():
        return False
    dest = workspace / VERIFY_DIRNAME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(verify_dir, dest)
    return True


@dataclass(frozen=True, slots=True)
class ArmWorkspacePaths:
    """The two per-arm workspace directories for one task, under a shared parent.

    ``root`` is the per-task temp directory removed wholesale on cleanup;
    ``baseline`` and ``candidate`` are the separate seeded copies the two arms run
    in.
    """

    root: Path
    baseline: Path
    candidate: Path


@contextmanager
def isolated_arm_workspaces(fixture: TaskFixture) -> Iterator[ArmWorkspacePaths]:
    """Create two fresh workspaces seeded from ``fixture.seed_dir``; always clean up.

    Copies ``seed/`` into ``<tmp>/baseline`` and ``<tmp>/candidate`` (separate
    directories — never shared) and yields their paths. The whole per-task temp
    tree is removed on exit, **including on exception**, so a crashed or raising
    run never leaks a workspace.
    """
    parent = Path(tempfile.mkdtemp(prefix=_WORKSPACE_PREFIX))
    try:
        baseline = parent / _BASELINE_ARM
        candidate = parent / _CANDIDATE_ARM
        shutil.copytree(fixture.seed_dir, baseline)
        shutil.copytree(fixture.seed_dir, candidate)
        yield ArmWorkspacePaths(root=parent, baseline=baseline, candidate=candidate)
    finally:
        shutil.rmtree(parent, ignore_errors=True)
